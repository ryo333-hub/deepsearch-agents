"""
主智能体组装与异步执行模块

负责把模型、主提示词、文件类工具和三个专家子智能体组装成 DeepAgent，
并提供 run_deep_agent 作为后续 API 层调用的统一入口。运行时还会为每个
session_id 创建独立工作目录，并把工具调用、子智能体调用和最终结果推送给前端。
"""

import asyncio
import shutil
from pathlib import Path

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.llm import model
from app.agent.prompts import main_agent_content
from app.agent.subagents.database_query_agent import database_query_agent
from app.agent.subagents.knowledge_base_agent import (
    build_knowledge_agent_task,
    knowledge_base_agent,
)
from app.agent.subagents.network_search_agent import network_search_agent
from app.api.context import (
    reset_selected_knowledge_base_context,
    reset_session_context,
    reset_thread_context,
    set_selected_knowledge_base_context,
    set_session_context,
    set_thread_context,
)
from app.api.monitor import monitor
from app.local_rag.storage import LocalRAGStorage

# 文件类工具由主智能体直接掌握，负责读取上传附件和生成最终交付文档
from app.tools.markdown_tools import generate_markdown
from app.tools.pdf_tools import convert_md_to_pdf
from app.tools.upload_file_read_tool import read_file_content
from app.utils.path_utils import resolve_path, resolve_session_directory, validate_thread_id

# 主智能体是调度中心：
# 1. tools 只放最终交付相关的文件工具
# 2. subagents 放网络、数据库、企业知识三类信息获取助手
# 3. checkpointer 通过 thread_id 保存同一会话中的执行上下文
# 4. 关闭 DeepAgents 默认 general-purpose，仅保留上面三个显式业务子智能体
register_harness_profile(
    "openai:deepseek-flash",
    HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
    ),
)

main_agent = create_deep_agent(
    model=model,
    system_prompt=main_agent_content["system_prompt"],
    tools=[generate_markdown, convert_md_to_pdf, read_file_content],
    checkpointer=InMemorySaver(),
    subagents=[database_query_agent, network_search_agent, knowledge_base_agent],
)

# 当前文件位于 app/agent/main_agent.py，parents[1] 即 app 目录
project_root_path = Path(__file__).parents[1].resolve()


def _build_session_file_instruction(uploaded_files: list[str]) -> str:
    """Build model-visible file guidance without exposing the host session path."""
    upload_section = ""
    if uploaded_files:
        upload_section = (
            "\n    [已上传文件] 以下文件已位于当前 session：\n"
            + "\n".join(f"    - {filename}" for filename in uploaded_files)
            + "\n    请按需使用 read_file_content，以这里列出的相对路径读取。\n"
        )

    return f"""
    【当前 session 文件规则】
    {upload_section}
    1. 三个文件工具的所有路径参数都使用当前 session 内相对路径，例如 input.pdf、uploads/data.pdf、reports/result.md。
    2. 不要添加 output/session_* 等目录前缀，不要传递 Windows 或服务器绝对路径。
    3. 禁止使用 ../、..\\、drive path、UNC path 或 device path，也不要尝试读取 .env、系统文件或当前 session 外文件。
    4. 如果工具拒绝某个路径，停止该路径尝试；不要绕过限制或猜测宿主机路径。
    5. Main Agent 可以直接调用 read_file_content、generate_markdown 和 convert_md_to_pdf 完成文件读取与生成。
    """


def validate_selected_knowledge_base(session_id: str, knowledge_base_id: str | None) -> str | None:
    """Validate a request's KB against its session before any model invocation."""
    validate_thread_id(session_id)
    if knowledge_base_id is None:
        return None
    token = set_thread_context(session_id)
    try:
        LocalRAGStorage().get_knowledge_base(knowledge_base_id)
    finally:
        reset_thread_context(token)
    return knowledge_base_id


def _build_knowledge_base_instruction(knowledge_base_id: str | None, query: str) -> str:
    """Give Main one authorized KB ID, never a host path or a guessed default."""
    if knowledge_base_id is None:
        return (
            "\n【当前任务知识库选择】未选择企业知识库。内部文档查询不能委派给企业知识助手；"
            "请说明当前任务未指定可查询的企业知识库。其他已获授权的数据来源可照常处理。"
            "不得从用户文本猜测或自行选择知识库 ID。\n"
        )
    delegated = build_knowledge_agent_task(knowledge_base_id, query,
                                           storage=LocalRAGStorage())
    return (
        "\n【当前任务知识库选择】以下知识库 ID 已按当前 session 验证。涉及企业内部文档时，"
        "通过 task 委派给企业知识助手，并在 task.description 中原样传递该 knowledge_base_id。"
        "可将 query 缩为相关的内部知识子问题，但不得改写 ID、改选其他知识库或传递宿主路径。\n"
        f"{delegated}\n"
    )


async def run_deep_agent(task_query, session_id, knowledge_base_id: str | None = None):
    """
    异步流式执行主智能体

    API 层会为每次任务传入用户问题和 session_id。本函数负责准备会话目录、
    复制上传文件、写入 ContextVar，并在流式执行过程中把关键事件上报给前端。
    :param task_query: 前端提交的原始任务问题
    :param session_id: 当前任务 ID，同时用于 thread_id、输出目录和 WebSocket 定向推送
    :param knowledge_base_id: API 为本次任务明确选定的知识库；不自动选择
    """
    knowledge_base_id = validate_selected_knowledge_base(session_id, knowledge_base_id)
    print(f"[MainAgent] 开始执行会话，session_id={session_id}")

    # 每个会话独立使用 output/session_{session_id}，避免不同用户的产物互相覆盖
    session_dir = resolve_session_directory(project_root_path / "output", session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    # 绝对路径只保留在应用内部和前端监控事件中，不写入模型消息。
    session_dir_str = str(session_dir).replace("\\", "/")

    # 上传文件先落在 updated/session_{session_id}，执行前复制到本次 output 工作目录
    # 这样读文件工具和生成文件工具都只需要围绕同一个 session_dir 工作
    updated_dir_path = resolve_session_directory(
        project_root_path / "updated", session_id
    )
    files = []
    if updated_dir_path.exists():
        for candidate in updated_dir_path.iterdir():
            source_path = Path(resolve_path(candidate.name, updated_dir_path))
            if not source_path.is_file():
                continue
            destination_path = Path(resolve_path(candidate.name, session_dir))
            shutil.copy2(source_path, destination_path)
            files.append(candidate.name)

    # ContextVar 让深层工具无需显式传参，也能拿到当前会话目录和 WebSocket thread_id
    session_dir_token = set_session_context(session_dir_str)
    session_id_token = set_thread_context(session_id)
    knowledge_base_token = set_selected_knowledge_base_context(knowledge_base_id)

    try:
        # 前端拿到工作目录后，可以展示本次任务生成的 Markdown/PDF 等产物
        monitor.report_session_dir(session_dir_str)

        # checkpointer 依赖 thread_id 区分会话记忆；同一 session_id 会复用同一条执行上下文
        config = {"configurable": {"thread_id": session_id}}

        # 模型只看到已选 KB ID 和逻辑相对路径；真实 session_dir 只在 ContextVar 中。
        path_instruction = _build_session_file_instruction(files)
        knowledge_instruction = _build_knowledge_base_instruction(knowledge_base_id, task_query)
        # astream 会持续产出模型节点、工具节点和子智能体节点的状态片段
        async for chunk in main_agent.astream(
            {"messages": [{"role": "user", "content": task_query + path_instruction + knowledge_instruction}]},
            config=config,
        ):
            # chunk 形如 {"model": {"messages": [...]}}，这里主要关心模型最新消息
            for node_name, state in chunk.items():
                if not state or "messages" not in state:
                    continue
                messages = state["messages"]
                if messages and isinstance(messages, list):
                    last_msg = messages[-1]
                    if node_name == "model":
                        if last_msg.tool_calls:
                            # DeepAgents 调用子智能体时，本质上会产生名为 task 的工具调用
                            for tool_call in last_msg.tool_calls:
                                if tool_call["name"] == "task":
                                    # 子智能体调用单独上报，前端可以展示“正在调用哪个专家助手”
                                    monitor.report_assistant(
                                        tool_call["args"]["subagent_type"],
                                        {
                                            "description": tool_call["args"][
                                                "description"
                                            ]
                                        },
                                    )
                        elif last_msg.content:
                            # 模型没有继续调用工具时，最新文本内容就是本轮可反馈给前端的结果
                            print(
                                f"主智能体执行结果，最终结果：{last_msg.content[:100]}"
                            )
                            monitor.report_task_result(last_msg.content)

    except asyncio.CancelledError:
        monitor.report_task_cancelled()
        raise
    except Exception as e:
        # 异步执行异常也走 monitor，保证前端能收到明确错误事件
        monitor._emit("error", f"执行主智能发生异常信息：{str(e)}")
    finally:
        # 任务结束后恢复 ContextVar，避免后续请求复用到本次会话目录或 thread_id
        reset_selected_knowledge_base_context(knowledge_base_token)
        reset_session_context(session_dir_token, session_id_token)


if __name__ == "__main__":
    import asyncio

    asyncio.run(
        run_deep_agent("从网络查询机器人信息，并生成Markdown文件", "test_session_001")
    )
