"""
Markdown 文件生成工具

供主智能体把最终整理后的内容写入当前会话工作目录。工具会把模型传入的
filename/path 交给 resolve_path 统一解析，避免模型直接操作真实绝对路径。
"""

from pathlib import Path

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated
from langchain_core.tools import tool

from app.api.context import get_session_context
from app.api.monitor import monitor
from app.utils.path_utils import PATH_ACCESS_DENIED_MESSAGE, resolve_path


@tool
def generate_markdown(
    content: Annotated[str, "要写入Markdown文档的文本内容"],
    filename: Annotated[
        str,
        "当前 session 内的 Markdown 相对文件路径，例如 report.md 或 "
        "reports/report.md；也可只传纯文件名并配合 path。禁止绝对路径和 traversal。",
    ],
    path: Annotated[
        str,
        "可选的当前 session 内相对子目录，例如 reports；不得使用绝对路径、../、..\\、盘符或 UNC 路径。",
    ] = "",
):
    """
    根据提供的文本内容生成 Markdown 文件

    :param content: 要写入 Markdown 文档的完整文本
    :param filename: 当前 session 内相对文件路径，缺少 .md 后缀时会自动补全
    :param path: 可选的当前 session 内相对子目录；使用时 filename 应为纯文件名
    :return: 文件生成结果说明
    """
    print(f"[MarkdownTool] 输入保存路径: {path or '当前会话目录'}")
    monitor.report_tool("Markdown文档生成工具", {"写入的文本内容": content})
    # session_dir 由 run_deep_agent 写入 ContextVar，保证文件写入当前会话工作目录
    session_dir = get_session_context()
    print(f"[MarkdownTool] 当前会话目录: {session_dir}")

    try:
        # filename 和 path 分别先过 Guard，避免在校验前做 Path/后缀变换时
        # 意外改变 ADS、盘符或其他 Windows 特殊路径的语义。
        safe_filename = Path(resolve_path(filename, session_dir))
        session_path = Path(session_dir).resolve(strict=False)
        if path and path != ".":
            if safe_filename.parent != session_path:
                return PATH_ACCESS_DENIED_MESSAGE
            safe_parent = Path(resolve_path(path, session_dir))
            output_path = safe_parent / safe_filename.name
        else:
            output_path = safe_filename

        output_name = output_path.name
        if not output_name.endswith(".md"):
            output_name += ".md"
        relative_output = output_path.with_name(output_name).relative_to(session_path)
        file_path = Path(resolve_path(relative_output.as_posix(), session_dir))
    except ValueError:
        return PATH_ACCESS_DENIED_MESSAGE

    parent_dir = file_path.parent

    print(
        f"[MarkdownTool] Debug: parent_dir={parent_dir}, filename={output_name}, full_path={file_path}"
    )

    try:
        # 允许模型指定 session_dir 下的子目录；不存在时自动创建
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
            print(f"[MarkdownTool] 已创建目录: {parent_dir}")

        file_path.write_text(content, encoding="utf-8")

        print(f"[MarkdownTool] 文件写入完成: {file_path}")
        logical_output_path = file_path.relative_to(session_path).as_posix()
        return f"Markdown文件 '{logical_output_path}' 已成功生成并保存。"
    except Exception as e:
        print(f"[MarkdownTool] 文件写入失败: {e}")
        return "生成Markdown文件失败。"


if __name__ == "__main__":
    # 本地调试入口：直接运行本文件可验证 Markdown 写入和路径解析效果
    def get_session_context():
        return "./examples/test_docs"

    test_content = "# 测试文档\n这是 Markdown 生成工具的本地测试内容"
    test_filename = "测试文件"
    test_path = "sub_dir"

    print("===== 开始测试：Markdown 文件生成 =====")
    result = generate_markdown.invoke(
        {"content": test_content, "filename": test_filename, "path": test_path}
    )

    print(f"\n调用结果：{result}")
    if "已成功生成" in result:
        file_path = Path(result.split("'")[1])
        print(f"验证结果：文件 {file_path} {'存在' if file_path.exists() else '不存在'}")
