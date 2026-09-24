"""
企业知识子智能体配置模块

将本地知识 Prompt 与只读 Local RAG Tool 组装成 DeepAgents 字典式子智能体。
Main Agent 通过 task 委派内部知识查询，并传递当前任务选定的知识库 ID。
"""

import json

from app.agent.prompts import sub_agents_content
from app.local_rag.storage import LocalRAGStorage
from app.tools.local_knowledge_base_tool import search_local_knowledge_base

# 本助手只查询当前会话的本地企业知识库，不注册 RAGFlow、网络或数据库工具。
knowledge_base_agent = {
    "name": sub_agents_content["local_knowledge"]["name"],
    "description": sub_agents_content["local_knowledge"]["description"],
    "system_prompt": sub_agents_content["local_knowledge"]["system_prompt"],
    "tools": [search_local_knowledge_base],
}


def build_knowledge_agent_task(knowledge_base_id: str, query: str, *,
                               storage: LocalRAGStorage | None = None) -> str:
    """Create a task description with a KB authorized in the active session.

    Future callers pass this description to the Knowledge Agent; the Tool
    independently rechecks access when invoked. No session path is exposed.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Knowledge task query must be nonempty text")
    active_storage = storage if storage is not None else LocalRAGStorage()
    active_storage.get_knowledge_base(knowledge_base_id)
    return "企业内部知识检索任务：" + json.dumps(
        {"knowledge_base_id": knowledge_base_id, "query": query}, ensure_ascii=False,
    )
