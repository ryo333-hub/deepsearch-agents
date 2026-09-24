"""Read-only enterprise knowledge lookup for the current request session.

The tool owns only orchestration. Storage authorization, query embedding,
vector validation, ranked mapping, and citations stay in Local RAG modules.
"""

from functools import lru_cache

from langchain_core.tools import tool

from app.api.context import (
    get_selected_knowledge_base_context,
    has_selected_knowledge_base_context,
)
from app.local_rag.evidence import build_evidence, format_evidence_for_context
from app.local_rag.retrieval import LocalRAGRetriever
from app.local_rag.schemas import Evidence
from app.local_rag.storage import KNOWLEDGE_BASE_ACCESS_ERROR, KnowledgeBaseAccessError


@lru_cache(maxsize=1)
def _get_retriever() -> LocalRAGRetriever:
    """One process-local retriever reuses the lazily loaded embedding adapter."""
    return LocalRAGRetriever()


@tool(response_format="content_and_artifact")
def search_local_knowledge_base(
    knowledge_base_id: str,
    query: str,
    top_k: int = 5,
) -> tuple[str, tuple[Evidence, ...]]:
    """查询当前会话中已建立索引的企业私有知识库，查找内部文档、运营 SOP、历史活动复盘、产品资料及分析报告。

    互联网最新信息请用网络搜索；结构化经营指标请用数据库工具。本工具只读，不创建、修改或重建知识库。

    Args:
        knowledge_base_id: 当前会话内的知识库 ID，不接受文件或会话路径。
        query: 要从企业资料中查证的具体问题。
        top_k: 最多返回的证据条数，默认 5。
    """
    # A Main task can use only its explicitly selected KB. Standalone Tool
    # calls retain their existing Storage-enforced current-session boundary.
    if (has_selected_knowledge_base_context()
            and knowledge_base_id != get_selected_knowledge_base_context()):
        raise KnowledgeBaseAccessError(KNOWLEDGE_BASE_ACCESS_ERROR)
    hits = _get_retriever().retrieve(knowledge_base_id, query, top_k=top_k)
    evidence = build_evidence(hits)
    if not evidence:
        return "未找到知识库证据。", evidence
    return f"找到 {len(evidence)} 条知识库证据：\n\n{format_evidence_for_context(evidence)}", evidence
