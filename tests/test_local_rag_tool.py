"""Offline LangChain tool contract and Local RAG read-only integration."""

from hashlib import sha256
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from app.api.context import reset_session_context, set_session_context, set_thread_context
from app.local_rag.embeddings import LocalEmbeddingAdapter
from app.local_rag.evidence import EvidenceBuildError
from app.local_rag.ingestion import ingest_document
from app.local_rag.retrieval import LocalRAGRetriever, RetrievalError
from app.local_rag.storage import KnowledgeBaseAccessError
from app.local_rag.vector_store import (
    LocalVectorStore, VectorIndexCompatibilityError, VectorIndexCorruptError,
    VectorIndexNotFoundError,
)
from app.tools import local_knowledge_base_tool as module
from app.tools.local_knowledge_base_tool import _get_retriever as cached_retriever_factory
from local_rag_test_support import OfflineDocumentTestCase, pdf_bytes
from test_local_rag_embeddings import FakeModel
from test_local_rag_evidence import KB, hit


class ToolContractTests(TestCase):
    def setUp(self):
        self.retriever = Mock()
        self.retriever.retrieve.return_value = (hit(1), hit(2, text="第二条资料"))
        mocked = patch.object(module, "_get_retriever", return_value=self.retriever)
        mocked.start()
        self.addCleanup(mocked.stop)

    def call(self, **changes):
        args = {"knowledge_base_id": KB, "query": "活动转化怎么样？", **changes}
        return module.search_local_knowledge_base.invoke(args)

    def test_name_description_and_business_boundaries(self):
        tool = module.search_local_knowledge_base
        self.assertEqual(tool.name, "search_local_knowledge_base")
        for expected in ("当前会话", "企业私有知识库", "运营 SOP", "活动复盘",
                         "互联网", "网络搜索", "经营指标", "数据库", "只读"):
            self.assertIn(expected, tool.description)
        for forbidden in ("NumPy", "vectors.npy", "384", "SentenceTransformer"):
            self.assertNotIn(forbidden, tool.description)

    def test_schema_exposes_only_kb_query_top_k(self):
        tool = module.search_local_knowledge_base
        self.assertEqual(set(tool.args), {"knowledge_base_id", "query", "top_k"})
        for forbidden in ("session_id", "thread_id", "session_dir", "index_path", "model_name"):
            self.assertNotIn(forbidden, tool.args)
        self.assertEqual(tool.response_format, "content_and_artifact")
        self.assertEqual(tool.args["knowledge_base_id"]["type"], "string")
        self.assertEqual(tool.args["query"]["type"], "string")
        self.assertEqual(tool.args["top_k"]["type"], "integer")

    def test_explicit_arguments_forwarded_unchanged(self):
        text = self.call(top_k=2)
        self.retriever.retrieve.assert_called_once_with(KB, "活动转化怎么样？", top_k=2)
        self.assertIn("找到 2 条知识库证据", text)

    def test_default_top_k_is_five(self):
        self.call()
        self.retriever.retrieve.assert_called_once_with(KB, "活动转化怎么样？", top_k=5)

    def test_content_and_structured_artifact_retain_citations(self):
        call = {"type": "tool_call", "name": module.search_local_knowledge_base.name,
                "args": {"knowledge_base_id": KB, "query": "活动转化怎么样？", "top_k": 2},
                "id": "local-rag-test-1"}
        result = module.search_local_knowledge_base.invoke(call)
        self.assertIsInstance(result, ToolMessage)
        self.assertEqual(result.tool_call_id, "local-rag-test-1")
        self.assertEqual([e.citation.citation_id for e in result.artifact], ["C1", "C2"])
        self.assertEqual([e.citation.score for e in result.artifact], [hit(1).score, hit(2).score])
        self.assertEqual([e.citation.document_name for e in result.artifact],
                         ["618活动复盘.md", "618活动复盘.md"])
        for expected in ("[C1]", "[C2]", "第 2 页", "第 4-6 行", "标题「活动效果」",
                         "字符 [40,51)", "0.812346"):
            self.assertIn(expected, result.content)

    def test_empty_retrieval_returns_clear_message_and_empty_artifact(self):
        self.retriever.retrieve.return_value = ()
        result = module.search_local_knowledge_base.invoke({
            "type": "tool_call", "name": module.search_local_knowledge_base.name,
            "args": {"knowledge_base_id": KB, "query": "问题"}, "id": "empty"})
        self.assertEqual(result.content, "未找到知识库证据。")
        self.assertEqual(result.artifact, ())

    def test_retrieval_error_types_propagate(self):
        for error in (KnowledgeBaseAccessError("KB 不可访问"),
                      VectorIndexNotFoundError("索引不存在"),
                      VectorIndexCorruptError("索引损坏"),
                      VectorIndexCompatibilityError("模型不兼容"),
                      RetrievalError("非法查询")):
            self.retriever.retrieve.side_effect = error
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                self.call()

    def test_evidence_error_propagates_without_fallback(self):
        self.retriever.retrieve.return_value = (hit(1), hit(1))
        with self.assertRaises(EvidenceBuildError):
            self.call()
        self.retriever.retrieve.assert_called_once()

    def test_tool_does_not_recalculate_or_reformat_citations(self):
        with patch.object(module, "build_evidence", wraps=module.build_evidence) as builder:
            with patch.object(module, "format_evidence_for_context",
                              wraps=module.format_evidence_for_context) as formatter:
                self.call()
        builder.assert_called_once_with(self.retriever.retrieve.return_value)
        formatter.assert_called_once_with(module.build_evidence(self.retriever.retrieve.return_value))

    def test_empty_kb_and_query_are_rejected_by_retriever(self):
        real = LocalRAGRetriever()
        with patch.object(module, "_get_retriever", return_value=real):
            with self.assertRaises((KnowledgeBaseAccessError, ValueError)):
                self.call(knowledge_base_id="")
            with self.assertRaises(RetrievalError):
                self.call(query="  ")

    def test_invalid_top_k_is_rejected_by_retriever(self):
        self.retriever.retrieve.side_effect = RetrievalError("top_k must be positive")
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaises(RetrievalError):
                self.call(top_k=value)
        with self.assertRaises(ValidationError):
            self.call(top_k="bad")

    def test_no_network_and_only_one_retrieval_call(self):
        with patch("socket.socket.connect", side_effect=AssertionError("network")) as connect:
            self.call()
            connect.assert_not_called()
        self.retriever.retrieve.assert_called_once()


class ToolStorageTests(OfflineDocumentTestCase):
    def setUp(self):
        super().setUp()
        self.model = FakeModel()
        self.adapter = LocalEmbeddingAdapter(model_factory=lambda _: self.model)
        self.retriever = LocalRAGRetriever(self.storage, self.adapter)
        mocked = patch.object(module, "_get_retriever", return_value=self.retriever)
        mocked.start()
        self.addCleanup(mocked.stop)

    def create_index(self, contents=("活动复盘里有转化率下降。",)):
        kb = self.storage.create_knowledge_base("私有运营资料")
        chunks = []
        for number, text in enumerate(contents):
            result = ingest_document(kb.knowledge_base_id,
                                     self.upload(f"运营资料{number}.txt", text),
                                     storage=self.storage, loader=self.loader)
            chunks.extend(result.chunks)
        LocalVectorStore(self.storage).build_index(
            kb.knowledge_base_id, self.adapter.embed_documents(chunks), self.adapter.metadata)
        return kb.knowledge_base_id, chunks

    def call(self, kb, **changes):
        return module.search_local_knowledge_base.invoke({
            "knowledge_base_id": kb, "query": "运营问题", **changes})

    def snapshot(self):
        return {str(p.relative_to(self.base)): sha256(p.read_bytes()).hexdigest()
                for p in self.base.rglob("*") if p.is_file()}

    def test_real_storage_retrieval_read_only_and_no_rebuild(self):
        kb, chunks = self.create_index(("转化率下降。", "库存补货规范。"))
        before = self.snapshot()
        with patch("app.local_rag.vector_store.LocalVectorStore.build_index",
                   side_effect=AssertionError("rebuild forbidden")):
            content = self.call(kb)
        self.assertEqual(before, self.snapshot())
        self.assertIn("[C1]", content)
        self.assertIn("[C2]", content)
        self.assertTrue(all(c.chunk_id in content for c in chunks))

    def test_no_index_and_missing_kb_rejected(self):
        kb = self.storage.create_knowledge_base("未索引资料")
        with self.assertRaises(VectorIndexNotFoundError):
            self.call(kb.knowledge_base_id)
        with self.assertRaises(KnowledgeBaseAccessError):
            self.call("kb_" + "f" * 32)

    def test_cross_session_and_kb_isolation(self):
        kb_a, chunks_a = self.create_index()
        kb_b, chunks_b = self.create_index(("另一知识库独有资料。",))
        result_a = self.call(kb_a)
        self.assertIn(chunks_a[0].chunk_id, result_a)
        self.assertNotIn(chunks_b[0].chunk_id, result_a)
        other_dir = self.upload_root / "session_session-B"
        other_dir.mkdir()
        session = set_session_context(str(other_dir))
        thread = set_thread_context("session-B")
        try:
            with self.assertRaises(KnowledgeBaseAccessError):
                self.call(kb_a)
        finally:
            reset_session_context(session, thread)

    def test_adapter_instance_reused_for_repeated_calls(self):
        kb, _ = self.create_index()
        self.call(kb)
        first_model = self.adapter._model
        self.call(kb)
        self.assertIs(self.adapter._model, first_model)

    def test_process_default_retriever_is_cached(self):
        with patch.object(module, "LocalRAGRetriever") as construct:
            cached_retriever_factory.cache_clear()
            try:
                a = cached_retriever_factory()
                b = cached_retriever_factory()
                self.assertIs(a, b)
                construct.assert_called_once_with()
            finally:
                cached_retriever_factory.cache_clear()

    def test_pdf_page_preserved_in_agent_output(self):
        kb = self.storage.create_knowledge_base("PDF")
        name = self.upload("运营手册.pdf", pdf_bytes(["Inventory guide"]))
        result = ingest_document(kb.knowledge_base_id, name,
                                 storage=self.storage, loader=self.loader)
        LocalVectorStore(self.storage).build_index(
            kb.knowledge_base_id,
            self.adapter.embed_documents(result.chunks), self.adapter.metadata)
        self.assertIn("第 1 页", self.call(kb.knowledge_base_id))
