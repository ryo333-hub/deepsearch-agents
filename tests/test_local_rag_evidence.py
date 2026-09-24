"""Evidence fidelity, exact source spans and bounded pure formatting."""

from unittest import TestCase
from unittest.mock import patch

from pydantic import ValidationError

from app.local_rag.config import MAX_EVIDENCE_TEXT_LENGTH
from app.local_rag.evidence import EvidenceBuildError, build_evidence, format_evidence_for_context
from app.local_rag.embeddings import LocalEmbeddingAdapter
from app.local_rag.ingestion import ingest_document
from app.local_rag.retrieval import LocalRAGRetriever
from app.local_rag.schemas import Citation, Evidence, RetrievalHit
from app.local_rag.vector_store import LocalVectorStore
from local_rag_test_support import OfflineDocumentTestCase
from test_local_rag_embeddings import FakeModel


KB = "kb_" + "a" * 32
DOC = "doc_" + "b" * 32


def hit(rank=1, *, text="第一行\n第二行\n第三行", start_char=40, start_line=4,
        page=2, heading="活动效果", score=0.81234567, document_name="618活动复盘.md",
        chunk_id=None, knowledge_base_id=KB):
    chunk_id = chunk_id or "chunk_" + f"{rank:032x}"
    citation = Citation(citation_id=f"C{rank}", knowledge_base_id=knowledge_base_id,
                        document_id=DOC, document_name=document_name, chunk_id=chunk_id,
                        score=score, score_type="inner_product", page=page,
                        start_line=start_line, end_line=start_line + text[:-1].count("\n"),
                        heading=heading, source_block_index=3,
                        start_char=start_char, end_char=start_char + len(text))
    return RetrievalHit(chunk_id=chunk_id, document_id=DOC, rank=rank,
                        score=score, text=text, citation=citation)


class EvidenceTests(TestCase):
    def test_single_hit_preserves_text_and_all_source_fields(self):
        original = hit()
        result = build_evidence((original,))
        self.assertEqual(len(result), 1)
        item = result[0]
        self.assertIsInstance(item, Evidence)
        self.assertEqual(item.text, original.text)
        self.assertFalse(item.truncated)
        self.assertEqual(item.citation, original.citation)
        self.assertEqual(item.citation.document_name, "618活动复盘.md")
        self.assertEqual((item.citation.page, item.citation.start_line,
                          item.citation.end_line, item.citation.heading), (2, 4, 6, "活动效果"))
        self.assertEqual((item.citation.source_block_index, item.citation.start_char,
                          item.citation.end_char), (3, 40, 51))
        self.assertEqual((item.citation.knowledge_base_id, item.citation.document_id,
                          item.citation.chunk_id), (KB, DOC, original.chunk_id))
        self.assertEqual(item.citation.score, 0.81234567)

    def test_multiple_hits_keep_order_score_and_assign_sequential_ids(self):
        original = (hit(3, text="第三"), hit(1, text="第一"), hit(2, text="第二"))
        result = build_evidence(original)
        self.assertEqual([e.text for e in result], [h.text for h in original])
        self.assertEqual([e.citation.chunk_id for e in result], [h.chunk_id for h in original])
        self.assertEqual([e.citation.citation_id for e in result], ["C1", "C2", "C3"])
        self.assertEqual([e.citation.score for e in result], [h.score for h in original])

    def test_empty_hits_and_empty_format_are_empty(self):
        self.assertEqual(build_evidence(()), ())
        self.assertEqual(format_evidence_for_context(()), "")

    def test_duplicate_chunk_is_rejected(self):
        with self.assertRaisesRegex(EvidenceBuildError, "Duplicate chunk"):
            build_evidence((hit(1), hit(2, chunk_id=hit(1).chunk_id)))

    def test_mixed_knowledge_bases_rejected(self):
        with self.assertRaises(EvidenceBuildError):
            build_evidence((hit(1), hit(2, knowledge_base_id="kb_" + "c" * 32)))

    def test_invalid_input_and_budget_rejected(self):
        for value in (None, "text", b"bytes", [object()]):
            with self.subTest(value=value), self.assertRaises(EvidenceBuildError):
                build_evidence(value)
        for value in (0, -1, 1.5, True, MAX_EVIDENCE_TEXT_LENGTH + 1):
            with self.subTest(value=value), self.assertRaises(EvidenceBuildError):
                build_evidence((hit(),), max_evidence_chars=value)

    def test_citation_missing_rejected(self):
        invalid = hit().model_copy(update={"citation": None})
        with self.assertRaises(EvidenceBuildError):
            build_evidence((invalid,))

    def test_empty_body_rejected(self):
        for text in ("", "   ", "\n"):
            invalid = hit().model_copy(update={"text": text})
            with self.subTest(text=text), self.assertRaises(EvidenceBuildError):
                build_evidence((invalid,))

    def test_nan_and_infinity_score_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            invalid = hit().model_copy(update={"score": value})
            with self.subTest(value=value), self.assertRaises(EvidenceBuildError):
                build_evidence((invalid,))

    def test_missing_score_type_rejected(self):
        original = hit()
        invalid = original.model_copy(update={
            "citation": original.citation.model_copy(update={"score_type": None}),
        })
        with self.assertRaises(EvidenceBuildError):
            build_evidence((invalid,))

    def test_invalid_char_range_rejected(self):
        original = hit()
        invalid = original.model_copy(update={
            "citation": original.citation.model_copy(update={"end_char": 41}),
        })
        with self.assertRaises(EvidenceBuildError):
            build_evidence((invalid,))

    def test_start_after_end_rejected(self):
        original = hit()
        invalid = original.model_copy(update={
            "citation": original.citation.model_copy(update={"start_char": 90}),
        })
        with self.assertRaises(EvidenceBuildError):
            build_evidence((invalid,))

    def test_invalid_line_range_rejected(self):
        original = hit()
        invalid = original.model_copy(update={
            "citation": original.citation.model_copy(update={"end_line": 2}),
        })
        with self.assertRaises(EvidenceBuildError):
            build_evidence((invalid,))

    def test_line_range_overstates_actual_text_rejected(self):
        original = hit()
        invalid = original.model_copy(update={
            "citation": original.citation.model_copy(update={"end_line": 10}),
        })
        with self.assertRaises(EvidenceBuildError):
            build_evidence((invalid,))

    def test_exact_budget_does_not_truncate(self):
        original = hit(text="ABCDE")
        result = build_evidence((original,), max_evidence_chars=5)[0]
        self.assertFalse(result.truncated)
        self.assertEqual(result.citation.end_char, 45)

    def test_truncation_reduces_char_and_line_ranges(self):
        original = hit()
        item = build_evidence((original,), max_evidence_chars=5)[0]
        self.assertEqual(item.text, original.text[:5])
        self.assertTrue(item.truncated)
        self.assertEqual((item.citation.start_char, item.citation.end_char), (40, 45))
        self.assertEqual((item.citation.start_line, item.citation.end_line), (4, 5))
        self.assertEqual((item.citation.page, item.citation.heading), (2, "活动效果"))
        self.assertEqual(item.citation.score, original.score)

    def test_prefix_ending_at_newline_does_not_cite_next_line(self):
        item = build_evidence((hit(),), max_evidence_chars=4)[0]
        self.assertEqual(item.text, "第一行\n")
        self.assertEqual(item.citation.end_line, 4)
        self.assertEqual(item.citation.end_char, 44)

    def test_nontruncated_missing_source_offset_keeps_existing_contract(self):
        original = hit()
        citation = original.citation.model_copy(update={
            "source_block_index": None, "start_char": None, "end_char": None,
        })
        item = build_evidence((original.model_copy(update={"citation": citation}),))[0]
        self.assertEqual(item.text, original.text)
        self.assertFalse(item.truncated)

    def test_truncation_without_source_offset_rejected(self):
        original = hit()
        citation = original.citation.model_copy(update={
            "source_block_index": None, "start_char": None, "end_char": None,
        })
        with self.assertRaises(EvidenceBuildError):
            build_evidence((original.model_copy(update={"citation": citation}),),
                           max_evidence_chars=5)

    def test_pdf_page_without_lines_stays_page_scoped(self):
        original = hit()
        citation = original.citation.model_copy(update={
            "start_line": None, "end_line": None,
        })
        item = build_evidence((original.model_copy(update={"citation": citation}),),
                              max_evidence_chars=5)[0]
        self.assertEqual(item.citation.page, 2)
        self.assertIsNone(item.citation.start_line)
        self.assertEqual(item.citation.end_char, 45)

    def test_evidence_schema_rejects_mismatched_text_span(self):
        original = build_evidence((hit(),))[0]
        with self.assertRaises(ValidationError):
            Evidence(text="unrelated", citation=original.citation)

    def test_evidence_schema_rejects_unlocatable_truncation(self):
        original = hit()
        citation = original.citation.model_copy(update={
            "source_block_index": None, "start_char": None, "end_char": None,
        })
        with self.assertRaises(ValidationError):
            Evidence(text="第一行", citation=citation, truncated=True)

    def test_json_roundtrip_preserves_truncated_flag_and_source(self):
        original = build_evidence((hit(),), max_evidence_chars=5)[0]
        self.assertEqual(Evidence.model_validate_json(original.model_dump_json()), original)

    def test_formatter_stable_and_uses_actual_source_range(self):
        first = build_evidence((hit(),), max_evidence_chars=5)[0]
        output = format_evidence_for_context((first,))
        self.assertEqual(output, format_evidence_for_context((first,)))
        for expected in ("[C1]", "618活动复盘.md", "第 2 页", "第 4-5 行",
                         "标题「活动效果」", "字符 [40,45)", "0.812346", "已裁剪：是", first.text):
            self.assertIn(expected, output)
        self.assertNotIn("第三行", output)

    def test_formatter_rejects_misnumbered_or_duplicate_citation(self):
        items = build_evidence((hit(1), hit(2)))
        with self.assertRaises(EvidenceBuildError):
            format_evidence_for_context((items[1], items[0]))
        with self.assertRaises(EvidenceBuildError):
            format_evidence_for_context((items[0], items[0]))

    def test_formatter_rejects_malformed_evidence(self):
        item = build_evidence((hit(),))[0]
        for value in ("bad", item.model_copy(update={"text": ""})):
            with self.subTest(value=value), self.assertRaises(EvidenceBuildError):
                format_evidence_for_context((value,))

    def test_builder_and_formatter_do_not_use_network(self):
        with patch("socket.socket.connect", side_effect=AssertionError("network")) as connect:
            items = build_evidence((hit(),))
            format_evidence_for_context(items)
            connect.assert_not_called()


class EvidencePipelineTests(OfflineDocumentTestCase):
    def test_loader_chunker_storage_embedding_index_retrieval_to_evidence(self):
        kb = self.storage.create_knowledge_base("虚构经营资料")
        chunks = []
        for name, content in (
            ("618活动复盘.md", "618 美妆活动流量上涨，但结算页转化率未同步提升。"),
            ("美妆销售分析.txt", "美妆销售分析显示访问量上涨，订单转化率和客单价下降。"),
            ("库存补货SOP.txt", "库存低于安全阈值时执行补货流程。"),
            ("广告投放规范.md", "广告素材应记录预算和投放时段。"),
        ):
            result = ingest_document(kb.knowledge_base_id, self.upload(name, content),
                                     storage=self.storage, loader=self.loader)
            chunks.extend(result.chunks)
        adapter = LocalEmbeddingAdapter(model_factory=lambda _: FakeModel())
        rows = adapter.embed_documents(chunks)
        LocalVectorStore(self.storage).build_index(kb.knowledge_base_id, rows, adapter.metadata)
        hits = LocalRAGRetriever(self.storage, adapter).retrieve(
            kb.knowledge_base_id,
            "为什么 618 美妆活动流量上涨但销售表现没有同步提升？", top_k=4,
        )
        evidence = build_evidence(hits, max_evidence_chars=24)
        self.assertEqual(len(evidence), len(hits))
        source = {chunk.chunk_id: chunk for chunk in chunks}
        self.assertEqual([e.citation.chunk_id for e in evidence], [h.chunk_id for h in hits])
        self.assertEqual([e.citation.citation_id for e in evidence],
                         [f"C{i}" for i in range(1, len(hits) + 1)])
        for item, hit in zip(evidence, hits, strict=True):
            chunk = source[item.citation.chunk_id]
            self.assertEqual(item.text, chunk.text[:24])
            self.assertEqual(item.citation.score, hit.score)
            self.assertEqual(item.citation.start_char, chunk.start_char)
            self.assertEqual(item.citation.end_char, chunk.start_char + len(item.text))
            self.assertEqual(item.citation.document_id, chunk.document_id)
        self.assertIn("[C1]", format_evidence_for_context(evidence))
