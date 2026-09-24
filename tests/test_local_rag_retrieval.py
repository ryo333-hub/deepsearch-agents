"""Deterministic ranking against real isolated Storage/Vector Store fixtures."""

from hashlib import sha256
import json
from unittest.mock import patch

import numpy as np

from app.api.context import reset_session_context, set_session_context, set_thread_context
from app.local_rag.config import EmbeddingConfig
from app.local_rag.embeddings import LocalEmbeddingAdapter, EmbeddingInputTooLong
from app.local_rag.ingestion import ingest_document
from app.local_rag.retrieval import LocalRAGRetriever, RetrievalError
from app.local_rag.schemas import ChunkEmbedding, EmbeddingVector, RetrievalHit
from app.local_rag.storage import KnowledgeBaseAccessError
from app.local_rag.vector_store import (
    LocalVectorStore, VectorIndexCompatibilityError, VectorIndexCorruptError, VectorIndexNotFoundError,
)
from local_rag_test_support import OfflineDocumentTestCase, pdf_bytes
from test_local_rag_embeddings import FakeModel


def padded(values):
    return tuple(float(x) for x in values) + (0.0,) * (384 - len(values))


class QueryModel(FakeModel):
    def __init__(self, values=(1, 0)):
        super().__init__()
        self.values = values

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return [padded(self.values) for _ in texts]


class RetrievalTests(OfflineDocumentTestCase):
    def setUp(self):
        super().setUp()
        self.model = QueryModel()
        self.adapter = LocalEmbeddingAdapter(model_factory=lambda _: self.model)
        self.store = LocalVectorStore(self.storage)
        self.retriever = LocalRAGRetriever(self.storage, self.adapter)

    def build(self, values=((1, 0), (0.8, 0.6), (0, 1)), sources=None, adapter=None):
        kb = self.storage.create_knowledge_base('检索测试')
        chunks = []
        for i, _ in enumerate(values):
            name, content = sources[i] if sources else (f'资料{i}.md', f'经营资料正文{i}。')
            uploaded = self.upload(name, content)
            ingested = ingest_document(kb.knowledge_base_id, uploaded,
                                       storage=self.storage, loader=self.loader)
            self.assertEqual(len(ingested.chunks), 1)
            chunks.extend(ingested.chunks)
        rows = tuple(ChunkEmbedding(chunk_id=c.chunk_id, dimension=384, vector=padded(v))
                     for c, v in zip(chunks, values, strict=True))
        self.store.build_index(kb.knowledge_base_id, rows, (adapter or self.adapter).metadata)
        return kb.knowledge_base_id, chunks

    def index_json(self, kb):
        root = self.base / 'store' / 'sessions' / 'session_session-A' / kb / 'vector_index'
        pointer = json.loads((root / 'current.json').read_text())
        return root / 'generations' / pointer['generation'] / 'index.json'

    def mutate_index(self, kb, **changes):
        path = self.index_json(kb)
        data = json.loads(path.read_text(encoding='utf-8'))
        data.update(changes)
        path.write_text(json.dumps(data), encoding='utf-8')

    def test_single_vector(self):
        kb, chunks = self.build(((1, 0),))
        hits = self.retriever.retrieve(kb, '查询', top_k=1)
        self.assertIsInstance(hits, tuple)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk_id, chunks[0].chunk_id)
        self.assertEqual(hits[0].score, 1.0)

    def test_top_one(self):
        kb, chunks = self.build(((0, 1), (1, 0), (0.8, 0.6)))
        hits = self.retriever.retrieve(kb, '查询', top_k=1)
        self.assertEqual([h.chunk_id for h in hits], [chunks[1].chunk_id])

    def test_top_k_and_descending_score(self):
        kb, chunks = self.build(((0, 1), (0.8, 0.6), (1, 0)))
        hits = self.retriever.retrieve(kb, '查询', top_k=2)
        self.assertEqual([h.chunk_id for h in hits], [chunks[2].chunk_id, chunks[1].chunk_id])
        np.testing.assert_allclose([h.score for h in hits], [1, 0.8])
        self.assertEqual([h.rank for h in hits], [1, 2])

    def test_top_k_larger_than_index_returns_all(self):
        kb, _ = self.build()
        self.assertEqual(len(self.retriever.retrieve(kb, '查询', top_k=1000)), 3)

    def test_top_k_zero_rejected(self):
        with self.assertRaises(RetrievalError):
            self.retriever.retrieve('invalid', '查询', top_k=0)

    def test_top_k_negative_rejected(self):
        with self.assertRaises(RetrievalError):
            self.retriever.retrieve('invalid', '查询', top_k=-1)

    def test_top_k_non_integer_rejected(self):
        for value in (True, 1.5, '2', None):
            with self.subTest(value=value), self.assertRaises(RetrievalError):
                self.retriever.retrieve('invalid', '查询', top_k=value)
        self.assertFalse(self.adapter.is_loaded)

    def test_empty_query_rejected_before_model_load(self):
        for query in ('', ' \n ', None, 42):
            with self.subTest(query=query), self.assertRaises(RetrievalError):
                self.retriever.retrieve('invalid', query)
        self.assertFalse(self.adapter.is_loaded)

    def test_query_token_limit_uses_adapter(self):
        kb, _ = self.build()
        with self.assertRaises(EmbeddingInputTooLong):
            self.retriever.retrieve(kb, '长' * 512)
        self.assertEqual(self.model.calls, [])

    def test_query_prefix_and_model_reuse(self):
        kb, _ = self.build()
        self.assertFalse(self.adapter.is_loaded)
        self.retriever.retrieve(kb, '经营问题')
        self.retriever.retrieve(kb, '第二个问题')
        self.assertIs(self.adapter._model, self.model)
        self.assertEqual([c[0] for c in self.model.calls], [['query: 经营问题'], ['query: 第二个问题']])
        self.assertTrue(all(c[1]['normalize_embeddings'] for c in self.model.calls))

    def malformed_query(self, result):
        kb, _ = self.build()
        with patch.object(self.adapter, 'embed_query', return_value=result):
            with self.assertRaises(RetrievalError):
                self.retriever.retrieve(kb, '查询')

    def test_query_dimension_mismatch(self):
        self.malformed_query(EmbeddingVector(vector=(1.0, 0.0), dimension=2))

    def test_nan_query_rejected(self):
        self.malformed_query(EmbeddingVector.model_construct(vector=padded((float('nan'),)), dimension=384))

    def test_infinity_query_rejected(self):
        self.malformed_query(EmbeddingVector.model_construct(vector=padded((float('inf'),)), dimension=384))

    def test_zero_query_rejected(self):
        self.malformed_query(EmbeddingVector(vector=padded((0,)), dimension=384))

    def test_non_unit_query_rejected_for_normalized_index(self):
        self.malformed_query(EmbeddingVector(vector=padded((2,)), dimension=384))

    def test_malformed_query_contract_rejected(self):
        self.malformed_query([1.0, 0.0])

    def test_ties_preserve_original_index_row_order(self):
        kb, chunks = self.build(((0, 1), (1, 0), (1, 0), (0, -1)))
        a = self.retriever.retrieve(kb, '查询', top_k=4)
        b = self.retriever.retrieve(kb, '查询', top_k=4)
        self.assertEqual(a, b)
        self.assertEqual([h.chunk_id for h in a], [chunks[i].chunk_id for i in (1, 2, 0, 3)])

    def test_negative_scores_not_thresholded(self):
        kb, _ = self.build(((-1, 0), (0, 1)))
        self.assertEqual([h.score for h in self.retriever.retrieve(kb, '查询')], [0.0, -1.0])

    def test_score_is_not_clipped(self):
        kb, _ = self.build(((1.00001, 0),))
        hit = self.retriever.retrieve(kb, '查询')[0]
        self.assertGreater(hit.score, 1.0)

    def test_non_normalized_vectors_use_cosine_not_raw_dot(self):
        model = QueryModel((2, 0))
        adapter = LocalEmbeddingAdapter(EmbeddingConfig(normalize_embeddings=False), model_factory=lambda _: model)
        kb, chunks = self.build(((0.8, 0.6), (0, 10), (0.1, 0)), adapter=adapter)
        hits = LocalRAGRetriever(self.storage, adapter).retrieve(kb, '查询')
        self.assertEqual([h.chunk_id for h in hits], [chunks[i].chunk_id for i in (2, 0, 1)])
        np.testing.assert_allclose([h.score for h in hits], [1, 0.8, 0], atol=1e-7)
        self.assertTrue(all(h.citation.score_type == 'cosine_similarity' for h in hits))

    def test_chunk_document_text_and_citation_mapping(self):
        kb, chunks = self.build(((0, 1), (1, 0)))
        hits = self.retriever.retrieve(kb, '查询')
        for hit, chunk in zip(hits, reversed(chunks), strict=True):
            self.assertEqual((hit.chunk_id, hit.document_id, hit.text),
                             (chunk.chunk_id, chunk.document_id, chunk.text))
            self.assertEqual(hit.citation.knowledge_base_id, kb)
            self.assertEqual(hit.citation.chunk_id, hit.chunk_id)
            self.assertEqual(hit.citation.document_id, hit.document_id)
            self.assertEqual(hit.citation.score, hit.score)
            self.assertEqual(hit.citation.score_type, 'inner_product')
            self.assertEqual(hit.citation.citation_id, f'C{hit.rank}')

    def test_line_heading_and_source_span_preserved(self):
        kb, chunks = self.build(((1, 0),), [('资料0.md', '# 标题0')])
        citation = self.retriever.retrieve(kb, '查询')[0].citation
        self.assertEqual(citation.document_name, '资料0.md')
        self.assertEqual(citation.heading, '标题0')
        for name in ('start_line', 'end_line', 'source_block_index', 'start_char', 'end_char'):
            self.assertIsNotNone(getattr(citation, name))
            self.assertEqual(getattr(citation, name), getattr(chunks[0], name))

    def test_pdf_page_preserved(self):
        kb, chunks = self.build(((1, 0),), [('manual.pdf', pdf_bytes(['Inventory guide']))])
        citation = self.retriever.retrieve(kb, '查询')[0].citation
        self.assertEqual(citation.page, 1)
        self.assertEqual(citation.page, chunks[0].page)

    def test_missing_index_on_empty_kb_rejected(self):
        kb = self.storage.create_knowledge_base('空知识库')
        with self.assertRaises(VectorIndexNotFoundError):
            self.retriever.retrieve(kb.knowledge_base_id, '查询')

    def test_missing_chunk_rejected(self):
        kb, chunks = self.build()
        path = self.base / 'store' / 'sessions' / 'session_session-A' / kb / 'documents' / chunks[0].document_id / 'chunks.jsonl'
        path.write_text('', encoding='utf-8')
        with self.assertRaises(KnowledgeBaseAccessError):
            self.retriever.retrieve(kb, '查询')

    def test_model_mismatch_rejected(self):
        kb, _ = self.build()
        self.mutate_index(kb, embedding_model_name='another-model')
        with self.assertRaises(VectorIndexCompatibilityError):
            self.retriever.retrieve(kb, '查询')

    def test_revision_mismatch_rejected(self):
        kb, _ = self.build()
        self.mutate_index(kb, embedding_model_revision='another-revision')
        with self.assertRaises(VectorIndexCompatibilityError):
            self.retriever.retrieve(kb, '查询')

    def test_index_dimension_mismatch_rejected(self):
        kb, _ = self.build()
        self.mutate_index(kb, embedding_dimension=768)
        with self.assertRaises(VectorIndexCompatibilityError):
            self.retriever.retrieve(kb, '查询')

    def test_normalization_mismatch_rejected(self):
        kb, _ = self.build()
        self.mutate_index(kb, normalize_embeddings=False)
        with self.assertRaises(VectorIndexCompatibilityError):
            self.retriever.retrieve(kb, '查询')

    def test_chunk_version_mismatch_rejected(self):
        kb, _ = self.build()
        self.mutate_index(kb, chunk_version='old-version')
        with self.assertRaises(VectorIndexCompatibilityError):
            self.retriever.retrieve(kb, '查询')

    def test_source_digest_mismatch_rejected(self):
        kb, _ = self.build()
        self.mutate_index(kb, source_chunks_sha256='0' * 64)
        with self.assertRaises(VectorIndexCompatibilityError):
            self.retriever.retrieve(kb, '查询')

    def test_damaged_vectors_rejected(self):
        kb, _ = self.build()
        self.index_json(kb).with_name('vectors.npy').write_bytes(b'corrupt')
        with self.assertRaises(VectorIndexCorruptError):
            self.retriever.retrieve(kb, '查询')

    def test_knowledge_base_isolation(self):
        first, chunks_a = self.build(((1, 0),))
        second, chunks_b = self.build(((1, 0),))
        a = self.retriever.retrieve(first, '查询')
        b = self.retriever.retrieve(second, '查询')
        self.assertEqual(a[0].chunk_id, chunks_a[0].chunk_id)
        self.assertEqual(b[0].chunk_id, chunks_b[0].chunk_id)
        self.assertNotEqual(a[0].chunk_id, b[0].chunk_id)

    def test_session_isolation_blocks_guess_before_embedding(self):
        kb, chunks_a = self.build(((1, 0),))
        other_dir = self.upload_root / 'session_session-B'
        other_dir.mkdir()
        s = set_session_context(str(other_dir))
        t = set_thread_context('session-B')
        try:
            self.session_dir = other_dir
            other, chunks_b = self.build(((1, 0),))
            with self.assertRaises(KnowledgeBaseAccessError):
                self.retriever.retrieve(kb, '查询')
            self.assertFalse(self.adapter.is_loaded)
            self.assertEqual(self.retriever.retrieve(other, '查询')[0].chunk_id, chunks_b[0].chunk_id)
            self.assertNotEqual(chunks_a[0].chunk_id, chunks_b[0].chunk_id)
        finally:
            reset_session_context(s, t)
        with self.assertRaises(KnowledgeBaseAccessError):
            self.retriever.retrieve(other, '查询')

    def test_path_input_rejected(self):
        for value in ('../kb', 'E:\\secret', 'kb_' + 'a' * 32):
            with self.subTest(value=value), self.assertRaises(KnowledgeBaseAccessError):
                self.retriever.retrieve(value, '查询')
        self.assertFalse(self.adapter.is_loaded)

    def test_hit_contract_rejects_mismatched_citation(self):
        kb, _ = self.build(((1, 0),))
        hit = self.retriever.retrieve(kb, '查询')[0]
        with self.assertRaises(ValueError):
            RetrievalHit.model_validate({**hit.model_dump(), 'score': 0.3})

    def test_retrieval_does_not_write_files(self):
        kb, _ = self.build()
        def snapshot():
            return {str(p.relative_to(self.base)): sha256(p.read_bytes()).hexdigest()
                    for p in self.base.rglob('*') if p.is_file()}
        before = snapshot()
        self.retriever.retrieve(kb, '查询')
        self.assertEqual(before, snapshot())

    def test_multiple_chunks_in_same_document_keep_source_locations(self):
        kb = self.storage.create_knowledge_base('多分块文档')
        name = self.upload('复盘.md', '# 复盘\n\n流量上涨但转化不足。\n\n需要改进结算流程。')
        result = ingest_document(kb.knowledge_base_id, name, storage=self.storage, loader=self.loader)
        self.assertGreater(len(result.chunks), 1)
        rows = tuple(ChunkEmbedding(chunk_id=c.chunk_id, dimension=384, vector=padded((1, 0)))
                     for c in result.chunks)
        self.store.build_index(kb.knowledge_base_id, rows, self.adapter.metadata)
        hits = self.retriever.retrieve(kb.knowledge_base_id, '查询', top_k=100)
        self.assertEqual([h.chunk_id for h in hits], [c.chunk_id for c in result.chunks])
        self.assertEqual([h.text for h in hits], [c.text for c in result.chunks])
        self.assertEqual([h.citation.start_line for h in hits], [c.start_line for c in result.chunks])

    def test_mapping_missing_chunk_never_silently_skipped(self):
        kb, chunks = self.build(((1, 0),))
        original = self.storage.get_document_ingestion
        calls = 0
        def missing_after_index_validation(*args):
            nonlocal calls
            calls += 1
            loaded = original(*args)
            return loaded if calls == 1 else loaded.model_copy(update={'chunks': ()})
        with patch.object(self.storage, 'get_document_ingestion', side_effect=missing_after_index_validation):
            with self.assertRaises(VectorIndexCorruptError):
                self.retriever.retrieve(kb, '查询')

    def test_empty_manifest_index_is_rejected(self):
        kb, _ = self.build(((1, 0),))
        self.mutate_index(kb, vector_count=0, chunk_ids=[], document_ids=[])
        with self.assertRaises(VectorIndexCorruptError):
            self.retriever.retrieve(kb, '查询')
