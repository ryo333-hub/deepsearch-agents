"""Adapter contracts without downloading/loading weights or contacting services."""

import builtins
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

from app.local_rag.config import (
    EMBEDDING_DIMENSION, EMBEDDING_DOCUMENT_PREFIX, EMBEDDING_MAX_INPUT_TOKENS,
    EMBEDDING_MODEL_REVISION, EmbeddingConfig,
)
from app.local_rag.embeddings import (
    EmbeddingDependencyError, EmbeddingError, EmbeddingInputError,
    EmbeddingInputTooLong, EmbeddingLoadError, EmbeddingOutputError,
    LocalEmbeddingAdapter, _load_local_model, get_embedding_adapter,
)
from app.local_rag.schemas import ChunkRecord, EmbeddingVector
from local_rag_test_support import OfflineDocumentTestCase


def chunk(text="活动转化率复盘", **kwargs):
    return ChunkRecord(document_id="doc_" + "1" * 32, chunk_index=0,
                       text=text, text_hash=sha256(text.encode()).hexdigest(), **kwargs)


class FakeTokenizer:
    model_max_length = EMBEDDING_MAX_INPUT_TOKENS

    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {"input_ids": [1] * (len(text) + 2)}


class FakeModel:
    max_seq_length = EMBEDDING_MAX_INPUT_TOKENS

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return EMBEDDING_DIMENSION

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        rows = []
        for text in texts:
            vector = [0.0] * EMBEDDING_DIMENSION
            vector[sum(text.encode()) % EMBEDDING_DIMENSION] = 1.0 if kwargs['normalize_embeddings'] else 2.0
            rows.append(vector)
        return rows


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.blockers = []
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.getaddrinfo"):
            p = patch(target, side_effect=AssertionError("Network prohibited"))
            self.blockers.append(p.start())
            self.addCleanup(p.stop)
        self.addCleanup(self.assert_no_network)
        self.model = FakeModel()
        self.factory = Mock(return_value=self.model)
        self.adapter = LocalEmbeddingAdapter(EmbeddingConfig(batch_size=2), model_factory=self.factory)

    def assert_no_network(self):
        for blocker in self.blockers:
            blocker.assert_not_called()

    def test_module_import_does_not_import_ml_libraries(self):
        real_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'transformers', 'sentence_transformers', 'numpy'}:
                self.fail("Embedding module imported a heavy ML dependency")
            return real_import(name, *args, **kwargs)
        spec = importlib.util.spec_from_file_location('_embedding_import_test',
                                                     Path('app/local_rag/embeddings.py').resolve())
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'_embedding_import_test': module}), patch('builtins.__import__', guarded):
            spec.loader.exec_module(module)
            self.assertFalse(module.LocalEmbeddingAdapter().is_loaded)

    def test_construction_and_metadata_are_lazy(self):
        self.assertFalse(self.adapter.is_loaded)
        self.assertEqual(self.adapter.metadata.model_revision, EMBEDDING_MODEL_REVISION)
        self.assertEqual(self.adapter.metadata.embedding_dimension, 384)
        self.assertEqual(self.adapter.metadata.max_input_tokens, 512)
        self.factory.assert_not_called()

    def test_model_reused_between_documents_and_queries(self):
        self.adapter.embed_documents([chunk()])
        self.adapter.embed_query('转化率')
        self.adapter.embed_query('inventory')
        self.factory.assert_called_once_with(self.adapter.config)

    def test_default_adapter_reused_without_loading(self):
        get_embedding_adapter.cache_clear()
        self.addCleanup(get_embedding_adapter.cache_clear)
        self.assertIs(get_embedding_adapter(), get_embedding_adapter())
        self.assertFalse(get_embedding_adapter().is_loaded)

    def test_concurrent_first_calls_load_only_once(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(self.adapter.embed_query, ['a', 'b', 'c', 'd']))
        self.assertEqual(len(results), 4)
        self.factory.assert_called_once()

    def test_documents_batched_and_order_preserved(self):
        chunks = [chunk(t) for t in ['first', '第二', 'third', '第四', 'last']]
        result = self.adapter.embed_documents(chunks)
        self.assertEqual([len(texts) for texts, _ in self.model.calls], [2, 2, 1])
        self.assertEqual([v.chunk_id for v in result], [c.chunk_id for c in chunks])
        for c, r in zip(chunks, result):
            self.assertEqual(r.vector[sum((EMBEDDING_DOCUMENT_PREFIX + c.text).encode()) % 384], 1)
            self.assertEqual(r.dimension, 384)

    def test_query_has_own_prefix_and_same_dimension(self):
        d = self.adapter.embed_documents([chunk()])[0]
        q = self.adapter.embed_query('  活动表现  ')
        self.assertEqual(q.dimension, d.dimension)
        self.assertEqual(self.model.calls[-1][0], ['query: 活动表现'])
        self.assertFalse(hasattr(q, 'chunk_id'))

    def test_normalization_true_and_cpu_forwarded(self):
        self.adapter.embed_query('x')
        opts = self.model.calls[0][1]
        self.assertTrue(opts['normalize_embeddings'])
        self.assertEqual(opts['device'], 'cpu')
        self.assertEqual(opts['prompt'], '')
        self.assertFalse(opts['show_progress_bar'])

    def test_normalization_false_for_both_paths(self):
        adapter = LocalEmbeddingAdapter(EmbeddingConfig(normalize_embeddings=False), model_factory=self.factory)
        d = adapter.embed_documents([chunk()])[0]
        q = adapter.embed_query('x')
        self.assertEqual(max(d.vector), 2)
        self.assertEqual(max(q.vector), 2)
        self.assertTrue(all(not opts['normalize_embeddings'] for _, opts in self.model.calls))

    def test_all_input_tokens_checked_before_any_encode(self):
        bad = chunk('中' * 1000)
        with self.assertRaises(EmbeddingInputTooLong) as raised:
            self.adapter.embed_documents([chunk('a'), chunk('b'), bad])
        self.assertEqual(raised.exception.input_id, bad.chunk_id)
        self.assertEqual(raised.exception.token_count, 1011)
        self.assertEqual(raised.exception.max_input_tokens, 512)
        self.assertIn(bad.chunk_id, str(raised.exception))
        self.assertEqual(self.model.calls, [])

    def test_exact_token_boundary_including_prefix_and_special_tokens(self):
        self.adapter.embed_documents([chunk('x' * (512 - len(EMBEDDING_DOCUMENT_PREFIX) - 2))])
        self.assertEqual(len(self.model.calls), 1)
        with self.assertRaises(EmbeddingInputTooLong):
            self.adapter.embed_documents([chunk('x' * (513 - len(EMBEDDING_DOCUMENT_PREFIX) - 2))])
        opts = self.model.tokenizer.calls[0][1]
        self.assertFalse(opts['truncation'])
        self.assertTrue(opts['add_special_tokens'])

    def test_query_overflow_reports_query_context(self):
        with self.assertRaises(EmbeddingInputTooLong) as raised:
            self.adapter.embed_query('字' * 512)
        self.assertEqual(raised.exception.input_id, 'query')
        self.assertEqual(raised.exception.token_count, 521)
        self.assertEqual(self.model.calls, [])

    def test_empty_document_sequence_does_not_load(self):
        self.assertEqual(self.adapter.embed_documents([]), ())
        self.factory.assert_not_called()

    def test_empty_or_nontext_query_rejected_before_loading(self):
        for text in ['', ' \n ', None, 123]:
            with self.subTest(text=text), self.assertRaises(EmbeddingInputError):
                self.adapter.embed_query(text)
        self.factory.assert_not_called()

    def test_whitespace_chunk_rejected_before_loading(self):
        with self.assertRaises(EmbeddingInputError):
            self.adapter.embed_documents([chunk(' \n ')])
        self.factory.assert_not_called()

    def test_invalid_chunk_types_rejected(self):
        for docs in ['text', [None], [{}], (x for x in [])]:
            with self.subTest(docs=type(docs)), self.assertRaises(EmbeddingInputError):
                self.adapter.embed_documents(docs)
        self.factory.assert_not_called()

    def test_unchecked_invalid_record_rejected(self):
        c = chunk().model_copy(update={'chunk_id': '../unsafe'})
        with self.assertRaises(EmbeddingInputError):
            self.adapter.embed_documents([c])
        self.factory.assert_not_called()

    def test_text_hash_mismatch_rejected(self):
        c = chunk().model_copy(update={'text': 'changed'})
        with self.assertRaisesRegex(EmbeddingInputError, 'text_hash mismatch'):
            self.adapter.embed_documents([c])

    def test_duplicate_chunk_id_rejected(self):
        c = chunk()
        with self.assertRaisesRegex(EmbeddingInputError, 'duplicate chunk_id'):
            self.adapter.embed_documents([c, c])

    def test_wrong_model_dimension_rejected_before_encode(self):
        self.model.get_sentence_embedding_dimension = lambda: 768
        with self.assertRaises(EmbeddingLoadError):
            self.adapter.embed_query('x')
        self.assertEqual(self.model.calls, [])

    def test_wrong_model_or_tokenizer_limit_rejected(self):
        for attr in ['model', 'tokenizer']:
            with self.subTest(attr=attr):
                model = FakeModel()
                if attr == 'model': model.max_seq_length = 256
                else: model.tokenizer.model_max_length = 256
                adapter = LocalEmbeddingAdapter(model_factory=lambda _: model)
                with self.assertRaises(EmbeddingLoadError): adapter.embed_query('x')

    def test_missing_tokenizer_rejected(self):
        self.model.tokenizer = None
        with self.assertRaisesRegex(EmbeddingLoadError, 'tokenizer'):
            self.adapter.embed_query('x')

    def test_tokenizer_failure_or_malformed_tokens_rejected(self):
        for output in [{}, {'input_ids': [[1, 2]]}, {'input_ids': []}]:
            with self.subTest(output=output):
                tok = Mock(return_value=output, model_max_length=512)
                self.model.tokenizer = tok
                with self.assertRaises(EmbeddingInputError): self.adapter.embed_query('x')
        self.assertEqual(self.model.calls, [])

    def test_model_load_failure_explicit_no_fallback(self):
        self.factory.side_effect = OSError('private path')
        with self.assertRaisesRegex(EmbeddingLoadError, 'OSError') as raised:
            self.adapter.embed_query('x')
        self.assertNotIn('private path', str(raised.exception))
        self.assertFalse(self.adapter.is_loaded)
        self.factory.assert_called_once()

    def test_dependency_missing_explicit(self):
        real_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == 'sentence_transformers': raise ModuleNotFoundError(name)
            return real_import(name, *args, **kwargs)
        with patch('builtins.__import__', guarded), self.assertRaises(EmbeddingDependencyError):
            _load_local_model(EmbeddingConfig())

    def test_encode_exception_does_not_retry_or_expose_text(self):
        self.model.encode = Mock(side_effect=RuntimeError('private text'))
        with self.assertRaises(EmbeddingError) as raised:
            self.adapter.embed_query('secret')
        self.assertNotIn('private text', str(raised.exception))
        self.model.encode.assert_called_once()

    def assert_bad_output(self, output):
        self.model.encode = Mock(return_value=output)
        with self.assertRaises(EmbeddingOutputError): self.adapter.embed_query('x')

    def test_vector_count_mismatch(self):
        self.assert_bad_output([])
        self.assert_bad_output([[1.0] * 384, [1.0] * 384])

    def test_dimension_mismatch(self):
        self.assert_bad_output([[1.0] * 383])

    def test_document_query_dimension_mismatch(self):
        self.adapter.embed_documents([chunk()])
        self.assert_bad_output([[1.0] * 768])

    def test_nan_rejected(self):
        self.assert_bad_output([[float('nan')] + [0.0] * 383])

    def test_infinity_rejected(self):
        for v in [float('inf'), float('-inf')]: self.assert_bad_output([[v] + [0.0] * 383])

    def test_nonnumeric_and_malformed_outputs_rejected(self):
        for raw in [None, [None], [1], [['1'] * 384], [[True] * 384], [[[]] * 384]]:
            with self.subTest(raw=type(raw)): self.assert_bad_output(raw)

    def test_zero_and_unnormalized_vectors_rejected(self):
        self.assert_bad_output([[0.0] * 384])
        self.assert_bad_output([[2.0] + [0.0] * 383])

    def test_results_are_immutable_and_contract_checks_dimension(self):
        result = self.adapter.embed_query('x')
        with self.assertRaises(ValueError): result.dimension = 1
        with self.assertRaises(ValueError): EmbeddingVector(vector=(1.0,), dimension=2)
        with self.assertRaises(ValueError): EmbeddingVector(vector=(float('nan'),), dimension=1)

    def test_config_validation(self):
        for kwargs in [{'batch_size': 0}, {'batch_size': True}, {'batch_size': 33},
                       {'normalize_embeddings': 'yes'}, {'model_dir': Path('relative')}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): EmbeddingConfig(**kwargs)

    def fake_package(self, constructor):
        package = types.ModuleType('sentence_transformers')
        package.SentenceTransformer = constructor
        modules = types.ModuleType('sentence_transformers.models')
        for name in ['Normalize', 'Pooling', 'Transformer']:
            setattr(modules, name, type(name, (), {}))
        return {'sentence_transformers': package, 'sentence_transformers.models': modules}

    def test_loader_missing_local_model_never_calls_constructor(self):
        constructor = Mock()
        packages = self.fake_package(constructor)
        with patch.dict(sys.modules, packages), patch.object(Path, 'is_dir', return_value=False):
            with self.assertRaisesRegex(EmbeddingLoadError, 'automatic download is disabled'):
                _load_local_model(EmbeddingConfig())
        constructor.assert_not_called()

    def test_loader_passes_offline_cpu_flags_and_removes_builtin_normalize(self):
        constructor = Mock()
        packages = self.fake_package(constructor)
        types_module = packages['sentence_transformers.models']
        class Modules(list):
            def eval(self): return self
        model = Modules([types_module.Transformer(), types_module.Pooling(), types_module.Normalize()])
        model[0].do_lower_case = False
        constructor.return_value = model
        with patch.dict(sys.modules, packages), patch.object(Path, 'is_dir', return_value=True):
            self.assertIs(_load_local_model(EmbeddingConfig()), model)
        self.assertEqual(len(model), 2)
        kwargs = constructor.call_args.kwargs
        self.assertEqual(kwargs['device'], 'cpu')
        self.assertTrue(kwargs['local_files_only'])
        self.assertFalse(kwargs['trust_remote_code'])

    def test_loader_failure_is_clear(self):
        packages = self.fake_package(Mock(side_effect=OSError('broken weights')))
        with patch.dict(sys.modules, packages), patch.object(Path, 'is_dir', return_value=True):
            with self.assertRaisesRegex(EmbeddingLoadError, 'OSError'):
                _load_local_model(EmbeddingConfig())


class PersistedEmbeddingTests(OfflineDocumentTestCase):
    def test_reloaded_chunks_embed_without_mutating_storage(self):
        from app.local_rag.ingestion import ingest_document
        kb = self.storage.create_knowledge_base('经营复盘')
        filename = self.upload('复盘.md', '# 活动复盘\n\n美妆点击率提高，转化不足。\n\n# SOP\n\n库存不足时补货。')
        ingested = ingest_document(kb.knowledge_base_id, filename, storage=self.storage, loader=self.loader)
        persisted = self.storage.get_document_ingestion(kb.knowledge_base_id, ingested.document.document_id)
        before = {p: p.read_bytes() for p in (self.base / 'store').rglob('*') if p.is_file()}
        adapter = LocalEmbeddingAdapter(model_factory=lambda _: FakeModel())
        result = adapter.embed_documents(persisted.chunks)
        self.assertEqual([v.chunk_id for v in result], [c.chunk_id for c in persisted.chunks])
        self.assertEqual(before, {p: p.read_bytes() for p in (self.base / 'store').rglob('*') if p.is_file()})
        self.assertEqual(self.storage.get_knowledge_base(kb.knowledge_base_id).index_status, 'chunked')


if __name__ == '__main__':
    unittest.main()
