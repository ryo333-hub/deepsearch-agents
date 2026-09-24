"""Offline persistence/compatibility tests. No retrieval behavior belongs here."""

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np

from app.api.context import reset_session_context, set_session_context, set_thread_context
from app.local_rag import config
from app.local_rag.embeddings import LocalEmbeddingAdapter
from app.local_rag.ingestion import ingest_document
from app.local_rag.schemas import ChunkEmbedding
from app.local_rag.storage import KnowledgeBaseAccessError, LocalRAGStorage, LocalRAGStorageError
from app.local_rag.vector_store import (
    LocalVectorStore, VectorIndexCompatibilityError, VectorIndexCorruptError,
    VectorIndexExistsError, VectorIndexNotFoundError, VectorStoreError,
)
from local_rag_test_support import OfflineDocumentTestCase
from test_local_rag_embeddings import FakeModel


class VectorStoreTests(OfflineDocumentTestCase):
    def setUp(self):
        super().setUp()
        self.vectors = LocalVectorStore(self.storage)

    def kb_with_documents(self, *contents):
        kb = self.storage.create_knowledge_base('虚构经营知识库')
        chunks = []
        for index, content in enumerate(contents):
            name = self.upload(f'资料-{index}.md', content)
            result = ingest_document(kb.knowledge_base_id, name,
                                     storage=self.storage, loader=self.loader)
            chunks.extend(result.chunks)
        return kb, tuple(chunks)

    def embeddings(self, chunks, adapter=None):
        adapter = adapter or LocalEmbeddingAdapter(model_factory=lambda _: FakeModel())
        return adapter, adapter.embed_documents(chunks)

    def build(self, kb, chunks, *, rebuild=False, adapter=None, rows=None):
        adapter, generated = self.embeddings(chunks, adapter)
        return self.vectors.build_index(kb.knowledge_base_id, rows or generated,
                                        adapter.metadata, rebuild=rebuild), adapter

    def index_paths(self, kb):
        root = self.base / 'store' / 'sessions' / 'session_session-A' / kb.knowledge_base_id / 'vector_index'
        pointer = json.loads((root / 'current.json').read_text(encoding='utf-8'))
        generation = root / 'generations' / pointer['generation']
        return root, generation, generation / 'index.json', generation / 'vectors.npy'

    def rewrite_index(self, kb, mutate):
        _, _, path, _ = self.index_paths(kb)
        data = json.loads(path.read_text(encoding='utf-8'))
        mutate(data)
        path.write_text(json.dumps(data), encoding='utf-8')

    def test_empty_kb_build_and_absent_load(self):
        kb = self.storage.create_knowledge_base('空库')
        adapter = LocalEmbeddingAdapter(model_factory=lambda _: FakeModel())
        with self.assertRaisesRegex(VectorStoreError, '空知识库'):
            self.vectors.build_index(kb.knowledge_base_id, (), adapter.metadata)
        with self.assertRaises(VectorIndexNotFoundError):
            self.vectors.load_index(kb.knowledge_base_id, adapter.metadata)

    def test_single_chunk_shape_dtype_readonly_and_mapping(self):
        kb, chunks = self.kb_with_documents('618 活动复盘显示美妆转化率不足。')
        built, _ = self.build(kb, chunks)
        self.assertEqual(built.vectors.shape, (1, 384))
        self.assertEqual(built.vectors.dtype, np.float32)
        self.assertFalse(built.vectors.flags.writeable)
        self.assertEqual(built.metadata.chunk_ids, (chunks[0].chunk_id,))
        self.assertEqual(built.metadata.document_ids, (chunks[0].document_id,))
        self.assertEqual(self.storage.get_knowledge_base(kb.knowledge_base_id).index_status, 'indexed')
        self.assertEqual(self.storage.get_document_ingestion(kb.knowledge_base_id,
                                                              chunks[0].document_id).document.ingestion_status,
                         'chunked')

    def test_multiple_chunks_order_and_exact_reload(self):
        kb, chunks = self.kb_with_documents('美妆销量转化库存。' * 150,
                                             '广告投放 ROI 复盘。' * 150)
        built, adapter = self.build(kb, chunks)
        reloaded = LocalVectorStore(LocalRAGStorage(self.base / 'store')).load_index(
            kb.knowledge_base_id, adapter.metadata)
        self.assertEqual(reloaded.metadata, built.metadata)
        np.testing.assert_array_equal(reloaded.vectors, built.vectors)
        self.assertEqual(list(reloaded.metadata.chunk_ids), [c.chunk_id for c in chunks])

    def test_manifest_contains_only_required_index_metadata(self):
        kb, chunks = self.kb_with_documents('补货 SOP')
        built, adapter = self.build(kb, chunks)
        expected = {
            'index_version', 'knowledge_base_id', 'session_id',
            'embedding_model_name', 'embedding_model_revision',
            'embedding_dimension', 'normalize_embeddings', 'chunk_version',
            'vector_count', 'dtype', 'chunk_ids', 'document_ids',
            'vectors_sha256', 'source_chunks_sha256', 'created_at',
        }
        self.assertEqual(set(built.metadata.model_dump()), expected)
        self.assertEqual(built.metadata.index_version, 1)
        self.assertEqual(built.metadata.session_id, 'session-A')
        self.assertEqual(built.metadata.embedding_model_name, adapter.metadata.model_name)
        self.assertEqual(built.metadata.embedding_model_revision, adapter.metadata.model_revision)
        self.assertEqual(built.metadata.chunk_version, config.CHUNKING_VERSION)
        self.assertEqual(built.metadata.vector_count, len(chunks))
        self.assertEqual(built.metadata.dtype, 'float32')

    def test_no_chunk_text_is_duplicated_in_index_metadata(self):
        secret = '虚构内部资料 SECRET-ORANGE-731'
        kb, chunks = self.kb_with_documents(secret)
        self.build(kb, chunks)
        _, _, index_path, _ = self.index_paths(kb)
        self.assertNotIn(secret, index_path.read_text(encoding='utf-8'))

    def test_existing_index_requires_explicit_rebuild(self):
        kb, chunks = self.kb_with_documents('复盘')
        first, adapter = self.build(kb, chunks)
        with self.assertRaises(VectorIndexExistsError):
            self.vectors.build_index(kb.knowledge_base_id,
                                     adapter.embed_documents(chunks), adapter.metadata)
        current = self.vectors.load_index(kb.knowledge_base_id, adapter.metadata)
        self.assertEqual(current.metadata, first.metadata)

    def test_explicit_rebuild_replaces_generation_and_cleans_old(self):
        kb, chunks = self.kb_with_documents('美妆经营资料。' * 100)
        first, adapter = self.build(kb, chunks)
        old_name = json.loads(self.index_paths(kb)[0].joinpath('current.json').read_text())['generation']
        second = self.vectors.build_index(kb.knowledge_base_id,
                                         adapter.embed_documents(chunks), adapter.metadata,
                                         rebuild=True)
        root, _, _, _ = self.index_paths(kb)
        new_name = json.loads((root / 'current.json').read_text())['generation']
        self.assertNotEqual(old_name, new_name)
        self.assertNotEqual(first.metadata.created_at, second.metadata.created_at)
        self.assertEqual([p.name for p in (root / 'generations').iterdir()], [new_name])

    def test_rebuild_pointer_failure_preserves_old_current(self):
        kb, chunks = self.kb_with_documents('经营复盘。' * 80)
        first, adapter = self.build(kb, chunks)
        real_replace = os.replace
        def fail_pointer(source, destination):
            if Path(destination).name == 'current.json':
                raise OSError('simulated')
            return real_replace(source, destination)
        with patch('app.local_rag.vector_store.os.replace', side_effect=fail_pointer):
            with self.assertRaises(VectorStoreError):
                self.vectors.build_index(kb.knowledge_base_id,
                                         adapter.embed_documents(chunks), adapter.metadata,
                                         rebuild=True)
        loaded = self.vectors.load_index(kb.knowledge_base_id, adapter.metadata)
        self.assertEqual(loaded.metadata, first.metadata)
        self.assertEqual(len(list(self.index_paths(kb)[0].joinpath('generations').iterdir())), 1)

    def test_initial_pointer_failure_leaves_no_current_or_generation(self):
        kb, chunks = self.kb_with_documents('经营复盘。' * 80)
        adapter, rows = self.embeddings(chunks)
        real_replace = os.replace
        def fail_pointer(source, destination):
            if Path(destination).name == 'current.json':
                raise OSError('simulated')
            return real_replace(source, destination)
        with patch('app.local_rag.vector_store.os.replace', side_effect=fail_pointer):
            with self.assertRaises(VectorStoreError):
                self.vectors.build_index(kb.knowledge_base_id, rows, adapter.metadata)
        root = self.base / 'store' / 'sessions' / 'session_session-A' / kb.knowledge_base_id / 'vector_index'
        self.assertFalse((root / 'current.json').exists())
        self.assertEqual(list((root / 'generations').iterdir()), [])
        self.assertEqual(self.storage.get_knowledge_base(kb.knowledge_base_id).index_status, 'chunked')

    def test_model_name_mismatch_rejected(self):
        kb, chunks = self.kb_with_documents('复盘')
        _, adapter = self.build(kb, chunks)
        with self.assertRaisesRegex(VectorIndexCompatibilityError, 'model_name'):
            self.vectors.load_index(kb.knowledge_base_id,
                                    replace(adapter.metadata, model_name='other/model'))

    def test_model_revision_mismatch_rejected(self):
        kb, chunks = self.kb_with_documents('复盘')
        _, adapter = self.build(kb, chunks)
        with self.assertRaisesRegex(VectorIndexCompatibilityError, 'revision'):
            self.vectors.load_index(kb.knowledge_base_id,
                                    replace(adapter.metadata, model_revision='other'))

    def test_runtime_dimension_mismatch_rejected(self):
        kb, chunks = self.kb_with_documents('复盘')
        _, adapter = self.build(kb, chunks)
        with self.assertRaisesRegex(VectorIndexCompatibilityError, 'dimension'):
            self.vectors.load_index(kb.knowledge_base_id,
                                    replace(adapter.metadata, embedding_dimension=768))

    def test_normalization_mismatch_rejected(self):
        kb, chunks = self.kb_with_documents('复盘')
        _, adapter = self.build(kb, chunks)
        with self.assertRaisesRegex(VectorIndexCompatibilityError, 'normalize'):
            self.vectors.load_index(kb.knowledge_base_id,
                                    replace(adapter.metadata, normalize_embeddings=False))

    def test_chunk_version_mismatch_rejected(self):
        kb, chunks = self.kb_with_documents('复盘')
        _, adapter = self.build(kb, chunks)
        with patch.object(config, 'CHUNKING_VERSION', 'other-version'):
            with self.assertRaises(VectorIndexCompatibilityError):
                self.vectors.load_index(kb.knowledge_base_id, adapter.metadata)

    def test_embedding_count_mismatch_rejected_before_write(self):
        kb, chunks = self.kb_with_documents('复盘')
        adapter, rows = self.embeddings(chunks)
        for bad in [(), rows + rows]:
            with self.assertRaises(VectorStoreError):
                self.vectors.build_index(kb.knowledge_base_id, bad, adapter.metadata)
        self.assertFalse(self.index_paths_if_exists(kb))

    def index_paths_if_exists(self, kb):
        root = self.base / 'store' / 'sessions' / 'session_session-A' / kb.knowledge_base_id / 'vector_index'
        return root.exists()

    def test_chunk_id_order_mismatch_rejected(self):
        kb, chunks = self.kb_with_documents('销量库存转化。' * 200)
        adapter, rows = self.embeddings(chunks)
        self.assertGreater(len(rows), 1)
        with self.assertRaisesRegex(VectorStoreError, '顺序'):
            self.vectors.build_index(kb.knowledge_base_id, tuple(reversed(rows)), adapter.metadata)

    def test_duplicate_chunk_id_rejected(self):
        kb, chunks = self.kb_with_documents('销量库存转化。' * 200)
        adapter, rows = self.embeddings(chunks)
        forged = (rows[0], rows[1].model_copy(update={'chunk_id': rows[0].chunk_id}), *rows[2:])
        with self.assertRaises(VectorStoreError):
            self.vectors.build_index(kb.knowledge_base_id, forged, adapter.metadata)

    def test_embedding_dimension_mismatch_rejected(self):
        kb, chunks = self.kb_with_documents('复盘')
        adapter, rows = self.embeddings(chunks)
        bad = rows[0].model_copy(update={'dimension': 383, 'vector': rows[0].vector[:-1]})
        with self.assertRaisesRegex(VectorStoreError, '维度'):
            self.vectors.build_index(kb.knowledge_base_id, (bad,), adapter.metadata)

    def test_nan_and_infinity_rejected_before_write(self):
        for value in [float('nan'), float('inf'), float('-inf')]:
            with self.subTest(value=value):
                kb, chunks = self.kb_with_documents(f'复盘-{value}')
                adapter, rows = self.embeddings(chunks)
                vector = (value, *rows[0].vector[1:])
                bad = ChunkEmbedding.model_construct(chunk_id=rows[0].chunk_id,
                                                     vector=vector, dimension=384)
                with self.assertRaisesRegex(VectorStoreError, 'NaN|Infinity'):
                    self.vectors.build_index(kb.knowledge_base_id, (bad,), adapter.metadata)

    def test_zero_and_unnormalized_vectors_rejected(self):
        kb, chunks = self.kb_with_documents('复盘')
        adapter, rows = self.embeddings(chunks)
        for vector in [(0.0,) * 384, (2.0, *([0.0] * 383))]:
            bad = ChunkEmbedding(chunk_id=rows[0].chunk_id, vector=vector, dimension=384)
            with self.assertRaisesRegex(VectorStoreError, '范数'):
                self.vectors.build_index(kb.knowledge_base_id, (bad,), adapter.metadata)

    def test_kb_isolation(self):
        first, a_chunks = self.kb_with_documents('A-资料')
        second, b_chunks = self.kb_with_documents('B-资料')
        a_index, adapter = self.build(first, a_chunks)
        b_index, _ = self.build(second, b_chunks, adapter=adapter)
        self.assertNotEqual(a_index.metadata.knowledge_base_id, b_index.metadata.knowledge_base_id)
        self.assertTrue(set(a_index.metadata.chunk_ids).isdisjoint(b_index.metadata.chunk_ids))
        self.assertNotEqual(self.index_paths(first)[0], self.index_paths(second)[0])

    def test_session_isolation_and_guessed_id_rejected(self):
        kb, chunks = self.kb_with_documents('A 私有资料')
        _, adapter = self.build(kb, chunks)
        set_thread_context('session-B')
        try:
            with self.assertRaises(KnowledgeBaseAccessError):
                self.vectors.load_index(kb.knowledge_base_id, adapter.metadata)
        finally:
            set_thread_context('session-A')

    def test_illegal_kb_id_rejected(self):
        adapter = LocalEmbeddingAdapter(model_factory=lambda _: FakeModel())
        for value in ['../session-A', 'kb_' + 'a' * 31, 'C:\\outside']:
            with self.subTest(value=value), self.assertRaises(KnowledgeBaseAccessError):
                self.vectors.load_index(value, adapter.metadata)

    def test_corrupt_pointer_rejected(self):
        kb, chunks = self.kb_with_documents('复盘')
        _, adapter = self.build(kb, chunks)
        root = self.index_paths(kb)[0]
        (root / 'current.json').write_text('{bad', encoding='utf-8')
        with self.assertRaises(VectorIndexCorruptError):
            self.vectors.load_index(kb.knowledge_base_id, adapter.metadata)

    def test_missing_vectors_rejected(self):
        kb, chunks = self.kb_with_documents('复盘')
        _, adapter = self.build(kb, chunks)
        self.index_paths(kb)[3].unlink()
        with self.assertRaises(VectorIndexCorruptError):
            self.vectors.load_index(kb.knowledge_base_id, adapter.metadata)

    def test_corrupt_vectors_and_hash_mismatch_rejected(self):
        kb, chunks = self.kb_with_documents('复盘')
        _, adapter = self.build(kb, chunks)
        path = self.index_paths(kb)[3]
        path.write_bytes(path.read_bytes()[:-3] + b'BAD')
        with self.assertRaises(VectorIndexCorruptError):
            self.vectors.load_index(kb.knowledge_base_id, adapter.metadata)

    def test_manifest_missing_field_rejected(self):
        kb, chunks = self.kb_with_documents('复盘')
        _, adapter = self.build(kb, chunks)
        self.rewrite_index(kb, lambda data: data.pop('embedding_model_revision'))
        with self.assertRaises(VectorIndexCorruptError):
            self.vectors.load_index(kb.knowledge_base_id, adapter.metadata)

    def test_every_manifest_field_is_required(self):
        kb, chunks = self.kb_with_documents('复盘')
        built, adapter = self.build(kb, chunks)
        path = self.index_paths(kb)[2]
        original = built.metadata.model_dump(mode='json')
        for field in original:
            with self.subTest(field=field):
                data = dict(original)
                del data[field]
                path.write_text(json.dumps(data), encoding='utf-8')
                with self.assertRaises(VectorIndexCorruptError):
                    self.vectors.load_index(kb.knowledge_base_id, adapter.metadata)

    def test_pending_document_blocks_build_and_stale_load(self):
        kb, chunks = self.kb_with_documents('原有复盘')
        _, adapter = self.build(kb, chunks)
        self.storage.register_document_metadata(
            kb.knowledge_base_id, document_name='pending.txt', source_type='txt',
            content_hash='f' * 64, size_bytes=10,
        )
        with self.assertRaises(VectorStoreError):
            self.vectors.load_index(kb.knowledge_base_id, adapter.metadata)
        with self.assertRaises(VectorStoreError):
            self.vectors.build_index(kb.knowledge_base_id, adapter.embed_documents(chunks),
                                     adapter.metadata, rebuild=True)
        self.vectors.delete_index(kb.knowledge_base_id)
        self.assertEqual(self.storage.get_knowledge_base(kb.knowledge_base_id).index_status, 'pending')

    def test_modified_source_with_same_id_is_rejected(self):
        kb, chunks = self.kb_with_documents('ABCDE')
        _, adapter = self.build(kb, chunks)
        root = self.index_paths(kb)[0].parent
        chunk_file = root / 'documents' / chunks[0].document_id / 'chunks.jsonl'
        record = json.loads(chunk_file.read_text(encoding='utf-8'))
        record['text'] = 'VWXYZ'
        record['text_hash'] = sha256(b'VWXYZ').hexdigest()
        chunk_file.write_text(json.dumps(record) + '\n', encoding='utf-8')
        with self.assertRaises(VectorIndexCompatibilityError):
            self.vectors.load_index(kb.knowledge_base_id, adapter.metadata)

    def test_npy_header_shape_dtype_and_pickle_rejected_before_allocation(self):
        kb, chunks = self.kb_with_documents('复盘')
        _, adapter = self.build(kb, chunks)
        path = self.index_paths(kb)[3]
        for shape, dtype, order in [((10**10, 384), '<f4', False),
                                     ((1,384), '<f8', False), ((1,384), '|O', False),
                                     ((1,384), '<f4', True)]:
            with self.subTest(shape=shape,dtype=dtype,order=order):
                stream=BytesIO()
                np.lib.format.write_array_header_1_0(stream,{
                    'shape':shape,'fortran_order':order,'descr':dtype})
                payload=stream.getvalue()
                path.write_bytes(payload)
                self.rewrite_index(kb,lambda d: d.update(vectors_sha256=sha256(payload).hexdigest()))
                with patch('app.local_rag.vector_store.np.load', side_effect=AssertionError('Do not allocate')) as load:
                    with self.assertRaises(VectorIndexCorruptError):
                        self.vectors.load_index(kb.knowledge_base_id, adapter.metadata)
                    load.assert_not_called()

    def test_corrupt_numeric_vectors_with_updated_checksum_still_rejected(self):
        kb, chunks = self.kb_with_documents('复盘')
        _, adapter = self.build(kb, chunks)
        path=self.index_paths(kb)[3]
        for value in [np.nan, np.inf, 0.0]:
            with self.subTest(value=value):
                stream=BytesIO();np.save(stream,np.full((1,384),value,dtype=np.float32),allow_pickle=False)
                payload=stream.getvalue();path.write_bytes(payload)
                self.rewrite_index(kb,lambda d: d.update(vectors_sha256=sha256(payload).hexdigest()))
                with self.assertRaises(VectorIndexCorruptError):
                    self.vectors.load_index(kb.knowledge_base_id, adapter.metadata)

    def test_generation_path_traversal_rejected(self):
        kb, chunks = self.kb_with_documents('复盘')
        _, adapter = self.build(kb, chunks)
        pointer=self.index_paths(kb)[0]/'current.json'
        for name in ['../other', '..\\other', 'C:\\outside', 'gen_'+'a'*32+'/../other']:
            with self.subTest(name=name):
                pointer.write_text(json.dumps({'pointer_version':1,'generation':name}),encoding='utf-8')
                with self.assertRaises(VectorIndexCorruptError):
                    self.vectors.load_index(kb.knowledge_base_id, adapter.metadata)

    def test_linked_vector_directory_rejected_without_touching_target(self):
        kb, chunks = self.kb_with_documents('复盘')
        adapter, _ = self.embeddings(chunks)
        kb_dir=self.base/'store'/'sessions'/'session_session-A'/kb.knowledge_base_id
        target=self.base/'outside-vector-target';target.mkdir()
        marker=target/'keep.txt';marker.write_text('keep',encoding='utf-8')
        self.create_junction(kb_dir/'vector_index',target)
        for action in [lambda: self.vectors.load_index(kb.knowledge_base_id,adapter.metadata),
                       lambda: self.vectors.delete_index(kb.knowledge_base_id)]:
            with self.assertRaises(KnowledgeBaseAccessError):action()
        self.assertEqual(marker.read_text(encoding='utf-8'),'keep')

    def test_delete_checks_every_file_before_mutation(self):
        kb, chunks = self.kb_with_documents('复盘')
        built, adapter = self.build(kb, chunks)
        root,_,metadata,vector=self.index_paths(kb)
        link=self.base/'protected-link.npy';os.link(vector,link)
        self.addCleanup(link.unlink,missing_ok=True)
        before=(metadata.read_bytes(),(root/'current.json').read_bytes())
        with self.assertRaises(VectorStoreError):self.vectors.delete_index(kb.knowledge_base_id)
        self.assertEqual(before,(metadata.read_bytes(),(root/'current.json').read_bytes()))
        self.assertEqual(self.storage.get_knowledge_base(kb.knowledge_base_id).index_status,'indexed')

    def test_kb_summary_failure_reports_committed_index(self):
        kb, chunks = self.kb_with_documents('复盘')
        adapter,rows=self.embeddings(chunks)
        with patch.object(self.storage,'_write',side_effect=LocalRAGStorageError('simulated')):
            with self.assertWarnsRegex(RuntimeWarning,'committed'):
                built=self.vectors.build_index(kb.knowledge_base_id,rows,adapter.metadata)
        loaded=self.vectors.load_index(kb.knowledge_base_id,adapter.metadata)
        np.testing.assert_array_equal(loaded.vectors,built.vectors)
        self.assertEqual(self.storage.get_knowledge_base(kb.knowledge_base_id).index_status,'chunked')

    def test_bad_embedding_scalar_not_silently_coerced(self):
        kb,chunks=self.kb_with_documents('复盘')
        adapter,rows=self.embeddings(chunks)
        for scalar in ['1.0',True]:
            with self.subTest(scalar=scalar):
                bad=rows[0].model_copy(update={'vector':(scalar,)+rows[0].vector[1:]})
                with self.assertRaises(VectorStoreError):
                    self.vectors.build_index(kb.knowledge_base_id,(bad,),adapter.metadata)

    def test_vectors_and_chunk_ids_count_mismatch_rejected(self):
        kb, chunks = self.kb_with_documents('复盘')
        _, adapter = self.build(kb, chunks)
        self.rewrite_index(kb, lambda data: data['chunk_ids'].append(data['chunk_ids'][0]))
        with self.assertRaises(VectorIndexCorruptError):
            self.vectors.load_index(kb.knowledge_base_id, adapter.metadata)

    def test_current_chunks_changed_rejected(self):
        kb, chunks = self.kb_with_documents('复盘')
        _, adapter = self.build(kb, chunks)
        self.rewrite_index(kb, lambda data: data['chunk_ids'].__setitem__(0, 'chunk_' + 'f' * 32))
        with self.assertRaises(VectorIndexCompatibilityError):
            self.vectors.load_index(kb.knowledge_base_id, adapter.metadata)

    def test_index_files_with_extra_hardlink_rejected(self):
        kb, chunks = self.kb_with_documents('复盘')
        _, adapter = self.build(kb, chunks)
        vector = self.index_paths(kb)[3]
        link = self.base / 'extra-vector-link.npy'
        try:
            os.link(vector, link)
        except OSError:
            self.skipTest('Hardlinks unavailable')
        self.addCleanup(link.unlink, missing_ok=True)
        with self.assertRaises(VectorIndexCorruptError):
            self.vectors.load_index(kb.knowledge_base_id, adapter.metadata)

    def test_delete_absent_and_existing_index(self):
        kb, chunks = self.kb_with_documents('复盘')
        self.assertFalse(self.vectors.delete_index(kb.knowledge_base_id))
        _, adapter = self.build(kb, chunks)
        self.assertTrue(self.vectors.delete_index(kb.knowledge_base_id))
        self.assertFalse(self.vectors.delete_index(kb.knowledge_base_id))
        self.assertEqual(self.storage.get_knowledge_base(kb.knowledge_base_id).index_status, 'chunked')
        with self.assertRaises(VectorIndexNotFoundError):
            self.vectors.load_index(kb.knowledge_base_id, adapter.metadata)
        self.assertIsNotNone(self.storage.get_document_ingestion(kb.knowledge_base_id,
                                                                  chunks[0].document_id))

    def test_delete_refuses_unknown_content(self):
        kb, chunks = self.kb_with_documents('复盘')
        self.build(kb, chunks)
        root = self.index_paths(kb)[0]
        (root / 'unknown.txt').write_text('do not delete', encoding='utf-8')
        with self.assertRaisesRegex(VectorStoreError, '未知'):
            self.vectors.delete_index(kb.knowledge_base_id)
        self.assertTrue((root / 'unknown.txt').exists())

    def test_process_level_reload(self):
        kb, chunks = self.kb_with_documents('618 美妆活动复盘。' * 80,
                                             '库存补货 SOP。' * 80)
        built, adapter = self.build(kb, chunks)
        code = r'''
import json, os, sys
from hashlib import sha256
def audit(event,args):
 if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
  raise RuntimeError('Network prohibited in reload subprocess')
sys.addaudithook(audit)
from app.api.context import set_thread_context
from app.local_rag.embeddings import LocalEmbeddingAdapter
from app.local_rag.storage import LocalRAGStorage
from app.local_rag.vector_store import LocalVectorStore
set_thread_context('session-A')
index=LocalVectorStore(LocalRAGStorage(os.environ['VECTOR_TEST_ROOT'])).load_index(
    os.environ['VECTOR_TEST_KB'], LocalEmbeddingAdapter().metadata)
print(json.dumps({'shape':list(index.vectors.shape),'chunks':list(index.metadata.chunk_ids),
                  'vector_bytes_sha256':sha256(index.vectors.tobytes()).hexdigest(),
                  'metadata':index.metadata.model_dump(mode='json')}))
'''
        env = os.environ.copy()
        env['VECTOR_TEST_ROOT'] = str(self.base / 'store')
        env['VECTOR_TEST_KB'] = kb.knowledge_base_id
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=Path.cwd(), env=env,
                                capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        loaded = json.loads(result.stdout)
        self.assertEqual(loaded['shape'], list(built.vectors.shape))
        self.assertEqual(loaded['chunks'], list(built.metadata.chunk_ids))
        self.assertEqual(loaded['vector_bytes_sha256'],sha256(built.vectors.tobytes()).hexdigest())
        self.assertEqual(loaded['metadata'],built.metadata.model_dump(mode='json'))


if __name__ == '__main__':
    import unittest
    unittest.main()
