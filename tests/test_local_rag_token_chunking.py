"""Token-aware splitting, exact source spans and shared encoder preflight."""

import builtins
import hashlib
import json
from pathlib import Path
import sys
import types
from unittest.mock import Mock, patch

from app.local_rag import config
from app.local_rag.chunking import chunk_document, citation_from_chunk
from app.local_rag.embeddings import LocalEmbeddingAdapter
from app.local_rag.ingestion import ingest_document
from app.local_rag.schemas import ChunkRecord, DocumentRecord, ParsedDocument, TextBlock
from app.local_rag.storage import KnowledgeBaseAccessError, LocalRAGStorageError
from app.local_rag.tokenization import (
    DocumentTokenBudget, EmbeddingDependencyError, EmbeddingInputError,
    EmbeddingInputTooLong, EmbeddingLoadError, _load_tokenizer,
    count_tokens, get_document_token_budget, prepare_text, validate_prepared,
)
from local_rag_test_support import CharacterTokenizer, OfflineDocumentTestCase, pdf_bytes
from test_local_rag_embeddings import FakeModel


class TokenChunkingTests(OfflineDocumentTestCase):
    def split(self, text, name='运营.txt', budget=None):
        parsed = self.loader.load(self.upload(name, text))
        doc = DocumentRecord(document_name=parsed.document_name, source_type=parsed.source_type,
                             content_hash=parsed.content_hash, size_bytes=parsed.size_bytes)
        chunks = chunk_document(parsed, doc.document_id, token_budget=budget or self.token_budget)
        return parsed, doc, chunks

    def assert_traceable(self, parsed, doc, chunks):
        for i, c in enumerate(chunks):
            b = next(b for b in parsed.blocks if b.block_index == c.source_block_index)
            self.assertEqual(c.text, b.text[c.start_char:c.end_char])
            self.assertEqual(c.chunk_index, i)
            self.assertEqual(c.heading, b.heading)
            self.assertEqual(c.page, b.page)
            if b.start_line:
                self.assertEqual(c.start_line, b.start_line + b.text[:c.start_char].count('\n'))
                self.assertEqual(c.end_line, b.start_line + b.text[:c.end_char-1].count('\n'))
            self.assertLessEqual(self.token_budget.validate_document(c.text, c.chunk_id), 512)
            citation = citation_from_chunk(c, doc, 'kb_' + 'a' * 32, f'C{i+1}')
            for field in ['source_block_index', 'start_char', 'end_char', 'page', 'start_line', 'end_line', 'heading']:
                self.assertEqual(getattr(citation, field), getattr(c, field))
        # Every non-whitespace source character survives; overlap does not invent
        # content, and trimming may omit only source whitespace at the boundaries.
        for b in parsed.blocks:
            spans = [c for c in chunks if c.source_block_index == b.block_index]
            covered = bytearray(len(b.text))
            for c in spans: covered[c.start_char:c.end_char] = b'1' * len(c.text)
            self.assertTrue(all(covered[i] or ch.isspace() for i, ch in enumerate(b.text)))

    def test_short_chinese_block_stays_intact(self):
        p,d,c = self.split('618 美妆活动点击率提升，转化率低于预期。')
        self.assertEqual(len(c), 1)
        self.assert_traceable(p,d,c)

    def test_short_english_block_stays_intact(self):
        p,d,c = self.split('The campaign increased clicks but conversion was below target.')
        self.assertEqual(len(c), 1)
        self.assert_traceable(p,d,c)

    def test_mixed_language_content_survives(self):
        p,d,c = self.split('广告复盘 ROI below target；补货 SOP 需要检查库存。\n' * 80)
        self.assertGreater(len(c), 1)
        self.assert_traceable(p,d,c)

    def test_1000_chinese_characters_are_split_and_encode(self):
        p,d,c = self.split('中' * 1000)
        self.assertGreater(len(c), 1)
        self.assert_traceable(p,d,c)
        adapter = LocalEmbeddingAdapter(model_factory=lambda _: FakeModel())
        self.assertEqual(len(adapter.embed_documents(c)), len(c))

    def test_long_english_paragraph(self):
        p,d,c = self.split('Inventory replenishment follows the safety stock policy. ' * 100)
        self.assert_traceable(p,d,c)
        self.assertLess(len(c), 20)

    def test_very_long_unbroken_paragraph_terminates_without_loss(self):
        p,d,c = self.split('库存' * 10_000)
        self.assertGreater(len(c), 20)
        self.assertLess(len(c), 60)
        self.assert_traceable(p,d,c)

    def test_single_heading_section_keeps_heading_and_source_ranges(self):
        p,d,c = self.split('# 618 活动复盘\n\n' + '美妆转化率下降。' * 300, '运营.md')
        self.assertTrue(all(x.heading == '618 活动复盘' for x in c))
        self.assert_traceable(p,d,c)

    def test_pdf_page_and_character_spans_survive_secondary_splits(self):
        p,d,c = self.split(pdf_bytes(['Campaign conversion ' * 160, 'Inventory SOP ' * 180]), '运营.pdf')
        self.assertEqual({x.page for x in c}, {1,2})
        self.assertGreater(sum(x.page == 1 for x in c), 1)
        self.assert_traceable(p,d,c)

    def test_txt_line_ranges_are_recomputed_for_each_subchunk(self):
        p,d,c = self.split('\n'.join(f'第{i}行：' + '库存周转分析' * 18 for i in range(30)))
        self.assertGreater(len({(x.start_line,x.end_line) for x in c}), 2)
        self.assert_traceable(p,d,c)

    def test_one_long_source_line_has_distinct_character_spans(self):
        p,d,c = self.split('美妆销售分析' * 400)
        self.assertTrue(all(x.start_line == x.end_line == 1 for x in c))
        self.assertEqual(len({(x.start_char,x.end_char) for x in c}), len(c))
        self.assert_traceable(p,d,c)

    def test_overlap_is_bounded_by_tokens_and_never_breaks_budget(self):
        p,d,c = self.split('甲乙丙丁' * 500)
        for a,b in zip(c,c[1:]):
            self.assertGreater(b.start_char, a.start_char)
            self.assertGreater(b.end_char, a.end_char)
            overlap = p.blocks[0].text[b.start_char:a.end_char]
            self.assertLessEqual(self.token_budget.overlap_tokens(overlap), config.CHUNK_OVERLAP_TOKENS)
        self.assert_traceable(p,d,c)

    def test_no_overlap_across_paragraphs_or_headings(self):
        p,d,c = self.split('# 复盘\n\n' + '甲'*800 + '\n\n# SOP\n\n' + '乙'*800, '运营.md')
        self.assertTrue(all(not ('甲' in x.text and '乙' in x.text) for x in c))
        self.assert_traceable(p,d,c)

    def test_all_chunks_pass_identical_embedding_preflight(self):
        p,d,c = self.split('销量 Sales ROI 618，转化表现低于预期。'*200)
        model = FakeModel()
        adapter = LocalEmbeddingAdapter(model_factory=lambda _: model)
        for x in c:
            prepared = prepare_text(x.text, x.chunk_id)
            self.assertEqual(self.token_budget.document_tokens(x.text), adapter._check_tokens(model, prepared, x.chunk_id))
        self.assertEqual(model.calls, [])
        self.assert_traceable(p,d,c)

    def test_empty_blocks_do_not_load_tokenizer(self):
        parsed = ParsedDocument(document_name='empty.txt',source_type='txt',content_hash='0'*64,
                                size_bytes=0,blocks=(TextBlock(block_index=0,text='  \n '),))
        with patch('app.local_rag.tokenization._load_tokenizer', side_effect=AssertionError('No loading')):
            with self.assertRaisesRegex(ValueError, '无可分块'):
                chunk_document(parsed, 'doc_'+'a'*32, token_budget=DocumentTokenBudget())

    def test_tokenizer_unavailable_fails_without_fallback(self):
        with patch('app.local_rag.tokenization._load_tokenizer', side_effect=EmbeddingLoadError('Unavailable')):
            with self.assertRaises(EmbeddingLoadError): self.split('运营资料', budget=DocumentTokenBudget())

    def test_chunking_never_calls_model_loader(self):
        with patch('app.local_rag.embeddings._load_local_model', side_effect=AssertionError('Encoder prohibited')) as load:
            self.split('广告投放复盘' * 500)
        load.assert_not_called()

    def test_shorter_candidate_can_have_more_tokens_without_unsafe_output(self):
        class Nonmonotone(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                ids = super().__call__(text, **kwargs)['input_ids']
                if len(text) % 7 == 0: ids = ids * 3
                return {'input_ids': ids}
        budget=DocumentTokenBudget(tokenizer=Nonmonotone())
        p,d,c=self.split('甲乙丙丁。'*600,budget=budget)
        self.assertTrue(all(budget.document_tokens(x.text)<=512 for x in c))
        self.assert_traceable(p,d,c)

    def test_trailing_whitespace_never_emits_duplicate_overlap(self):
        p,d,c=self.split('库存' * 400 + ' '*2000)
        self.assert_traceable(p,d,c)
        self.assertTrue(all(b.end_char>a.end_char for a,b in zip(c,c[1:])))

    def test_invalid_or_partial_source_span_rejected(self):
        values=dict(document_id='doc_'+'1'*32,chunk_index=0,text='test',text_hash='0'*64)
        for extra in [{'start_char':0},{'source_block_index':0,'start_char':4,'end_char':4}]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):ChunkRecord(**values,**extra)

    def test_ingest_reload_embed_long_business_document(self):
        source='# 618 活动复盘\n\n' + '美妆品类点击率上升但转化率不足，库存断货影响成交。'*100
        source+='\n\n# 库存补货 SOP\n\n' + '库存周转低于阈值时及时补货，广告投放须跟踪 ROI。'*70
        filename=self.upload('虚构经营资料.md',source)
        kb=self.storage.create_knowledge_base('虚构企业')
        ingested=ingest_document(kb.knowledge_base_id,filename,storage=self.storage,loader=self.loader)
        loaded=self.storage.get_document_ingestion(kb.knowledge_base_id,ingested.document.document_id)
        self.assertEqual(ingested,loaded)
        before={p:p.read_bytes() for p in (self.base/'store').rglob('*') if p.is_file()}
        adapter=LocalEmbeddingAdapter(model_factory=lambda _:FakeModel())
        vectors=adapter.embed_documents(loaded.chunks)
        self.assertEqual([x.chunk_id for x in vectors],[x.chunk_id for x in loaded.chunks])
        self.assert_traceable(self.loader.load(filename),loaded.document,loaded.chunks)
        self.assertEqual(before,{p:p.read_bytes() for p in (self.base/'store').rglob('*') if p.is_file()})

    def test_old_version_reload_and_reingest_refused_without_writes(self):
        kb=self.storage.create_knowledge_base('旧数据')
        name=self.upload('历史.txt','美妆经营复盘')
        with patch.object(config,'CHUNKING_VERSION','local-rag-chunk-v1'):
            old=ingest_document(kb.knowledge_base_id,name,storage=self.storage,loader=self.loader)
        before={p:p.read_bytes() for p in (self.base/'store').rglob('*') if p.is_file()}
        with self.assertRaisesRegex(LocalRAGStorageError,'重新入库'):
            self.storage.get_document_ingestion(kb.knowledge_base_id,old.document.document_id)
        with self.assertRaisesRegex(ValueError,'显式重建'):
            ingest_document(kb.knowledge_base_id,name,storage=self.storage,loader=self.loader)
        self.assertEqual(before,{p:p.read_bytes() for p in (self.base/'store').rglob('*') if p.is_file()})
        new_kb=self.storage.create_knowledge_base('显式重建')
        new=ingest_document(new_kb.knowledge_base_id,name,storage=self.storage,loader=self.loader)
        self.assertEqual(new.document.chunking_version,config.CHUNKING_VERSION)
        self.assertNotEqual(new.document.document_id,old.document.document_id)


class TokenBudgetTests(OfflineDocumentTestCase):
    def test_lazy_construction_and_cache(self):
        get_document_token_budget.cache_clear()
        self.addCleanup(get_document_token_budget.cache_clear)
        with patch('app.local_rag.tokenization._load_tokenizer') as load:
            self.assertIs(get_document_token_budget(),get_document_token_budget())
            load.assert_not_called()

    def test_budget_includes_prefix_and_special_tokens(self):
        budget=self.token_budget
        self.assertEqual(budget.embedding_max_tokens,512)
        self.assertEqual(budget.effective_document_token_budget,501) # fake: nine prefix chars + two specials
        self.assertEqual(budget.validate_document('中'*501),512)
        with self.assertRaises(EmbeddingInputTooLong):budget.validate_document('中'*502)

    def test_shared_preparation_for_queries_and_documents(self):
        self.assertEqual(prepare_text('  销量\n','c'),'passage: 销量')
        self.assertEqual(prepare_text(' 销量 ','q',query=True),'query: 销量')
        for value in ['', ' \n', None]:
            with self.assertRaises(EmbeddingInputError):prepare_text(value,'c')

    def test_overlap_excludes_prefix_and_specials(self):
        self.assertEqual(self.token_budget.overlap_tokens('中文'),2)
        self.assertEqual(self.token_budget.document_tokens('中文'),13)

    def test_tokenizer_loaded_only_once(self):
        with patch('app.local_rag.tokenization._load_tokenizer',return_value=CharacterTokenizer()) as load:
            budget=DocumentTokenBudget()
            budget.validate_document('a');budget.validate_document('b')
            load.assert_called_once()

    def test_missing_tokenizers_dependency(self):
        with patch.dict(sys.modules,{'tokenizers':None}), self.assertRaises(EmbeddingDependencyError):
            _load_tokenizer(config.EmbeddingConfig(model_dir=self.base/'missing'))

    def test_loader_never_imports_encoder_and_disables_truncation(self):
        root=self.base/'model';root.mkdir()
        (root/'tokenizer_config.json').write_text(json.dumps({'model_max_length':512}),encoding='utf-8')
        native=Mock();native.encode.return_value.ids=[1,2,3]
        package=types.ModuleType('tokenizers');package.Tokenizer=Mock()
        package.Tokenizer.from_file.return_value=native
        real_import=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in {'torch','transformers','sentence_transformers'}:
                self.fail('Encoder library import during tokenizer-only loading')
            return real_import(name,*args,**kwargs)
        with patch.dict(sys.modules,{'tokenizers':package}),patch('builtins.__import__',guarded):
            tok=_load_tokenizer(config.EmbeddingConfig(model_dir=root))
            self.assertEqual(validate_prepared(tok,'passage: a','a'),3)
        native.no_truncation.assert_called_once();native.no_padding.assert_called_once()

    def test_missing_snapshot_explicit_no_network(self):
        with self.assertRaisesRegex(EmbeddingLoadError,'automatic download is disabled'):
            _load_tokenizer(config.EmbeddingConfig(model_dir=self.base/'missing'))

    def test_wrong_maximum_rejected(self):
        root=self.base/'model';root.mkdir()
        (root/'tokenizer_config.json').write_text('{"model_max_length":256}',encoding='utf-8')
        with self.assertRaises(EmbeddingLoadError):_load_tokenizer(config.EmbeddingConfig(model_dir=root))

    def test_corrupt_tokenizer_rejected(self):
        root=self.base/'model';root.mkdir()
        (root/'tokenizer_config.json').write_text('{"model_max_length":512}',encoding='utf-8')
        (root/'tokenizer.json').write_text('not json',encoding='utf-8')
        with self.assertRaises(EmbeddingLoadError):_load_tokenizer(config.EmbeddingConfig(model_dir=root))
