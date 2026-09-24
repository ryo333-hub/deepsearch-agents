import hashlib
import unittest
from unittest.mock import patch

from app.local_rag import config
from app.local_rag.chunking import chunk_document, citation_from_chunk
from app.local_rag.schemas import DocumentRecord
from local_rag_test_support import OfflineDocumentTestCase, pdf_bytes


class ChunkingTests(OfflineDocumentTestCase):
    def chunks(self, text, name="a.txt"):
        parsed = self.loader.load(self.upload(name, text))
        doc = DocumentRecord(document_name=parsed.document_name, source_type=parsed.source_type,
                             content_hash=parsed.content_hash, size_bytes=parsed.size_bytes)
        return parsed, doc, chunk_document(parsed, doc.document_id)

    def test_short_text_one_chunk(self):
        _, _, chunks = self.chunks("星河项目内部代号为 ORANGE-731。")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_index, 0)

    def test_long_text_bounded_and_indices_contiguous(self):
        _, _, chunks = self.chunks("星河项目的说明。" * 600)
        self.assertGreater(len(chunks), 1)
        self.assertEqual([c.chunk_index for c in chunks], list(range(len(chunks))))
        self.assertTrue(all(0 < len(c.text) <= config.MAX_CHUNK_CHARS for c in chunks))

    def test_only_whitespace_between_paragraphs_never_creates_empty_chunk(self):
        _, _, chunks = self.chunks("\nfirst\n\n \nsecond\n\n")
        self.assertEqual([c.text for c in chunks], ["first", "second"])

    def test_hash_and_locations_stable_but_ids_backend_random(self):
        parsed, doc, first = self.chunks("first\r\nsecond\n\nlast")
        second = chunk_document(parsed, doc.document_id)
        self.assertEqual([c.text_hash for c in first], [c.text_hash for c in second])
        self.assertNotEqual(first[0].chunk_id, second[0].chunk_id)
        self.assertEqual(first[0].text_hash, hashlib.sha256("first\nsecond".encode()).hexdigest())
        self.assertTrue(all(c.chunking_version == config.CHUNKING_VERSION for c in first))

    def test_markdown_never_overlaps_across_headings(self):
        _, _, chunks = self.chunks("# Root\n## A\n" + "甲。" * 800 + "\n## B\n乙内容。", "a.md")
        a = [c for c in chunks if c.heading == "Root > A"]
        b = [c for c in chunks if c.heading == "Root > B"]
        self.assertTrue(a and b)
        self.assertTrue(all("甲" not in c.text for c in b))
        self.assertTrue(all("乙" not in c.text for c in a))

    def test_pdf_chunks_never_cross_pages(self):
        _, _, chunks = self.chunks(pdf_bytes(["First " * 350, "Second " * 350]), "a.pdf")
        self.assertEqual({c.page for c in chunks}, {1, 2})
        self.assertTrue(all("Second" not in c.text for c in chunks if c.page == 1))
        self.assertTrue(all(c.start_line is None for c in chunks))

    def test_line_locations_include_overlap_and_match_original_file(self):
        lines = [f"line-{index}: " + "x" * 100 for index in range(1, 31)]
        _, _, chunks = self.chunks("\n".join(lines))
        for chunk in chunks:
            source = "\n".join(lines[chunk.start_line - 1:chunk.end_line])
            self.assertIn(chunk.text, source)
        self.assertTrue(any(b.start_line <= a.end_line for a, b in zip(chunks, chunks[1:])))

    def test_overlap_hard_split_is_bounded(self):
        text = "".join(chr(0x4E00 + i) for i in range(2500))
        _, _, chunks = self.chunks(text)
        overlap = chunks[0].end_char - chunks[1].start_char
        self.assertGreater(overlap, 0)
        self.assertLessEqual(self.token_budget.overlap_tokens(text[chunks[1].start_char:chunks[0].end_char]),
                             config.CHUNK_OVERLAP_TOKENS)
        self.assertEqual(chunks[0].text[-overlap:], chunks[1].text[:overlap])
        self.assertTrue(chunks[-1].text.endswith(text[-30:]))
        reconstructed = chunks[0].text + "".join(b.text[a.end_char - b.start_char:]
                                               for a, b in zip(chunks, chunks[1:]))
        self.assertEqual(reconstructed, text)

    def test_very_long_single_paragraph_terminates(self):
        _, _, chunks = self.chunks("字" * 12_000)
        self.assertLess(len(chunks), 32)
        self.assertTrue(all(len(c.text) <= 1000 for c in chunks))

    def test_repeated_content_does_not_loop(self):
        _, _, chunks = self.chunks("重复。" * 4000)
        self.assertLess(len(chunks), 30)

    def test_chinese_sentence_boundary_preferred(self):
        text = "甲" * 350 + "。" + "乙" * 350 + "！"
        _, _, chunks = self.chunks(text)
        self.assertTrue(chunks[0].text.endswith("。"))
        self.assertNotIn("乙", chunks[0].text)

    def test_english_sentence_boundary_preferred(self):
        text = "First word " * 30 + ". " + "Second word " * 30 + "."
        _, _, chunks = self.chunks(text)
        self.assertTrue(chunks[0].text.endswith("."))
        self.assertNotIn("Second", chunks[0].text)

    def test_newline_preferred_over_sentence_boundary(self):
        _, _, chunks = self.chunks("a" * 650 + "\n" + "b" * 500 + ".")
        self.assertEqual(chunks[0].end_line, 1)
        self.assertNotIn("b", chunks[0].text)

    def test_chunk_count_limit_fails_instead_of_returning_partial(self):
        with patch.object(config, "MAX_CHUNKS_PER_DOCUMENT", 1):
            with self.assertRaisesRegex(ValueError, "Chunk 数量"):
                self.chunks("a" * 2500)

    def test_invalid_overlap_configuration_rejected(self):
        with patch.object(config, "CHUNK_OVERLAP_TOKENS", 512):
            with self.assertRaises(ValueError):
                self.chunks("hello")

    def test_pdf_citation_has_real_page_and_no_score(self):
        _, doc, chunks = self.chunks(pdf_bytes(["first", "ORANGE-731"]), "a.pdf")
        citation = citation_from_chunk(chunks[-1], doc, "kb_" + "a" * 32)
        self.assertEqual(citation.page, 2)
        self.assertIsNone(citation.score)
        self.assertIsNone(citation.score_type)

    def test_txt_citation_original_lines(self):
        _, doc, chunks = self.chunks("\n\nORANGE-731\nThursday 16:20\n")
        citation = citation_from_chunk(chunks[0], doc, "kb_" + "a" * 32)
        self.assertEqual((citation.start_line, citation.end_line), (3, 4))

    def test_markdown_citation_heading_and_lines(self):
        _, doc, chunks = self.chunks("# 星河\n## 制度\n\nORANGE-731", "a.md")
        citation = citation_from_chunk(chunks[-1], doc, "kb_" + "a" * 32)
        self.assertEqual((citation.heading, citation.start_line, citation.end_line), ("星河 > 制度", 4, 4))

    def test_citation_rejects_different_document(self):
        _, doc, chunks = self.chunks("one")
        _, other, _ = self.chunks("two")
        with self.assertRaises(ValueError):
            citation_from_chunk(chunks[0], other, "kb_" + "a" * 32)


if __name__ == "__main__":
    unittest.main()
