"""Offline metadata/security tests. All storage lives in temporary directories."""

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from app.api.context import reset_session_context, set_session_context, set_thread_context
from app.local_rag.config import LOCAL_RAG_DATA_ROOT, MAX_EVIDENCE_TEXT_LENGTH
from app.local_rag.schemas import ChunkRecord, Citation, DocumentRecord, Evidence
from app.local_rag.storage import (
    KNOWLEDGE_BASE_ACCESS_ERROR, KnowledgeBaseAccessError,
    LocalRAGStorage, LocalRAGStorageError,
)


def digest(text="fake document"):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class LocalRAGStorageTests(unittest.TestCase):
    def setUp(self):
        project_tmp = Path.cwd() / ".tmp"
        project_tmp.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=project_tmp, prefix="local-rag-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "data"
        self.storage = LocalRAGStorage(self.root)
        self.session_token = set_session_context(str(self.base / "unrelated-upload-session"))
        self.thread_token = set_thread_context("session-A")
        self.addCleanup(reset_session_context, self.session_token, self.thread_token)
        self.network_mocks = []
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.getaddrinfo"):
            blocker = patch(target, side_effect=AssertionError("Networking prohibited"))
            self.network_mocks.append(blocker.start())
            self.addCleanup(blocker.stop)
        self.addCleanup(self.assert_no_network)

    def assert_no_network(self):
        for mock in self.network_mocks:
            mock.assert_not_called()

    def kb_dir(self, kb, session="session-A"):
        return self.root / "sessions" / f"session_{session}" / kb.knowledge_base_id

    def create(self, name="测试知识库"):
        return self.storage.create_knowledge_base(name)

    def register(self, kb, **overrides):
        values = dict(document_name="测试制度.txt", source_type="txt", content_hash=digest(), size_bytes=28)
        values.update(overrides)
        return self.storage.register_document_metadata(kb.knowledge_base_id, **values)

    def change_manifest(self, kb, **changes):
        path = self.kb_dir(kb) / "manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(changes)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def assert_denied(self, kb_id):
        with self.assertRaisesRegex(KnowledgeBaseAccessError, KNOWLEDGE_BASE_ACCESS_ERROR):
            self.storage.get_knowledge_base(kb_id)

    def make_directory_link(self, link, target):
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            if os.name != "nt":
                self.skipTest("Directory symlinks unavailable")
            env = os.environ.copy()
            env["LOCAL_RAG_TEST_LINK"] = str(link)
            env["LOCAL_RAG_TEST_TARGET"] = str(target)
            completed = subprocess.run([
                "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                "New-Item -ItemType Junction -Path $env:LOCAL_RAG_TEST_LINK "
                "-Target $env:LOCAL_RAG_TEST_TARGET -ErrorAction Stop | Out-Null",
            ], env=env, capture_output=True, check=False)
            if completed.returncode:
                self.skipTest("Directory symlink/junction unavailable")
        def remove_link():
            if link.is_symlink():
                link.unlink()
            elif link.is_junction():
                link.rmdir()
        self.addCleanup(remove_link)

    def test_constructor_and_empty_list_do_not_create_storage(self):
        self.assertEqual(self.storage.list_knowledge_bases(), [])
        self.assertFalse(self.root.exists())

    def test_default_root_is_separate_backend_storage(self):
        self.assertEqual(LOCAL_RAG_DATA_ROOT, Path(__file__).resolve().parents[1] / ".data" / "local_rag")

    def test_create_persists_manifest_and_only_documents_directory(self):
        kb = self.create()
        self.assertEqual(kb.session_id, "session-A")
        self.assertEqual(kb.index_status, "empty")
        self.assertEqual({p.name for p in self.kb_dir(kb).iterdir()}, {"manifest.json", "documents"})
        self.assertEqual(self.storage.get_knowledge_base(kb.knowledge_base_id), kb)

    def test_generated_ids_are_opaque_and_distinct(self):
        first, second = self.create(), self.create()
        self.assertRegex(first.knowledge_base_id, r"^kb_[0-9a-f]{12}4[0-9a-f]{3}[89ab][0-9a-f]{15}$")
        self.assertNotEqual(first.knowledge_base_id, second.knowledge_base_id)
        self.assertNotIn("session-A", first.knowledge_base_id)

    def test_display_name_cannot_control_directory(self):
        name = r"../../C:\outside/预算:2026"
        kb = self.create(name)
        self.assertEqual(kb.name, name)
        self.assertEqual(self.kb_dir(kb).parent.name, "session_session-A")
        self.assertTrue(self.kb_dir(kb).is_dir())
        self.assertFalse((self.base / "outside").exists())

    def test_list_contains_only_summary_without_paths(self):
        kb = self.create()
        self.register(kb)
        values = self.storage.list_knowledge_bases()
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0].document_count, 1)
        self.assertEqual(values[0].index_status, "pending")
        self.assertEqual(set(values[0].model_dump()), {"knowledge_base_id", "name", "created_at", "document_count", "index_status"})
        self.assertNotIn(str(self.root), values[0].model_dump_json())

    def test_sessions_list_only_their_own_kbs(self):
        a = self.create("A")
        set_thread_context("session-B")
        self.assertEqual(self.storage.list_knowledge_bases(), [])
        b = self.create("B")
        self.assertEqual([item.knowledge_base_id for item in self.storage.list_knowledge_bases()], [b.knowledge_base_id])
        set_thread_context("session-A")
        self.assertEqual([item.knowledge_base_id for item in self.storage.list_knowledge_bases()], [a.knowledge_base_id])

    def test_foreign_kb_and_unknown_kb_have_same_error(self):
        a = self.create()
        set_thread_context("session-B")
        self.assert_denied(a.knowledge_base_id)
        self.assert_denied("kb_" + "0" * 32)

    def test_cannot_register_document_in_foreign_kb(self):
        a = self.create()
        original = (self.kb_dir(a) / "manifest.json").read_bytes()
        set_thread_context("session-B")
        with self.assertRaises(KnowledgeBaseAccessError):
            self.register(a)
        self.assertEqual((self.kb_dir(a) / "manifest.json").read_bytes(), original)

    def test_invalid_kb_ids_rejected(self):
        for value in (None, 12, "", "kb_abc", "kb_" + "A" * 32, "kb_" + "a" * 32 + "\n"):
            with self.subTest(value=value):
                self.assert_denied(value)

    def test_path_forms_rejected_as_kb_ids(self):
        for value in ("../other", r"..\other", r"C:\Windows\file", "C:relative", r"\\host\share", r"\\?\C:\file", "/tmp/file", "kb_x:stream", "file.", "CON"):
            with self.subTest(value=value):
                self.assert_denied(value)
        self.assertFalse(self.root.exists())

    def test_session_missing_rejects_all_apis(self):
        set_thread_context(None)
        actions = [lambda: self.create(), self.storage.list_knowledge_bases,
                   lambda: self.storage.get_knowledge_base("kb_" + "a" * 32),
                   lambda: self.storage.register_document_metadata("kb_" + "a" * 32, document_name="a.txt", source_type="txt", content_hash=digest(), size_bytes=1)]
        for action in actions:
            with self.assertRaisesRegex(ValueError, "session"):
                action()
        self.assertFalse(self.root.exists())

    def test_illegal_sessions_use_existing_validation(self):
        for value in ("../escape", "a/b", "C:foo", "中文", "a" * 129, "A\n", ""):
            with self.subTest(value=value):
                set_thread_context(value)
                with self.assertRaises(ValueError):
                    self.create()
        self.assertFalse(self.root.exists())

    def test_storage_does_not_use_upload_directory_context(self):
        kb = self.create()
        self.assertTrue(self.kb_dir(kb).exists())
        self.assertFalse((self.base / "unrelated-upload-session").exists())

    def test_case_alias_cannot_mix_sessions_on_windows(self):
        a = self.create()
        set_thread_context("session-a")
        self.assert_denied(a.knowledge_base_id)
        if os.name == "nt":
            with self.assertRaises(KnowledgeBaseAccessError):
                self.create()
        else:
            self.assertEqual(self.storage.list_knowledge_bases(), [])

    def test_manifest_session_mismatch_rejected_and_hidden(self):
        kb = self.create()
        self.change_manifest(kb, session_id="session-B")
        self.assert_denied(kb.knowledge_base_id)
        self.assertEqual(self.storage.list_knowledge_bases(), [])

    def test_manifest_kb_id_mismatch_rejected(self):
        kb = self.create()
        self.change_manifest(kb, knowledge_base_id="kb_" + "a" * 32)
        self.assert_denied(kb.knowledge_base_id)

    def test_manifest_schema_version_rejected(self):
        kb = self.create()
        for value in (2, True, "1", None):
            with self.subTest(value=value):
                self.change_manifest(kb, schema_version=value)
                self.assert_denied(kb.knowledge_base_id)

    def test_missing_manifest_version_rejected(self):
        kb = self.create()
        path = self.kb_dir(kb) / "manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        del value["schema_version"]
        path.write_text(json.dumps(value), encoding="utf-8")
        self.assert_denied(kb.knowledge_base_id)

    def test_corrupt_manifest_rejected_without_absolute_path(self):
        kb = self.create()
        (self.kb_dir(kb) / "manifest.json").write_text("{broken", encoding="utf-8")
        self.assert_denied(kb.knowledge_base_id)
        self.assertEqual(self.storage.list_knowledge_bases(), [])

    def test_manifest_chinese_utf8_and_stable_json(self):
        kb = self.create("内部联调知识库")
        raw = (self.kb_dir(kb) / "manifest.json").read_bytes()
        self.assertIn("内部联调知识库".encode(), raw)
        self.assertNotIn(b"\\u", raw)
        value = json.loads(raw)
        self.assertEqual(raw.decode(), json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")

    def test_invalid_display_names_rejected_before_io(self):
        for name in ("", " ", "a\nb", "a\tb", "a\x00b", "a\x7fb", "a\u202eb", "x" * 121):
            with self.subTest(name=name), self.assertRaises(ValidationError):
                self.create(name)
        self.assertFalse(self.root.exists())

    def test_document_registration_generates_id_and_persists_metadata(self):
        kb = self.create()
        doc = self.register(kb)
        self.assertRegex(doc.document_id, r"^doc_[0-9a-f]{32}$")
        self.assertEqual(self.storage.get_knowledge_base(kb.knowledge_base_id).documents, (doc,))
        self.assertEqual(list((self.kb_dir(kb) / "documents").iterdir()), [])

    def test_duplicate_hash_returns_existing_document_without_rewrite(self):
        kb = self.create()
        first = self.register(kb)
        original = (self.kb_dir(kb) / "manifest.json").read_bytes()
        second = self.register(kb, document_name="different.txt")
        self.assertEqual(first, second)
        self.assertEqual(len(self.storage.get_knowledge_base(kb.knowledge_base_id).documents), 1)
        self.assertEqual((self.kb_dir(kb) / "manifest.json").read_bytes(), original)

    def test_same_hash_in_different_kbs_is_not_shared(self):
        first = self.register(self.create())
        second = self.register(self.create())
        self.assertNotEqual(first.document_id, second.document_id)

    def test_registration_rejects_caller_identity_and_path(self):
        kb = self.create()
        for name in ("document_id", "session_id", "storage_path"):
            with self.subTest(name=name), self.assertRaises(TypeError):
                self.register(kb, **{name: "unauthorized"})

    def test_bad_document_metadata_rejected(self):
        kb = self.create()
        for change in ({"content_hash":"invalid"}, {"size_bytes":-1}, {"size_bytes":True}, {"source_type":"exe"}, {"document_name":r"C:\secret.txt"}, {"document_name":".env"}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                self.register(kb, **change)
        self.assertEqual(self.storage.get_knowledge_base(kb.knowledge_base_id).documents, ())

    def test_fresh_instance_can_read_persisted_manifest(self):
        kb = self.create()
        self.register(kb)
        self.assertEqual(len(LocalRAGStorage(self.root).get_knowledge_base(kb.knowledge_base_id).documents), 1)

    def test_atomic_replace_failure_preserves_manifest_and_cleans_temp(self):
        kb = self.create()
        path = self.kb_dir(kb) / "manifest.json"
        original = path.read_bytes()
        with patch("app.local_rag.storage.os.replace", side_effect=OSError("fake failure")):
            with self.assertRaises(LocalRAGStorageError):
                self.register(kb)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.kb_dir(kb).glob("*.tmp")), [])

    def test_initial_write_failure_leaves_no_partial_kb(self):
        with patch("app.local_rag.storage.os.replace", side_effect=OSError("fake failure")):
            with self.assertRaises(LocalRAGStorageError):
                self.create()
        self.assertEqual(self.storage.list_knowledge_bases(), [])
        self.assertEqual(list((self.root / "sessions" / "session_session-A").iterdir()), [])

    def test_concurrent_instances_do_not_lose_document_updates(self):
        kb = self.create()
        def register(index):
            token = set_thread_context("session-A")
            session_token = set_session_context(None)
            try:
                return LocalRAGStorage(self.root).register_document_metadata(kb.knowledge_base_id, document_name=f"{index}.txt", source_type="txt", content_hash=digest(str(index)), size_bytes=1)
            finally:
                reset_session_context(session_token, token)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(register, range(8)))
        self.assertEqual(len(self.storage.get_knowledge_base(kb.knowledge_base_id).documents), 8)

    def test_kb_junction_escape_is_rejected(self):
        outside = self.base / "outside"
        outside.mkdir()
        kb_id = "kb_" + "a" * 32
        link = self.root / "sessions" / "session_session-A" / kb_id
        self.make_directory_link(link, outside)
        self.assert_denied(kb_id)
        self.assertEqual(self.storage.list_knowledge_bases(), [])
        self.assertEqual(list(outside.iterdir()), [])

    def test_session_junction_to_other_session_is_rejected(self):
        b_dir = self.root / "sessions" / "session_session-B"
        b_dir.mkdir(parents=True)
        self.make_directory_link(self.root / "sessions" / "session_session-A", b_dir)
        with self.assertRaises(KnowledgeBaseAccessError):
            self.create()
        with self.assertRaises(KnowledgeBaseAccessError):
            self.storage.list_knowledge_bases()
        self.assertEqual(list(b_dir.iterdir()), [])

    def test_sessions_root_junction_escape_rejected(self):
        outside = self.base / "outside"
        outside.mkdir()
        self.make_directory_link(self.root / "sessions", outside)
        with self.assertRaises(KnowledgeBaseAccessError):
            self.create()
        self.assertEqual(list(outside.iterdir()), [])

    def test_storage_root_junction_escape_rejected(self):
        outside = self.base / "outside"
        outside.mkdir()
        self.make_directory_link(self.root, outside)
        with self.assertRaises(KnowledgeBaseAccessError):
            self.create()
        self.assertEqual(list(outside.iterdir()), [])

    def test_manifest_hardlink_rejected(self):
        kb = self.create()
        manifest = self.kb_dir(kb) / "manifest.json"
        other = self.base / "outside.json"
        other.write_bytes(manifest.read_bytes())
        manifest.unlink()
        try:
            os.link(other, manifest)
        except OSError:
            self.skipTest("Hard links unavailable")
        self.assert_denied(kb.knowledge_base_id)


class LocalRAGContractTests(unittest.TestCase):
    def document(self):
        return DocumentRecord(document_name="测试制度.pdf", source_type="pdf", content_hash=digest(), size_bytes=20)

    def citation(self, **changes):
        doc = self.document()
        chunk = ChunkRecord(document_id=doc.document_id, chunk_index=0, text="测试事实", text_hash=digest("测试事实"))
        values = dict(citation_id="C1", document_id=doc.document_id, document_name=doc.document_name, chunk_id=chunk.chunk_id, knowledge_base_id="kb_" + "a" * 32)
        values.update(changes)
        return Citation(**values)

    def test_chunk_id_generated_without_vector(self):
        chunk = ChunkRecord(document_id=self.document().document_id, chunk_index=0, text="测试事实", text_hash=digest("测试事实"))
        self.assertRegex(chunk.chunk_id, r"^chunk_[0-9a-f]{32}$")
        self.assertNotIn("embedding", chunk.model_dump())

    def test_citation_json_roundtrip_without_score_or_location(self):
        citation = self.citation()
        self.assertIsNone(citation.score)
        self.assertIsNone(citation.score_type)
        self.assertIsNone(citation.page)
        self.assertEqual(Citation.model_validate_json(citation.model_dump_json()), citation)

    def test_citation_retains_real_location_and_score(self):
        citation = self.citation(page=2, score=0.82, score_type="cosine_similarity")
        self.assertEqual(citation.page, 2)
        self.assertEqual(citation.score, 0.82)

    def test_citation_rejects_paths_secrets_and_internal_fields(self):
        for changes in ({"document_name":r"C:\secret.pdf"}, {"document_name":"../secret.pdf"}, {"document_name":".env"}, {"document_name":".env.local"}, {"heading":r"C:\secret"}, {"api_key":"fake-key"}, {"session_id":"session-A"}, {"chroma_path":"internal"}, {"metadata":{"path":"internal"}}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                self.citation(**changes)

    def test_nonfinite_scores_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                self.citation(score=value)

    def test_invalid_locations_rejected(self):
        for values in ({"page":0}, {"start_line":2}, {"start_line":5,"end_line":2}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                self.citation(**values)

    def test_evidence_serializes_bounded_fake_text_and_citation(self):
        evidence = Evidence(text="星河项目代号 ORANGE-731", citation=self.citation(start_line=1, end_line=2))
        self.assertEqual(Evidence.model_validate_json(evidence.model_dump_json()), evidence)
        self.assertEqual(set(evidence.model_dump()), {"text", "citation", "truncated"})

    def test_evidence_empty_or_oversized_text_rejected(self):
        for text in ("", "a" * (MAX_EVIDENCE_TEXT_LENGTH + 1)):
            with self.assertRaises(ValidationError):
                Evidence(text=text, citation=self.citation())


if __name__ == "__main__":
    unittest.main()
