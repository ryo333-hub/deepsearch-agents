"""Temporary offline fixtures shared by second-layer Local RAG tests."""

import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter
from reportlab.pdfgen.canvas import Canvas

from app.api.context import reset_session_context, set_session_context, set_thread_context
from app.local_rag.loaders import UploadedDocumentLoader
from app.local_rag.storage import LocalRAGStorage
from app.local_rag.tokenization import DocumentTokenBudget


class CharacterTokenizer:
    """Offline test tokenizer: one character/token plus two special tokens."""
    model_max_length = 512

    def __call__(self, text, *, add_special_tokens, **kwargs):
        assert kwargs.get("truncation") is False
        return {"input_ids": [1] * (len(text) + (2 if add_special_tokens else 0))}


def pdf_bytes(pages):
    stream = io.BytesIO()
    canvas = Canvas(stream)
    for text in pages:
        if text:
            canvas.drawString(40, 780, text)
        canvas.showPage()
    canvas.save()
    return stream.getvalue()


def blank_pdf_bytes():
    stream = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.write(stream)
    return stream.getvalue()


class OfflineDocumentTestCase(unittest.TestCase):
    def setUp(self):
        project_tmp = Path.cwd() / ".tmp"
        project_tmp.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=project_tmp, prefix="rag-ingestion-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.upload_root = self.base / "uploads"
        self.session_dir = self.upload_root / "session_session-A"
        self.session_dir.mkdir(parents=True)
        session_token = set_session_context(str(self.session_dir))
        thread_token = set_thread_context("session-A")
        self.addCleanup(reset_session_context, session_token, thread_token)
        self.loader = UploadedDocumentLoader(self.upload_root)
        self.storage = LocalRAGStorage(self.base / "store")
        self.token_budget = DocumentTokenBudget(tokenizer=CharacterTokenizer())
        token_patch = patch("app.local_rag.chunking.get_document_token_budget", return_value=self.token_budget)
        token_patch.start()
        self.addCleanup(token_patch.stop)
        self.network_mocks = []
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.getaddrinfo"):
            blocker = patch(target, side_effect=AssertionError("Network prohibited"))
            self.network_mocks.append(blocker.start())
            self.addCleanup(blocker.stop)
        self.addCleanup(self.assert_no_network)

    def assert_no_network(self):
        for mock in self.network_mocks:
            mock.assert_not_called()

    def upload(self, name, content):
        path = self.session_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
        return name

    def create_junction(self, link, target):
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            if os.name != "nt":
                self.skipTest("Symlink unavailable")
            env = os.environ.copy()
            env["RAG_FIXTURE_LINK"] = str(link)
            env["RAG_FIXTURE_TARGET"] = str(target)
            result = subprocess.run([
                "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                "New-Item -ItemType Junction -Path $env:RAG_FIXTURE_LINK "
                "-Target $env:RAG_FIXTURE_TARGET -ErrorAction Stop | Out-Null",
            ], env=env, capture_output=True, check=False)
            if result.returncode:
                self.skipTest("Junction unavailable")
        def cleanup():
            if link.is_symlink():
                link.unlink()
            elif link.is_junction():
                link.rmdir()
        self.addCleanup(cleanup)
