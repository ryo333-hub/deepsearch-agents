"""Offline extraction from the current session's original upload space.

Only a trusted backend/test may configure upload_root. No URL fetching, OCR,
Markdown rendering, scripting, includes, or arbitrary host-path entry point.
"""

import hashlib
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError
from pypdf import PdfReader

from app.api.context import get_thread_context
from app.local_rag import config
from app.local_rag.schemas import ParsedDocument, TextBlock
from app.utils.path_utils import resolve_path, resolve_session_directory, validate_thread_id


class DocumentLoadError(ValueError):
    """Safe extraction error with no host paths or file contents."""


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


@dataclass(frozen=True)
class DocumentSnapshot:
    document_name: str
    source_type: str
    data: bytes

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


class UploadedDocumentLoader:
    def __init__(self, upload_root: str | Path = config.LOCAL_RAG_UPLOAD_ROOT):
        self._upload_root = Path(os.path.abspath(upload_root))

    def read(self, filename: str) -> DocumentSnapshot:
        """Read once into a bounded snapshot; reuse bytes for hashing/parsing."""
        try:
            session_id = validate_thread_id(get_thread_context())
            root = self._upload_root
            if root.resolve() != root or root.is_symlink() or root.is_junction():
                raise ValueError
            raw_session = root / f"session_{session_id}"
            if raw_session.is_symlink() or raw_session.is_junction():
                raise ValueError
            session_dir = resolve_session_directory(root, session_id)
            if session_dir.exists() and session_dir.resolve().name != raw_session.name:
                raise ValueError
            file_path = Path(resolve_path(filename, session_dir))
            # Same-session links are unnecessary; reject hardlinks too.
            lexical = session_dir / filename.replace("\\", "/")
            current = lexical
            while current != session_dir:
                if current.is_symlink() or current.is_junction():
                    raise ValueError
                current = current.parent
            ext = file_path.suffix.lower()
            source_type = {".txt": "txt", ".md": "markdown", ".markdown": "markdown", ".pdf": "pdf"}.get(ext)
            if source_type is None:
                raise DocumentLoadError("不支持的文档格式，仅支持 TXT、Markdown 和 PDF。")
            if not file_path.is_file() or file_path.stat().st_nlink != 1:
                raise ValueError
            if file_path.stat().st_size > config.MAX_DOCUMENT_BYTES:
                raise DocumentLoadError("文档超过最大文件大小限制。")
            with file_path.open("rb") as stream:
                data = stream.read(config.MAX_DOCUMENT_BYTES + 1)
            if len(data) > config.MAX_DOCUMENT_BYTES:
                raise DocumentLoadError("文档超过最大文件大小限制。")
            return DocumentSnapshot(file_path.name, source_type, data)
        except DocumentLoadError:
            raise
        except (OSError, RuntimeError, ValueError, TypeError, AttributeError):
            raise DocumentLoadError("上传文档不存在或不允许访问当前 session 外的文件。") from None

    def load(self, filename: str) -> ParsedDocument:
        return parse_snapshot(self.read(filename))


def _paragraph_blocks(lines: list[str], markdown: bool) -> tuple[TextBlock, ...]:
    blocks = []
    start = None
    heading = None
    headings = []
    fence = None

    def flush(end):
        nonlocal start
        if start is not None:
            text = "\n".join(lines[start:end])
            if text.strip():
                blocks.append(TextBlock(block_index=len(blocks), text=text,
                                        start_line=start + 1, end_line=end, heading=heading))
            start = None

    index = 0
    while index < len(lines):
        line = lines[index]
        if markdown:
            fence_match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
            if fence:
                if fence_match and fence_match[1][0] == fence[0] and len(fence_match[1]) >= fence[1] and not fence_match[2].strip():
                    fence = None
                    flush(index + 1)
                index += 1
                continue
            if fence_match:
                flush(index)
                start = index
                fence = (fence_match[1][0], len(fence_match[1]))
                index += 1
                continue
            match = re.match(r"^ {0,3}(#{1,6})(?:\s+|$)(.*?)\s*$", line)
            setext = (index + 1 < len(lines) and line.strip()
                      and re.fullmatch(r" {0,3}(=+|-+)\s*", lines[index + 1]))
            if match or setext:
                flush(index)
                level = len(match[1]) if match else (1 if lines[index + 1].lstrip().startswith("=") else 2)
                title = re.sub(r"\s+#+\s*$", "", match[2]).strip() if match else line.strip()
                headings = [(depth, name) for depth, name in headings if depth < level]
                if title:
                    headings.append((level, title))
                heading = " > ".join(name for _, name in headings) or None
                start = index
                index += 1 if match else 2
                flush(index)
                continue
        if not line.strip():
            flush(index)
        elif start is None:
            start = index
        index += 1
    flush(len(lines))
    return tuple(blocks)


def parse_snapshot(snapshot: DocumentSnapshot) -> ParsedDocument:
    """Parse an already authorized in-memory snapshot, without rereading files."""
    if len(snapshot.data) > config.MAX_DOCUMENT_BYTES:
        raise DocumentLoadError("文档超过最大文件大小限制。")
    common = dict(document_name=snapshot.document_name, source_type=snapshot.source_type,
                  content_hash=snapshot.content_hash, size_bytes=len(snapshot.data))
    try:
        if snapshot.source_type in ("txt", "markdown"):
            try:
                text = normalize_newlines(snapshot.data.decode("utf-8-sig"))
            except UnicodeDecodeError:
                raise DocumentLoadError("文本不是有效 UTF-8 编码，请先转换为 UTF-8。") from None
            if len(text) > config.MAX_EXTRACTED_CHARS:
                raise DocumentLoadError("提取文本超过最大字符数限制。")
            if "\x00" in text:
                raise DocumentLoadError("文本包含不支持的 NUL 字符。")
            if not text.strip():
                raise DocumentLoadError("文档无有效文本。")
            lines = text.split("\n")
            if text.endswith("\n"):
                lines.pop()  # Terminal newline is not another physical line.
            blocks = _paragraph_blocks(lines, snapshot.source_type == "markdown")
            return ParsedDocument(**common, blocks=blocks, line_count=len(lines))
        if snapshot.source_type != "pdf":
            raise DocumentLoadError("不支持的文档格式。")
        reader = PdfReader(io.BytesIO(snapshot.data))
        if reader.is_encrypted:
            raise DocumentLoadError("暂不支持加密 PDF。")
        count = len(reader.pages)
        if count > config.MAX_PDF_PAGES:
            raise DocumentLoadError("PDF 超过最大页数限制。")
        blocks, empty_pages, total = [], [], 0
        for page_number, page in enumerate(reader.pages, 1):
            text = normalize_newlines(page.extract_text() or "")
            total += len(text)
            if total > config.MAX_EXTRACTED_CHARS:
                raise DocumentLoadError("提取文本超过最大字符数限制。")
            if text.strip():
                blocks.append(TextBlock(block_index=len(blocks), text=text, page=page_number))
            else:
                empty_pages.append(page_number)
        if not blocks:
            raise DocumentLoadError("PDF 无可提取文本，可能是扫描件或图片型 PDF。")
        return ParsedDocument(**common, blocks=tuple(blocks), page_count=count, empty_pages=tuple(empty_pages))
    except DocumentLoadError:
        raise
    except ValidationError:
        raise DocumentLoadError("文档名称或定位信息无法安全保存。") from None
    except Exception:
        raise DocumentLoadError("PDF 或文档解析失败，未生成完整提取结果。") from None
