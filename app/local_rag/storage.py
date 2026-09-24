"""Small, synchronous metadata store for a single local application process.

Only the backend may configure data_root. No Agent-facing path/session inputs.
The process lock prevents lost updates across instances/threads in this process;
multi-process writers and hostile concurrent filesystem mutation are outside
this MVP. Resolved containment and link rejection are checked before file IO.
"""

import json
import hashlib
import os
import re
from pathlib import Path
from threading import RLock
from uuid import uuid4
from collections.abc import Callable

from app.api.context import get_thread_context
from app.local_rag.config import LOCAL_RAG_DATA_ROOT, MAX_MANIFEST_BYTES
from app.local_rag import config
from app.local_rag.schemas import (
    ChunkRecord, DocumentRecord, IngestionResult, KnowledgeBaseManifest,
    KnowledgeBaseSummary, ParsedDocument,
)
from app.utils.path_utils import validate_thread_id


KNOWLEDGE_BASE_ACCESS_ERROR = "knowledge base 不存在或无访问权限"
_KB_ID_PATTERN = re.compile(r"kb_[0-9a-f]{32}")
_WRITE_LOCK = RLock()


class KnowledgeBaseAccessError(ValueError):
    """Uniform error for absent, malformed, foreign or unsafe KB metadata."""


class LocalRAGStorageError(RuntimeError):
    """Sanitized storage failure; no backend paths in its message."""


class LocalRAGStorage:
    def __init__(self, data_root: str | Path = LOCAL_RAG_DATA_ROOT):
        # Trusted backend configuration, not a path accepted by a public tool.
        self._root = Path(os.path.abspath(data_root))

    def _checked(self, path: Path, boundary: Path) -> Path:
        """Reject links (including links within another session) and escapes."""
        try:
            if not path.is_relative_to(self._root) or not path.is_relative_to(boundary):
                raise ValueError
            if self._root.resolve() != self._root:
                raise ValueError
            current = self._root
            for component in (None, *path.relative_to(self._root).parts):
                if component is not None:
                    current = current / component
                if current.is_symlink() or current.is_junction():
                    raise ValueError
            resolved = path.resolve()
            if not resolved.is_relative_to(boundary.resolve()):
                raise ValueError
        except (OSError, RuntimeError, ValueError):
            raise KnowledgeBaseAccessError(KNOWLEDGE_BASE_ACCESS_ERROR) from None
        return path

    def _session(self) -> tuple[str, Path]:
        try:
            session_id = validate_thread_id(get_thread_context())
        except ValueError:
            raise ValueError("Local RAG 需要合法的当前 session") from None
        path = self._root / "sessions" / f"session_{session_id}"
        self._checked(path, self._root)
        # Windows is case-insensitive: session-A must not alias session-a.
        if path.exists() and path.resolve().name != path.name:
            raise KnowledgeBaseAccessError(KNOWLEDGE_BASE_ACCESS_ERROR)
        return session_id, path

    def _kb_path(self, session_dir: Path, knowledge_base_id: str) -> Path:
        if not isinstance(knowledge_base_id, str) or not _KB_ID_PATTERN.fullmatch(knowledge_base_id):
            raise KnowledgeBaseAccessError(KNOWLEDGE_BASE_ACCESS_ERROR)
        return self._checked(session_dir / knowledge_base_id, session_dir)

    def _load(self, session_id: str, session_dir: Path, knowledge_base_id: str) -> KnowledgeBaseManifest:
        try:
            kb_dir = self._kb_path(session_dir, knowledge_base_id)
            path = self._checked(kb_dir / "manifest.json", session_dir)
            info = path.stat()
            if info.st_size > MAX_MANIFEST_BYTES or info.st_nlink != 1:
                raise ValueError
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not {"schema_version", "session_id", "knowledge_base_id"} <= data.keys():
                raise ValueError
            manifest = KnowledgeBaseManifest.model_validate(data)
            if manifest.session_id != session_id or manifest.knowledge_base_id != knowledge_base_id:
                raise ValueError
            return manifest
        except (OSError, UnicodeError, ValueError):
            raise KnowledgeBaseAccessError(KNOWLEDGE_BASE_ACCESS_ERROR) from None

    def _write(self, manifest: KnowledgeBaseManifest, session_dir: Path) -> None:
        kb_dir = self._kb_path(session_dir, manifest.knowledge_base_id)
        target = self._checked(kb_dir / "manifest.json", session_dir)
        payload = json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if len(payload.encode("utf-8")) > MAX_MANIFEST_BYTES:
            raise LocalRAGStorageError("Local RAG manifest 超出大小限制")
        temporary = kb_dir / f".manifest-{uuid4().hex}.tmp"
        created = False
        try:
            self._checked(temporary, session_dir)
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                created = True
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            self._checked(temporary, session_dir)
            self._checked(target, session_dir)
            os.replace(temporary, target)
        except OSError:
            raise LocalRAGStorageError("Local RAG manifest 保存失败") from None
        finally:
            if created:
                self._checked(temporary, session_dir).unlink(missing_ok=True)

    def create_knowledge_base(self, name: str) -> KnowledgeBaseManifest:
        with _WRITE_LOCK:
            session_id, session_dir = self._session()
            manifest = KnowledgeBaseManifest(name=name, session_id=session_id)
            kb_dir = self._kb_path(session_dir, manifest.knowledge_base_id)
            documents_dir = self._checked(kb_dir / "documents", session_dir)
            created = False
            try:
                session_dir.mkdir(parents=True, exist_ok=True)
                self._session()
                self._checked(kb_dir, session_dir).mkdir(exist_ok=False)
                created = True
                self._checked(documents_dir, session_dir).mkdir()
                self._write(manifest, session_dir)
            except (OSError, LocalRAGStorageError):
                if created:
                    # Only remove our own empty directories; never recursive delete.
                    if self._checked(documents_dir, session_dir).is_dir():
                        documents_dir.rmdir()
                    self._checked(kb_dir, session_dir).rmdir()
                raise LocalRAGStorageError("Local RAG 知识库创建失败") from None
            return manifest

    def get_knowledge_base(self, knowledge_base_id: str) -> KnowledgeBaseManifest:
        with _WRITE_LOCK:
            session_id, session_dir = self._session()
            return self._load(session_id, session_dir, knowledge_base_id)

    def list_knowledge_bases(self) -> list[KnowledgeBaseSummary]:
        with _WRITE_LOCK:
            session_id, session_dir = self._session()
            if not session_dir.exists():
                return []
            result = []
            # Enumerate only this session; never scan the sessions root.
            for entry in sorted(session_dir.iterdir(), key=lambda path: path.name):
                if not _KB_ID_PATTERN.fullmatch(entry.name):
                    continue
                try:
                    manifest = self._load(session_id, session_dir, entry.name)
                except KnowledgeBaseAccessError:
                    continue  # Never expose corrupt, foreign or linked entries.
                result.append(KnowledgeBaseSummary(
                    knowledge_base_id=manifest.knowledge_base_id, name=manifest.name,
                    created_at=manifest.created_at, document_count=len(manifest.documents),
                    index_status=manifest.index_status,
                ))
            return result

    def register_document_metadata(
        self, knowledge_base_id: str, *, document_name: str,
        source_type: str, content_hash: str, size_bytes: int,
    ) -> DocumentRecord:
        with _WRITE_LOCK:
            session_id, session_dir = self._session()
            manifest = self._load(session_id, session_dir, knowledge_base_id)
            document = DocumentRecord(
                document_name=document_name, source_type=source_type,
                content_hash=content_hash, size_bytes=size_bytes,
            )
            for existing in manifest.documents:
                if existing.content_hash == document.content_hash:
                    return existing
            updated = KnowledgeBaseManifest.model_validate({
                **manifest.model_dump(), "documents": (*manifest.documents, document),
                "index_status": "pending",
            })
            self._write(updated, session_dir)
            return document

    def get_document_ingestion(self, knowledge_base_id: str, document_id: str) -> IngestionResult:
        """Reload only committed documents from this session's manifest."""
        with _WRITE_LOCK:
            session_id, session_dir = self._session()
            manifest = self._load(session_id, session_dir, knowledge_base_id)
            document = next((doc for doc in manifest.documents if doc.document_id == document_id), None)
            if document is None or document.ingestion_status not in {"chunked", "indexed"}:
                raise KnowledgeBaseAccessError(KNOWLEDGE_BASE_ACCESS_ERROR)
            if document.chunking_version != config.CHUNKING_VERSION:
                raise LocalRAGStorageError("分块版本或配置已变化，需要显式重新入库；未修改旧数据。")
            try:
                directory = self._kb_path(session_dir, knowledge_base_id) / "documents" / document.document_id
                def read_file(name, maximum):
                    path = self._checked(directory / name, session_dir)
                    info = path.stat()
                    if info.st_size > maximum or info.st_nlink != 1:
                        raise ValueError
                    return path.read_text(encoding="utf-8")
                metadata = DocumentRecord.model_validate_json(read_file("metadata.json", MAX_MANIFEST_BYTES))
                if metadata != document:
                    raise ValueError
                lines = read_file("chunks.jsonl", config.MAX_CHUNKS_FILE_BYTES).splitlines()
                if not 0 < len(lines) <= config.MAX_CHUNKS_PER_DOCUMENT or len(lines) != document.chunk_count:
                    raise ValueError
                chunks = tuple(ChunkRecord.model_validate_json(line) for line in lines)
                self._validate_chunks(document, chunks)
                return IngestionResult(document=document, chunks=chunks)
            except (OSError, UnicodeError, ValueError):
                raise KnowledgeBaseAccessError(KNOWLEDGE_BASE_ACCESS_ERROR) from None

    @staticmethod
    def _validate_chunks(document: DocumentRecord, chunks: tuple[ChunkRecord, ...]) -> None:
        if not chunks or len(chunks) > config.MAX_CHUNKS_PER_DOCUMENT:
            raise ValueError("Invalid chunk count")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Duplicate chunk ID")
        for index, chunk in enumerate(chunks):
            if (chunk.document_id != document.document_id or chunk.chunk_index != index
                    or chunk.chunking_version != document.chunking_version
                    or chunk.text_hash != hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()):
                raise ValueError("Inconsistent chunk metadata")
            if (chunk.source_block_index is None or chunk.start_char is None
                    or chunk.end_char - chunk.start_char != len(chunk.text)):
                raise ValueError("Missing or inconsistent chunk source span")
            if index:
                previous = chunks[index - 1]
                if (chunk.source_block_index < previous.source_block_index
                        or (chunk.source_block_index == previous.source_block_index
                            and (chunk.start_char <= previous.start_char or chunk.end_char <= previous.end_char))):
                    raise ValueError("Chunk source spans must advance in order")
            if document.source_type == "pdf":
                if chunk.page is None or document.page_count is None or chunk.page > document.page_count:
                    raise ValueError("Invalid PDF page location")
            elif (chunk.start_line is None or document.line_count is None
                  or chunk.end_line > document.line_count):
                raise ValueError("Invalid text line location")

    def save_parsed_document(
        self, knowledge_base_id: str, parsed: ParsedDocument,
        build_chunks: Callable[[ParsedDocument, str], tuple[ChunkRecord, ...]],
    ) -> IngestionResult:
        """Backend transaction: allocate ID, build chunks, publish files, then manifest.

        A crash before manifest publication can leave an unreferenced directory;
        readers never treat that as a committed document. No automatic overwrites
        or cleanup of pre-existing directories are performed.
        """
        with _WRITE_LOCK:
            session_id, session_dir = self._session()
            manifest = self._load(session_id, session_dir, knowledge_base_id)
            existing = next((doc for doc in manifest.documents if doc.content_hash == parsed.content_hash), None)
            if existing and existing.ingestion_status in {"chunked", "indexed"}:
                if existing.chunking_version != config.CHUNKING_VERSION:
                    raise ValueError("分块版本已变化，需要显式重建。")
                return self.get_document_ingestion(knowledge_base_id, existing.document_id).model_copy(update={"duplicate": True})
            document = existing or DocumentRecord(
                document_name=parsed.document_name, source_type=parsed.source_type,
                content_hash=parsed.content_hash, size_bytes=parsed.size_bytes,
            )
            if document.source_type != parsed.source_type or document.size_bytes != parsed.size_bytes:
                raise ValueError("Existing document metadata does not match input")
            chunks = build_chunks(parsed, document.document_id)
            blocks = {block.block_index: block for block in parsed.blocks}
            for chunk in chunks:
                block = blocks.get(chunk.source_block_index)
                if (block is None or chunk.start_char is None
                        or block.text[chunk.start_char:chunk.end_char] != chunk.text
                        or chunk.page != block.page or chunk.heading != block.heading):
                    raise ValueError("Chunk source span does not match the parsed document")
            document = DocumentRecord.model_validate({
                **document.model_dump(), "page_count": parsed.page_count, "line_count": parsed.line_count,
                "ingestion_status": "chunked", "chunking_version": config.CHUNKING_VERSION,
                "chunk_count": len(chunks),
            })
            self._validate_chunks(document, chunks)
            documents = tuple(document if doc.document_id == document.document_id else doc for doc in manifest.documents)
            if existing is None:
                documents += (document,)
            updated = KnowledgeBaseManifest.model_validate({
                **manifest.model_dump(), "documents": documents,
                "index_status": "chunked" if all(doc.ingestion_status == "chunked" for doc in documents) else "pending",
            })
            kb_dir = self._kb_path(session_dir, knowledge_base_id)
            parent = self._checked(kb_dir / "documents", session_dir)
            destination = self._checked(parent / document.document_id, session_dir)
            if destination.exists():
                raise LocalRAGStorageError("存在未提交的文档目录，请先检查，未覆盖已有数据。")
            temporary = self._checked(parent / f".ingestion-{uuid4().hex}", session_dir)
            metadata = json.dumps(document.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            serialized = "".join(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n" for chunk in chunks)
            if len(serialized.encode("utf-8")) > config.MAX_CHUNKS_FILE_BYTES:
                raise LocalRAGStorageError("Chunk 文件超过大小限制。")
            created, published, committed = False, False, False
            try:
                self._checked(temporary, session_dir).mkdir()
                created = True
                for name, content in (("metadata.json", metadata), ("chunks.jsonl", serialized)):
                    path = self._checked(temporary / name, session_dir)
                    with path.open("x", encoding="utf-8", newline="\n") as stream:
                        stream.write(content)
                        stream.flush()
                        os.fsync(stream.fileno())
                self._checked(destination, session_dir)
                os.replace(self._checked(temporary, session_dir), destination)
                published = True
                self._write(updated, session_dir)
                committed = True
            except OSError:
                raise LocalRAGStorageError("文档入库保存失败，未提交完整入库状态。") from None
            finally:
                if created and not committed:
                    owned = destination if published else temporary
                    # Only fixed files created by this transaction; no recursive deletion.
                    for name in ("metadata.json", "chunks.jsonl"):
                        self._checked(owned / name, session_dir).unlink(missing_ok=True)
                    self._checked(owned, session_dir).rmdir()
            return IngestionResult(document=document, chunks=chunks)
