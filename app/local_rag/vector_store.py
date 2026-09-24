"""Session/KB-scoped NumPy vector persistence. Retrieval deliberately absent.

Each generation is immutable. The tiny current.json pointer is atomically
replaced only after both generation files have been flushed and validated, so a
failed build never makes a partial index current. All paths come from the same
LocalRAGStorage authorization and containment checks as source chunks.
"""

from dataclasses import dataclass
from collections.abc import Sequence
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
from threading import RLock
from uuid import uuid4
import warnings

import numpy as np

from app.local_rag import config
from app.local_rag.embeddings import EmbeddingMetadata
from app.local_rag.schemas import ChunkEmbedding, ChunkRecord, VectorIndexManifest
from app.local_rag.storage import (
    KnowledgeBaseAccessError, LocalRAGStorage, LocalRAGStorageError, _WRITE_LOCK,
)


_GENERATION_PATTERN = re.compile(r"gen_[0-9a-f]{32}")
_VECTOR_LOCK = RLock()


class VectorStoreError(RuntimeError):
    """Sanitized persistence/compatibility failure without backend paths."""


class VectorIndexNotFoundError(VectorStoreError):
    pass


class VectorIndexExistsError(VectorStoreError):
    pass


class VectorIndexCompatibilityError(VectorStoreError):
    pass


class VectorIndexCorruptError(VectorStoreError):
    pass


@dataclass(frozen=True)
class LoadedVectorIndex:
    metadata: VectorIndexManifest
    vectors: np.ndarray

    def __post_init__(self):
        self.vectors.flags.writeable = False


class LocalVectorStore:
    """Backend service; KB IDs are accepted, host paths never are."""

    def __init__(self, storage: LocalRAGStorage | None = None):
        self.storage = storage or LocalRAGStorage()

    def _context(self, knowledge_base_id: str):
        session_id, session_dir = self.storage._session()
        manifest = self.storage._load(session_id, session_dir, knowledge_base_id)
        kb_dir = self.storage._kb_path(session_dir, knowledge_base_id)
        root = self.storage._checked(kb_dir / "vector_index", session_dir)
        return session_id, session_dir, manifest, root

    def _source_chunks(self, manifest) -> tuple[ChunkRecord, ...]:
        chunks = []
        for document in manifest.documents:
            if document.ingestion_status not in {"chunked", "indexed"}:
                raise VectorStoreError("知识库包含未完成分块的文档，无法构建索引")
            if document.chunking_version != config.CHUNKING_VERSION:
                raise VectorIndexCompatibilityError("Chunk 版本与当前运行配置不一致")
            loaded = self.storage.get_document_ingestion(
                manifest.knowledge_base_id, document.document_id,
            )
            chunks.extend(loaded.chunks)
            if len(chunks) > config.MAX_VECTOR_INDEX_RECORDS:
                raise VectorStoreError("向量索引记录数量超过限制")
        if not chunks:
            raise VectorStoreError("空知识库不能构建向量索引")
        return tuple(chunks)

    @staticmethod
    def _source_digest(chunks) -> str:
        # Bind vectors to the complete ordered source records, not just IDs.
        # Only this digest is persisted; source text is never duplicated.
        digest = sha256()
        for chunk in chunks:
            digest.update(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False,
                                     sort_keys=True, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    @staticmethod
    def _validate_runtime(metadata: VectorIndexManifest, runtime: EmbeddingMetadata,
                          session_id: str, knowledge_base_id: str) -> None:
        expected = {
            "knowledge_base_id": knowledge_base_id,
            "session_id": session_id,
            "embedding_model_name": runtime.model_name,
            "embedding_model_revision": runtime.model_revision,
            "embedding_dimension": runtime.embedding_dimension,
            "normalize_embeddings": runtime.normalize_embeddings,
            "chunk_version": config.CHUNKING_VERSION,
            "dtype": "float32",
        }
        for field, value in expected.items():
            if getattr(metadata, field) != value:
                raise VectorIndexCompatibilityError(f"Vector index {field} is incompatible")

    @staticmethod
    def _matrix(embeddings, chunks, runtime: EmbeddingMetadata) -> np.ndarray:
        if not isinstance(embeddings, Sequence) or isinstance(embeddings, (str, bytes)):
            raise VectorStoreError("embeddings 必须是有限序列") from None
        if len(embeddings) != len(chunks):
            raise VectorStoreError("Chunk 与向量数量不一致")
        rows = tuple(embeddings)
        if len(rows) != len(chunks):
            raise VectorStoreError("Chunk 与向量数量不一致")
        if [row.chunk_id for row in rows if isinstance(row, ChunkEmbedding)] != [c.chunk_id for c in chunks]:
            raise VectorStoreError("Chunk ID 与向量顺序不一致")
        if any(not isinstance(row, ChunkEmbedding) or row.dimension != runtime.embedding_dimension for row in rows):
            raise VectorStoreError("向量契约或维度不一致")
        if len({row.chunk_id for row in rows}) != len(rows):
            raise VectorStoreError("向量索引包含重复 Chunk ID")
        if (type(runtime.embedding_dimension) is not int or runtime.embedding_dimension <= 0
                or type(runtime.normalize_embeddings) is not bool
                or len(chunks) * runtime.embedding_dimension * 4 + 128 > config.MAX_VECTOR_INDEX_BYTES):
            raise VectorStoreError("向量维度、归一化配置或文件大小超过限制")
        try:
            rows = tuple(ChunkEmbedding.model_validate(row.model_dump(warnings=False)) for row in rows)
        except ValueError:
            raise VectorStoreError("向量契约无效，可能包含非数值、NaN 或 Infinity") from None
        try:
            with np.errstate(over="raise", invalid="raise"):
                matrix = np.asarray([row.vector for row in rows], dtype="<f4")
        except (TypeError, ValueError, OverflowError, FloatingPointError):
            raise VectorStoreError("向量无法转换为 float32 矩阵") from None
        if matrix.shape != (len(chunks), runtime.embedding_dimension):
            raise VectorStoreError("向量矩阵 shape 不正确")
        if not np.isfinite(matrix).all():
            raise VectorStoreError("向量包含 NaN 或 Infinity")
        norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
        if (not np.isfinite(norms).all() or np.any(norms == 0)
                or (runtime.normalize_embeddings
                    and not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4))):
            raise VectorStoreError("向量范数与 Embedding 配置不一致")
        return np.ascontiguousarray(matrix)

    def build_index(self, knowledge_base_id: str, embeddings,
                    runtime: EmbeddingMetadata, *, rebuild: bool = False) -> LoadedVectorIndex:
        if type(rebuild) is not bool or not isinstance(runtime, EmbeddingMetadata):
            raise VectorStoreError("无效的 Vector Store 构建参数")
        with _WRITE_LOCK, _VECTOR_LOCK:
            session_id, session_dir, source, root = self._context(knowledge_base_id)
            current = self.storage._checked(root / "current.json", session_dir)
            if current.exists() and not rebuild:
                raise VectorIndexExistsError("向量索引已存在；需要显式 rebuild")
            chunks = self._source_chunks(source)
            matrix = self._matrix(embeddings, chunks, runtime)
            generation_name = "gen_" + uuid4().hex
            generations = self.storage._checked(root / "generations", session_dir)
            temporary = self.storage._checked(root / (".generation-" + uuid4().hex), session_dir)
            generation = self.storage._checked(generations / generation_name, session_dir)
            created, published, pointer_committed = False, False, False
            try:
                root.mkdir(exist_ok=True)
                self.storage._checked(root, session_dir)
                generations.mkdir(exist_ok=True)
                temporary.mkdir()
                created = True
                vectors_path = self.storage._checked(temporary / "vectors.npy", session_dir)
                with vectors_path.open("xb") as stream:
                    np.save(stream, matrix, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                vector_bytes = vectors_path.read_bytes()
                if len(vector_bytes) > config.MAX_VECTOR_INDEX_BYTES:
                    raise VectorStoreError("向量文件超过大小限制")
                index = VectorIndexManifest(
                    index_version=config.VECTOR_INDEX_VERSION, dtype="float32",
                    knowledge_base_id=knowledge_base_id, session_id=session_id,
                    embedding_model_name=runtime.model_name,
                    embedding_model_revision=runtime.model_revision,
                    embedding_dimension=runtime.embedding_dimension,
                    normalize_embeddings=runtime.normalize_embeddings,
                    chunk_version=config.CHUNKING_VERSION,
                    vector_count=len(chunks),
                    chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
                    document_ids=tuple(chunk.document_id for chunk in chunks),
                    vectors_sha256=sha256(vector_bytes).hexdigest(),
                    source_chunks_sha256=self._source_digest(chunks),
                    created_at=datetime.now(timezone.utc),
                )
                self._write_json(temporary / "index.json", index.model_dump(mode="json"),
                                 session_dir, config.MAX_VECTOR_INDEX_METADATA_BYTES)
                persisted = self._read_json(
                    temporary / "index.json", session_dir,
                    config.MAX_VECTOR_INDEX_METADATA_BYTES,
                    VectorIndexCorruptError("Vector index metadata 缺失或损坏"),
                )
                if VectorIndexManifest.model_validate(persisted) != index:
                    raise VectorIndexCorruptError("Vector index metadata 写入校验失败")
                self._load_generation(temporary, session_dir, index, matrix_expected=matrix)
                os.replace(temporary, generation)
                published = True
                pointer_tmp = self.storage._checked(root / (".current-" + uuid4().hex + ".tmp"), session_dir)
                try:
                    self._write_json(pointer_tmp, {"pointer_version": 1, "generation": generation_name},
                                     session_dir, 1024)
                    os.replace(pointer_tmp, current)
                    pointer_committed = True
                finally:
                    self.storage._checked(pointer_tmp, session_dir).unlink(missing_ok=True)
                loaded = self._load_current(session_id, session_dir, source, root, runtime)
                # current.json is the commit point. The KB status is only a
                # summary; failing to update it must not report a rolled-back
                # build when a complete, validated new index is already current.
                try:
                    self._set_index_status(source, session_dir, "indexed")
                except (LocalRAGStorageError, OSError):
                    warnings.warn("Vector index committed; KB status summary update failed",
                                  RuntimeWarning, stacklevel=2)
                self._cleanup_other_generations(root, session_dir, generation_name)
                return loaded
            except VectorStoreError:
                raise
            except (OSError, ValueError) as exc:
                raise VectorStoreError(f"向量索引保存失败 ({type(exc).__name__})") from None
            finally:
                if created and not pointer_committed:
                    owned = generation if published else temporary
                    self._remove_generation(owned, session_dir, missing_ok=True)

    def load_index(self, knowledge_base_id: str,
                   runtime: EmbeddingMetadata) -> LoadedVectorIndex:
        if not isinstance(runtime, EmbeddingMetadata):
            raise VectorStoreError("无效的 Embedding runtime metadata")
        with _WRITE_LOCK, _VECTOR_LOCK:
            session_id, session_dir, source, root = self._context(knowledge_base_id)
            return self._load_current(session_id, session_dir, source, root, runtime)

    def _load_current(self, session_id, session_dir, source, root,
                      runtime: EmbeddingMetadata) -> LoadedVectorIndex:
        pointer = self._read_json(root / "current.json", session_dir, 1024,
                                  VectorIndexNotFoundError("向量索引不存在"))
        if (not isinstance(pointer, dict) or set(pointer) != {"pointer_version", "generation"}
                or type(pointer["pointer_version"]) is not int or pointer["pointer_version"] != 1
                or not isinstance(pointer["generation"], str)
                or not _GENERATION_PATTERN.fullmatch(pointer["generation"])):
            raise VectorIndexCorruptError("Vector Store current pointer 已损坏")
        generation = self.storage._checked(root / "generations" / pointer["generation"], session_dir)
        metadata_data = self._read_json(generation / "index.json", session_dir,
                                        config.MAX_VECTOR_INDEX_METADATA_BYTES,
                                        VectorIndexCorruptError("Vector index metadata 缺失或损坏"))
        try:
            metadata = VectorIndexManifest.model_validate(metadata_data)
        except ValueError:
            raise VectorIndexCorruptError("Vector index metadata 不完整或无效") from None
        self._validate_runtime(metadata, runtime, session_id, source.knowledge_base_id)
        chunks = self._source_chunks(source)
        if (metadata.chunk_ids != tuple(chunk.chunk_id for chunk in chunks)
                or metadata.document_ids != tuple(chunk.document_id for chunk in chunks)
                or metadata.source_chunks_sha256 != self._source_digest(chunks)):
            raise VectorIndexCompatibilityError("Vector index 与当前持久化 Chunk 不一致")
        return self._load_generation(generation, session_dir, metadata)

    def _load_generation(self, generation: Path, session_dir: Path,
                         metadata: VectorIndexManifest, matrix_expected=None) -> LoadedVectorIndex:
        path = self.storage._checked(generation / "vectors.npy", session_dir)
        try:
            info = path.stat()
            if (not path.is_file() or info.st_nlink != 1
                    or not 0 < info.st_size <= config.MAX_VECTOR_INDEX_BYTES):
                raise ValueError
            payload = path.read_bytes()
            if sha256(payload).hexdigest() != metadata.vectors_sha256:
                raise ValueError
            # Validate the bounded header and exact body size BEFORE NumPy can
            # allocate from a corrupt header's declared shape. Decode the same
            # bytes whose checksum was validated, not a second filesystem read.
            stream = BytesIO(payload)
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError
            if (shape != (metadata.vector_count, metadata.embedding_dimension)
                    or dtype != np.dtype("<f4") or fortran
                    or len(payload) - stream.tell() != metadata.vector_count * metadata.embedding_dimension * 4):
                raise ValueError
            stream.seek(0)
            vectors = np.load(stream, allow_pickle=False)
            if (vectors.dtype != np.dtype("float32")
                    or vectors.shape != (metadata.vector_count, metadata.embedding_dimension)
                    or not vectors.flags.c_contiguous or not np.isfinite(vectors).all()):
                raise ValueError
            if matrix_expected is not None and not np.array_equal(vectors, matrix_expected):
                raise ValueError
            norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
            if (not np.isfinite(norms).all() or np.any(norms == 0)
                    or (metadata.normalize_embeddings
                        and not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4))):
                raise ValueError
            return LoadedVectorIndex(metadata, vectors)
        except (OSError, ValueError, TypeError, EOFError):
            raise VectorIndexCorruptError("vectors.npy 缺失、损坏或与 metadata 不一致") from None

    def delete_index(self, knowledge_base_id: str) -> bool:
        with _WRITE_LOCK, _VECTOR_LOCK:
            _, session_dir, source, root = self._context(knowledge_base_id)
            if not root.exists():
                return False
            self.storage._checked(root, session_dir)
            allowed = {"current.json", "generations"}
            if any(entry.name not in allowed for entry in root.iterdir()):
                raise VectorStoreError("Vector Store 目录包含未知内容，拒绝删除")
            generations = self.storage._checked(root / "generations", session_dir)
            targets = []
            if generations.exists():
                for generation in list(generations.iterdir()):
                    if not _GENERATION_PATTERN.fullmatch(generation.name):
                        raise VectorStoreError("Vector Store generations 包含未知内容，拒绝删除")
                    self._generation_files(generation, session_dir)
                    targets.append(generation)
            current = self.storage._checked(root / "current.json", session_dir)
            if current.exists() and (not current.is_file() or current.stat().st_nlink != 1):
                raise VectorStoreError("Vector Store current 文件不安全，拒绝删除")
            # All targets are checked before the first mutation. Remove the
            # pointer before data so interruption exposes absence, not partial data.
            if source.index_status == "indexed":
                self._set_index_status(source, session_dir, "chunked")
            current.unlink(missing_ok=True)
            for generation in targets:
                self._remove_generation(generation, session_dir)
            if generations.exists():
                generations.rmdir()
            root.rmdir()
            return True

    def _set_index_status(self, manifest, session_dir, status):
        if manifest.index_status == status:
            return
        updated = type(manifest).model_validate({**manifest.model_dump(), "index_status": status})
        self.storage._write(updated, session_dir)

    def _cleanup_other_generations(self, root, session_dir, keep):
        generations = self.storage._checked(root / "generations", session_dir)
        for entry in list(generations.iterdir()):
            if entry.name != keep and _GENERATION_PATTERN.fullmatch(entry.name):
                try:
                    self._remove_generation(entry, session_dir)
                except (OSError, VectorStoreError, KnowledgeBaseAccessError):
                    pass  # Current generation is already committed and valid.

    def _generation_files(self, path, session_dir):
        path = self.storage._checked(path, session_dir)
        if not path.is_dir() or path.is_symlink() or path.is_junction():
            raise VectorStoreError("Vector generation 不是安全目录")
        entries = {entry.name for entry in path.iterdir()}
        if not entries <= {"index.json", "vectors.npy"}:
            raise VectorStoreError("Vector generation 包含未知内容，拒绝删除")
        targets = []
        for name in ("index.json", "vectors.npy"):
            target = self.storage._checked(path / name, session_dir)
            if target.exists() and (not target.is_file() or target.stat().st_nlink != 1):
                raise VectorStoreError("Vector generation 文件链接数异常")
            targets.append(target)
        return targets

    def _remove_generation(self, path, session_dir, missing_ok=False):
        path = self.storage._checked(path, session_dir)
        if not path.exists() and missing_ok:
            return
        for target in self._generation_files(path, session_dir):
            target.unlink(missing_ok=True)
        path.rmdir()

    def _write_json(self, path, value, session_dir, maximum):
        path = self.storage._checked(path, session_dir)
        payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
        if len(payload) > maximum:
            raise VectorStoreError("Vector Store metadata 超过大小限制")
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    def _read_json(self, path, session_dir, maximum, missing_error):
        path = self.storage._checked(path, session_dir)
        try:
            info = path.stat()
            if not path.is_file() or info.st_nlink != 1 or not 0 < info.st_size <= maximum:
                raise ValueError
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise missing_error from None
        except (OSError, UnicodeError, ValueError):
            raise VectorIndexCorruptError("Vector Store JSON 文件已损坏") from None
