"""CPU-only, offline E5 embeddings for already-authorized persisted chunks.

Import/construction performs no model IO. Storage remains responsible for
session/KB authorization; this adapter never resolves paths from chunk IDs,
writes vectors, changes ingestion status, or contacts an embedding API.
Keep one adapter (or use get_embedding_adapter) to reuse its lazy model.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from math import isclose, isfinite, sqrt
from numbers import Real
from threading import RLock
from typing import Any

from app.local_rag.config import (
    EMBEDDING_DIMENSION, EMBEDDING_DOCUMENT_PREFIX, EMBEDDING_MAX_INPUT_TOKENS,
    EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_REVISION, EMBEDDING_QUERY_PREFIX,
    EmbeddingConfig,
)
from app.local_rag.schemas import ChunkEmbedding, ChunkRecord, EmbeddingVector
from app.local_rag.tokenization import (
    EmbeddingError, EmbeddingDependencyError, EmbeddingLoadError,
    EmbeddingInputError, EmbeddingInputTooLong, prepare_text, validate_prepared,
)


class EmbeddingOutputError(EmbeddingError):
    pass


@dataclass(frozen=True)
class EmbeddingMetadata:
    model_name: str
    model_revision: str
    embedding_dimension: int
    normalize_embeddings: bool
    batch_size: int
    max_input_tokens: int
    document_prefix: str = EMBEDDING_DOCUMENT_PREFIX
    query_prefix: str = EMBEDDING_QUERY_PREFIX
    device: str = "cpu"


def _load_local_model(config: EmbeddingConfig):
    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.models import Normalize, Pooling, Transformer
    except ImportError:
        raise EmbeddingDependencyError(
            "sentence-transformers and its CPU dependencies must be installed in the project environment"
        ) from None
    if not config.model_dir.is_dir():
        raise EmbeddingLoadError(
            f"Local snapshot missing for {EMBEDDING_MODEL_NAME}@{EMBEDDING_MODEL_REVISION}; "
            "prepare the pinned model explicitly (automatic download is disabled)"
        )
    try:
        model = SentenceTransformer(
            str(config.model_dir), device="cpu", local_files_only=True,
            trust_remote_code=False, tokenizer_kwargs={"use_fast": True},
        )
        # The pinned E5 snapshot ends in Normalize. Let the adapter's setting
        # control normalization for BOTH query and documents, including False.
        if (len(model) != 3 or not isinstance(model[0], Transformer)
                or not isinstance(model[1], Pooling) or not isinstance(model[2], Normalize)):
            raise EmbeddingLoadError("Unexpected E5 module layout in local snapshot")
        del model[2]
        if model[0].do_lower_case:
            raise EmbeddingLoadError("Unexpected tokenizer preprocessing in local snapshot")
        model.eval()
        return model
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingLoadError(f"Local E5 model loading failed ({type(exc).__name__})") from None


class LocalEmbeddingAdapter:
    """Small batches, exact token budgeting, no partial result on failure.

    model_factory is a trusted in-process testing seam, not a backend fallback.
    Metadata describes the pinned snapshot; ensure provisioning uses that commit.
    """

    def __init__(self, config: EmbeddingConfig | None = None, *,
                 model_factory: Callable[[EmbeddingConfig], Any] | None = None):
        self.config = config or EmbeddingConfig()
        self._model_factory = model_factory or _load_local_model
        self._model = None
        self._lock = RLock()

    @property
    def metadata(self) -> EmbeddingMetadata:
        # Reading metadata also never loads the model.
        return EmbeddingMetadata(
            EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_REVISION, EMBEDDING_DIMENSION,
            self.config.normalize_embeddings, self.config.batch_size, EMBEDDING_MAX_INPUT_TOKENS,
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self):
        if self._model is None:
            try:
                model = self._model_factory(self.config)
                if not callable(getattr(model, "tokenizer", None)):
                    raise EmbeddingLoadError("Model tokenizer is unavailable")
                if model.get_sentence_embedding_dimension() != EMBEDDING_DIMENSION:
                    raise EmbeddingLoadError("Local model dimension differs from the pinned model")
                # Reject rather than guess if the model/tokenizer contract changed.
                if (model.max_seq_length != EMBEDDING_MAX_INPUT_TOKENS
                        or model.tokenizer.model_max_length != EMBEDDING_MAX_INPUT_TOKENS):
                    raise EmbeddingLoadError("Local model/tokenizer input limit differs from the pinned model")
                self._model = model
            except EmbeddingError:
                raise
            except Exception as exc:
                raise EmbeddingLoadError(f"Model initialization failed ({type(exc).__name__})") from None
        return self._model

    def _check_tokens(self, model, text: str, input_id: str):
        return validate_prepared(model.tokenizer, text, input_id)

    def _vectors(self, raw, count: int) -> list[EmbeddingVector]:
        try:
            if len(raw) != count:
                raise EmbeddingOutputError(f"Expected {count} vectors, got {len(raw)}")
            result = []
            for row in raw:
                if len(row) != EMBEDDING_DIMENSION:
                    raise EmbeddingOutputError(f"Vector dimension must be {EMBEDDING_DIMENSION}")
                if any(isinstance(v, bool) or not isinstance(v, Real) or not isfinite(v) for v in row):
                    raise EmbeddingOutputError("Vector contains nonnumeric, NaN or Infinity values")
                vector = tuple(float(v) for v in row)
                norm = sqrt(sum(v * v for v in vector))
                if not isfinite(norm) or norm == 0:
                    raise EmbeddingOutputError("Vector has zero or nonfinite norm")
                if self.config.normalize_embeddings and not isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
                    raise EmbeddingOutputError("Model did not return a normalized vector")
                result.append(EmbeddingVector(vector=vector, dimension=EMBEDDING_DIMENSION))
            return result
        except EmbeddingError:
            raise
        except (TypeError, ValueError, OverflowError):
            raise EmbeddingOutputError("Malformed embedding output") from None

    def _encode(self, inputs: list[tuple[str, str]]) -> list[EmbeddingVector]:
        with self._lock:
            model = self._ensure_loaded()
            # Validate ALL inputs before the first encode, including later batches.
            for input_id, text in inputs:
                self._check_tokens(model, text, input_id)
            result = []
            for offset in range(0, len(inputs), self.config.batch_size):
                batch = [text for _, text in inputs[offset:offset + self.config.batch_size]]
                try:
                    raw = model.encode(
                        batch, batch_size=self.config.batch_size, device="cpu",
                        normalize_embeddings=self.config.normalize_embeddings,
                        convert_to_numpy=True, show_progress_bar=False, prompt="",
                    )
                except Exception as exc:
                    raise EmbeddingError(f"Local encoding failed ({type(exc).__name__})") from None
                result.extend(self._vectors(raw, len(batch)))
            return result

    def embed_documents(self, chunks: Sequence[ChunkRecord]) -> tuple[ChunkEmbedding, ...]:
        if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)):
            raise EmbeddingInputError("documents must be a sequence of ChunkRecord")
        inputs = []
        seen = set()
        for chunk in chunks:
            if not isinstance(chunk, ChunkRecord):
                raise EmbeddingInputError("documents must contain only ChunkRecord")
            try:
                # Also reject unchecked model_construct/model_copy payloads.
                chunk = ChunkRecord.model_validate(chunk.model_dump())
            except ValueError:
                raise EmbeddingInputError("Invalid ChunkRecord") from None
            text = prepare_text(chunk.text, chunk.chunk_id)
            if sha256(chunk.text.encode("utf-8")).hexdigest() != chunk.text_hash:
                raise EmbeddingInputError(f"{chunk.chunk_id}: text_hash mismatch")
            if chunk.chunk_id in seen:
                raise EmbeddingInputError(f"{chunk.chunk_id}: duplicate chunk_id")
            seen.add(chunk.chunk_id)
            inputs.append((chunk.chunk_id, text))
        if not inputs:
            return ()
        vectors = self._encode(inputs)
        return tuple(ChunkEmbedding(chunk_id=input_id, **vector.model_dump())
                     for (input_id, _), vector in zip(inputs, vectors, strict=True))

    def embed_query(self, text: str) -> EmbeddingVector:
        prepared = prepare_text(text, "query", query=True)
        return self._encode([("query", prepared)])[0]


@lru_cache(maxsize=1)
def get_embedding_adapter() -> LocalEmbeddingAdapter:
    """Process-local reusable adapter; calling this still does not load weights."""
    return LocalEmbeddingAdapter()
