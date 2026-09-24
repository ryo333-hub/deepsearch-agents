"""Shared E5 input preparation and exact token budgeting, with no encoder IO.

The standalone loader uses the same pinned tokenizer.json as SentenceTransformer.
Only the small Rust tokenizers package is imported lazily; no torch/transformers
import, downloading, truncation, padding, or model fallback is involved.
"""

from functools import lru_cache
import json
from threading import RLock

from app.local_rag.config import (
    EMBEDDING_DOCUMENT_PREFIX, EMBEDDING_QUERY_PREFIX, EMBEDDING_MAX_INPUT_TOKENS,
    MAX_CHUNK_TEXT_LENGTH, EmbeddingConfig,
)


class EmbeddingError(ValueError):
    """Predictable local failure; no online/model/random-vector fallback."""


class EmbeddingDependencyError(EmbeddingError):
    pass


class EmbeddingLoadError(EmbeddingError):
    pass


class EmbeddingInputError(EmbeddingError):
    pass


class EmbeddingInputTooLong(EmbeddingInputError):
    def __init__(self, input_id: str, token_count: int, max_input_tokens: int):
        self.input_id = input_id
        self.token_count = token_count
        self.max_input_tokens = max_input_tokens
        super().__init__(f"{input_id}: token_count={token_count} exceeds "
                         f"max_input_tokens={max_input_tokens}; rechunk explicitly before embedding")


def prepare_text(text: str, input_id: str, *, query: bool = False) -> str:
    if not isinstance(text, str) or not text.strip():
        raise EmbeddingInputError(f"{input_id}: text must be a nonempty string")
    if len(text) > MAX_CHUNK_TEXT_LENGTH:
        raise EmbeddingInputError(f"{input_id}: exceeds the input character safety limit")
    return (EMBEDDING_QUERY_PREFIX if query else EMBEDDING_DOCUMENT_PREFIX) + text.strip()


def count_tokens(tokenizer, prepared: str, input_id: str, *, special_tokens: bool = True) -> int:
    try:
        ids = tokenizer(
            prepared, add_special_tokens=special_tokens, truncation=False, padding=False,
            return_attention_mask=False, return_token_type_ids=False, verbose=False,
        )["input_ids"]
        if not isinstance(ids, list) or any(type(t) is not int for t in ids):
            raise TypeError("Invalid tokenizer result")
        if prepared and not ids:
            raise ValueError("Tokenizer dropped nonempty input")
    except Exception as exc:
        raise EmbeddingInputError(f"{input_id}: tokenizer failed ({type(exc).__name__})") from None
    return len(ids)


def validate_prepared(tokenizer, prepared: str, input_id: str) -> int:
    count = count_tokens(tokenizer, prepared, input_id)
    if count > EMBEDDING_MAX_INPUT_TOKENS:
        raise EmbeddingInputTooLong(input_id, count, EMBEDDING_MAX_INPUT_TOKENS)
    return count


class _LocalTokenizer:
    model_max_length = EMBEDDING_MAX_INPUT_TOKENS

    def __init__(self, backend):
        self.backend = backend

    def __call__(self, text, *, add_special_tokens, truncation, padding, **_):
        if truncation or padding:
            raise ValueError("Length checks cannot truncate or pad")
        return {"input_ids": self.backend.encode(text, add_special_tokens=add_special_tokens).ids}


def _load_tokenizer(config: EmbeddingConfig):
    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise EmbeddingDependencyError("The project tokenizers dependency is required") from None
    try:
        settings = json.loads((config.model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
        if settings.get("model_max_length") != EMBEDDING_MAX_INPUT_TOKENS:
            raise EmbeddingLoadError("Local tokenizer limit differs from the pinned model")
        backend = Tokenizer.from_file(str(config.model_dir / "tokenizer.json"))
        backend.no_truncation()
        backend.no_padding()
        return _LocalTokenizer(backend)
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingLoadError(
            f"Pinned local tokenizer unavailable ({type(exc).__name__}); automatic download is disabled"
        ) from None


class DocumentTokenBudget:
    """Lazy tokenizer-only budget, also injectable with an encoder's tokenizer.

    effective_document_token_budget is a nominal allowance: max minus the
    measured empty-prefix/special-token overhead. Subword boundaries can change
    with the body, so every candidate is STILL checked as a complete input.
    """

    embedding_max_tokens = EMBEDDING_MAX_INPUT_TOKENS

    def __init__(self, config: EmbeddingConfig | None = None, *, tokenizer=None):
        self.config = config or EmbeddingConfig()
        self._tokenizer = tokenizer
        self._lock = RLock()

    @property
    def tokenizer(self):
        with self._lock:
            if self._tokenizer is None:
                self._tokenizer = _load_tokenizer(self.config)
            if (not callable(self._tokenizer)
                    or self._tokenizer.model_max_length != EMBEDDING_MAX_INPUT_TOKENS):
                raise EmbeddingLoadError("Model tokenizer is unavailable or has an incompatible limit")
            return self._tokenizer

    @property
    def effective_document_token_budget(self) -> int:
        return self.embedding_max_tokens - count_tokens(self.tokenizer, EMBEDDING_DOCUMENT_PREFIX, "prefix")

    def document_tokens(self, text: str, input_id: str = "chunk candidate") -> int:
        return count_tokens(self.tokenizer, prepare_text(text, input_id), input_id)

    def validate_document(self, text: str, input_id: str = "chunk candidate") -> int:
        return validate_prepared(self.tokenizer, prepare_text(text, input_id), input_id)

    def overlap_tokens(self, text: str) -> int:
        # Overlap measures source context only; no query/document wrappers.
        return count_tokens(self.tokenizer, text, "overlap", special_tokens=False)


@lru_cache(maxsize=1)
def get_document_token_budget() -> DocumentTokenBudget:
    return DocumentTokenBudget()
