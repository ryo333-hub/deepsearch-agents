"""Backend-owned storage configuration. Importing this module creates nothing."""

from pathlib import Path
from dataclasses import dataclass
import hashlib
import json


LOCAL_RAG_DATA_ROOT = Path(__file__).resolve().parents[2] / ".data" / "local_rag"
MAX_KNOWLEDGE_BASE_NAME_LENGTH = 120
MAX_EVIDENCE_TEXT_LENGTH = 8000
MAX_CHUNK_TEXT_LENGTH = 32000
MAX_MANIFEST_BYTES = 4 * 1024 * 1024

# Existing upload endpoint writes to app/updated/session_<thread_id>.
LOCAL_RAG_UPLOAD_ROOT = Path(__file__).resolve().parents[1] / "updated"
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 100
MAX_EXTRACTED_CHARS = 500_000
MAX_CHUNK_CHARS = 1000
CHUNK_OVERLAP_TOKENS = 64  # Context target (one eighth of the 512-token window), not mandatory padding.
MAX_CHUNKS_PER_DOCUMENT = 2000
MAX_CHUNKS_FILE_BYTES = 16 * 1024 * 1024
MAX_VECTOR_INDEX_BYTES = 256 * 1024 * 1024
MAX_VECTOR_INDEX_RECORDS = 100_000
MAX_VECTOR_INDEX_METADATA_BYTES = 32 * 1024 * 1024

# One explicitly prepared local snapshot. Loading never downloads from the Hub.
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
EMBEDDING_DIMENSION = 384
EMBEDDING_MAX_INPUT_TOKENS = 512
EMBEDDING_DOCUMENT_PREFIX = "passage: "
EMBEDDING_QUERY_PREFIX = "query: "

# Persisted in both document metadata and every chunk. Length-affecting settings
# are fingerprinted; changing any of them requires explicit re-ingestion.
CHUNKING_VERSION = "local-rag-chunk-v2-" + hashlib.sha256(json.dumps({
    "algorithm": "structure-first-token-budget-v2",
    "model": EMBEDDING_MODEL_NAME, "revision": EMBEDDING_MODEL_REVISION,
    "max_tokens": EMBEDDING_MAX_INPUT_TOKENS, "prefix": EMBEDDING_DOCUMENT_PREFIX,
    "special_tokens": True, "outer_whitespace": "strip",
    "max_chars": MAX_CHUNK_CHARS, "overlap_tokens": CHUNK_OVERLAP_TOKENS,
}, sort_keys=True).encode()).hexdigest()[:16]
VECTOR_INDEX_VERSION = 1


@dataclass(frozen=True)
class EmbeddingConfig:
    """Backend-owned configuration, never populated with a model-supplied path.

    model_dir must hold the explicitly provisioned revision above. Different
    models need their own reviewed preprocessing contract, not a fallback.
    """

    model_dir: Path = (LOCAL_RAG_DATA_ROOT / "models" / "multilingual-e5-small"
                       / EMBEDDING_MODEL_REVISION)
    batch_size: int = 4
    normalize_embeddings: bool = True

    def __post_init__(self):
        if not isinstance(self.model_dir, Path) or not self.model_dir.is_absolute():
            raise ValueError("model_dir must be an absolute backend-owned Path")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 32:
            raise ValueError("embedding batch_size must be between 1 and 32")
        if type(self.normalize_embeddings) is not bool:
            raise ValueError("normalize_embeddings must be a bool")
