"""Versioned local RAG metadata. IDs are opaque; no host paths or credentials.

Models accept persisted IDs for deserialization. Public storage creation APIs
generate IDs themselves and do not accept caller-supplied identity or paths.
Evidence text is bounded, not automatically redacted: ingestion must later
decide what document content is permitted to leave the local machine.
"""

from datetime import datetime, timezone
from pathlib import PureWindowsPath
from typing import Annotated, Literal, Self
from uuid import uuid4
import unicodedata

from pydantic import (
    AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field,
    StringConstraints, field_validator, model_validator,
)

from app.local_rag.config import (
    MAX_CHUNK_TEXT_LENGTH, MAX_EVIDENCE_TEXT_LENGTH,
    MAX_KNOWLEDGE_BASE_NAME_LENGTH,
)
from app.utils.path_utils import validate_thread_id, validate_upload_filename


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


KnowledgeBaseId = Annotated[str, StringConstraints(strict=True, pattern=r"^kb_[0-9a-f]{32}$")]
DocumentId = Annotated[str, StringConstraints(strict=True, pattern=r"^doc_[0-9a-f]{32}$")]
ChunkId = Annotated[str, StringConstraints(strict=True, pattern=r"^chunk_[0-9a-f]{32}$")]
ContentHash = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]
SessionId = Annotated[str, StringConstraints(strict=True), AfterValidator(validate_thread_id)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]


def _display_name(value: str) -> str:
    if not value.strip() or any(unicodedata.category(c).startswith("C") for c in value):
        raise ValueError("Display name must be nonempty and contain no control characters")
    return value


KnowledgeBaseName = Annotated[
    str, StringConstraints(strict=True, min_length=1, max_length=MAX_KNOWLEDGE_BASE_NAME_LENGTH),
    AfterValidator(_display_name),
]


def _document_name(value: str) -> str:
    validate_upload_filename(value)
    _display_name(value)
    if value.casefold() == ".env" or value.casefold().startswith(".env."):
        raise ValueError("Environment files cannot be document sources")
    return value


DocumentName = Annotated[
    str, StringConstraints(strict=True, min_length=1, max_length=255), AfterValidator(_document_name),
]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class _Location(_Record):
    page: PositiveInt | None = None
    start_line: PositiveInt | None = None
    end_line: PositiveInt | None = None
    heading: Annotated[str, StringConstraints(strict=True, max_length=300)] | None = None

    @field_validator("heading")
    @classmethod
    def check_heading(cls, value: str | None) -> str | None:
        if value is not None:
            _display_name(value)
            if (PureWindowsPath(value).drive or value.startswith(("/", "\\"))
                    or "../" in value or "..\\" in value or value.casefold() == ".env"):
                raise ValueError("Heading must be a document heading, not a host path")
        return value

    @model_validator(mode="after")
    def check_line_range(self) -> Self:
        if (self.start_line is None) != (self.end_line is None):
            raise ValueError("Both start_line and end_line are required for a line range")
        if self.start_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line must not precede start_line")
        return self


class DocumentRecord(_Record):
    document_id: DocumentId = Field(default_factory=lambda: "doc_" + uuid4().hex)
    document_name: DocumentName
    source_type: Literal["pdf", "txt", "markdown"]
    content_hash: ContentHash
    size_bytes: NonNegativeInt
    created_at: AwareDatetime = Field(default_factory=_utc_now)
    page_count: PositiveInt | None = None
    line_count: NonNegativeInt | None = None
    ingestion_status: Literal["pending", "parsed", "chunked", "indexed", "failed"] = "pending"
    chunking_version: str | None = None
    chunk_count: NonNegativeInt | None = None


class _ChunkLocation(_Location):
    # Half-open character range in ParsedDocument.blocks[source_block_index].text
    # after loader newline normalization, BEFORE trimming/splitting. Optional only
    # so old records remain parseable for an explicit version/rebuild decision.
    source_block_index: NonNegativeInt | None = None
    start_char: NonNegativeInt | None = None
    end_char: PositiveInt | None = None

    @model_validator(mode="after")
    def check_source_span(self) -> Self:
        fields = (self.source_block_index, self.start_char, self.end_char)
        if any(x is not None for x in fields):
            if any(x is None for x in fields) or self.end_char <= self.start_char:
                raise ValueError("A complete, nonempty source span is required")
        return self


class ChunkRecord(_ChunkLocation):
    chunk_id: ChunkId = Field(default_factory=lambda: "chunk_" + uuid4().hex)
    document_id: DocumentId
    chunk_index: NonNegativeInt
    text: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=MAX_CHUNK_TEXT_LENGTH)]
    text_hash: ContentHash
    chunking_version: str | None = None


class Citation(_ChunkLocation):
    citation_id: Annotated[str, StringConstraints(strict=True, pattern=r"^C[1-9][0-9]*$")]
    document_id: DocumentId
    document_name: DocumentName
    chunk_id: ChunkId
    knowledge_base_id: KnowledgeBaseId
    score: Annotated[float, Field(strict=True, allow_inf_nan=False)] | None = None
    score_type: Literal["cosine_similarity", "cosine_distance", "inner_product", "l2_distance"] | None = None


class Evidence(_Record):
    text: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=MAX_EVIDENCE_TEXT_LENGTH)]
    citation: Citation
    truncated: bool = False

    @field_validator("truncated", mode="before")
    @classmethod
    def check_truncated_type(cls, value):
        if type(value) is not bool:
            raise ValueError("truncated must be a bool")
        return value

    @model_validator(mode="after")
    def check_cited_text_span(self) -> Self:
        if self.citation.start_char is not None:
            if self.citation.end_char - self.citation.start_char != len(self.text):
                raise ValueError("Citation character span must match evidence text")
        elif self.truncated:
            raise ValueError("Truncated evidence requires an exact source character span")
        return self


class KnowledgeBaseManifest(_Record):
    knowledge_base_id: KnowledgeBaseId = Field(default_factory=lambda: "kb_" + uuid4().hex)
    name: KnowledgeBaseName
    session_id: SessionId
    created_at: AwareDatetime = Field(default_factory=_utc_now)
    documents: tuple[DocumentRecord, ...] = ()
    schema_version: Literal[1] = 1
    # Keep first-layer statuses readable; this ingestion layer never emits ready/indexed.
    index_status: Literal["empty", "pending", "parsed", "chunked", "indexed", "ready", "failed"] = "empty"

    @field_validator("schema_version", mode="before")
    @classmethod
    def check_schema_version(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("Unsupported manifest schema_version")
        return value

    @model_validator(mode="after")
    def check_documents(self) -> Self:
        if len({doc.document_id for doc in self.documents}) != len(self.documents):
            raise ValueError("Duplicate document ID")
        if len({doc.content_hash for doc in self.documents}) != len(self.documents):
            raise ValueError("Duplicate document content hash")
        return self


class KnowledgeBaseSummary(_Record):
    knowledge_base_id: KnowledgeBaseId
    name: KnowledgeBaseName
    created_at: AwareDatetime
    document_count: NonNegativeInt
    index_status: Literal["empty", "pending", "parsed", "chunked", "indexed", "ready", "failed"]


class TextBlock(_Location):
    """Internal extraction unit: original location, normalized newlines only."""

    block_index: NonNegativeInt
    text: str


class ParsedDocument(_Record):
    document_name: DocumentName
    source_type: Literal["pdf", "txt", "markdown"]
    content_hash: ContentHash
    size_bytes: NonNegativeInt
    blocks: tuple[TextBlock, ...]
    page_count: PositiveInt | None = None
    line_count: NonNegativeInt | None = None
    empty_pages: tuple[PositiveInt, ...] = ()


class IngestionResult(_Record):
    document: DocumentRecord
    chunks: tuple[ChunkRecord, ...]
    duplicate: bool = False


class EmbeddingVector(_Record):
    """Finite, immutable vector; this is not a persisted vector-store record."""

    vector: tuple[Annotated[float, Field(strict=True, allow_inf_nan=False)], ...]
    dimension: PositiveInt

    @model_validator(mode="after")
    def check_dimension(self) -> Self:
        if len(self.vector) != self.dimension:
            raise ValueError("Vector length must equal dimension")
        return self


class ChunkEmbedding(EmbeddingVector):
    chunk_id: ChunkId


class RetrievalHit(_Record):
    """Query-local ranked hit. Higher score means greater vector similarity."""

    chunk_id: ChunkId
    document_id: DocumentId
    rank: PositiveInt
    score: Annotated[float, Field(strict=True, allow_inf_nan=False)]
    text: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=MAX_CHUNK_TEXT_LENGTH)]
    citation: Citation

    @model_validator(mode="after")
    def check_citation(self) -> Self:
        if (self.citation.chunk_id != self.chunk_id
                or self.citation.document_id != self.document_id
                or self.citation.score != self.score
                or self.citation.citation_id != f"C{self.rank}"):
            raise ValueError("Retrieval hit and citation must describe the same ranked chunk")
        return self


class VectorIndexManifest(_Record):
    """Metadata row order is the only mapping to vectors.npy row order."""

    index_version: Literal[1]
    knowledge_base_id: KnowledgeBaseId
    session_id: SessionId
    embedding_model_name: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=255)]
    embedding_model_revision: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=255)]
    embedding_dimension: PositiveInt
    normalize_embeddings: bool
    chunk_version: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=255)]
    vector_count: PositiveInt
    dtype: Literal["float32"]
    chunk_ids: tuple[ChunkId, ...]
    document_ids: tuple[DocumentId, ...]
    vectors_sha256: ContentHash
    source_chunks_sha256: ContentHash
    created_at: AwareDatetime

    @field_validator("index_version", mode="before")
    @classmethod
    def check_index_version(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("Unsupported vector index version")
        return value

    @field_validator("normalize_embeddings", mode="before")
    @classmethod
    def check_normalization_type(cls, value):
        if type(value) is not bool:
            raise ValueError("normalize_embeddings must be a bool")
        return value

    @model_validator(mode="after")
    def check_rows(self) -> Self:
        if (len(self.chunk_ids) != self.vector_count
                or len(self.document_ids) != self.vector_count):
            raise ValueError("Vector metadata row counts do not match")
        if len(set(self.chunk_ids)) != self.vector_count:
            raise ValueError("Duplicate chunk ID in vector index")
        return self
