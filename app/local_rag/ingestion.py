"""Offline orchestration only: no vectors, retrieval, tools or model calls."""

from app.local_rag import config
from app.local_rag.chunking import chunk_document
from app.local_rag.loaders import UploadedDocumentLoader, parse_snapshot
from app.local_rag.schemas import IngestionResult
from app.local_rag.storage import LocalRAGStorage


def ingest_document(knowledge_base_id: str, filename: str, *,
                    storage: LocalRAGStorage | None = None,
                    loader: UploadedDocumentLoader | None = None) -> IngestionResult:
    """Business inputs are KB ID and session-relative upload filename.

    storage/loader are trusted backend dependencies, never model tool arguments.
    Check KB authorization before opening the current session's uploaded file.
    Old chunking versions are never reused or overwritten. For this development
    stage, explicitly ingest the original upload into a new knowledge base to
    rebuild; no in-place migration or deletion of old data is performed.
    """
    storage = storage if storage is not None else LocalRAGStorage()
    loader = loader if loader is not None else UploadedDocumentLoader()
    manifest = storage.get_knowledge_base(knowledge_base_id)
    snapshot = loader.read(filename)
    existing = next((doc for doc in manifest.documents if doc.content_hash == snapshot.content_hash), None)
    if existing and existing.ingestion_status in {"chunked", "indexed"}:
        if existing.chunking_version != config.CHUNKING_VERSION:
            raise ValueError("分块版本已变化，需要显式重建；本轮不会自动覆盖已有 Chunk。")
        return storage.get_document_ingestion(knowledge_base_id, existing.document_id).model_copy(update={"duplicate": True})
    parsed = parse_snapshot(snapshot)
    return storage.save_parsed_document(knowledge_base_id, parsed, chunk_document)
