"""Exact local vector retrieval; no threshold, reranking or external services.

Session authorization is inherited from Storage's ContextVar. A retriever holds
no corpus text cache. Citation IDs are local to one returned tuple, not global.
"""

import numpy as np

from app.local_rag.embeddings import LocalEmbeddingAdapter, get_embedding_adapter
from app.local_rag.schemas import Citation, EmbeddingVector, RetrievalHit
from app.local_rag.storage import LocalRAGStorage, _WRITE_LOCK
from app.local_rag.vector_store import LocalVectorStore, VectorIndexCorruptError


class RetrievalError(ValueError):
    """Invalid retrieval parameters or query vector."""


class LocalRAGRetriever:
    def __init__(self, storage: LocalRAGStorage | None = None,
                 adapter: LocalEmbeddingAdapter | None = None):
        self.storage = storage or LocalRAGStorage()
        self.adapter = adapter if adapter is not None else get_embedding_adapter()
        self.vector_store = LocalVectorStore(self.storage)

    def retrieve(self, knowledge_base_id: str, query: str, *,
                 top_k: int = 5) -> tuple[RetrievalHit, ...]:
        """Return up to top_k hits, score DESC, ties in original index order.

        Absent/empty, stale or corrupt indexes raise the existing Storage/Vector
        Store errors. No automatic rebuild or silent omission of missing chunks.
        """
        if type(top_k) is not int or top_k <= 0:
            raise RetrievalError("top_k must be a positive integer")
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("query must be nonempty text")
        # Authorize before even loading model weights. Revalidate the index after
        # encoding; model inference does not need to hold the Storage write lock.
        self.storage.get_knowledge_base(knowledge_base_id)
        runtime = self.adapter.metadata
        encoded = self.adapter.embed_query(query)
        vector = self._query_vector(encoded, runtime.embedding_dimension,
                                    runtime.normalize_embeddings)
        with _WRITE_LOCK:
            index = self.vector_store.load_index(knowledge_base_id, runtime)
            # load_index owns all model/revision/normalization/source checks.
            # Keep its validated snapshot and source mapping under the same
            # in-process lock used by ingestion/build/delete.
            scores = self._scores(index.vectors, vector, index.metadata.normalize_embeddings)
            order = np.argsort(-scores, kind="stable")[:top_k]
            documents = {}
            hits = []
            for rank, row in enumerate(order, start=1):
                chunk_id = index.metadata.chunk_ids[row]
                document_id = index.metadata.document_ids[row]
                if document_id not in documents:
                    loaded = self.storage.get_document_ingestion(knowledge_base_id, document_id)
                    documents[document_id] = (
                        loaded.document, {chunk.chunk_id: chunk for chunk in loaded.chunks},
                    )
                document, chunks = documents[document_id]
                chunk = chunks.get(chunk_id)
                if chunk is None or chunk.document_id != document_id:
                    raise VectorIndexCorruptError("Index chunk is missing or mapped to another document")
                score = float(scores[row])
                citation = Citation(
                    citation_id=f"C{rank}", knowledge_base_id=knowledge_base_id,
                    document_id=document_id, document_name=document.document_name,
                    chunk_id=chunk_id, score=score,
                    score_type="inner_product" if runtime.normalize_embeddings else "cosine_similarity",
                    **{name: getattr(chunk, name) for name in (
                        "page", "start_line", "end_line", "heading",
                        "source_block_index", "start_char", "end_char",
                    )},
                )
                hits.append(RetrievalHit(chunk_id=chunk_id, document_id=document_id,
                                         rank=rank, score=score, text=chunk.text, citation=citation))
            return tuple(hits)

    @staticmethod
    def _query_vector(encoded, dimension, normalized):
        if not isinstance(encoded, EmbeddingVector):
            raise RetrievalError("Query embedding must be an EmbeddingVector")
        try:
            encoded = EmbeddingVector.model_validate(encoded.model_dump(warnings=False))
        except ValueError:
            raise RetrievalError("Query vector is malformed or contains NaN/Infinity") from None
        if encoded.dimension != dimension:
            raise RetrievalError("Query embedding dimension differs from index runtime")
        vector = np.asarray(encoded.vector, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            raise RetrievalError("Query vector has zero or nonfinite norm")
        if normalized and not np.isclose(norm, 1.0, rtol=1e-4, atol=1e-4):
            raise RetrievalError("Query vector does not match normalization configuration")
        return vector

    @staticmethod
    def _scores(matrix, query, normalized):
        # Float64 arithmetic avoids float32 intermediate overflow. Non-normalized
        # inputs are normalized only for cosine calculation, never persisted.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if normalized:
                scores = matrix @ query
            else:
                scores = (matrix @ (query / np.linalg.norm(query))) / np.linalg.norm(
                    matrix.astype(np.float64), axis=1,
                )
        if not np.isfinite(scores).all():
            raise RetrievalError("Similarity calculation produced NaN/Infinity")
        return scores
