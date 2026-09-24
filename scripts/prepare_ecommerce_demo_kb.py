"""Prepare one session-owned SOP KB using existing local RAG; no LLM/network."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'demo/ecommerce/knowledge_base/02_库存经营SOP.md'


def prepare(thread_id):
    from app.api.context import set_thread_context, reset_thread_context
    from app.utils.path_utils import validate_thread_id
    from app.local_rag.storage import LocalRAGStorage
    from app.local_rag.ingestion import ingest_document
    from app.local_rag.loaders import UploadedDocumentLoader
    from app.local_rag.embeddings import LocalEmbeddingAdapter
    from app.local_rag.vector_store import LocalVectorStore
    validate_thread_id(thread_id)
    token = set_thread_context(thread_id)
    try:
        storage = LocalRAGStorage()
        adapter = LocalEmbeddingAdapter()
        vectors = LocalVectorStore(storage)
        content_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        name = '电商库存 SOP'
        for summary in storage.list_knowledge_bases():
            manifest = storage.get_knowledge_base(summary.knowledge_base_id)
            if (manifest.name == name and len(manifest.documents) == 1
                    and manifest.documents[0].content_hash == content_hash
                    and manifest.index_status == 'indexed'):
                vectors.load_index(manifest.knowledge_base_id, adapter.metadata)
                return {'thread_id': thread_id, 'knowledge_base_id': manifest.knowledge_base_id,
                        'name': name, 'reused': True}
        # Source staging is separate from session uploads: Main must use RAG,
        # not receive SOP as an attachment merely because we prepared the KB.
        kb = storage.create_knowledge_base(name)
        (ROOT / '.tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / '.tmp', prefix='demo-kb-') as tmp:
            staging = Path(tmp) / ('session_' + thread_id)
            staging.mkdir()
            (staging / SOURCE.name).write_bytes(SOURCE.read_bytes())
            ingested = ingest_document(kb.knowledge_base_id, SOURCE.name, storage=storage,
                                       loader=UploadedDocumentLoader(tmp))
            vectors.build_index(kb.knowledge_base_id, adapter.embed_documents(ingested.chunks), adapter.metadata)
        return {'thread_id': thread_id, 'knowledge_base_id': kb.knowledge_base_id,
                'name': name, 'reused': False}
    finally:
        reset_thread_context(token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thread-id', default=None)
    args = parser.parse_args()
    result = prepare(args.thread_id or str(uuid4()))
    result['page_url'] = 'http://localhost:5173/?thread_id=' + result['thread_id']
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
