import atexit
import os
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

COLLECTION_NAME = "rag_documents"
EMBEDDING_DIM = 384  # Matches BAAI/bge-small-en-v1.5 output dimension
UPSERT_BATCH_SIZE = 50


class QdrantStorage:
    _instance: Optional["QdrantStorage"] = None
    _atexit_registered = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QdrantStorage, cls).__new__(cls)

            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")

            # Connect to Qdrant Cloud if both URL and API Key are present
            if qdrant_url and qdrant_api_key and qdrant_api_key.strip():
                cls._instance.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)
            else:
                # Fallback to persistent local vector database
                db_path = os.getenv("QDRANT_LOCAL_PATH", "./qdrant_db")
                cls._instance.client = QdrantClient(path=db_path, timeout=60)

            cls._instance._ensure_collection()
            if not cls._atexit_registered:
                atexit.register(close_qdrant_storage)
                cls._atexit_registered = True
        return cls._instance

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
        try:
            from qdrant_client.models import PayloadSchemaType
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="source_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    def upsert_chunks(self, source_id: str, chunks: List[str], vectors: List[List[float]]) -> int:
        if not chunks or not vectors:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"text": chunk, "source_id": source_id},
                )
            )

        # Batch upserts to prevent payload overflow and HTTP write timeouts on large documents
        total_points = len(points)
        for i in range(0, total_points, UPSERT_BATCH_SIZE):
            batch = points[i : i + UPSERT_BATCH_SIZE]
            self.client.upsert(collection_name=COLLECTION_NAME, points=batch, timeout=60)

        return total_points

    def _source_filter(self, source_id: str) -> Filter:
        return Filter(
            must=[
                FieldCondition(
                    key="source_id",
                    match=MatchValue(value=source_id),
                )
            ]
        )

    def delete_source(self, source_id: str) -> None:
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=self._source_filter(source_id),
            wait=True,
        )

    def list_sources(self) -> List[str]:
        sources = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                if not point.payload or not point.payload.get("source_id"):
                    continue
                src = point.payload["source_id"]
                if src not in sources:
                    sources.append(src)
            if offset is None:
                break
        return sources

    def search(self, query_vector: List[float], top_k: int = 5, source_id: Optional[str] = None) -> Dict[str, Any]:
        result = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=self._source_filter(source_id) if source_id else None,
            limit=top_k,
            with_payload=True,
        )
        matches = []
        sources = []
        for hit in result.points:
            if not hit.payload or "text" not in hit.payload:
                continue
            matched_source = hit.payload.get("source_id", "unknown")
            if matched_source not in sources:
                sources.append(matched_source)
            matches.append(
                {
                    "text": hit.payload["text"],
                    "source_id": matched_source,
                    "score": getattr(hit, "score", None),
                }
            )

        contexts = [match["text"] for match in matches]
        return {"contexts": contexts, "sources": sources, "matches": matches}

    def close(self) -> None:
        client = getattr(self, "client", None)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def get_qdrant_storage() -> QdrantStorage:
    return QdrantStorage()


def close_qdrant_storage() -> None:
    storage = QdrantStorage._instance
    if storage is None:
        return
    try:
        storage.close()
    except Exception:
        pass
    finally:
        QdrantStorage._instance = None
