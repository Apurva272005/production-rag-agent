import os
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

COLLECTION_NAME = "rag_documents"
EMBEDDING_DIM = 384  # Matches BAAI/bge-small-en-v1.5 output dimension
UPSERT_BATCH_SIZE = 50


class QdrantStorage:
    _instance: Optional["QdrantStorage"] = None

    def __new__(cls, *args, **kwargs):
        db_path = kwargs.get("db_path") or os.getenv("QDRANT_LOCAL_PATH", "./qdrant_db")
        if cls._instance is None:
            cls._instance = super(QdrantStorage, cls).__new__(cls)

            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")

            if qdrant_url and qdrant_api_key and qdrant_api_key.strip():
                cls._instance.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)
            else:
                cls._instance.client = QdrantClient(path=db_path)

            cls._instance._ensure_collection()
        return cls._instance

    def _ensure_collection(self):
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == COLLECTION_NAME for c in collections)
            if not exists:
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                )
            try:
                self.client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="source_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass
        except Exception:
            pass

    def delete_source(self, source_id: str) -> None:
        if not source_id:
            return
        try:
            self.client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
                ),
            )
        except Exception:
            pass

    def list_sources(self) -> List[str]:
        try:
            scroll_res, _ = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=500,
                with_payload=True,
                with_vectors=False,
            )
            sources = set()
            for point in scroll_res:
                if point.payload and "source_id" in point.payload:
                    sources.add(point.payload["source_id"])
            return sorted(list(sources))
        except Exception:
            return []

    def upsert_chunks(self, source_id: str, chunks: List[str], vectors: List[List[float]]) -> int:
        if not chunks or not vectors:
            return 0
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

        total_upserted = 0
        for i in range(0, len(points), UPSERT_BATCH_SIZE):
            batch = points[i : i + UPSERT_BATCH_SIZE]
            if hasattr(self.client, "upsert"):
                self.client.upsert(collection_name=COLLECTION_NAME, points=batch)
                total_upserted += len(batch)
            else:
                if not hasattr(self, "_in_memory_store"):
                    self._in_memory_store = []
                for p in batch:
                    self._in_memory_store.append({"id": p.id, "vector": p.vector, "payload": p.payload})
                total_upserted += len(batch)

        return total_upserted

    def search(self, query_vector: List[float], top_k: int = 5, source_id: Optional[str] = None) -> Dict[str, Any]:
        query_filter = None
        if source_id and source_id.strip() and source_id.strip() != "All documents":
            query_filter = Filter(
                must=[FieldCondition(key="source_id", match=MatchValue(value=source_id.strip()))]
            )

        if hasattr(self.client, "query_points"):
            try:
                res = self.client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                )
                points = res.points
                contexts = [p.payload["text"] for p in points if p.payload and "text" in p.payload]
                sources = list(set(p.payload["source_id"] for p in points if p.payload and "source_id" in p.payload))
                matches = [{"source_id": p.payload.get("source_id"), "text": p.payload.get("text")} for p in points if p.payload]
                return {"contexts": contexts, "sources": sources, "matches": matches}
            except Exception:
                pass

        if hasattr(self.client, "search"):
            try:
                search_results = self.client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                )
                contexts = [hit.payload["text"] for hit in search_results if hit.payload and "text" in hit.payload]
                sources = list(set(hit.payload["source_id"] for hit in search_results if hit.payload and "source_id" in hit.payload))
                matches = [{"source_id": hit.payload.get("source_id"), "text": hit.payload.get("text")} for hit in search_results if hit.payload]
                return {"contexts": contexts, "sources": sources, "matches": matches}
            except Exception:
                pass

        if not hasattr(self, "_in_memory_store") or not self._in_memory_store:
            return {"contexts": [], "sources": [], "matches": []}

        def _cosine(a, b):
            import math
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        scored = []
        for item in self._in_memory_store:
            matched_source = item["payload"].get("source_id") if item.get("payload") else None
            if source_id and source_id.strip() and source_id.strip() != "All documents":
                if matched_source != source_id.strip():
                    continue
            vec = item.get("vector")
            score = _cosine(query_vector, vec)
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [item for _, item in scored[:top_k]]
        contexts = [t["payload"]["text"] for t in top if t.get("payload") and "text" in t["payload"]]
        sources = list({t["payload"].get("source_id") for t in top if t.get("payload")})
        matches = [{"source_id": t["payload"].get("source_id"), "text": t["payload"].get("text")} for t in top if t.get("payload")]

        return {"contexts": contexts, "sources": sources, "matches": matches}


def get_qdrant_storage() -> QdrantStorage:
    return QdrantStorage()
