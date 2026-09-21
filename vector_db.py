import os
import uuid
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

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
                self.client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="visibility",
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

    def list_sources(self, visibility: Optional[str] = None) -> List[str]:
        try:
            scroll_filter = None
            if visibility:
                scroll_filter = Filter(
                    must=[FieldCondition(key="visibility", match=MatchValue(value=visibility))]
                )
            scroll_res, _ = self.client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=scroll_filter,
                limit=500,
                with_payload=True,
                with_vectors=False,
            )
            sources = set()
            for point in scroll_res:
                if point.payload and "source_id" in point.payload:
                    if visibility and point.payload.get("visibility") != visibility:
                        continue
                    sources.add(point.payload["source_id"])
            return sorted(list(sources))
        except Exception:
            if hasattr(self, "_in_memory_store"):
                sources = set()
                for item in self._in_memory_store:
                    p = item.get("payload", {})
                    if visibility and p.get("visibility") != visibility:
                        continue
                    if "source_id" in p:
                        sources.add(p["source_id"])
                return sorted(list(sources))
            return []

    def upsert_chunks(
        self,
        source_id: str,
        chunks: List[Union[str, Dict[str, Any]]],
        vectors: List[List[float]],
        visibility: str = "public",
    ) -> int:
        if not chunks or not vectors:
            return 0

        points = []
        for chunk_item, vector in zip(chunks, vectors):
            point_id = str(uuid.uuid4())
            if isinstance(chunk_item, dict):
                text = chunk_item.get("text", "")
                page_number = chunk_item.get("page_number", 1)
                ocr_used = chunk_item.get("ocr_used", False)
            else:
                text = str(chunk_item)
                page_number = 1
                ocr_used = False

            payload = {
                "text": text,
                "source_id": source_id,
                "visibility": visibility,
                "page_number": page_number,
                "ocr_used": ocr_used,
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        total_upserted = 0
        for i in range(0, len(points), UPSERT_BATCH_SIZE):
            batch = points[i : i + UPSERT_BATCH_SIZE]
            if hasattr(self.client, "upsert"):
                try:
                    self.client.upsert(collection_name=COLLECTION_NAME, points=batch)
                    total_upserted += len(batch)
                except Exception:
                    # In-memory fallback
                    if not hasattr(self, "_in_memory_store"):
                        self._in_memory_store = []
                    for p in batch:
                        self._in_memory_store.append({"id": p.id, "vector": p.vector, "payload": p.payload})
                    total_upserted += len(batch)
            else:
                if not hasattr(self, "_in_memory_store"):
                    self._in_memory_store = []
                for p in batch:
                    self._in_memory_store.append({"id": p.id, "vector": p.vector, "payload": p.payload})
                total_upserted += len(batch)

        return total_upserted

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        source_id: Optional[str] = None,
        visibility: Optional[str] = None,
    ) -> Dict[str, Any]:
        filter_conditions = []
        if source_id and source_id.strip() and source_id.strip() != "All documents":
            filter_conditions.append(FieldCondition(key="source_id", match=MatchValue(value=source_id.strip())))
        if visibility:
            filter_conditions.append(FieldCondition(key="visibility", match=MatchValue(value=visibility.strip())))

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

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
                matches = [
                    {
                        "source_id": p.payload.get("source_id"),
                        "text": p.payload.get("text"),
                        "page_number": p.payload.get("page_number", 1),
                        "ocr_used": p.payload.get("ocr_used", False),
                        "visibility": p.payload.get("visibility", "public"),
                    }
                    for p in points if p.payload
                ]
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
                matches = [
                    {
                        "source_id": hit.payload.get("source_id"),
                        "text": hit.payload.get("text"),
                        "page_number": hit.payload.get("page_number", 1),
                        "ocr_used": hit.payload.get("ocr_used", False),
                        "visibility": hit.payload.get("visibility", "public"),
                    }
                    for hit in search_results if hit.payload
                ]
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
            payload = item.get("payload", {})
            matched_source = payload.get("source_id")
            matched_vis = payload.get("visibility", "public")

            if source_id and source_id.strip() and source_id.strip() != "All documents":
                if matched_source != source_id.strip():
                    continue
            if visibility and matched_vis != visibility:
                continue

            vec = item.get("vector")
            score = _cosine(query_vector, vec)
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [item for _, item in scored[:top_k]]
        contexts = [t["payload"]["text"] for t in top if t.get("payload") and "text" in t["payload"]]
        sources = list({t["payload"].get("source_id") for t in top if t.get("payload")})
        matches = [
            {
                "source_id": t["payload"].get("source_id"),
                "text": t["payload"].get("text"),
                "page_number": t["payload"].get("page_number", 1),
                "ocr_used": t["payload"].get("ocr_used", False),
                "visibility": t["payload"].get("visibility", "public"),
            }
            for t in top if t.get("payload")
        ]

        return {"contexts": contexts, "sources": sources, "matches": matches}


def get_qdrant_storage() -> QdrantStorage:
    return QdrantStorage()
