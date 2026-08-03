import os
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION_NAME = "rag_documents"
EMBEDDING_DIM = 384  # Matches BAAI/bge-small-en-v1.5 output dimension


class QdrantStorage:
    _instance: Optional["QdrantStorage"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QdrantStorage, cls).__new__(cls)
            
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")

            # Connect to Qdrant Cloud if both URL and API Key are present
            if qdrant_url and qdrant_api_key and qdrant_api_key.strip():
                cls._instance.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            else:
                # Fallback to persistent local vector database
                db_path = os.getenv("QDRANT_LOCAL_PATH", "./qdrant_db")
                cls._instance.client = QdrantClient(path=db_path)

            cls._instance._ensure_collection()
        return cls._instance

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )

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
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        return len(points)

    def search(self, query_vector: List[float], top_k: int = 5) -> Dict[str, Any]:
        search_results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
        )
        contexts = [hit.payload["text"] for hit in search_results if hit.payload and "text" in hit.payload]
        sources = list(set(hit.payload["source_id"] for hit in search_results if hit.payload and "source_id" in hit.payload))
        return {"contexts": contexts, "sources": sources}


def get_qdrant_storage() -> QdrantStorage:
    return QdrantStorage()
