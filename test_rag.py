import os
import unittest
from data_loader import embed_texts
from vector_db import QdrantStorage


class TestRAGComponents(unittest.TestCase):
    def test_embedding_generation(self):
        sample_texts = ["Attendance policy requires 75% minimum attendance.", "CE363 is Machine Learning."]
        vectors = embed_texts(sample_texts)
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 384)

    def test_qdrant_storage_and_search(self):
        storage = QdrantStorage(db_path="./test_qdrant_db")
        sample_texts = ["Students must submit assignments before the deadline.", "Late submissions incur a 10% penalty."]
        vectors = embed_texts(sample_texts)
        storage.upsert_chunks(source_id="test_policy", chunks=sample_texts, vectors=vectors)

        query = "What is the penalty for late submissions?"
        query_vec = embed_texts([query])[0]
        results = storage.search(query_vec, top_k=2)

        self.assertIn("test_policy", results["sources"])
        self.assertTrue(len(results["contexts"]) > 0)
        self.assertTrue(any("penalty" in c for c in results["contexts"]))

if __name__ == "__main__":
    unittest.main()
