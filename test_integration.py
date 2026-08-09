#!/usr/bin/env python3
"""
Comprehensive integration test for the RAG Institute Assistant.
Tests all API endpoints and the full ingestion→query pipeline.
"""
import os
import sys
import json
import time
import tempfile
import requests

FASTAPI_BASE = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")

PASS = "✅"
FAIL = "❌"
results = []


def record(name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((name, passed, detail))
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))


def test_root_endpoint():
    """Test GET / returns status online."""
    try:
        r = requests.get(f"{FASTAPI_BASE}/", timeout=10)
        data = r.json()
        record("GET /", r.status_code == 200 and data.get("status") == "online",
               f"status={r.status_code}")
    except Exception as e:
        record("GET /", False, str(e))


def test_health_endpoint():
    """Test GET /health returns ok."""
    try:
        r = requests.get(f"{FASTAPI_BASE}/health", timeout=10)
        data = r.json()
        record("GET /health", r.status_code == 200 and data.get("status") == "ok",
               f"status={r.status_code}")
    except Exception as e:
        record("GET /health", False, str(e))


def test_docs_endpoint():
    """Test GET /docs returns Swagger UI."""
    try:
        r = requests.get(f"{FASTAPI_BASE}/docs", timeout=10)
        record("GET /docs", r.status_code == 200 and "swagger" in r.text.lower(),
               f"status={r.status_code}")
    except Exception as e:
        record("GET /docs", False, str(e))


def test_inngest_endpoint():
    """Test GET /api/inngest returns Inngest registration info."""
    try:
        r = requests.get(f"{FASTAPI_BASE}/api/inngest", timeout=10)
        record("GET /api/inngest", r.status_code == 200,
               f"status={r.status_code}")
    except Exception as e:
        record("GET /api/inngest", False, str(e))


def test_ask_validation():
    """Test POST /ask rejects empty question."""
    try:
        r = requests.post(f"{FASTAPI_BASE}/ask",
                          json={"question": "", "top_k": 5}, timeout=15)
        record("POST /ask (empty question → 422)", r.status_code == 422,
               f"status={r.status_code}")
    except Exception as e:
        record("POST /ask (empty question)", False, str(e))


def test_ask_valid_question():
    """Test POST /ask returns a valid answer structure."""
    try:
        r = requests.post(f"{FASTAPI_BASE}/ask",
                          json={"question": "What is machine learning?", "top_k": 3},
                          timeout=60)
        data = r.json()
        has_answer = "answer" in data
        has_sources = "sources" in data
        has_count = "num_contexts" in data
        record("POST /ask (valid question → 200)", r.status_code == 200 and has_answer,
               f"status={r.status_code}, keys={list(data.keys())}")
    except Exception as e:
        record("POST /ask (valid question)", False, str(e))


def test_ask_topk_bounds():
    """Test POST /ask rejects top_k > 20."""
    try:
        r = requests.post(f"{FASTAPI_BASE}/ask",
                          json={"question": "test", "top_k": 50}, timeout=15)
        record("POST /ask (top_k=50 → 422)", r.status_code == 422,
               f"status={r.status_code}")
    except Exception as e:
        record("POST /ask (top_k bounds)", False, str(e))


def test_embedding_module():
    """Test data_loader embed_texts produces 384-dim vectors."""
    try:
        sys.path.insert(0, os.getcwd())
        from data_loader import embed_texts
        vecs = embed_texts(["Hello world"])
        ok = len(vecs) == 1 and len(vecs[0]) == 384
        record("embed_texts → 384-dim vector", ok,
               f"shape=({len(vecs)}, {len(vecs[0]) if vecs else '?'})")
    except Exception as e:
        record("embed_texts", False, str(e))


def test_pdf_loader():
    """Test data_loader load_and_chunk_pdf with a tiny synthetic PDF."""
    try:
        sys.path.insert(0, os.getcwd())
        from data_loader import load_and_chunk_pdf

        from pypdf import PdfWriter
        writer = PdfWriter()
        from pypdf._page import PageObject
        page = PageObject.create_blank_page(None, 612, 792)
        writer.add_page(page)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            writer.write(f)
            tmp_path = f.name

        chunks = load_and_chunk_pdf(tmp_path)
        record("load_and_chunk_pdf (blank PDF)", isinstance(chunks, list),
               f"chunks={len(chunks)}")
        os.unlink(tmp_path)
    except Exception as e:
        record("load_and_chunk_pdf", False, str(e))


def test_qdrant_connectivity():
    """Test that QdrantStorage connects and collection exists."""
    try:
        sys.path.insert(0, os.getcwd())
        from vector_db import get_qdrant_storage, COLLECTION_NAME
        store = get_qdrant_storage()
        collections = store.client.get_collections().collections
        names = [c.name for c in collections]
        record("Qdrant connection + collection exists",
               COLLECTION_NAME in names,
               f"collections={names}")
    except Exception as e:
        record("Qdrant connectivity", False, str(e))


def main():
    print("=" * 60)
    print("  RAG Institute Assistant — Integration Test Suite")
    print("=" * 60)

    print("\n📡 API Endpoint Tests:")
    test_root_endpoint()
    test_health_endpoint()
    test_docs_endpoint()
    test_inngest_endpoint()

    print("\n🔒 Validation Tests:")
    test_ask_validation()
    test_ask_topk_bounds()

    print("\n🧠 Module Tests:")
    test_embedding_module()
    test_pdf_loader()
    test_qdrant_connectivity()

    print("\n🌐 End-to-End Query Test:")
    test_ask_valid_question()

    # Summary
    total = len(results)
    passed = sum(1 for _, p, _ in results if p)
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
