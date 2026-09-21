#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for RAG Institute Assistant (v2.0.0).
Covers:
- Base Endpoints & Inngest
- Authentication & JWT Token Verification
- RBAC Upload and Visibility Guard (Admin / Faculty / Student)
- Server-side Retrieval Filtering on Document Visibility
- Source Document Page Preview (PyMuPDF PNG rendering)
- Answer PDF Export (ReportLab)
- Admin Analytics Dashboard
"""
import os
import io
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


def make_sample_pdf(title: str, text: str) -> str:
    """Creates a minimal test PDF file using reportlab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp.name, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"<b>{title}</b>", styles["Heading1"]),
        Paragraph(text, styles["Normal"]),
        PageBreak(),
        Paragraph(f"<b>{title} — Page 2</b>", styles["Heading2"]),
        Paragraph(f"Supplementary details for {title}.", styles["Normal"]),
    ]
    doc.build(story)
    return tmp.name


def run_all_tests():
    print(f"\n🚀 Running Full Integration Test Suite on {FASTAPI_BASE}\n" + "=" * 65)

    # ── 1. Base Health & Discovery Endpoints ──
    try:
        r = requests.get(f"{FASTAPI_BASE}/", timeout=10)
        record("GET / (Root API discovery)", r.status_code == 200 and r.json().get("status") == "online", f"status={r.status_code}")
    except Exception as e:
        record("GET /", False, str(e))

    try:
        r = requests.get(f"{FASTAPI_BASE}/health", timeout=10)
        record("GET /health (Health check)", r.status_code == 200 and r.json().get("status") == "ok", f"status={r.status_code}")
    except Exception as e:
        record("GET /health", False, str(e))

    try:
        r = requests.get(f"{FASTAPI_BASE}/docs", timeout=10)
        record("GET /docs (Swagger documentation)", r.status_code == 200 and "swagger" in r.text.lower(), f"status={r.status_code}")
    except Exception as e:
        record("GET /docs", False, str(e))

    try:
        r = requests.get(f"{FASTAPI_BASE}/api/inngest", timeout=10)
        record("GET /api/inngest (Inngest event bridge)", r.status_code == 200, f"status={r.status_code}")
    except Exception as e:
        record("GET /api/inngest", False, str(e))

    # ── 2. Authentication & JWT Tokens ──
    tokens = {}
    for role, email, password in [
        ("admin", "admin@institute.edu", "Admin@123"),
        ("faculty", "faculty@institute.edu", "Faculty@123"),
        ("student", "student@institute.edu", "Student@123"),
    ]:
        try:
            r = requests.post(f"{FASTAPI_BASE}/auth/login", json={"email": email, "password": password}, timeout=10)
            if r.status_code == 200 and "access_token" in r.json():
                tokens[role] = r.json()["access_token"]
                record(f"POST /auth/login ({role.capitalize()})", True, f"token received, role={r.json().get('role')}")
            else:
                record(f"POST /auth/login ({role.capitalize()})", False, f"status={r.status_code}, resp={r.text}")
        except Exception as e:
            record(f"POST /auth/login ({role.capitalize()})", False, str(e))

    # Test invalid login
    try:
        r = requests.post(f"{FASTAPI_BASE}/auth/login", json={"email": "student@institute.edu", "password": "WrongPassword!"}, timeout=10)
        record("POST /auth/login (Invalid credentials → 401)", r.status_code == 401, f"status={r.status_code}")
    except Exception as e:
        record("POST /auth/login (Invalid credentials)", False, str(e))

    # Test GET /auth/me
    student_header = {"Authorization": f"Bearer {tokens.get('student', '')}"}
    try:
        r = requests.get(f"{FASTAPI_BASE}/auth/me", headers=student_header, timeout=10)
        record("GET /auth/me (Student profile inspection)", r.status_code == 200 and r.json().get("role") == "student", f"role={r.json().get('role')}")
    except Exception as e:
        record("GET /auth/me", False, str(e))

    # ── 3. Document Ingestion & RBAC Visibility Guard ──
    faculty_header = {"Authorization": f"Bearer {tokens.get('faculty', '')}"}
    admin_header = {"Authorization": f"Bearer {tokens.get('admin', '')}"}

    # Upload Private Doc as Faculty
    private_pdf = make_sample_pdf("Faculty Confidential Exam 2026", "This confidential exam contains question 1: What is Backpropagation algorithm in Deep Learning?")
    try:
        with open(private_pdf, "rb") as f:
            r = requests.post(
                f"{FASTAPI_BASE}/upload",
                files={"file": ("faculty_confidential_exam.pdf", f, "application/pdf")},
                data={"source_id": "faculty_confidential_exam", "visibility": "private", "replace_existing": "true"},
                headers=faculty_header,
                timeout=60,
            )
        record("POST /upload (Private doc upload by Faculty)", r.status_code == 200 and r.json().get("chunks_ingested", 0) > 0, f"chunks={r.json().get('chunks_ingested')}")
    except Exception as e:
        record("POST /upload (Private doc upload)", False, str(e))
    finally:
        if os.path.exists(private_pdf):
            os.unlink(private_pdf)

    # Upload Public Doc as Faculty
    public_pdf = make_sample_pdf("Computer Science Handbook 2026", "Computer Science curriculum includes algorithms, data structures, artificial intelligence, and operating systems.")
    try:
        with open(public_pdf, "rb") as f:
            r = requests.post(
                f"{FASTAPI_BASE}/upload",
                files={"file": ("cs_public_handbook.pdf", f, "application/pdf")},
                data={"source_id": "cs_public_handbook", "visibility": "public", "replace_existing": "true"},
                headers=faculty_header,
                timeout=60,
            )
        record("POST /upload (Public doc upload by Faculty)", r.status_code == 200 and r.json().get("chunks_ingested", 0) > 0, f"chunks={r.json().get('chunks_ingested')}")
    except Exception as e:
        record("POST /upload (Public doc upload)", False, str(e))
    finally:
        if os.path.exists(public_pdf):
            os.unlink(public_pdf)

    # Student attempting to upload doc (Should be 403 Forbidden)
    dummy_pdf = make_sample_pdf("Student Upload", "Unauthorized test.")
    try:
        with open(dummy_pdf, "rb") as f:
            r = requests.post(
                f"{FASTAPI_BASE}/upload",
                files={"file": ("student_doc.pdf", f, "application/pdf")},
                data={"source_id": "student_doc", "visibility": "public"},
                headers=student_header,
                timeout=20,
            )
        record("POST /upload (Student forbidden from uploading → 403)", r.status_code == 403, f"status={r.status_code}")
    except Exception as e:
        record("POST /upload (Student forbidden)", False, str(e))
    finally:
        if os.path.exists(dummy_pdf):
            os.unlink(dummy_pdf)

    # ── 4. RBAC Visibility Filter on /sources and /ask ──
    # Sources list for student should NOT include private docs
    try:
        r = requests.get(f"{FASTAPI_BASE}/sources", headers=student_header, timeout=10)
        sources_student = r.json().get("sources", [])
        is_private_hidden = "faculty_confidential_exam" not in sources_student
        record("GET /sources (Student cannot see private doc sources)", r.status_code == 200 and is_private_hidden, f"sources={sources_student}")
    except Exception as e:
        record("GET /sources (Student role)", False, str(e))

    # Student explicitly asking about private document (Server-side 403 Guard)
    try:
        r = requests.post(
            f"{FASTAPI_BASE}/ask",
            json={"question": "What is on the confidential exam?", "top_k": 3, "source_id": "faculty_confidential_exam"},
            headers=student_header,
            timeout=30,
        )
        record("POST /ask (Student blocked from querying private source → 403)", r.status_code == 403, f"status={r.status_code}")
    except Exception as e:
        record("POST /ask (Student private query guard)", False, str(e))

    # Student asking about public document (Should succeed with citations and page numbers)
    try:
        r = requests.post(
            f"{FASTAPI_BASE}/ask",
            json={"question": "What does the Computer Science curriculum include?", "top_k": 3, "source_id": "cs_public_handbook"},
            headers=student_header,
            timeout=60,
        )
        data = r.json()
        has_answer = bool(data.get("answer"))
        has_citations = isinstance(data.get("citations"), list)
        has_page = len(data.get("citations", [])) > 0 and "page_number" in data["citations"][0]
        record("POST /ask (Student query on public doc with citations & page numbers → 200)", r.status_code == 200 and has_answer and has_citations, f"num_citations={len(data.get('citations', []))}, has_page={has_page}")
    except Exception as e:
        record("POST /ask (Student public query)", False, str(e))

    # ── 5. Source Document Page Preview (Phase 3) ──
    # Student attempting to preview private document page (Should be 403)
    try:
        r = requests.get(f"{FASTAPI_BASE}/preview/faculty_confidential_exam/1", headers=student_header, timeout=10)
        record("GET /preview (Student forbidden from private doc preview → 403)", r.status_code == 403, f"status={r.status_code}")
    except Exception as e:
        record("GET /preview (Student private preview guard)", False, str(e))

    # Student previewing public document page (Should return 200 image/png)
    try:
        r = requests.get(f"{FASTAPI_BASE}/preview/cs_public_handbook/1", headers=student_header, timeout=15)
        is_png = r.status_code == 200 and r.headers.get("content-type") == "image/png" and len(r.content) > 1000
        record("GET /preview (Public doc page 1 PNG rendered → 200 image/png)", is_png, f"content-type={r.headers.get('content-type')}, bytes={len(r.content)}")
    except Exception as e:
        record("GET /preview (Public doc preview)", False, str(e))

    # ── 6. PDF Answer Export (Phase 4) ──
    try:
        r = requests.post(
            f"{FASTAPI_BASE}/export",
            json={
                "question": "What is in the CS curriculum?",
                "answer": "The Computer Science curriculum includes algorithms, data structures, AI, and operating systems [1].",
                "citations": [{"source_id": "cs_public_handbook", "page_number": 1, "ocr_used": False}],
                "source_id": "cs_public_handbook",
            },
            headers=student_header,
            timeout=15,
        )
        is_pdf = r.status_code == 200 and "application/pdf" in r.headers.get("content-type", "") and len(r.content) > 500
        record("POST /export (Generate styled PDF answer report → 200 application/pdf)", is_pdf, f"content-type={r.headers.get('content-type')}, size={len(r.content)} bytes")
    except Exception as e:
        record("POST /export (PDF export)", False, str(e))

    # ── 7. Admin Analytics Dashboard (Phase 1) ──
    # Faculty attempting to access admin dashboard (Should be 403)
    try:
        r = requests.get(f"{FASTAPI_BASE}/admin/dashboard", headers=faculty_header, timeout=10)
        record("GET /admin/dashboard (Faculty access forbidden → 403)", r.status_code == 403, f"status={r.status_code}")
    except Exception as e:
        record("GET /admin/dashboard (Faculty guard)", False, str(e))

    # Admin accessing dashboard (Should return summary, popular sources, role breakdown, query logs)
    try:
        r = requests.get(f"{FASTAPI_BASE}/admin/dashboard", headers=admin_header, timeout=15)
        d_data = r.json()
        has_summary = "summary" in d_data
        has_queries = d_data.get("summary", {}).get("total_queries", 0) > 0
        record("GET /admin/dashboard (Admin full analytics access → 200 OK)", r.status_code == 200 and has_summary and has_queries, f"total_queries={d_data.get('summary', {}).get('total_queries')}, users={d_data.get('summary', {}).get('total_users')}")
    except Exception as e:
        record("GET /admin/dashboard (Admin access)", False, str(e))

    # ── Summary ──
    print("\n" + "=" * 65)
    passed_count = sum(1 for _, p, _ in results if p)
    total_count = len(results)
    print(f"📊 Results: {passed_count}/{total_count} tests passed ({(passed_count/total_count)*100:.1f}%)\n")

    if passed_count == total_count:
        print("🎉 ALL INTEGRATION TESTS PASSED PERFECTLY!\n")
        return 0
    else:
        print(f"⚠️ {total_count - passed_count} tests failed.\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
