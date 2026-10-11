import os
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
import shutil
import io
import json
import logging
import time
from urllib.parse import urlencode
from datetime import timedelta, datetime
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Response, status, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from sqlalchemy.orm import Session
from sqlalchemy import func
import inngest
import inngest.fast_api

# ReportLab for Phase 4 Export
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# PyMuPDF for Phase 3 Page Preview
try:
    import pymupdf
except Exception:
    try:
        import fitz as pymupdf
    except Exception:
        pymupdf = None

from database import get_db, SessionLocal, Base, engine, ensure_rbac_schema
from models_db import AuditLog, DocumentRecord, QueryLog
from clerk_auth import CurrentUser, get_current_user, get_current_token, record_revoked_token, require_roles
from rbac_filters import build_access_filter
from routers.clerk_admin_routes import router as clerk_admin_router
from routers.clerk_webhook_routes import router as clerk_webhook_router
from clerk_api import ClerkAPIError, ClerkConfigurationError, check_connection, list_users, revoke_session, secret_diagnostics, validate_clerk_secret
from clerk_policy import validate_domain_settings
from custom_types import (
    AskRequest,
    AskResponse,
    CitationItem,
    IngestPDFData,
    IngestResult,
    ExportRequest,
)
from data_loader import load_and_chunk_pdf_structured, embed_texts
from vector_db import get_qdrant_storage

logger = logging.getLogger("rag_api")

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
validate_clerk_secret()
validate_domain_settings()

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

# Initialize database tables. Clerk is the identity and role source of truth.
Base.metadata.create_all(bind=engine)
ensure_rbac_schema()

DEFAULT_OPENROUTER_MODELS = [
    "minimax/minimax-m3:free",
    "poolside/laguna-s-2.1:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
]
QUERY_RATE_LIMIT_PER_MINUTE = int(os.getenv("QUERY_RATE_LIMIT_PER_MINUTE", "30"))


def get_openrouter_models() -> list[str]:
    configured_model = os.getenv("OPENROUTER_MODEL", "").strip()
    if configured_model:
        return [configured_model, *[model for model in DEFAULT_OPENROUTER_MODELS if model != configured_model]]
    return list(DEFAULT_OPENROUTER_MODELS)


def build_context_block(matches: list[dict[str, Any]]) -> str:
    if not matches:
        return "No relevant context was found in the ingested documents."

    context_items = []
    for index, match in enumerate(matches, start=1):
        source_id = match.get("source_id", "unknown")
        page_num = match.get("page_number", 1)
        ocr_badge = " [OCR]" if match.get("ocr_used") else ""
        text = match.get("text", "").strip()
        if text:
            context_items.append(f"[{index}] Source: {source_id} (Page {page_num}{ocr_badge})\n{text}")
    return "\n\n".join(context_items) or "No relevant context was found in the ingested documents."


def build_chat_messages(question: str, context_block: str, has_context: bool) -> list[dict[str, str]]:
    system_prompt = (
        "You are a helpful ChatGPT-style assistant for a document-grounded RAG app. "
        "Use the retrieved source context as the primary authority. "
        "Give a clear, complete, conversational answer with useful structure. "
        "Start with the direct answer, then add explanation, bullets, or steps when helpful. "
        "When you use document facts, cite the context number like [1] or [2]. "
        "Do not invent institute policies, dates, fees, rules, or document-specific facts. "
        "If the retrieved context is insufficient, say what is missing and then, only if useful, "
        "add a separate 'General guidance' section that is clearly not from the uploaded documents."
    )

    context_instruction = (
        "Relevant source context is available below. Answer the question using it."
        if has_context
        else "No relevant uploaded-document context was retrieved. Be transparent about that."
    )

    user_prompt = (
        f"{context_instruction}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        f"User question:\n{question}\n\n"
        "Write the best possible answer for the user. Keep it grounded, useful, and easy to read."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def document_is_accessible(user: CurrentUser, document: DocumentRecord) -> bool:
    if user.role == "admin":
        return True
    if document.visibility in {"public", "student"}:
        return True
    if user.role != "faculty" or document.visibility != "faculty":
        return False
    return document.owner_id == user.id or (
        bool(user.department) and document.department == user.department
    )


def document_is_manageable(user: CurrentUser, document: DocumentRecord) -> bool:
    if user.role == "admin":
        return True
    return (
        user.role == "faculty"
        and document.owner_id == user.id
        and document.department == user.department
    )


def validate_ingest_scope(user: CurrentUser, visibility: str, department: Optional[str]) -> str:
    allowed = {"public", "student", "faculty", "admin"}
    if visibility not in allowed:
        raise HTTPException(status_code=400, detail=f"visibility must be one of {sorted(allowed)}")
    if user.role == "faculty":
        if visibility == "admin":
            raise HTTPException(status_code=403, detail="Faculty cannot ingest admin-only documents")
        if not user.department:
            raise HTTPException(status_code=403, detail="Faculty account has no department assigned")
        if department and department != user.department:
            raise HTTPException(status_code=403, detail="Faculty can only manage their own department")
        return user.department
    return department or user.department or ""


def enforce_query_rate_limit(db: Session, user: CurrentUser) -> None:
    cutoff = datetime.utcnow() - timedelta(minutes=1)
    recent = (
        db.query(QueryLog)
        .filter(QueryLog.user_id == user.id, QueryLog.timestamp >= cutoff)
        .count()
    )
    if recent >= QUERY_RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Query rate limit exceeded; try again shortly")


def write_audit(db: Session, user: CurrentUser, action: str, resource_id: Optional[str], details: str = "") -> None:
    db.add(
        AuditLog(
            actor_id=user.id,
            actor_role=user.role,
            action=action,
            resource_id=resource_id,
            details=details,
        )
    )
    db.commit()


def ingest_pdf_file(
    pdf_path: str,
    source_id: str,
    visibility: str = "public",
    uploaded_by: str = "system",
    replace_existing: bool = True,
    owner_id: Optional[str] = None,
    department: Optional[str] = None,
    db: Optional[Session] = None,
) -> IngestResult:
    """Ingests PDF, computes embeddings, indexes into Qdrant, records in DB, and stores file for preview."""
    structured_chunks = load_and_chunk_pdf_structured(pdf_path)

    if not structured_chunks:
        return IngestResult(status="empty", source_id=source_id, chunks_ingested=0, visibility=visibility)

    texts = [c["text"] for c in structured_chunks]
    vectors = embed_texts(texts)
    store = get_qdrant_storage()

    if replace_existing:
        store.delete_source(source_id)

    count = store.upsert_chunks(
        source_id=source_id,
        chunks=structured_chunks,
        vectors=vectors,
        visibility=visibility,
        owner_id=owner_id,
        department=department,
        doc_id=source_id,
    )

    # Persist PDF file to storage directory for Phase 3 page preview
    dest_path = os.path.join(STORAGE_DIR, f"{source_id}.pdf")
    if os.path.abspath(pdf_path) != os.path.abspath(dest_path):
        try:
            shutil.copy2(pdf_path, dest_path)
        except Exception:
            pass

    # Save / update record in Database
    should_close_db = False
    if db is None:
        db = SessionLocal()
        should_close_db = True

    try:
        doc_record = db.query(DocumentRecord).filter(DocumentRecord.source_id == source_id).first()
        if not doc_record:
            doc_record = DocumentRecord(
                source_id=source_id,
                file_name=os.path.basename(pdf_path),
                uploaded_by=uploaded_by,
                visibility=visibility,
                owner_id=owner_id,
                department=department,
                doc_id=source_id,
                file_path=dest_path,
                chunks_count=count,
            )
            db.add(doc_record)
        else:
            doc_record.visibility = visibility
            doc_record.uploaded_by = uploaded_by
            doc_record.file_path = dest_path
            doc_record.chunks_count = count
            doc_record.owner_id = owner_id
            doc_record.department = department
            doc_record.doc_id = source_id
            doc_record.uploaded_at = datetime.utcnow()
        db.commit()
    finally:
        if should_close_db:
            db.close()

    return IngestResult(status="success", source_id=source_id, chunks_ingested=count, visibility=visibility, department=department)


# ── Inngest Client & Async Ingestion Workflow ───────────────────────────────
try:
    inngest_client = inngest.Inngest(app_id="rag_institute_assistant")
except Exception:
    class _InngestStub:
        def create_function(self, *a, **k):
            def _decorator(fn=None, **_kw):
                if fn is None:
                    return lambda f: f
                return fn
            return _decorator

    inngest_client = _InngestStub()


@inngest_client.create_function(
    fn_id="ingest-pdf",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(limit=2, period=timedelta(seconds=60)),
    rate_limit=inngest.RateLimit(limit=1, period=timedelta(hours=4), key="event.data.source_id"),
)
async def ingest_pdf(ctx: inngest.Context, step: inngest.Step) -> Dict[str, Any]:
    pdf_path = ctx.event.data.get("pdf_path")
    source_id = ctx.event.data.get("source_id")
    visibility = ctx.event.data.get("visibility", "public")
    owner_id = ctx.event.data.get("user_id")
    role = ctx.event.data.get("role", "student")
    department = ctx.event.data.get("department")

    if not pdf_path or not source_id:
        raise ValueError("Missing pdf_path or source_id in event data")
    if role not in {"admin", "faculty"}:
        raise ValueError("Ingestion event is not authorized")
    if role == "faculty" and visibility == "admin":
        raise ValueError("Faculty cannot ingest admin-only documents")

    if not os.path.exists(pdf_path):
        return {
            "status": "completed_via_direct_upload",
            "source_id": source_id,
            "message": f"File {pdf_path} was already processed.",
        }

    replace_existing = ctx.event.data.get("replace_existing", True)

    chunks = await step.run("extract-chunks", lambda: load_and_chunk_pdf_structured(pdf_path))
    if not chunks:
        return IngestResult(status="empty", source_id=source_id, chunks_ingested=0, visibility=visibility).model_dump()

    texts = [c["text"] for c in chunks]
    vectors = await step.run("embed-chunks", lambda: embed_texts(texts))

    def _store():
        store = get_qdrant_storage()
        if replace_existing:
            store.delete_source(source_id)
        return store.upsert_chunks(
            source_id,
            chunks,
            vectors,
            visibility=visibility,
            owner_id=owner_id,
            department=department,
            doc_id=source_id,
        )

    count = await step.run("store-qdrant", _store)
    return IngestResult(status="success", source_id=source_id, chunks_ingested=count, visibility=visibility).model_dump()


# ── FastAPI Backend Server ──────────────────────────────────────────────────
app = FastAPI(
    title="RAG Institute Assistant API",
    description="Production RAG pipeline with RBAC, OCR, Page Previews, and Answer Export",
    version="2.0.0",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        os.getenv("STREAMLIT_URL", "http://127.0.0.1:8501"),
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Clerk admin routes. There is intentionally no local password-login router.
app.include_router(clerk_admin_router)
app.include_router(clerk_webhook_router)


@app.get("/")
async def root(request: Request):
    clerk_params = [
        (key, value)
        for key, value in request.query_params.multi_items()
        if key.startswith("__clerk") or key.startswith("_clerk")
    ]
    if clerk_params:
        streamlit_url = os.getenv("STREAMLIT_URL", "http://127.0.0.1:8501").rstrip("/")
        return RedirectResponse(
            url=f"{streamlit_url}/?{urlencode(clerk_params)}",
            status_code=302,
        )
    return {
        "status": "online",
        "service": "RAG Institute Assistant API",
        "version": "2.0.0",
        "endpoints": ["/auth/clerk", "/ingest", "/upload", "/upload-batch", "/ask", "/sources", "/preview", "/export", "/admin/users", "/admin/roles", "/admin/logs", "/docs", "/health"],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/clerk")
async def clerk_health(request: Request):
    """Local-only Clerk connectivity probe with safe, actionable reason codes."""
    client_host = request.client.host if request.client else None
    if client_host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=403, detail="Local access only")
    try:
        check_connection()
    except ClerkConfigurationError as exc:
        logger.exception("Clerk health check configuration failure")
        return Response(
            content=json.dumps({"status": "failed", "reason": "server_misconfigured", "message": str(exc)}),
            status_code=500,
            media_type="application/json",
        )
    except ClerkAPIError as exc:
        logger.exception("Clerk health check failed: %s: %s", type(exc).__name__, str(exc))
        reason = "invalid_secret" if exc.reason == "invalid_secret" else "auth_service_unavailable"
        return Response(
            content=json.dumps({"status": "failed", "reason": reason}),
            status_code=503 if reason == "auth_service_unavailable" else 502,
            media_type="application/json",
        )
    return {"status": "ok", "reason": "ok"}


@app.get("/auth/clerk", response_class=HTMLResponse, include_in_schema=False)
async def clerk_sign_in(request: Request):
    publishable_key = os.getenv("CLERK_PUBLISHABLE_KEY", "")
    frontend_api_url = os.getenv("CLERK_ISSUER", "").rstrip("/")
    streamlit_url = os.getenv("STREAMLIT_URL", "http://127.0.0.1:8501")
    fastapi_url = os.getenv("FASTAPI_BASE_URL", str(request.base_url)).rstrip("/")
    auth_callback_url = f"{fastapi_url}/auth/clerk"
    template = os.getenv("CLERK_JWT_TEMPLATE", "rag-api")
    force_sign_out = request.query_params.get("force") == "1"
    if not publishable_key or not frontend_api_url:
        raise HTTPException(status_code=503, detail="Clerk publishable key and issuer are required")
    return HTMLResponse(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Sign in</title>
<script defer crossorigin="anonymous" src="{frontend_api_url}/npm/@clerk/ui@1/dist/ui.browser.js"></script>
<script defer crossorigin="anonymous" data-clerk-publishable-key="{publishable_key}" src="{frontend_api_url}/npm/@clerk/clerk-js@6/dist/clerk.browser.js"></script>
</head><body><div id="clerk-sign-in"></div><p id="status">Loading secure sign-in...</p>
<script>
const returnUrl = {streamlit_url!r};
const authCallbackUrl = {auth_callback_url!r};
const template = {template!r};
const forceSignOut = {str(force_sign_out).lower()};
window.addEventListener('load', async () => {{
  try {{
    await Clerk.load({{ ui: {{ ClerkUI: window.__internal_ClerkUICtor }} }});
    if (forceSignOut && Clerk.session) {{
      await Clerk.signOut();
    }}
    if (!Clerk.session) {{
      Clerk.mountSignIn(document.getElementById('clerk-sign-in'), {{
        signInForceRedirectUrl: authCallbackUrl,
        signInFallbackRedirectUrl: authCallbackUrl,
        signUpForceRedirectUrl: authCallbackUrl,
        signUpFallbackRedirectUrl: authCallbackUrl
      }});
      document.getElementById('status').textContent = 'Sign in with your institute account.';
      return;
    }}
    const token = await Clerk.session.getToken({{ template }});
    window.location.replace(returnUrl + '?clerk_token=' + encodeURIComponent(token));
  }} catch (error) {{
    document.getElementById('status').textContent = 'Unable to load Clerk sign-in: ' + error.message;
  }}
}});
</script></body></html>"""
    )


@app.post("/auth/logout")
async def clerk_logout(
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
    token: str = Depends(get_current_token),
):
    """Revoke both the Clerk session and this exact token before clearing the UI."""
    record_revoked_token(db, token, current_user.claims, current_user.id)
    session_id = current_user.claims.get("sid")
    if session_id:
        try:
            revoke_session(session_id)
        except Exception:
            # The local denylist still blocks this token if Clerk is temporarily unavailable.
            pass
    write_audit(db, current_user, "logout", session_id)
    return {"status": "logged_out"}


@app.get("/auth/me")
async def clerk_me(current_user: CurrentUser = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "role": current_user.role,
        "department": current_user.department,
    }


@app.get("/sources")
async def sources(
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    store = get_qdrant_storage()
    access_filter = build_access_filter(current_user)
    qdrant_sources = store.list_sources(access_filter=access_filter)
    db_sources = []
    for document in db.query(DocumentRecord).all():
        if document_is_accessible(current_user, document):
            db_sources.append(document.source_id)
    return {"sources": sorted(set(qdrant_sources + db_sources))}


@app.post("/ingest", response_model=IngestResult)
async def ingest(
    req: IngestPDFData,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_roles("admin", "faculty")),
) -> IngestResult:
    try:
        department = validate_ingest_scope(current_user, req.visibility, req.department)
        result = ingest_pdf_file(
            pdf_path=req.pdf_path,
            source_id=req.source_id,
            visibility=req.visibility,
            uploaded_by=current_user.id,
            replace_existing=req.replace_existing,
            owner_id=current_user.id,
            department=department,
            db=db,
        )
        write_audit(db, current_user, "ingest", req.source_id, f"visibility={req.visibility}; department={department}")
        return result
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err))
    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"PDF ingestion failed: {err}")


@app.post("/upload", response_model=IngestResult)
async def upload_file(
    file: UploadFile = File(...),
    source_id: Optional[str] = Form(None),
    visibility: str = Form("public"),
    department: Optional[str] = Form(None),
    replace_existing: bool = Form(True),
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_roles("admin", "faculty")),
) -> IngestResult:
    target_source = source_id or os.path.splitext(file.filename)[0]
    target_source = "".join(c if (c.isalnum() or c in "_-") else "_" for c in target_source)
    storage_path = os.path.join(STORAGE_DIR, f"{target_source}.pdf")

    try:
        department = validate_ingest_scope(current_user, visibility, department)
        content = await file.read()
        with open(storage_path, "wb") as f:
            f.write(content)

        result = ingest_pdf_file(
            pdf_path=storage_path,
            source_id=target_source,
            visibility=visibility,
            uploaded_by=current_user.id,
            replace_existing=replace_existing,
            owner_id=current_user.id,
            department=department,
            db=db,
        )
        write_audit(db, current_user, "ingest", target_source, f"visibility={visibility}; department={department}")
        return result
    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"File upload failed: {err}")


@app.post("/upload-batch")
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    visibility: str = Form("public"),
    replace_existing: bool = Form(True),
    department: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_roles("admin", "faculty")),
) -> Dict[str, Any]:
    results = []
    total_chunks = 0
    for file in files:
        raw_name = os.path.splitext(file.filename)[0]
        target_source = "".join(c if (c.isalnum() or c in "_-") else "_" for c in raw_name)
        storage_path = os.path.join(STORAGE_DIR, f"{target_source}.pdf")
        try:
            department = validate_ingest_scope(current_user, visibility, department)
            content = await file.read()
            with open(storage_path, "wb") as f:
                f.write(content)

            res = ingest_pdf_file(
                pdf_path=storage_path,
                source_id=target_source,
                visibility=visibility,
                uploaded_by=current_user.id,
                replace_existing=replace_existing,
                owner_id=current_user.id,
                department=department,
                db=db,
            )
            write_audit(db, current_user, "ingest", target_source, f"visibility={visibility}; department={department}")
            results.append(res.model_dump())
            total_chunks += res.chunks_ingested
        except Exception as err:
            results.append({"status": "failed", "source_id": file.filename, "error": str(err), "chunks_ingested": 0})

    return {
        "status": "completed",
        "total_files": len(files),
        "total_chunks_ingested": total_chunks,
        "details": results,
    }


@app.delete("/documents/{source_id}")
async def delete_document(
    source_id: str,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_roles("admin", "faculty")),
):
    document = db.query(DocumentRecord).filter(DocumentRecord.source_id == source_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if not document_is_manageable(current_user, document):
        raise HTTPException(status_code=403, detail="You can only delete documents within your ownership or department scope")
    get_qdrant_storage().delete_source(source_id)
    if document.file_path and os.path.exists(document.file_path):
        os.unlink(document.file_path)
    db.delete(document)
    db.commit()
    write_audit(db, current_user, "delete", source_id)
    return {"status": "deleted", "source_id": source_id}


@app.post("/ask", response_model=AskResponse)
async def ask(
    req: AskRequest,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
) -> AskResponse:
    # 1. Embed query
    query_vecs = embed_texts([req.question])
    if not query_vecs:
        raise HTTPException(status_code=400, detail="Failed to generate query embedding.")
    query_vec = query_vecs[0]

    raw_source = getattr(req, "source_id", None)
    source_id = raw_source.strip() if raw_source and isinstance(raw_source, str) else None

    if source_id and source_id != "All documents":
        doc_record = db.query(DocumentRecord).filter(DocumentRecord.source_id == source_id).first()
        if doc_record and not document_is_accessible(current_user, doc_record):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: Document '{source_id}' is outside your role or department scope.",
            )

    # 3. Query Qdrant
    store = get_qdrant_storage()
    found = store.search(
        query_vector=query_vec,
        top_k=req.top_k,
        source_id=source_id,
        access_filter=build_access_filter(current_user),
    )

    contexts = found.get("contexts", [])
    sources_list = found.get("sources", [])
    matches = found.get("matches", [])

    # Build structured citations with page numbers and OCR badges
    citations: List[CitationItem] = []
    for idx, match in enumerate(matches, start=1):
        citations.append(
            CitationItem(
                index=idx,
                source_id=match.get("source_id", "unknown"),
                page_number=match.get("page_number", 1),
                ocr_used=match.get("ocr_used", False),
                text=match.get("text", "")[:350],
            )
        )

    # Log query to database
    enforce_query_rate_limit(db, current_user)
    user_identifier = current_user.id
    user_role_str = current_user.role
    try:
        q_log = QueryLog(
            user_id=user_identifier,
            user_role=user_role_str,
            question=req.question,
            source_id=source_id,
            num_contexts=len(contexts),
        )
        db.add(q_log)
        db.commit()
    except Exception:
        pass

    # 4. Synthesize LLM answer via OpenRouter
    context_block = build_context_block(matches)
    messages = build_chat_messages(req.question, context_block, bool(contexts))

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_openrouter_api_key_here":
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY environment variable is missing or unconfigured in .env",
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=30.0,
        default_headers={
            "HTTP-Referer": "http://127.0.0.1:8000",
            "X-Title": "RAG Institute Assistant",
        },
    )

    answer = None
    last_error = None
    models_to_try = get_openrouter_models()

    for model_name in models_to_try:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                max_tokens=1600,
                temperature=0.3,
                messages=messages,
                timeout=25.0,
            )

            choice = completion.choices[0] if completion.choices else None
            if choice and choice.message:
                raw_text = choice.message.content or ""
                if not raw_text.strip() and hasattr(choice.message, "reasoning"):
                    raw_text = getattr(choice.message, "reasoning") or ""
                if raw_text.strip():
                    answer = raw_text.strip()
                    break
        except Exception as err:
            last_error = str(err)
            continue

    # Fallback to dynamically discovered free models if all static models failed
    if not answer:
        try:
            import requests
            resp = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            if resp.status_code == 200:
                available = resp.json().get("data", [])
                dynamic_free = [
                    m["id"] for m in available
                    if ":free" in m.get("id", "") and m.get("id") not in models_to_try
                ]
                for model_name in dynamic_free[:5]:
                    try:
                        completion = client.chat.completions.create(
                            model=model_name,
                            max_tokens=1600,
                            temperature=0.3,
                            messages=messages,
                            timeout=25.0,
                        )
                        choice = completion.choices[0] if completion.choices else None
                        if choice and choice.message:
                            raw_text = choice.message.content or ""
                            if raw_text.strip():
                                answer = raw_text.strip()
                                break
                    except Exception as err:
                        last_error = str(err)
                        continue
        except Exception:
            pass

    if not answer:
        if last_error:
            raise HTTPException(
                status_code=502,
                detail=f"OpenRouter LLM completion failed across free models: {last_error}",
            )
        answer = "The LLM model returned an empty response. Please try asking again."

    return AskResponse(
        answer=answer,
        sources=sources_list,
        num_contexts=len(contexts),
        citations=citations,
    )


# ── Phase 3: Source Document Page Preview ───────────────────────────────────
@app.get("/preview/{source_id}/{page_number}")
async def preview_page(
    source_id: str,
    page_number: int,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Renders a high-resolution PNG image of the requested PDF page with RBAC validation."""
    doc_record = db.query(DocumentRecord).filter(DocumentRecord.source_id == source_id).first()
    
    # Check visibility permissions
    if doc_record and not document_is_accessible(current_user, doc_record):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: document is outside your role or department scope",
        )

    pdf_path = os.path.join(STORAGE_DIR, f"{source_id}.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"Source PDF file '{source_id}.pdf' not found in storage.")

    if pymupdf is None:
        raise HTTPException(status_code=500, detail="PyMuPDF is not available on server for image rendering.")

    try:
        doc = pymupdf.open(pdf_path)
        if page_number < 1 or page_number > len(doc):
            doc.close()
            raise HTTPException(status_code=400, detail=f"Page number {page_number} is out of bounds (1-{len(doc)}).")

        page = doc[page_number - 1]
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        doc.close()
        return Response(content=img_bytes, media_type="image/png")
    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to render PDF page image: {err}")


# ── Phase 4: Download / Export Answers as PDF ────────────────────────────────
@app.post("/export")
async def export_answer_pdf(
    req: ExportRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Generates a professional PDF report containing question, grounded answer, and citations."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ExportTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )
    meta_style = ParagraphStyle(
        "ExportMeta",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=14,
    )
    h2_style = ParagraphStyle(
        "ExportH2",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ExportBody",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
    )

    story = []

    # Header
    story.append(Paragraph("📄 RAG Institute Assistant — Q&A Report", title_style))
    user_email = current_user.email or current_user.id
    timestamp_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    story.append(Paragraph(f"Exported for: <b>{user_email}</b> • Generated on: <b>{timestamp_str}</b>", meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0"), spaceAfter=12))

    # Question Box
    story.append(Paragraph("<b>Question:</b>", h2_style))
    story.append(Paragraph(req.question.replace("\n", "<br/>"), body_style))
    story.append(Spacer(1, 10))

    # Answer Box
    story.append(Paragraph("<b>Grounded Answer:</b>", h2_style))
    formatted_answer = req.answer.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
    story.append(Paragraph(formatted_answer, body_style))
    story.append(Spacer(1, 14))

    # Citations Table
    if req.citations:
        story.append(Paragraph("<b>Retrieved Source Citations:</b>", h2_style))
        table_rows = [
            [
                Paragraph("<b>#</b>", body_style),
                Paragraph("<b>Source Document</b>", body_style),
                Paragraph("<b>Page</b>", body_style),
                Paragraph("<b>Extraction Method</b>", body_style),
            ]
        ]
        for idx, cite in enumerate(req.citations, start=1):
            source = str(cite.get("source_id", "N/A"))
            page_str = f"Page {cite.get('page_number', 1)}"
            ocr_str = "OCR Extracted" if cite.get("ocr_used") else "Native Text"
            table_rows.append([
                Paragraph(str(idx), body_style),
                Paragraph(source, body_style),
                Paragraph(page_str, body_style),
                Paragraph(ocr_str, body_style),
            ])

        t = Table(table_rows, colWidths=[30, 260, 90, 140])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ]))
        story.append(t)

    doc.build(story)
    buffer.seek(0)

    filename = f"rag_answer_{int(time.time())}.pdf"
    return Response(
        content=buffer.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── Phase 1: Admin Analytics Dashboard ──────────────────────────────────────
@app.get("/admin/dashboard")
async def admin_dashboard(
    db: Session = Depends(get_db),
    admin: CurrentUser = Depends(require_roles("admin")),
):
    """Admin-only analytics dashboard aggregating query volume, sources, and user activity."""
    total_queries = db.query(QueryLog).count()
    try:
        total_users = len(list_users(limit=100))
    except Exception:
        total_users = 0
    total_documents = db.query(DocumentRecord).count()
    public_documents = db.query(DocumentRecord).filter(DocumentRecord.visibility == "public").count()
    private_documents = db.query(DocumentRecord).filter(DocumentRecord.visibility == "private").count()

    # Most queried sources
    source_stats = (
        db.query(QueryLog.source_id, func.count(QueryLog.id).label("count"))
        .filter(QueryLog.source_id.isnot(None))
        .group_by(QueryLog.source_id)
        .order_by(func.count(QueryLog.id).desc())
        .limit(10)
        .all()
    )
    popular_sources = [{"source_id": s[0], "query_count": s[1]} for s in source_stats]

    # Queries per role
    role_stats = (
        db.query(QueryLog.user_role, func.count(QueryLog.id).label("count"))
        .group_by(QueryLog.user_role)
        .all()
    )
    role_breakdown = {r[0] or "unknown": r[1] for r in role_stats}

    # Recent queries
    recent_logs = (
        db.query(QueryLog)
        .order_by(QueryLog.timestamp.desc())
        .limit(15)
        .all()
    )
    recent_queries = [
        {
            "id": l.id,
            "user": l.user_id,
            "role": l.user_role,
            "question": l.question,
            "source_id": l.source_id,
            "num_contexts": l.num_contexts,
            "timestamp": l.timestamp.isoformat(),
        }
        for l in recent_logs
    ]

    return {
        "summary": {
            "total_queries": total_queries,
            "total_users": total_users,
            "total_documents": total_documents,
            "public_documents": public_documents,
            "private_documents": private_documents,
        },
        "popular_sources": popular_sources,
        "role_breakdown": role_breakdown,
        "recent_queries": recent_queries,
    }


# Mount Inngest endpoint
try:
    if not os.getenv("INNGEST_SIGNING_KEY"):
        os.environ["INNGEST_DEV"] = "1"
    inngest.fast_api.serve(app, inngest_client, [ingest_pdf])
except Exception:
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
