import os
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
import shutil
import io
import time
from datetime import timedelta, datetime
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Response, status
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

from database import get_db, SessionLocal, Base, engine
from models_db import User, DocumentRecord, QueryLog
from auth import (
    get_current_user,
    get_optional_current_user,
    require_role,
    init_default_users,
)
from routers import auth_routes
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

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

# Initialize Database tables and default accounts
Base.metadata.create_all(bind=engine)
with SessionLocal() as init_session:
    init_default_users(init_session)

DEFAULT_OPENROUTER_MODELS = [
    "minimax/minimax-m3:free",
    "poolside/laguna-s-2.1:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
]


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


def ingest_pdf_file(
    pdf_path: str,
    source_id: str,
    visibility: str = "public",
    uploaded_by: str = "system",
    replace_existing: bool = True,
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
                file_path=dest_path,
                chunks_count=count,
            )
            db.add(doc_record)
        else:
            doc_record.visibility = visibility
            doc_record.uploaded_by = uploaded_by
            doc_record.file_path = dest_path
            doc_record.chunks_count = count
            doc_record.uploaded_at = datetime.utcnow()
        db.commit()
    finally:
        if should_close_db:
            db.close()

    return IngestResult(status="success", source_id=source_id, chunks_ingested=count, visibility=visibility)


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

    if not pdf_path or not source_id:
        raise ValueError("Missing pdf_path or source_id in event data")

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
        return store.upsert_chunks(source_id, chunks, vectors, visibility=visibility)

    count = await step.run("store-qdrant", _store)
    return IngestResult(status="success", source_id=source_id, chunks_ingested=count, visibility=visibility).model_dump()


# ── FastAPI Backend Server ──────────────────────────────────────────────────
app = FastAPI(
    title="RAG Institute Assistant API",
    description="Production RAG pipeline with RBAC, OCR, Page Previews, and Answer Export",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Auth router
app.include_router(auth_routes.router)


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "RAG Institute Assistant API",
        "version": "2.0.0",
        "endpoints": ["/auth/login", "/ingest", "/upload", "/upload-batch", "/ask", "/sources", "/preview", "/export", "/admin/dashboard", "/docs", "/health"],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/sources")
async def sources(
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    store = get_qdrant_storage()
    is_restricted = (current_user is None or current_user.role == "student")
    
    if is_restricted:
        public_docs = db.query(DocumentRecord.source_id).filter(DocumentRecord.visibility == "public").all()
        public_ids = [d[0] for d in public_docs]
        qdrant_sources = store.list_sources(visibility="public")
        all_public = sorted(list(set(public_ids + qdrant_sources)))
        return {"sources": all_public}

    all_docs = db.query(DocumentRecord.source_id).all()
    db_ids = [d[0] for d in all_docs]
    qdrant_sources = store.list_sources()
    return {"sources": sorted(list(set(db_ids + qdrant_sources)))}


@app.post("/ingest", response_model=IngestResult)
async def ingest(
    req: IngestPDFData,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin", "faculty")),
) -> IngestResult:
    try:
        return ingest_pdf_file(
            pdf_path=req.pdf_path,
            source_id=req.source_id,
            visibility=req.visibility,
            uploaded_by=current_user.email,
            replace_existing=req.replace_existing,
            db=db,
        )
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err))
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"PDF ingestion failed: {err}")


@app.post("/upload", response_model=IngestResult)
async def upload_file(
    file: UploadFile = File(...),
    source_id: Optional[str] = Form(None),
    visibility: str = Form("public"),
    replace_existing: bool = Form(True),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin", "faculty")),
) -> IngestResult:
    target_source = source_id or os.path.splitext(file.filename)[0]
    target_source = "".join(c if (c.isalnum() or c in "_-") else "_" for c in target_source)
    storage_path = os.path.join(STORAGE_DIR, f"{target_source}.pdf")

    try:
        content = await file.read()
        with open(storage_path, "wb") as f:
            f.write(content)

        return ingest_pdf_file(
            pdf_path=storage_path,
            source_id=target_source,
            visibility=visibility,
            uploaded_by=current_user.email,
            replace_existing=replace_existing,
            db=db,
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"File upload failed: {err}")


@app.post("/upload-batch")
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    visibility: str = Form("public"),
    replace_existing: bool = Form(True),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin", "faculty")),
) -> Dict[str, Any]:
    results = []
    total_chunks = 0
    for file in files:
        raw_name = os.path.splitext(file.filename)[0]
        target_source = "".join(c if (c.isalnum() or c in "_-") else "_" for c in raw_name)
        storage_path = os.path.join(STORAGE_DIR, f"{target_source}.pdf")
        try:
            content = await file.read()
            with open(storage_path, "wb") as f:
                f.write(content)

            res = ingest_pdf_file(
                pdf_path=storage_path,
                source_id=target_source,
                visibility=visibility,
                uploaded_by=current_user.email,
                replace_existing=replace_existing,
                db=db,
            )
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


@app.post("/ask", response_model=AskResponse)
async def ask(
    req: AskRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
) -> AskResponse:
    # 1. Embed query
    query_vecs = embed_texts([req.question])
    if not query_vecs:
        raise HTTPException(status_code=400, detail="Failed to generate query embedding.")
    query_vec = query_vecs[0]

    # 2. RBAC Visibility Filter: Students MUST only retrieve public chunks server-side
    visibility_filter = None
    if current_user is None or current_user.role == "student":
        visibility_filter = "public"

    raw_source = getattr(req, "source_id", None)
    source_id = raw_source.strip() if raw_source and isinstance(raw_source, str) else None

    # If student explicitly asked for a private source, block it server-side
    if visibility_filter == "public" and source_id and source_id != "All documents":
        doc_record = db.query(DocumentRecord).filter(DocumentRecord.source_id == source_id).first()
        if doc_record and doc_record.visibility == "private":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: Document '{source_id}' is private and restricted from student role.",
            )

    # 3. Query Qdrant
    store = get_qdrant_storage()
    found = store.search(
        query_vector=query_vec,
        top_k=req.top_k,
        source_id=source_id,
        visibility=visibility_filter,
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
    user_identifier = current_user.email if current_user else "anonymous"
    user_role_str = current_user.role if current_user else "guest"
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
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """Renders a high-resolution PNG image of the requested PDF page with RBAC validation."""
    doc_record = db.query(DocumentRecord).filter(DocumentRecord.source_id == source_id).first()
    
    # Check visibility permissions
    if doc_record and doc_record.visibility == "private":
        if current_user is None or current_user.role == "student":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access forbidden: You do not have permission to view pages of this private document.",
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
    current_user: Optional[User] = Depends(get_optional_current_user),
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
    user_email = current_user.email if current_user else "Authenticated User"
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
    admin: User = Depends(require_role("admin")),
):
    """Admin-only analytics dashboard aggregating query volume, sources, and user activity."""
    total_queries = db.query(QueryLog).count()
    total_users = db.query(User).count()
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
