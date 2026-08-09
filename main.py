import os
import tempfile
from datetime import timedelta
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import inngest
import inngest.fast_api

from custom_types import AskRequest, AskResponse, IngestPDFData, IngestResult
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import get_qdrant_storage

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

DEFAULT_OPENROUTER_MODELS = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "google/gemma-4-26b-a4b-it:free",
    "openai/gpt-oss-20b:free",
]


def get_openrouter_models() -> list[str]:
    configured_model = os.getenv("OPENROUTER_MODEL", "").strip()
    if configured_model:
        return [configured_model, *[model for model in DEFAULT_OPENROUTER_MODELS if model != configured_model]]
    return DEFAULT_OPENROUTER_MODELS


def build_context_block(matches: list[dict[str, Any]]) -> str:
    if not matches:
        return "No relevant context was found in the ingested documents."

    context_items = []
    for index, match in enumerate(matches, start=1):
        source_id = match.get("source_id", "unknown")
        text = match.get("text", "").strip()
        if text:
            context_items.append(f"[{index}] Source: {source_id}\n{text}")
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


def ingest_pdf_file(pdf_path: str, source_id: str, replace_existing: bool = True) -> IngestResult:
    chunks = load_and_chunk_pdf(pdf_path)

    if not chunks:
        return IngestResult(status="empty", source_id=source_id, chunks_ingested=0)

    vectors = embed_texts(chunks)
    store = get_qdrant_storage()
    if replace_existing:
        store.delete_source(source_id)
    count = store.upsert_chunks(source_id, chunks, vectors)

    return IngestResult(status="success", source_id=source_id, chunks_ingested=count)


# ── Inngest Client & Orchestrated Ingestion Workflow ────────────────────────
inngest_client = inngest.Inngest(app_id="rag_institute_assistant")


@inngest_client.create_function(
    fn_id="ingest-pdf",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(limit=2, period=timedelta(seconds=60)),
    rate_limit=inngest.RateLimit(limit=1, period=timedelta(hours=4), key="event.data.source_id"),
)
async def ingest_pdf(ctx: inngest.Context, step: inngest.Step) -> Dict[str, Any]:
    pdf_path = ctx.event.data.get("pdf_path")
    source_id = ctx.event.data.get("source_id")

    if not pdf_path or not source_id:
        raise ValueError("Missing pdf_path or source_id in event data")

    replace_existing = ctx.event.data.get("replace_existing", True)

    # Step 1: Read and chunk PDF via LlamaIndex
    chunks = await step.run("extract-chunks", lambda: load_and_chunk_pdf(pdf_path))

    if not chunks:
        return IngestResult(status="empty", source_id=source_id, chunks_ingested=0).model_dump()

    # Step 2: Compute local HuggingFace embeddings (bge-small-en-v1.5)
    vectors = await step.run("embed-chunks", lambda: embed_texts(chunks))

    # Step 3: Store vectors in Qdrant Vector Store
    def _store():
        store = get_qdrant_storage()
        if replace_existing:
            store.delete_source(source_id)
        return store.upsert_chunks(source_id, chunks, vectors)

    count = await step.run("store-qdrant", _store)

    return IngestResult(status="success", source_id=source_id, chunks_ingested=count).model_dump()


# ── FastAPI Backend Server ──────────────────────────────────────────────────
app = FastAPI(
    title="RAG Institute Assistant API",
    description="Production-ready RAG pipeline using LlamaIndex, Inngest, Qdrant, FastEmbed, and OpenRouter",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "RAG Institute Assistant API",
        "endpoints": ["/ingest", "/ask", "/sources", "/api/inngest", "/docs", "/health"],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ingest")
async def get_ingest_info():
    return {
        "message": "The /ingest endpoint expects HTTP POST requests with JSON payload.",
        "expected_payload": {"pdf_path": "/path/to/file.pdf", "source_id": "doc_name", "replace_existing": True},
        "streamlit_ui": "http://localhost:8501",
    }


@app.post("/ingest", response_model=IngestResult)
async def ingest(req: IngestPDFData) -> IngestResult:
    try:
        target_replace = getattr(req, "replace_existing", True)
        return ingest_pdf_file(req.pdf_path, req.source_id, target_replace)
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err))
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"PDF ingestion failed: {err}")


@app.post("/upload", response_model=IngestResult)
async def upload_file(
    file: UploadFile = File(...),
    source_id: Optional[str] = Form(None),
    replace_existing: bool = Form(True),
) -> IngestResult:
    temp_path = None
    try:
        target_source = source_id or os.path.splitext(file.filename)[0]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        return ingest_pdf_file(temp_path, target_source, replace_existing)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"File upload failed: {err}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


@app.get("/sources")
async def sources():
    store = get_qdrant_storage()
    return {"sources": store.list_sources()}


@app.get("/ask")
async def get_ask_info():
    return {
        "message": "The /ask endpoint expects HTTP POST requests with JSON payload.",
        "expected_payload": {"question": "Your question here", "top_k": 5, "source_id": "optional_source_id"},
        "streamlit_ui": "http://localhost:8501",
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    # 1. Embed user query locally
    query_vecs = embed_texts([req.question])
    if not query_vecs:
        raise HTTPException(status_code=400, detail="Failed to generate query embedding.")
    query_vec = query_vecs[0]

    # 2. Query Qdrant vector database
    store = get_qdrant_storage()
    raw_source = getattr(req, "source_id", None)
    source_id = raw_source.strip() if raw_source and isinstance(raw_source, str) else None
    found = store.search(query_vec, req.top_k, source_id=source_id)

    contexts = found.get("contexts", [])
    sources_list = found.get("sources", [])
    matches = found.get("matches", [])
    context_block = build_context_block(matches)
    messages = build_chat_messages(req.question, context_block, bool(contexts))

    # 3. Call OpenRouter Free-Tier LLM via OpenAI-compatible client
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

    for model_name in get_openrouter_models():
        try:
            completion = client.chat.completions.create(
                model=model_name,
                max_tokens=1600,
                temperature=0.3,
                messages=messages,
                timeout=30.0,
            )

            if completion.choices and completion.choices[0].message.content:
                answer = completion.choices[0].message.content.strip()
                break
        except Exception as err:
            last_error = str(err)
            continue

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
    )


# Mount Inngest endpoint
# Enable Inngest local dev mode if no production signing key is provided
if not os.getenv("INNGEST_SIGNING_KEY"):
    os.environ["INNGEST_DEV"] = "1"

inngest.fast_api.serve(app, inngest_client, [ingest_pdf])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
