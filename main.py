import os
from datetime import timedelta
from typing import Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import inngest
import inngest.fast_api

from custom_types import AskRequest, AskResponse, IngestResult
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import get_qdrant_storage

load_dotenv()

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

    # Step 1: Read and chunk PDF via LlamaIndex
    chunks = await step.run("extract-chunks", lambda: load_and_chunk_pdf(pdf_path))

    if not chunks:
        return IngestResult(status="empty", source_id=source_id, chunks_ingested=0).model_dump()

    # Step 2: Compute local HuggingFace embeddings (bge-small-en-v1.5)
    vectors = await step.run("embed-chunks", lambda: embed_texts(chunks))

    # Step 3: Store vectors in Qdrant Vector Store
    def _store():
        store = get_qdrant_storage()
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
        "endpoints": ["/ask", "/api/inngest", "/docs", "/health"],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    # 1. Embed user query locally
    query_vecs = embed_texts([req.question])
    if not query_vecs:
        raise HTTPException(status_code=400, detail="Failed to generate query embedding.")
    query_vec = query_vecs[0]

    # 2. Query Qdrant vector database
    store = get_qdrant_storage()
    found = store.search(query_vec, req.top_k)

    contexts = found.get("contexts", [])
    sources = found.get("sources", [])

    if not contexts:
        context_block = "No relevant context found in ingested documents."
    else:
        context_block = "\n\n".join(f"- {c}" for c in contexts)

    user_content = (
        "Use the following retrieved context to answer the user's question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {req.question}\n"
        "Answer concisely using only the context provided above."
    )

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
        default_headers={
            "HTTP-Referer": "http://127.0.0.1:8000",
            "X-Title": "RAG Institute Assistant",
        },
    )

    free_models = [
        "meta-llama/llama-3.1-8b-instruct:free",
        "google/gemma-2-9b-it:free",
        "mistralai/mistral-7b-instruct:free",
        "qwen/qwen-2.5-7b-instruct:free",
    ]

    answer = None
    last_error = None

    for model_name in free_models:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                max_tokens=1024,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions using only provided document context."},
                    {"role": "user", "content": user_content},
                ],
            )

            if completion.choices and completion.choices[0].message.content:
                answer = completion.choices[0].message.content.strip()
                break
        except Exception as err:
            last_error = str(err)
            continue

    if not answer:
        if last_error:
            raise HTTPException(status_code=502, detail=f"OpenRouter LLM completion failed across free models: {last_error}")
        answer = "The LLM model returned an empty response. Please try asking again."

    return AskResponse(
        answer=answer,
        sources=sources,
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


