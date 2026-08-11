# RAG Institute Assistant

Lightweight RAG (Retrieval-Augmented Generation) assistant using `FastAPI`, `Inngest`, `Qdrant`, local embedding fallbacks, and `Streamlit` for a simple UI.

This repo includes safe fallbacks so tests and local development work without downloading large models. To enable full model-backed behavior, install the optional packages listed in `requirements.txt`.

**Key files**
- `main.py` — FastAPI backend and Inngest functions
- `app.py` / `streamlit_app.py` — Streamlit UI
- `data_loader.py` — embedding & PDF chunking (uses `fastembed`/`llama-index` if available, otherwise deterministic fallback)
- `vector_db.py` — Qdrant wrapper with in-memory fallback for tests

## Setup

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Install heavier ML dependencies to enable real embeddings:

```bash
# Example (may require GPU/extra disk):
pip install llama-index fastembed transformers torch
```

## Environment variables
Create a `.env` file at the repo root or export these in your shell:

- `OPENROUTER_API_KEY` — required for the LLM completion calls.
- `QDRANT_URL` — optional: Qdrant Cloud URL (if using remote Qdrant).
- `QDRANT_API_KEY` — optional: API key for Qdrant Cloud.
- `QDRANT_LOCAL_PATH` — optional: local path for qdrant persistence (default `./qdrant_db`).
- `INNGEST_SIGNING_KEY` — optional: when set, Inngest will not run in dev mode.
- `INNGEST_DEV_EVENT_URL` — optional: URL for the Inngest dev event endpoint (default `http://127.0.0.1:8288/e/local`).
- `FASTAPI_BASE_URL` — optional: used by the Streamlit UI (default `http://127.0.0.1:8000`).

## Run locally

1. Start the FastAPI app (development):

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

2. (Optional) Start Inngest dev server for event-driven ingestion:

```bash
npx inngest-cli@latest dev
```

3. Start the Streamlit UI:

```bash
streamlit run app.py
```

Now open: `http://localhost:8501` to use the UI.

## Docker (quick local Qdrant)

If you want to run a local Qdrant instance via Docker, use the provided `docker-compose.yml`:

```bash
docker compose up -d qdrant
```

This starts Qdrant on `http://localhost:6333` and persists data to a Docker volume named `qdrant_data`.

After Qdrant is up, set `QDRANT_URL`/`QDRANT_API_KEY` (if using cloud) or ensure `QDRANT_LOCAL_PATH` is set if you prefer a local on-disk path.

## Example `.env`

Copy `.env.example` to `.env` and fill in values:

```bash
cp .env.example .env
# edit .env with your API keys
```

## Tests

Run the unit tests:

```bash
pytest -q
```

## Notes
- `data_loader.py` provides deterministic hashed embeddings when `fastembed` is not installed so development and CI are deterministic and fast.
- `vector_db.py` will use the installed `qdrant-client` when available; otherwise it uses an in-memory store for upsert/search so the app and tests still function without a running Qdrant server.

If you'd like, I can update `requirements.txt` to pin versions or add a `docker-compose.yml` for a full reproducible local stack.
