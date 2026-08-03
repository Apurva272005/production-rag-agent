import os
import time
import tempfile
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────
INNGEST_EVENT_KEY = os.getenv("INNGEST_EVENT_KEY", "")  # optional, for prod Inngest
INNGEST_DEV_URL = os.getenv("INNGEST_DEV_EVENT_URL", "http://127.0.0.1:8288/e/local")
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Institute Assistant", page_icon="📄", layout="centered")


def send_event(event_name: str, data: dict) -> dict:
    """Send an event to the local Inngest dev server."""
    # Try sending array of events first (Inngest standard)
    payload = [{"name": event_name, "data": data}]
    try:
        resp = requests.post(INNGEST_DEV_URL, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        # Fallback to single object format
        payload = {"name": event_name, "data": data}
        resp = requests.post(INNGEST_DEV_URL, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()



st.title("📄 RAG Institute Assistant")
st.caption("Upload documents, then ask questions grounded in their content.")

tab_ingest, tab_ask = st.tabs(["📥 Ingest Document", "💬 Ask a Question"])

# ── Tab 1: Ingest ───────────────────────────────────────────────────────────
with tab_ingest:
    st.subheader("Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    source_id = st.text_input(
        "Source ID (unique name for this document)",
        placeholder="e.g., attendance_policy_2026",
    )

    if st.button("Ingest Document", type="primary", disabled=not (uploaded_file and source_id)):
        with st.spinner("Saving file and triggering ingestion..."):
            try:
                # Save uploaded file to a temp path the backend can read
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                send_event(
                    "rag/ingest_pdf",
                    {"pdf_path": tmp_path, "source_id": source_id},
                )
                st.success(
                    f"Ingestion triggered for '{source_id}'. "
                    "Check the Inngest dev server dashboard for progress/status."
                )
            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not reach the Inngest dev server. "
                    "Make sure it's running (`npx inngest-cli@latest dev`) "
                    "and your FastAPI app is up (`uv run uvicorn main:app --reload`)."
                )
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()
    st.caption(
        "Note: ingestion is throttled (max 2/min) and rate-limited per source "
        "(1 per 4 hours) to protect the pipeline from overload."
    )

# ── Tab 2: Ask ──────────────────────────────────────────────────────────────
with tab_ask:
    st.subheader("Ask a question")
    question = st.text_input("Your question", placeholder="e.g., What is the attendance policy for CE363?")
    top_k = st.slider("Number of context chunks to retrieve (top_k)", min_value=1, max_value=10, value=5)

    if st.button("Ask", type="primary", disabled=not question):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                # Calls a synchronous FastAPI endpoint (see /ask in main.py)
                # so the UI gets an immediate answer instead of firing an
                # Inngest event and waiting blindly.
                resp = requests.post(
                    f"{FASTAPI_BASE_URL}/ask",
                    json={"question": question, "top_k": top_k},
                    timeout=60,
                )
                resp.raise_for_status()
                result = resp.json()

                st.markdown("### Answer")
                st.write(result["answer"])

                with st.expander(f"Sources ({result['num_contexts']} chunks used)"):
                    for src in result["sources"]:
                        st.write(f"- {src}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not reach the FastAPI backend. "
                    "Make sure it's running (`uv run uvicorn main:app --reload`)."
                )
            except Exception as e:
                st.error(f"Query failed: {e}")

    st.divider()
    st.caption("Answers are generated strictly from ingested document content, with source attribution.")
