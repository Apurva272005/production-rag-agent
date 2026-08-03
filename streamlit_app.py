import os
import tempfile
import requests
import streamlit as st

# ── Config & Environment ──────────────────────────────────────────────────
INNGEST_DEV_URL = os.getenv("INNGEST_DEV_EVENT_URL", "http://127.0.0.1:8288/e/local")
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="RAG Institute Assistant",
    page_icon="📄",
    layout="centered",
)


def send_inngest_event(event_name: str, data: dict) -> dict:
    """Send an event to the local Inngest dev server."""
    payload = [{"name": event_name, "data": data}]
    try:
        resp = requests.post(INNGEST_DEV_URL, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        payload = {"name": event_name, "data": data}
        resp = requests.post(INNGEST_DEV_URL, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()


st.title("📄 RAG Institute Assistant")
st.caption("Upload documents via LlamaIndex & Inngest, then ask questions grounded in vector context.")

tab_ingest, tab_ask = st.tabs(["📥 Ingest Document", "💬 Ask a Question"])

# ── Tab 1: Ingest ───────────────────────────────────────────────────────────
with tab_ingest:
    st.subheader("Upload a PDF Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    source_id = st.text_input(
        "Source ID (unique name for this document)",
        placeholder="e.g., attendance_policy_2026",
    )

    if st.button("Ingest Document", type="primary", disabled=not (uploaded_file and source_id)):
        with st.spinner("Saving file and triggering Inngest pipeline..."):
            try:
                # Save uploaded file to a temporary location for background worker reading
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                send_inngest_event(
                    "rag/ingest_pdf",
                    {"pdf_path": tmp_path, "source_id": source_id},
                )
                st.success(
                    f"Ingestion event triggered for '{source_id}'. "
                    "Track progress in the Inngest Dev Dashboard (http://127.0.0.1:8288)."
                )
            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not reach the Inngest dev server at "
                    f"{INNGEST_DEV_URL}. Make sure it is running (`npx inngest-cli@latest dev`)."
                )
            except Exception as e:
                st.error(f"Ingestion trigger failed: {e}")

    st.divider()
    st.caption(
        "Note: Ingestion is throttled (max 2/min) and rate-limited per source ID "
        "(1 per 4 hours) to protect server resources."
    )

# ── Tab 2: Ask ──────────────────────────────────────────────────────────────
with tab_ask:
    st.subheader("Ask a Question")
    question = st.text_input("Your question", placeholder="e.g., What is the attendance policy?")
    top_k = st.slider("Number of context chunks to retrieve (top_k)", min_value=1, max_value=10, value=5)

    if st.button("Ask", type="primary", disabled=not question):
        with st.spinner("Searching Qdrant vector index and generating answer via OpenRouter..."):
            try:
                resp = requests.post(
                    f"{FASTAPI_BASE_URL}/ask",
                    json={"question": question, "top_k": top_k},
                    timeout=60,
                )
                resp.raise_for_status()
                result = resp.json()

                st.markdown("### Answer")
                st.write(result.get("answer", "No answer returned."))

                sources = result.get("sources", [])
                num_contexts = result.get("num_contexts", 0)

                with st.expander(f"Sources & Attribution ({num_contexts} chunks used)"):
                    if sources:
                        for src in sources:
                            st.write(f"- `{src}`")
                    else:
                        st.write("No matching document sources found.")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not reach the FastAPI backend. "
                    f"Make sure it is running (`uv run python main.py` or `uvicorn main:app --reload`)."
                )
            except Exception as e:
                st.error(f"Query failed: {e}")

    st.divider()
    st.caption("Answers are generated strictly from ingested document content using Llama 3.1 8B Instruct.")
