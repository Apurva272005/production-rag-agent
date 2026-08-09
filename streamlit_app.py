import os
import re
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


def clean_source_id(name: str) -> str:
    """Sanitize filename into a clean Source ID string."""
    base = os.path.splitext(name)[0]
    return re.sub(r"[^a-zA-Z0-9_-]", "_", base).strip("_")


def ingest_document(pdf_path: str, source_id: str) -> dict:
    resp = requests.post(
        f"{FASTAPI_BASE_URL}/ingest",
        json={"pdf_path": pdf_path, "source_id": source_id, "replace_existing": True},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_sources() -> list[str]:
    try:
        resp = requests.get(f"{FASTAPI_BASE_URL}/sources", timeout=10)
        resp.raise_for_status()
        return resp.json().get("sources", [])
    except Exception:
        return []


if "active_source_id" not in st.session_state:
    st.session_state.active_source_id = ""


st.title("📄 RAG Institute Assistant")
st.caption("Upload documents via LlamaIndex & Inngest, then ask questions answered by an LLM using retrieved document context.")

tab_ingest, tab_ask = st.tabs(["📥 Ingest Document", "💬 Ask a Question"])

# ── Tab 1: Ingest ───────────────────────────────────────────────────────────
with tab_ingest:
    st.subheader("Upload a PDF Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    default_source = ""
    if uploaded_file:
        default_source = clean_source_id(uploaded_file.name)

    source_id = st.text_input(
        "Source ID (unique name for this document)",
        value=default_source,
        placeholder="e.g., attendance_policy_2026",
    )

    if st.button("Ingest Document", type="primary", disabled=not (uploaded_file and source_id)):
        with st.spinner("Saving file, extracting text, embedding chunks, and updating Qdrant..."):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                result = ingest_document(tmp_path, source_id)
                st.session_state.active_source_id = source_id
                st.success(
                    f"Ingested '{source_id}' successfully "
                    f"({result.get('chunks_ingested', 0)} chunks). You can ask questions now."
                )
            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not reach the FastAPI backend at "
                    f"{FASTAPI_BASE_URL}. Make sure it is running (`uv run python main.py`)."
                )
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    st.divider()
    st.caption(
        "Ingestion replaces previous chunks with the same Source ID, so re-uploading a book keeps retrieval clean."
    )

# ── Tab 2: Ask ──────────────────────────────────────────────────────────────
with tab_ask:
    st.subheader("Ask a Question")
    question = st.text_input("Your question", placeholder="e.g., What is the attendance policy?")
    top_k = st.slider("Number of context chunks to retrieve (top_k)", min_value=1, max_value=10, value=5)
    available_sources = fetch_sources()
    source_options = ["All documents", *available_sources]
    default_index = 0
    if st.session_state.active_source_id in available_sources:
        default_index = source_options.index(st.session_state.active_source_id)
    selected_source = st.selectbox("Document scope", source_options, index=default_index)
    source_id_filter = None if selected_source == "All documents" else selected_source

    if st.button("Ask", type="primary", disabled=not question):
        with st.spinner("Searching Qdrant vector index and generating answer via OpenRouter..."):
            try:
                payload = {"question": question, "top_k": top_k}
                if source_id_filter:
                    payload["source_id"] = source_id_filter
                resp = requests.post(
                    f"{FASTAPI_BASE_URL}/ask",
                    json=payload,
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
    st.caption("Answers are generated by the configured OpenRouter LLM and grounded in the retrieved document context.")
