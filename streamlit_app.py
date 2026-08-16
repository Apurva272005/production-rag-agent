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


def ingest_document_direct(uploaded_file, source_id: str) -> dict:
    """Upload single PDF directly to backend server via multipart file upload."""
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
    data = {"source_id": source_id, "replace_existing": "true"}
    resp = requests.post(
        f"{FASTAPI_BASE_URL}/upload",
        files=files,
        data=data,
        timeout=600,
    )
    if resp.status_code >= 400:
        try:
            res_data = resp.json()
            detail = res_data.get("detail") if isinstance(res_data, dict) else resp.text
            if detail:
                raise RuntimeError(detail)
        except Exception as err:
            if isinstance(err, RuntimeError):
                raise err
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
    resp.raise_for_status()
    return resp.json()


st.title("📄 RAG Institute Assistant")
st.caption("Upload documents via LlamaIndex & Inngest, then ask questions answered by an LLM using retrieved document context.")

tab_ingest, tab_ask = st.tabs(["📥 Ingest Document(s)", "💬 Ask a Question"])

# ── Tab 1: Ingest Multiple Documents ─────────────────────────────────────────
with tab_ingest:
    st.subheader("Upload PDF Documents")
    uploaded_files = st.file_uploader(
        "Choose one or multiple PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.write(f"📁 **{len(uploaded_files)} PDF file(s) selected:**")
        file_source_map = {}
        for idx, f in enumerate(uploaded_files):
            raw_name = os.path.splitext(f.name)[0]
            clean_source = "".join(c if (c.isalnum() or c in "_-") else "_" for c in raw_name)
            file_source_map[f] = clean_source
            st.caption(f"• `{f.name}` ({f.size / (1024*1024):.1f} MB) ➔ Source ID: `{clean_source}`")

        if st.button("Ingest All Documents", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_files = len(uploaded_files)
            success_count = 0
            total_chunks = 0

            for idx, file in enumerate(uploaded_files):
                src_id = file_source_map[file]
                status_text.text(f"Processing {idx+1}/{total_files}: '{file.name}'...")
                
                try:
                    res = ingest_document_direct(file, src_id)
                    chunks = res.get("chunks_ingested", 0)
                    total_chunks += chunks
                    success_count += 1
                    st.success(f"✅ Ingested '{src_id}' ({chunks} chunks stored).")
                except requests.exceptions.ConnectionError:
                    st.error(
                        f"Could not reach the FastAPI backend at {FASTAPI_BASE_URL}. "
                        "Make sure it is running (`uv run python main.py`)."
                    )
                    break
                except Exception as e:
                    st.error(f"❌ Ingestion failed for '{file.name}': {e}")
                
                progress_bar.progress((idx + 1) / total_files)

            status_text.text("Finished processing all documents!")
            if success_count > 0:
                st.balloons()
                st.info(f"🎉 **Batch Ingestion Summary**: Successfully ingested {success_count}/{total_files} file(s) ({total_chunks} total chunks stored in Qdrant Cloud).")

    st.divider()
    st.caption("Ingestion replaces previous chunks with the same Source ID, so re-uploading a book keeps retrieval clean.")

# ── Tab 2: Ask ──────────────────────────────────────────────────────────────
with tab_ask:
    st.subheader("Ask a Question")
    question = st.text_input("Your question", placeholder="e.g., Explain supervised learning.")

    available_sources = ["All documents"]
    try:
        s_resp = requests.get(f"{FASTAPI_BASE_URL}/sources", timeout=5)
        if s_resp.status_code == 200:
            available_sources.extend(s_resp.json().get("sources", []))
    except Exception:
        pass

    selected_source = st.selectbox("Document Scope Filter", available_sources)
    top_k = st.slider("Number of context chunks to retrieve (top_k)", min_value=1, max_value=20, value=5)

    if st.button("Ask", type="primary", disabled=not question):
        with st.spinner("Searching Qdrant vector index and generating answer via LLM..."):
            try:
                payload = {"question": question, "top_k": top_k}
                if selected_source and selected_source != "All documents":
                    payload["source_id"] = selected_source

                resp = requests.post(
                    f"{FASTAPI_BASE_URL}/ask",
                    json=payload,
                    timeout=60,
                )
                if resp.status_code >= 400:
                    try:
                        detail = resp.json().get("detail")
                        if detail:
                            raise RuntimeError(detail)
                    except Exception:
                        pass
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
                    f"Could not reach the FastAPI backend at {FASTAPI_BASE_URL}. "
                    "Make sure it is running (`uv run python main.py`)."
                )
            except Exception as e:
                st.error(f"Query failed: {e}")

    st.divider()
    st.caption("Answers are generated strictly from ingested document content using free LLM models via OpenRouter.")
