import os
import io
import time
import requests
import streamlit as st

# ── Config & Environment ──────────────────────────────────────────────────
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")
INNGEST_DEV_URL = os.getenv("INNGEST_DEV_EVENT_URL", "http://127.0.0.1:8288/e/local")

st.set_page_config(
    page_title="RAG Institute Assistant",
    page_icon="🎓",
    layout="wide",
)

# ── Session State Initialization ──────────────────────────────────────────
if "token" not in st.session_state:
    st.session_state.token = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "user_role" not in st.session_state:
    st.session_state.user_role = None


def get_auth_headers() -> dict:
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}


def login_user(email: str, password: str) -> bool:
    try:
        resp = requests.post(
            f"{FASTAPI_BASE_URL}/auth/login",
            json={"email": email, "password": password},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.token = data["access_token"]
            st.session_state.user_email = data["email"]
            st.session_state.user_role = data["role"]
            return True
        else:
            err = resp.json().get("detail", "Login failed")
            st.error(f"Authentication failed: {err}")
            return False
    except Exception as e:
        st.error(f"Cannot connect to backend server: {e}")
        return False


def logout_user():
    st.session_state.token = None
    st.session_state.user_email = None
    st.session_state.user_role = None
    st.rerun()


# ── Sidebar: Authentication & User Profile ──────────────────────────────────
with st.sidebar:
    st.title("🎓 Institute Portal")

    if not st.session_state.token:
        st.subheader("🔐 Sign In")
        login_email = st.text_input("Email", placeholder="e.g., student@institute.edu")
        login_pass = st.text_input("Password", type="password")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Sign In", type="primary", use_container_width=True):
                if login_email and login_pass:
                    if login_user(login_email, login_pass):
                        st.rerun()
                else:
                    st.warning("Please enter both email and password.")

        st.markdown("---")
        st.caption("**Demo Credentials:**")
        st.caption("• **Admin:** `admin@institute.edu` / `Admin@123`")
        st.caption("• **Faculty:** `faculty@institute.edu` / `Faculty@123`")
        st.caption("• **Student:** `student@institute.edu` / `Student@123`")

    else:
        role_emoji = {"admin": "🛡️", "faculty": "👨‍🏫", "student": "🎓"}.get(st.session_state.user_role, "👤")
        st.success(f"Signed in as **{st.session_state.user_email}**")
        st.info(f"Role: **{role_emoji} {st.session_state.user_role.upper()}**")

        if st.button("🚪 Sign Out", use_container_width=True):
            logout_user()

        st.markdown("---")
        st.caption(f"Backend Server: `{FASTAPI_BASE_URL}`")


# ── Main Header ─────────────────────────────────────────────────────────────
st.title("📄 RAG Institute Assistant")
st.caption("Production RAG pipeline with Role-Based Access Control, OCR for scanned documents, Citation Page Preview, and Answer PDF Export.")

# If not authenticated, prompt to sign in
if not st.session_state.token:
    st.warning("⚠️ Please sign in from the sidebar to access document queries and management.")
    st.stop()


# ── Role-Based Tab Rendering ────────────────────────────────────────────────
user_role = st.session_state.user_role

if user_role == "student":
    tabs = st.tabs(["💬 Ask a Question"])
    tab_ask = tabs[0]
    tab_ingest = None
    tab_admin = None
elif user_role == "faculty":
    tabs = st.tabs(["📥 Ingest Document(s)", "💬 Ask a Question"])
    tab_ingest = tabs[0]
    tab_ask = tabs[1]
    tab_admin = None
else:  # Admin
    tabs = st.tabs(["📥 Ingest Document(s)", "💬 Ask a Question", "📊 Admin Analytics Dashboard"])
    tab_ingest = tabs[0]
    tab_ask = tabs[1]
    tab_admin = tabs[2]


# ── Tab 1: Ingest (Faculty & Admin Only) ─────────────────────────────────────
if tab_ingest:
    with tab_ingest:
        st.subheader("📥 Upload & Ingest Documents")
        st.caption("Uploaded documents are processed by PyMuPDF / OCR, embedded with FastEmbed (384d), and indexed in Qdrant.")

        ingest_mode = st.radio("Ingestion Mode", ["Single Document", "Batch Documents"], horizontal=True)

        visibility = st.radio(
            "Document Visibility & Access Control",
            ["public", "private"],
            format_func=lambda x: "🌐 Public (Accessible to all students and faculty)" if x == "public" else "🔒 Private (Restricted to Faculty & Admin only)",
            help="Private documents cannot be retrieved or previewed by students.",
        )

        if ingest_mode == "Single Document":
            col_f1, col_f2 = st.columns([1, 1])
            with col_f1:
                uploaded_file = st.file_uploader("Select PDF Document", type=["pdf"], key="single_pdf")
            with col_f2:
                default_name = os.path.splitext(uploaded_file.name)[0] if uploaded_file else ""
                source_id = st.text_input("Source Identifier (Unique Doc ID)", value=default_name, placeholder="e.g., ai_syllabus_2026")

            if st.button("🚀 Ingest Document", type="primary", disabled=not (uploaded_file and source_id)):
                with st.spinner("Uploading, extracting text/OCR, and indexing vectors..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        data = {"source_id": source_id, "visibility": visibility, "replace_existing": "true"}
                        resp = requests.post(
                            f"{FASTAPI_BASE_URL}/upload",
                            files=files,
                            data=data,
                            headers=get_auth_headers(),
                            timeout=600,
                        )
                        if resp.status_code == 200:
                            res = resp.json()
                            st.success(f"✅ Successfully ingested '{res['source_id']}' with {res['chunks_ingested']} chunks! Visibility: **{visibility.upper()}**")
                        else:
                            st.error(f"Ingestion failed: {resp.text}")
                    except Exception as e:
                        st.error(f"Error during upload: {e}")

        else:
            uploaded_files = st.file_uploader("Select Multiple PDF Files", type=["pdf"], accept_multiple_files=True, key="batch_pdf")
            if st.button("🚀 Ingest All Selected Files", type="primary", disabled=not uploaded_files):
                with st.spinner(f"Ingesting {len(uploaded_files)} documents in batch..."):
                    try:
                        files_payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
                        data = {"visibility": visibility, "replace_existing": "true"}
                        resp = requests.post(
                            f"{FASTAPI_BASE_URL}/upload-batch",
                            files=files_payload,
                            data=data,
                            headers=get_auth_headers(),
                            timeout=900,
                        )
                        if resp.status_code == 200:
                            batch_res = resp.json()
                            st.success(f"✅ Batch ingestion complete! {batch_res['total_files']} files processed, {batch_res['total_chunks_ingested']} total chunks indexed.")
                            st.json(batch_res["details"])
                        else:
                            st.error(f"Batch upload failed: {resp.text}")
                    except Exception as e:
                        st.error(f"Batch error: {e}")


# ── Tab 2: Ask a Question (All Roles) ───────────────────────────────────────
with tab_ask:
    st.subheader("💬 Ask Grounded Questions")

    # Fetch available sources filtered by user role
    sources = ["All documents"]
    try:
        r = requests.get(f"{FASTAPI_BASE_URL}/sources", headers=get_auth_headers(), timeout=10)
        if r.status_code == 200:
            sources += r.json().get("sources", [])
    except Exception:
        pass

    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        question = st.text_input("Your Question", placeholder="e.g., What are the prerequisites for this course?", key="ask_input")
    with col_q2:
        selected_source = st.selectbox("Document Filter", sources, index=0)

    top_k = st.slider("Number of context chunks to retrieve (top_k)", min_value=1, max_value=15, value=5)

    if st.button("🔍 Get Answer", type="primary", disabled=not question):
        with st.spinner("Searching vector database and synthesizing answer with citations..."):
            try:
                payload = {
                    "question": question,
                    "top_k": top_k,
                    "source_id": selected_source if selected_source != "All documents" else None,
                }
                resp = requests.post(
                    f"{FASTAPI_BASE_URL}/ask",
                    json=payload,
                    headers=get_auth_headers(),
                    timeout=90,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    answer_text = data.get("answer", "")
                    citations = data.get("citations", [])
                    sources_list = data.get("sources", [])

                    st.markdown("### 💡 Answer")
                    st.markdown(answer_text)

                    # Export / Download Answer Button
                    st.markdown("---")
                    col_dl, col_space = st.columns([1, 3])
                    with col_dl:
                        try:
                            export_resp = requests.post(
                                f"{FASTAPI_BASE_URL}/export",
                                json={
                                    "question": question,
                                    "answer": answer_text,
                                    "citations": [c for c in citations],
                                    "source_id": selected_source,
                                },
                                headers=get_auth_headers(),
                                timeout=30,
                            )
                            if export_resp.status_code == 200:
                                st.download_button(
                                    label="📥 Download Answer (PDF Report)",
                                    data=export_resp.content,
                                    file_name=f"rag_answer_{int(time.time())}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True,
                                )
                        except Exception as dl_err:
                            st.caption(f"Export button unavailable: {dl_err}")

                    # Retrieved Citations & Page Preview Expanders
                    st.markdown("### 📚 Source Citations & Page Preview")
                    if not citations:
                        st.info("No specific document chunks were retrieved for this query.")
                    else:
                        for cite in citations:
                            c_idx = cite.get("index", 1)
                            c_source = cite.get("source_id", "unknown")
                            c_page = cite.get("page_number", 1)
                            c_ocr = cite.get("ocr_used", False)
                            c_text = cite.get("text", "")

                            badge = "🏷️ [OCR Extracted]" if c_ocr else "📄 [Native Text]"
                            expander_label = f"[{c_idx}] Source: **{c_source}** • Page {c_page} • {badge}"

                            with st.expander(expander_label, expanded=False):
                                st.markdown(f"**Chunk Context:**\n> {c_text}")

                                # Interactive Page Preview button
                                preview_key = f"preview_{c_source}_{c_page}_{c_idx}"
                                if st.button(f"🔍 Render Page {c_page} Preview Image", key=preview_key):
                                    with st.spinner("Rendering original document page image..."):
                                        try:
                                            p_resp = requests.get(
                                                f"{FASTAPI_BASE_URL}/preview/{c_source}/{c_page}",
                                                headers=get_auth_headers(),
                                                timeout=20,
                                            )
                                            if p_resp.status_code == 200:
                                                st.image(p_resp.content, caption=f"{c_source} — Page {c_page}", use_column_width=True)
                                            elif p_resp.status_code == 403:
                                                st.error("🔒 Access Denied: You do not have permission to preview private documents.")
                                            else:
                                                st.warning(f"Could not render preview: {p_resp.text}")
                                        except Exception as p_err:
                                            st.error(f"Preview rendering error: {p_err}")

                elif resp.status_code == 403:
                    st.error(f"🔒 Access Forbidden: {resp.json().get('detail', 'Unauthorized query')}")
                else:
                    st.error(f"Query failed: {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")


# ── Tab 3: Admin Analytics Dashboard (Admin Only) ───────────────────────────
if tab_admin:
    with tab_admin:
        st.subheader("📊 System Analytics & Audit Dashboard")

        try:
            r = requests.get(f"{FASTAPI_BASE_URL}/admin/dashboard", headers=get_auth_headers(), timeout=15)
            if r.status_code == 200:
                stats = r.json()
                summary = stats.get("summary", {})
                popular = stats.get("popular_sources", [])
                roles = stats.get("role_breakdown", {})
                recent = stats.get("recent_queries", [])

                # Metric Cards
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Queries", summary.get("total_queries", 0))
                m2.metric("Registered Users", summary.get("total_users", 0))
                m3.metric("Total Documents", summary.get("total_documents", 0))
                m4.metric("Public Documents", summary.get("public_documents", 0))
                m5.metric("Private Documents", summary.get("private_documents", 0))

                st.markdown("---")

                # Popular Sources & Role Breakdown
                c_left, c_right = st.columns([1, 1])
                with c_left:
                    st.markdown("#### 📈 Most Queried Documents")
                    if popular:
                        for p in popular:
                            st.write(f"• **{p['source_id']}**: {p['query_count']} queries")
                    else:
                        st.info("No source query logs recorded yet.")

                with c_right:
                    st.markdown("#### 👥 Queries by User Role")
                    if roles:
                        for r_name, count in roles.items():
                            st.write(f"• **{r_name.capitalize()}**: {count} queries")
                    else:
                        st.info("No role distribution data yet.")

                st.markdown("---")
                st.markdown("#### 🕒 Recent Query Audit Log")
                if recent:
                    table_rows = []
                    for item in recent:
                        table_rows.append({
                            "Timestamp": item.get("timestamp", "")[:19].replace("T", " "),
                            "User": item.get("user"),
                            "Role": item.get("role"),
                            "Source Filter": item.get("source_id") or "All",
                            "Question": item.get("question"),
                            "Contexts": item.get("num_contexts"),
                        })
                    st.dataframe(table_rows, use_container_width=True)
                else:
                    st.info("No queries have been logged yet.")

            elif r.status_code == 403:
                st.error("🔒 403 Forbidden: Administrator role required to view analytics dashboard.")
            else:
                st.error(f"Failed to load dashboard: {r.text}")
        except Exception as e:
            st.error(f"Dashboard error: {e}")
