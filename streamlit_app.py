from __future__ import annotations

import html
import json
import logging
import os
import time
from pathlib import Path
from urllib.parse import urlencode

import requests
import streamlit as st
from dotenv import load_dotenv

from ui_components import alert_card, initials, inject_styles, page_header, role_badge, stat_card


load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
CLERK_SIGN_IN_URL = f"{FASTAPI_BASE_URL}/auth/clerk?{urlencode({'force': '1'})}"
MEMBERSHIP_DENIED_MESSAGE = "You are not a CHARUSAT Member"
logger = logging.getLogger("streamlit_auth")

SESSION_STATE_DEFAULTS = {
    "token": None,
    "user": None,
    "user_email": None,
    "user_role": None,
    "user_department": None,
    "error_message": None,
    "auth_action": None,
    "active_view": "Ask",
    "chat_history": [],
}
AUTH_STATE_KEYS = ("token", "user", "user_email", "user_role", "user_department", "error_message", "auth_action")


def init_session_state() -> None:
    for key, default in SESSION_STATE_DEFAULTS.items():
        st.session_state.setdefault(key, default)


st.set_page_config(page_title="RAG Institute Assistant", page_icon="🎓", layout="wide")
init_session_state()
inject_styles()


def get_auth_headers() -> dict[str, str]:
    token = st.session_state.get("token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def clear_auth_state() -> None:
    for key in AUTH_STATE_KEYS:
        st.session_state[key] = SESSION_STATE_DEFAULTS[key]
    st.query_params.clear()


def parse_detail(response: requests.Response) -> tuple[str | None, str | None]:
    try:
        detail = response.json().get("detail", {})
    except ValueError:
        return None, None
    if isinstance(detail, dict):
        return detail.get("message"), detail.get("reason")
    return str(detail), None


def load_clerk_session() -> bool:
    token = st.session_state.get("token")
    if not token:
        token = st.query_params.get("clerk_token")
        if token:
            st.session_state["token"] = token
            st.query_params.clear()
    if not st.session_state.get("token"):
        return False
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/auth/me", headers=get_auth_headers(), timeout=10)
        if response.status_code == 200:
            profile = response.json()
            st.session_state["user"] = profile
            st.session_state["user_email"] = profile.get("email") or profile.get("id")
            st.session_state["user_role"] = profile.get("role", "student")
            st.session_state["user_department"] = profile.get("department")
            st.session_state["error_message"] = None
            return True
        _, reason = parse_detail(response)
        clear_auth_state()
        st.session_state["auth_action"] = "membership_denied" if reason == "domain_not_allowed" else "auth_error"
        if response.status_code == 401 and reason == "session_invalid":
            st.session_state["error_message"] = "Your session expired. Please sign in again."
        elif response.status_code == 403 and reason == "domain_not_allowed":
            st.session_state["error_message"] = MEMBERSHIP_DENIED_MESSAGE
        elif response.status_code == 503:
            st.session_state["error_message"] = "Service unavailable, please try again."
        else:
            st.session_state["error_message"] = "Your session is invalid. Please sign in again."
    except requests.RequestException as exc:
        logger.exception("Streamlit could not reach FastAPI at %s: %s: %s", FASTAPI_BASE_URL, type(exc).__name__, str(exc))
        clear_auth_state()
        st.session_state["auth_action"] = "auth_error"
        st.session_state["error_message"] = f"Backend not reachable at {FASTAPI_BASE_URL}. Start FastAPI and try again."
    return False


def forward_clerk_callback() -> bool:
    callback_params = [(key, value) for key, value in st.query_params.items() if key.startswith("__clerk") or key.startswith("_clerk")]
    if not callback_params:
        return False
    callback_url = f"{FASTAPI_BASE_URL}/auth/clerk?{urlencode(callback_params)}"
    st.html(f"<script>window.top.location.replace({json.dumps(callback_url)});</script>", unsafe_allow_javascript=True)
    st.info("Completing secure sign-in...")
    return True


def logout_user(switch_account: bool = False) -> None:
    token = st.session_state.get("token")
    if token:
        try:
            requests.post(f"{FASTAPI_BASE_URL}/auth/logout", headers={"Authorization": f"Bearer {token}"}, timeout=10)
        except requests.RequestException:
            logger.exception("Logout request failed")
    clear_auth_state()
    st.session_state["auth_action"] = "switch" if switch_account else "logout"
    st.rerun()


def login_view() -> None:
    action = st.session_state.get("auth_action")
    message = st.session_state.get("error_message")
    st.markdown(
        '<div class="portal-hero"><div class="portal-mark">⌘</div>'
        '<div class="portal-eyebrow">CHARUSAT institute portal</div>'
        '<div class="portal-title">RAG Institute Assistant</div>'
        '<div class="portal-subtitle">Ask grounded questions, explore institute documents, and get answers with clear page-level citations.</div></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    if message:
        alert_card(message, "error" if action in {"membership_denied", "auth_error"} else "warning")
    _, center, _ = st.columns([1, 1.2, 1])
    with center:
        st.link_button("Sign in with Clerk", CLERK_SIGN_IN_URL, type="primary", width="stretch")
        st.caption("Use your verified CHARUSAT email address to continue.")
    st.markdown('<div class="empty-state">Secure access is managed by Clerk. Your role and permissions are verified by the backend.</div>', unsafe_allow_html=True)


def render_sidebar() -> str:
    role = st.session_state.get("user_role") or "student"
    email = st.session_state.get("user_email") or "Unknown user"
    name = email.split("@", 1)[0].replace(".", " ").replace("_", " ").title()
    with st.sidebar:
        st.markdown("## 🎓 Institute Portal")
        st.caption("Knowledge workspace")
        st.markdown(f'<div class="profile-card"><span class="profile-avatar">{initials(email)}</span><span class="profile-name">{html.escape(name)}</span><div class="profile-email">{html.escape(email)}</div><div style="margin-top:.65rem">{role_badge(role)}</div></div>', unsafe_allow_html=True)
        st.write("")
        choices = ["Ask"] + (["Documents"] if role in {"faculty", "admin"} else []) + (["Admin"] if role == "admin" else [])
        if st.session_state.get("active_view") not in choices:
            st.session_state["active_view"] = "Ask"
        selected = st.radio("Workspace", choices, key="active_view", format_func=lambda item: {"Ask": "💬  Ask", "Documents": "📁  Documents", "Admin": "📊  Admin"}[item])
        st.markdown("---")
        st.caption(f"Backend: `{FASTAPI_BASE_URL}`")
        if st.button("Sign out", icon=":material/logout:", width="stretch"):
            logout_user()
    return selected


def request_sources() -> list[str]:
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/sources", headers=get_auth_headers(), timeout=10)
        if response.status_code == 200:
            return ["All documents", *response.json().get("sources", [])]
    except requests.RequestException as exc:
        logger.exception("Failed to load sources: %s", exc)
    return ["All documents"]


def documents_view() -> None:
    role = st.session_state.get("user_role") or "student"
    page_header("Workspace / documents", "Manage institute documents", "Upload source material, control visibility, and keep the knowledge base current.")
    st.write("")
    with st.container(border=True):
        st.markdown("#### Add documents")
        ingest_mode = st.segmented_control("Ingestion mode", ["Single document", "Batch documents"], default="Single document")
        visibility_options = ["public", "student", "faculty"] + (["admin"] if role == "admin" else [])
        col_a, col_b = st.columns(2)
        with col_a:
            visibility = st.selectbox("Visibility", visibility_options, format_func=lambda value: {"public": "Public · everyone", "student": "Student · students and above", "faculty": "Faculty · scoped access", "admin": "Admin · administrators only"}[value])
        with col_b:
            department = st.text_input("Department", value=st.session_state.get("user_department") or "", disabled=role == "faculty")
        if ingest_mode == "Single document":
            uploaded = st.file_uploader("Drop a PDF here or browse", type=["pdf"], key="single_pdf")
            source_id = st.text_input("Document identifier", value=os.path.splitext(uploaded.name)[0] if uploaded else "", placeholder="e.g. ai_syllabus_2026")
            if st.button("Ingest document", icon=":material/upload_file:", type="primary", disabled=not (uploaded and source_id)):
                with st.status("Processing document...", expanded=False) as status_box:
                    try:
                        response = requests.post(f"{FASTAPI_BASE_URL}/upload", files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}, data={"source_id": source_id, "visibility": visibility, "department": department, "replace_existing": "true"}, headers=get_auth_headers(), timeout=600)
                        if response.status_code == 200:
                            status_box.update(label="Document ready", state="complete")
                            st.toast("Document ingested successfully", icon="✅")
                        else:
                            status_box.update(label="Ingestion failed", state="error")
                            st.error(response.text)
                    except requests.RequestException as exc:
                        status_box.update(label="Backend unavailable", state="error")
                        st.error(f"Upload failed: {exc}")
        else:
            uploaded_files = st.file_uploader("Drop PDFs here or browse", type=["pdf"], accept_multiple_files=True, key="batch_pdf")
            if st.button("Ingest selected documents", icon=":material/queue_play_next:", type="primary", disabled=not uploaded_files):
                progress = st.progress(0, text="Preparing batch...")
                try:
                    response = requests.post(f"{FASTAPI_BASE_URL}/upload-batch", files=[("files", (file.name, file.getvalue(), "application/pdf")) for file in uploaded_files], data={"visibility": visibility, "department": department, "replace_existing": "true"}, headers=get_auth_headers(), timeout=900)
                    progress.progress(1.0, text="Batch complete")
                    if response.status_code == 200:
                        st.toast("Batch ingestion complete", icon="✅")
                        st.json(response.json().get("details", []))
                    else:
                        st.error(response.text)
                except requests.RequestException as exc:
                    progress.empty()
                    st.error(f"Batch upload failed: {exc}")
    st.write("")
    with st.container(border=True):
        st.markdown("#### Available documents")
        sources = request_sources()[1:]
        if sources:
            st.dataframe([{"Document": source, "Status": "Ready", "Access": "Role-filtered"} for source in sources], hide_index=True, width="stretch")
            selected = st.selectbox("Select a document to delete", sources, key="delete_source")
            confirm = st.checkbox("I understand this permanently removes the document", key="confirm_delete")
            if st.button("Delete selected document", icon=":material/delete:", disabled=not confirm):
                response = requests.delete(f"{FASTAPI_BASE_URL}/documents/{selected}", headers=get_auth_headers(), timeout=60)
                if response.status_code == 200:
                    st.toast("Document deleted", icon="✅")
                    st.rerun()
                else:
                    st.error(response.text)
        else:
            st.markdown('<div class="empty-state">No documents are visible to your role yet.</div>', unsafe_allow_html=True)


def copy_answer(answer: str) -> None:
    st.html(f"<button style='border:1px solid #d9e2ec;border-radius:9px;padding:.45rem .8rem;background:transparent;cursor:pointer' onclick='navigator.clipboard.writeText({json.dumps(answer)})'>Copy answer</button>", unsafe_allow_javascript=True)


def ask_view() -> None:
    page_header("Workspace / ask", "Ask the institute knowledge base", "Get grounded answers from your accessible documents, with citations you can inspect and export.")
    sources = request_sources()
    with st.container(border=True):
        col_source, col_k = st.columns([3, 1])
        with col_source:
            selected_source = st.selectbox("Document scope", sources, label_visibility="collapsed")
        with col_k:
            top_k = st.slider("Context chunks", 1, 15, 5, label_visibility="collapsed")
        if not st.session_state.get("chat_history"):
            st.markdown('<div class="empty-state"><strong>Start with a question</strong><br>Try asking about prerequisites, course outcomes, schedules, or policies.</div>', unsafe_allow_html=True)
            suggestions = ["What are the prerequisites for this course?", "Summarize the error analysis chapter", "Explain the admission process", "What are the key learning outcomes?"]
            with st.container(horizontal=True):
                for index, suggestion in enumerate(suggestions):
                    if st.button(suggestion, key=f"suggestion_{index}"):
                        st.session_state["question_input"] = suggestion
                        st.rerun()
        with st.form("ask_form", clear_on_submit=True):
            question = st.text_input("Your question", key="question_input", placeholder="Ask a question about your institute documents...")
            submitted = st.form_submit_button("Ask", icon=":material/send:", type="primary")
    if submitted and question.strip():
        question = question.strip()
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.status("Searching documents...", expanded=False) as status_box:
            try:
                response = requests.post(f"{FASTAPI_BASE_URL}/ask", json={"question": question, "top_k": top_k, "source_id": selected_source if selected_source != "All documents" else None}, headers=get_auth_headers(), timeout=180)
                if response.status_code == 200:
                    status_box.update(label="Answer generated", state="complete")
                    data = response.json()
                    st.session_state["chat_history"].append({"role": "assistant", "content": data.get("answer", ""), "citations": data.get("citations", []), "question": question})
                    st.toast("Answer ready", icon="✨")
                else:
                    status_box.update(label="Could not generate answer", state="error")
                    st.error(response.text)
            except requests.RequestException as exc:
                status_box.update(label="Backend unavailable", state="error")
                st.error(f"Request failed: {exc}")
    for index, message in enumerate(st.session_state.get("chat_history", [])):
        with st.chat_message("user" if message["role"] == "user" else "assistant"):
            st.markdown(message["content"])
            if message["role"] != "assistant":
                continue
            citations = message.get("citations", [])
            with st.expander(f"Sources · {len(citations)} chunks", expanded=False):
                if not citations:
                    st.caption("No specific source chunks were returned.")
                for citation_index, cite in enumerate(citations):
                    source, page = cite.get("source_id", "unknown"), cite.get("page_number", 1)
                    st.markdown(f'<div class="source-card"><div class="source-title">{html.escape(source)}</div><div class="source-meta">Page {page} · {"OCR extracted" if cite.get("ocr_used") else "Native text"}</div><div class="source-meta">{html.escape(cite.get("text", "")[:350])}</div></div>', unsafe_allow_html=True)
                    if st.button("Preview page", key=f"preview_{index}_{citation_index}", icon=":material/preview:"):
                        preview = requests.get(f"{FASTAPI_BASE_URL}/preview/{source}/{page}", headers=get_auth_headers(), timeout=20)
                        if preview.status_code == 200:
                            st.image(preview.content, caption=f"{source} · page {page}", width="stretch")
                        else:
                            st.error(preview.text)
            left, right = st.columns(2)
            with left:
                copy_answer(message["content"])
            with right:
                export = requests.post(f"{FASTAPI_BASE_URL}/export", json={"question": message.get("question", ""), "answer": message["content"], "citations": citations, "source_id": "All documents"}, headers=get_auth_headers(), timeout=30)
                if export.status_code == 200:
                    st.download_button("Export answer as PDF", export.content, file_name=f"rag_answer_{int(time.time())}.pdf", mime="application/pdf", icon=":material/download:")


def admin_view() -> None:
    page_header("Workspace / administration", "Administration", "Review usage, manage Clerk roles, and inspect sensitive activity.")
    try:
        dashboard = requests.get(f"{FASTAPI_BASE_URL}/admin/dashboard", headers=get_auth_headers(), timeout=15)
        users_response = requests.get(f"{FASTAPI_BASE_URL}/admin/users", headers=get_auth_headers(), timeout=15)
        logs_response = requests.get(f"{FASTAPI_BASE_URL}/admin/logs", headers=get_auth_headers(), timeout=15)
        stats = dashboard.json() if dashboard.status_code == 200 else {}
        summary = stats.get("summary", {})
        metrics = st.columns(3)
        with metrics[0]: stat_card("Registered users", summary.get("total_users", 0), "👥")
        with metrics[1]: stat_card("Documents", summary.get("total_documents", 0), "📄")
        with metrics[2]: stat_card("Queries", summary.get("total_queries", 0), "◈")
        users = users_response.json() if users_response.status_code == 200 else []
        logs = logs_response.json() if logs_response.status_code == 200 else []
        tab_users, tab_roles, tab_logs = st.tabs(["Users", "Roles", "Logs"])
        with tab_users:
            search = st.text_input("Search users", placeholder="Search by email or user ID")
            filtered = [user for user in users if not search or search.lower() in json.dumps(user).lower()]
            st.dataframe([{"Email": user.get("email") or "Unknown", "User ID": user.get("id"), "Role": (user.get("public_metadata") or {}).get("role", "student")} for user in filtered], hide_index=True, width="stretch")
        with tab_roles:
            if users:
                selected_user = st.selectbox("User", users, format_func=lambda user: user.get("email") or user.get("id", "Unknown"))
                metadata = selected_user.get("public_metadata") or {}
                new_role = st.selectbox("Role", ["student", "faculty", "admin"], index=["student", "faculty", "admin"].index(metadata.get("role", "student")))
                new_department = st.text_input("Department", value=metadata.get("department", ""))
                if st.button("Save role", icon=":material/save:", type="primary"):
                    response = requests.put(f"{FASTAPI_BASE_URL}/admin/roles/{selected_user.get('id')}", json={"role": new_role, "department": new_department or None}, headers=get_auth_headers(), timeout=30)
                    if response.status_code == 200:
                        st.toast("Role updated", icon="✅")
                        st.rerun()
                    else:
                        st.error(response.text)
            else:
                st.info("No users returned from Clerk.")
        with tab_logs:
            st.dataframe(logs, hide_index=True, width="stretch") if logs else st.markdown('<div class="empty-state">No audit events yet.</div>', unsafe_allow_html=True)
    except requests.RequestException as exc:
        logger.exception("Admin dashboard request failed: %s", exc)
        st.error(f"Admin data is unavailable: {exc}")


if forward_clerk_callback():
    st.stop()

if not load_clerk_session():
    with st.sidebar:
        st.markdown("## 🎓 Institute Portal")
        st.caption("Secure document intelligence")
        st.markdown('<div class="portal-card-flat">Sign in with your verified institute account to continue.</div>', unsafe_allow_html=True)
    login_view()
    st.stop()

active_view = render_sidebar()
if active_view == "Documents":
    documents_view()
elif active_view == "Admin":
    admin_view()
else:
    ask_view()
