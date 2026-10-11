from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import main
from clerk_auth import CurrentUser, get_current_token, get_current_user


PERMISSIONS = {
    "POST /upload": {"admin", "faculty"},
    "POST /ingest": {"admin", "faculty"},
    "DELETE /documents/{id}": {"admin", "faculty"},
    "POST /ask": {"admin", "faculty", "student"},
    "GET /sources": {"admin", "faculty", "student"},
    "GET /admin/users": {"admin"},
    "PUT /admin/roles/{id}": {"admin"},
    "GET /admin/logs": {"admin"},
    "POST /auth/logout": {"admin", "faculty", "student"},
}


@pytest.mark.parametrize("role", ["admin", "faculty", "student"])
def test_permission_matrix_is_explicit(role):
    expected = {
        endpoint: role in allowed
        for endpoint, allowed in PERMISSIONS.items()
    }
    assert expected["POST /ask"] is True
    assert expected["GET /admin/users"] is (role == "admin")
    assert expected["POST /upload"] is (role in {"admin", "faculty"})


def test_protected_routes_return_401_without_clerk_token():
    client = TestClient(main.app)
    assert client.get("/sources").status_code == 401
    assert client.post("/ask", json={"question": "hello"}).status_code == 401
    assert client.get("/admin/logs").status_code == 401


def test_root_forwards_clerk_callback_to_streamlit(monkeypatch):
    monkeypatch.setenv("STREAMLIT_URL", "http://127.0.0.1:8501")
    response = TestClient(main.app).get(
        "/?__clerk_handshake=abc&__clerk_status=complete",
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert response.headers["location"] == (
        "http://127.0.0.1:8501/?__clerk_handshake=abc&__clerk_status=complete"
    )


def test_root_without_clerk_callback_remains_json():
    response = TestClient(main.app).get("/", follow_redirects=False)
    assert response.status_code == 200
    assert response.json()["service"] == "RAG Institute Assistant API"


@pytest.mark.parametrize("role", ["student", "faculty"])
def test_admin_route_returns_403_for_non_admin(role):
    user = CurrentUser(f"{role}_1", role, "cs", f"{role}@example.com", {})
    main.app.dependency_overrides[get_current_user] = lambda: user
    try:
        response = TestClient(main.app).get("/admin/logs")
        assert response.status_code == 403
    finally:
        main.app.dependency_overrides.clear()


def test_student_upload_returns_403_before_file_processing():
    user = CurrentUser("student_1", "student", None, "student@example.com", {})
    main.app.dependency_overrides[get_current_user] = lambda: user
    try:
        response = TestClient(main.app).post(
            "/upload",
            files={"file": ("document.pdf", b"not a real pdf", "application/pdf")},
        )
        assert response.status_code == 403
    finally:
        main.app.dependency_overrides.clear()


def test_authenticated_logout_revokes_session_and_returns_success():
    user = CurrentUser(
        "student_1",
        "student",
        None,
        "student@example.com",
        {"sid": "sess_test", "exp": 4102444800},
    )
    main.app.dependency_overrides[get_current_user] = lambda: user
    main.app.dependency_overrides[get_current_token] = lambda: "test-token-for-logout"
    try:
        with patch("main.revoke_session") as revoke:
            response = TestClient(main.app).post("/auth/logout")
            revoke.assert_called_once_with("sess_test")
        assert response.status_code == 200
        assert response.json()["status"] == "logged_out"
    finally:
        main.app.dependency_overrides.clear()
