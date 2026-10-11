from unittest.mock import Mock

import pytest
import requests
from fastapi.testclient import TestClient

import clerk_auth
from clerk_api import ClerkAPIError, get_user, validate_clerk_secret
from clerk_auth import CurrentUser


def _configure_domains(monkeypatch):
    monkeypatch.setenv("STUDENT_DOMAINS", "charusat.edu.in,edu.charusat.org")
    monkeypatch.setenv("FACULTY_DOMAINS", "charusat.org,charusat.ac.in")
    monkeypatch.setenv("ADMIN_EMAILS", "admin@charusat.org")
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_secret")


def _clerk_user(addresses, primary_id, external_accounts=None):
    return {
        "primary_email_address_id": primary_id,
        "email_addresses": addresses,
        "external_accounts": external_accounts or [],
    }


def _address(identifier, email, status="verified"):
    return {
        "id": identifier,
        "email_address": email,
        "verification": {"status": status},
    }


def test_primary_email_verified_direct(monkeypatch):
    _configure_domains(monkeypatch)
    monkeypatch.setattr(
        clerk_auth,
        "get_user",
        lambda user_id: _clerk_user([_address("email_1", "student@CHARUSAT.EDU.IN")], "email_1"),
    )
    clerk_auth._identity_cache.clear()
    assert clerk_auth.resolve_verified_email("user_1") == ("student@charusat.edu.in", "ok")


def test_primary_email_verified_via_microsoft_external_account(monkeypatch):
    _configure_domains(monkeypatch)
    monkeypatch.setattr(
        clerk_auth,
        "get_user",
        lambda user_id: _clerk_user(
            [_address("email_1", "faculty@charusat.org", "unverified")],
            "email_1",
            [{
                "provider": "oauth_microsoft",
                "email_address": "faculty@charusat.org",
                "verification": {"status": "VERIFIED"},
            }],
        ),
    )
    clerk_auth._identity_cache.clear()
    assert clerk_auth.resolve_verified_email("user_1") == ("faculty@charusat.org", "ok")


def test_primary_email_is_selected_even_when_not_first(monkeypatch):
    _configure_domains(monkeypatch)
    monkeypatch.setattr(
        clerk_auth,
        "get_user",
        lambda user_id: _clerk_user(
            [_address("email_1", "other@gmail.com"), _address("email_2", "faculty@charusat.ac.in")],
            "email_2",
        ),
    )
    clerk_auth._identity_cache.clear()
    assert clerk_auth.resolve_verified_email("user_1") == ("faculty@charusat.ac.in", "ok")


def test_unverified_primary_falls_back_to_verified_email(monkeypatch):
    _configure_domains(monkeypatch)
    monkeypatch.setattr(
        clerk_auth,
        "get_user",
        lambda user_id: _clerk_user(
            [_address("email_1", "old@gmail.com", "unverified"), _address("email_2", "student@edu.charusat.org")],
            "email_1",
        ),
    )
    clerk_auth._identity_cache.clear()
    assert clerk_auth.resolve_verified_email("user_1") == ("student@edu.charusat.org", "ok")


def test_no_emails_returns_no_email(monkeypatch):
    _configure_domains(monkeypatch)
    monkeypatch.setattr(clerk_auth, "get_user", lambda user_id: _clerk_user([], None))
    clerk_auth._identity_cache.clear()
    assert clerk_auth.resolve_verified_email("user_1") == (None, "no_email")


@pytest.mark.parametrize(
    "failure",
    [
        ClerkAPIError("auth_service_unavailable", 401),
        requests.Timeout("timeout"),
        ClerkAPIError("auth_service_unavailable", 500),
    ],
)
def test_clerk_lookup_failures_are_service_unavailable_and_do_not_delete(monkeypatch, failure):
    _configure_domains(monkeypatch)
    import main

    clerk_auth._identity_cache.clear()
    monkeypatch.setattr(
        clerk_auth,
        "_decode_token",
        lambda token: CurrentUser("user_api_error", "student", None, None, {}),
    )
    monkeypatch.setattr(clerk_auth, "get_user", Mock(side_effect=failure))
    delete = Mock()
    monkeypatch.setattr(clerk_auth, "delete_user", delete)

    response = TestClient(main.app).get(
        "/auth/me",
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 503
    assert response.json()["detail"]["reason"] == "auth_service_unavailable"
    delete.assert_not_called()


def test_deleted_clerk_user_returns_session_invalid(monkeypatch):
    _configure_domains(monkeypatch)
    import main

    clerk_auth._identity_cache.clear()
    monkeypatch.setattr(
        clerk_auth,
        "_decode_token",
        lambda token: CurrentUser("deleted_user", "student", None, None, {}),
    )
    monkeypatch.setattr(clerk_auth, "get_user", Mock(side_effect=ClerkAPIError("session_invalid", 404)))

    response = TestClient(main.app).get(
        "/auth/me",
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 401
    assert response.json()["detail"]["reason"] == "session_invalid"


def test_verified_gmail_is_domain_rejected_and_deleted(monkeypatch):
    _configure_domains(monkeypatch)
    import main

    clerk_auth._identity_cache.clear()
    monkeypatch.setattr(
        clerk_auth,
        "_decode_token",
        lambda token: CurrentUser("gmail_user", "student", None, None, {}),
    )
    monkeypatch.setattr(
        clerk_auth,
        "get_user",
        lambda user_id: _clerk_user([_address("email_1", "user@gmail.com")], "email_1"),
    )
    delete = Mock(return_value=True)
    monkeypatch.setattr(clerk_auth, "delete_user", delete)

    response = TestClient(main.app).get(
        "/auth/me",
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 403
    assert response.json()["detail"]["reason"] == "domain_not_allowed"
    delete.assert_called_once_with("gmail_user")


@pytest.mark.parametrize("value", [None, "", " pk_test_secret", '"sk_test_secret"', "sk test_secret", "pk_test_secret"])
def test_invalid_clerk_secret_fails_validation(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)
    else:
        monkeypatch.setenv("CLERK_SECRET_KEY", value)
    with pytest.raises(RuntimeError):
        validate_clerk_secret()


def test_get_user_retries_a_server_error(monkeypatch):
    _configure_domains(monkeypatch)
    first = Mock(status_code=500)
    second = Mock(status_code=200)
    second.json.return_value = {"id": "user_1"}
    monkeypatch.setattr("clerk_api.requests.get", Mock(side_effect=[first, second]))
    assert get_user("user_1") == {"id": "user_1"}
