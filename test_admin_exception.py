from unittest.mock import Mock

import clerk_auth
from clerk_policy import role_from_email


def configure(monkeypatch, exception="apurvaankhiwala27@gmail.com"):
    monkeypatch.setenv("STUDENT_DOMAINS", "charusat.edu.in,edu.charusat.org")
    monkeypatch.setenv("FACULTY_DOMAINS", "charusat.org,charusat.ac.in")
    monkeypatch.setenv("ADMIN_EMAILS", "admin@charusat.org")
    monkeypatch.setenv("ADMIN_EXCEPTION_EMAILS", exception)


def address(email, status="verified"):
    return {
        "id": "primary",
        "email_address": email,
        "verification": {"status": status},
    }


def clerk_user(email, status="verified"):
    return {
        "primary_email_address_id": "primary",
        "email_addresses": [address(email, status)],
    }


def test_admin_exception_is_exact_and_case_insensitive(monkeypatch):
    configure(monkeypatch)
    assert role_from_email("apurvaankhiwala27@gmail.com") == "admin"
    assert role_from_email("APURVAANKHIWALA27@GMAIL.COM") == "admin"
    assert role_from_email("apurva.ankhiwala27@gmail.com") is None
    assert role_from_email("apurvaankhiwala27+x@gmail.com") is None
    assert role_from_email("someone@gmail.com") is None


def test_exception_requires_verified_primary_email(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(clerk_auth, "get_user", lambda _: clerk_user("apurvaankhiwala27@gmail.com", "unverified"))
    assert clerk_auth.resolve_verified_email("user_exception") == (
        "apurvaankhiwala27@gmail.com",
        "email_unverified",
    )


def test_existing_student_and_faculty_roles_remain(monkeypatch):
    configure(monkeypatch)
    assert role_from_email("student@charusat.edu.in") == "student"
    assert role_from_email("student@edu.charusat.org") == "student"
    assert role_from_email("faculty@charusat.org") == "faculty"
    assert role_from_email("faculty@charusat.ac.in") == "faculty"


def test_exception_must_be_primary_when_multiple_emails(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(
        clerk_auth,
        "get_user",
        lambda _: {
            "primary_email_address_id": "other",
            "email_addresses": [
                address("apurvaankhiwala27@gmail.com"),
                {**address("student@charusat.edu.in", "unverified"), "id": "other"},
            ],
        },
    )
    assert clerk_auth.resolve_verified_email("user_secondary_exception") == (
        "apurvaankhiwala27@gmail.com",
        "domain_not_allowed",
    )


def test_removing_exception_revokes_admin_on_next_request(monkeypatch):
    configure(monkeypatch)
    user = Mock(return_value=clerk_user("apurvaankhiwala27@gmail.com"))
    monkeypatch.setattr(clerk_auth, "get_user", user)
    assert clerk_auth.resolve_verified_email("user_revoked") == (
        "apurvaankhiwala27@gmail.com",
        "ok",
    )

    monkeypatch.setenv("ADMIN_EXCEPTION_EMAILS", "")
    assert clerk_auth.resolve_verified_email("user_revoked") == (
        "apurvaankhiwala27@gmail.com",
        "domain_not_allowed",
    )
    assert user.call_count == 2
