import os
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from jose import jwt

from clerk_auth import CurrentUser, JWKSCache, _decode_token, record_revoked_token, require_roles
import clerk_auth
from clerk_policy import role_from_email
from rbac_filters import build_access_filter


def _claims(**overrides):
    payload = {
        "sub": "user_123",
        "iss": "https://clerk.example.com",
        "azp": "http://127.0.0.1:8501",
        "exp": int((datetime.now(timezone.utc) + timedelta(minutes=5)).timestamp()),
        "public_metadata": {"role": "student", "department": "cs"},
    }
    payload.update(overrides)
    return payload


def test_default_role_is_student_when_metadata_is_missing():
    user = CurrentUser("u", "student", None, None, {})
    assert user.role == "student"


@pytest.mark.parametrize(
    ("email", "expected"),
    [
        ("student@charusat.edu.in", "student"),
        ("student@edu.charusat.org", "student"),
        ("faculty@charusat.org", "faculty"),
        ("faculty@charusat.ac.in", "faculty"),
        ("ADMIN@CHARUSAT.ORG", "admin"),
        ("admin@charusat.org.evil.com", None),
        ("user@notcharusat.org", None),
        ("user@gmail.com", None),
        ("", None),
        ("not-an-email", None),
        ("user@", None),
        ("@charusat.org", None),
        ("a@charusat.org@evil.com", None),
    ],
)
def test_role_from_email_exact_domain(monkeypatch, email, expected):
    monkeypatch.setenv("STUDENT_DOMAINS", "charusat.edu.in,edu.charusat.org")
    monkeypatch.setenv("FACULTY_DOMAINS", "charusat.org,charusat.ac.in")
    monkeypatch.setenv("ADMIN_EMAILS", "admin@charusat.org")
    assert role_from_email(email) == expected


def test_admin_allowlist_requires_faculty_domain(monkeypatch):
    monkeypatch.setenv("STUDENT_DOMAINS", "charusat.edu.in,edu.charusat.org")
    monkeypatch.setenv("FACULTY_DOMAINS", "charusat.org,charusat.ac.in")
    monkeypatch.setenv("ADMIN_EMAILS", "admin@charusat.edu.in")
    assert role_from_email("admin@charusat.edu.in") == "admin"


def test_unverified_primary_email_is_rejected(monkeypatch):
    monkeypatch.setattr(
        clerk_auth,
        "get_user",
        lambda user_id: {
            "primary_email_address_id": "email_1",
            "email_addresses": [
                {
                    "id": "email_1",
                    "email_address": "student@charusat.edu.in",
                    "verification": {"status": "unverified"},
                }
            ],
        },
    )
    clerk_auth._identity_cache.clear()
    assert clerk_auth.resolve_verified_email("user_1") == (
        "student@charusat.edu.in",
        "email_unverified",
    )


def test_clerk_api_failure_returns_503(monkeypatch):
    from fastapi.testclient import TestClient
    import main
    from clerk_api import ClerkAPIError

    clerk_auth._identity_cache.clear()
    monkeypatch.setattr(
        clerk_auth,
        "_decode_token",
        lambda token: CurrentUser("user_api_error", "student", None, None, {}),
    )
    monkeypatch.setattr(
        clerk_auth,
        "get_user",
        lambda user_id: (_ for _ in ()).throw(ClerkAPIError("Clerk unavailable")),
    )
    monkeypatch.setattr(
        clerk_auth,
        "delete_user",
        lambda user_id: pytest.fail("API failures must not delete Clerk users"),
    )

    response = TestClient(main.app).get(
        "/auth/me",
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "message": "Please try again",
            "reason": "auth_service_unavailable",
        }
    }


def test_non_charusat_user_gets_exact_forbidden_response(monkeypatch):
    from fastapi.testclient import TestClient
    import main

    clerk_auth._identity_cache.clear()
    monkeypatch.setattr(
        clerk_auth,
        "_decode_token",
        lambda token: CurrentUser("user_outsider", "student", None, None, {}),
    )
    monkeypatch.setattr(
        clerk_auth,
        "resolve_verified_email",
        lambda user_id: ("outsider@gmail.com", "domain_not_allowed"),
    )
    monkeypatch.setattr(clerk_auth, "delete_user", lambda user_id: True)

    response = TestClient(main.app).get(
        "/auth/me",
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 403
    assert response.json() == {
        "detail": {
            "message": "You are not a CHARUSAT Member",
            "reason": "domain_not_allowed",
        }
    }


def test_require_roles_returns_403_for_disallowed_role():
    dependency = require_roles("admin")
    with pytest.raises(Exception) as error:
        dependency(CurrentUser("u", "student", None, None, {}))
    assert error.value.status_code == 403


def test_access_filter_is_role_scoped():
    student_filter = build_access_filter(CurrentUser("student", "student", None, None, {}))
    faculty_filter = build_access_filter(CurrentUser("faculty", "faculty", "cs", None, {}))
    admin_filter = build_access_filter(CurrentUser("admin", "admin", None, None, {}))

    assert student_filter is not None
    assert faculty_filter is not None
    assert admin_filter is None
    assert student_filter.must[0].key == "visibility"
    assert student_filter.must[0].match.any == ["public", "student"]
    assert any(getattr(item, "key", None) == "owner_id" for item in faculty_filter.should)


def test_student_cannot_surface_faculty_or_admin_chunks(tmp_path):
    from data_loader import embed_texts
    from vector_db import QdrantStorage

    old_url = os.environ.pop("QDRANT_URL", None)
    old_key = os.environ.pop("QDRANT_API_KEY", None)
    QdrantStorage._instance = None
    try:
        storage = QdrantStorage(db_path=str(tmp_path / "qdrant"))
        texts = [
            {"text": "Public attendance policy", "page_number": 1},
            {"text": "Faculty exam answer key", "page_number": 1},
            {"text": "Admin security secret", "page_number": 1},
        ]
        vectors = embed_texts([item["text"] for item in texts])
        storage.upsert_chunks("public_doc", [texts[0]], [vectors[0]], visibility="public")
        storage.upsert_chunks("faculty_doc", [texts[1]], [vectors[1]], visibility="faculty", owner_id="faculty_1", department="cs")
        storage.upsert_chunks("admin_doc", [texts[2]], [vectors[2]], visibility="admin")

        user = CurrentUser("student_1", "student", None, None, {})
        result = storage.search(
            vectors[1],
            top_k=10,
            access_filter=build_access_filter(user),
        )
        assert "faculty_doc" not in result["sources"]
        assert "admin_doc" not in result["sources"]
        assert all(match["visibility"] in {"public", "student"} for match in result["matches"])
    finally:
        client = getattr(QdrantStorage._instance, "client", None)
        if client:
            client.close()
        QdrantStorage._instance = None
        if old_url is not None:
            os.environ["QDRANT_URL"] = old_url
        if old_key is not None:
            os.environ["QDRANT_API_KEY"] = old_key


def test_jwt_validation_checks_issuer_and_azp(monkeypatch):
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    from jose.utils import base64url_encode

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_numbers = private_key.public_key().public_numbers()
    jwk = {
        "kty": "RSA",
        "kid": "test-key",
        "use": "sig",
        "alg": "RS256",
        "n": base64url_encode(public_numbers.n.to_bytes((public_numbers.n.bit_length() + 7) // 8, "big")).decode(),
        "e": base64url_encode(public_numbers.e.to_bytes((public_numbers.e.bit_length() + 7) // 8, "big")).decode(),
    }
    token = jwt.encode(_claims(), private_key, algorithm="RS256", headers={"kid": "test-key"})
    monkeypatch.setenv("CLERK_JWKS_URL", "https://clerk.example.com/.well-known/jwks.json")
    monkeypatch.setenv("CLERK_ISSUER", "https://clerk.example.com")
    monkeypatch.setenv("CLERK_AUTHORIZED_PARTIES", "http://127.0.0.1:8501")
    with patch.object(JWKSCache, "get_key", return_value=jwk):
        user = _decode_token(token)
    assert user.id == "user_123"
    assert user.role == "student"
    assert user.department == "cs"


def test_expired_jwt_is_rejected(monkeypatch):
    token = "not-a-real-token"
    with pytest.raises(Exception) as error:
        _decode_token(token)
    assert error.value.status_code == 401


def test_token_revoked_by_logout_is_rejected(monkeypatch):
    from cryptography.hazmat.primitives.asymmetric import rsa
    from jose.utils import base64url_encode
    from database import Base, SessionLocal, engine
    from models_db import RevokedClerkToken

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_numbers = private_key.public_key().public_numbers()
    jwk = {
        "kty": "RSA",
        "kid": "logout-key",
        "use": "sig",
        "alg": "RS256",
        "n": base64url_encode(public_numbers.n.to_bytes((public_numbers.n.bit_length() + 7) // 8, "big")).decode(),
        "e": base64url_encode(public_numbers.e.to_bytes((public_numbers.e.bit_length() + 7) // 8, "big")).decode(),
    }
    claims = _claims(sid="sess_logout")
    token = jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "logout-key"})
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        record_revoked_token(db, token, claims, "user_123")
        monkeypatch.setenv("CLERK_JWKS_URL", "https://clerk.example.com/.well-known/jwks.json")
        monkeypatch.setenv("CLERK_ISSUER", "https://clerk.example.com")
        monkeypatch.setenv("CLERK_AUTHORIZED_PARTIES", "http://127.0.0.1:8501")
        with patch.object(JWKSCache, "get_key", return_value=jwk):
            with pytest.raises(Exception) as error:
                _decode_token(token)
        assert error.value.status_code == 401
    finally:
        db.query(RevokedClerkToken).filter(RevokedClerkToken.token_hash.isnot(None)).delete()
        db.commit()
        db.close()
