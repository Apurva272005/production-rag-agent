"""Clerk JWT verification and FastAPI authorization dependencies."""

from __future__ import annotations

import os
import json
import hashlib
import logging
import threading
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Callable, Optional

import requests
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from clerk_api import ClerkAPIError, ClerkConfigurationError, delete_user, get_user, update_public_metadata
from clerk_policy import is_admin_exception_email, is_configured_admin_email, role_from_email
from database import get_db
from models_db import AuditLog


ROLES = {"admin", "faculty", "student"}
_bearer = HTTPBearer(auto_error=False)
logger = logging.getLogger("clerk_auth")
_identity_cache: dict[str, tuple[float, str | None, str | None, str | None]] = {}
_identity_cache_lock = threading.Lock()
_admin_exception_audited: set[str] = set()


@dataclass(frozen=True)
class ClerkSettings:
    secret_key: str
    jwks_url: str
    issuer: str
    authorized_parties: frozenset[str]


def get_settings() -> ClerkSettings:
    return ClerkSettings(
        secret_key=os.getenv("CLERK_SECRET_KEY", "").strip(),
        jwks_url=os.getenv("CLERK_JWKS_URL", "").strip(),
        issuer=os.getenv("CLERK_ISSUER", "").strip(),
        authorized_parties=frozenset(
            value.strip()
            for value in os.getenv("CLERK_AUTHORIZED_PARTIES", "").split(",")
            if value.strip()
        ),
    )


class JWKSCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self._expires_at = 0.0
        self._keys: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get_key(self, jwks_url: str, kid: str) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            if now >= self._expires_at or kid not in self._keys:
                response = requests.get(jwks_url, timeout=10)
                response.raise_for_status()
                payload = response.json()
                self._keys = {key["kid"]: key for key in payload.get("keys", []) if key.get("kid")}
                self._expires_at = now + self.ttl_seconds
            key = self._keys.get(kid)
            if not key:
                raise KeyError(f"No Clerk JWKS key found for kid '{kid}'")
            return key


jwks_cache = JWKSCache()


def _unauthorized(detail: str = "Missing or invalid Clerk session token") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _metadata(claims: dict[str, Any]) -> dict[str, Any]:
    metadata = claims.get("public_metadata")
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    metadata = claims.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("public"), dict):
        return metadata["public"]
    metadata = claims.get("publicMetadata")
    return metadata if isinstance(metadata, dict) else {}


def token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def record_revoked_token(db: Any, token: str, claims: dict[str, Any], user_id: str) -> None:
    from models_db import RevokedClerkToken

    expires_at = datetime.fromtimestamp(int(claims.get("exp", 0)), tz=timezone.utc).replace(tzinfo=None)
    db.merge(
        RevokedClerkToken(
            token_hash=token_fingerprint(token),
            session_id=claims.get("sid"),
            user_id=user_id,
            expires_at=expires_at,
        )
    )
    db.commit()


def is_token_revoked(token: str) -> bool:
    """Check the persistent denylist; tolerate pre-migration test databases."""
    try:
        from database import SessionLocal
        from models_db import RevokedClerkToken

        db = SessionLocal()
        try:
            now = datetime.utcnow()
            return db.query(RevokedClerkToken).filter(
                RevokedClerkToken.token_hash == token_fingerprint(token),
                RevokedClerkToken.expires_at > now,
            ).first() is not None
        finally:
            db.close()
    except Exception:
        return False


@dataclass(frozen=True)
class CurrentUser:
    id: str
    role: str
    department: Optional[str]
    email: Optional[str]
    claims: dict[str, Any]


def _member_forbidden() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={"message": "You are not a CHARUSAT Member", "reason": "domain_not_allowed"},
    )


def _audit_membership_attempt(db: Any, user_id: str, details: str) -> None:
    try:
        db.add(
            AuditLog(
                actor_id=user_id,
                actor_role="unknown",
                action="membership_denied",
                resource_id=user_id,
                details=details,
            )
        )
        db.commit()
    except Exception:
        db.rollback()


def _audit_admin_exception_login(db: Any, user_id: str, email: str, session_key: str) -> None:
    """Record one exception-admin session without recording tokens or secrets."""
    with _identity_cache_lock:
        if session_key in _admin_exception_audited:
            return
        _admin_exception_audited.add(session_key)
    try:
        db.add(
            AuditLog(
                actor_id=user_id,
                actor_role="admin",
                action="admin_exception_login",
                resource_id=user_id,
                details=f"verified admin email={email}",
            )
        )
        db.commit()
    except Exception:
        with _identity_cache_lock:
            _admin_exception_audited.discard(session_key)
        db.rollback()
        logger.exception("Admin exception audit failed for user_id=%s", user_id)


def resolve_verified_email(user_id: str) -> tuple[str | None, str]:
    # Resolve the Clerk user on every protected request so env changes revoke access immediately.
    user = get_user(user_id)
    addresses = user.get("email_addresses") or []
    primary_id = user.get("primary_email_address_id")
    primary = next((address for address in addresses if address.get("id") == primary_id), None)
    def is_verified(address: dict[str, Any]) -> bool:
        verification = address.get("verification") or {}
        status_value = str(verification.get("status") or "").lower()
        strategy = str(verification.get("strategy") or "").lower()
        if status_value == "verified" and strategy.startswith("from_oauth_"):
            return True
        if status_value == "verified":
            return True
        address_email = str(address.get("email_address") or "").strip().lower()
        for external in user.get("external_accounts") or []:
            provider = str(external.get("provider") or "").lower()
            external_email = str(
                external.get("email_address") or external.get("email") or ""
            ).strip().lower()
            external_status = str((external.get("verification") or {}).get("status") or "").lower()
            if (
                address_email
                and address_email == external_email
                and provider in {"microsoft", "oauth_microsoft"}
                and external_status == "verified"
            ):
                return True
        return False

    candidates = [primary] if primary else []
    candidates.extend(address for address in addresses if address is not primary)
    verified = next(
        (address for address in candidates if address and is_verified(address)),
        None,
    )
    if not verified:
        has_email = any(isinstance(address.get("email_address"), str) and address.get("email_address", "").strip() for address in addresses)
        fallback_email = next(
            (
                str(address.get("email_address")).strip().lower()
                for address in candidates
                if address and isinstance(address.get("email_address"), str) and address.get("email_address", "").strip()
            ),
            None,
        )
        result = (fallback_email if has_email else None, None, "email_unverified" if has_email else "no_email")
    else:
        email = str(verified.get("email_address")).strip().lower()
        resolved_role = role_from_email(email)
        # The configured Gmail exception is valid only for a verified primary email.
        if resolved_role == "admin" and verified is not primary:
            resolved_role = None
        result = (email, resolved_role, None if resolved_role else "domain_not_allowed")
    return result[0], result[2] or "ok"


def _decode_token(token: str) -> CurrentUser:
    settings = get_settings()
    if not settings.jwks_url or not settings.issuer:
        raise _unauthorized("Clerk authentication is not configured on the server")

    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        if not kid or header.get("alg") != "RS256":
            raise _unauthorized("Invalid Clerk token header")
        key = jwks_cache.get_key(settings.jwks_url, kid)
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            issuer=settings.issuer,
            options={"verify_aud": False},
        )
    except HTTPException:
        raise
    except (JWTError, KeyError, requests.RequestException, ValueError) as exc:
        raise _unauthorized("Invalid or expired Clerk session token") from exc

    if is_token_revoked(token):
        raise _unauthorized("Clerk session has been revoked; sign in again")

    authorized_party = claims.get("azp")
    if not authorized_party:
        raise _unauthorized("Clerk token is missing authorized party")
    if settings.authorized_parties and authorized_party not in settings.authorized_parties:
        raise _unauthorized("Clerk token was issued for an unauthorized party")

    user_id = claims.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise _unauthorized("Clerk token is missing subject")

    metadata = _metadata(claims)
    role = metadata.get("role", "student")
    if role not in ROLES:
        role = "student"
    department = metadata.get("department")
    return CurrentUser(
        id=user_id,
        role=role,
        department=department if isinstance(department, str) and department.strip() else None,
        email=claims.get("email") or claims.get("email_address"),
        claims=claims,
    )


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Any = Depends(get_db),
) -> CurrentUser:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise _unauthorized()
    decoded = _decode_token(credentials.credentials)
    try:
        verified_email, resolution_reason = resolve_verified_email(decoded.id)
    except ClerkConfigurationError as exc:
        logger.exception("Clerk configuration failure during user lookup for user_id=%s", decoded.id)
        _audit_membership_attempt(db, decoded.id, "invalid Clerk server configuration")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Authentication is not configured correctly on the server", "reason": "server_misconfigured"},
        ) from exc
    except (ClerkAPIError, RuntimeError, requests.RequestException) as exc:
        logger.exception(
            "Clerk user lookup failed for user_id=%s: %s: %s",
            decoded.id,
            type(exc).__name__,
            str(exc),
        )
        _audit_membership_attempt(db, decoded.id, f"verified email lookup failed: {type(exc).__name__}")
        if getattr(exc, "reason", None) == "session_invalid":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"message": "Your session expired. Please sign in again.", "reason": "session_invalid"},
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": "Please try again", "reason": "auth_service_unavailable"},
        ) from exc

    if resolution_reason != "ok":
        if resolution_reason == "domain_not_allowed" and not is_configured_admin_email(verified_email):
            try:
                delete_user(decoded.id)
            except Exception as exc:
                logger.exception("Non-member Clerk deletion failed for user_id=%s", decoded.id)
                _audit_membership_attempt(db, decoded.id, f"non-member deletion failed: {type(exc).__name__}")
            else:
                _audit_membership_attempt(db, decoded.id, f"non-member deleted: {verified_email}")
        else:
            _audit_membership_attempt(db, decoded.id, resolution_reason)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "message": "You are not a CHARUSAT Member" if resolution_reason == "domain_not_allowed" else "Your CHARUSAT email could not be verified.",
                "reason": resolution_reason,
            },
        )

    resolved_role = role_from_email(verified_email)

    metadata = _metadata(decoded.claims)
    if metadata.get("role") != resolved_role:
        try:
            update_public_metadata(decoded.id, {**metadata, "role": resolved_role})
        except Exception as exc:
            logger.exception("Clerk role metadata update failed for user_id=%s", decoded.id)
            _audit_membership_attempt(db, decoded.id, f"role metadata update failed: {type(exc).__name__}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"message": "Please try again", "reason": "auth_service_unavailable"},
            ) from exc

    if resolved_role == "admin" and is_admin_exception_email(verified_email):
        session_key = str(decoded.claims.get("sid") or decoded.id)
        _audit_admin_exception_login(db, decoded.id, verified_email or "", session_key)

    return CurrentUser(
        id=decoded.id,
        role=resolved_role,
        department=decoded.department,
        email=verified_email,
        claims=decoded.claims,
    )


def get_current_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> str:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise _unauthorized()
    return credentials.credentials


def require_roles(*allowed_roles: str) -> Callable:
    allowed = set(allowed_roles)

    def dependency(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if current_user.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role}' is not allowed for this operation",
            )
        return current_user

    return dependency


require_role = require_roles
