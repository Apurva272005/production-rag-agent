"""Small Clerk Backend API client used only by admin routes and scripts."""

from __future__ import annotations

import os
import logging
import time
from typing import Any

import certifi
import requests


CLERK_API_BASE = "https://api.clerk.com/v1"
CLERK_TIMEOUT = (10, 10)
logger = logging.getLogger("clerk_api")


class ClerkAPIError(RuntimeError):
    """Raised when the Clerk Backend API cannot provide an authoritative result."""

    def __init__(self, reason: str, status_code: int | None = None, message: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code
        self.message = message


class ClerkConfigurationError(RuntimeError):
    """Raised when the server cannot safely call Clerk."""


def validate_clerk_secret() -> None:
    raw_secret = os.getenv("CLERK_SECRET_KEY", "")
    if (
        not raw_secret
        or raw_secret != raw_secret.strip()
        or any(character.isspace() for character in raw_secret)
        or any(quote in raw_secret for quote in ("'", '"'))
        or not raw_secret.startswith("sk_")
    ):
        raise ClerkConfigurationError(
            "CLERK_SECRET_KEY must be a quoted-free, whitespace-free sk_ key"
        )


def secret_diagnostics() -> dict[str, Any]:
    """Return safe key diagnostics; never return the secret itself."""
    value = os.getenv("CLERK_SECRET_KEY", "")
    return {
        "status": "present" if value else "absent",
        "length": len(value),
        "prefix": value[:3] if value else "",
        "has_quotes_or_whitespace": any(c.isspace() for c in value)
        or any(c in value for c in ('"', "'")),
    }


def _safe_body(response: requests.Response) -> str:
    body = str(getattr(response, "text", "")).replace("\n", " ").replace("\r", " ")
    return body[:1000]


def _log_failure(url: str, response: requests.Response | None = None, exc: Exception | None = None) -> None:
    details: dict[str, Any] = {
        "url": url,
        "timeout_seconds": CLERK_TIMEOUT,
        "secret": secret_diagnostics(),
    }
    if response is not None:
        details.update({"status_code": response.status_code, "response_body": _safe_body(response)})
    if exc is not None:
        details.update({"exception_type": type(exc).__name__, "exception_message": str(exc)})
    logger.warning("Clerk Backend API failure: %s", details)


def _headers() -> dict[str, str]:
    validate_clerk_secret()
    return {
        "Authorization": f"Bearer {os.environ['CLERK_SECRET_KEY']}",
        "Content-Type": "application/json",
    }


def _request_get(url: str, headers: dict[str, str], **kwargs: Any) -> requests.Response:
    kwargs.setdefault("timeout", CLERK_TIMEOUT)
    kwargs.setdefault("verify", certifi.where())
    try:
        return requests.get(url, headers=headers, **kwargs)
    except requests.exceptions.SSLError as exc:
        _log_failure(url, exc=exc)
        raise ClerkAPIError("auth_service_unavailable", message="TLS certificate validation failed") from exc
    except requests.RequestException as exc:
        _log_failure(url, exc=exc)
        raise


def list_users(limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    response = requests.get(
        f"{CLERK_API_BASE}/users",
        headers=_headers(),
        params={"limit": limit, "offset": offset},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def get_user(user_id: str) -> dict[str, Any]:
    try:
        headers = _headers()
    except ClerkConfigurationError:
        logger.error("Clerk Backend API configuration invalid: secret=%s", secret_diagnostics())
        raise
    url = f"{CLERK_API_BASE}/users/{user_id}"
    for attempt in range(2):
        try:
            response = _request_get(url, headers)
        except requests.RequestException as exc:
            _log_failure(url, exc=exc)
            if attempt == 0:
                time.sleep(0.2)
                continue
            raise ClerkAPIError("auth_service_unavailable", message=str(exc)) from exc

        if response.status_code == 404:
            _log_failure(url, response=response)
            raise ClerkAPIError("session_invalid", 404)
        if response.status_code in {401, 403}:
            _log_failure(url, response=response)
            raise ClerkAPIError("auth_service_unavailable", response.status_code, "invalid secret key or wrong instance")
        if response.status_code == 429 or 500 <= response.status_code <= 599:
            _log_failure(url, response=response)
            if attempt == 0:
                time.sleep(0.5 if response.status_code == 429 else 0.2)
                continue
            raise ClerkAPIError("auth_service_unavailable", response.status_code, _safe_body(response))
        try:
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            _log_failure(url, response=response, exc=exc)
            raise ClerkAPIError("auth_service_unavailable", response.status_code, str(exc)) from exc

    raise ClerkAPIError("auth_service_unavailable")


def check_connection() -> None:
    """Validate credentials and connectivity without exposing user data."""
    headers = _headers()
    url = f"{CLERK_API_BASE}/users"
    for attempt in range(2):
        try:
            response = _request_get(url, headers, params={"limit": 1})
        except requests.RequestException as exc:
            if attempt == 0:
                time.sleep(0.2)
                continue
            raise ClerkAPIError("auth_service_unavailable", message=str(exc)) from exc
        if response.status_code == 401 or response.status_code == 403:
            _log_failure(url, response=response)
            raise ClerkAPIError("invalid_secret", response.status_code, "invalid secret key or wrong instance")
        if response.status_code == 429 or response.status_code >= 500:
            _log_failure(url, response=response)
            if attempt == 0:
                time.sleep(0.5 if response.status_code == 429 else 0.2)
                continue
            raise ClerkAPIError("auth_service_unavailable", response.status_code, _safe_body(response))
        try:
            response.raise_for_status()
            response.json()
            return
        except (requests.RequestException, ValueError) as exc:
            _log_failure(url, response=response, exc=exc)
            raise ClerkAPIError("auth_service_unavailable", response.status_code, str(exc)) from exc
    raise ClerkAPIError("auth_service_unavailable")


def get_primary_verified_email(user_id: str) -> str | None:
    """Return only the user's verified primary email; never use a JWT email claim."""
    user = get_user(user_id)
    primary_id = user.get("primary_email_address_id")
    for address in user.get("email_addresses") or []:
        if address.get("id") != primary_id:
            continue
        if (address.get("verification") or {}).get("status") != "verified":
            return None
        email = address.get("email_address")
        return email.strip() if isinstance(email, str) and email.strip() else None
    return None


def delete_user(user_id: str) -> bool:
    response = requests.delete(
        f"{CLERK_API_BASE}/users/{user_id}",
        headers=_headers(),
        timeout=15,
    )
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return True


def update_public_metadata(user_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    # Clerk moved public/private metadata updates to the dedicated metadata endpoint.
    url = f"{CLERK_API_BASE}/users/{user_id}/metadata"
    try:
        response = requests.patch(
            url,
            headers=_headers(),
            json={"public_metadata": metadata},
            timeout=CLERK_TIMEOUT,
            verify=certifi.where(),
        )
        if not response.ok:
            _log_failure(url, response=response)
        response.raise_for_status()
        return response.json()
    except ClerkConfigurationError:
        logger.error("Clerk metadata update blocked by configuration: secret=%s", secret_diagnostics())
        raise
    except requests.RequestException as exc:
        _log_failure(url, exc=exc)
        raise ClerkAPIError("auth_service_unavailable", message=str(exc)) from exc
    except ValueError as exc:
        _log_failure(url, response=response, exc=exc)
        raise ClerkAPIError("auth_service_unavailable", response.status_code, str(exc)) from exc


def revoke_session(session_id: str) -> bool:
    """Revoke a Clerk session. Returns False when it was already gone."""
    response = requests.post(
        f"{CLERK_API_BASE}/sessions/{session_id}/revoke",
        headers=_headers(),
        timeout=15,
    )
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return True
