"""Safe local diagnostic for the FastAPI + Clerk configuration.

Run from the project root with the same virtual environment as the backend:
    python diagnose_clerk.py
"""

from __future__ import annotations

import base64
import json
import os
import socket
import ssl
import sys
from urllib.parse import urlparse

import certifi
import requests
from dotenv import load_dotenv


ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(ROOT, ".env")
TIMEOUT = (10, 10)


def safe_body(response: requests.Response) -> str:
    body = response.text.replace("\n", " ").replace("\r", " ")[:1000]
    for value in (os.getenv("CLERK_SECRET_KEY", ""), os.getenv("OPENROUTER_API_KEY", "")):
        if value:
            body = body.replace(value, "[REDACTED]")
    return body


def instance_from_publishable_key(key: str) -> str | None:
    if not key.startswith(("pk_test_", "pk_live_")):
        return None
    encoded = key.split("_", 2)[-1]
    try:
        decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode()
    except (ValueError, UnicodeDecodeError):
        return None
    return decoded.rstrip("$") or None


def report(number: int, name: str, fn) -> object | None:
    try:
        result = fn()
        print(f"PASS {number}. {name}")
        return True if result is None else result
    except Exception as exc:  # This script must show the real diagnostic failure.
        print(f"FAIL {number}. {name}: {type(exc).__name__}: {exc}")
        return None


def main() -> int:
    load_dotenv(dotenv_path=ENV_PATH, override=False)
    failures = 0

    def env_check():
        secret = os.getenv("CLERK_SECRET_KEY", "")
        if not secret or secret != secret.strip() or any(c.isspace() for c in secret):
            raise RuntimeError("CLERK_SECRET_KEY is missing or contains whitespace")
        if any(c in secret for c in ('"', "'")):
            raise RuntimeError("CLERK_SECRET_KEY contains quotes")
        if not secret.startswith(("sk_test_", "sk_live_")):
            raise RuntimeError("CLERK_SECRET_KEY must start with sk_test_ or sk_live_")
        print(f"    secret: present, length={len(secret)}, prefix={secret[:3]}")
        return secret

    secret = report(1, ".env and Clerk secret", env_check)
    if secret is None:
        failures += 1

    def network_check():
        host = urlparse("https://api.clerk.com").hostname
        assert host
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        print(f"    DNS: {host} -> {sorted({item[4][0] for item in addresses})}")
        response = requests.get("https://api.clerk.com", timeout=TIMEOUT, verify=certifi.where())
        print(f"    HTTPS: HTTP {response.status_code}")

    if report(2, "DNS and HTTPS to api.clerk.com", network_check) is None:
        failures += 1

    def backend_check():
        if secret is None:
            raise RuntimeError("skipped because secret validation failed")
        response = requests.get(
            "https://api.clerk.com/v1/users",
            params={"limit": 1},
            headers={"Authorization": f"Bearer {secret}"},
            timeout=TIMEOUT,
            verify=certifi.where(),
        )
        if response.status_code != 200:
            label = "wrong key or wrong instance" if response.status_code in {401, 403} else "Clerk response"
            raise RuntimeError(f"HTTP {response.status_code} ({label}): {safe_body(response)}")

    if report(3, "Clerk Backend API GET /v1/users?limit=1", backend_check) is None:
        failures += 1

    def jwks_check():
        jwks_url = os.getenv("CLERK_JWKS_URL", "").strip()
        issuer = os.getenv("CLERK_ISSUER", "").strip().rstrip("/")
        if not jwks_url or not issuer:
            raise RuntimeError("CLERK_JWKS_URL and CLERK_ISSUER must both be set")
        if urlparse(jwks_url).hostname != urlparse(issuer).hostname:
            raise RuntimeError(f"issuer host {urlparse(issuer).hostname!r} differs from JWKS host {urlparse(jwks_url).hostname!r}")
        response = requests.get(jwks_url, timeout=TIMEOUT, verify=certifi.where())
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {safe_body(response)}")
        keys = response.json().get("keys", [])
        if not keys:
            raise RuntimeError("JWKS response contains no keys")
        print(f"    issuer/JWKS host: {urlparse(issuer).hostname}; keys={len(keys)}")

    if report(4, "JWKS fetch and issuer host match", jwks_check) is None:
        failures += 1

    def instance_check():
        publishable = os.getenv("CLERK_PUBLISHABLE_KEY", "").strip()
        issuer_host = urlparse(os.getenv("CLERK_ISSUER", "")).hostname
        decoded_host = instance_from_publishable_key(publishable)
        print(f"    publishable instance: {decoded_host}")
        print(f"    issuer instance: {issuer_host}")
        if not decoded_host or decoded_host != issuer_host:
            raise RuntimeError("publishable key and issuer are not from the same Clerk instance")

    if report(5, "Clerk publishable key and issuer instance match", instance_check) is None:
        failures += 1

    def fastapi_check():
        backend_url = os.getenv("BACKEND_URL", os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")).strip().rstrip("/")
        response = requests.get(f"{backend_url}/health", timeout=TIMEOUT)
        print(f"    Streamlit backend URL: {backend_url}")
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {safe_body(response)}")

    if report(6, "FastAPI reachability from Streamlit configuration", fastapi_check) is None:
        failures += 1

    print(f"\nDiagnostic result: {'PASS' if failures == 0 else f'{failures} check(s) failed'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
