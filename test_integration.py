"""Clerk-aware live smoke tests for a running FastAPI service."""

from __future__ import annotations

import os
import sys

import requests


BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")


def check(name: str, condition: bool, detail: str = "") -> None:
    marker = "PASS" if condition else "FAIL"
    print(f"[{marker}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        raise SystemExit(1)


def main() -> int:
    root = requests.get(f"{BASE_URL}/", timeout=10)
    check("GET /", root.status_code == 200 and root.json().get("status") == "online", str(root.status_code))

    health = requests.get(f"{BASE_URL}/health", timeout=10)
    check("GET /health", health.status_code == 200 and health.json().get("status") == "ok", str(health.status_code))

    unauthenticated = requests.get(f"{BASE_URL}/sources", timeout=10)
    check("GET /sources rejects unauthenticated requests", unauthenticated.status_code == 401, str(unauthenticated.status_code))

    token = os.getenv("CLERK_TEST_TOKEN", "").strip()
    admin_token = os.getenv("CLERK_ADMIN_TEST_TOKEN", "").strip()
    if not token:
        print("[SKIP] Authenticated smoke checks: set CLERK_TEST_TOKEN to run them")
        return 0

    headers = {"Authorization": f"Bearer {token}"}
    profile = requests.get(f"{BASE_URL}/auth/me", headers=headers, timeout=10)
    check("GET /auth/me with Clerk token", profile.status_code == 200, str(profile.status_code))

    sources = requests.get(f"{BASE_URL}/sources", headers=headers, timeout=10)
    check("GET /sources with Clerk token", sources.status_code == 200, str(sources.status_code))

    if admin_token:
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        logs = requests.get(f"{BASE_URL}/admin/logs", headers=admin_headers, timeout=10)
        check("GET /admin/logs with admin Clerk token", logs.status_code == 200, str(logs.status_code))
    else:
        print("[SKIP] Admin smoke check: set CLERK_ADMIN_TEST_TOKEN to run it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
