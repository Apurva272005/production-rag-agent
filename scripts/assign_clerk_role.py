#!/usr/bin/env python3
"""Assign a Clerk public-metadata role from an admin shell."""

from __future__ import annotations

import argparse
import json
import os

import requests
from dotenv import load_dotenv


def main() -> int:
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
    parser = argparse.ArgumentParser()
    parser.add_argument("clerk_user_id")
    parser.add_argument("role", choices=("admin", "faculty", "student"))
    parser.add_argument("--department", default=None)
    args = parser.parse_args()

    secret = os.getenv("CLERK_SECRET_KEY", "").strip()
    if not secret:
        parser.error("CLERK_SECRET_KEY is not configured")

    metadata = {"role": args.role}
    if args.department:
        metadata["department"] = args.department.strip()
    response = requests.patch(
        f"https://api.clerk.com/v1/users/{args.clerk_user_id}",
        headers={"Authorization": f"Bearer {secret}", "Content-Type": "application/json"},
        json={"public_metadata": metadata},
        timeout=15,
    )
    response.raise_for_status()
    user = response.json()
    print(json.dumps({"id": user.get("id"), "public_metadata": user.get("public_metadata", metadata)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
