from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

from fastapi import APIRouter, HTTPException, Request

from clerk_api import update_public_metadata


router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


def _verify_svix_signature(body: bytes, headers) -> bool:
    secret = os.getenv("CLERK_WEBHOOK_SIGNING_SECRET", "").strip()
    if not secret:
        return False
    timestamp = headers.get("svix-timestamp", "")
    message_id = headers.get("svix-id", "")
    signatures = headers.get("svix-signature", "")
    try:
        if abs(time.time() - int(timestamp)) > 300:
            return False
        encoded_secret = secret.split("_", 1)[1] if "_" in secret else secret
        key = base64.b64decode(encoded_secret)
        signed = f"{message_id}.{timestamp}.".encode() + body
        digest = base64.b64encode(hmac.new(key, signed, hashlib.sha256).digest()).decode()
        return any(
            hmac.compare_digest(value.removeprefix("v1,"), digest)
            for value in signatures.split(" ")
            if value.startswith("v1,")
        )
    except (ValueError, IndexError, base64.binascii.Error):
        return False


@router.post("/clerk")
async def clerk_webhook(request: Request):
    body = await request.body()
    if not _verify_svix_signature(body, request.headers):
        raise HTTPException(status_code=401, detail="Invalid Clerk webhook signature")
    payload = json.loads(body)
    if payload.get("type") == "user.created":
        user_id = payload.get("data", {}).get("id")
        if user_id:
            update_public_metadata(user_id, {"role": "student"})
    return {"status": "accepted"}
