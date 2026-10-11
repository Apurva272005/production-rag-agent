from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from clerk_api import list_users, update_public_metadata
from clerk_auth import CurrentUser, require_roles
from database import get_db
from models_db import AuditLog


router = APIRouter(prefix="/admin", tags=["Clerk administration"])


class RoleUpdate(BaseModel):
    role: str = Field(pattern="^(admin|faculty|student)$")
    department: Optional[str] = None


def _audit(db: Session, actor: CurrentUser, action: str, resource_id: str, details: dict[str, Any]) -> None:
    db.add(
        AuditLog(
            actor_id=actor.id,
            actor_role=actor.role,
            action=action,
            resource_id=resource_id,
            details=json.dumps(details),
        )
    )
    db.commit()


@router.get("/users")
def users(_: CurrentUser = Depends(require_roles("admin"))) -> list[dict[str, Any]]:
    return [
        {
            "id": user.get("id"),
            "email": (user.get("email_addresses") or [{}])[0].get("email_address"),
            "public_metadata": user.get("public_metadata") or {},
        }
        for user in list_users()
    ]


@router.put("/roles/{clerk_user_id}")
def set_role(
    clerk_user_id: str,
    req: RoleUpdate,
    actor: CurrentUser = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    metadata = {"role": req.role}
    if req.department:
        metadata["department"] = req.department.strip()
    updated = update_public_metadata(clerk_user_id, metadata)
    _audit(db, actor, "role_change", clerk_user_id, metadata)
    return {"id": updated.get("id", clerk_user_id), "public_metadata": updated.get("public_metadata", metadata)}


@router.put("/users/{clerk_user_id}")
def update_user_role(
    clerk_user_id: str,
    req: RoleUpdate,
    actor: CurrentUser = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    return set_role(clerk_user_id, req, actor, db)


@router.get("/logs")
def logs(
    _: CurrentUser = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    return [
        {
            "id": item.id,
            "actor_id": item.actor_id,
            "actor_role": item.actor_role,
            "action": item.action,
            "resource_id": item.resource_id,
            "details": item.details,
            "timestamp": item.timestamp.isoformat(),
        }
        for item in db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(200).all()
    ]
