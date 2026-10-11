"""Server-side Qdrant access filters derived from verified Clerk claims."""

from __future__ import annotations

from typing import Optional

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from clerk_auth import CurrentUser


def build_access_filter(user: CurrentUser) -> Optional[Filter]:
    if user.role == "admin":
        return None

    public_or_student = FieldCondition(
        key="visibility",
        match=MatchAny(any=["public", "student"]),
    )

    if user.role == "student":
        return Filter(must=[public_or_student])

    faculty_scope = []
    if user.department:
        faculty_scope.append(
            FieldCondition(key="department", match=MatchValue(value=user.department))
        )
    faculty_scope.append(FieldCondition(key="owner_id", match=MatchValue(value=user.id)))

    faculty_documents = Filter(
        must=[
            FieldCondition(key="visibility", match=MatchValue(value="faculty")),
            Filter(should=faculty_scope),
        ]
    )
    own_documents = FieldCondition(key="owner_id", match=MatchValue(value=user.id))
    return Filter(should=[public_or_student, faculty_documents, own_documents])


def allowed_visibilities(user: CurrentUser) -> list[str]:
    if user.role == "admin":
        return ["public", "student", "faculty", "admin"]
    if user.role == "faculty":
        return ["public", "student", "faculty"]
    return ["public", "student"]
