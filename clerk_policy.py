"""CHARUSAT email policy used by the backend authorization boundary."""

from __future__ import annotations

import os


ROLES = {"admin", "faculty", "student"}


def parsed_setting(name: str) -> list[str]:
    raw_setting = os.getenv(name, "").strip().strip("\"'")
    values = []
    for raw_value in raw_setting.split(","):
        value = raw_value.strip().strip("\"'").strip().lower()
        if value and value not in values:
            values.append(value)
    return values


def validate_domain_settings() -> None:
    missing = [name for name in ("STUDENT_DOMAINS", "FACULTY_DOMAINS") if not parsed_setting(name)]
    if not parsed_setting("ADMIN_EMAILS") and not parsed_setting("ADMIN_EXCEPTION_EMAILS"):
        missing.append("ADMIN_EMAILS or ADMIN_EXCEPTION_EMAILS")
    if missing:
        raise RuntimeError(f"Required Clerk membership settings are empty: {', '.join(missing)}")


def _configured_values(name: str) -> set[str]:
    return set(parsed_setting(name))


def is_configured_admin_email(email: str | None) -> bool:
    normalized = (email or "").strip().lower()
    return bool(normalized) and normalized in (
        _configured_values("ADMIN_EMAILS") | _configured_values("ADMIN_EXCEPTION_EMAILS")
    )


def is_admin_exception_email(email: str | None) -> bool:
    normalized = (email or "").strip().lower()
    return bool(normalized) and normalized in _configured_values("ADMIN_EXCEPTION_EMAILS")


def role_from_email(email: str | None) -> str | None:
    """Return the CHARUSAT role for one exact, normalized email address."""
    normalized = (email or "").strip().lower()
    if normalized.count("@") != 1:
        return None

    local_part, domain = normalized.rsplit("@", 1)
    if not local_part or not domain:
        return None

    student_domains = _configured_values("STUDENT_DOMAINS")
    faculty_domains = _configured_values("FACULTY_DOMAINS")
    if is_configured_admin_email(normalized):
        return "admin"
    if domain in student_domains:
        return "student"
    if domain in faculty_domains:
        return "faculty"
    return None
