from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st


def inject_styles() -> None:
    css = Path(__file__).with_name("style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def role_badge(role: str) -> str:
    safe_role = role if role in {"admin", "faculty", "student"} else "student"
    return f'<span class="role-badge role-{safe_role}">{safe_role}</span>'


def initials(email: str | None) -> str:
    value = (email or "U").split("@", 1)[0].replace(".", " ").replace("_", " ")
    parts = [part for part in value.split() if part]
    return "".join(part[0] for part in parts[:2]).upper() or "U"


def page_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="portal-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="portal-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="portal-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def alert_card(message: str, tone: str = "error") -> None:
    st.markdown(f'<div class="portal-alert {tone}">{message}</div>', unsafe_allow_html=True)


def stat_card(label: str, value: Any, icon: str = "") -> None:
    st.markdown(
        f'<div class="portal-card"><div class="stat-label">{icon} {label}</div>'
        f'<div class="stat-value">{value}</div></div>',
        unsafe_allow_html=True,
    )
