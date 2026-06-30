from __future__ import annotations

from typing import Iterable

import pandas as pd
import streamlit as st


def render_empty_state(title: str, body: str, steps: Iterable[str] | None = None) -> None:
    st.info(body)
    if steps:
        st.markdown(f"**{title}**")
        for step in steps:
            st.markdown(f"- {step}")


def render_status_pill(label: str, is_ready: bool, detail: str) -> None:
    status = "Ready" if is_ready else "Pending"
    color = "#166534" if is_ready else "#92400e"
    background = "#dcfce7" if is_ready else "#fef3c7"
    st.markdown(
        (
            f"<div style='padding:0.65rem 0.9rem;border-radius:8px;"
            f"background:{background};margin-bottom:0.6rem;'>"
            f"<div style='font-size:0.8rem;color:{color};font-weight:600;'>{label} · {status}</div>"
            f"<div style='font-size:0.92rem;color:#1f2937;'>{detail}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_dataframe_preview(data: pd.DataFrame, title: str, limit: int = 10) -> None:
    st.subheader(title)
    if data.empty:
        st.caption("No rows available yet.")
        return
    st.dataframe(data.head(limit), use_container_width=True, hide_index=True)
