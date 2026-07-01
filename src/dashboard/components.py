from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
import streamlit as st


GLOBAL_STYLES = """
<style>
    :root {
        --bg: #f5f7fb;
        --panel: rgba(255,255,255,0.88);
        --panel-strong: #ffffff;
        --panel-border: rgba(148, 163, 184, 0.22);
        --text: #0f172a;
        --muted: #5b6473;
        --blue: #2563eb;
        --blue-soft: #dbeafe;
        --green: #0f9f6e;
        --green-soft: #dff8ee;
        --orange: #f97316;
        --orange-soft: #fff1e8;
        --danger: #ef4444;
        --danger-soft: #fee2e2;
        --heat-1: #fff7ed;
        --heat-2: #fed7aa;
        --heat-3: #fb923c;
        --heat-4: #ea580c;
        --heat-5: #b91c1c;
        --shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
        --radius-xl: 22px;
        --radius-lg: 18px;
        --radius-md: 14px;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 28%),
            radial-gradient(circle at top right, rgba(15, 159, 110, 0.08), transparent 30%),
            linear-gradient(180deg, #f8fafc 0%, #f5f7fb 100%);
        color: var(--text);
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1440px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown div,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label p {
        color: var(--text) !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] .stRadio > div {
        gap: 0.28rem;
    }

    [data-testid="stSidebar"] .stRadio label {
        background: transparent;
        border-radius: 12px;
        padding: 0.28rem 0.3rem;
        transition: background 160ms ease;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(37, 99, 235, 0.06);
    }

    [data-testid="stSidebar"] .stRadio label p {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }

    .uc-sidebar-brand {
        padding: 0.6rem 0.1rem 0.95rem;
        margin-bottom: 0.35rem;
    }

    .uc-sidebar-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: var(--text);
        margin: 0;
        line-height: 1.1;
    }

    .uc-sidebar-subtitle {
        margin-top: 0.35rem;
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.45;
    }

    .uc-sidebar-section {
        margin: 0.7rem 0 0.5rem;
        font-size: 0.8rem;
        font-weight: 800;
        text-transform: uppercase;
        color: var(--muted);
        letter-spacing: 0.02em;
    }

    [data-testid="stMetric"] {
        background: transparent;
        border: 0;
    }

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: var(--muted);
        font-weight: 600;
        letter-spacing: 0;
    }

    .uc-shell {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .uc-hero {
        background:
            linear-gradient(135deg, rgba(255,255,255,0.95), rgba(255,255,255,0.9)),
            linear-gradient(135deg, rgba(37,99,235,0.10), rgba(15,159,110,0.08));
        border: 1px solid var(--panel-border);
        border-radius: var(--radius-xl);
        box-shadow: var(--shadow);
        padding: 1.65rem 1.8rem;
        position: relative;
        overflow: hidden;
    }

    .uc-hero::after {
        content: "";
        position: absolute;
        inset: auto -8% -30% auto;
        width: 18rem;
        height: 18rem;
        background: radial-gradient(circle, rgba(37,99,235,0.12), transparent 68%);
        pointer-events: none;
    }

    .uc-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.4rem 0.7rem;
        background: rgba(255,255,255,0.75);
        border: 1px solid rgba(37, 99, 235, 0.12);
        border-radius: 999px;
        color: var(--blue);
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 0.95rem;
    }

    .uc-title {
        font-size: clamp(2.15rem, 3vw, 3.3rem);
        line-height: 1.02;
        font-weight: 800;
        margin: 0;
        color: var(--text);
    }

    .uc-subtitle {
        margin-top: 0.45rem;
        font-size: 1.15rem;
        color: var(--muted);
        font-weight: 500;
    }

    .uc-hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.7rem;
        margin-top: 1rem;
    }

    .uc-chip {
        padding: 0.55rem 0.8rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(148,163,184,0.18);
        color: var(--text);
        font-size: 0.88rem;
        font-weight: 600;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
    }

    .uc-story {
        background: rgba(255,255,255,0.72);
        border: 1px solid var(--panel-border);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow);
        padding: 1rem 1.1rem;
        height: 100%;
    }

    .uc-story-title, .uc-panel-title {
        font-size: 1rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 0.7rem;
    }

    .uc-story-step {
        display: grid;
        grid-template-columns: 2.1rem 1fr;
        gap: 0.7rem;
        align-items: start;
        padding: 0.75rem 0;
        border-top: 1px solid rgba(148,163,184,0.12);
    }

    .uc-story-step:first-child {
        border-top: 0;
        padding-top: 0;
    }

    .uc-story-index {
        width: 2rem;
        height: 2rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(37,99,235,0.12), rgba(15,159,110,0.14));
        color: var(--blue);
        font-weight: 800;
        font-size: 0.9rem;
    }

    .uc-story-copy strong {
        display: block;
        margin-bottom: 0.2rem;
        font-size: 0.98rem;
    }

    .uc-story-copy span {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.45;
    }

    .uc-card {
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow);
        padding: 1.05rem 1.1rem;
        transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
        backdrop-filter: blur(8px);
    }

    .uc-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 24px 44px rgba(15,23,42,0.10);
        border-color: rgba(37, 99, 235, 0.22);
    }

    .uc-card-kpi {
        min-height: 152px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .uc-kpi-label {
        color: var(--muted);
        font-size: 0.88rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .uc-kpi-value {
        font-size: clamp(1.9rem, 2.6vw, 2.8rem);
        line-height: 1;
        font-weight: 800;
        color: var(--text);
        margin: 0.55rem 0 0.4rem;
    }

    .uc-kpi-detail {
        font-size: 0.92rem;
        color: var(--muted);
        line-height: 1.4;
    }

    .uc-kpi-accent {
        width: 100%;
        height: 0.42rem;
        border-radius: 999px;
        margin-top: 0.95rem;
        background: linear-gradient(90deg, rgba(37,99,235,0.95), rgba(15,159,110,0.95));
    }

    .uc-panel {
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: var(--radius-xl);
        box-shadow: var(--shadow);
        padding: 1.1rem 1.15rem;
        margin-top: 0.9rem;
    }

    .uc-panel-compact {
        padding: 0.95rem 1rem;
        border-radius: var(--radius-lg);
    }

    .uc-panel-copy {
        color: var(--muted);
        font-size: 0.98rem;
        line-height: 1.55;
    }

    .uc-status-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin-top: 0.85rem;
    }

    .uc-status-pill {
        padding: 0.9rem 1rem;
        border-radius: 16px;
        border: 1px solid transparent;
        transition: transform 180ms ease, box-shadow 180ms ease;
    }

    .uc-status-pill:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 30px rgba(15,23,42,0.08);
    }

    .uc-status-pill.ready {
        background: linear-gradient(180deg, #f0fdf4 0%, #e7f9ef 100%);
        border-color: rgba(15,159,110,0.18);
    }

    .uc-status-pill.pending {
        background: linear-gradient(180deg, #fff7ed 0%, #fff3e6 100%);
        border-color: rgba(249,115,22,0.18);
    }

    .uc-status-label {
        font-size: 0.78rem;
        font-weight: 800;
        color: var(--muted);
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    .uc-status-headline {
        font-size: 1rem;
        color: var(--text);
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .uc-status-detail {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.45;
    }

    .uc-status-pill.compact {
        padding: 0.75rem 0.85rem;
        border-radius: 14px;
    }

    .uc-status-pill.compact .uc-status-label {
        font-size: 0.72rem;
    }

    .uc-status-pill.compact .uc-status-headline {
        font-size: 0.95rem;
        margin-bottom: 0.18rem;
    }

    .uc-status-pill.compact .uc-status-detail {
        font-size: 0.83rem;
        line-height: 1.35;
    }

    .uc-insight {
        border-radius: var(--radius-lg);
        padding: 1rem 1.05rem;
        border: 1px solid rgba(148,163,184,0.16);
        background: linear-gradient(180deg, rgba(255,255,255,0.9) 0%, rgba(248,250,252,0.96) 100%);
        box-shadow: 0 12px 28px rgba(15,23,42,0.06);
    }

    .uc-insight-title {
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0;
        text-transform: uppercase;
        color: var(--blue);
        margin-bottom: 0.45rem;
    }

    .uc-insight-copy {
        color: var(--text);
        font-size: 1rem;
        line-height: 1.55;
    }

    .uc-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 0.2rem;
    }

    .uc-legend-item {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.75rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.86);
        border: 1px solid rgba(148,163,184,0.16);
        color: var(--text);
        font-size: 0.88rem;
        font-weight: 600;
    }

    .uc-legend-swatch {
        width: 0.8rem;
        height: 0.8rem;
        border-radius: 999px;
        display: inline-block;
    }

    .uc-table-caption {
        color: var(--muted);
        font-size: 0.92rem;
        margin-top: 0.3rem;
    }

    .uc-presentation-band {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.9rem;
        margin-top: 0.85rem;
    }

    .uc-band-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(255,255,255,0.78));
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        box-shadow: 0 14px 32px rgba(15,23,42,0.06);
    }

    .uc-band-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        color: var(--muted);
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .uc-band-copy {
        font-size: 0.96rem;
        color: var(--text);
        line-height: 1.45;
        font-weight: 600;
    }

    @media (max-width: 1100px) {
        .uc-status-row,
        .uc-presentation-band {
            grid-template-columns: 1fr;
        }
    }
</style>
"""


def inject_global_styles() -> None:
    st.markdown(GLOBAL_STYLES, unsafe_allow_html=True)


def render_empty_state(title: str, body: str, steps: Iterable[str] | None = None) -> None:
    step_markup = ""
    if steps:
        items = "".join(f"<li>{step}</li>" for step in steps)
        step_markup = f"<ol style='margin:0.6rem 0 0 1.1rem;color:#5b6473;line-height:1.6;'>{items}</ol>"

    st.markdown(
        (
            "<div class='uc-panel'>"
            f"<div class='uc-panel-title'>{title}</div>"
            f"<div class='uc-panel-copy'>{body}</div>"
            f"{step_markup}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_status_pill(label: str, is_ready: bool, detail: str, compact: bool = False) -> None:
    status = "ready" if is_ready else "pending"
    headline = "Ready" if is_ready else "Pending"
    compact_class = " compact" if compact else ""
    st.markdown(
        (
            f"<div class='uc-status-pill {status}{compact_class}'>"
            f"<div class='uc-status-label'>{label}</div>"
            f"<div class='uc-status-headline'>{headline}</div>"
            f"<div class='uc-status-detail'>{detail}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    st.markdown(
        (
            "<div class='uc-sidebar-brand'>"
            "<div class='uc-sidebar-title'>UrbanCool AI</div>"
            "<div class='uc-sidebar-subtitle'>Urban Heat Intelligence Platform</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_dataframe_preview(data: pd.DataFrame, title: str, limit: int = 10, caption: str | None = None) -> None:
    render_section_header(title, caption)
    if data.empty:
        st.caption("No rows available yet.")
        return
    st.dataframe(data.head(limit), use_container_width=True, hide_index=True)


def render_section_header(title: str, subtitle: str | None = None) -> None:
    subtitle_markup = f"<div class='uc-panel-copy' style='margin-top:0.25rem;'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        (
            "<div style='margin:0.2rem 0 0.7rem;'>"
            f"<div class='uc-panel-title' style='font-size:1.15rem;'>{title}</div>"
            f"{subtitle_markup}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_hero_banner(
    title: str,
    subtitle: str,
    eyebrow: str,
    chips: Sequence[str],
) -> None:
    chip_markup = "".join(f"<span class='uc-chip'>{chip}</span>" for chip in chips)
    st.markdown(
        (
            "<div class='uc-hero'>"
            f"<div class='uc-eyebrow'>{eyebrow}</div>"
            f"<h1 class='uc-title'>{title}</h1>"
            f"<div class='uc-subtitle'>{subtitle}</div>"
            f"<div class='uc-hero-meta'>{chip_markup}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_story_panel(steps: Sequence[tuple[str, str]]) -> None:
    items = []
    for index, (title, copy) in enumerate(steps, start=1):
        items.append(
            (
                "<div class='uc-story-step'>"
                f"<div class='uc-story-index'>{index}</div>"
                "<div class='uc-story-copy'>"
                f"<strong>{title}</strong>"
                f"<span>{copy}</span>"
                "</div></div>"
            )
        )
    st.markdown(
        (
            "<div class='uc-story'>"
            "<div class='uc-story-title'>Guided Walkthrough</div>"
            f"{''.join(items)}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, detail: str, accent: str = "linear-gradient(90deg, rgba(37,99,235,0.95), rgba(15,159,110,0.95))") -> None:
    st.markdown(
        (
            "<div class='uc-card uc-card-kpi'>"
            f"<div class='uc-kpi-label'>{label}</div>"
            f"<div class='uc-kpi-value'>{value}</div>"
            f"<div class='uc-kpi-detail'>{detail}</div>"
            f"<div class='uc-kpi-accent' style='background:{accent};'></div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_insight_card(title: str, copy: str, tone: str = "blue") -> None:
    palette = {
        "blue": "#2563eb",
        "green": "#0f9f6e",
        "orange": "#f97316",
        "red": "#dc2626",
    }
    color = palette.get(tone, palette["blue"])
    st.markdown(
        (
            "<div class='uc-insight'>"
            f"<div class='uc-insight-title' style='color:{color};'>{title}</div>"
            f"<div class='uc-insight-copy'>{copy}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def open_panel(title: str, subtitle: str | None = None, compact: bool = False) -> None:
    subtitle_markup = f"<div class='uc-panel-copy' style='margin-top:0.18rem;'>{subtitle}</div>" if subtitle else ""
    css_class = "uc-panel uc-panel-compact" if compact else "uc-panel"
    st.markdown(
        (
            f"<div class='{css_class}'>"
            f"<div class='uc-panel-title'>{title}</div>"
            f"{subtitle_markup}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_legend(items: Sequence[tuple[str, str]]) -> None:
    markup = "".join(
        (
            "<span class='uc-legend-item'>"
            f"<span class='uc-legend-swatch' style='background:{color};'></span>"
            f"{label}</span>"
        )
        for label, color in items
    )
    st.markdown(f"<div class='uc-legend'>{markup}</div>", unsafe_allow_html=True)


def render_presentation_band(items: Sequence[tuple[str, str]]) -> None:
    markup = "".join(
        (
            "<div class='uc-band-card'>"
            f"<div class='uc-band-title'>{title}</div>"
            f"<div class='uc-band-copy'>{copy}</div>"
            "</div>"
        )
        for title, copy in items
    )
    st.markdown(f"<div class='uc-presentation-band'>{markup}</div>", unsafe_allow_html=True)
