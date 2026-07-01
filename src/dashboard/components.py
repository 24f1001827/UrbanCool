from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
import streamlit as st


GLOBAL_STYLES = """
<style>
    :root {
        --bg: #f8fafc;
        --panel: rgba(255,255,255,0.96);
        --panel-strong: #ffffff;
        --panel-border: rgba(148, 163, 184, 0.20);
        --surface: rgba(255,255,255,0.98);
        --surface-soft: rgba(248,250,252,0.98);
        --text: #111827;
        --muted: #4b5563;
        --muted-soft: #6b7280;
        --blue: #2563eb;
        --blue-soft: #dbeafe;
        --green: #16a34a;
        --green-soft: #dcfce7;
        --orange: #f97316;
        --orange-soft: #ffedd5;
        --danger: #ef4444;
        --danger-soft: #fee2e2;
        --heat-1: #fff7ed;
        --heat-2: #fed7aa;
        --heat-3: #fb923c;
        --heat-4: #ea580c;
        --heat-5: #b91c1c;
        --shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
        --shadow-soft: 0 8px 20px rgba(15, 23, 42, 0.05);
        --radius-xl: 22px;
        --radius-lg: 18px;
        --radius-md: 14px;
        --radius-sm: 10px;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(37, 99, 235, 0.05), transparent 28%),
            radial-gradient(circle at top right, rgba(22, 163, 74, 0.04), transparent 30%),
            linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        color: var(--text);
    }

    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        display: none;
    }

    .block-container {
        padding-top: 0.35rem;
        padding-bottom: 1.5rem;
        max-width: 1480px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
        min-width: 20.5rem;
        width: 20.5rem;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 0.9rem;
        padding-left: 0.9rem;
        padding-right: 0.9rem;
    }

    [data-testid="stSidebar"]  {
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
        gap: 0.52rem;
    }

    [data-testid="stSidebar"] .stRadio label {
        position: relative;
        background: rgba(255,255,255,0.78);
        border: 1px solid rgba(148,163,184,0.16);
        border-radius: 14px;
        padding: 0.84rem 2.4rem 0.84rem 3.2rem;
        box-shadow: 0 6px 16px rgba(15,23,42,0.04);
        transition: transform 180ms ease, background 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
        overflow: hidden;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(37, 99, 235, 0.08);
        border-color: rgba(37,99,235,0.22);
        box-shadow: 0 14px 26px rgba(15,23,42,0.08);
        transform: translateX(3px);
    }

    [data-testid="stSidebar"] .stRadio label p {
        font-size: 1.02rem !important;
        font-weight: 700 !important;
        line-height: 1.2;
    }

    [data-testid="stSidebar"] .stRadio label::before {
        content: "";
        position: absolute;
        left: 0.25rem;
        top: 0.45rem;
        bottom: 0.45rem;
        width: 0.28rem;
        border-radius: 999px;
        background: transparent;
        transition: background 180ms ease;
    }

    [data-testid="stSidebar"] .stRadio label::after {
        content: "›";
        position: absolute;
        right: 0.95rem;
        top: 50%;
        transform: translateY(-52%);
        color: var(--muted-soft);
        font-size: 1.35rem;
        font-weight: 400;
        line-height: 1;
    }

    [data-testid="stSidebar"] .stRadio label:nth-of-type(1)::before,
    [data-testid="stSidebar"] .stRadio label:nth-of-type(2)::before,
    [data-testid="stSidebar"] .stRadio label:nth-of-type(3)::before,
    [data-testid="stSidebar"] .stRadio label:nth-of-type(4)::before,
    [data-testid="stSidebar"] .stRadio label:nth-of-type(5)::before,
    [data-testid="stSidebar"] .stRadio label:nth-of-type(6)::before,
    [data-testid="stSidebar"] .stRadio label:nth-of-type(7)::before {
        width: 2rem;
        height: 2rem;
        left: 0.75rem;
        top: 50%;
        bottom: auto;
        transform: translateY(-50%);
        border-radius: 999px;
    }

    [data-testid="stSidebar"] .stRadio label:nth-of-type(1)::before { background: linear-gradient(135deg, #fee2e2, #fecaca); }
    [data-testid="stSidebar"] .stRadio label:nth-of-type(2)::before { background: linear-gradient(135deg, #dbeafe, #bfdbfe); }
    [data-testid="stSidebar"] .stRadio label:nth-of-type(3)::before { background: linear-gradient(135deg, #e0e7ff, #c7d2fe); }
    [data-testid="stSidebar"] .stRadio label:nth-of-type(4)::before { background: linear-gradient(135deg, #ede9fe, #ddd6fe); }
    [data-testid="stSidebar"] .stRadio label:nth-of-type(5)::before { background: linear-gradient(135deg, #dbeafe, #bfdbfe); }
    [data-testid="stSidebar"] .stRadio label:nth-of-type(6)::before { background: linear-gradient(135deg, #e2e8f0, #cbd5e1); }
    [data-testid="stSidebar"] .stRadio label:nth-of-type(7)::before { background: linear-gradient(135deg, #dcfce7, #bbf7d0); }

    [data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.96));
        border-color: rgba(248, 113, 113, 0.18);
        box-shadow: 0 18px 28px rgba(248, 113, 113, 0.10);
        transform: translateX(3px);
    }

    [data-testid="stSidebar"] .stRadio label:has(input:checked)::before {
        background: linear-gradient(180deg, rgba(248,63,43,0.96), rgba(255,158,95,0.96));
    }

    [data-testid="stSidebar"] .stRadio label:has(input:checked) p {
        color: #ef4444 !important;
    }

    [data-testid="stSidebar"] .stRadio label:hover p {
        color: var(--text) !important;
    }

    .uc-sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        padding: 0.15rem 0.1rem 0.95rem;
        margin-bottom: 0.4rem;
    }

    .uc-sidebar-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: var(--text);
        margin: 0;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }

    .uc-sidebar-subtitle {
        margin-top: 0.2rem;
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.35;
    }

    .uc-sidebar-section {
        margin: 0.75rem 0 0.55rem;
        font-size: 0.82rem;
        font-weight: 800;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.06em;
    }

    .uc-brand-mark {
        width: 2.7rem;
        height: 2.7rem;
        flex: 0 0 auto;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }

    .uc-brand-mark svg {
        display: block;
        width: 100%;
        height: 100%;
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

    .uc-page-header {
        margin: 0.2rem 0 0.8rem;
    }

    .uc-page-title {
        font-size: clamp(2.8rem, 4.2vw, 4.2rem);
        line-height: 0.98;
        letter-spacing: -0.04em;
        font-weight: 800;
        margin: 0;
        color: var(--text);
    }

    .uc-page-subtitle {
        margin-top: 0.6rem;
        font-size: 1.05rem;
        line-height: 1.45;
        color: var(--muted);
        font-weight: 500;
        max-width: 52rem;
    }

    .uc-chip-row {
        display: inline-flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-top: 1rem;
    }

    .uc-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.5rem 0.85rem;
        background: rgba(255,255,255,0.9);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 999px;
        color: var(--blue);
        font-size: 0.88rem;
        font-weight: 700;
        box-shadow: var(--shadow-soft);
    }

    .uc-chip {
        padding: 0.62rem 1rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.94);
        border: 1px solid rgba(148,163,184,0.18);
        color: var(--text);
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: var(--shadow-soft);
    }

    .uc-summary-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin-top: 1rem;
        align-items: stretch;
    }

    .uc-summary-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.97), rgba(248,250,252,0.96));
        border: 1px solid rgba(148,163,184,0.16);
        border-radius: 18px;
        box-shadow: var(--shadow-soft);
        padding: 0.95rem 1rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }

    .uc-summary-label {
        font-size: 0.76rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
    }

    .uc-summary-value {
        margin-top: 0.25rem;
        font-size: 1.08rem;
        line-height: 1.3;
        font-weight: 800;
        color: var(--text);
    }

    .uc-summary-detail {
        margin-top: 0.24rem;
        font-size: 0.9rem;
        line-height: 1.4;
        color: var(--muted);
    }

    .uc-panel-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: var(--text);
        margin-bottom: 0.35rem;
        letter-spacing: -0.01em;
    }

    .uc-story {
        background: rgba(255,255,255,0.86);
        border: 1px solid var(--panel-border);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-soft);
        padding: 1rem 1.05rem;
    }

    .uc-story-step {
        display: grid;
        grid-template-columns: 2.1rem 1fr;
        gap: 0.7rem;
        align-items: start;
        padding: 0.75rem 0;
        border-top: 1px solid rgba(148,163,184,0.12);
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
        min-height: 228px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        gap: 0.7rem;
        position: relative;
        overflow: hidden;
    }

    .uc-kpi-label {
        color: var(--muted);
        font-size: 0.86rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .uc-kpi-top {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        min-height: 3.2rem;
    }

    .uc-kpi-icon {
        width: 2.8rem;
        height: 2.8rem;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        font-weight: 800;
        color: var(--text);
        background: rgba(255,255,255,0.86);
        box-shadow: inset 0 0 0 1px rgba(148,163,184,0.14);
    }

    .uc-kpi-value {
        font-size: clamp(2.25rem, 3vw, 3.25rem);
        line-height: 1;
        font-weight: 800;
        color: var(--text);
        margin: 0.15rem 0 0.1rem;
        letter-spacing: -0.04em;
        min-height: 4.8rem;
        display: flex;
        align-items: flex-start;
        word-break: break-word;
    }

    .uc-kpi-detail {
        font-size: 0.94rem;
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
        padding: 1rem 1.05rem;
        margin-top: 0.75rem;
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
        position: relative;
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

    .uc-presentation-band-shell {
        display: flex;
        width: 100%;
        justify-content: center;
    }

    .uc-presentation-band {
        width: 100%;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
        margin-top: 0.85rem;
        align-items: stretch;
    }

    .uc-band-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(255,255,255,0.78));
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        box-shadow: 0 14px 32px rgba(15,23,42,0.06);
        height: 100%;
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

    .uc-hero-layout {
        display: grid;
        grid-template-columns: minmax(0, 1.35fr) minmax(300px, 0.95fr);
        gap: 1rem;
        align-items: stretch;
    }

    .uc-hero-copy {
        position: relative;
        z-index: 1;
    }

    .uc-hero-summary {
        display: grid;
        gap: 0.75rem;
    }

    .uc-hero-summary-card {
        background: rgba(255,255,255,0.86);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 18px;
        padding: 0.85rem 0.95rem;
        box-shadow: 0 14px 30px rgba(15,23,42,0.06);
    }

    .uc-hero-summary-label {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.02em;
        font-weight: 800;
        color: var(--muted);
    }

    .uc-hero-summary-value {
        margin-top: 0.3rem;
        font-size: 1.08rem;
        line-height: 1.25;
        color: var(--text);
        font-weight: 800;
    }

    .uc-hero-summary-detail {
        margin-top: 0.25rem;
        font-size: 0.9rem;
        line-height: 1.4;
        color: var(--muted);
    }

    section.main [data-testid="stWidgetLabel"],
    section.main [data-testid="stWidgetLabel"] *,
    section.main [data-testid="stWidgetLabel"] p,
    section.main [data-testid="stWidgetLabel"] span,
    section.main label,
    section.main .stSlider p,
    section.main .stSelectbox p,
    section.main .stMultiSelect p,
    section.main .stNumberInput p,
    section.main .stCaption,
    section.main .stMarkdown p,
    section.main .stMarkdown span {
        color: var(--text) !important;
        opacity: 1 !important;
    }

    section.main [data-baseweb="select"] > div,
    section.main [data-baseweb="input"] > div,
    section.main [data-baseweb="textarea"] > div {
        background: rgba(255,255,255,0.9);
        border-color: rgba(148,163,184,0.24);
        color: var(--text);
    }

    section.main [data-baseweb="select"] * {
        color: var(--text) !important;
    }

    section.main [data-testid="stSlider"] *,
    section.main [data-testid="stSlider"] label {
        color: var(--text) !important;
    }

    section.main [data-testid="stSelectbox"] *,
    section.main [data-testid="stMultiSelect"] *,
    section.main [data-testid="stNumberInput"] *,
    section.main [data-testid="stTextInput"] *,
    section.main [data-testid="stDateInput"] *,
    section.main [data-testid="stTimeInput"] * {
        color: var(--text) !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 10px 24px rgba(15,23,42,0.05);
    }

    [data-testid="stDataFrame"] > div {
        width: 100%;
    }

    @media (max-width: 1100px) {
        .uc-status-row,
        .uc-presentation-band,
        .uc-summary-grid {
            grid-template-columns: 1fr;
        }

        [data-testid="stSidebar"] {
            min-width: auto;
            width: auto;
        }
    }
    /* Fix Streamlit filter labels */

[data-testid="stWidgetLabel"] {
    color: #111827 !important;
}

[data-testid="stWidgetLabel"] p {
    color: #111827 !important;
}

[data-testid="stWidgetLabel"] span {
    color: #111827 !important;
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
    icon = "✓" if is_ready else "•"
    border_color = "rgba(15,159,110,0.7)" if is_ready else "rgba(249,115,22,0.65)"
    text_color = "rgba(15,159,110,0.95)" if is_ready else "rgba(249,115,22,0.9)"
    st.markdown(
        (
            f"<div class='uc-status-pill {status}{compact_class}'>"
            f"<div class='uc-status-label'>{label}</div>"
            f"<div style='position:absolute;right:0.95rem;top:0.85rem;width:1.35rem;height:1.35rem;border-radius:999px;border:1.5px solid {border_color};color:{text_color};display:flex;align-items:center;justify-content:center;font-size:0.84rem;font-weight:800;background:rgba(255,255,255,0.74);'>{icon}</div>"
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
            "<div class='uc-brand-mark'>"
            "<svg viewBox='0 0 64 64' fill='none' xmlns='http://www.w3.org/2000/svg' aria-hidden='true'>"
            "<path d='M32 7C25 13.5 18.5 21.5 18.5 31.2C18.5 41.9 25.9 49.5 32 57C38.1 49.5 45.5 41.9 45.5 31.2C45.5 21.5 39 13.5 32 7Z' fill='#10b981' fill-opacity='0.12'/>"
            "<path d='M32 8.5C26.4 14 21.5 20.8 21.5 30.2C21.5 39.2 27.3 46 32 52.1C36.7 46 42.5 39.2 42.5 30.2C42.5 20.8 37.6 14 32 8.5Z' stroke='#10b981' stroke-width='3.2' stroke-linecap='round' stroke-linejoin='round'/>"
            "<path d='M32 24V40' stroke='#10b981' stroke-width='3.2' stroke-linecap='round'/>"
            "<path d='M24.5 31H39.5' stroke='#10b981' stroke-width='3.2' stroke-linecap='round'/>"
            "<path d='M18.8 45.4C24.2 46.6 28.2 46.7 32 43.8C35.8 46.7 39.8 46.6 45.2 45.4' stroke='#059669' stroke-width='2.6' stroke-linecap='round'/>"
            "</svg>"
            "</div>"
            "<div>"
            "<div class='uc-sidebar-title'>UrbanCool AI</div>"
            "<div class='uc-sidebar-subtitle'>Urban Heat Intelligence Platform</div>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_dataframe_preview(
    data: pd.DataFrame,
    title: str,
    limit: int = 10,
    caption: str | None = None,
    height: int | None = None,
) -> None:
    render_section_header(title, caption)
    if data.empty:
        st.caption("No rows available yet.")
        return
    dataframe_kwargs = {"use_container_width": True, "hide_index": True}
    if height is not None:
        dataframe_kwargs["height"] = height
    st.dataframe(data.head(limit), **dataframe_kwargs)


def render_section_header(title: str, subtitle: str | None = None) -> None:
    subtitle_markup = f"<div class='uc-panel-copy' style='margin-top:0.28rem;color:var(--muted);'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        (
            "<div style='margin:0.05rem 0 0.7rem;'>"
            f"<div class='uc-panel-title'>{title}</div>"
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
    summary_items: Sequence[tuple[str, str, str]] | None = None,
) -> None:
    chip_markup = "".join(f"<span class='uc-chip'>{chip}</span>" for chip in chips)
    eyebrow_markup = f"<div class='uc-eyebrow'>{eyebrow}</div>" if eyebrow else ""
    summary_markup = ""
    if summary_items:
        summary_markup = (
            "<div class='uc-summary-grid'>"
            + "".join(
                (
                    "<div class='uc-summary-card'>"
                    f"<div class='uc-summary-label'>{label}</div>"
                    f"<div class='uc-summary-value'>{value}</div>"
                    f"<div class='uc-summary-detail'>{detail}</div>"
                    "</div>"
                )
                for label, value, detail in summary_items
            )
            + "</div>"
        )
    st.markdown(
        (
            "<div class='uc-page-header'>"
            f"{eyebrow_markup}"
            f"<h1 class='uc-page-title'>{title}</h1>"
            f"<div class='uc-page-subtitle'>{subtitle}</div>"
            f"<div class='uc-chip-row'>{chip_markup}</div>"
            f"{summary_markup}"
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
            "<div class='uc-story-title'>Key Signals</div>"
            f"{''.join(items)}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, detail: str, accent: str = "linear-gradient(90deg, rgba(37,99,235,0.95), rgba(15,159,110,0.95))") -> None:
    icon_lookup = {
        "Mean Land Surface Temperature": "🌡",
        "Hotspot Count": "🔥",
        "High Risk Wards": "▣",
        "Dominant Driver": "↗",
        "Model Coverage": "◫",
        "Ward Rank": "1",
        "Mean LST": "🌡",
        "Hotspot Share": "%",
        "Sample Points": "◌",
        "Leading Global Driver": "↗",
        "Mean SHAP Magnitude": "∑",
        "Ward Insight Coverage": "◫",
        "Displayed Points": "◌",
        "Peak Hotspot Score": "⚑",
        "Allocated Budget": "₹",
        "Remaining Budget": "₹",
        "Plan Items": "▦",
        "Total Cooling": "↧",
        "Estimated Cooling": "↧",
        "Affected Area": "◌",
        "Confidence": "✓",
        "Coverage": "◫",
    }
    icon = icon_lookup.get(label, "•")
    st.markdown(
        (
            "<div class='uc-card uc-card-kpi'>"
            f"<div class='uc-kpi-top'><div class='uc-kpi-icon'>{icon}</div><div class='uc-kpi-label'>{label}</div></div>"
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


def render_presentation_band(items: Sequence[tuple[str, str]], centered: bool = True) -> None:
    markup = "".join(
        (
            "<div class='uc-band-card'>"
            f"<div class='uc-band-title'>{title}</div>"
            f"<div class='uc-band-copy'>{copy}</div>"
            "</div>"
        )
        for title, copy in items
    )
    shell_class = "uc-presentation-band-shell" if centered else ""
    st.markdown(f"<div class='{shell_class}'><div class='uc-presentation-band'>{markup}</div></div>", unsafe_allow_html=True)
