"""
utils/styling.py
=================
CSS injection and small reusable HTML component builders (KPI cards,
badges, section titles, alert cards) used across every page so the look
stays consistent without duplicating markup.
"""

import streamlit as st
from config import CSS_DIR, COLORS


def inject_css():
    css_path = CSS_DIR / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", icon: str = ""):
    st.markdown(
        f"""
        <div style="margin-bottom: 6px;">
            <div style="font-size:0.78rem;font-weight:700;letter-spacing:0.12em;
                        color:{COLORS['accent_blue_light']};text-transform:uppercase;">
                {icon} INDUSTRIAL PREDICTIVE MAINTENANCE
            </div>
            <h1 style="margin:2px 0 2px 0;">{title}</h1>
            <div style="color:{COLORS['text_secondary']};font-size:0.95rem;">{subtitle}</div>
        </div>
        <hr style="margin:10px 0 22px 0;">
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str):
    st.markdown(
        f'<div class="section-title"><div class="bar"></div><h3>{text}</h3></div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "", delta: str = None, delta_positive: bool = True):
    delta_html = ""
    if delta:
        cls = "kpi-delta-up" if delta_positive else "kpi-delta-down"
        arrow = "▲" if delta_positive else "▼"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "blue"):
    return f'<span class="badge badge-{kind}">{text}</span>'


def medal(rank: int) -> str:
    return {1: '<span class="rank-gold">🥇</span>', 2: '<span class="rank-silver">🥈</span>',
            3: '<span class="rank-bronze">🥉</span>'}.get(rank, f"#{rank}")


def alert_card(title: str, items: list[str], risk: str = "low"):
    risk_class = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high",
                  "Critical": "risk-critical"}.get(risk, "risk-low")
    items_html = "".join(f'<div class="alert-item">• {i}</div>' for i in items)
    st.markdown(
        f"""
        <div class="alert-card {risk_class}">
            <div class="alert-title">{title}</div>
            {items_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer():
    st.markdown(
        """
        <div class="app-footer">
            Industrial Predictive Maintenance System for Bearing Fault Diagnosis &middot;
            Built with Streamlit, TensorFlow, PyTorch &amp; Plotly
        </div>
        """,
        unsafe_allow_html=True,
    )
