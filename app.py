"""
app.py
======
Entry point for the Industrial Predictive Maintenance Dashboard.
This is the "App" page the main landing page and it renders the
Executive Dashboard overview. Additional pages live under `pages/` and
are auto-discovered by Streamlit's native multipage router; the shared
sidebar navigation and page order are defined in utils/styling.render_sidebar.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from config import (
    PROJECT_SHORT_TITLE, APP_NAME, COLORS, PLOTLY_TEMPLATE, N_CLASSES, N_FEATURES,
    TARGET_ACCURACY, DATASET_INFO, MODEL_NAMES,
)
from utils.styling import inject_css, page_header, section_title, kpi_card, footer, render_sidebar
from utils.data_loader import load_metrics, best_model_row
from utils.benchmark import benchmark_all_models

st.set_page_config(
    page_title=PROJECT_SHORT_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# --------------------------------------------------------------------------
# SIDEBAR (shared across every page — see utils/styling.render_sidebar)
# --------------------------------------------------------------------------
render_sidebar(active="app")

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
page_header(
    "Executive Dashboard",
    "Real-time overview of the six-model bearing fault diagnosis system"
)

st.markdown(
    f"""
    <div class="panel-card" style="margin-bottom:22px;">
        <b>{APP_NAME}</b> is a production-style monitoring console built around the
        <b>Case Western Reserve University (CWRU) Bearing Data Center</b> dataset the
        standard benchmark for rolling-element bearing fault detection. It compares
        six deep-learning and meta-learning models side by side
        <b>Two-Dimensional Convolutional Neural Network (2D CNN), Long Short-Term Memory (LSTM), Transformer,
        Model-Agnostic Meta-Learning (MAML), Meta Stochastic Gradient Descent (Meta-SGD) and
        Feature-Based Contrastive Learning (FBCL)</b> spanning both
        traditional supervised deep learning and meta-continual-learning approaches
        to bearing fault classification and predictive maintenance.
    </div>
    """,
    unsafe_allow_html=True,
)

metrics = load_metrics()
best = best_model_row(metrics)
bench = benchmark_all_models()
measured = bench.set_index("model_name")["measured_inference_ms"].to_dict() if not bench.empty else {}

# --------------------------------------------------------------------------
# KPI ROW 1
# --------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Total Models", "6", "<span style='font-size:0.68rem;'>Two-Dimensional Convolutional Neural Network (2D CNN) · Long Short-Term Memory (LSTM) · Transformer · Model-Agnostic Meta-Learning (MAML) · Meta Stochastic Gradient Descent (Meta-SGD) · Feature-Based Contrastive Learning (FBCL)</span>")
with c2:
    kpi_card("Best Performing Model", f"<span style='font-size:1.05rem;'>{best['model_name']}</span>", f"Rank #1 · avg score {best['average_score']:.4f}")
with c3:
    # Compare the displayed (rounded) accuracy against the displayed target.
    displayed_accuracy = round(best["accuracy"] * 100)
    displayed_target = round(TARGET_ACCURACY * 100)

    target_met = displayed_accuracy >= displayed_target

    delta_label = "Target met" if target_met else (
        f"{displayed_target - displayed_accuracy}% below target"
    )

    kpi_card(
        "Highest Accuracy",
        f"{best['accuracy'] * 100:.2f}%",
        f"Target ≥ {TARGET_ACCURACY * 100:.0f}%",
        delta=delta_label,
        delta_positive=target_met
    )
with c4:
    kpi_card("Number of Classes", str(N_CLASSES), "10 bearing health states")

st.write("")
c5, c6, c7, c8 = st.columns(4)
with c5:
    kpi_card("Precision (Best Model)", f"{best['precision']*100:.2f}%", best["model_name"])
with c6:
    kpi_card("Recall (Best Model)", f"{best['recall']*100:.2f}%", best["model_name"])
with c7:
    kpi_card("F1 Score (Best Model)", f"{best['f1_score']*100:.2f}%", best["model_name"])
with c8:
    best_measured = measured.get(best["model_name"])
    inf_val = f"{best_measured:.2f} ms" if best_measured is not None and best_measured == best_measured else f"~{best['inference_time_ms']:.1f} ms"
    kpi_card("Inference Time", inf_val, "Measured on this machine" if best_measured else "Estimated")

st.write("")
c9, c10 = st.columns(2)
with c9:
    kpi_card("Number of Features", str(N_FEATURES), "9 base statistical + 10 engineered")
with c10:
    kpi_card("Dataset", "CWRU Bearing Data Center", DATASET_INFO["Augmented training samples"])

st.write("")
st.write("")

# --------------------------------------------------------------------------
# CHARTS
# --------------------------------------------------------------------------

# Model Type Distribution centered alone
left, center, right = st.columns([1, 1.3, 1])

with center:
    section_title("Model Type Distribution")
    type_counts = metrics["model_type"].value_counts().reset_index()
    type_counts.columns = ["model_type", "count"]
    fig2 = px.pie(type_counts, names="model_type", values="count", hole=0.55,
                  color_discrete_sequence=[COLORS["accent_blue"], COLORS["accent_cyan"], COLORS["accent_amber"]])
    fig2.update_traces(textinfo="label+value")
    fig2.update_layout(template=PLOTLY_TEMPLATE, height=380, margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

footer()