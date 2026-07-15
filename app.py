"""
app.py
======
Entry point for the Industrial Predictive Maintenance Dashboard.
This file renders Page 1 — Executive Dashboard. Pages 2-8 live under
`pages/` and are auto-discovered by Streamlit's native multipage router.

"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from config import (
    PROJECT_SHORT_TITLE, COLORS, PLOTLY_TEMPLATE, N_CLASSES, N_FEATURES,
    TARGET_ACCURACY, DATASET_INFO, MODEL_NAMES,
)
from utils.styling import inject_css, page_header, section_title, kpi_card, footer
from utils.data_loader import load_metrics, best_model_row
from utils.model_loader import load_all_models, get_load_errors
from utils.benchmark import benchmark_all_models

st.set_page_config(
    page_title=PROJECT_SHORT_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center;padding:10px 0 18px 0;">
            <div style="font-size:2.2rem;">⚙️</div>
            <div style="font-weight:800;font-size:1.05rem;color:{COLORS['text_primary']};">
                PREDICTIVE MAINTENANCE
            </div>
            <div style="font-size:0.72rem;color:{COLORS['accent_blue_light']};
                        letter-spacing:0.1em;">BEARING FAULT DIAGNOSTICS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("SYSTEM STATUS")
    registry = load_all_models()
    n_ok = len(registry["models"])
    n_err = len(registry["errors"])
    if n_err == 0:
        st.success(f" {n_ok}/6 models loaded")
    else:
        st.warning(f" {n_ok}/6 models loaded, {n_err} failed")
        with st.expander("Load errors"):
            for name, msg in registry["errors"].items():
                st.caption(f"**{name}**: {msg}")
    st.markdown("---")
    st.caption("NAVIGATION")
    st.page_link("app.py", label="Executive Dashboard")
    st.page_link("pages/1_Model_Comparison.py", label="Model Comparison")
    st.page_link("pages/2_Live_Prediction.py", label="Live Prediction")
    st.page_link("pages/4_Maintenance_Recommendations.py", label="Maintenance Recs")
    st.page_link("pages/5_Dataset_Explorer.py", label="Dataset Explorer")
    st.page_link("pages/6_Model_Performance.py", label="Model Performance")
    st.page_link("pages/7_About.py", label="About")

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
page_header(
    "Executive Dashboard",
    "Real-time overview of the six-model bearing fault diagnosis system"
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
    kpi_card("Total Models", "6", "2D CNN · LSTM · Transformer · MAML · Meta-SGD · FBCL")
with c2:
    kpi_card("Best Performing Model", best["model_name"], f"Rank #1 · avg score {best['average_score']:.4f}")
with c3:
    kpi_card("Highest Accuracy", f"{best['accuracy']*100:.2f}%",
              f"Target ≥ {TARGET_ACCURACY*100:.0f}%",
              delta="Target met" if best["accuracy"] >= TARGET_ACCURACY else "Below target",
              delta_positive=best["accuracy"] >= TARGET_ACCURACY)
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
left, right = st.columns([1.3, 1])

with left:
    section_title("Model Accuracy Comparison")
    df_sorted = metrics.sort_values("accuracy", ascending=True)
    colors = [COLORS["accent_green"] if m == best["model_name"] else COLORS["accent_blue"]
              for m in df_sorted["model_name"]]
    fig = go.Figure(go.Bar(
        x=df_sorted["accuracy"] * 100, y=df_sorted["model_name"], orientation="h",
        marker_color=colors, text=[f"{v*100:.2f}%" for v in df_sorted["accuracy"]],
        textposition="outside",
    ))
    fig.add_vline(x=TARGET_ACCURACY * 100, line_dash="dash", line_color=COLORS["accent_amber"],
                  annotation_text="95% Target")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=380, margin=dict(l=10, r=10, t=10, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       xaxis_title="Accuracy (%)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

with right:
    section_title("Model Type Distribution")
    type_counts = metrics["model_type"].value_counts().reset_index()
    type_counts.columns = ["model_type", "count"]
    fig2 = px.pie(type_counts, names="model_type", values="count", hole=0.55,
                  color_discrete_sequence=[COLORS["accent_blue"], COLORS["accent_cyan"], COLORS["accent_amber"]])
    fig2.update_traces(textinfo="label+value")
    fig2.update_layout(template=PLOTLY_TEMPLATE, height=380, margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

section_title("All Models — Snapshot")
display_df = metrics[["model_name", "model_type", "framework", "accuracy", "precision",
                       "recall", "f1_score", "computed_rank"]].copy()
display_df.columns = ["Model", "Type", "Framework", "Accuracy", "Precision", "Recall", "F1 Score", "Rank"]
for c in ["Accuracy", "Precision", "Recall", "F1 Score"]:
    display_df[c] = (display_df[c] * 100).round(2).astype(str) + "%"
st.dataframe(display_df.sort_values("Rank"), use_container_width=True, hide_index=True)

footer()
