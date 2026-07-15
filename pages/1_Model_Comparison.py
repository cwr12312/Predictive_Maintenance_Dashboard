"""Page 2 (nav) — Model Comparison: ranks the six models with interactive charts."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from config import PROJECT_SHORT_TITLE, COLORS, PLOTLY_TEMPLATE, MODEL_NAMES
from utils.styling import inject_css, page_header, section_title, medal, footer
from utils.data_loader import load_metrics

st.set_page_config(page_title=f"Model Comparison · {PROJECT_SHORT_TITLE}", layout="wide")
inject_css()
page_header("Model Comparison", "Ranking the six approved deep-learning & meta-learning models")

metrics = load_metrics().sort_values("computed_rank")

# --------------------------------------------------------------------------
# RANKED TABLE
# --------------------------------------------------------------------------
section_title("Leaderboard")
cols = st.columns([0.5, 2, 1.2, 1.2, 1.2, 1.2, 1.4])
headers = ["Rank", "Model", "Accuracy", "Precision", "Recall", "F1 Score", "Inference (est.)"]
for c, h in zip(cols, headers):
    c.markdown(f"**{h}**")

for _, row in metrics.iterrows():
    cols = st.columns([0.5, 2, 1.2, 1.2, 1.2, 1.2, 1.4])
    cols[0].markdown(medal(int(row["computed_rank"])), unsafe_allow_html=True)
    muted = COLORS['text_muted']
    cols[1].markdown(f"**{row['model_name']}**  \n<span style='color:{muted};font-size:0.78rem;'>{row['model_type']} · {row['framework']}</span>",
                      unsafe_allow_html=True)
    cols[2].markdown(f"{row['accuracy']*100:.2f}%")
    cols[3].markdown(f"{row['precision']*100:.2f}%")
    cols[4].markdown(f"{row['recall']*100:.2f}%")
    cols[5].markdown(f"{row['f1_score']*100:.2f}%")
    cols[6].markdown(f"{row['inference_time_ms']:.1f} ms")

st.write("")
st.write("")

# --------------------------------------------------------------------------
# CHART GRID
# --------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [" Bar Comparison", " Radar Chart", " Heatmap", " Scatter", " Bubble"]
)

metric_cols = ["accuracy", "precision", "recall", "f1_score"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

with tab1:
    section_title("Metric-by-Metric Bar Comparison")
    fig = go.Figure()
    for m, lbl in zip(metric_cols, metric_labels):
        fig.add_trace(go.Bar(name=lbl, x=metrics["model_name"], y=metrics[m] * 100))
    fig.update_layout(barmode="group", template=PLOTLY_TEMPLATE, height=460,
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       yaxis_title="Score (%)", legend_title="Metric")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    section_title("Radar Chart — Multi-Metric Profile")
    fig = go.Figure()
    for _, row in metrics.iterrows():
        vals = [row[m] * 100 for m in metric_cols] + [row[metric_cols[0]] * 100]
        fig.add_trace(go.Scatterpolar(r=vals, theta=metric_labels + [metric_labels[0]],
                                       fill="toself", name=row["model_name"], opacity=0.55))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=520,
                       polar=dict(radialaxis=dict(visible=True, range=[85, 100])),
                       paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    section_title("Performance Heatmap")
    heat_df = metrics.set_index("model_name")[metric_cols] * 100
    heat_df.columns = metric_labels
    fig = px.imshow(heat_df.values, x=heat_df.columns, y=heat_df.index, text_auto=".2f",
                     color_continuous_scale="Blues", aspect="auto")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=420, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    section_title("Accuracy vs. Inference Time")
    fig = px.scatter(metrics, x="inference_time_ms", y="accuracy", text="model_name",
                      color="model_type", size="f1_score", size_max=28,
                      labels={"inference_time_ms": "Inference Time (ms, estimated)", "accuracy": "Accuracy"})
    fig.update_traces(textposition="top center")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=460, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    section_title("Bubble Plot — Precision vs Recall (bubble = F1)")
    fig = px.scatter(metrics, x="precision", y="recall", size="f1_score", color="model_name",
                      size_max=45, hover_data=["accuracy"])
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=460, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

footer()
