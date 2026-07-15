"""Page 3 (nav) — Live Prediction: upload CSV or enter features manually, run any of the six models."""

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from config import (
    PROJECT_SHORT_TITLE, COLORS, PLOTLY_TEMPLATE, MODEL_NAMES, CLASS_NAMES,
    BASE_FEATURES, FEATURE_DESCRIPTIONS, CLASS_DISPLAY_NAMES,
)
from utils.styling import inject_css, page_header, section_title, footer, badge
from utils.model_loader import get_model, get_load_errors
from utils.preprocessing import preprocess_manual_entry, preprocess_csv_upload
from utils.maintenance import get_recommendation

st.set_page_config(page_title=f"Live Prediction · {PROJECT_SHORT_TITLE}", layout="wide")
inject_css()
page_header("Live Prediction", "Run real inference with any of the six trained models")

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# --------------------------------------------------------------------------
# MODEL SELECTOR
# --------------------------------------------------------------------------
section_title("1 · Select a Model")
model_name = st.selectbox("Model", MODEL_NAMES, index=1, label_visibility="collapsed")
wrapper = get_model(model_name)
errors = get_load_errors()

if wrapper is None:
    st.error(f" **{model_name}** failed to load and cannot be used for prediction.\n\n"
              f"Error detail: `{errors.get(model_name, 'unknown error')}`")
    st.stop()
else:
    st.markdown(badge(f"{model_name} ready", "green"), unsafe_allow_html=True)

st.write("")
section_title("2 · Provide Input")
input_mode = st.radio("Input method", ["Manual Feature Entry", "Upload CSV"], horizontal=True,
                       label_visibility="collapsed")

engineered_df = None
scaled = None

if input_mode == "Manual Feature Entry":
    st.caption("Enter the 9 base statistical features extracted from a vibration signal segment. "
               "The remaining 10 engineered features are computed automatically.")
    defaults = {"max": 0.8, "min": -0.7, "mean": 0.0, "sd": 0.35, "rms": 0.4,
                "skewness": 0.1, "kurtosis": 4.0, "crest": 3.2, "form": 1.2}
    cols = st.columns(3)
    values = {}
    for i, feat in enumerate(BASE_FEATURES):
        with cols[i % 3]:
            values[feat] = st.number_input(
                f"{feat}", value=float(defaults[feat]), format="%.4f",
                help=FEATURE_DESCRIPTIONS.get(feat, ""), key=f"manual_{feat}",
            )
    if st.button("🔮 Run Prediction", type="primary", key="manual_predict"):
        engineered_df, scaled = preprocess_manual_entry(values)

else:
    st.caption("Upload a CSV containing either the 9 base statistical columns "
               f"({', '.join(BASE_FEATURES)}) or the full 19 engineered feature columns. "
               "Every row will be classified.")
    up = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if up is not None:
        try:
            df_raw = pd.read_csv(up)
            st.dataframe(df_raw.head(10), use_container_width=True)
            if st.button("🔮 Run Batch Prediction", type="primary", key="csv_predict"):
                engineered_df, scaled = preprocess_csv_upload(df_raw)
        except Exception as e:
            st.error(f"Could not read/process the uploaded CSV: {e}")

# --------------------------------------------------------------------------
# INFERENCE + RESULTS
# --------------------------------------------------------------------------
if scaled is not None:
    section_title("3 · Prediction Results")
    t0 = time.perf_counter()
    preds, probs = wrapper.predict(scaled)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_sample_ms = elapsed_ms / max(1, scaled.shape[0])

    is_batch = scaled.shape[0] > 1

    if not is_batch:
        pred_idx = int(preds[0])
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[0, pred_idx])
        rec = get_recommendation(pred_class, confidence)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Predicted Bearing Fault", rec["display_name"])
        with c2:
            st.metric("Prediction Confidence", f"{confidence*100:.2f}%")
        with c3:
            st.metric("Fault Severity", rec["severity"])
        with c4:
            st.metric("Risk Level", rec["risk"])

        c5, c6, c7 = st.columns(3)
        with c5:
            st.metric("Prediction Time", f"{per_sample_ms:.2f} ms")
        with c6:
            st.metric("Model Used", model_name)
        with c7:
            st.metric("Probability (Top Class)", f"{confidence*100:.2f}%")

        gcol, pcol = st.columns([1, 1.4])
        with gcol:
            section_title("Confidence Gauge")
            gauge_color = COLORS["accent_green"] if confidence >= 0.8 else (
                COLORS["accent_amber"] if confidence >= 0.5 else COLORS["accent_red"])
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=confidence * 100,
                number={"suffix": "%"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": gauge_color},
                       "steps": [{"range": [0, 50], "color": "#2A1B1B"},
                                 {"range": [50, 80], "color": "#2A2416"},
                                 {"range": [80, 100], "color": "#16281C"}]},
            ))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=300, margin=dict(l=20, r=20, t=30, b=10),
                               paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        with pcol:
            section_title("Prediction Probability Chart")
            prob_df = pd.DataFrame({"Class": [CLASS_DISPLAY_NAMES[c] for c in CLASS_NAMES],
                                     "Probability": probs[0] * 100})
            prob_df = prob_df.sort_values("Probability", ascending=True)
            fig2 = go.Figure(go.Bar(x=prob_df["Probability"], y=prob_df["Class"], orientation="h",
                                     marker_color=COLORS["accent_blue"]))
            fig2.update_layout(template=PLOTLY_TEMPLATE, height=300, margin=dict(l=10, r=10, t=10, b=10),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                xaxis_title="Probability (%)")
            st.plotly_chart(fig2, use_container_width=True)

        st.session_state.prediction_history.insert(0, {
            "Model": model_name, "Prediction": rec["display_name"],
            "Confidence": f"{confidence*100:.2f}%", "Risk": rec["risk"],
            "Time (ms)": f"{per_sample_ms:.2f}",
        })

    else:
        pred_classes = [CLASS_NAMES[i] for i in preds]
        conf = probs.max(axis=1)
        result_df = engineered_df.copy()
        result_df["Predicted Class"] = [CLASS_DISPLAY_NAMES[c] for c in pred_classes]
        result_df["Confidence"] = (conf * 100).round(2)
        result_df["Risk"] = [get_recommendation(c, cf)["risk"] for c, cf in zip(pred_classes, conf)]

        st.success(f" Classified {len(result_df)} samples with **{model_name}** "
                   f"in {elapsed_ms:.1f} ms total ({per_sample_ms:.2f} ms/sample).")
        st.dataframe(result_df, use_container_width=True)

        dist = pd.Series(pred_classes).value_counts().reset_index()
        dist.columns = ["Class", "Count"]
        fig3 = px.bar(dist, x="Class", y="Count", color="Class", template=PLOTLY_TEMPLATE)
        fig3.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(" Download Predictions CSV", csv_bytes, "predictions.csv", "text/csv")

        for c, cf in zip(pred_classes, conf):
            st.session_state.prediction_history.insert(0, {
                "Model": model_name, "Prediction": CLASS_DISPLAY_NAMES[c],
                "Confidence": f"{cf*100:.2f}%",
                "Risk": get_recommendation(c, cf)["risk"], "Time (ms)": f"{per_sample_ms:.2f}",
            })

# --------------------------------------------------------------------------
# HISTORY
# --------------------------------------------------------------------------
st.write("")
section_title("Prediction History (this session)")
if st.session_state.prediction_history:
    hist_df = pd.DataFrame(st.session_state.prediction_history[:50])
    st.dataframe(hist_df, use_container_width=True, hide_index=True)
    if st.button("Clear History"):
        st.session_state.prediction_history = []
        st.rerun()
else:
    st.info("No predictions made yet in this session.")

footer()
