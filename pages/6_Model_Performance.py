"""Page 7 (nav) — Model Performance: confusion matrix, classification report, ROC/PR curves, training curves."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

from config import PROJECT_SHORT_TITLE, COLORS, PLOTLY_TEMPLATE, MODEL_NAMES, CLASS_NAMES, ALL_FEATURES
from utils.styling import inject_css, page_header, section_title, footer
from utils.model_loader import get_model
from utils.preprocessing import generate_synthetic_dataset, scale_features
from utils.data_loader import load_metrics, get_row

st.set_page_config(page_title=f"Model Performance · {PROJECT_SHORT_TITLE}", layout="wide")
inject_css()
page_header("Model Performance", "Deep dive into any single model's evaluation behaviour")

model_name = st.selectbox("Select model", MODEL_NAMES, index=1)
wrapper = get_model(model_name)
official = get_row(model_name)

if wrapper is None:
    st.error(f"{model_name} failed to load — showing official metrics only.")
else:
    st.success(f"{model_name} loaded — live evaluation below runs real inference.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Official Accuracy", f"{official['accuracy']*100:.2f}%")
c2.metric("Official Precision", f"{official['precision']*100:.2f}%")
c3.metric("Official Recall", f"{official['recall']*100:.2f}%")
c4.metric("Official F1 Score", f"{official['f1_score']*100:.2f}%")
st.caption("Official metrics sourced directly from the project's MODEL_C.xlsx comparison workbook (test-set results).")

if wrapper is not None:
    @st.cache_data(show_spinner=False)
    def eval_set():
        df = generate_synthetic_dataset(n_per_class=40, seed=99)
        X = scale_features(df[ALL_FEATURES])
        y = df["fault_class"].map(lambda c: CLASS_NAMES.index(c)).values
        return X, y

    X_eval, y_eval = eval_set()
    y_pred, y_probs = wrapper.predict(X_eval)

    tab1, = st.tabs(
        [" Confusion Matrix"]
    )

    with tab1:
        section_title("Confusion Matrix (live inference, illustrative eval set)")
        cm = confusion_matrix(y_eval, y_pred, labels=range(len(CLASS_NAMES)))
        fig = px.imshow(cm, x=CLASS_NAMES, y=CLASS_NAMES, text_auto=True,
                         color_continuous_scale="Blues", aspect="auto")
        fig.update_layout(template=PLOTLY_TEMPLATE, height=560, paper_bgcolor="rgba(0,0,0,0)",
                           xaxis_title="Predicted", yaxis_title="True")
        st.plotly_chart(fig, use_container_width=True)

footer()