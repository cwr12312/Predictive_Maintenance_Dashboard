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

st.warning(
    "The confusion matrix, ROC and PR curves below are computed by running **real, live "
    "inference** with the loaded model — but on an illustrative synthetic evaluation set "
    "(the original held-out CWRU test CSV was not part of the uploaded bundle). Treat these "
    "charts as a behavioural diagnostic, not a replacement for the official metrics above."
)

if wrapper is not None:
    @st.cache_data(show_spinner=False)
    def eval_set():
        df = generate_synthetic_dataset(n_per_class=40, seed=99)
        X = scale_features(df[ALL_FEATURES])
        y = df["fault_class"].map(lambda c: CLASS_NAMES.index(c)).values
        return X, y

    X_eval, y_eval = eval_set()
    y_pred, y_probs = wrapper.predict(X_eval)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [" Confusion Matrix", " Classification Report", " ROC Curves",
         " Precision-Recall Curves", " Training Curves"]
    )

    with tab1:
        section_title("Confusion Matrix (live inference, illustrative eval set)")
        cm = confusion_matrix(y_eval, y_pred, labels=range(len(CLASS_NAMES)))
        fig = px.imshow(cm, x=CLASS_NAMES, y=CLASS_NAMES, text_auto=True,
                         color_continuous_scale="Blues", aspect="auto")
        fig.update_layout(template=PLOTLY_TEMPLATE, height=560, paper_bgcolor="rgba(0,0,0,0)",
                           xaxis_title="Predicted", yaxis_title="True")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        section_title("Classification Report")
        report = classification_report(y_eval, y_pred, target_names=CLASS_NAMES,
                                        output_dict=True, zero_division=0)
        rep_df = pd.DataFrame(report).T.round(3)
        st.dataframe(rep_df, use_container_width=True)

    with tab3:
        section_title("ROC Curves (One-vs-Rest)")
        y_bin = label_binarize(y_eval, classes=range(len(CLASS_NAMES)))
        fig = go.Figure()
        for i, cname in enumerate(CLASS_NAMES):
            if y_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{cname} (AUC={roc_auc:.2f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"),
                                  name="Chance"))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=520, paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", xaxis_title="False Positive Rate",
                           yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        section_title("Precision-Recall Curves (One-vs-Rest)")
        y_bin = label_binarize(y_eval, classes=range(len(CLASS_NAMES)))
        fig = go.Figure()
        for i, cname in enumerate(CLASS_NAMES):
            if y_bin[:, i].sum() == 0:
                continue
            prec, rec, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
            fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=cname))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=520, paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        section_title("Training / Validation Curves")
        extra = wrapper.extra or {}
        ckpt = extra.get("checkpoint")
        if ckpt and "train_accs" in ckpt and "val_accs" in ckpt:
            epochs = list(range(1, len(ckpt["train_accs"]) + 1))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=ckpt["train_accs"], name="Train Accuracy"))
            fig.add_trace(go.Scatter(x=epochs, y=ckpt["val_accs"], name="Validation Accuracy"))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=420, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Epoch", yaxis_title="Accuracy",
                               title="Fine-tuning Accuracy Curve (from saved checkpoint)")
            st.plotly_chart(fig, use_container_width=True)
            if "pretraining_losses" in ckpt:
                fig2 = go.Figure(go.Scatter(y=ckpt["pretraining_losses"], mode="lines",
                                             line=dict(color=COLORS["accent_cyan"])))
                fig2.update_layout(template=PLOTLY_TEMPLATE, height=340, paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Epoch", yaxis_title="Loss",
                                    title="Self-Supervised Pretraining Loss (masked-feature reconstruction)")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info(
                f"Epoch-by-epoch training/validation curves were not serialized inside the "
                f"**{model_name}** checkpoint (only final test metrics were saved by the "
                "original training script). Re-run training with history logging enabled and "
                "re-export the checkpoint to populate this chart, or refer to the notebook's "
                "printed console output."
            )
else:
    st.info("Live evaluation charts require the model to load successfully.")

footer()
