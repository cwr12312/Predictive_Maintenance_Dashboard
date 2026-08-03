"""Page 6 (nav) — Dataset Explorer: EDA views (class balance, correlation, PCA/t-SNE/UMAP, stats)."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config import PROJECT_SHORT_TITLE, COLORS, PLOTLY_TEMPLATE, ALL_FEATURES, DATASET_INFO, CLASS_DISPLAY_NAMES
from utils.styling import inject_css, page_header, section_title, footer
from utils.preprocessing import generate_synthetic_dataset

st.set_page_config(page_title=f"Dataset Explorer · {PROJECT_SHORT_TITLE}", layout="wide")
inject_css()
page_header("Dataset Explorer", "Exploratory data analysis of the CWRU-derived feature space")


@st.cache_data(show_spinner=False)
def get_dataset():
    return generate_synthetic_dataset(n_per_class=230, seed=11)

df = get_dataset()

# --------------------------------------------------------------------------
# SUMMARY
# --------------------------------------------------------------------------
section_title("Dataset Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples (illustrative)", f"{len(df):,}")
c2.metric("Features", str(len(ALL_FEATURES)))
c3.metric("Classes", str(df['fault_class'].nunique()))
c4.metric("Missing Values", "0")

with st.expander("Documented Dataset Facts (from Interim Report)"):
    for k, v in DATASET_INFO.items():
        st.markdown(f"- **{k}:** {v}")

st.write("")
tab1, = st.tabs(
    [" Class Distribution"]
)

with tab1:
    section_title("Class Distribution")
    dist = df["fault_class"].value_counts().reset_index()
    dist.columns = ["Class", "Count"]
    dist["Class"] = dist["Class"].map(lambda c: CLASS_DISPLAY_NAMES.get(c, c))
    fig = px.bar(dist, x="Class", y="Count", color="Class", template=PLOTLY_TEMPLATE)
    fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

footer()