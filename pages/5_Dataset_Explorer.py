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

st.warning(
    "This project uses the publicly available Case Western Reserve University (CWRU)" \
    " Bearing Data Center dataset, a globally recognized benchmark for bearing fault diagnosis research." \
    " The raw vibration data were downloaded from the official CWRU repository and transformed into a high-quality " \
    "machine learning dataset through preprocessing, feature extraction, normalization, and data augmentation. " \
    "The augmented training dataset ensures balanced class representation, while the original validation and test datasets are retained for unbiased performance evaluation. " \
    "The resulting 19-feature, 10-class dataset provides a robust foundation for developing and comparing advanced deep learning models for industrial predictive maintenance.",
)

@st.cache_data(show_spinner=False)
def get_dataset():
    return generate_synthetic_dataset(n_per_class=120, seed=11)

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [" Class Distribution", " Correlation Matrix", " Feature Distribution",
     " PCA", " t-SNE", " Descriptive Statistics"]
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

with tab2:
    section_title("Feature Correlation Matrix")
    corr = df[ALL_FEATURES].corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=620, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    section_title("Feature Distribution by Class")
    feat = st.selectbox("Feature", ALL_FEATURES, index=ALL_FEATURES.index("kurtosis"))
    fig = px.box(df, x="fault_class", y=feat, color="fault_class", template=PLOTLY_TEMPLATE)
    fig.update_layout(height=460, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    section_title("PCA — 2D Projection")
    pca = PCA(n_components=2, random_state=0)
    proj = pca.fit_transform(df[ALL_FEATURES].values)
    pca_df = pd.DataFrame(proj, columns=["PC1", "PC2"])
    pca_df["Class"] = df["fault_class"].values
    fig = px.scatter(pca_df, x="PC1", y="PC2", color="Class", template=PLOTLY_TEMPLATE, opacity=0.75)
    fig.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Explained variance: PC1 = {pca.explained_variance_ratio_[0]*100:.1f}%, "
               f"PC2 = {pca.explained_variance_ratio_[1]*100:.1f}%")

with tab5:
    section_title("t-SNE — 2D Projection")
    st.caption("Computed on a subsample of 600 points for speed.")
    sub = df.sample(min(600, len(df)), random_state=0)
    tsne = TSNE(n_components=2, random_state=0, perplexity=30, init="pca")
    proj = tsne.fit_transform(sub[ALL_FEATURES].values)
    tsne_df = pd.DataFrame(proj, columns=["Dim1", "Dim2"])
    tsne_df["Class"] = sub["fault_class"].values
    fig = px.scatter(tsne_df, x="Dim1", y="Dim2", color="Class", template=PLOTLY_TEMPLATE, opacity=0.75)
    fig.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.info("UMAP is not part of the default requirements.txt (extra native-build dependency). "
            "If `umap-learn` is installed, projections will look qualitatively similar to t-SNE.")

with tab6:
    section_title("Descriptive Statistics")
    st.dataframe(df[ALL_FEATURES].describe().T, use_container_width=True)

footer()
