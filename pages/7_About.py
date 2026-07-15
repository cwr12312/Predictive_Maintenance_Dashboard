"""Page 8 (nav) — About: project metadata, objectives, methodology, architecture, credits."""

import streamlit as st

from config import (
    PROJECT_SHORT_TITLE, PROJECT_TITLE, PROJECT_DESCRIPTION, OBJECTIVES, METHODOLOGY,
    SYSTEM_ARCHITECTURE, DATASET_INFO, DEVELOPMENT_TOOLS,
    MODEL_REGISTRY,
)
from utils.styling import inject_css, page_header, section_title, footer

st.set_page_config(page_title=f"About · {PROJECT_SHORT_TITLE}", layout="wide")
inject_css()
page_header("About This Project", "Capstone project metadata and technical documentation")

st.markdown(f"### {PROJECT_TITLE}")
st.write(PROJECT_DESCRIPTION)

st.write("")
c1, c2 = st.columns(2)
with c1:
    section_title("Objectives")
    for o in OBJECTIVES:
        st.markdown(f"- {o}")

    section_title("Methodology")
    st.write(METHODOLOGY)

with c2:
    section_title("System Architecture")
    st.write(SYSTEM_ARCHITECTURE)

    section_title("Dataset Information")
    for k, v in DATASET_INFO.items():
        st.markdown(f"- **{k}:** {v}")

st.write("")
section_title("Deep Learning & Meta-Learning Models")
for name, cfg in MODEL_REGISTRY.items():
    with st.expander(f"{name}  ·  {cfg['framework']}"):
        st.write(cfg["description"])
        st.caption(f"Input shape: {cfg['input_shape']}  ·  Checkpoint: `{cfg['file'].name}`")

st.write("")
c3, c4 = st.columns(2)
with c3:
    section_title("Development Tools")
    st.markdown(", ".join(DEVELOPMENT_TOOLS))
footer()
