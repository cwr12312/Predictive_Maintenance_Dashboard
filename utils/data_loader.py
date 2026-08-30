"""
utils/data_loader.py
=====================
Loads the official six-model evaluation metrics (extracted from the
project's MODEL_C.xlsx comparison workbook) and provides small helpers
for ranking / best-model lookups used across the Executive Dashboard and
Model Comparison pages.
"""

import pandas as pd
import streamlit as st

from config import METRICS_CSV


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    df = pd.read_csv(METRICS_CSV)
    df = df.sort_values("average_score", ascending=False).reset_index(drop=True)
    df["computed_rank"] = df["average_score"].rank(ascending=False, method="min").astype(int)
    return df


def best_model_row(df: pd.DataFrame = None) -> pd.Series:
    df = df if df is not None else load_metrics()
    return df.sort_values("average_score", ascending=False).iloc[0]


def get_row(model_name: str, df: pd.DataFrame = None) -> pd.Series:
    df = df if df is not None else load_metrics()
    match = df[df["model_name"] == model_name]
    if match.empty:
        raise KeyError(f"No metrics row for model '{model_name}'")
    return match.iloc[0]
