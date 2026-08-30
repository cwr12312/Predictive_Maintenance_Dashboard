"""
utils/benchmark.py
===================
Measures *actual* per-sample inference latency for each loaded model on
this machine, used by the Executive Dashboard and Model Performance pages.
Falls back to the estimated `inference_time_ms` column in
data/model_metrics.csv if a model failed to load or benchmarking fails.
"""

import numpy as np
import pandas as pd
import streamlit as st

from config import N_FEATURES
from utils.model_loader import load_all_models
from utils.preprocessing import load_reference_scaler


@st.cache_data(show_spinner=False)
def benchmark_all_models(n_warmup: int = 2, n_runs: int = 10) -> pd.DataFrame:
    registry = load_all_models()
    models = registry["models"]

    rng = np.random.default_rng(0)
    dummy = rng.normal(size=(8, N_FEATURES)).astype(np.float32)

    rows = []
    for name, wrapper in models.items():
        try:
            for _ in range(n_warmup):
                wrapper.predict(dummy)
            times = []
            for _ in range(n_runs):
                _, _, per_sample_ms = wrapper.timed_predict(dummy)
                times.append(per_sample_ms)
            rows.append({"model_name": name, "measured_inference_ms": float(np.median(times))})
        except Exception:
            rows.append({"model_name": name, "measured_inference_ms": np.nan})
    return pd.DataFrame(rows)
