"""
utils/preprocessing.py
=======================
Reproduces the feature-engineering pipeline described in the Interim Report
and used by every training script (fbcl_model.py, maml.py, transfomer.py,
lstm_model.py, 2d_cnn.py):

    9 base statistical features  ->  + 10 engineered ratio/interaction terms
    ->  19-dimensional feature vector  ->  StandardScaler  ->  model input

Because the raw scaler objects (`scaler.pkl` fitted during training) were
not part of the uploaded bundle, this module fits a *reference* scaler at
app start-up from representative synthetic statistics consistent with the
dataset documentation (Table 2 / Table 4, Interim Report: CWRU, 10 balanced
classes, mean-zero / unit-variance engineered space). If the user supplies
their own `feature_scaler.pkl` / `label_encoder.pkl` in `data/`, those are
used automatically instead — see `load_reference_scaler()`.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import ALL_FEATURES, BASE_FEATURES, ENGINEERED_FEATURES, DATA_DIR


def engineer_features(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the 9 base statistical features into the full 19-feature set
    exactly as enumerated in Table 4 of the Interim Report.

    Parameters
    ----------
    df_base : DataFrame with (at least) columns BASE_FEATURES

    Returns
    -------
    DataFrame with columns == ALL_FEATURES, in the fixed training order.
    """
    df = df_base.copy()
    missing = [c for c in BASE_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required base feature columns: {missing}")

    eps = 1e-8  # guards against divide-by-zero on edge-case manual entries

    df["peak_to_peak"] = df["max"] - df["min"]
    df["rms_mean_ratio"] = df["rms"] / (df["mean"].replace(0, eps))
    df["crest_form_ratio"] = df["crest"] / (df["form"].replace(0, eps))
    df["kurtosis_squared"] = df["kurtosis"] ** 2
    df["skewness_abs"] = df["skewness"].abs()
    df["rms_sd_ratio"] = df["rms"] / (df["sd"].replace(0, eps))
    df["max_min_ratio"] = df["max"] / (df["min"].replace(0, eps))
    df["kurtosis_crest"] = df["kurtosis"] * df["crest"]
    df["mean_squared"] = df["mean"] ** 2
    df["rms_squared"] = df["rms"] ** 2

    return df[ALL_FEATURES]


def _synthetic_reference_scaler() -> StandardScaler:
    """
    Build a StandardScaler fitted on documented, class-representative
    statistics so the pipeline is runnable end-to-end even without the
    original serialized scaler. Ranges follow the physical interpretation
    given in the Interim Report (kurtosis ~3 healthy / 5-30 faulted, RMS
    rising with severity, etc.) across the 10 balanced fault classes.
    """
    rng = np.random.default_rng(42)
    n_per_class = 250
    rows = []
    severities = [1.0, 1.6, 2.4]  # relative energy scaling for 007/014/021

    class_specs = [
        ("Ball_007", 0), ("Ball_014", 1), ("Ball_021", 2),
        ("IR_007", 0), ("IR_014", 1), ("IR_021", 2),
        ("Normal", None),
        ("OR_007_6", 0), ("OR_014_6", 1), ("OR_021_6", 2),
    ]
    for name, sev_idx in class_specs:
        sev = severities[sev_idx] if sev_idx is not None else 0.4
        base_rms = 0.6 * sev if name != "Normal" else 0.35
        base_kurt = 3.0 if name == "Normal" else float(rng.uniform(5, 22)) * (sev / 1.6)
        for _ in range(n_per_class):
            rms = max(0.05, rng.normal(base_rms, 0.08))
            mean = rng.normal(0.0, 0.02)
            sd = max(0.02, rng.normal(base_rms * 0.9, 0.05))
            kurt = max(1.5, rng.normal(base_kurt, base_kurt * 0.15 + 0.3))
            skew = rng.normal(0.0 if name == "Normal" else 0.3, 0.4)
            crest = max(1.0, rng.normal(3.0 if name == "Normal" else 4.5 + sev, 0.6))
            form = max(1.0, rng.normal(1.2, 0.1))
            mx = rms * crest + abs(rng.normal(0, 0.05))
            mn = -mx * rng.uniform(0.7, 1.0)
            rows.append([mx, mn, mean, sd, rms, skew, kurt, crest, form])

    df = pd.DataFrame(rows, columns=BASE_FEATURES)
    df_full = engineer_features(df)
    scaler = StandardScaler()
    scaler.fit(df_full.values)
    return scaler


_scaler_cache: StandardScaler | None = None


def load_reference_scaler() -> StandardScaler:
    """
    Loads a persisted StandardScaler from data/feature_scaler.pkl if present
    (drop your original training-time scaler there for exact reproduction),
    otherwise falls back to a synthetic, documentation-consistent reference
    scaler built on first use and cached for the session.
    """
    global _scaler_cache
    if _scaler_cache is not None:
        return _scaler_cache

    custom_path = DATA_DIR / "feature_scaler.pkl"
    if custom_path.exists():
        with open(custom_path, "rb") as f:
            _scaler_cache = pickle.load(f)
            return _scaler_cache

    _scaler_cache = _synthetic_reference_scaler()
    return _scaler_cache


def scale_features(df_full: pd.DataFrame) -> np.ndarray:
    """Apply the reference StandardScaler to a full 19-feature DataFrame."""
    scaler = load_reference_scaler()
    return scaler.transform(df_full[ALL_FEATURES].values.astype(np.float32))


def preprocess_manual_entry(base_values: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Convert a dict of the 9 base feature values (manual UI entry) into the
    engineered + scaled representation ready for any of the six models.

    Returns (engineered_unscaled_df, scaled_array)
    """
    df_base = pd.DataFrame([base_values])[BASE_FEATURES]
    df_full = engineer_features(df_base)
    scaled = scale_features(df_full)
    return df_full, scaled


def preprocess_csv_upload(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Accepts an uploaded CSV. Two accepted layouts:
      1) Already has all 19 ALL_FEATURES columns -> used directly.
      2) Has only the 9 BASE_FEATURES columns -> engineered automatically.
    Any extra columns (e.g. a target/label column) are ignored for inference.
    """
    cols = set(df_raw.columns)
    if set(ALL_FEATURES).issubset(cols):
        df_full = df_raw[ALL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    elif set(BASE_FEATURES).issubset(cols):
        df_base = df_raw[BASE_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
        df_full = engineer_features(df_base)
    else:
        missing = set(BASE_FEATURES) - cols
        raise ValueError(
            "Uploaded CSV must contain either the 19 engineered feature columns or the 9 "
            f"base statistical columns. Missing base columns: {sorted(missing)}"
        )
    scaled = scale_features(df_full)
    return df_full, scaled


def generate_synthetic_dataset(n_per_class: int = 150, seed: int = 7) -> pd.DataFrame:
    """
    Generates an illustrative, documentation-consistent dataset (used only
    by the Dataset Explorer page for EDA visuals) since the raw CWRU CSVs
    were not part of the uploaded bundle. Distributional shape follows the
    same generative logic as `_synthetic_reference_scaler`, but returns
    unscaled features + string labels for exploration.
    """
    rng = np.random.default_rng(seed)
    severities = [1.0, 1.6, 2.4]
    class_specs = [
        ("Ball_007", 0), ("Ball_014", 1), ("Ball_021", 2),
        ("IR_007", 0), ("IR_014", 1), ("IR_021", 2),
        ("Normal", None),
        ("OR_007_6", 0), ("OR_014_6", 1), ("OR_021_6", 2),
    ]
    rows = []
    for name, sev_idx in class_specs:
        sev = severities[sev_idx] if sev_idx is not None else 0.4
        base_rms = 0.6 * sev if name != "Normal" else 0.35
        base_kurt = 3.0 if name == "Normal" else float(rng.uniform(5, 22)) * (sev / 1.6)
        for _ in range(n_per_class):
            rms = max(0.05, rng.normal(base_rms, 0.08))
            mean = rng.normal(0.0, 0.02)
            sd = max(0.02, rng.normal(base_rms * 0.9, 0.05))
            kurt = max(1.5, rng.normal(base_kurt, base_kurt * 0.15 + 0.3))
            skew = rng.normal(0.0 if name == "Normal" else 0.3, 0.4)
            crest = max(1.0, rng.normal(3.0 if name == "Normal" else 4.5 + sev, 0.6))
            form = max(1.0, rng.normal(1.2, 0.1))
            mx = rms * crest + abs(rng.normal(0, 0.05))
            mn = -mx * rng.uniform(0.7, 1.0)
            rows.append([mx, mn, mean, sd, rms, skew, kurt, crest, form, name])

    df = pd.DataFrame(rows, columns=BASE_FEATURES + ["fault_class"])
    engineered = engineer_features(df[BASE_FEATURES])
    df_out = pd.concat([engineered, df["fault_class"]], axis=1)
    return df_out
