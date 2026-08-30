"""
utils/shap_utils.py
====================
Model-agnostic explainability layer. Uses SHAP's generic `Explainer` (which
auto-selects a Permutation/Kernel algorithm) wrapped around each model's
`predict_proba`, so all six heterogeneous architectures (Keras and PyTorch)
share the same explanation code path. If SHAP cannot explain a particular
model for any reason, the caller receives a clear status instead of a
crash, and the UI falls back to permutation feature importance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import ALL_FEATURES, CLASS_NAMES


def _predict_fn_factory(wrapper):
    def _fn(X):
        return wrapper.predict_proba(np.asarray(X, dtype=np.float32))
    return _fn


def compute_shap_values(wrapper, background: np.ndarray, instances: np.ndarray, max_evals: int = 300):
    """
    Returns a dict:
      {"status": "ok", "values": shap.Explanation} on success, or
      {"status": "fallback", "importances": np.ndarray} using permutation importance, or
      {"status": "error", "message": str}
    """
    try:
        import shap
        predict_fn = _predict_fn_factory(wrapper)
        masker = shap.maskers.Independent(background, max_samples=min(100, len(background)))
        explainer = shap.Explainer(predict_fn, masker, output_names=CLASS_NAMES)
        sv = explainer(instances, max_evals=max_evals, silent=True)
        return {"status": "ok", "values": sv}
    except Exception as exc:  # noqa: BLE001
        fallback = _permutation_importance(wrapper, background, instances)
        if fallback is not None:
            return {"status": "fallback", "importances": fallback, "message": str(exc)}
        return {"status": "error", "message": str(exc)}


def _permutation_importance(wrapper, background: np.ndarray, instances: np.ndarray):
    """A lightweight, dependency-free fallback: measures how much shuffling
    each feature (holding it at the background mean) changes the predicted
    probability of the winning class, averaged across the given instances."""
    try:
        base_probs = wrapper.predict_proba(instances)
        base_classes = base_probs.argmax(axis=1)
        n_feat = instances.shape[1]
        importances = np.zeros(n_feat)
        bg_mean = background.mean(axis=0)

        for f in range(n_feat):
            perturbed = instances.copy()
            perturbed[:, f] = bg_mean[f]
            new_probs = wrapper.predict_proba(perturbed)
            drop = base_probs[np.arange(len(instances)), base_classes] - \
                   new_probs[np.arange(len(instances)), base_classes]
            importances[f] = np.mean(np.abs(drop))
        return importances
    except Exception:
        return None


def global_permutation_importance(wrapper, background: np.ndarray, sample: np.ndarray) -> pd.DataFrame:
    imp = _permutation_importance(wrapper, background, sample)
    if imp is None:
        return pd.DataFrame(columns=["feature", "importance"])
    df = pd.DataFrame({"feature": ALL_FEATURES, "importance": imp})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)
