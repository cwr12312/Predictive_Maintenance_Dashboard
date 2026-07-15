"""
config.py
=========
Central configuration for the Industrial Predictive Maintenance Dashboard.

Every path, constant, class label, feature name and colour used across the
app is defined here so that the rest of the codebase never hard-codes
"magic values". This mirrors the exact preprocessing and dataset
specification described in the project's Proposal and Interim Report.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# PROJECT ROOT / DIRECTORY LAYOUT
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
CSS_DIR = ROOT_DIR / "css"
ASSETS_DIR = ROOT_DIR / "assets"
CHARTS_DIR = ROOT_DIR / "charts"
REPORTS_DIR = ROOT_DIR / "reports"
OUTPUTS_DIR = ROOT_DIR / "outputs"
LOGS_DIR = ROOT_DIR / "logs"

for _d in (CHARTS_DIR, REPORTS_DIR, OUTPUTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

METRICS_CSV = DATA_DIR / "model_metrics.csv"

# --------------------------------------------------------------------------
# PROJECT METADATA  (Page 8 - About)
# --------------------------------------------------------------------------
PROJECT_TITLE = "Industrial Predictive Maintenance System for Bearing Fault Diagnosis Using Advanced Deep Learning Models"
PROJECT_SHORT_TITLE = "Bearing Fault Predictive Maintenance Dashboard"
PROJECT_DESCRIPTION = (
    "A dual-path, interpretable machine learning system for diagnosing rolling-element "
    "bearing faults from vibration-derived statistical features. This dashboard packages "
    "six advanced deep-learning and meta-learning models — 2D CNN, LSTM, Transformer, "
    "MAML, Meta-SGD and FBCL — trained on the Case Western Reserve University (CWRU) "
    "Bearing Data Center benchmark, into a single production-style monitoring, prediction "
    "and explainability console."
)
OBJECTIVES = [
    "Classify ten industrial bearing health states (Normal, Ball, Inner-Race and "
    "Outer-Race faults at three severities) with \u226595% accuracy.",
    "Deliver sub-150ms inference suitable for near-real-time monitoring on commodity hardware.",
    "Compare feature-driven deep learning against meta-learning / continual-learning approaches.",
    "Provide physically interpretable, SHAP-based explanations tied to vibration-analysis theory.",
    "Package the full pipeline into an industrial-grade monitoring dashboard.",
]
METHODOLOGY = (
    "Raw CWRU vibration recordings were reduced to 9 time-domain statistical features "
    "(max, min, mean, sd, rms, skewness, kurtosis, crest, form). Ten additional engineered "
    "features (ratios and polynomial interactions of the base statistics) were derived, "
    "producing a 19-dimensional feature vector. The training split was augmented via SMOTE "
    "and bootstrapped Gaussian-noise sampling to ~100,000 balanced samples before training "
    "the six deep-learning / meta-learning architectures used in this dashboard."
)
SYSTEM_ARCHITECTURE = (
    "Streamlit front-end -> cached model registry (TensorFlow/Keras + PyTorch) -> shared "
    "preprocessing pipeline (StandardScaler + feature engineering) -> per-model inference "
    "adapters -> SHAP / permutation explainability layer -> Plotly visualisation layer."
)
DATASET_INFO = {
    "Source": "Case Western Reserve University (CWRU) Bearing Data Center",
    "Original samples": "2,300 (9 features + 1 target, perfectly balanced, 230 per class)",
    "Augmented training samples": "~100,000 (balanced via SMOTE + Gaussian bootstrapping)",
    "Validation / Test samples": "327 / 329 (held out, untouched by augmentation)",
    "Engineered feature count": "19 (9 base statistical + 10 derived)",
    "Number of classes": "10",
    "Operating condition": "1 HP load, 1772 RPM, drive-end accelerometer",
}
DEVELOPMENT_TOOLS = [
    "Python 3.10+", "TensorFlow / Keras", "PyTorch", "Streamlit", "Plotly & Plotly Express",
    "Pandas / NumPy", "scikit-learn", "SHAP", "Custom CSS (Industrial Dark Theme)",
]

# --------------------------------------------------------------------------
# CLASS LABELS  (Table 5 & 6, Interim Report — fault_encoded mapping)
# --------------------------------------------------------------------------
CLASS_NAMES = [
    "Ball_007", "Ball_014", "Ball_021",
    "IR_007", "IR_014", "IR_021",
    "Normal",
    "OR_007_6", "OR_014_6", "OR_021_6",
]
N_CLASSES = len(CLASS_NAMES)

CLASS_DISPLAY_NAMES = {
    "Ball_007": "Ball Fault (0.007in)",
    "Ball_014": "Ball Fault (0.014in)",
    "Ball_021": "Ball Fault (0.021in)",
    "IR_007": "Inner Race Fault (0.007in)",
    "IR_014": "Inner Race Fault (0.014in)",
    "IR_021": "Inner Race Fault (0.021in)",
    "Normal": "Normal / Healthy",
    "OR_007_6": "Outer Race Fault (0.007in, 6 o'clock)",
    "OR_014_6": "Outer Race Fault (0.014in, 6 o'clock)",
    "OR_021_6": "Outer Race Fault (0.021in, 6 o'clock)",
}

FAULT_FAMILY = {  # maps a specific class to its broad family, used for maintenance recs
    "Ball_007": "Ball Fault", "Ball_014": "Ball Fault", "Ball_021": "Ball Fault",
    "IR_007": "Inner Race Fault", "IR_014": "Inner Race Fault", "IR_021": "Inner Race Fault",
    "Normal": "Normal",
    "OR_007_6": "Outer Race Fault", "OR_014_6": "Outer Race Fault", "OR_021_6": "Outer Race Fault",
}

SEVERITY_MAP = {  # inches of induced defect -> qualitative severity
    "007": "Mild", "014": "Moderate", "021": "Severe",
}

# --------------------------------------------------------------------------
# FEATURES  (Table 4, Interim Report + base statistics)
# --------------------------------------------------------------------------
BASE_FEATURES = ["max", "min", "mean", "sd", "rms", "skewness", "kurtosis", "crest", "form"]

ENGINEERED_FEATURES = [
    "peak_to_peak", "rms_mean_ratio", "crest_form_ratio", "kurtosis_squared",
    "skewness_abs", "rms_sd_ratio", "max_min_ratio", "kurtosis_crest",
    "mean_squared", "rms_squared",
]

ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES  # 19 total, fixed column order
N_FEATURES = len(ALL_FEATURES)

FEATURE_DESCRIPTIONS = {
    "max": "Maximum amplitude of the vibration signal segment",
    "min": "Minimum amplitude of the vibration signal segment",
    "mean": "Mean amplitude (DC offset) of the signal",
    "sd": "Standard deviation — overall signal dispersion",
    "rms": "Root Mean Square — total vibration energy; rises with defect severity",
    "skewness": "Distribution asymmetry of the signal amplitude",
    "kurtosis": "Impulsiveness indicator; >3 suggests early-stage pitting/impacts",
    "crest": "Crest factor (peak / RMS) — impulsiveness relative to energy",
    "form": "Form factor (RMS / mean absolute value) — waveform shape indicator",
    "peak_to_peak": "max - min, total signal swing",
    "rms_mean_ratio": "rms / mean — energy relative to central tendency",
    "crest_form_ratio": "crest / form — combined shape/impulsiveness ratio",
    "kurtosis_squared": "kurtosis^2 — emphasises strongly impulsive signals",
    "skewness_abs": "|skewness| — magnitude of asymmetry, direction-agnostic",
    "rms_sd_ratio": "rms / sd — energy relative to dispersion",
    "max_min_ratio": "max / min — amplitude range ratio",
    "kurtosis_crest": "kurtosis * crest — compound impulsiveness feature",
    "mean_squared": "mean^2 — nonlinear central-tendency term",
    "rms_squared": "rms^2 — proportional to vibration power",
}

TARGET_ACCURACY = 0.95
TARGET_INFERENCE_MS = 150
TARGET_FALSE_POSITIVE_RATE = 0.05

# --------------------------------------------------------------------------
# MODEL REGISTRY
# --------------------------------------------------------------------------
MODEL_REGISTRY = {
    "2D CNN": {
        "id": "cnn2d",
        "framework": "keras",
        "file": MODELS_DIR / "cnn_2d_model.keras",
        "input_shape": (5, 4, 1),
        "description": "2D Convolutional Network. Features zero-padded from 19 to 20 and "
                        "reshaped into a 5x4 pseudo-image so spatial convolution filters can "
                        "learn local statistical-feature interactions.",
        "supports_shap": True,
        "shap_mode": "gradient",
    },
    "LSTM": {
        "id": "lstm",
        "framework": "keras",
        "file": MODELS_DIR / "lstm_model.keras",
        "input_shape": (1, 19),
        "description": "Bidirectional-style stacked LSTM treating the 19-feature vector as a "
                        "single timestep sequence; two LSTM layers extract sequential feature "
                        "dependencies before dense classification heads.",
        "supports_shap": True,
        "shap_mode": "gradient",
    },
    "Transformer": {
        "id": "transformer",
        "framework": "keras",
        "file": MODELS_DIR / "final_transformer_model.keras",
        "input_shape": (19, 1),
        "description": "Self-attention Transformer classifier: each of the 19 features is "
                        "treated as a token, allowing multi-head attention to learn "
                        "inter-feature relationships directly.",
        "supports_shap": True,
        "shap_mode": "gradient",
    },
    "MAML": {
        "id": "maml",
        "framework": "torch",
        "file": MODELS_DIR / "maml_model.pth",
        "input_shape": (19,),
        "description": "Model-Agnostic Meta-Learning transformer encoder, self-supervised "
                        "pretrained via masked-feature reconstruction, then fine-tuned for "
                        "fast adaptation across fault classes.",
        "supports_shap": True,
        "shap_mode": "kernel",
    },
    "Meta-SGD": {
        "id": "meta_sgd",
        "framework": "torch",
        "file": MODELS_DIR / "meta_sgd.pth",
        "input_shape": (19,),
        "description": "Meta-SGD transformer encoder that learns both initial weights and "
                        "per-parameter learning rates for rapid adaptation to new fault "
                        "conditions.",
        "supports_shap": True,
        "shap_mode": "kernel",
    },
    "FBCL": {
        "id": "fbcl",
        "framework": "torch",
        "file": MODELS_DIR / "fbcl_model.pth",
        "input_shape": (19,),
        "description": "Feature Boosting Continual Learning transformer: a shared backbone "
                        "with additive boosting heads trained sequentially across defect "
                        "severities via knowledge distillation, mitigating catastrophic "
                        "forgetting.",
        "supports_shap": True,
        "shap_mode": "kernel",
    },
}
MODEL_NAMES = list(MODEL_REGISTRY.keys())

# --------------------------------------------------------------------------
# UI THEME  — Industrial Dark / Blue
# --------------------------------------------------------------------------
COLORS = {
    "bg_primary": "#0B1220",
    "bg_secondary": "#111A2E",
    "bg_card": "#152238",
    "bg_card_hover": "#1B2C4A",
    "border": "#22314D",
    "accent_blue": "#2E8FFF",
    "accent_blue_light": "#5CB2FF",
    "accent_cyan": "#22D3EE",
    "accent_green": "#22C55E",
    "accent_amber": "#F59E0B",
    "accent_red": "#EF4444",
    "text_primary": "#E8EEFC",
    "text_secondary": "#93A2C2",
    "text_muted": "#5E6C8C",
    "gold": "#FFD700",
    "silver": "#C0C0C0",
    "bronze": "#CD7F32",
}

RISK_COLORS = {
    "Low": COLORS["accent_green"],
    "Medium": COLORS["accent_amber"],
    "High": COLORS["accent_red"],
    "Critical": "#B91C1C",
}

PLOTLY_TEMPLATE = "plotly_dark"
