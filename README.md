# Industrial Predictive Maintenance Dashboard

**Industrial Predictive Maintenance System for Bearing Fault Diagnosis Using Advanced Deep Learning Models**

A production-styled Streamlit dashboard for diagnosing rolling-element bearing faults using
six trained deep-learning / meta-learning / continual-learning models — **2D CNN, LSTM,
Transformer, MAML, Meta-SGD and FBCL**.

---

## 1. Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Windows users can instead double-click **`run_dashboard.bat`**, which creates a virtual
environment, installs dependencies and launches the app automatically.

The app opens at `http://localhost:8501`.

## 2. Pages

1. **Executive Dashboard** — KPI cards, best-model highlight, accuracy chart, model snapshot table.
2. **Model Comparison** — ranked leaderboard with medals, bar/radar/heatmap/scatter/bubble charts.
3. **Live Prediction** — manual entry or CSV upload, run any of the 6 models, confidence gauge,
   probability chart, session prediction history.
4. **Explainable AI** — SHAP global/summary/local (waterfall) explanations with automatic
   permutation-importance fallback per model.
5. **Maintenance Recommendations** — rule-based, colour-coded action cards per fault family.
6. **Dataset Explorer** — class balance, correlation matrix, per-feature boxplots, PCA, t-SNE,
   descriptive statistics.
7. **Model Performance** — confusion matrix, classification report, ROC & PR curves (live
   inference), training curves where available.
8. **About** — objectives, methodology, architecture, dataset facts, model descriptions, credits.

## 3. Troubleshooting

- **A model shows "failed to load"**: check the sidebar "Load errors" expander on the
  Executive Dashboard for the exact exception (usually a TensorFlow/PyTorch version
  mismatch — try the versions pinned in `requirements.txt`).