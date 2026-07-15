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

---

## 2. Project Structure

```
Predictive_Maintenance_Dashboard/
├── app.py                      # Page 1 - Executive Dashboard (entry point)
├── config.py                   # All constants: paths, classes, features, colours, model registry
├── requirements.txt
├── README.md
├── run_dashboard.bat
├── .gitignore
├── models/                     # Trained model checkpoints 
│   ├── cnn_2d_model.keras
│   ├── final_transformer_model.keras
│   ├── lstm_model.keras
│   ├── maml_model.pth
│   ├── meta_sgd.pth
│   └── fbcl_model.pth
├── data/
│   └── model_metrics.csv       # Official 6-model metrics.
├── utils/
│   ├── styling.py               # CSS injection + reusable KPI/badge/alert components
│   ├── preprocessing.py         # 9->19 feature engineering + scaling pipeline
│   ├── torch_architectures.py   # Exact nn.Module defs for MAML / Meta-SGD / FBCL
│   ├── model_loader.py          # Cached loaders + unified ModelWrapper.predict_proba()
│   ├── benchmark.py             # Live per-model inference-latency measurement
│   ├── data_loader.py           # Metrics CSV loading / ranking helpers
│   ├── maintenance.py           # Fault -> maintenance recommendation rules engine
│   └── shap_utils.py            # SHAP explainability with permutation-importance fallback
├── pages/                       # Streamlit native multipage router
│   ├── 1_Model_Comparison.py
│   ├── 2_Live_Prediction.py
│   ├── 4_Maintenance_Recommendations.py
│   ├── 5_Dataset_Explorer.py
│   ├── 6_Model_Performance.py
│   └── 7_About.py
├── css/style.css                # Industrial dark theme
├── assets/, charts/, reports/, outputs/, logs/   # Working directories for generated artefacts
```

---

## 3. Pages

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

## 4. Troubleshooting

- **A model shows "failed to load"**: check the sidebar "Load errors" expander on the
  Executive Dashboard for the exact exception (usually a TensorFlow/PyTorch version
  mismatch — try the versions pinned in `requirements.txt`).