# Industrial Predictive Maintenance Dashboard

**Industrial Predictive Maintenance System for Bearing Fault Diagnosis Using Advanced Deep Learning Models**

A production-styled Streamlit dashboard for diagnosing rolling-element bearing faults using
six trained deep-learning / meta-learning / continual-learning models — **Two-Dimensional
Convolutional Neural Network (2D CNN), Long Short-Term Memory (LSTM), Transformer,
Model-Agnostic Meta-Learning (MAML), Meta Stochastic Gradient Descent (Meta-SGD) and
Feature-Based Contrastive Learning (FBCL)** — evaluated on the CWRU Bearing Data Center dataset.
An LLM reasoning layer sits on top of these six models, entirely within the Live Prediction
page, to explain predictions in plain language, answer natural-language questions, reason
about root causes, mine maintenance notes, generate reports, generate scenarios, and fuse
multiple modalities — see section 4 below.

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

1. **App** — the landing page; renders the Executive Dashboard (KPI cards, best-model
   highlight, accuracy target, model-type breakdown).
2. **About** — objectives, methodology, architecture, dataset facts, model descriptions,
   and a summary of the LLM reasoning layer.
3. **Model Comparison** — ranked leaderboard with medals and comparison charts.
4. **Live Prediction** — manual entry or CSV upload, run any of the 6 models, confidence
   gauge, probability chart, a **Plain Language Explanation** (what was detected /
   confidence / potential impact / recommended action, generated from the real
   prediction), plus the full LLM reasoning layer (see section 4), a **Generate Full
   Report** button that opens a report in a pop-up dialog with a download button, and
   session prediction history.
5. **Maintenance Recommendations** — rule-based, colour-coded action cards per fault family.
6. **Dataset Explorer** — class balance and dataset statistics, computed live from the
   project's real augmented training dataset.
7. **Model Performance** — confusion matrix from live inference on the real held-out test set.

No separate LLM/AI page exists — the entire reasoning layer lives inside the Live
Prediction page.

## 3. Troubleshooting

- **A model shows "failed to load"**: check the sidebar "Load errors" expander (visible on
  every page) for the exact exception — usually a TensorFlow/PyTorch version mismatch; try
  the versions pinned in `requirements.txt`.
- **Sidebar navigation**: `.streamlit/config.toml` sets `showSidebarNavigation = false` so
  Streamlit's automatic (filename-derived) page list doesn't duplicate the app's own
  custom navigation in `utils/styling.render_sidebar()`.
- **A secondary button (Report / Root Cause / etc.) used to make results disappear**: this
  was a real bug — `scaled`/`engineered_df` were plain local variables that reset to `None`
  on every rerun that wasn't the exact click of "Run Prediction", including the rerun
  triggered by clicking any button inside the results section. Fixed by persisting them in
  `st.session_state` (`live_scaled` / `live_engineered_df`).

## 4. LLM Reasoning Layer (Live Prediction page)

The six ML models above are the only source of fault predictions and never require an LLM.
On top of them, the Live Prediction page implements, live in the running app today:

- **Plain Language Explanation** — every prediction, in four parts: what was detected, how
  confident the model is, the potential impact, and the recommended action.
- **Natural Language Querying** — ask things like *"which bearings are trending toward
  failure"* or *"which model performed best"*; answered from this session's real logged
  predictions and the real model metrics — never fabricated.
- **Root Cause Reasoning** — combines real structured sensor features (RMS, kurtosis, crest
  factor) with an optional technician note to suggest possible causes, as decision support.
- **Maintenance History Mining** — paste free-text maintenance notes/logs and extract
  recurring faults and failure events, as a starting point for human review.
- **Automated Reporting** — a report generated from the real current prediction, including a
  real session summary (counts and risk breakdown of predictions actually logged so far),
  shown in a pop-up dialog with a download button.
- **Scenario Generation** — a natural-language description of the current predicted fault,
  for technician training, testing, or documentation.
- **Multimodal Fusion** — combines the real vibration-model prediction with a thermal
  reading, an acoustic note, and/or maintenance text you enter, into one fused assessment.
  The six trained models still only ever see vibration data (there is no thermal/acoustic
  *model* in this project), but the fusion *reasoning* layer itself is fully implemented.

Each of these runs live if an LLM API key is configured, otherwise it falls back to a
clearly-labelled, deterministic template grounded in this project's real data — the UI
always shows a **Live AI** or **Template / Rule-Based** badge so you know which you got.

No key is hard-coded anywhere in the project. To enable live mode, set ONE of the following:

**Option A — Streamlit secrets** (create `.streamlit/secrets.toml`):
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
# or, to use OpenAI instead:
# OPENAI_API_KEY = "sk-..."
```

**Option B — environment variable:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
streamlit run app.py
```

Optionally override the model name with `DASHBOARD_LLM_MODEL` (Anthropic) or
`DASHBOARD_LLM_MODEL_OPENAI` (OpenAI). See `utils/llm_assistant.py` for the exact call
sites — every LLM-dependent function there returns a `mode` of `"live"` or `"template"`
so the UI always tells you honestly which one produced the response.