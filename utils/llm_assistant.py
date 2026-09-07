"""
utils/llm_assistant.py
=======================
Large Language Model reasoning layer for the Industrial Predictive
Maintenance Dashboard. Everything here is surfaced from within the
existing Live Prediction page — there is no separate LLM/AI page.

DESIGN PRINCIPLE
-----------------
The six trained ML models (Two-Dimensional Convolutional Neural Network (2D CNN), Long Short-Term Memory (LSTM),
Transformer, Model-Agnostic Meta-Learning (MAML), Meta Stochastic Gradient Descent (Meta-SGD),
Feature-Based Contrastive Learning (FBCL))
remain completely unchanged and are the ONLY source of fault predictions
and metrics in this dashboard. Everything in this module is an *additional,
optional* language-reasoning layer that sits ON TOP of those predictions —
it explains, summarises, and answers questions about real dashboard data.
It never retrains, replaces, or overrides any of the six models.

LIVE LLM CONNECTION (optional)
-------------------------------
No API key is hard-coded anywhere in this file. This dashboard uses Google
Gemini exclusively as its LLM provider. To enable real, live LLM calls, set
the following as an environment variable OR in `.streamlit/secrets.toml`
(see README.md for the exact steps):

    GEMINI_API_KEY = "YOUR_GEMINI_KEY"

If it isn't configured, every function below transparently and honestly
falls back to a deterministic, clearly-labelled template / rule-based
response built ONLY from real dashboard data (data/model_metrics.csv,
config.py class/maintenance definitions, live prediction context, and this
session's real logged predictions). No response is ever fabricated or
mislabeled as coming from a live model — callers always receive a `mode`
of "live" or "template" alongside the text so the UI can show the user
exactly which one they got.
"""

from __future__ import annotations

import os
import json
from typing import Optional

import streamlit as st

from config import CLASS_DISPLAY_NAMES, FAULT_FAMILY, LLM_DEFAULT_GEMINI_MODEL
from utils.maintenance import RECOMMENDATIONS
from utils.data_loader import load_metrics, best_model_row


# ==========================================================================
# PROVIDER / KEY RESOLUTION  (no keys ever hard-coded)
# ==========================================================================

def _get_secret(name: str) -> Optional[str]:
    """Reads a secret from st.secrets first, then falls back to env vars."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


def get_active_provider() -> Optional[str]:
    """Returns 'gemini' if a Gemini API key is configured, else None."""
    if _get_secret("GEMINI_API_KEY"):
        return "gemini"
    return None


def is_llm_configured() -> bool:
    return get_active_provider() is not None


def _call_gemini(system: str, user: str) -> str:
    from google import genai  # imported lazily so the package is only required if actually used

    client = genai.Client(
        api_key=_get_secret("GEMINI_API_KEY")
    )

    model = _get_secret("DASHBOARD_LLM_MODEL") or LLM_DEFAULT_GEMINI_MODEL

    prompt = f"""
System instructions:
{system}

User request:
{user}
"""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    text = getattr(response, "text", None)

    if not text:
        raise RuntimeError("Gemini returned an empty response.")

    return text.strip()


def call_llm(system: str, user: str) -> tuple[Optional[str], str]:
    """
    Attempts a real, live LLM call to Gemini if an API key is configured.

    Returns:
        (response_text, mode)

        "live"     = real API call succeeded
        "template" = no API key configured or API call failed

    API errors are printed to the terminal so they can be diagnosed.
    The API key itself is NEVER printed.
    """

    provider = get_active_provider()

    # --------------------------------------------------------------
    # Check whether a Gemini API key was found
    # --------------------------------------------------------------
    if provider is None:
        print("========================================")
        print("LLM ERROR: No Gemini API key detected.")
        print("========================================")
        print("Expected GEMINI_API_KEY in:")
        print("  .streamlit/secrets.toml")
        print("or as an environment variable.")
        print("========================================")
        return None, "template"

    # --------------------------------------------------------------
    # Provider detected
    # --------------------------------------------------------------
    print("========================================")
    print(f"LLM PROVIDER DETECTED: {provider}")
    print("========================================")

    try:
        response = _call_gemini(system, user)
        print("LLM SUCCESS: Gemini response received.")
        return response, "live"

    except Exception as e:

        # ----------------------------------------------------------
        # IMPORTANT:
        # Print the REAL error instead of silently hiding it.
        # NEVER print the API key.
        # ----------------------------------------------------------
        print("")
        print("========================================")
        print("           LLM API ERROR")
        print("========================================")
        print(f"Provider: {provider}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("========================================")
        print("The dashboard is falling back to template mode.")
        print("========================================")
        print("")

        return None, "template"


def family_for_display(predicted_display: str) -> str:
    """
    Maps a human-readable class label (e.g. "Inner Race Fault — 0.014in")
    back to its broader fault family (e.g. "Inner Race Fault") so the
    reasoning functions below can look up the right maintenance rules in
    utils/maintenance.RECOMMENDATIONS. Falls back to "Ball Fault" if no
    match is found rather than raising, since this is only used to pick
    which canned guidance/template to reach for.
    """
    for cname, cdisp in CLASS_DISPLAY_NAMES.items():
        if cdisp == predicted_display:
            return FAULT_FAMILY.get(cname, "Ball Fault")
    return "Ball Fault"


# ==========================================================================
# A. PLAIN LANGUAGE EXPLANATION
# ==========================================================================

def explain_prediction_plain_language(predicted_display: str, confidence: float,
                                       top_features: list[tuple[str, float]]) -> tuple[str, str]:
    """
    Translates a technical model prediction into a plain-language
    explanation for a plant technician, covering exactly four things:
    what was detected, how confident the model is, what could happen if
    the fault is ignored, and what to do next. Grounded in the real
    prediction/confidence passed in and the real maintenance rules in
    utils/maintenance.py — never fabricated.
    """
    system = (
        "You are a plain-language assistant embedded in an industrial bearing "
        "predictive-maintenance dashboard. Explain a model's prediction to a plant "
        "technician who is not a data scientist. Respond in Markdown with exactly these "
        "four bolded sections, in this order: **What was detected**, **Confidence**, "
        "**Potential impact**, **Recommended action**. Be concise and concrete. Never "
        "invent numbers that were not given to you."
    )
    feat_str = ", ".join(f"{f} ({v:+.3f})" for f, v in top_features) or "no dominant feature"
    user = (
        f"Predicted bearing condition: '{predicted_display}'. Model confidence: "
        f"{confidence*100:.1f}%. Features that most influenced the prediction: {feat_str}."
    )
    text, mode = call_llm(system, user)
    if text:
        return text, mode

    family = family_for_display(predicted_display)
    rec = RECOMMENDATIONS.get(family, RECOMMENDATIONS["Ball Fault"])
    actions_md = "\n".join(f"- {a}" for a in rec["actions"][:4])

    if family == "Normal":
        what = ("No fault signature was detected — the vibration pattern is consistent "
                "with a healthy, normally-operating bearing.")
        impact = "No corrective action is needed at this time; continue routine monitoring."
    else:
        what = (f"The system detected a **{predicted_display}** — a defect in the "
                 f"{family.lower()} area of the bearing, which shows up as an abnormal "
                 f"vibration pattern compared to a healthy bearing.")
        impact = (f"This is classified as **{rec['risk']}-risk**. If left unaddressed, "
                   f"{family.lower()} damage typically progresses to increased vibration and "
                   f"heat, and can eventually lead to unplanned bearing failure and downtime.")

    confidence_text = (
        f"The model is **{confidence*100:.1f}% confident** in this prediction."
        + (" This is a strong, reliable reading." if confidence >= 0.85 else
           " This is a moderate reading — consider a follow-up check to confirm." if confidence >= 0.6 else
           " This is a low-confidence reading — treat it as a preliminary flag and verify manually.")
    )

    template = (
        f"**What was detected**\n{what}\n\n"
        f"**Confidence**\n{confidence_text}\n\n"
        f"**Potential impact**\n{impact}\n\n"
        f"**Recommended action**\n{actions_md}"
    )
    return template, mode


# ==========================================================================
# B. NATURAL LANGUAGE QUERYING
# ==========================================================================

def answer_dashboard_query(question: str, session_history: list[dict] | None = None) -> tuple[str, str]:
    """
    Answers a free-text question about the dashboard's real data — model
    metrics/rankings, maintenance rules, and this session's real logged
    predictions (e.g. "which bearings are trending toward failure"). Never
    fabricates figures; if session_history is empty, says so honestly
    rather than inventing a trend.
    """
    metrics = load_metrics()
    best = best_model_row(metrics)
    metrics_context = metrics[["model_name", "accuracy", "precision", "recall", "f1_score",
                                "average_score", "inference_time_ms", "model_type"]].to_dict(orient="records")
    history_context = (session_history or [])[:50]

    system = (
        "You are a data assistant for an industrial predictive-maintenance dashboard. "
        "Answer ONLY using the JSON context provided below — model metrics AND this "
        "session's real logged predictions. Never invent figures. If the session history "
        "is empty or doesn't cover the question (e.g. a multi-week trend), say so plainly "
        "instead of guessing. Be concise (2-4 sentences), plant-technician-friendly."
    )
    user = (
        f"Model metrics (JSON, real values from this project): {json.dumps(metrics_context)}\n"
        f"This session's logged predictions, most recent first (JSON): {json.dumps(history_context)}\n\n"
        f"Question: {question}"
    )
    text, mode = call_llm(system, user)
    if text:
        return text, mode

    # ---- Rule-based grounded fallback (no LLM configured) ----
    q = question.lower()

    if ("trend" in q or "failure" in q or "week" in q or "shift" in q) and "model" not in q:
        if not history_context:
            return ("No predictions have been logged in this session yet, so there's no "
                    "trend to report. Run some predictions above and ask again — this tool "
                    "only reports on real, logged predictions, never invented ones."), "template"
        risky = [h for h in history_context if h.get("Risk") in ("High", "Critical")]
        if not risky:
            return (f"Across the {len(history_context)} prediction(s) logged this session, "
                    f"none were flagged High or Critical risk — nothing is currently trending "
                    f"toward failure."), "template"
        counts: dict[str, int] = {}
        for h in risky:
            counts[h.get("Prediction", "Unknown")] = counts.get(h.get("Prediction", "Unknown"), 0) + 1
        top = sorted(counts.items(), key=lambda x: -x[1])
        lines = "; ".join(f"{name} ({n}x)" for name, n in top)
        return (f"Of the {len(history_context)} prediction(s) logged this session, "
                f"{len(risky)} were High/Critical risk: {lines}. (Reflects this session's "
                f"real logged predictions.)"), "template"

    if "best" in q and "model" in q:
        return (f"The best-performing model is **{best['model_name']}** with "
                f"{best['accuracy']*100:.2f}% accuracy (average score {best['average_score']:.4f})."), "template"
    if "worst" in q or "lowest" in q:
        worst = metrics.sort_values("average_score").iloc[0]
        return (f"The lowest-ranked model is **{worst['model_name']}** "
                f"({worst['accuracy']*100:.2f}% accuracy)."), "template"
    if "fast" in q or "inference" in q or "latency" in q or "speed" in q:
        fastest = metrics.sort_values("inference_time_ms").iloc[0]
        return (f"**{fastest['model_name']}** has the fastest estimated inference time "
                f"at {fastest['inference_time_ms']:.1f} ms/sample."), "template"
    if "accuracy" in q:
        lines = "; ".join(f"{r['model_name']}: {r['accuracy']*100:.2f}%" for _, r in metrics.iterrows())
        return f"Accuracy by model — {lines}.", "template"
    if "how many" in q and ("class" in q or "fault" in q):
        return f"The system classifies {len(CLASS_DISPLAY_NAMES)} bearing health states.", "template"
    if "maintenance" in q or "recommend" in q or "action" in q:
        for family in RECOMMENDATIONS:
            if family.lower().split()[0] in q:
                rec = RECOMMENDATIONS[family]
                return (f"For a **{family}** (risk: {rec['risk']}), recommended actions are: "
                        + "; ".join(rec["actions"])), "template"
        return ("Maintenance guidance depends on the predicted fault family — visit the "
                "**Maintenance Recommendations** page, or ask e.g. 'What should I do for "
                "an inner race fault?'"), "template"
    if history_context and ("how many" in q or "count" in q or "predictions" in q):
        return f"{len(history_context)} prediction(s) have been logged so far this session.", "template"

    return (
        "I can answer questions about model rankings, accuracy, inference speed, "
        "maintenance recommendations, and this session's logged predictions (e.g. which "
        "bearings are trending toward failure). Try: 'Which model performed best?', "
        "'Which bearings are trending toward failure?', or 'What should I do for a ball fault?'"
    ), "template"


# ==========================================================================
# C. ROOT CAUSE REASONING (decision support, not a diagnosis)
# ==========================================================================

def root_cause_reasoning(predicted_display: str, confidence: float, features: dict,
                          technician_notes: str = "") -> tuple[str, str]:
    """
    Suggests 2-4 plausible root causes for the predicted fault by combining
    the real vibration-derived features (rms/kurtosis/crest) passed in from
    the Live Prediction page with an optional free-text technician note.
    Explicitly framed as decision support for a human to verify, never a
    guaranteed diagnosis — both the live-LLM prompt and the template
    fallback below end with that same disclaimer.
    """
    system = (
        "You are a decision-support assistant for industrial vibration analysis. You "
        "combine structured sensor features (FFT/RMS/kurtosis-derived) with technician "
        "notes to suggest POSSIBLE root causes — you never state a guaranteed diagnosis, "
        "and you always recommend manual verification."
    )
    user = (
        f"Predicted condition: {predicted_display} (confidence {confidence*100:.1f}%).\n"
        f"Vibration-derived features: {json.dumps(features, default=str)}\n"
        f"Technician notes / maintenance log: {technician_notes or 'None provided'}\n\n"
        "List 2-4 plausible root causes as decision support, referencing specific feature "
        "values where relevant, and end with a reminder that this is not a guaranteed diagnosis."
    )
    text, mode = call_llm(system, user)
    if text:
        return text, mode

    # ---- Rule-based fallback: turn each individual feature reading into a
    # plain-English possible cause only when it crosses a meaningful
    # threshold, rather than always listing all three regardless of value. ----
    bullets = []
    kurt = features.get("kurtosis")
    rms = features.get("rms")
    crest = features.get("crest")
    if isinstance(kurt, (int, float)) and kurt > 6:
        bullets.append(f"Elevated kurtosis ({kurt:.2f}) suggests impulsive, impact-like "
                        f"vibration consistent with localized pitting or spalling.")
    if isinstance(rms, (int, float)) and rms > 0.5:
        bullets.append(f"High RMS energy ({rms:.3f}) indicates increased overall vibration "
                        f"amplitude, consistent with defect growth or mechanical looseness.")
    if isinstance(crest, (int, float)) and crest > 4:
        bullets.append(f"High crest factor ({crest:.2f}) points to sharp, short-duration "
                        f"impacts relative to overall signal energy — typical of early-stage defects.")
    if technician_notes.strip():
        bullets.append("Technician notes are on file for this asset — consider whether this "
                        "is a recurring/unresolved fault rather than a newly-developing one.")
    if not bullets:
        bullets.append("The feature pattern is broadly consistent with the predicted fault "
                        "family; no single feature stands out as dominant.")
    bullets.append("⚠ This is decision support only, not a diagnosis — confirm with manual "
                    "vibration or thermal inspection before scheduling repairs.")
    return "\n".join(f"- {b}" for b in bullets), mode


# ==========================================================================
# D. MINE UNSTRUCTURED MAINTENANCE HISTORY
# ==========================================================================

def summarize_maintenance_notes(notes_text: str) -> tuple[str, str]:
    """
    Extracts structured information (recurring faults, equipment/date
    mentions, repeat-failure patterns) from free-text maintenance notes.
    This is explicitly presented as extraction/summarisation to support a
    human review — not as automatically-generated training labels.
    """
    system = (
        "Extract structured information from free-text industrial maintenance notes. "
        "Identify: (1) recurring faults or equipment mentioned, (2) any pattern of repeat "
        "failures, (3) a short list of distinct failure events found in the text. Present "
        "this as a structured summary a reliability engineer could use to spot patterns — "
        "not as ground-truth labels. Be concise."
    )
    text, mode = call_llm(system, notes_text)
    if text:
        return text, mode

    keywords = ["bearing", "vibration", "inner race", "outer race", "ball fault",
                "overheating", "leak", "replaced", "lubrication", "alignment", "noise", "seal"]
    lines = [ln.strip() for ln in notes_text.splitlines() if ln.strip()]

    if not lines:
        return "No maintenance notes were provided to extract information from.", mode

    # Tally which keywords appear, and how often, to surface recurring themes.
    theme_counts: dict[str, int] = {}
    for ln in lines:
        low = ln.lower()
        for k in keywords:
            if k in low:
                theme_counts[k] = theme_counts.get(k, 0) + 1
    found = sorted(theme_counts, key=lambda k: -theme_counts[k])
    recurring = [k for k in found if theme_counts[k] > 1]

    parts = [f"**Structured Extraction — {len(lines)} log entr{'y' if len(lines) == 1 else 'ies'} found**"]

    parts.append("\n**Distinct events identified:**")
    for ln in lines[:12]:
        parts.append(f"- {ln}")

    if found:
        parts.append("\n**Fault types / equipment mentioned:** " + ", ".join(found))
    else:
        parts.append("\n**Fault types / equipment mentioned:** none of the tracked keywords "
                      "were found in this note set.")

    if recurring:
        parts.append(f"\n**Recurring theme(s):** {', '.join(recurring)} — appearing in more "
                      f"than one entry, which may indicate a repeat or unresolved issue worth "
                      f"flagging for review.")
    else:
        parts.append("\n**Recurring theme(s):** none — nothing in this note set repeats across "
                      "multiple entries.")

    parts.append("\n_Candidate structure only — recommended for human review before use as "
                  "training labels._")
    return "\n".join(parts), mode


# ==========================================================================
# E. AUTOMATED MAINTENANCE / HEALTH REPORTING
# ==========================================================================

def generate_maintenance_report(asset_name: str, predicted_display: str, confidence: float,
                                 model_name: str, risk: str, actions: list[str],
                                 best_model_name: str, best_accuracy: float,
                                 session_summary_lines: list[str] | None = None) -> tuple[str, str]:
    """
    Builds the Markdown "Predictions Report" shown/downloaded from the
    Live Prediction page's report modal. Every fact passed in (asset name,
    predicted class, confidence, risk, actions, best-model comparison, and
    the optional real session summary) comes from actual dashboard state —
    the LLM/template is only asked to phrase it, never to supply figures.
    """
    system = (
        "Draft a concise, professional maintenance/system health report in Markdown for a "
        "plant manager, using ONLY the facts given below. Do not invent data or figures. "
        "Title the report 'Predictions Report'."
    )
    session_block = ""
    if session_summary_lines:
        session_block = "\nThis session's real logged predictions: " + "; ".join(session_summary_lines)
    user = (
        f"Asset: {asset_name}\nDetected condition: {predicted_display}\nModel used: {model_name}\n"
        f"Confidence: {confidence*100:.1f}%\nRisk level: {risk}\n"
        f"Recommended actions: {'; '.join(actions)}\n"
        f"Best-performing system model overall: {best_model_name} ({best_accuracy*100:.2f}% test accuracy)"
        f"{session_block}\n\n"
        "Write the report with these section headers: Asset Health Summary, Detected/Predicted "
        "Faults, Model Confidence, Session Summary (only if session data was given), Key "
        "Observations, Recommended Maintenance Actions."
    )
    text, mode = call_llm(system, user)
    if text:
        return text, mode

    observation = ("No abnormal condition detected." if risk == "Low" else
                   "Abnormal vibration signature detected consistent with the predicted fault family.")
    session_md = ""
    if session_summary_lines:
        session_md = "\n### Session Summary\n" + "\n".join(f"- {s}" for s in session_summary_lines) + "\n"

    template = f"""## Predictions Report

**Asset:** {asset_name}

### Asset Health Summary
Condition: **{predicted_display}** &nbsp;|&nbsp; Risk level: **{risk}**

### Detected / Predicted Faults
- Predicted class: {predicted_display}
- Model used: {model_name}
- Prediction confidence: {confidence*100:.1f}%

### Model Confidence
{confidence*100:.1f}% from {model_name}. For reference, the best overall system model is
**{best_model_name}** at **{best_accuracy*100:.2f}%** test accuracy.
{session_md}
### Key Observations
- Risk classified as **{risk}**.
- {observation}

### Recommended Maintenance Actions
{chr(10).join(f"- {a}" for a in actions)}
"""
    return template, mode


# ==========================================================================
# F. SCENARIO GENERATION / DOCUMENTATION
# ==========================================================================

def generate_fault_scenario(class_display: str, family: str, risk: str) -> tuple[str, str]:
    """
    Produces a short, realistic narrative of how the given fault might
    develop and get noticed on a factory floor — used for technician
    training materials, documentation, or test-data generation. Purely
    illustrative prose, not tied to any specific real asset or reading.
    """
    system = (
        "Write a short (4-6 sentence), realistic industrial scenario describing how this "
        "bearing fault might develop and be noticed on a factory floor, for technician "
        "training/documentation/testing purposes. Do not include real company or people names."
    )
    user = f"Fault: {class_display} (family: {family}, risk: {risk})"
    text, mode = call_llm(system, user)
    if text:
        return text, mode

    template = (
        f"**Scenario — {class_display}:** A technician on routine rounds notices a subtle "
        f"change in operating sound near a drive-end bearing. Vibration monitoring flags an "
        f"increase consistent with a {family.lower()} pattern. Over the following shifts, RMS "
        f"and kurtosis trend upward, and the system raises a {risk.lower()}-risk alert. "
        f"Following the recommended inspection, maintenance confirms early-stage {family.lower()} "
        f"damage and schedules corrective action before it can cause unplanned downtime."
    )
    return template, mode


# ==========================================================================
# G. MULTIMODAL FUSION
# ==========================================================================
# The six trained models only ever see vibration-derived features — there is
# no thermal or acoustic *model* in this project, and this function never
# invents one. What IS implemented and real: a working reasoning layer that
# fuses the real vibration prediction with whatever additional modality
# readings a technician enters right now (a thermal reading, an acoustic
# note) into one combined, decision-support assessment. If a modality field
# is left blank it is simply excluded — nothing about it is fabricated.

# ==========================================================================
# H. FREE-FORM "MANAGE AGENT" CHAT (floating panel — utils/styling.py)
# ==========================================================================
# This is intentionally a thin wrapper around call_llm() above — it is NOT
# a second agent/backend. The floating "Manage Agent" chat panel and every
# other LLM feature in this dashboard share this one provider-resolution
# and template-fallback layer.

def chat_with_agent(history: list[dict]) -> tuple[str, str]:
    """
    Free-form conversational entry point for the floating "Manage Agent"
    chat panel. `history` is a list of {"role": "user"|"assistant",
    "content": str} dicts (session_state.agent_chat_history). Returns
    (reply_text, mode) exactly like every other function in this file.
    """
    system = (
        "You are the 'Manage Agent' assistant embedded in an Industrial "
        "Predictive Maintenance Dashboard for rolling-element bearing fault "
        "diagnosis. The dashboard runs six real trained models (2D CNN, LSTM, "
        "Transformer, MAML, Meta-SGD, FBCL) on the CWRU bearing dataset. "
        "Answer the user's questions helpfully and concisely — you may "
        "discuss the dashboard, its models, predictions, metrics, and "
        "maintenance recommendations, or general bearing-fault-diagnosis "
        "topics. Never claim to retrain, replace, or change any of the six "
        "models — you are a reasoning layer on top of them, not a "
        "replacement for them."
    )
    convo = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-12:]
    )
    text, mode = call_llm(system, convo)
    if text:
        return text, mode

    last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    fallback = (
        "_(Template mode — no live Gemini key configured. Set GEMINI_API_KEY "
        "to enable live Gemini replies.)_\n\n"
        f"I received your message: \u201c{last_user}\u201d. I can help explain "
        "predictions, model metrics, and maintenance recommendations shown "
        "elsewhere in this dashboard — try asking about a specific model or "
        "fault type."
    )
    return fallback, mode


def multimodal_fusion_reasoning(predicted_display: str, confidence: float, family: str, risk: str,
                                 thermal_temp_c: float | None = None,
                                 thermal_baseline_c: float | None = None,
                                 acoustic_note: str = "",
                                 maintenance_note: str = "") -> tuple[str, str]:
    """
    Combines the real vibration-model prediction with whichever optional
    modalities the technician actually filled in (thermal reading,
    acoustic note, maintenance text) into a single fused assessment. Any
    modality left blank/None is simply omitted from `modalities` below —
    nothing about a missing modality is guessed or invented.
    """
    # Vibration is always present since it's the real model output; every
    # other entry below is added conditionally based on what was supplied.
    modalities = [f"Vibration (ML model): {predicted_display}, {confidence*100:.1f}% confidence, "
                  f"classified {risk}-risk."]
    if thermal_temp_c is not None:
        line = f"Thermal: {thermal_temp_c:.1f}°C reading"
        if thermal_baseline_c is not None:
            delta = thermal_temp_c - thermal_baseline_c
            line += f" vs. {thermal_baseline_c:.1f}°C baseline ({delta:+.1f}°C)"
        modalities.append(line + ".")
    if acoustic_note.strip():
        modalities.append(f"Acoustic: technician-reported — \"{acoustic_note.strip()}\"")
    if maintenance_note.strip():
        modalities.append(f"Maintenance text: \"{maintenance_note.strip()}\"")

    if len(modalities) == 1:
        # Nothing besides vibration was provided — there's no fusion to do,
        # so say so plainly instead of pretending to combine modalities.
        return ("Only the vibration modality has data — add a thermal reading, acoustic note, "
                "or maintenance note above to fuse them into a combined assessment."), "template"

    system = (
        "You are a multimodal industrial-monitoring reasoning layer. You are given real "
        "readings from up to four modalities (vibration-model prediction, thermal, acoustic, "
        "maintenance text). Combine them into ONE fused assessment: state whether the "
        "modalities agree or conflict, and give a combined risk read. This is decision "
        "support, never a guaranteed diagnosis. Never invent a modality reading you were not given."
    )
    user = "Modality readings:\n" + "\n".join(f"- {m}" for m in modalities)
    text, mode = call_llm(system, user)
    if text:
        return text, mode

    # Template fallback — simple, honest rule-based fusion of only the
    # modalities that were actually provided.
    lines = [f"**Fused assessment** (combining {len(modalities)} modalit{'y' if len(modalities)==1 else 'ies'}):"]
    lines += [f"- {m}" for m in modalities]
    flags = []
    if thermal_temp_c is not None and thermal_baseline_c is not None and (thermal_temp_c - thermal_baseline_c) > 10:
        flags.append("thermal reading is notably above baseline")
    if acoustic_note.strip():
        flags.append("an acoustic anomaly was reported by a technician")
    if risk in ("High", "Critical"):
        flags.append("the vibration model already flags elevated risk")
    if flags:
        lines.append(f"\n**Agreement check:** {'; '.join(flags)} — these modalities corroborate "
                      f"each other, which increases confidence this is a real, developing issue "
                      f"rather than a sensor artifact.")
    else:
        lines.append("\n**Agreement check:** no additional modality raised a flag beyond the "
                      "vibration model — no corroborating evidence of an issue from the other "
                      "inputs provided.")
    lines.append("\n⚠ Decision support only — verify with a physical inspection before acting.")
    return "\n".join(lines), mode