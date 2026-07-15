"""Page 5 (nav) — Maintenance Recommendations: rule-based guidance per predicted fault."""

import streamlit as st

from config import PROJECT_SHORT_TITLE, CLASS_NAMES, CLASS_DISPLAY_NAMES, RISK_COLORS
from utils.styling import inject_css, page_header, section_title, alert_card, footer
from utils.maintenance import get_recommendation, RECOMMENDATIONS

st.set_page_config(page_title=f"Maintenance Recommendations · {PROJECT_SHORT_TITLE}", layout="wide")
inject_css()
page_header("Maintenance Recommendations", "Actionable guidance mapped to each predicted fault condition")

section_title("Select a Fault Class")
choice = st.selectbox(
    "Predicted class", CLASS_NAMES,
    format_func=lambda c: CLASS_DISPLAY_NAMES.get(c, c),
    index=CLASS_NAMES.index("Normal"),
)
confidence = st.slider("Simulated prediction confidence", 0.0, 1.0, 0.92, 0.01)

rec = get_recommendation(choice, confidence)
st.write("")
alert_card(
    f"{rec['display_name']}  ·  Severity: {rec['severity']}  ·  Risk: {rec['risk']}",
    rec["actions"],
    risk=rec["risk"],
)

st.write("")
st.write("")
section_title("Full Recommendation Reference")

family_examples = {
    "Normal": ["Machine Healthy", "Continue Routine Monitoring"],
    "Ball Fault": ["Inspect Bearing", "Monitor Temperature", "Schedule Maintenance"],
    "Inner Race Fault": ["Immediate Inspection", "Bearing Replacement Recommended"],
    "Outer Race Fault": ["High Risk Alert", "Shutdown Recommendation", "Lubrication Check", "Alignment Inspection"],
}

cols = st.columns(4)
for col, (family, rules) in zip(cols, RECOMMENDATIONS.items()):
    with col:
        alert_card(f"{family} — Risk: {rules['risk']}", rules["actions"], risk=rules["risk"])

footer()
