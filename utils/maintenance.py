"""
utils/maintenance.py
=====================
Maps a predicted fault class (+ confidence) to maintenance recommendations,
severity and risk level, following the structure requested for Page 5.
"""

from config import FAULT_FAMILY, SEVERITY_MAP, CLASS_DISPLAY_NAMES

RECOMMENDATIONS = {
    "Normal": {
        "risk": "Low",
        "actions": ["Machine Healthy", "Continue Routine Monitoring", "No maintenance action required"],
    },
    "Ball Fault": {
        "risk": "Medium",
        "actions": ["Inspect Bearing", "Monitor Temperature", "Schedule Maintenance",
                    "Track vibration trend over the next 2-4 weeks"],
    },
    "Inner Race Fault": {
        "risk": "High",
        "actions": ["Immediate Inspection", "Bearing Replacement Recommended",
                    "Reduce load / speed until inspected", "Log for root-cause analysis"],
    },
    "Outer Race Fault": {
        "risk": "Critical",
        "actions": ["High Risk Alert", "Shutdown Recommendation", "Lubrication Check",
                    "Alignment Inspection", "Notify maintenance supervisor immediately"],
    },
}


def get_recommendation(class_name: str, confidence: float = None) -> dict:
    """
    class_name: one of config.CLASS_NAMES (e.g. 'OR_014_6')
    Returns dict with family, severity, risk, actions, display_name.
    """
    family = FAULT_FAMILY.get(class_name, "Ball Fault")
    rec = RECOMMENDATIONS.get(family, RECOMMENDATIONS["Ball Fault"]).copy()

    sev_code = None
    for code in ("007", "014", "021"):
        if code in class_name:
            sev_code = code
            break
    severity = SEVERITY_MAP.get(sev_code, "N/A") if family != "Normal" else "N/A"

    risk = rec["risk"]
    # Escalate risk if confidence is low (model unsure) -> recommend manual verification
    low_confidence = confidence is not None and confidence < 0.60
    actions = list(rec["actions"])
    if low_confidence and family != "Normal":
        actions.append("⚠ Model confidence is low — confirm with manual vibration inspection")

    return {
        "family": family,
        "display_name": CLASS_DISPLAY_NAMES.get(class_name, class_name),
        "severity": severity,
        "risk": risk,
        "actions": actions,
    }
