"""FABRICATED notes, for proving the reconciliation control actually fires.

Nothing here is from the dataset. Each entry takes a real patient's note and
plants one specific contradiction against what the structured tables say, so
the check in reconcile.py can be seen catching something. Never load these into
the database and never present them as data.
"""
from __future__ import annotations

# patient -> (what was planted, the altered note text)
PLANTED_CONFLICTS: dict[str, tuple[str, str]] = {
    "Stein, Larry": (
        "note records a beta-blocker the medication table does not contain, so the "
        "brief's headline HFrEF gap would be wrong",
        "Progress Note — AWV. Problem list active: Chronic Systolic Heart Failure "
        "(HFrEF) | Pure Hypercholesterolaemia | GERD without Oesophagitis. "
        "Medications: Atorvastatin 40 mg | Fenofibrate 145 mg | Esomeprazole 40 mg | "
        "Carvedilol 12.5 mg started at cardiology follow-up last month, tolerating "
        "well. Vitals stable — BP: 154/98, BMI: 36.0. Addressed preventive "
        "screening gaps. Lab orders placed based on guideline intervals."),
    "Black, Tyler": (
        "note records metformin stopped for intolerance, which the medication "
        "table cannot express because it has no discontinuation field",
        "Annual Wellness Visit. Active conditions: Type 2 Diabetes Mellitus. "
        "Medications reconciled: Empagliflozin 10 mg | Semaglutide 1 mg (Ozempic). "
        "Metformin discontinued three months ago for persistent GI intolerance; "
        "patient does not wish to retry. Vitals — BP: 135/74, BMI: 27.7. "
        "Preventive labs ordered per condition-specific guidelines. Patient "
        "educated on medication adherence."),
    "Bender, Jessica": (
        "note records a blood pressure that disagrees with the flowsheet value, "
        "which is itself physiologically impossible",
        "Annual preventive visit. Chronic conditions: Pure Hypercholesterolaemia. "
        "Medication list: Rosuvastatin 20 mg | Fenofibrate 145 mg | Evolocumab "
        "140 mg (Repatha). Today's vitals: BP 118/76, BMI 28.4. Patient reports "
        "adherence. Lifestyle counselling provided. Monitoring labs ordered."),
}


def planted_note(patient: str) -> list[dict] | None:
    """Return the fabricated note for a patient, or None."""
    entry = PLANTED_CONFLICTS.get(patient)
    if not entry:
        return None
    _, text = entry
    return [{"date": "FIXTURE — not real data", "author": "FIXTURE", "text": text}]
