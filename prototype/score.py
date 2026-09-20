"""A transparent per-patient score, so the ranking can be argued with.

The panel review ranks patients and nothing checks that the order is right --
the largest uncovered hazard in the system (KNOWN_ISSUES.md). A score does not
make the order correct, but it makes it *legible*: every number here decomposes
into named contributions a clinician can read down and disagree with, line by
line.

The division of labour is the same rule as everywhere else. A model writes the
weights ONCE, offline, into `score_weights.json` -- clinical knowledge is what
it is for. Code applies them on every run. The model never scores a patient,
so the score is identical every time and a weight change is a reviewable diff
rather than a prompt nobody audits.

Three components, because sickness is not priority. A very sick, well-managed
patient needs nothing this week:

    burden       rolled-up conditions. How much illness is on file.
    instability  abnormal labs and BP stage. Whether it is controlled now.
    neglect      overdue orders and missing guideline therapy. What is undone.

Neglect carries the most weight and most of the dynamic range. It is the only
component that describes something the panel manager can act on.

Two constraints this dataset forces, both documented in the README:

  * Conditions are rolled up to the 3-character ICD-10 category before
    scoring. I10 and I11.9 are one hypertension, not two, and E78.00/E78.1/
    E78.5 are one dyslipidaemia, not three. Scoring raw codes would inflate
    comorbidity exactly as grouping on them halves cohorts.
  * Burden alone cannot rank this panel: the median patient has 2 distinct
    conditions and the maximum is 4. Without instability and neglect the
    score sorts 100 people into about four buckets.

Not a validated instrument. Charlson and Elixhauser are published, validated
ICD-10 comorbidity indices and a real deployment should anchor the burden
component on one of them rather than on weights a model wrote. This exists to
make the ranking inspectable and editable, which is the precondition for
validating it, not a substitute for doing so.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .tools import AS_OF, PHYSIOLOGIC_LIMITS, connect

WEIGHTS = Path(__file__).parent / "score_weights.json"


@dataclass
class Contribution:
    component: str          # burden | instability | neglect
    label: str              # what a reader sees
    points: float
    because: str            # the fact it came from


@dataclass
class PatientScore:
    patient: str
    total: float = 0.0
    contributions: list[Contribution] = field(default_factory=list)

    def by_component(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for c in self.contributions:
            out[c.component] = round(out.get(c.component, 0.0) + c.points, 2)
        return out

    def explain(self) -> str:
        """The whole score as text. If this cannot be read aloud it is no use."""
        lines = [f"{self.patient} — {self.total:g} points"]
        for comp in ("burden", "instability", "neglect"):
            rows = [c for c in self.contributions if c.component == comp]
            if rows:
                lines.append(f"  {comp} ({sum(r.points for r in rows):g})")
                lines.extend(f"    +{r.points:g}  {r.label} — {r.because}" for r in rows)
        return "\n".join(lines)


def load_weights() -> dict:
    return json.loads(WEIGHTS.read_text())


def _rollup(icd10: str) -> str:
    """I11.9 and I10 are both I1-hypertension; E78.00, E78.1 and E78.5 are one lipid.

    Three characters is the ICD-10 category, which tools.py already treats as
    the unit of a condition when it warns about unused siblings.
    """
    return (icd10 or "").split(".")[0][:3].upper()


def score_panel(weights: dict | None = None) -> list[PatientScore]:
    """Score every patient. Deterministic: same database, same numbers."""
    w = weights or load_weights()
    con = connect()
    try:
        names = dict(con.execute("SELECT PAT_ID, PAT_NAME FROM patient").fetchall())
        dx = con.execute("SELECT DISTINCT PAT_ID, icd10, dx_name FROM v_diagnosis").fetchall()
        labs = con.execute("""
            SELECT PAT_ID, COMPONENT_NAME, value, REFERENCE_LOW, REFERENCE_HIGH
            FROM v_lab_result WHERE value IS NOT NULL
        """).fetchall()
        vitals = con.execute("""
            SELECT PAT_ID, max(systolic) FROM v_vitals
            WHERE systolic IS NOT NULL GROUP BY PAT_ID
        """).fetchall()
    finally:
        con.close()

    scores = {p: PatientScore(patient=n) for p, n in names.items()}

    # --- burden: rolled-up conditions, each counted once ----------------------
    seen: dict[str, set[str]] = {}
    for pid, icd10, dx_name in dx:
        cat = _rollup(icd10)
        if cat in seen.setdefault(pid, set()):
            continue                       # I10 after I11.9 adds nothing
        seen[pid].add(cat)
        pts = w["burden"]["categories"].get(cat, w["burden"]["default"])
        if pts and pid in scores:
            scores[pid].contributions.append(Contribution(
                "burden", dx_name, pts, f"{icd10} (category {cat})"))

    # --- instability: values outside range, and impossible ones separately ----
    for pid, analyte, value, lo, hi in labs:
        if pid not in scores:
            continue
        hard = PHYSIOLOGIC_LIMITS.get(analyte)
        if hard and not (hard[0] <= value <= hard[1]):
            # Not a sick patient -- a bad record. It lowers confidence in the
            # rest of the row, so it scores, but as a data problem.
            scores[pid].contributions.append(Contribution(
                "instability", f"{analyte} not physiologic", w["instability"]["impossible"],
                f"{value:g}, outside {hard[0]}-{hard[1]}"))
        elif lo is not None and hi is not None and not (lo <= value <= hi):
            pts = w["instability"]["analytes"].get(analyte, w["instability"]["default"])
            if pts:
                scores[pid].contributions.append(Contribution(
                    "instability", f"{analyte} out of range", pts,
                    f"{value:g} against {lo:g}-{hi:g}"))

    for pid, systolic in vitals:
        if pid not in scores or systolic is None:
            continue
        for threshold, pts, label in w["instability"]["bp_stages"]:
            if systolic >= threshold:
                scores[pid].contributions.append(Contribution(
                    "instability", label, pts, f"systolic {systolic:g}"))
                break

    # --- neglect: what is undone. The only component anyone can act on --------
    from .panel import _pending_orders
    for o in _pending_orders()["actionable"]:
        pid = next((k for k, v in names.items() if v == o["patient"]), None)
        if pid is None:
            continue
        band = next((pts for months, pts in w["neglect"]["order_months"]
                     if o["months_open"] >= months), 0)
        if o["patient_has_the_condition_it_monitors"] is True:
            band += w["neglect"]["live_indication_bonus"]
        if band:
            scores[pid].contributions.append(Contribution(
                "neglect", f"{o['test']} never resulted", band,
                f"{o['months_open']} months open"))

    from .floor import compute_floor
    for f in compute_floor():
        if f.get("category") != "HFrEF therapy gap":
            continue
        for who in f.get("patients") or []:
            pid = next((k for k, v in names.items() if v == who), None)
            if pid:
                scores[pid].contributions.append(Contribution(
                    "neglect", "guideline therapy missing",
                    w["neglect"]["guideline_gap"], f["headline"]))

    for s in scores.values():
        s.total = round(sum(c.points for c in s.contributions), 2)
    return sorted(scores.values(), key=lambda s: (-s.total, s.patient))


if __name__ == "__main__":
    for s in score_panel()[:10]:
        print(s.explain())
        print()
