"""The guaranteed floor: findings that are computed, not discovered.

A five-run stability eval showed the agents are reliable on anything backed by
a deterministic sweep and unreliable on anything competing for a reporting slot.
Schwartz, Mary's eleven-month potassium appeared in 5 of 5 runs because it sits
near the top of a sorted, complete list. Padilla, Elizabeth's INR-on-a-DOAC
appeared in 1 of 5, because it is one candidate among many and usually loses.

Both are true of the data on every run. The difference is entirely whether a
model had to choose to mention it.

So the categories where a miss actually harms someone are computed here and
injected into the findings store BEFORE the supervisor runs. The specialists
still investigate freely and add whatever they find; the floor is what cannot be
lost. The claim becomes precise: these categories are guaranteed by
construction, and discovery beyond them is agentic and varies.

Each category is deliberate. A floor that covered everything would make the
agents ornamental; one that covered nothing is what we measured at 1/5.
"""
from __future__ import annotations

from .guidelines import GUIDELINES, check_guideline
from .tools import AS_OF, ScreeningSession, connect

STALE_MONTHS = 6          # an order open longer than this has outlived its question


def _drug_monitoring_mismatches() -> list[dict]:
    """A test ordered to monitor a drug the patient is not taking. Padilla's case."""
    from .panel import _pending_orders
    out = []
    p = _pending_orders()
    for o in p["actionable"] + p["superseded"] + p["no_live_indication"]:
        d = o.get("drug_monitoring_check")
        if d and not d["patient_is_on_it"]:
            out.append({
                "headline": f"{o['test']} ordered for a patient not on the drug it monitors",
                "patients": [o["patient"]],
                "severity": "high",
                "evidence": (f"{o['test']} ordered {o['ordered']} ({o['months_open']} "
                             f"months open). It monitors {d['test_monitors']}; this "
                             f"patient is on {d['patient_actually_on']}. {d['note']}"),
                "recommended_action": ("Confirm what the patient is actually taking and "
                                       "cancel or replace the order with the right test."),
            })
    return out


def _hypertensive_crisis() -> list[dict]:
    """Crisis-range BP with a plausible pulse pressure, and no antihypertensive."""
    from .panel import blood_pressure_staging
    con = connect()
    try:
        untreated = {r[0] for r in con.execute("""
            SELECT p.PAT_NAME FROM patient p WHERE NOT EXISTS (
              SELECT 1 FROM v_medication m WHERE m.PAT_ID = p.PAT_ID
              AND m.generic_class IN ('ACEi','ARB','ARNi','Thiazide','Thiazide-like',
                                      'CCB','CCB (Rate ctrl)','Beta-blocker','MRA'))
        """).fetchall()}
    finally:
        con.close()
    crisis = blood_pressure_staging()["crisis"]
    if not crisis:
        return []
    return [{
        "headline": "Blood pressure in the hypertensive crisis range",
        "patients": [r["patient"] for r in crisis],
        "severity": "high",
        "evidence": ("ACC/AHA crisis threshold (>180 or >120) with a plausible pulse "
                     "pressure: " + "; ".join(f"{r['patient']} {r['bp']}" for r in crisis)
                     + ". Of these, not on any antihypertensive: "
                     + (", ".join(sorted(r["patient"] for r in crisis
                                         if r["patient"] in untreated)) or "none")),
        "recommended_action": "Confirm the reading and review antihypertensive therapy.",
    }]


def _stale_orders_with_indication() -> list[dict]:
    """Orders past the staleness threshold for a condition the patient still has."""
    from .panel import _pending_orders
    stale = [o for o in _pending_orders()["actionable"]
             if o["months_open"] >= STALE_MONTHS
             and o["patient_has_the_condition_it_monitors"] is True]
    if not stale:
        return []
    stale.sort(key=lambda o: -o["months_open"])
    return [{
        "headline": f"Orders open {STALE_MONTHS}+ months for a condition the patient has",
        "patients": sorted({o["patient"] for o in stale}),
        "severity": "high",
        "evidence": "; ".join(f"{o['patient']}: {o['test']}, {o['months_open']}mo"
                              for o in stale[:12])
                    + (f" (+{len(stale) - 12} more)" if len(stale) > 12 else ""),
        "recommended_action": "Re-order or close out. The original question is unanswered.",
    }]


def _impossible_values() -> list[dict]:
    bad = ScreeningSession().implausible_patients()
    if not bad:
        return []
    con = connect()
    try:
        names = dict(con.execute("SELECT PAT_ID, PAT_NAME FROM patient").fetchall())
    finally:
        con.close()
    return [{
        "headline": "Physiologically impossible laboratory values",
        "patients": sorted(names[p] for p in bad),
        "severity": "high",
        "evidence": (f"{len(bad)} patients hold at least one value outside physiologic "
                     f"limits. Examples: " + "; ".join(
                         f"{names[p]} {v[0]}" for p, v in list(bad.items())[:5])),
        "recommended_action": ("Do not act on these values. Investigate the extraction "
                               "before any lab-derived alert is trusted."),
    }]


def _guideline_gaps_high_stakes() -> list[dict]:
    """HFrEF therapy gaps only -- mortality benefit, small population, unambiguous."""
    out = []
    for g in GUIDELINES:
        if "I50.32" not in g.get("icd10_any", []):
            continue
        r = check_guideline(g["id"])
        if r.get("gaps"):
            out.append({
                "headline": f"{g['title']} — guideline gap",
                "patients": [x["name"] for x in r["gaps"]],
                "severity": "high",
                "evidence": (f"{g['id']}: {g['recommendation']} "
                             f"{r['gap_count']} of {r['in_population']} in population "
                             f"lack it. Source: {g['source']}."),
                "recommended_action": ("Review for a contraindication, or document why "
                                       "the therapy is absent."),
            })
    return out


CATEGORIES = {
    "drug monitoring mismatch": _drug_monitoring_mismatches,
    "hypertensive crisis": _hypertensive_crisis,
    "stale order with live indication": _stale_orders_with_indication,
    "impossible values": _impossible_values,
    "HFrEF therapy gap": _guideline_gaps_high_stakes,
}


# The floor's size is known before the run. seed_floor() asserts against it so
# that a data layer which has quietly stopped answering shows up as a refusal
# rather than as a panel with nothing wrong.
EXPECTED_FLOOR = 9


def compute_floor() -> list[dict]:
    """Every guaranteed finding. Deterministic -- identical on every run."""
    rows = []
    for category, fn in CATEGORIES.items():
        for f in fn():
            rows.append({**f, "category": category})
    return rows
