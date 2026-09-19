"""Guideline concordance: does this panel's care match what guidelines recommend?

THE GUIDELINE PACK BELOW IS A DEMO SUBSET, NOT A CLINICAL REFERENCE. Each entry
is a paraphrase of a widely accepted recommendation, written for this prototype
and attributed to its source body. It omits the caveats real guidelines carry
and must not be used to make care decisions.

FOUR recommendations, chosen so each is defensible rather than merely available.
The population is small, the recommendation is unambiguous, and the absence is
worth a clinician's time. Four were cut for failing that test:

  anticoagulation in AF    its own caveat disqualified it -- CHA2DS2-VASc needs
                           prior stroke and vascular disease, and neither is in
                           this extract
  metformin first-line     21 gaps in a population of 28, and the
                           contraindications that would explain them are not
                           recorded
  statin in hyperlipidaemia  31 gaps in a population of 50; at that volume it is
                           a population-health campaign, not a weekly worklist,
                           and the supervisor set it aside on every run
  TSH monitoring           real but minor, and not worth another surface to
                           defend

A real deployment carries a maintained library with a version and an owner. The
point here is the mechanism, not the coverage.

The division of labour matters. The pack supplies WHAT is recommended. The
deterministic tools supply WHO the patient is. The model supplies the judgment
that neither encodes: whether a recommendation actually applies to this patient,
whether an apparent gap has a defensible reason, and how to rank what is worth
a clinician's attention. That judgment is the part no lookup table holds.
"""
from __future__ import annotations

from .tools import connect

# population: evaluated by the deterministic tool below, not by the model.
GUIDELINES = [
    {
        "id": "G1",
        "title": "Statin therapy in type 2 diabetes",
        "population": "type 2 diabetes, age 40-75",
        "recommendation": "Moderate-intensity statin is recommended for adults "
                          "aged 40-75 with diabetes, irrespective of baseline LDL.",
        "source": "ADA Standards of Care; ACC/AHA cholesterol guideline",
        "icd10_any": ["E11.9", "E11.51", "E11.65"],
        "age_min": 40, "age_max": 75,
        "expected_classes": ["Statin", "PCSK9i"],
    },
    {
        "id": "G2",
        "title": "Beta-blocker in HFrEF",
        "population": "heart failure with reduced ejection fraction",
        "recommendation": "An evidence-based beta-blocker is recommended in HFrEF "
                          "to reduce mortality and hospitalisation.",
        "source": "ACC/AHA/HFSA heart failure guideline",
        "icd10_any": ["I50.32", "I50.33", "I50.9"],
        "expected_classes": ["Beta-blocker"],
    },
    {
        "id": "G3",
        "title": "Renin-angiotensin inhibition in HFrEF",
        "population": "heart failure with reduced ejection fraction",
        "recommendation": "ARNi, ACE inhibitor or ARB is recommended in HFrEF, "
                          "with ARNi preferred where tolerated.",
        "source": "ACC/AHA/HFSA heart failure guideline",
        "icd10_any": ["I50.32", "I50.33", "I50.9"],
        "expected_classes": ["ARNi", "ACEi", "ARB"],
    },
    {
        "id": "G4",
        "title": "SGLT2 inhibition in HFrEF",
        "population": "heart failure with reduced ejection fraction",
        "recommendation": "An SGLT2 inhibitor is recommended in HFrEF regardless "
                          "of diabetes status.",
        "source": "ACC/AHA/HFSA heart failure guideline",
        "icd10_any": ["I50.32", "I50.33", "I50.9"],
        "expected_classes": ["SGLT2i"],
    },
]

DISCLAIMER = ("Demo subset of paraphrased recommendations, not a clinical "
              "reference. Attribution names the source body; the wording is this "
              "prototype's and omits caveats the real guidelines carry.")


def list_guidelines() -> dict:
    """The guideline pack this prototype checks against.

    Returns:
        disclaimer, and a list of {id, title, population, recommendation, source}.
    """
    return {"disclaimer": DISCLAIMER,
            "guidelines": [{k: g[k] for k in
                            ("id", "title", "population", "recommendation", "source")
                            if k in g} | ({"caveat": g["caveat"]} if "caveat" in g else {})
                           for g in GUIDELINES]}


def _by_id(gid: str) -> dict | None:
    return next((g for g in GUIDELINES if g["id"] == gid), None)


def check_guideline(guideline_id: str) -> dict:
    """Find who the guideline applies to and whether the expected therapy is present.

    Deterministic. Returns the population and, for each patient, what they are
    actually on -- it does NOT decide whether a gap is justified.

    Args:
        guideline_id: an id from list_guidelines, e.g. "G1".
    Returns:
        in_population, concordant and gaps, each with the patient's actual regimen.
    """
    g = _by_id(guideline_id)
    if not g:
        return {"error": f"unknown guideline {guideline_id}"}

    con = connect()
    try:
        if "icd10_any" in g:
            codes = ",".join("?" * len(g["icd10_any"]))
            sql = (f"SELECT DISTINCT d.PAT_ID FROM v_diagnosis d "
                   f"WHERE d.icd10 IN ({codes})")
            params = list(g["icd10_any"])
        else:
            cls = ",".join("?" * len(g["requires_classes"]))
            sql = (f"SELECT DISTINCT m.PAT_ID FROM v_medication m "
                   f"WHERE m.generic_class IN ({cls})")
            params = list(g["requires_classes"])
        pop = [r[0] for r in con.execute(sql, params).fetchall()]

        if g.get("age_min") or g.get("age_max"):
            lo, hi = g.get("age_min", 0), g.get("age_max", 200)
            ok = {r[0] for r in con.execute(
                "SELECT PAT_ID FROM patient WHERE PAT_AGE BETWEEN ? AND ?", [lo, hi]).fetchall()}
            pop = [p for p in pop if p in ok]
        if not pop:
            return {"guideline": g["title"], "in_population": 0, "concordant": [], "gaps": []}

        ph = ",".join("?" * len(pop))
        names = dict(con.execute(
            f"SELECT PAT_ID, PAT_NAME FROM patient WHERE PAT_ID IN ({ph})", pop).fetchall())
        ages = dict(con.execute(
            f"SELECT PAT_ID, PAT_AGE FROM patient WHERE PAT_ID IN ({ph})", pop).fetchall())
        regimen: dict[str, list[str]] = {}
        for pid, cls, agent in con.execute(
            f"SELECT DISTINCT PAT_ID, generic_class, DISPLAY_NAME FROM v_medication "
            f"WHERE PAT_ID IN ({ph})", pop).fetchall():
            regimen.setdefault(pid, []).append(f"{agent} [{cls}]")

        met, gaps = [], []
        for pid in sorted(pop):
            classes = {r.split("[")[-1].rstrip("]") for r in regimen.get(pid, [])}
            if "expected_classes" in g:
                satisfied = bool(classes & set(g["expected_classes"]))
            else:
                n = con.execute(
                    "SELECT count(*) FROM v_lab_result WHERE PAT_ID = ? AND COMPONENT_NAME = ?",
                    [pid, g["expected_analyte"]]).fetchone()[0]
                satisfied = n > 0
            row = {"pat_id": pid, "name": names[pid], "age": ages[pid],
                   "full_regimen": sorted(regimen.get(pid, []))}
            (met if satisfied else gaps).append(row)
    finally:
        con.close()

    return {"guideline": g["title"], "source": g["source"],
            "recommendation": g["recommendation"],
            "caveat": g.get("caveat", ""),
            "in_population": len(pop),
            "concordant_count": len(met), "gap_count": len(gaps),
            "gaps": gaps, "concordant": met[:5]}
