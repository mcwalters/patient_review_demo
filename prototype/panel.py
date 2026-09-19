"""Panel review: a supervisor agent that delegates to three specialists.

The user is the panel manager -- the population-health RN or care coordinator
who works a list rather than a schedule and can meaningfully review perhaps
fifteen patients a week out of a hundred. Attention is the scarce resource, so
the job is ranking, not retrieval.

The supervisor owns an open goal and decides which specialist to consult and in
what order. Nothing scripts its path. Each specialist reaches the data only
through deterministic tools, so no model writes SQL anywhere in the system.

  supervisor
    ├── data_integrity      which records cannot be trusted, and why
    ├── guideline_concordance   who is missing recommended therapy
    └── followup            what was started and never finished
"""
from __future__ import annotations

import asyncio
import os

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "accorded-lake")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-west1")

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

from .guidelines import check_guideline, list_guidelines
from .tools import ScreeningSession, connect

MODEL = "gemini-2.5-pro"


# ---------------------------------------------------------------- shared tools
def patient_snapshot(pat_id: str) -> dict:
    """Everything on file for one patient: demographics, problems, meds, recent labs, vitals.

    Args:
        pat_id: the PAT_ID, e.g. "P200001".
    Returns:
        demographics, diagnoses, medications, latest labs and vitals.
    """
    con = connect()
    try:
        d = con.execute("SELECT PAT_NAME, PAT_AGE, SEX_NAME FROM patient WHERE PAT_ID=?",
                        [pat_id]).fetchone()
        if not d:
            return {"error": f"no such patient {pat_id}"}
        dx = con.execute("SELECT DISTINCT icd10, dx_name FROM v_diagnosis WHERE PAT_ID=?",
                         [pat_id]).fetchall()
        rx = con.execute("SELECT DISTINCT DISPLAY_NAME, generic_class FROM v_medication "
                         "WHERE PAT_ID=?", [pat_id]).fetchall()
        labs = con.execute("""
            SELECT COMPONENT_NAME, value, unit, REFERENCE_LOW, REFERENCE_HIGH,
                   is_abnormal, RESULT_DATE FROM (
              SELECT *, row_number() OVER (PARTITION BY COMPONENT_NAME
                       ORDER BY RESULT_DATE DESC) rn
              FROM v_lab_result WHERE PAT_ID=?) WHERE rn=1 ORDER BY is_abnormal DESC""",
            [pat_id]).fetchall()
        vit = con.execute("SELECT systolic, diastolic, bmi, heart_rate, RECORD_DATE "
                          "FROM v_vitals WHERE PAT_ID=? ORDER BY RECORD_DATE DESC LIMIT 1",
                          [pat_id]).fetchone()
        pend = con.execute("SELECT test_name, ORDER_DATE FROM v_lab_order "
                           "WHERE PAT_ID=? AND is_pending", [pat_id]).fetchall()
    finally:
        con.close()
    return {
        "pat_id": pat_id, "name": d[0], "age": d[1], "sex": d[2],
        "diagnoses": [{"icd10": r[0], "name": r[1]} for r in dx],
        "medications": [{"agent": r[0], "class": r[1]} for r in rx],
        "latest_labs": [{"analyte": r[0], "value": r[1], "unit": r[2],
                         "ref_low": r[3], "ref_high": r[4], "abnormal": r[5],
                         "date": str(r[6])} for r in labs],
        "latest_vitals": ({"systolic": vit[0], "diastolic": vit[1], "bmi": vit[2],
                           "heart_rate": vit[3], "date": str(vit[4])} if vit else None),
        "orders_never_resulted": [{"test": r[0], "ordered": str(r[1])} for r in pend],
    }


# ------------------------------------------------------- data integrity agent
def cohort_statistic(metric: str) -> dict:
    """Panel-level statistics for judging whether a pattern is plausible.

    Args:
        metric: one of "condition_prevalence", "prescribing_rates",
                "duplicate_therapy", "lab_ranges", "demographics",
                "orphan_lab_results".
    Returns:
        the requested statistic, computed over all 100 patients.
    """
    con = connect()
    try:
        if metric == "condition_prevalence":
            rows = con.execute("SELECT icd10, any_value(dx_name), count(DISTINCT PAT_ID) "
                               "FROM v_diagnosis GROUP BY 1 ORDER BY 3 DESC").fetchall()
            return {"panel": 100, "rows": [{"icd10": r[0], "name": r[1], "patients": r[2]}
                                           for r in rows]}
        if metric == "prescribing_rates":
            rows = con.execute("SELECT generic_class, count(DISTINCT PAT_ID), "
                               "list_sort(list(DISTINCT DISPLAY_NAME)) FROM v_medication "
                               "GROUP BY 1 ORDER BY 2 DESC").fetchall()
            return {"panel": 100, "rows": [{"drug_class": r[0], "patients": r[1],
                                            "agents": r[2]} for r in rows]}
        if metric == "duplicate_therapy":
            rows = con.execute("""SELECT p.PAT_NAME, m.generic_class,
                count(DISTINCT m.MEDICATION_ID), string_agg(DISTINCT m.DISPLAY_NAME,' + ')
                FROM v_medication m JOIN patient p USING (PAT_ID) GROUP BY 1,2
                HAVING count(DISTINCT m.MEDICATION_ID)>1 ORDER BY 3 DESC""").fetchall()
            return {"rows": [{"patient": r[0], "class": r[1], "n_agents": r[2],
                              "agents": r[3]} for r in rows]}
        if metric == "lab_ranges":
            rows = con.execute("""SELECT COMPONENT_NAME, any_value(unit), count(*),
                round(min(value),2), round(median(value),2), round(max(value),2),
                any_value(REFERENCE_LOW), any_value(REFERENCE_HIGH)
                FROM v_lab_result GROUP BY 1 ORDER BY 1""").fetchall()
            return {"rows": [{"analyte": r[0], "unit": r[1], "n": r[2], "min": r[3],
                              "median": r[4], "max": r[5], "ref_low": r[6],
                              "ref_high": r[7]} for r in rows]}
        if metric == "demographics":
            age = con.execute("""SELECT min(PAT_AGE), round(avg(PAT_AGE),1), median(PAT_AGE),
                max(PAT_AGE), count(*) FILTER (WHERE PAT_AGE>=65) FROM patient""").fetchone()
            ins = con.execute("""SELECT e.INSURANCE, count(DISTINCT e.PAT_ID),
                min(p.PAT_AGE), median(p.PAT_AGE), max(p.PAT_AGE) FROM pat_enc e
                JOIN patient p USING (PAT_ID) GROUP BY 1""").fetchall()
            return {"age": {"min": age[0], "mean": age[1], "median": age[2],
                            "max": age[3], "over_65": age[4]},
                    "insurance": [{"payer": r[0], "patients": r[1], "min_age": r[2],
                                   "median_age": r[3], "max_age": r[4]} for r in ins],
                    "note": "All 153 encounters are Annual Wellness Visits."}
        if metric == "orphan_lab_results":
            rows = con.execute("""SELECT r.COMPONENT_NAME, count(DISTINCT r.PAT_ID) pts,
                (SELECT count(DISTINCT m.PAT_ID) FROM v_medication m
                 WHERE lower(m.DISPLAY_NAME) LIKE '%'||lower(split_part(r.COMPONENT_NAME,' ',1))||'%')
                FROM v_lab_result r GROUP BY 1 ORDER BY 2 DESC""").fetchall()
            return {"note": "Analytes that monitor a drug, next to how many patients "
                            "are actually on a drug of that name.",
                    "rows": [{"analyte": r[0], "patients_with_result": r[1],
                              "patients_on_matching_drug": r[2]} for r in rows]}
        return {"error": f"unknown metric {metric!r}"}
    finally:
        con.close()


def patients_with_value(analyte: str, below: str, above: str) -> dict:
    """Which patients hold a lab value outside a bound. Use to confirm a suspicion.

    Args:
        analyte: exact analyte name, e.g. "Sodium".
        below: value below which to flag, "" to skip.
        above: value above which to flag, "" to skip.
    Returns:
        matching patients with the value and date.
    """
    clauses, params = ["COMPONENT_NAME = ?"], [analyte]
    if below.strip():
        clauses.append("value < ?"); params.append(float(below))
    if above.strip():
        clauses.append("value > ?"); params.append(float(above))
    con = connect()
    try:
        rows = con.execute(
            f"SELECT p.PAT_NAME, r.value, r.unit, r.RESULT_DATE FROM v_lab_result r "
            f"JOIN patient p USING (PAT_ID) WHERE {' AND '.join(clauses)} "
            f"ORDER BY r.value DESC", params).fetchall()
    finally:
        con.close()
    return {"analyte": analyte, "matches": [{"patient": r[0], "value": r[1],
            "unit": r[2], "date": str(r[3])} for r in rows]}


INTEGRITY_INSTRUCTION = """\
You investigate whether this EHR extract can be trusted. You are not checking
types or nulls -- other tooling does that. You are asking whether a clinician
would believe these records.

INVESTIGATE, do not just report. Form a suspicion from a statistic, then chase
it with a second and third call before concluding. A rate that looks wrong may
be explained by the population; a contradiction may be one bad row or a
systematic extraction fault, and those need different responses. Use
patients_with_value and patient_snapshot to find out which.

Judge against real practice: is this prevalence plausible for a primary-care
panel, is this prescribing rate plausible for an expensive specialist drug, are
there combinations that should never co-occur, do demographics line up with
coverage and visit type, are values physiologically possible.

Report only what you verified, with the numbers the tools returned -- never a
number you recalled. For each finding give: what you observed, what you expected,
whether it is one record or systematic, and what it would break for a panel
manager acting on this data. Say explicitly when a suspicion did NOT hold up.
"""

GUIDELINE_INSTRUCTION = """\
You check whether this panel's care matches guideline recommendations.

Call list_guidelines first, then check_guideline for each that is relevant.
check_guideline tells you who is in the population and who lacks the expected
therapy. It does NOT decide whether a gap is real -- that is your judgment.

For each gap, look at the patient's FULL regimen before calling it a gap. Ask
whether something else in the regimen reasonably substitutes, whether the
apparent gap has a plausible clinical explanation, and whether the guideline's
own caveats make it inapplicable here. A patient on a PCSK9 inhibitor is not
untreated for cholesterol just because they are not on a statin.

Be clear about what this dataset cannot tell you: there is no intolerance
history, no contraindication list, no patient preference, and no note content
beyond templated prose. So you can identify a gap worth a human looking at; you
cannot conclude that care was wrong. Say so.

Rank what you found by how likely it is to matter, and keep it short.
"""

FOLLOWUP_INSTRUCTION = """\
You find care that was started and never finished.

The big one is orders that never returned a result. An absent result generates
no alert anywhere -- it simply is not there -- so nobody notices. Use
cohort_statistic and patient_snapshot to find them and judge which still matter.

An unreturned test matters more when the patient has the condition it monitors,
when it is old, and when nothing since supersedes it. It matters less when a
later result for the same analyte exists. Check before concluding.

Report the ones worth chasing, with the patient, the test, how long it has been
open, and why it still matters. Be brief.
"""


def _pending_orders() -> dict:
    """Lab orders placed at a visit that never returned a result.

    Returns:
        every unresulted order with patient, test and how long it has been open.
    """
    con = connect()
    try:
        rows = con.execute("""
            SELECT p.PAT_NAME, o.PAT_ID, o.test_name, o.ORDER_DATE,
                   date_diff('month', o.ORDER_DATE, DATE '2026-05-27') AS months_open
            FROM v_lab_order o JOIN patient p USING (PAT_ID)
            WHERE o.is_pending ORDER BY months_open DESC""").fetchall()
    finally:
        con.close()
    return {"total": len(rows),
            "orders": [{"patient": r[0], "pat_id": r[1], "test": r[2],
                        "ordered": str(r[3]), "months_open": r[4]} for r in rows]}


def build_supervisor() -> LlmAgent:
    integrity = LlmAgent(
        name="data_integrity", model=MODEL,
        description="Investigates whether records in the extract can be trusted. "
                    "Ask it before acting on any patient list.",
        instruction=INTEGRITY_INSTRUCTION,
        tools=[cohort_statistic, patients_with_value, patient_snapshot])

    guideline = LlmAgent(
        name="guideline_concordance", model=MODEL,
        description="Finds patients missing therapy that guidelines recommend, and "
                    "judges whether each apparent gap is real.",
        instruction=GUIDELINE_INSTRUCTION,
        tools=[list_guidelines, check_guideline, patient_snapshot])

    followup = LlmAgent(
        name="followup", model=MODEL,
        description="Finds care started and never finished -- above all, lab orders "
                    "that never returned a result.",
        instruction=FOLLOWUP_INSTRUCTION,
        tools=[_pending_orders, cohort_statistic, patient_snapshot])

    return LlmAgent(
        name="panel_review", model=MODEL,
        description="Reviews a patient panel and decides who needs attention.",
        instruction="""\
You advise a panel manager -- the nurse or care coordinator who works a list of
100 patients between visits and can meaningfully review perhaps fifteen a week.
Attention is their scarce resource. Your job is ranking, not retrieval: almost
every patient has something, so a long list is the same as no list.

You have three specialists. Decide which to consult and in what order; nothing
scripts your path.

  data_integrity          which records cannot be trusted, and why
  guideline_concordance   who is missing recommended therapy
  followup                what was started and never finished

Consult data_integrity EARLY. A shortlist built on records that cannot be
correct is worse than no shortlist, because it spends the scarce resource on
noise. Let what it finds change how much weight you give the others.

You may also call patient_snapshot yourself to check a specific patient before
putting them on the list.

Finish with a ranked shortlist of no more than TWELVE patients. For each: the
name, the single reason they are on the list, and what the panel manager should
actually do. Then state plainly what you deliberately left off and why, and what
you could not determine from this data. A short, honest list beats a long one.
""",
        tools=[AgentTool(agent=integrity), AgentTool(agent=guideline),
               AgentTool(agent=followup), patient_snapshot])


async def review_async(goal: str, verbose: bool = True) -> tuple[str, list]:
    runner = InMemoryRunner(agent=build_supervisor(), app_name="panel")
    s = await runner.session_service.create_session(app_name="panel", user_id="demo")
    trace, final = [], ""
    async for ev in runner.run_async(
        user_id="demo", session_id=s.id,
        new_message=types.Content(role="user", parts=[types.Part(text=goal)])
    ):
        if ev.content and ev.content.parts:
            for p in ev.content.parts:
                if getattr(p, "function_call", None):
                    call = {"agent": ev.author, "tool": p.function_call.name,
                            "args": dict(p.function_call.args)}
                    trace.append(call)
                    if verbose:
                        a = {k: v for k, v in call["args"].items() if k != "request"}
                        print(f"  [{call['agent']:<20}] -> {call['tool']}({str(a)[:60]})")
        if ev.is_final_response() and ev.content:
            final = "".join(p.text for p in ev.content.parts if getattr(p, "text", None))
    return final, trace


def review(goal: str = "Who on this panel needs attention this week?", verbose: bool = True):
    return asyncio.run(review_async(goal, verbose))
