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
from google.genai import types

from typing import Literal

from pydantic import BaseModel, Field

from .guidelines import check_guideline, list_guidelines
from .tools import AS_OF, ScreeningSession, connect

MODEL = "gemini-2.5-pro"

# Vertex list price for gemini-2.5-pro, USD per 1M tokens, as configured for this
# project. Update if the rate changes; the token counts are measured either way.
PRICE_IN_PER_M = 1.25
PRICE_OUT_PER_M = 10.00


class Usage:
    """Token and latency accounting for one run.

    A product lead who cannot answer "what does a run cost?" has not finished the
    product. The counts come from the API's own usage_metadata, not an estimate.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def add(self, agent: str, prompt: int, output: int, seconds: float) -> None:
        self.calls.append({"agent": agent, "input_tokens": prompt,
                           "output_tokens": output, "seconds": round(seconds, 1)})

    def summary(self) -> dict:
        tin = sum(c["input_tokens"] for c in self.calls)
        tout = sum(c["output_tokens"] for c in self.calls)
        cost = tin / 1e6 * PRICE_IN_PER_M + tout / 1e6 * PRICE_OUT_PER_M
        per_agent: dict[str, dict] = {}
        for c in self.calls:
            a = per_agent.setdefault(c["agent"], {"calls": 0, "in": 0, "out": 0})
            a["calls"] += 1
            a["in"] += c["input_tokens"]
            a["out"] += c["output_tokens"]
        return {"model_calls": len(self.calls), "input_tokens": tin,
                "output_tokens": tout, "usd": round(cost, 4),
                "per_agent": per_agent}


class ExtractedFinding(BaseModel):
    headline: str = Field(description="one line naming the problem")
    patients: list[str] = Field(default_factory=list,
                                description="names exactly as written; [] if panel-level")
    severity: Literal["high", "medium", "low"]
    evidence: str = Field(description="the numbers and facts the report gave")
    recommended_action: str


class ExtractedFindings(BaseModel):
    findings: list[ExtractedFinding]


EXTRACTOR_INSTRUCTION = """\
You convert a specialist's report into structured findings. You add nothing and
you drop nothing.

Emit one finding per distinct problem the report describes, including ones
stated only in passing. Copy patient names exactly as written. Copy the numbers
from the report; never supply one it did not give. If the report explicitly says
a suspicion did NOT hold up, record that too, at severity low -- a checked and
dismissed hypothesis is a result.

Do not merge two problems into one finding because they share a patient, and do
not split one problem into several because it names several patients.

DEDUPLICATE. A report often states the same problem twice -- once as a panel
statistic and again with the patients named. That is ONE finding, not two. Emit
a single row and keep the version carrying the patient names; fold any extra
numbers into its evidence. Two findings are distinct only when they describe
different problems, not when they describe one problem at different levels of
detail.
"""


class Findings:
    """Structured findings shared by every agent in the run.

    Findings do NOT travel between agents as prose. Two runs proved why: with
    AgentTool summarising, 120 pending orders became "none found" and a list of
    names became "the service did not provide the names"; with summarisation
    off, the supervisor short-circuited and returned a specialist's report as
    its own answer. Prose is lossy in both directions.

    So each specialist writes structured rows here, and the supervisor reads
    them back deterministically. Python carries the data; the model carries the
    judgment about what it means.
    """

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def seed_floor(self) -> int:
        """Inject the guaranteed findings before any agent runs.

        These are computed, not discovered, so they are present on every run by
        construction. The five-run eval measured the alternative: Padilla's
        INR-on-a-DOAC surfaced in 1 run of 5 when a model had to choose to
        mention it. It is identical data every time; the variance was entirely
        in the choosing.
        """
        from .floor import compute_floor
        for f in compute_floor():
            self.record("guaranteed", **f)
        return len(self.rows)

    def record(self, agent: str, **kw) -> dict:
        fid = f"F{len(self.rows) + 1:02d}"
        self.rows.append({"finding_id": fid, "agent": agent, **kw})
        return {"recorded": True, "finding_id": fid, "total_findings": len(self.rows)}

    def all(self) -> list[dict]:
        return list(self.rows)


# ---------------------------------------------------------------- shared tools
def patient_snapshot(patient: str) -> dict:
    """Everything on file for one patient: demographics, problems, meds, recent labs, vitals.

    Accepts either a PAT_ID or the patient's name, because findings carry names.

    Args:
        patient: a PAT_ID like "P200001", or a name as recorded like "Stein, Larry".
    Returns:
        demographics, diagnoses, medications, latest labs, vitals, pending orders.
    """
    con = connect()
    try:
        # Findings record names, not ids. Requiring an id here made an earlier
        # run conclude the tool was broken, which it effectively was.
        row = con.execute(
            "SELECT PAT_ID, PAT_NAME, PAT_AGE, SEX_NAME FROM patient "
            "WHERE PAT_ID = ? OR lower(PAT_NAME) = lower(?)", [patient, patient]).fetchone()
        if not row:
            near = [r[0] for r in con.execute(
                "SELECT PAT_NAME FROM patient WHERE lower(PAT_NAME) LIKE lower(?) LIMIT 5",
                [f"%{patient.split(',')[0]}%"]).fetchall()]
            return {"error": f"no patient matches {patient!r}",
                    "did_you_mean": near,
                    "hint": "pass a PAT_ID like P200001 or a name exactly as recorded"}
        pat_id, d = row[0], row[1:]
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
                count(DISTINCT m.MEDICATION_ID), string_agg(DISTINCT m.DISPLAY_NAME,' + '),
                min(m.START_DATE), max(m.START_DATE),
                date_diff('day', min(m.START_DATE), max(m.START_DATE))
                FROM v_medication m JOIN patient p USING (PAT_ID) GROUP BY 1,2
                HAVING count(DISTINCT m.MEDICATION_ID)>1 ORDER BY 7 DESC""").fetchall()
            return {
                "IMPORTANT": "These are NOT necessarily concurrent. order_med has "
                             "START_DATE for every row but END_DATE and DISCON_TIME are "
                             "100% NULL and ORDER_STATUS is 'Active' on all 522 rows -- "
                             "nothing is ever recorded as stopped. Start dates within a "
                             "duplicated class are 113 to 1376 days apart (mean 814). That "
                             "pattern is sequential switching, not simultaneous therapy. "
                             "Judge by first_started / last_started / days_apart below; do "
                             "not call this triple therapy without evidence of overlap.",
                "rows": [{"patient": r[0], "class": r[1], "n_agents": r[2],
                          "agents": r[3], "first_started": str(r[4]),
                          "last_started": str(r[5]), "days_apart": r[6]} for r in rows]}
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


def blood_pressure_staging() -> dict:
    """Every patient's latest BP, staged against the named ACC/AHA thresholds.

    Use this instead of judging blood pressure by eye. An earlier run labelled
    five patients with "severe hypertension" when only two met any severe
    threshold and one of the five was 111/104 -- a reading that is not
    hypertension at all. Severity against a published cut-off is arithmetic, so
    it is computed here; deciding what to DO about a stage is your job.

    Also checks pulse pressure. Systolic and diastolic appear to have been
    generated independently in this extract, so many pairs are not physiologically
    possible -- report those as data defects, not as blood-pressure findings.

    Returns:
        thresholds used, a count per stage, and the patients in each of the
        stages that matter, plus implausible readings called out separately.
    """
    con = connect()
    try:
        rows = con.execute("""
            SELECT p.PAT_NAME, v.systolic, v.diastolic, v.RECORD_DATE,
                   v.systolic - v.diastolic AS pulse_pressure
            FROM (SELECT *, row_number() OVER (PARTITION BY PAT_ID
                           ORDER BY RECORD_DATE DESC) rn FROM v_vitals) v
            JOIN patient p USING (PAT_ID)
            WHERE v.rn = 1 ORDER BY v.systolic DESC""").fetchall()
    finally:
        con.close()

    def stage(sys_, dia):
        if sys_ > 180 or dia > 120:
            return "hypertensive crisis"
        if sys_ >= 140 or dia >= 90:
            return "stage 2"
        if sys_ >= 130 or dia >= 80:
            return "stage 1"
        if sys_ >= 120:
            return "elevated"
        return "normal"

    staged, implausible, counts = [], [], {}
    for name, sys_, dia, date, pp in rows:
        st = stage(sys_, dia)
        counts[st] = counts.get(st, 0) + 1
        rec = {"patient": name, "bp": f"{sys_}/{dia}", "stage": st,
               "pulse_pressure": pp, "recorded": str(date)}
        if pp <= 0:
            rec["data_defect"] = "diastolic >= systolic; physically impossible"
            implausible.append(rec)
        elif pp < 20 or pp > 100:
            rec["data_defect"] = f"pulse pressure {pp} outside the plausible 20-100 range"
            implausible.append(rec)
        else:
            staged.append(rec)

    return {
        "thresholds": {
            "hypertensive crisis": "systolic > 180 or diastolic > 120",
            "stage 2": "systolic >= 140 or diastolic >= 90",
            "stage 1": "systolic 130-139 or diastolic 80-89",
            "elevated": "systolic 120-129 and diastolic < 80",
            "normal": "below 120/80",
            "source": "ACC/AHA 2017 categories",
        },
        "counts_all_readings": counts,
        "USE_THESE_LABELS": "Call a patient 'hypertensive crisis' or 'stage 2' only "
                            "if this tool says so. Do not invent words like 'severe'.",
        "crisis": [r for r in staged if r["stage"] == "hypertensive crisis"],
        "stage_2": [r for r in staged if r["stage"] == "stage 2"][:20],
        "implausible_readings": implausible,
        "implausible_note": f"{len(implausible)} patients have a latest BP whose "
                            f"pulse pressure is not physiologically possible. Treat "
                            f"these as data defects, not clinical findings.",
    }


def medication_timeline(patient: str) -> dict:
    """Every medication order for one patient with its start date, in order.

    Use this before calling a same-class repeat "duplicate therapy". This
    extract records no stop date for anything, so two agents of one class may be
    a switch rather than a combination -- the start dates are the only evidence
    either way.

    Args:
        patient: a PAT_ID like "P200001", or a name as recorded.
    Returns:
        the orders in start-date order, and a per-class summary of the spread.
    """
    con = connect()
    try:
        row = con.execute("SELECT PAT_ID, PAT_NAME FROM patient "
                          "WHERE PAT_ID = ? OR lower(PAT_NAME) = lower(?)",
                          [patient, patient]).fetchone()
        if not row:
            return {"error": f"no patient matches {patient!r}"}
        pid, name = row
        orders = con.execute("""
            SELECT SIMPLE_GENERIC_C_NAME, DISPLAY_NAME, START_DATE, END_DATE,
                   ORDER_STATUS_C_NAME
            FROM order_med WHERE PAT_ID = ? ORDER BY START_DATE""", [pid]).fetchall()
    finally:
        con.close()
    per: dict[str, list] = {}
    for cls, agent, start, *_ in orders:
        per.setdefault(cls, []).append((str(start), agent))
    return {
        "patient": name, "pat_id": pid,
        "note": "END_DATE and ORDER_STATUS are uninformative in this extract -- "
                "every row is Active with no end date. Start dates are the only "
                "timing signal.",
        "orders": [{"class": o[0], "agent": o[1], "started": str(o[2]),
                    "ended": o[3], "status": o[4]} for o in orders],
        "same_class_repeats": [
            {"class": cls, "n": len(v), "first": v[0][0], "last": v[-1][0],
             "days_apart": (__import__("datetime").date.fromisoformat(v[-1][0])
                            - __import__("datetime").date.fromisoformat(v[0][0])).days,
             "sequence": [f"{d} {a}" for d, a in v]}
            for cls, v in per.items() if len(v) > 1],
    }


def find_patients(condition: str, drug_class: str) -> dict:
    """List patients carrying a diagnosis and/or on a medication class.

    Free-text matched against the diagnosis names and drug classes actually in
    the dataset. Pass "" for either to ignore it; pass both to get the
    intersection.

    Args:
        condition: part of a diagnosis name, e.g. "heart failure" or "diabetes".
        drug_class: part of a drug class or agent name, e.g. "statin", "DOAC".
    Returns:
        matched_diagnoses / matched_classes so you can see what the text hit,
        and the patients.
    """
    con = connect()
    try:
        dx_names, cls_names = [], []
        if condition.strip():
            dx_names = [r[0] for r in con.execute(
                "SELECT DISTINCT dx_name FROM v_diagnosis WHERE lower(dx_name) LIKE lower(?)",
                [f"%{condition.strip()}%"]).fetchall()]
        if drug_class.strip():
            cls_names = [r[0] for r in con.execute(
                "SELECT DISTINCT generic_class FROM v_medication "
                "WHERE lower(generic_class) LIKE lower(?) OR lower(DISPLAY_NAME) LIKE lower(?)",
                [f"%{drug_class.strip()}%", f"%{drug_class.strip()}%"]).fetchall()]
        if condition.strip() and not dx_names:
            return {"error": f"no diagnosis name contains {condition!r}",
                    "hint": "call cohort_statistic('condition_prevalence') to see them all"}
        if drug_class.strip() and not cls_names:
            return {"error": f"no drug class or agent contains {drug_class!r}",
                    "hint": "call cohort_statistic('prescribing_rates') to see them all"}

        where, params = [], []
        if dx_names:
            where.append(f"p.PAT_ID IN (SELECT PAT_ID FROM v_diagnosis WHERE dx_name IN "
                         f"({','.join('?' * len(dx_names))}))")
            params += dx_names
        if cls_names:
            where.append(f"p.PAT_ID IN (SELECT PAT_ID FROM v_medication WHERE generic_class IN "
                         f"({','.join('?' * len(cls_names))}))")
            params += cls_names
        if not where:
            return {"error": "give a condition, a drug_class, or both"}
        rows = con.execute(
            f"SELECT p.PAT_ID, p.PAT_NAME, p.PAT_AGE FROM patient p "
            f"WHERE {' AND '.join(where)} ORDER BY p.PAT_NAME", params).fetchall()
    finally:
        con.close()
    return {"matched_diagnoses": dx_names, "matched_classes": cls_names,
            "count": len(rows),
            "patients": [{"pat_id": r[0], "name": r[1], "age": r[2]} for r in rows]}


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


SEVERITY_WORDS = """\
DO NOT INVENT SEVERITY LABELS FOR BLOOD PRESSURE. Call blood_pressure_staging
and use the stage it returns. An earlier run reported "unaddressed severe
hypertension" for five patients, ranked them first through fifth, and only two
met any severe threshold -- one was 111/104, which is not hypertension. Only 7
of 153 readings in this panel reach hypertensive crisis.

The same tool flags readings whose pulse pressure is impossible. Those are data
defects and belong in a data-quality finding, not a blood-pressure one.
"""

NO_STOP_DATES = """\
ONE TRAP THAT CATCHES EVERY AGENT HERE. A patient holding two agents of the
same class is NOT evidence of concurrent therapy in this extract. Nothing is
ever recorded as stopped -- END_DATE and DISCON_TIME are entirely empty and all
522 orders read Active -- so a switch and a combination look identical. Across
the 20 duplicated patient-class pairs the start dates are 113 to 1376 days
apart, mean 814. Rogers, Jessica's four statin orders span 2022 to 2026.

Call medication_timeline before describing anything as duplicate, double or
triple therapy, and report what the start dates show. If they are months or
years apart, it is sequential switching -- say that instead. The reportable
defect is the missing discontinuation data, not the patient.
"""

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

STATE EVERY FINDING IN YOUR REPLY. Your report is parsed into structured
findings automatically, so anything you write down is captured -- but only what
you write down. Do not leave a conclusion implicit.

ALWAYS NAME THE PATIENTS. A finding without names cannot be acted on, and the
supervisor cannot recover names you leave out. Give the name exactly as recorded
-- no titles, no honorifics, no Mr/Ms. Do not infer anything about a patient
that is not in the record.

Report only what you verified, with the numbers the tools returned -- never a
number you recalled. For each finding give: what you observed, what you expected,
whether it is one record or systematic, and what it would break for a panel
manager acting on this data. Say explicitly when a suspicion did NOT hold up.
"""

INTEGRITY_INSTRUCTION += NO_STOP_DATES + SEVERITY_WORDS

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

STATE EVERY FINDING IN YOUR REPLY. Your report is parsed into structured
findings automatically, so anything you write down is captured -- but only what
you write down. Do not leave a conclusion implicit.

ALWAYS NAME THE PATIENTS. A finding without names cannot be acted on, and the
supervisor cannot recover names you leave out. Give the name exactly as recorded
-- no titles, no honorifics, no Mr/Ms. Do not infer anything about a patient
that is not in the record.

Rank what you found by how likely it is to matter, and keep it short.
"""

GUIDELINE_INSTRUCTION += NO_STOP_DATES + SEVERITY_WORDS

FOLLOWUP_INSTRUCTION = """\
You find care that was started and never finished.

The big one is orders that never returned a result. An absent result generates
no alert anywhere -- it simply is not there -- so nobody notices. Use
cohort_statistic and patient_snapshot to find them and judge which still matter.

An unreturned test matters more when the patient has the condition it monitors,
when it is old, and when nothing since supersedes it. It matters less when a
later result for the same analyte exists. Check before concluding.

_pending_orders returns ALL of them with the triage already computed: months
open, whether a later result supersedes the order, and whether the patient still
has the condition the test monitors.

REVIEW ALL OF THEM. REPORT FEW. Those are different things, and getting the
second wrong destroys the value of the first. You are writing for someone who
can work about fifteen patients a week, so a finding per open order is the same
as no list at all -- one run recorded twenty-six, which is a transcript, not
triage.

Record an order individually only when it clears a real bar: the test is
high-stakes for that patient (a potassium on an ACE inhibitor plus a diuretic,
a kidney screen in diabetes, a level for a narrow-therapeutic-index drug), or
it has been open long enough that the original clinical question is now
unanswered. At most EIGHT individual findings.

Fold everything else into ONE grouped finding -- "N further open orders,
routine monitoring, none individually urgent" -- with the count and a couple of
examples. Say how many you reviewed and how many you are reporting
individually; reviewing seventy-four and surfacing six is the right shape of
answer.

STATE EVERY FINDING IN YOUR REPLY. Your report is parsed into structured
findings automatically, so anything you write down is captured -- but only what
you write down. Do not leave a conclusion implicit.

ALWAYS NAME THE PATIENTS. A finding without names cannot be acted on, and the
supervisor cannot recover names you leave out. Give the name exactly as recorded
-- no titles, no honorifics, no Mr/Ms. Do not infer anything about a patient
that is not in the record.

Report the ones worth chasing, with the patient, the test, how long it has been
open, and why it still matters. Be brief.
"""

FOLLOWUP_INSTRUCTION += NO_STOP_DATES


# Which analytes monitor which DRUG. An INR monitors warfarin, not apixaban;
# a digoxin level monitors digoxin. The agent kept getting this from memory and
# getting it wrong -- one run reported "stale INR for a patient on warfarin"
# when the patient was on a DOAC and the other had no anticoagulant at all.
# Same rule as everywhere else: no model computes what code can compute.
_ANALYTE_MONITORS_DRUG = {
    "INR / PT": ("VKA", "INR monitors warfarin. It is NOT the monitoring test for "
                        "a DOAC -- ordering it for a DOAC patient is an error, and "
                        "ordering it for a patient on no anticoagulant needs a reason."),
    "aPTT": ("VKA", "aPTT does not monitor DOAC therapy either."),
    "Digoxin Level": ("__digoxin__", "A digoxin level is only meaningful in a patient "
                                     "taking digoxin."),
}

# Which analytes monitor which conditions. Used to decide, deterministically,
# whether an unreturned test still has a live indication.
_ANALYTE_INDICATION = {
    "HbA1c": "E11%", "Fasting Glucose": "E11%", "Urine Albumin/Creat Ratio": "E11%",
    "LDL Cholesterol": "E78%", "HDL Cholesterol": "E78%", "Total Cholesterol": "E78%",
    "Triglycerides": "E78%", "Lipid Panel — Total Chol": "E78%",
    "TSH": "E0%", "Free T3": "E0%", "Free T4": "E0%",
    "BNP": "I50%", "NT-proBNP": "I50%",
    "INR / PT": "I48%", "aPTT": "I48%",
    "Eosinophil Count": "J4%", "IgE Total": "J4%", "Peak Flow": "J4%", "SpO2": "J4%",
}


def _pending_orders() -> dict:
    """EVERY unreturned lab order, triaged deterministically. Nothing is sampled.

    Coverage used to depend on which patients the agent happened to look at: one
    run made 21 tool calls and found Schwartz, Mary's eleven-month-old potassium
    order, the next made 6 and did not. Sampling is not an acceptable basis for a
    safety net, so all 120 are returned here with the facts already computed --
    how long open, whether a later result supersedes it, and whether the patient
    still carries the condition the test monitors.

    Your job is to judge which of the ACTIONABLE ones matter and why. You do not
    need to call patient_snapshot to establish the three facts below; they are
    already here.

    Returns:
        counts, then the orders in three buckets: actionable, superseded, and
        no_live_indication.
    """
    con = connect()
    try:
        rows = con.execute("""
            SELECT p.PAT_NAME, o.PAT_ID, o.test_name, o.ORDER_DATE,
                   date_diff('month', o.ORDER_DATE, DATE '2026-05-27') AS months_open
            FROM v_lab_order o JOIN patient p USING (PAT_ID)
            WHERE o.is_pending ORDER BY months_open DESC""").fetchall()

        out = []
        for name, pid, test, ordered, months in rows:
            superseded = con.execute(
                "SELECT count(*) FROM v_lab_result WHERE PAT_ID = ? "
                "AND COMPONENT_NAME = ? AND RESULT_DATE > ?",
                [pid, test, ordered]).fetchone()[0] > 0
            # Does this test monitor a drug, and is the patient actually on it?
            drug_check = None
            if test in _ANALYTE_MONITORS_DRUG:
                cls, note = _ANALYTE_MONITORS_DRUG[test]
                if cls == "__digoxin__":
                    on_it = con.execute(
                        "SELECT count(*) FROM v_medication WHERE PAT_ID = ? "
                        "AND lower(DISPLAY_NAME) LIKE '%digoxin%'", [pid]).fetchone()[0] > 0
                    actual = "digoxin" if on_it else None
                else:
                    on_it = con.execute(
                        "SELECT count(*) FROM v_medication WHERE PAT_ID = ? "
                        "AND generic_class = ?", [pid, cls]).fetchone()[0] > 0
                    actual = ",".join(r[0] for r in con.execute(
                        "SELECT DISTINCT generic_class FROM v_medication WHERE PAT_ID = ? "
                        "AND generic_class IN ('VKA','DOAC')", [pid]).fetchall()) or None
                drug_check = {"test_monitors": cls.strip("_") if cls != "__digoxin__" else "digoxin",
                              "patient_is_on_it": on_it,
                              "patient_actually_on": actual or "nothing of that kind",
                              "note": note}

            pattern = _ANALYTE_INDICATION.get(test)
            if pattern is None:
                indication = None          # not a condition-specific monitor
            else:
                indication = con.execute(
                    "SELECT count(*) FROM v_diagnosis WHERE PAT_ID = ? AND icd10 LIKE ?",
                    [pid, pattern]).fetchone()[0] > 0
            rec = {"patient": name, "pat_id": pid, "test": test,
                   "ordered": str(ordered), "months_open": months,
                   "superseded_by_later_result": superseded,
                   "patient_has_the_condition_it_monitors": indication}
            if drug_check:
                rec["drug_monitoring_check"] = drug_check
            out.append(rec)
    finally:
        con.close()

    actionable = [o for o in out if not o["superseded_by_later_result"]
                  and o["patient_has_the_condition_it_monitors"] is not False]
    superseded = [o for o in out if o["superseded_by_later_result"]]
    no_ind = [o for o in out if not o["superseded_by_later_result"]
              and o["patient_has_the_condition_it_monitors"] is False]
    actionable.sort(key=lambda o: (-o["months_open"],
                                   o["patient_has_the_condition_it_monitors"] is not True))
    return {
        "total": len(out),
        "counts": {"actionable": len(actionable), "superseded": len(superseded),
                   "no_live_indication": len(no_ind)},
        "REVIEW_ALL_ACTIONABLE": "Every actionable order is listed. Work the list; "
                                 "do not sample it, and say how many you judged.",
        "DRUG_MONITORING": "Where an order carries drug_monitoring_check, that field "
                           "is authoritative about which drug the test monitors and "
                           "what the patient is actually on. Use it verbatim. Do not "
                           "state a drug class from memory.",
        "actionable": actionable,
        "superseded": superseded[:15],
        "no_live_indication": no_ind[:15],
    }


async def _run_agent(agent: LlmAgent, prompt: str, app: str,
                     usage: "Usage | None" = None) -> str:
    """Run one agent to completion and return its final text."""
    import time as _time
    runner = InMemoryRunner(agent=agent, app_name=app)
    sess = await runner.session_service.create_session(app_name=app, user_id="demo")
    out, t0 = "", _time.time()
    async for ev in runner.run_async(
        user_id="demo", session_id=sess.id,
        new_message=types.Content(role="user", parts=[types.Part(text=prompt)])
    ):
        # every model turn carries its own usage; sum them rather than guess
        um = getattr(ev, "usage_metadata", None)
        if um is not None and usage is not None:
            usage.add(agent.name,
                      getattr(um, "prompt_token_count", 0) or 0,
                      getattr(um, "candidates_token_count", 0) or 0,
                      _time.time() - t0)
        if ev.is_final_response() and ev.content:
            out = "".join(p.text for p in ev.content.parts if getattr(p, "text", None))
    return out


async def _extract(report: str, usage: "Usage | None" = None) -> list[dict]:
    """Turn a specialist's prose into structured findings.

    Recording used to be a tool the specialist called, which made it optional --
    one run described Schwartz, Mary's eleven-month-old potassium order in prose
    and never called record_finding, so the finding did not exist as far as the
    supervisor was concerned. Extraction is now unconditional: the specialist
    just reports, and everything it reports is structured on the way out.
    """
    if not report.strip():
        return []
    extractor = LlmAgent(name="finding_extractor", model=MODEL,
                         instruction=EXTRACTOR_INSTRUCTION,
                         output_schema=ExtractedFindings)
    raw = await _run_agent(extractor, f"Specialist report:\n\n{report}", "extract", usage)
    try:
        return [f.model_dump() for f in
                ExtractedFindings.model_validate_json(raw).findings]
    except Exception:
        return []


def _tracer(trace: list, agent_name: str):
    """AgentTool runs a specialist in a nested invocation, so its internal tool
    calls do not reach the parent event stream. A before-tool callback on each
    specialist records them, which is the part of the work worth showing."""
    def cb(tool, args, tool_context, **kwargs):
        trace.append({"agent": agent_name, "tool": getattr(tool, "name", str(tool)),
                      "args": {k: v for k, v in (args or {}).items() if k != "request"}})
        return None
    return cb


def build_supervisor(trace: list | None = None,
                     findings: "Findings | None" = None,
                     usage: "Usage | None" = None) -> LlmAgent:
    trace = trace if trace is not None else []
    findings = findings if findings is not None else Findings()
    usage = usage if usage is not None else Usage()
    if not findings.all():
        findings.seed_floor()

    def get_agent_activity() -> dict:
        """Which specialists ran and how many tool calls each made.

        Use this instead of guessing whether a specialist did any work. A
        previous run claimed one agent had been "strangely silent" while using
        three of its findings -- an invented explanation for a failure that had
        not happened.

        Returns:
            per-agent tool-call counts and findings recorded.
        """
        calls: dict[str, int] = {}
        for t in trace:
            calls[t["agent"]] = calls.get(t["agent"], 0) + 1
        recorded: dict[str, int] = {}
        for f in findings.all():
            recorded[f["agent"]] = recorded.get(f["agent"], 0) + 1
        return {"agents": [{"agent": a, "tool_calls": n,
                            "findings_recorded": recorded.get(a, 0)}
                           for a, n in sorted(calls.items())]}

    def get_all_findings() -> dict:
        """Every finding the specialists recorded, in full. Call after consulting them.

        Returns:
            findings: list of {finding_id, agent, headline, patients, severity,
                      evidence, recommended_action}.
        """
        return {"count": len(findings.all()), "findings": findings.all()}
    integrity = LlmAgent(
        name="data_integrity", model=MODEL,
        description="Investigates whether records in the extract can be trusted. "
                    "Ask it before acting on any patient list.",
        instruction=INTEGRITY_INSTRUCTION,
        before_tool_callback=_tracer(trace, "data_integrity"),
        tools=[cohort_statistic,
               patients_with_value, find_patients, medication_timeline,
               blood_pressure_staging, patient_snapshot])

    guideline = LlmAgent(
        name="guideline_concordance", model=MODEL,
        description="Finds patients missing therapy that guidelines recommend, and "
                    "judges whether each apparent gap is real.",
        instruction=GUIDELINE_INSTRUCTION,
        before_tool_callback=_tracer(trace, "guideline_concordance"),
        tools=[list_guidelines,
               check_guideline, find_patients, medication_timeline,
               blood_pressure_staging, patient_snapshot])

    followup = LlmAgent(
        name="followup", model=MODEL,
        description="Finds care started and never finished -- above all, lab orders "
                    "that never returned a result.",
        instruction=FOLLOWUP_INSTRUCTION,
        before_tool_callback=_tracer(trace, "followup"),
        tools=[_pending_orders, cohort_statistic,
               find_patients, medication_timeline, patient_snapshot])

    async def _consult(agent: LlmAgent, name: str, request: str) -> dict:
        report = await _run_agent(agent, request, f"spec_{name}", usage)
        rows = await _extract(report, usage)
        for r in rows:
            findings.record(name, **r)
        return {"specialist": name, "findings_recorded": len(rows),
                "note": "Its findings are already in the store. Read them with "
                        "get_all_findings; do not rely on this summary."}

    async def consult_data_integrity(request: str) -> dict:
        """Ask the data-integrity specialist which records cannot be trusted.

        It investigates panel statistics, chases suspicions to specific patients,
        and reports what held up and what did not. Its findings are recorded
        automatically.

        Args:
            request: what you want it to look into.
        Returns:
            how many findings it recorded. Read them with get_all_findings.
        """
        return await _consult(integrity, "data_integrity", request)

    async def consult_guideline_concordance(request: str) -> dict:
        """Ask the guideline specialist who is missing recommended therapy.

        It checks the guideline pack against the panel and judges each apparent
        gap against the patient's full regimen. Findings are recorded
        automatically.

        Args:
            request: what you want it to check.
        Returns:
            how many findings it recorded. Read them with get_all_findings.
        """
        return await _consult(guideline, "guideline_concordance", request)

    async def consult_followup(request: str) -> dict:
        """Ask the follow-up specialist what was started and never finished.

        Chiefly the orders that never returned a result. Findings are recorded
        automatically.

        Args:
            request: what you want it to look for.
        Returns:
            how many findings it recorded. Read them with get_all_findings.
        """
        return await _consult(followup, "followup", request)

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

Consult data_integrity EARLY and let what it finds change how much weight you
give each signal.

BUT DEGRADE, DO NOT REFUSE. A panel manager who is told "this data is unsafe,
come back later" has been given nothing, and their patients still need working
this week. Bad data is the normal condition of clinical data, not a reason to
stop. So:
  - Down-weight the signals the data problems actually touch, and route around
    them. Impossible sodium values make LAB-derived alerts unreliable; they say
    nothing about whether a diabetic is on a statin, or whether an order placed
    eleven months ago ever came back. Those remain actionable.
  - Put a patient on the list anyway when the reason does not depend on a value
    you cannot trust, and mark the ones where it does.
  - Withhold the list ONLY if literally no signal survives, which is not the
    case here.
State the data caveats clearly alongside the list, not instead of it.

The store already contains findings marked `guaranteed` before you start. Those
are computed deterministically, not discovered: drug-monitoring mismatches,
hypertensive crises, orders stale past six months with a live indication,
physiologically impossible values, and HFrEF therapy gaps. They are present on
every run and they are not optional -- work them into your shortlist on merit
alongside everything else, and do not re-derive or second-guess them.

Each specialist records its findings in a shared store as it works. Its chat
reply to you is only a status line -- do NOT build your list from it. After
consulting the specialists, call get_all_findings and build the shortlist from
THAT. It carries the patient names, counts and evidence in full.

CITE, DO NOT RESTATE. Every finding has a finding_id. Reference it as [F03]
rather than re-describing it, and never write a count or a lab value from
memory -- a previous run said "nineteen patients" where the store held
thirteen. If you need a number, it is in the finding's evidence field; quote it
exactly or leave it out.

DO NOT SPECULATE ABOUT YOUR OWN PROCESS. If you are about to say a specialist
found nothing, was silent, or was suppressed, call get_agent_activity first and
report what it says. A previous run claimed an agent had been "strangely
silent" while three of that agent's findings were on its own shortlist.

You may also call patient_snapshot yourself to check a specific patient before
putting them on the list.

Finish with a ranked shortlist of no more than TWELVE patients. For each: the
name, the single reason they are on the list, and what the panel manager should
actually do. Then state plainly what you deliberately left off and why, and what
you could not determine from this data. A short, honest list beats a long one.
""",
        tools=[consult_data_integrity, consult_guideline_concordance,
               consult_followup, get_all_findings, get_agent_activity,
               find_patients, patient_snapshot])


async def review_async(goal: str, verbose: bool = True) -> tuple[str, list, list, dict]:
    trace: list = []
    findings = Findings()
    usage = Usage()
    import time as _time
    _t0 = _time.time()
    runner = InMemoryRunner(agent=build_supervisor(trace, findings, usage),
                            app_name="panel")
    s = await runner.session_service.create_session(app_name="panel", user_id="demo")
    final = ""
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
        um = getattr(ev, "usage_metadata", None)
        if um is not None:
            usage.add("panel_review", getattr(um, "prompt_token_count", 0) or 0,
                      getattr(um, "candidates_token_count", 0) or 0, 0)
        if ev.is_final_response() and ev.content:
            final = "".join(p.text for p in ev.content.parts if getattr(p, "text", None))
    return final, trace, findings.all(), usage.summary() | {
        "wall_clock_seconds": round(_time.time() - _t0, 1)}


def review(goal: str = "Who on this panel needs attention this week?", verbose: bool = True):
    """Returns (report, trace, findings, usage)."""
    return asyncio.run(review_async(goal, verbose))
