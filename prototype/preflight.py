"""Pre-flight clinical plausibility check.

Schema validation catches type errors. It does not catch a patient on three
direct oral anticoagulants, a PCSK9 inhibitor rate of 16%, or an 18-year-old on
Medicare. Those are *clinical* implausibilities, and recognising them needs
medical knowledge the dataset has no way to encode. That is the job the model
is doing here -- it is the knowledge base, not the inference engine. Every
number it reasons over is computed by the deterministic tools below.

Findings are attached to the screening run so a cohort built on questionable
records is labelled as such rather than presented as fact.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "accorded-lake")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-west1")

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from .rules import SHARED
from .tools import PHYSIOLOGIC_LIMITS, connect

MODEL = "gemini-2.5-pro"
CACHE = Path(__file__).parent / "preflight_findings.json"

# Ordered worst-first: the first two are true whatever this panel is.
KINDS = ("impossible", "internally inconsistent", "population-dependent")

INSTRUCTION = SHARED + """\
You are auditing a 100-patient EHR extract for CLINICAL implausibility before
any algorithm is run against it. You are not checking types or nulls -- other
tooling does that. You are checking whether a clinician would believe these
records.

YOU DO NOT KNOW WHERE THIS PANEL CAME FROM. Nothing tells you it is a general
primary-care population, and it may well not be: an extract assembled to
exercise a tool, a specialty clinic's list, or a deliberately enriched cohort
would all carry disease and prescribing rates that look absurd against national
averages and are entirely ordinary in context. "Rare in the general population"
is therefore NOT a finding on its own. Treating it as one is how an audit tells
a cardiology service its lipid clinic is implausible.

So separate what you find into three kinds, and say which each one is:

  impossible              No patient anywhere could have this value. SpO2 above
                          100%, a sodium incompatible with life. True whatever
                          the panel is.
  internally inconsistent The record contradicts itself: a drug level for a drug
                          nobody is prescribed, coverage a patient is not
                          eligible for. True whatever the panel is.
  population-dependent    Surprising only if you assume a particular population.
                          A 16% PCSK9 rate is absurd in general practice and
                          unremarkable in a refractory-lipid clinic.

The first two are defects. The third is a QUESTION FOR WHOEVER SUPPLIED THE
DATA, and you must phrase it as one: state the rate, name the population that
would make it ordinary, and say what you would need to know to settle it. Do
not dress it up as an error. Getting this wrong in the other direction is worse
than missing it -- an audit that cries wolf about a sick panel being sick is an
audit nobody reads twice.

Call the inspection tools to get real numbers. Do not guess rates.

COUNT, DO NOT QUOTE THE EXTREME. A minimum or a maximum is one record and reads
like one bad row. Before you describe any range problem, call
count_outside_plausible with the bounds you would defend clinically and report
the count: "minimum total cholesterol 0.54" and "17 of 50 total cholesterol
results are under 50 mg/dL" are the same column and a different finding, and
only the second tells anyone how much of the data to distrust.

CITE THE DATES WHEN YOU SAY DUPLICATE. The shared rules tell you to call
medication_timeline before describing anything as duplicate, double or triple
therapy. You do not have that tool -- find_duplicate_therapy is the one that
carries the dates here, and it returns the spread between the first and last
order in every pair, how many pairs started on the same day, and how many of
the 522 orders record an end date. Those numbers ARE the finding: "20 patients
on more than one agent of a class" is the alarming reading and the wrong one,
and "the orders are 113 to 1376 days apart, mean 814, none on the same day, and
not one order in the extract records a stop" is the same rows read correctly.
Quote the spread or do not use the word.

A POPULATION EXPLAINS WHO IS IN THE PANEL, NOT WHAT A BODY CAN DO. Prevalence
of disease and patterns of prescribing are population facts, and those are the
ones to hold back as questions. A measurement outside what a human body can
produce is not: no cohort, however sick, makes a haemoglobin of 25.1 or an eGFR
of 173 credible. Call check_physiologic_limits and split the distribution --
the records beyond the hard limits are impossible and say so, and only what is
left is a question about the population. Labelling the whole column
population-dependent because part of it is explainable is the failure mode that
rule was written to avoid, pointed the other way.

RULE OUT THE UNITS FIRST. A value that looks impossible is a unit mix-up until
you have checked -- cholesterol reported in mmol/L and labelled mg/dL would look
impossibly low and be entirely correct. check_reference_ranges gives you the
unit each analyte carries and whether any analyte carries more than one. That
tool also gives you the reference ranges, which are data and can themselves be
wrong: a REFERENCE_LOW of 0 is not something a laboratory reports, and a range
anchored at zero both is a defect and explains the values sitting near zero.

Call flag_finding once per distinct problem, with the observed number, what
would be expected and under which assumption, and what it breaks for anyone
screening cohorts on this data. If a pattern is plausible, leave it alone.

Finish with a one-paragraph verdict on whether this extract is safe to build
clinical logic on, separating what is broken from what merely needs confirming
with the data's owner.
"""


def _tools_for(findings: list[dict]):
    def get_condition_prevalence() -> dict:
        """Prevalence of every diagnosis in the panel, as patient counts out of 100."""
        con = connect()
        try:
            rows = con.execute("""
                SELECT icd10, any_value(dx_name) AS name, count(DISTINCT PAT_ID) AS patients
                FROM v_diagnosis GROUP BY icd10 ORDER BY patients DESC
            """).fetchall()
        finally:
            con.close()
        return {"panel_size": 100,
                "conditions": [{"icd10": r[0], "name": r[1], "patients": r[2]} for r in rows]}

    def get_prescribing_rates() -> dict:
        """How many patients are on each medication class, with the agents used."""
        con = connect()
        try:
            rows = con.execute("""
                SELECT generic_class, count(DISTINCT PAT_ID) AS patients,
                       list_sort(list(DISTINCT DISPLAY_NAME)) AS agents
                FROM v_medication GROUP BY 1 ORDER BY 2 DESC
            """).fetchall()
        finally:
            con.close()
        return {"panel_size": 100,
                "classes": [{"drug_class": r[0], "patients": r[1], "agents": r[2]} for r in rows]}

    def find_duplicate_therapy() -> dict:
        """Patients on more than one agent of a class, WITH the start-date spread.

        The spread is the whole finding, and it is computed here rather than
        judged: nothing in this extract is ever recorded as stopped, so a switch
        and a combination are the same rows. Without the dates an auditor can
        only report "20 patients on duplicate therapy", which is the alarming
        reading and the wrong one -- the orders are years apart.
        """
        con = connect()
        try:
            rows = con.execute("""
                SELECT p.PAT_NAME, m.generic_class, count(DISTINCT m.MEDICATION_ID) AS n,
                       string_agg(DISTINCT m.DISPLAY_NAME, ' + ') AS agents,
                       min(m.START_DATE), max(m.START_DATE),
                       date_diff('day', min(m.START_DATE), max(m.START_DATE)) AS span
                FROM v_medication m JOIN patient p USING (PAT_ID)
                GROUP BY 1,2 HAVING count(DISTINCT m.MEDICATION_ID) > 1
                ORDER BY span DESC, 1
            """).fetchall()
            stopped = con.execute("""
                SELECT count(*), count(END_DATE), count(DISCON_TIME) FROM order_med
            """).fetchone()
        finally:
            con.close()
        spans = [r[6] for r in rows if r[6] is not None]
        return {
            "duplications": [{"patient": r[0], "drug_class": r[1],
                              "distinct_agents": r[2], "agents": r[3],
                              "first_start": str(r[4]), "last_start": str(r[5]),
                              "days_between_first_and_last": r[6]} for r in rows],
            "orders_total": stopped[0],
            "orders_with_an_end_date": stopped[1],
            "orders_with_a_discontinuation_time": stopped[2],
            "days_apart_min": min(spans) if spans else None,
            "days_apart_max": max(spans) if spans else None,
            "days_apart_mean": round(sum(spans) / len(spans)) if spans else None,
            "concurrent_same_day": sum(1 for x in spans if x == 0),
            "note": ("Read the spread before calling any of this duplicate therapy. "
                     "No order in this extract carries an end date or a "
                     "discontinuation time, so a patient who switched agents and a "
                     "patient taking both look identical in these rows. The "
                     "reportable defect is the missing discontinuation data."),
        }

    def get_demographics() -> dict:
        """Age distribution, sex, insurance, and how coverage lines up with age."""
        con = connect()
        try:
            age = con.execute("""SELECT min(PAT_AGE), round(avg(PAT_AGE),1), median(PAT_AGE),
                                 max(PAT_AGE), count(*) FILTER (WHERE PAT_AGE>=65) FROM patient""").fetchone()
            ins = con.execute("""SELECT e.INSURANCE, count(DISTINCT e.PAT_ID),
                                 min(p.PAT_AGE), median(p.PAT_AGE), max(p.PAT_AGE)
                                 FROM pat_enc e JOIN patient p USING (PAT_ID) GROUP BY 1""").fetchall()
            visit = con.execute("""SELECT DISTINCT PRC_NAME, ENC_TYPE_C_NAME FROM pat_enc""").fetchall()
        finally:
            con.close()
        return {"age": {"min": age[0], "mean": age[1], "median": age[2], "max": age[3],
                        "aged_65_plus": age[4], "panel_size": 100},
                "insurance": [{"payer": r[0], "patients": r[1], "min_age": r[2],
                               "median_age": r[3], "max_age": r[4]} for r in ins],
                "visit_types": [{"purpose": r[0], "encounter_type": r[1]} for r in visit]}

    def check_lab_plausibility() -> dict:
        """Min/max/median for each analyte against its reference range."""
        con = connect()
        try:
            rows = con.execute("""
                SELECT COMPONENT_NAME, any_value(unit), count(*),
                       round(min(value),2), round(median(value),2), round(max(value),2),
                       any_value(REFERENCE_LOW), any_value(REFERENCE_HIGH)
                FROM v_lab_result GROUP BY 1 ORDER BY 1
            """).fetchall()
        finally:
            con.close()
        return {"analytes": [{"analyte": r[0], "unit": r[1], "n": r[2], "min": r[3],
                              "median": r[4], "max": r[5], "ref_low": r[6],
                              "ref_high": r[7]} for r in rows]}

    def check_reference_ranges() -> dict:
        """The reference range each analyte is reported against, and whether it is sane.

        Ranges are data too, and these ones are not all credible: nine analytes
        report a REFERENCE_LOW of 0, which no laboratory does for a cholesterol
        or a haemoglobin. A range anchored at zero is a defect in its own right
        and it explains the values -- results drawn across [0, upper] run all
        the way down to nothing.
        """
        con = connect()
        try:
            rows = con.execute("""
                SELECT COMPONENT_NAME, any_value(unit), count(*),
                       any_value(REFERENCE_LOW), any_value(REFERENCE_HIGH),
                       round(min(value),2), round(max(value),2)
                FROM v_lab_result GROUP BY 1 ORDER BY 1
            """).fetchall()
            mixed = con.execute("""
                SELECT COMPONENT_NAME, string_agg(DISTINCT unit, ', ')
                FROM v_lab_result GROUP BY 1 HAVING count(DISTINCT unit) > 1
            """).fetchall()
        finally:
            con.close()
        return {
            "analytes": [{"analyte": r[0], "unit": r[1], "n": r[2],
                          "ref_low": r[3], "ref_high": r[4],
                          "min": r[5], "max": r[6],
                          "ref_low_is_zero": r[3] == 0} for r in rows],
            "analytes_with_a_zero_reference_low": sum(1 for r in rows if r[3] == 0),
            "analytes_with_mixed_units": [{"analyte": m[0], "units": m[1]} for m in mixed],
            "note": ("Every analyte carries one unit throughout -- a value that looks "
                     "wrong is not a unit mix-up. Checked because it is the first "
                     "thing to rule out: cholesterol reported in mmol/L and labelled "
                     "mg/dL would look impossibly low and be perfectly correct."),
        }

    def check_physiologic_limits() -> dict:
        """Results beyond the hard limits in tools.PHYSIOLOGIC_LIMITS.

        These bounds need no clinical judgement and no population: a value past
        them is a bad record whoever the patient is. Call this before deciding
        anything is population-dependent, because a distribution can be both --
        an eGFR of 130 is hyperfiltration and an eGFR of 173 is not a kidney.
        """
        con = connect()
        try:
            out = []
            for analyte, (lo, hi) in PHYSIOLOGIC_LIMITS.items():
                r = con.execute("""
                    SELECT count(*), count(*) FILTER (WHERE value < ? OR value > ?),
                           round(min(value),2), round(max(value),2)
                    FROM v_lab_result WHERE COMPONENT_NAME = ?
                """, [lo, hi, analyte]).fetchone()
                if r and r[0]:
                    out.append({"analyte": analyte, "hard_low": lo, "hard_high": hi,
                                "n": r[0], "beyond_limits": r[1],
                                "min": r[2], "max": r[3]})
        finally:
            con.close()
        return {"analytes": out,
                "note": ("beyond_limits > 0 means that many records are impossible, "
                         "whatever population this is. Report those as impossible and "
                         "judge only the remainder against the population.")}

    def count_outside_plausible(analyte: str, low: float, high: float) -> dict:
        """How many results for one analyte fall outside a range YOU specify.

        Use this before describing a range problem. A minimum is one record and
        reads like one bad row; the count is the finding. Total cholesterol has
        a minimum of 0.54, which sounds like an outlier, and 17 of its 50
        results are under 50 mg/dL, which is a third of the column.

        Args:
            analyte: exact COMPONENT_NAME, e.g. "Total Cholesterol".
            low: below this is not physiologically plausible.
            high: above this is not physiologically plausible.
        """
        con = connect()
        try:
            r = con.execute("""
                SELECT count(*), count(*) FILTER (WHERE value < ?),
                       count(*) FILTER (WHERE value > ?),
                       round(min(value),2), round(max(value),2)
                FROM v_lab_result WHERE COMPONENT_NAME = ?
            """, [low, high, analyte]).fetchone()
        finally:
            con.close()
        if not r or not r[0]:
            return {"analyte": analyte, "error": "no results under that exact name"}
        return {"analyte": analyte, "n": r[0], "below": r[1], "above": r[2],
                "outside": r[1] + r[2],
                "share_outside": round((r[1] + r[2]) / r[0], 3),
                "min": r[3], "max": r[4]}

    def flag_finding(title: str, severity: str, kind: str, observed: str,
                     expected: str, impact: str, affected: str,
                     plausible_if: str = "") -> dict:
        """Record one clinical-plausibility problem.

        Args:
            title: short name, e.g. "SpO2 above 100%".
            severity: "high", "medium" or "low".
            kind: "impossible", "internally inconsistent" or
                "population-dependent" -- the first two are defects whatever
                this panel is, the third is a question for the data's owner.
            observed: what the data actually shows, with numbers.
            expected: what would be expected, and under which assumption.
            impact: what this breaks for anyone screening cohorts on this data.
            affected: the patients, codes or classes involved.
            plausible_if: REQUIRED for population-dependent findings -- the
                population in which this rate would be unremarkable.
        """
        kind = kind if kind in KINDS else "population-dependent"
        f = {"title": title, "severity": severity, "kind": kind,
             "observed": observed, "expected": expected, "impact": impact,
             "affected": affected, "plausible_if": plausible_if}
        findings.append(f)
        return {"recorded": True, "count": len(findings)}

    return [get_condition_prevalence, get_prescribing_rates, find_duplicate_therapy,
            get_demographics, check_lab_plausibility, check_reference_ranges,
            check_physiologic_limits, count_outside_plausible, flag_finding]


async def run_async(verbose: bool = True) -> tuple[list[dict], str]:
    findings: list[dict] = []
    agent = LlmAgent(name="plausibility_linter", model=MODEL,
                     description="Audits an EHR extract for clinical implausibility.",
                     instruction=INSTRUCTION, tools=_tools_for(findings))
    runner = InMemoryRunner(agent=agent, app_name="preflight")
    s = await runner.session_service.create_session(app_name="preflight", user_id="demo")
    final = ""
    async for ev in runner.run_async(
        user_id="demo", session_id=s.id,
        new_message=types.Content(role="user", parts=[types.Part(
            text="Audit this extract for clinical plausibility. Inspect before judging.")])
    ):
        if verbose and ev.content and ev.content.parts:
            for p in ev.content.parts:
                if getattr(p, "function_call", None):
                    n = p.function_call.name
                    extra = dict(p.function_call.args).get("title", "")
                    print(f"  -> {n}{'  ' + extra if extra else ''}")
        if ev.is_final_response() and ev.content:
            final = "".join(p.text for p in ev.content.parts if getattr(p, "text", None))
    CACHE.write_text(json.dumps({"findings": findings, "verdict": final}, indent=1))
    return findings, final


def run(verbose: bool = True):
    return asyncio.run(run_async(verbose))


def cached() -> tuple[list[dict], str]:
    """Load the last run. The demo uses this so it never waits on the model."""
    if not CACHE.exists():
        return [], ""
    blob = json.loads(CACHE.read_text())
    return blob["findings"], blob["verdict"]
