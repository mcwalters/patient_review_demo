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

from .tools import connect

MODEL = "gemini-2.5-pro"
CACHE = Path(__file__).parent / "preflight_findings.json"

INSTRUCTION = """\
You are auditing a 100-patient EHR extract for CLINICAL implausibility before
any algorithm is run against it. You are not checking types or nulls -- other
tooling does that. You are checking whether a clinician would believe these
records.

Call the inspection tools to get real numbers. Do not guess rates. Then judge
them against what you know about real practice:
  - Is this prevalence plausible for a primary-care panel?
  - Is this prescribing rate plausible, especially for expensive or
    specialist-initiated drugs?
  - Are there combinations that should never co-occur in one patient?
  - Do demographics line up with the coverage and visit type?

Call flag_finding once per distinct problem. Be specific and quantitative:
give the observed number, say what would be expected in real practice, and
state what it would break if someone screened a cohort on this data. Do not
flag something merely because it is uncommon -- flag it because it is not
credible. If a pattern is plausible, leave it alone.

Finish with a one-paragraph verdict on whether this extract is safe to build
clinical logic on, and what to watch for.
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
        """Patients prescribed more than one distinct agent from the same drug class."""
        con = connect()
        try:
            rows = con.execute("""
                SELECT p.PAT_NAME, m.generic_class, count(DISTINCT m.MEDICATION_ID) AS n,
                       string_agg(DISTINCT m.DISPLAY_NAME, ' + ') AS agents
                FROM v_medication m JOIN patient p USING (PAT_ID)
                GROUP BY 1,2 HAVING count(DISTINCT m.MEDICATION_ID) > 1
                ORDER BY n DESC, 1
            """).fetchall()
        finally:
            con.close()
        return {"duplications": [{"patient": r[0], "drug_class": r[1],
                                  "distinct_agents": r[2], "agents": r[3]} for r in rows]}

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

    def flag_finding(title: str, severity: str, observed: str, expected: str,
                     impact: str, affected: str) -> dict:
        """Record one clinical-plausibility problem.

        Args:
            title: short name, e.g. "Triple anticoagulation".
            severity: "high", "medium" or "low".
            observed: what the data actually shows, with numbers.
            expected: what real-world practice would look like.
            impact: what this breaks for anyone screening cohorts on this data.
            affected: the patients, codes or classes involved.
        """
        f = {"title": title, "severity": severity, "observed": observed,
             "expected": expected, "impact": impact, "affected": affected}
        findings.append(f)
        return {"recorded": True, "count": len(findings)}

    return [get_condition_prevalence, get_prescribing_rates, find_duplicate_therapy,
            get_demographics, check_lab_plausibility, flag_finding]


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
