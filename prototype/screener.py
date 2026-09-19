"""The eligibility screening agent (ADK + Gemini on Vertex, ADC auth).

The agent reads a free-text protocol, searches the dataset's vocabulary to
ground every clinical concept, registers structured criteria, and runs the
cohort. It never writes SQL and never names a code the dataset lacks --
prototype/tools.py rejects anything that is not in the vocabulary.
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

from .tools import ScreeningSession

MODEL = "gemini-2.5-pro"

INSTRUCTION = """\
You screen a 100-patient synthetic EHR for eligibility against a study or care
program protocol given in free text.

Work in three phases.

PHASE 1 - GROUND EVERY CONCEPT.
Before registering anything, call search_diagnoses / search_medications /
search_analytes / search_vitals for each clinical concept in the protocol. You may only use
values these tools return. You may not invent an ICD-10 code, a drug class or
an analyte name; registration will reject anything the dataset lacks.

PHASE 2 - REGISTER CRITERIA with define_criterion, one call per criterion.
  - Expand concepts completely. This dataset splits conditions across several
    near-duplicate ICD-10 codes: "diabetes" is E11.9 AND E11.51 AND E11.65, and
    a lipid disorder spans three E78 codes. Registering only the exact-name
    match silently loses most of the cohort. Always include every code that
    belongs to the concept.
  - CONSIDER EVERY CANDIDATE THE SEARCH RETURNED, one by one, and decide in or
    out. A code that names a COMPLICATION or SUBTYPE of the concept is an
    instance of it and belongs in: "Hypertensive Heart Disease" (I11.9) IS
    hypertension and belongs in a hypertension criterion; "T2DM with
    Neuropathy" IS diabetes. A code that merely shares a word but names a
    different condition does not: I11.9 is NOT heart failure. Judge by what the
    condition is, not by string overlap, and account for every candidate in
    your rationale.
  - BE PRECISE ON EXCLUSIONS. Expanding an inclusion too far enrols the wrong
    patients; expanding an EXCLUSION too far denies care to people who qualify,
    which is the worse error. Include a code only if the condition it names is
    genuinely the excluded concept. Do not add a code because its name shares
    words: "Hypertensive Heart Disease" (I11.9) is NOT heart failure, and
    belongs nowhere near a heart-failure exclusion. When unsure, leave it out
    and say so in your rationale.
  - Do the same for drugs: a protocol saying "on a statin" means the Statin
    class; "lipid-lowering therapy" may span Statin, PCSK9i, Fibrate and more.
    Read the agent names the search returns and decide.
  - polarity is "include" for inclusion criteria and "exclude" for exclusions.
    For an exclusion, describe the thing being excluded (concept = heart
    failure) and set polarity="exclude" -- do not invert the codes yourself.
  - min_value / max_value are strings; use "" for an open end. "eGFR above 45"
    is min_value="45", max_value="". "A1c between 7 and 10" is "7" and "10".
  - Put your reasoning for the chosen codes in `rationale`. Say why you
    included each one.

PHASE 3 - call run_screening once, then report:
  - the counts (eligible / needs review / excluded),
  - which criteria drove the most exclusions,
  - and explicitly flag how many patients landed in needs_review because data
    was MISSING rather than because they failed. Missing data is not a pass.

Be concise in your final message. The structured results are rendered
separately; do not list every patient.
"""


def build(session: ScreeningSession) -> LlmAgent:
    """Bind the tools to a session and construct the agent."""

    def search_diagnoses(query: str) -> dict:
        """Search this dataset's ICD-10 diagnosis vocabulary for a clinical concept.

        Args:
            query: a clinical concept, e.g. "type 2 diabetes" or "heart failure".
        Returns:
            matches: list of {icd10, name, patients} actually present in the data.
        """
        return session.search_diagnoses(query)

    def search_medications(query: str) -> dict:
        """Search this dataset's medication classes and agents.

        Args:
            query: a drug, class or therapy, e.g. "metformin" or "anticoagulant".
        Returns:
            matches: list of {drug_class, patients, agents} present in the data.
        """
        return session.search_medications(query)

    def search_analytes(query: str) -> dict:
        """Search this dataset's lab analytes and their reference ranges.

        Args:
            query: a lab test, e.g. "HbA1c" or "kidney function".
        Returns:
            matches: list of {analyte, unit, ref_low, ref_high, n} present in the data.
        """
        return session.search_analytes(query)

    def search_vitals(query: str) -> dict:
        """Search the vitals recorded in this dataset (blood pressure, BMI, weight, etc).

        Args:
            query: a vital sign, e.g. "BMI" or "blood pressure".
        Returns:
            matches: list of {vital, note}. Use the exact `vital` string as field_name.
        """
        return session.search_vitals(query)

    def define_criterion(criterion_id: str, source_text: str, polarity: str, kind: str,
                         codes: list[str], field_name: str, min_value: str,
                         max_value: str, rationale: str) -> dict:
        """Register one eligibility criterion. Rejects values absent from the dataset.

        Args:
            criterion_id: short unique id, e.g. "i1" or "e2".
            source_text: the protocol fragment this came from, verbatim.
            polarity: "include" or "exclude".
            kind: "diagnosis", "medication", "lab", "vital" or "demographic".
            codes: ICD-10 codes (diagnosis) or drug class names (medication);
                   [] for lab, vital and numeric demographic criteria.
            field_name: analyte name, vital column, or "PAT_AGE"/"SEX_NAME"; "" otherwise.
            min_value: numeric lower bound as a string, "" for none.
            max_value: numeric upper bound as a string, "" for none.
            rationale: why these codes/classes represent the concept.
        Returns:
            patients_meeting and patients_unknown, or an error naming rejected values.
        """
        return session.define_criterion(criterion_id, source_text, polarity, kind,
                                        codes, field_name, min_value, max_value, rationale)

    def run_screening() -> dict:
        """Combine all registered criteria into the final cohort. Call once, last.

        Returns:
            counts of eligible / needs_review / excluded, and the criteria used.
        """
        r = session.run_screening()
        return {"counts": r.get("counts"), "criteria": list(r.get("criteria", {}))}

    return LlmAgent(
        name="eligibility_screener",
        model=MODEL,
        description="Screens an EHR cohort against a free-text protocol.",
        instruction=INSTRUCTION,
        tools=[search_diagnoses, search_medications, search_analytes, search_vitals,
               define_criterion, run_screening],
    )


async def screen_async(protocol: str, verbose: bool = True) -> tuple[dict, str, list]:
    """Run the agent over a protocol. Returns (results, final_message, trace)."""
    session = ScreeningSession()
    runner = InMemoryRunner(agent=build(session), app_name="screening")
    s = await runner.session_service.create_session(app_name="screening", user_id="demo")

    trace, final = [], ""
    async for ev in runner.run_async(
        user_id="demo", session_id=s.id,
        new_message=types.Content(role="user", parts=[types.Part(text=protocol)])
    ):
        if ev.content and ev.content.parts:
            for p in ev.content.parts:
                if getattr(p, "function_call", None):
                    call = {"tool": p.function_call.name, "args": dict(p.function_call.args)}
                    trace.append(call)
                    if verbose:
                        a = call["args"]
                        detail = a.get("query") or a.get("source_text") or ""
                        print(f"  -> {call['tool']}({str(detail)[:60]})")
        if ev.is_final_response() and ev.content:
            final = "".join(p.text for p in ev.content.parts if getattr(p, "text", None))

    return session.run_screening(), final, trace


def screen(protocol: str, verbose: bool = True):
    return asyncio.run(screen_async(protocol, verbose))
