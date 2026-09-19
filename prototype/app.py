"""Live demo UI: paste a protocol, watch the agent ground it, review the cohort.

    streamlit run prototype/app.py

Reviewer-first layout. The criteria plan and its rationale come BEFORE the
cohort, because approving the plan is the human-in-the-loop gate -- a reviewer
who only sees a patient list cannot catch an over-broad exclusion.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prototype import preflight, theme               # noqa: E402
from prototype.guidelines import (                   # noqa: E402
    DISCLAIMER, GUIDELINES, check_guideline)
from prototype.panel import review_async             # noqa: E402
from prototype.screener import screen_async          # noqa: E402
from prototype.tools import ScreeningSession         # noqa: E402

FINDINGS_ANCHOR = "findings-the-specialists-recorded"


def link_citations(markdown: str, anchor: str = FINDINGS_ANCHOR) -> str:
    """Turn [F03] and [F03, F05] in the supervisor's prose into anchor links.

    The brackets are kept inside the link text (escaped) so the citation still
    reads as [F03] rather than losing its brackets to markdown link syntax.
    """
    def repl(match: re.Match) -> str:
        ids = re.findall(r"F\d+", match.group(0))
        return " ".join(f"[\\[{i}\\]](#{anchor})" for i in ids)

    return re.sub(r"\[F\d+(?:\s*,\s*F\d+)*\]", repl, markdown)


st.set_page_config(page_title="Eligibility Screening — Qualified Health",
                   page_icon=str(theme.ASSETS / "q-mark.png"), layout="wide")
theme.apply()

EXAMPLES = {
    "Hypertension intensification": """Adults aged 18 to 75 with a diagnosis of hypertension who are already on at
least one antihypertensive medication, whose most recent systolic blood
pressure is 140 or higher.

Exclude patients with chronic kidney disease, defined as an eGFR below 30.
Exclude patients with heart failure.""",
    "Lipid management escalation": """Patients with a lipid disorder on statin therapy whose most recent LDL
cholesterol is still above 100. Exclude anyone already on a PCSK9 inhibitor.""",
    "Diabetes GLP-1 program": """Adults with type 2 diabetes on metformin with a BMI of 30 or above.
Exclude heart failure and anyone already on a GLP-1 receptor agonist.""",
}

theme.title("Eligibility screening", "from a free-text protocol",
            "Gemini 2.5 Pro on Vertex AI · application default credentials · "
            "100-patient synthetic EHR · the model grounds clinical concepts, "
            "deterministic code runs every query")

tab_panel, tab_screen, tab_preflight, tab_data, tab_guides = st.tabs(
    ["Panel review", "Screen a protocol", "Pre-flight data audit",
     "What the model may select", "Guidelines used"])

# ------------------------------------------------------------- panel review
with tab_panel:
    st.subheader("Who needs attention this week?")
    st.caption("A supervisor agent decides which specialists to consult and in what "
               "order — nothing scripts its path. It is told to check data integrity "
               "early, and to degrade rather than refuse when records cannot be trusted.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**data_integrity**  \nwhich records can't be trusted")
    c2.markdown("**guideline_concordance**  \nwho is missing recommended therapy")
    c3.markdown("**followup**  \nwhat was started and never finished")

    goal = st.text_input(
        "Goal for the supervisor",
        "Who on this panel needs my attention this week? I can review about a dozen.")
    if st.button("Run panel review", type="primary", key="run_panel"):
        with st.spinner("Supervisor consulting specialists… (3–5 min)"):
            report, ptrace, pfindings = asyncio.run(review_async(goal, verbose=False))
        st.session_state["panel"] = (report, ptrace, pfindings)

    if "panel" in st.session_state:
        report, ptrace, pfindings = st.session_state["panel"]

        # Who actually did the work. Shown because a run once claimed a
        # specialist had been "silent" while using three of its findings --
        # the agent's self-report is not evidence, the trace is.
        counts: dict[str, dict] = {}
        for t in ptrace:
            counts.setdefault(t["agent"], {"calls": 0, "findings": 0})["calls"] += 1
        for f in pfindings:
            counts.setdefault(f["agent"], {"calls": 0, "findings": 0})["findings"] += 1
        if counts:
            cols = st.columns(len(counts))
            for col, (agent, c) in zip(cols, counts.items()):
                col.metric(agent, c["calls"], f"{c['findings']} findings",
                           delta_color="off", help="tool calls made by this agent")

        st.markdown(link_citations(report))

        if pfindings:
            st.subheader("Findings the specialists recorded",
                         anchor=FINDINGS_ANCHOR)
            st.caption("The supervisor's report cites these by id. Rendered from the "
                       "structured store, not from its prose — numbers in a narrative "
                       "drift, and in one run they did.")
            order = {"high": 0, "medium": 1, "low": 2}
            rows = [{
                "id": f.get("finding_id", ""),
                "severity": f.get("severity", ""),
                "agent": f.get("agent", ""),
                "finding": f.get("headline", ""),
                # Names are recorded "Last, First", so a comma-joined list reads as
                # twice as many people. Semicolons keep each patient distinct.
                "patients": "; ".join(f.get("patients") or []) or "(panel-level)",
                "evidence": f.get("evidence", ""),
                "action": f.get("recommended_action", ""),
            } for f in sorted(pfindings, key=lambda x: order.get(x.get("severity"), 3))]
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True, height=340)

        with st.expander("Delegation trace — which specialist did what, in order"):
            for i, t in enumerate(ptrace, 1):
                st.code(f"{i:>2}. [{t['agent']}] {t['tool']}"
                        f"({json.dumps(t['args'])[:120]})", language=None)


# ---------------------------------------------------------------- pre-flight
with tab_preflight:
    st.subheader("Clinical plausibility of the extract")
    st.caption("Run before trusting any cohort. Schema validation catches type errors; "
               "this catches records no clinician would believe.")
    findings, verdict = preflight.cached()
    if not findings:
        st.info("No cached audit. Run `python -m prototype.preflight` to generate one.")
    else:
        order = {"high": 0, "medium": 1, "low": 2}
        for f in sorted(findings, key=lambda x: order.get(x["severity"], 3)):
            icon = {"high": "🔴", "medium": "🟠"}.get(f["severity"], "🟡")
            with st.expander(f"{icon}  {f['title']}", expanded=f["severity"] == "high"):
                st.markdown(f"**Observed** {f['observed']}")
                st.markdown(f"**Expected in practice** {f['expected']}")
                st.markdown(f"**Impact on screening** {f['impact']}")
                st.caption(f"Affected: {f['affected']}")
        if verdict:
            st.markdown("**Verdict**")
            st.write(verdict)

    st.divider()
    st.markdown("**Deterministic physiologic-range check** — independent of the model")
    sess = ScreeningSession()
    bad = sess.implausible_patients()
    st.metric("Patients holding an impossible lab value", f"{len(bad)} of 100")
    if bad:
        st.dataframe(pd.DataFrame(
            [{"patient": p, "impossible values": "; ".join(v)} for p, v in bad.items()]),
            width='stretch', hide_index=True)

# ------------------------------------------------------------------ vocabulary
with tab_data:
    st.subheader("The vocabulary the agent selects from")
    st.caption("The model cannot name a code outside these lists. Registration rejects "
               "anything absent from the dataset, so a hallucinated code cannot reach SQL.")
    v = ScreeningSession().vocab
    c1, c2, c3 = st.columns(3)
    c1.metric("Diagnoses", len(v.conditions))
    c2.metric("Drug classes", len(v.med_classes))
    c3.metric("Lab analytes", len(v.analytes))
    st.dataframe(pd.DataFrame(v.conditions), width='stretch', hide_index=True, height=260)
    st.dataframe(pd.DataFrame([{**m, "agents": ", ".join(m["agents"])} for m in v.med_classes]),
                 width='stretch', hide_index=True, height=260)

# --------------------------------------------------------------------- screen
with tab_screen:
    pick = st.selectbox("Start from an example, or write your own", list(EXAMPLES))
    protocol = st.text_area("Protocol", EXAMPLES[pick], height=150)
    go = st.button("Screen the panel", type="primary")

    if go and protocol.strip():
        with st.spinner("Grounding concepts and screening…"):
            results, final, trace = asyncio.run(screen_async(protocol, verbose=False))
        st.session_state["r"] = (results, final, trace)

    if "r" in st.session_state:
        results, final, trace = st.session_state["r"]

        # The agent can finish without registering anything -- an unparseable
        # protocol, or every concept rejected as absent from the dataset. Say so
        # plainly instead of crashing on a missing key.
        if "counts" not in results:
            st.error(results.get("error", "The agent registered no criteria."))
            st.caption("Nothing was screened. The trace below shows what it tried.")
            with st.expander("Agent trace", expanded=True):
                for i, c in enumerate(trace, 1):
                    st.code(f"{i:>2}. {c['tool']}({json.dumps(c['args'])[:160]})", language=None)
            if final:
                st.write(final)
            st.stop()

        counts = results["counts"]

        # ---- warnings first: they are the reason a human is in the loop
        if results["warnings"]:
            st.subheader("Review before acting")
            for w in results["warnings"]:
                (st.error if w["severity"] == "high" else st.warning)(w["message"])

        # ---- the plan, before the cohort
        st.subheader("How the agent read the protocol")
        st.caption("Approve this before the cohort. An over-broad exclusion is invisible "
                   "in a patient list but obvious here.")
        rows = []
        # inclusions first, then exclusions, each in id order -- registration
        # order is whatever the agent happened to do and reads as noise
        ordered = sorted(results["criteria"].items(),
                         key=lambda kv: (kv[1]["polarity"] != "include", kv[0]))
        for cid, c in ordered:
            imp = results["impact"][cid]
            rng = ""
            if c["min_value"] is not None or c["max_value"] is not None:
                rng = f"{c['min_value'] if c['min_value'] is not None else '−∞'} … " \
                      f"{c['max_value'] if c['max_value'] is not None else '∞'}"
            rows.append({
                "": cid, "from the protocol": c["source_text"], "in/out": c["polarity"],
                "type": c["kind"], "bound to": ", ".join(c["codes"]) or c["field_name"],
                "range": rng, "met": imp["met"], "no data": imp["unknown"],
                "why these codes": c["rationale"]})
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        # ---- cohort
        st.subheader("Cohort")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Eligible", counts["eligible"])
        k2.metric("Needs review", counts["needs_review"], help="Missing data, not a failure")
        k3.metric("Excluded", counts["excluded"])
        k4.metric("With data-quality flags", counts.get("with_data_quality_flags", 0))

        def table(group: str) -> pd.DataFrame:
            out = []
            for p in results[group]:
                row = {"patient": p["name"], "age": p["age"]}
                if p["data_quality_flags"]:
                    row["⚠ data quality"] = "; ".join(p["data_quality_flags"])
                for cid, d in p["criteria"].items():
                    row[cid] = ("✓" if d["passes"] else "✗") + \
                               ("?" if d["status"] == "unknown" else "")
                    if d["evidence"]:
                        row[f"{cid} evidence"] = d["evidence"]
                out.append(row)
            return pd.DataFrame(out)

        t1, t2, t3 = st.tabs([f"Eligible ({counts['eligible']})",
                              f"Needs review ({counts['needs_review']})",
                              f"Excluded ({counts['excluded']})"])
        with t1:
            st.dataframe(table("eligible"), width='stretch', hide_index=True)
        with t2:
            st.caption("These patients are not ineligible. A criterion could not be "
                       "evaluated because the data is missing. Missing is not a pass.")
            st.dataframe(table("needs_review"), width='stretch', hide_index=True)
        with t3:
            st.dataframe(table("excluded"), width='stretch', hide_index=True)

        with st.expander("Agent trace — every tool call, in order"):
            for i, c in enumerate(trace, 1):
                st.code(f"{i:>2}. {c['tool']}({json.dumps(c['args'])[:160]})", language=None)
        with st.expander("Agent's closing summary"):
            st.write(final)


# ------------------------------------------------------------ guidelines used
@st.cache_data(show_spinner=False)
def _guideline_coverage() -> dict:
    """Deterministic population / concordance counts per guideline."""
    return {g["id"]: check_guideline(g["id"]) for g in GUIDELINES}


with tab_guides:
    st.subheader("What guideline_concordance checks against")
    st.warning(f"**{DISCLAIMER}**")
    st.caption(
        "The division of labour matters. This pack supplies **what is recommended**. "
        "Deterministic SQL in `guidelines.py` supplies **who the patient is** — who "
        "falls in the population and what they are actually prescribed. The model "
        "supplies the judgment neither encodes: whether a recommendation genuinely "
        "applies, whether an apparent gap has a defensible reason, and whether it is "
        "worth a clinician's scarce attention. Nothing here is a lookup table for care."
    )

    cov = _guideline_coverage()
    st.dataframe(pd.DataFrame([{
        "id": g["id"],
        "guideline": g["title"],
        "population": g["population"],
        "in panel": cov[g["id"]].get("in_population", 0),
        "concordant": cov[g["id"]].get("concordant_count", 0),
        "gaps": cov[g["id"]].get("gap_count", 0),
        "source": g["source"],
    } for g in GUIDELINES]), width='stretch', hide_index=True)

    st.markdown("#### The recommendations in full")
    for g in GUIDELINES:
        c = cov[g["id"]]
        gaps, pop = c.get("gap_count", 0), c.get("in_population", 0)
        with st.expander(f"{g['id']} · {g['title']}  —  {gaps} of {pop} in population"):
            st.markdown(f"**Recommendation**  {g['recommendation']}")
            st.markdown(f"**Population**  {g['population']}")
            if g.get("caveat"):
                st.info(f"**Caveat**  {g['caveat']}")
            matched = ", ".join(g.get("icd10_any", [])) or \
                      ", ".join(g.get("requires_classes", []))
            expected = ", ".join(g.get("expected_classes", [])) or \
                       g.get("expected_analyte", "")
            st.markdown(f"**Identified by**  `{matched}`  ·  "
                        f"**Satisfied by**  `{expected}`")
            st.caption(f"Source: {g['source']}")
            if c.get("gaps"):
                st.markdown("**Patients the deterministic check flags as gaps** — "
                            "these are candidates for the model to judge, not "
                            "conclusions:")
                st.dataframe(pd.DataFrame([{
                    "patient": x["name"], "age": x["age"],
                    "full regimen": "; ".join(x["full_regimen"]) or "(no medications)",
                } for x in c["gaps"]]), width='stretch', hide_index=True)

    st.divider()
    st.caption(
        "**Why so few, and why these.** Eight recommendations covering the "
        "conditions this panel actually has. A real deployment would carry a "
        "maintained guideline library with versioning and an owner; the point here "
        "is the mechanism, not the coverage. Note that G5 cannot be fully evaluated "
        "— CHA₂DS₂-VASc needs prior stroke and vascular disease, and neither is in "
        "this dataset."
    )

theme.footer()
