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
from prototype.brief import write_brief_async        # noqa: E402
from prototype.fixtures import PLANTED_CONFLICTS, planted_note  # noqa: E402
from prototype.reconcile import reconcile_async      # noqa: E402
from prototype.panel import review_async             # noqa: E402
from prototype.screener import screen_async          # noqa: E402
from prototype.tools import ScreeningSession, connect  # noqa: E402

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


def link_patients(markdown: str, names: list[str]) -> str:
    """Turn patient names in the prose into links to their brief.

    One pass over an alternation of all the names, longest first, so a name is
    never re-scanned inside a URL this function just inserted and a short name
    cannot match inside a longer one.
    """
    if not names:
        return markdown
    # Idempotent: skip a name already used as link TEXT (preceded by "[") or
    # already sitting inside a link URL (preceded by "="). Without the "=" case
    # a second application nests the link inside its own href.
    pattern = re.compile(
        r"(?<![\[=])(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
        + r")(?!\]\()")
    return pattern.sub(lambda m: f"[{m.group(0)}](?patient={m.group(0)})", markdown)


st.set_page_config(page_title="Panel Review — Qualified Health",
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

theme.title("Panel review", "who needs attention this week",
            "A supervisor agent and three specialists over a 100-patient synthetic "
            "EHR · Gemini 2.5 Pro on Vertex AI, application default credentials · "
            "no model writes SQL, and no model computes what code can compute")

VIEWS = ["Panel review", "Patient brief", "Pre-flight data audit",
         "What the model may select", "Guidelines used"]

# Protocol screening is not part of the product being presented -- the panel
# manager's worklist is. The code stays and the view is still reachable at
# ?view=Screen%20a%20protocol if someone asks to see it, but it is off the nav
# so the demo has one story rather than two.
HIDDEN_VIEWS = ["Screen a protocol"]
ALL_VIEWS = VIEWS + HIDDEN_VIEWS

# Navigation is session state rather than st.tabs, because a patient name has to
# be able to send you to another view. st.tabs cannot be switched in code.
# The URL carries it, so a link in a table is a real link.
_qp = st.query_params
if _qp.get("patient"):
    st.session_state["brief_patient"] = _qp["patient"]
    st.session_state["view"] = "Patient brief"
    st.query_params.clear()
elif _qp.get("view") in ALL_VIEWS:
    st.session_state["view"] = _qp["view"]
    st.query_params.clear()

_current = st.session_state.get("view", VIEWS[0])
if _current in HIDDEN_VIEWS:                 # reached by URL, not on the nav
    view = _current
    st.caption(f"Viewing **{_current}** — not part of the presented product; "
               f"[back to the panel review](?view=Panel%20review).")
else:
    view = st.segmented_control("Section", VIEWS, key="view",
                                label_visibility="collapsed", default=_current) \
           or _current or VIEWS[0]


@st.cache_data(show_spinner=False)
def all_patient_names() -> list[str]:
    con = connect()
    try:
        return [r[0] for r in con.execute(
            "SELECT PAT_NAME FROM patient ORDER BY 1").fetchall()]
    finally:
        con.close()


def patient_link_column(df, name_col: str = "patient"):
    """Turn a patient-name column into links that open that patient's brief.

    The name itself is the link: the URL carries the name unencoded and
    LinkColumn's display_text pulls it back out, so the cell reads as the
    patient rather than as a URL.
    """
    out = df.copy()
    out[name_col] = out[name_col].map(lambda n: f"?patient={n}")
    return out, {name_col: st.column_config.LinkColumn(
        name_col, display_text=r"\?patient=(.*)",
        help="open this patient's pre-visit brief")}

# ------------------------------------------------------------- panel review
if view == "Panel review":
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
            report, ptrace, pfindings, pusage = asyncio.run(
                review_async(goal, verbose=False))
        st.session_state["panel"] = (report, ptrace, pfindings, pusage)

    if "panel" in st.session_state:
        report, ptrace, pfindings, pusage = st.session_state["panel"]

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

        all_named = sorted({p for f in pfindings for p in (f.get("patients") or [])})
        st.markdown(link_patients(link_citations(report), all_named))

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

        with st.expander("What this run cost"):
            u1, u2, u3, u4 = st.columns(4)
            u1.metric("Model calls", pusage.get("model_calls", 0))
            u2.metric("Input tokens", f"{pusage.get('input_tokens', 0):,}")
            u3.metric("Output tokens", f"{pusage.get('output_tokens', 0):,}")
            u4.metric("Cost", f"${pusage.get('usd', 0):.2f}",
                      f"{pusage.get('wall_clock_seconds', 0):.0f}s",
                      delta_color="off")
            if pusage.get("per_agent"):
                st.dataframe(pd.DataFrame([
                    {"agent": a, "calls": v["calls"], "input": v["in"],
                     "output": v["out"]} for a, v in pusage["per_agent"].items()]),
                    width='stretch', hide_index=True)
            st.caption("Token counts come from the API's usage_metadata, not an "
                       "estimate. Cost uses the Vertex list price for "
                       "gemini-2.5-pro at the rates in panel.py.")

        with st.expander("Delegation trace — which specialist did what, in order"):
            for i, t in enumerate(ptrace, 1):
                st.code(f"{i:>2}. [{t['agent']}] {t['tool']}"
                        f"({json.dumps(t['args'])[:120]})", language=None)



# ------------------------------------------------------------- patient brief
if view == "Patient brief":
    st.subheader("Pre-visit brief")
    st.caption("Everything on file for one patient. The facts are assembled "
               "deterministically; the model only decides what to raise first and "
               "writes it. Reached by clicking a name anywhere in the app, or pick "
               "one here.")

    current = st.session_state.get("brief_patient")
    who = st.selectbox("Patient", all_patient_names(),
                       index=(all_patient_names().index(current)
                              if current in all_patient_names() else 0),
                       key="brief_patient_select")
    if who != current:
        st.session_state["brief_patient"] = who

    findings_ctx = st.session_state.get("panel", (None, None, [], {}))[2] or []
    cache = st.session_state.setdefault("briefs", {})
    if who not in cache:
        with st.spinner(f"Assembling the brief for {who}…"):
            cache[who] = asyncio.run(write_brief_async(who, findings_ctx))
    narrative, pack = cache[who]

    if "error" in pack:
        st.error(pack["error"])
    else:
        p_, v_ = pack["patient"], pack["visit"]
        st.markdown(f"### {p_['name']}  ·  {p_['age']}  ·  {p_['sex']}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Next AWV (derived)", v_["status"], v_["detail"],
                  delta_color="off", help=v_["basis"])
        k2.metric("Conditions", len(pack["conditions"]))
        k3.metric("Abnormal labs", len(pack["labs"]["abnormal"]))
        k4.metric("Open orders", len(pack["outstanding_orders"]))

        for flag in pack["data_quality"]["flags"]:
            st.error(f"**Do not trust this record:** {flag}")

        left, right = st.columns([3, 2])
        with left:
            st.markdown(narrative)
            if pack["panel_findings"]:
                st.caption("**From the last panel review**")
                for f in pack["panel_findings"]:
                    st.write(f"`{f['finding_id']}` [{f['severity']}] {f['headline']}")
        with right:
            st.caption("**The facts behind it** — every number opposite comes from "
                       "here, not from the model.")
            with st.expander("Conditions", expanded=True):
                for d in pack["conditions"]:
                    st.write(f"`{d['icd10']}`  {d['name']}")
            with st.expander("Medications"):
                st.caption(pack["data_quality"]["medication_caveat"])
                for cls, agents in pack["medications"]["by_class"].items():
                    st.write(f"**{cls}** — " + ", ".join(
                        f"{a['agent']} (from {a['started']})" for a in agents))
            with st.expander("Abnormal labs"):
                if pack["labs"]["abnormal"]:
                    st.dataframe(pd.DataFrame(pack["labs"]["abnormal"]),
                                 width='stretch', hide_index=True)
                else:
                    st.write("None.")
            with st.expander("Open orders"):
                if pack["outstanding_orders"]:
                    st.dataframe(pd.DataFrame(pack["outstanding_orders"]),
                                 width='stretch', hide_index=True)
                else:
                    st.write("None.")
            with st.expander("Care gaps"):
                for g in pack["care_gaps"]:
                    st.write(f"**{g['id']}** {g['title']}")
                    st.caption(g["source"])
                if not pack["care_gaps"]:
                    st.write("None against the eight-guideline pack.")
        # ---- reconcile the brief against what the clinician actually wrote ----
        st.divider()
        st.markdown("#### Checked against the clinician's note")
        st.caption("The brief is assembled from structured tables; the note is what a "
                   "human recorded at the visit. Where they disagree, the brief is the "
                   "one that is wrong. This is a control, not a discovery step.")

        rc1, rc2 = st.columns([1, 1])
        run_real = rc1.button("Reconcile against the note", key=f"rec_{who}")
        can_plant = who in PLANTED_CONFLICTS
        run_plant = rc2.button("Demonstrate with a planted conflict",
                               key=f"plant_{who}", disabled=not can_plant,
                               help=None if can_plant else
                               "No fixture written for this patient")

        rec_cache = st.session_state.setdefault("reconciliations", {})
        if run_real:
            with st.spinner("Comparing the brief with the note…"):
                rec_cache[(who, "real")] = asyncio.run(
                    reconcile_async(pack, narrative))
        if run_plant:
            with st.spinner("Comparing the brief with a fabricated note…"):
                rec_cache[(who, "planted")] = asyncio.run(
                    reconcile_async(pack, narrative, planted_note(who)))

        for kind in ("real", "planted"):
            r = rec_cache.get((who, kind))
            if not r:
                continue
            if kind == "planted":
                st.warning(f"**Fabricated note — not data.** Planted: "
                           f"{PLANTED_CONFLICTS[who][0]}")
            if r.get("error"):
                st.error(r["error"])
            elif r["conflicts"]:
                for c in r["conflicts"]:
                    st.error(f"**{c['severity'].upper()} — {c['field']}**  \n"
                             f"Brief says: {c['brief_says']}  \n"
                             f"Note says: {c['note_says']}  \n"
                             f"_{c['why_it_matters']}_")
            else:
                st.success(f"No conflicts across {r['note_count']} note(s). "
                           f"Compared: {', '.join(r.get('checked', []))}.")
                st.caption("Expected on this extract: the notes are generated from the "
                           "same tables the brief is built from — 153/153 field "
                           "agreement — so there is nothing to disagree about. Use the "
                           "button on the right to see the control fire.")

        st.caption("Prompts a conversation; does not replace chart review. No drug "
                   "or dose is recommended anywhere in this brief.")


# ---------------------------------------------------------------- pre-flight
if view == "Pre-flight data audit":
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
if view == "What the model may select":
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
if view == "Screen a protocol":
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
        def _linked(group: str):
            df = table(group)
            if df.empty:
                return st.write("None.")
            linked, cfg = patient_link_column(df)
            return st.dataframe(linked, column_config=cfg, width='stretch',
                                hide_index=True)

        with t1:
            st.caption("Click a patient to open their pre-visit brief.")
            _linked("eligible")
        with t2:
            st.caption("These patients are not ineligible. A criterion could not be "
                       "evaluated because the data is missing. Missing is not a pass.")
            _linked("needs_review")
        with t3:
            _linked("excluded")

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


if view == "Guidelines used":
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
