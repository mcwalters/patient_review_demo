"""Live demo UI: paste a protocol, watch the agent ground it, review the cohort.

    streamlit run prototype/app.py

Reviewer-first layout. The criteria plan and its rationale come BEFORE the
cohort, because approving the plan is the human-in-the-loop gate -- a reviewer
who only sees a patient list cannot catch an over-broad exclusion.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import sys
from urllib.parse import quote
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prototype import panel_cache, preflight, score, theme  # noqa: E402
from prototype.guidelines import (                   # noqa: E402
    DISCLAIMER, GUIDELINES, check_guideline)
from prototype.brief import write_brief_async        # noqa: E402
from prototype.panel import (                         # noqa: E402
    Findings, review_async, uncited_high_severity)
from prototype.screener import screen_async          # noqa: E402
from prototype.report_md import (                    # noqa: E402
    FINDINGS_ANCHOR, fold_actions, link_citations, link_patients)

# Shared across every Streamlit session in this process -- see the guard below.
_IN_FLIGHT: dict[str, "threading.Thread | None"] = {"worker": None}
from prototype.tools import ScreeningSession, connect  # noqa: E402


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

VIEWS = ["Panel review", "Patient brief", "Priority score",
         "Pre-flight data audit",
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
    out[name_col] = out[name_col].map(lambda n: f"?patient={quote(n)}")
    return out, {name_col: st.column_config.LinkColumn(
        name_col, display_text=r"\?patient=(.*)",
        help="open this patient's pre-visit brief")}

# ------------------------------------------------------------- panel review
if view == "Panel review":
    st.subheader("Who needs attention this week?")
    st.caption("A supervisor agent briefs three specialists and consults them at "
               "once — it writes what to ask each, which is the judgement; it does "
               "not choose whether or in what order, because the answer was always "
               "all three and they cannot read each other. It reads the data-quality "
               "findings before it ranks, and degrades rather than refuses when "
               "records cannot be trusted.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**data_integrity**  \nwhich records can't be trusted")
    c2.markdown("**guideline_concordance**  \nwho is missing recommended therapy")
    c3.markdown("**followup**  \nwhat was started and never finished")

    goal = st.text_input(
        "What should the supervisor focus on?",
        "Anything that needs attention this week.",
        help="This steers what the supervisor emphasises and how it ranks. It "
             "cannot change the structure of the review: the three specialists, "
             "the data-integrity check, the guaranteed findings and the "
             "twelve-patient cap are fixed.")
    st.caption("A steer, not a configuration. The shortlist is capped at twelve "
               "regardless of what you ask for — say so here and it will be "
               "ignored.")
    # A review takes minutes, so it is entirely possible to start a second one
    # on top of the first -- a stray click, or a second browser tab left open.
    # Observed exactly that: two reviews ran concurrently, the completion line
    # read "31 findings" from one thread's store while the cache saved 21 from
    # the other's, and the run cost twice what it should have.
    #
    # The guard is module-level and not in session_state, because session_state
    # is per browser tab: a second tab is a different session and would not see
    # a flag stored there. Streamlit runs every session in one process, so a
    # module global is shared across all of them, which is what this needs.
    _prior = _IN_FLIGHT.get("worker")
    _running = _prior is not None and _prior.is_alive()
    if _running:
        st.info("A panel review is already running — started here or in another "
                "tab. Wait for it to finish rather than starting a second one; "
                "two concurrent runs cost twice as much and race to save.")

    if st.button("Run panel review", type="primary", key="run_panel",
                 disabled=_running) and not _running:
        # The run takes minutes. The callbacks append to these as it goes, so
        # the loop below can render what has actually happened rather than
        # showing a spinner and hoping.
        live_trace: list = []
        live_findings = Findings()
        box: dict = {}

        def _work():
            try:
                box["result"] = asyncio.run(review_async(
                    goal, verbose=False, trace=live_trace, findings=live_findings))
            except Exception as exc:                       # surfaced below
                box["error"] = f"{type(exc).__name__}: {exc}"

        worker = threading.Thread(target=_work, daemon=True)
        worker.start()
        _IN_FLIGHT["worker"] = worker

        LABEL = {"data_integrity": "which records can be trusted",
                 "guideline_concordance": "who is missing recommended therapy",
                 "followup": "what was started and never finished",
                 "panel_review": "supervisor deciding what to consult",
                 "guaranteed": "guaranteed findings"}
        started = time.time()

        # A plain placeholder, not st.status. The loop re-renders once a second,
        # and any expander inside something re-rendered on a timer collapses the
        # moment the user opens it. Everything here stays visible instead; the
        # trace gets a proper expander once the run is over and nothing is
        # redrawing it.
        progress = st.empty()
        while worker.is_alive():
            seen = list(live_trace)
            n_find = len(live_findings.all())
            by_agent: dict[str, int] = {}
            for t in seen:
                by_agent[t["agent"]] = by_agent.get(t["agent"], 0) + 1
            current = seen[-1]["agent"] if seen else "panel_review"
            with progress.container(border=True):
                st.markdown(f"**Checking {LABEL.get(current, current)}**")
                m1, m2, m3 = st.columns(3)
                m1.metric("Elapsed", f"{int(time.time() - started)}s")
                m2.metric("Tool calls", len(seen))
                m3.metric("Findings", n_find)
                if by_agent:
                    st.caption("  ·  ".join(
                        f"{LABEL.get(a, a)}: {n}" for a, n in by_agent.items()))
                for t in seen[-5:]:
                    st.markdown(
                        f"<span style='color:#5C6C80;font-size:0.85rem'>"
                        f"<code>{t['agent']}</code> → {t['tool']}</span>",
                        unsafe_allow_html=True)
            time.sleep(1.0)
        worker.join()

        if "error" in box:
            progress.error(box["error"])
            st.stop()
        progress.success(
            f"Done in {int(time.time() - started)}s · {len(live_trace)} tool calls · "
            f"{len(live_findings.all())} findings")
        st.session_state["panel"] = box["result"]
        # Refused rows never reach the findings list, so they would otherwise
        # vanish silently -- which is the wrong outcome for a rejected name.
        st.session_state["panel_rejected"] = list(live_findings.rejected)
        st.session_state["panel_is_saved_run"] = False
        _IN_FLIGHT["worker"] = None
        panel_cache.save(goal, box["result"], live_findings.rejected)

    # Open on the last saved run rather than a blank screen. A review takes
    # minutes, which is a slice of the session spent watching a
    # spinner, and it is the first thing anyone sees. The live button above
    # still runs it for real and overwrites this.
    if "panel" not in st.session_state:
        saved = panel_cache.load()
        if saved:
            st.session_state["panel"] = panel_cache.as_session_value(saved)
            st.session_state["panel_rejected"] = saved.get("rejected", [])
            st.session_state["panel_is_saved_run"] = True
            st.session_state["panel_saved_meta"] = saved

    if "panel" in st.session_state:
        report, ptrace, pfindings, pusage = st.session_state["panel"]

        if st.session_state.get("panel_is_saved_run"):
            meta = st.session_state.get("panel_saved_meta", {})
            st.info(
                f"**Showing a saved run from {panel_cache.age_phrase(meta)}** — "
                f"real output from a real run, not a fixture: "
                f"{len(pfindings)} findings, {len(ptrace)} tool calls, "
                f"{pusage.get('wall_clock_seconds', '?')}s, "
                f"${pusage.get('usd', 0):.2f}. Press **Run panel review** to "
                f"watch the agents do it live — about two and a half minutes, and "
                f"the result will differ, which is the point of the eval.")

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

        # The shortlist is capped at twelve out of a hundred and the supervisor
        # chooses what to leave off. Its own account of the omissions is a
        # self-report, so the report is checked against the store instead. A
        # high-severity finding that went unmentioned is raised ABOVE the
        # narrative -- the finding was always in the table below, but nothing
        # distinguished "deliberately deprioritised" from "silently dropped".
        missed = uncited_high_severity(report, pfindings)
        if missed:
            st.warning(
                f"**{len(missed)} finding"
                f"{'s' if len(missed) > 1 else ''} rated high by the store, not "
                f"cited individually in the report.** The shortlist is twelve "
                f"patients out of a hundred, so something has to come off it. "
                f"This is where the ranking was disagreed with — not a claim that "
                f"either side is wrong.")
            st.caption(
                "Known limitation: the floor rates a whole category high without "
                "grading inside it, so a six-month-overdue triglycerides arrives "
                "at the same severity as an eleven-month-overdue kidney screen in "
                "diabetes. Expect stale orders here; the supervisor is usually "
                "right to rank them below a hypertensive crisis. See KNOWN_ISSUES.md.")
            for f in missed:
                who = "; ".join(f.get("patients") or []) or "panel-level"
                st.markdown(f"- **{f['finding_id']}** · {f.get('headline','')} — {who}")

        rejected = st.session_state.get("panel_rejected") or []
        if rejected:
            st.warning(
                f"**{len(rejected)} finding"
                f"{'s' if len(rejected) > 1 else ''} refused: unrecognised patient "
                f"name.** The store rejects a name that is not in the patient table, "
                f"because the UI turns names into links to that person's brief.")
            for r in rejected:
                st.markdown(f"- `{r['agent']}` · {r['headline']} — "
                            f"unknown: {', '.join(r['unknown'])}")

        all_named = sorted({p for f in pfindings for p in (f.get("patients") or [])})
        st.markdown(link_patients(link_citations(fold_actions(report)), all_named))

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
    # Seeded from disk, so a brief written once is instant for every later
    # session and survives a server restart. session_state on its own made the
    # first click on each patient cost the model call again after every restart
    # and in every new tab -- about forty seconds warm, longer on a cold
    # server, and it is the first thing anyone clicks from the shortlist.
    cache = st.session_state.setdefault("briefs", {})
    if not cache:
        cache.update(panel_cache.load_briefs())
    if who not in cache:
        with st.spinner(f"Assembling the brief for {who}… under a minute, "
                        f"then it is saved and instant"):
            narrative, pack = asyncio.run(write_brief_async(who, findings_ctx))
        cache[who] = (narrative, pack)
        panel_cache.save_brief(who, narrative, pack)
    narrative, pack = cache[who]

    if "error" in pack:
        st.error(pack["error"])
    else:
        p_, v_ = pack["patient"], pack["visit"]
        st.markdown(f"### {p_['name']}  ·  {p_['age']}  ·  {p_['sex']}")
        lv = v_.get("last_visit") or {}
        if lv:
            st.caption(f"Last seen **{lv['date']}** in {lv['department']} by "
                       f"**{lv['provider']}** — {lv['provider_specialty']}, "
                       f"{lv['provider_type']}  ·  {lv['insurance']}")
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
                    st.write("None against the four-guideline pack.")
        # Reconciliation against the note now runs inside write_brief and appears
        # in the prose. A control behind a button is a control nobody presses.
        rec = pack.get("note_reconciliation") or {}
        if rec.get("conflicts"):
            st.caption(f"**Checked against {rec.get('note_count', 0)} clinician "
                       f"note(s)** — {len(rec['conflicts'])} conflict(s), detailed above.")
        elif rec.get("checked"):
            st.caption(f"**Checked against {rec.get('note_count', 0)} clinician "
                       f"note(s)** — no conflicts. Compared: "
                       f"{', '.join(rec['checked'])}.")

        st.caption("Prompts a conversation; does not replace chart review. No drug "
                   "or dose is recommended anywhere in this brief.")


# ------------------------------------------------------------ priority score
if view == "Priority score":
    st.subheader("A transparent score, so the ranking can be argued with")
    st.caption("The panel review ranks patients and nothing checks that the order is "
               "right. This does not make it correct — it makes it legible. A model "
               "wrote the weights once, offline; code applies them, so the score is "
               "identical every run and a weight change is a reviewable diff.")

    w = score.load_weights()
    st.info(f"**How it was weighted.** {w['rationale']}")

    scores = score.score_panel(w)
    rows = [{"patient": s_.patient, "score": s_.total,
             **{k: s_.by_component().get(k, 0) for k in
                ("burden", "instability", "neglect")}} for s_ in scores]
    c1, c2, c3 = st.columns(3)
    c1.metric("Patients scored", len(scores))
    c2.metric("Distinct scores", len({s_.total for s_ in scores}),
              help="Burden alone would sort this panel into about four buckets")
    c3.metric("Range", f"{scores[-1].total:g} – {scores[0].total:g}")

    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True, height=320)

    st.markdown("#### Every score reads as its reasons")
    st.caption("If a clinician cannot read down this list and disagree line by line, "
               "the score is not doing its job.")
    pick = st.selectbox("Patient", [s_.patient for s_ in scores], index=0,
                        key="score_patient")
    chosen = next(s_ for s_ in scores if s_.patient == pick)
    st.code(chosen.explain(), language=None)

    with st.expander("The weights themselves — the thing a clinician edits"):
        st.caption("prototype/score_weights.json. Not a validated instrument: "
                   "Charlson and Elixhauser are published and validated, and a real "
                   "deployment should anchor the burden component on one of them.")
        st.json(w)

    st.divider()
    st.markdown("**Does it agree with the agents?** — two independent mechanisms")
    if "panel" in st.session_state:
        report = st.session_state["panel"][0]
        top = [s_.patient for s_ in scores[:12]]
        agreed = [p for p in top if p in report]
        st.metric("Scorer's top 12 also named by the panel review", f"{len(agreed)} of 12")
        st.caption("Neither is ground truth, so this is convergent validity and "
                   "nothing more. The disagreements are the interesting ones: "
                   + ", ".join(p for p in top if p not in report))
    else:
        st.caption("Run or load a panel review to compare.")

# ---------------------------------------------------------------- pre-flight
if view == "Pre-flight data audit":
    st.subheader("Clinical plausibility of the extract")
    st.caption("Run before trusting any cohort. Schema validation catches type errors; "
               "this catches records no clinician would believe.")
    findings, verdict = preflight.cached()
    if not findings:
        st.info("No cached audit. Run `python -m prototype.preflight` to generate one.")
    else:
        # Separated because the two groups carry different authority. A value
        # that cannot exist is a defect whatever this panel is. A rate that is
        # high for the general population is only a defect if you assume this
        # is the general population -- and nothing here says it is. An extract
        # could be a specialty clinic or a deliberately enriched cohort, in
        # which case the "objection" is that the sick panel is sick.
        BLURB = {
            "impossible": ("Defects — true whatever this panel is",
                           "No patient could hold these values. Nothing about the "
                           "population makes them credible."),
            "internally inconsistent": ("Defects — the record contradicts itself",
                                        "Two parts of the same record disagree. "
                                        "Independent of what population this is."),
            "population-dependent": ("Questions for whoever supplied the data",
                                     "Surprising only against a general primary-care "
                                     "panel. Each names the population that would make "
                                     "it ordinary — these are questions, not errors."),
        }
        order = {"high": 0, "medium": 1, "low": 2}
        for kind in preflight.KINDS:
            group = [f for f in findings if f.get("kind", "population-dependent") == kind]
            if not group:
                continue
            title, blurb = BLURB[kind]
            st.markdown(f"#### {title}")
            st.caption(blurb)
            for f in sorted(group, key=lambda x: order.get(x["severity"], 3)):
                icon = ("❓" if kind == "population-dependent"
                        else {"high": "🔴", "medium": "🟠"}.get(f["severity"], "🟡"))
                with st.expander(f"{icon}  {f['title']}",
                                 expanded=kind != "population-dependent"
                                 and f["severity"] == "high"):
                    st.markdown(f"**Observed** {f['observed']}")
                    st.markdown(f"**Expected** {f['expected']}")
                    if f.get("plausible_if"):
                        st.info(f"**Would be unremarkable in** {f['plausible_if']}")
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
        "**Why so few, and why these.** Four recommendations, each chosen because "
        "the population is small, the recommendation is unambiguous, and the "
        "absence is worth a clinician's time. Four others were cut for failing "
        "that test: anticoagulation in AF (CHA₂DS₂-VASc needs prior stroke and "
        "vascular disease, neither of which is in this extract), metformin "
        "first-line (21 gaps in a population of 28, with the contraindications "
        "that would explain them unrecorded), statin in hyperlipidaemia (31 of 50 "
        "— a population-health campaign, not a weekly worklist), and TSH "
        "monitoring (real but minor). A real deployment carries a maintained "
        "library with a version and an owner; the point here is the mechanism, "
        "not the coverage."
    )

theme.footer()
