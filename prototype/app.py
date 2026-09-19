"""Live demo UI: paste a protocol, watch the agent ground it, review the cohort.

    streamlit run prototype/app.py

Reviewer-first layout. The criteria plan and its rationale come BEFORE the
cohort, because approving the plan is the human-in-the-loop gate -- a reviewer
who only sees a patient list cannot catch an over-broad exclusion.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prototype import preflight                      # noqa: E402
from prototype.screener import screen_async          # noqa: E402
from prototype.tools import ScreeningSession         # noqa: E402

st.set_page_config(page_title="Eligibility Screening", layout="wide")

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

st.title("Eligibility screening from a free-text protocol")
st.caption("Gemini 2.5 Pro on Vertex AI · ADC auth · 100-patient synthetic EHR · "
           "the model grounds concepts, deterministic code runs every query")

tab_screen, tab_preflight, tab_data = st.tabs(
    ["Screen a protocol", "Pre-flight data audit", "What the model may select"])

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
            use_container_width=True, hide_index=True)

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
    st.dataframe(pd.DataFrame(v.conditions), use_container_width=True, hide_index=True, height=260)
    st.dataframe(pd.DataFrame([{**m, "agents": ", ".join(m["agents"])} for m in v.med_classes]),
                 use_container_width=True, hide_index=True, height=260)

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
        for cid, c in results["criteria"].items():
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
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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
            st.dataframe(table("eligible"), use_container_width=True, hide_index=True)
        with t2:
            st.caption("These patients are not ineligible. A criterion could not be "
                       "evaluated because the data is missing. Missing is not a pass.")
            st.dataframe(table("needs_review"), use_container_width=True, hide_index=True)
        with t3:
            st.dataframe(table("excluded"), use_container_width=True, hide_index=True)

        with st.expander("Agent trace — every tool call, in order"):
            for i, c in enumerate(trace, 1):
                st.code(f"{i:>2}. {c['tool']}({json.dumps(c['args'])[:160]})", language=None)
        with st.expander("Agent's closing summary"):
            st.write(final)
