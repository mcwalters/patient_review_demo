"""Tier 1: the facts the agents reason over are provably correct.

Every assertion here re-derives an expected value with independent SQL and
compares it to what the tool returns. No model is involved and nothing costs
money, so this runs on every change.

This is the regression net around the rule the whole build rests on: no model
computes what code can compute. If the code computes it wrong, the rule buys
nothing.

    ./.venv/bin/python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prototype.guidelines import GUIDELINES, check_guideline          # noqa: E402
from prototype.panel import (                                         # noqa: E402
    _pending_orders, blood_pressure_staging, find_patients,
    medication_timeline, patient_snapshot)
from prototype.tools import AS_OF, PHYSIOLOGIC_LIMITS, ScreeningSession  # noqa: E402

DB = Path(__file__).resolve().parent.parent / "ehr.duckdb"


@pytest.fixture(scope="module")
def db():
    con = duckdb.connect(str(DB), read_only=True)
    yield con
    con.close()


def q1(con, sql, params=None):
    return con.execute(sql, params or []).fetchone()[0]


# --------------------------------------------------------------- pending orders
def test_pending_total_matches_sql(db):
    assert _pending_orders()["total"] == q1(
        db, "SELECT count(*) FROM v_lab_order WHERE is_pending")


def test_pending_buckets_partition_the_whole_set():
    r = _pending_orders()
    c = r["counts"]
    assert c["actionable"] + c["superseded"] + c["no_live_indication"] == r["total"]


def test_superseded_really_has_a_later_result(db):
    """Every order in the superseded bucket must have a later result for that analyte."""
    for o in _pending_orders()["superseded"]:
        n = q1(db, "SELECT count(*) FROM v_lab_result WHERE PAT_ID = ? "
                   "AND COMPONENT_NAME = ? AND RESULT_DATE > ?",
               [o["pat_id"], o["test"], o["ordered"]])
        assert n > 0, f"{o['patient']} {o['test']} marked superseded with no later result"


def test_actionable_orders_are_not_superseded(db):
    for o in _pending_orders()["actionable"]:
        n = q1(db, "SELECT count(*) FROM v_lab_result WHERE PAT_ID = ? "
                   "AND COMPONENT_NAME = ? AND RESULT_DATE > ?",
               [o["pat_id"], o["test"], o["ordered"]])
        assert n == 0, f"{o['patient']} {o['test']} is actionable but was superseded"


def test_months_open_arithmetic(db):
    o = _pending_orders()["actionable"][0]
    assert o["months_open"] == q1(
        db, "SELECT date_diff('month', ?::DATE, ?::DATE)", [o["ordered"], AS_OF])


# ----------------------------------------------------------- drug monitoring
def test_inr_flags_patients_not_on_warfarin(db):
    """The fact the agent kept getting backwards."""
    checked = [o for o in (_pending_orders()["actionable"]
                           + _pending_orders()["superseded"]
                           + _pending_orders()["no_live_indication"])
               if o["test"] == "INR / PT" and "drug_monitoring_check" in o]
    assert checked, "no INR order carried a drug-monitoring check"
    for o in checked:
        on_vka = q1(db, "SELECT count(*) FROM v_medication WHERE PAT_ID = ? "
                        "AND generic_class = 'VKA'", [o["pat_id"]]) > 0
        assert o["drug_monitoring_check"]["patient_is_on_it"] == on_vka
        assert o["drug_monitoring_check"]["test_monitors"] == "VKA"


def test_digoxin_level_without_digoxin(db):
    orders = [o for o in _pending_orders()["actionable"]
              + _pending_orders()["superseded"] + _pending_orders()["no_live_indication"]
              if o["test"] == "Digoxin Level"]
    for o in orders:
        on_it = q1(db, "SELECT count(*) FROM v_medication WHERE PAT_ID = ? "
                       "AND lower(DISPLAY_NAME) LIKE '%digoxin%'", [o["pat_id"]]) > 0
        assert o["drug_monitoring_check"]["patient_is_on_it"] == on_it


# ------------------------------------------------------------ blood pressure
def test_bp_stages_match_acc_aha_thresholds(db):
    r = blood_pressure_staging()
    for rec in r["crisis"]:
        s, d = (int(x) for x in rec["bp"].split("/"))
        assert s > 180 or d > 120, f"{rec['patient']} staged crisis at {rec['bp']}"
    for rec in r["stage_2"]:
        s, d = (int(x) for x in rec["bp"].split("/"))
        assert not (s > 180 or d > 120)
        assert s >= 140 or d >= 90


def test_bp_crisis_count_matches_sql(db):
    expected = q1(db, """
        SELECT count(*) FROM (SELECT systolic s, diastolic d FROM
          (SELECT *, row_number() OVER (PARTITION BY PAT_ID ORDER BY RECORD_DATE DESC) rn
           FROM v_vitals) WHERE rn = 1)
        WHERE (s > 180 OR d > 120) AND (s - d) BETWEEN 20 AND 100""")
    assert len(blood_pressure_staging()["crisis"]) == expected


def test_implausible_pulse_pressures_are_separated():
    for rec in blood_pressure_staging()["implausible_readings"]:
        pp = rec["pulse_pressure"]
        assert pp <= 0 or pp < 20 or pp > 100
    for rec in blood_pressure_staging()["crisis"] + blood_pressure_staging()["stage_2"]:
        assert 20 <= rec["pulse_pressure"] <= 100


# --------------------------------------------------------------- lab plausibility
def test_physiologic_limits_flag_exactly_the_out_of_range(db):
    flagged = ScreeningSession().implausible_patients()
    for analyte, (lo, hi) in PHYSIOLOGIC_LIMITS.items():
        for pid, in db.execute(
            "SELECT DISTINCT PAT_ID FROM v_lab_result WHERE COMPONENT_NAME = ? "
            "AND (value < ? OR value > ?)", [analyte, lo, hi]).fetchall():
            assert pid in flagged, f"{pid} has out-of-range {analyte} but was not flagged"


# ------------------------------------------------------------------ guidelines
@pytest.mark.parametrize("gid", [g["id"] for g in GUIDELINES])
def test_guideline_population_and_gaps_partition(gid):
    r = check_guideline(gid)
    assert r["concordant_count"] + r["gap_count"] == r["in_population"]


def test_guideline_gaps_really_lack_the_therapy(db):
    for g in GUIDELINES:
        if "expected_classes" not in g:
            continue
        for gap in check_guideline(g["id"])["gaps"]:
            n = q1(db, f"""SELECT count(*) FROM v_medication m JOIN patient p USING (PAT_ID)
                   WHERE p.PAT_NAME = ? AND m.generic_class IN
                   ({','.join('?' * len(g['expected_classes']))})""",
                   [gap["name"], *g["expected_classes"]])
            assert n == 0, f"{gap['name']} flagged as a {g['id']} gap but is on the therapy"


# ------------------------------------------------------------------- lookups
def test_patient_snapshot_accepts_id_or_name():
    by_name = patient_snapshot("Stein, Larry")
    by_id = patient_snapshot(by_name["pat_id"])
    assert by_name["name"] == by_id["name"] == "Stein, Larry"


def test_patient_snapshot_reports_a_miss_helpfully():
    r = patient_snapshot("Nobody, Here")
    assert "error" in r and "hint" in r


def test_find_patients_expands_a_concept(db):
    r = find_patients("heart failure", "")
    expected = q1(db, "SELECT count(DISTINCT PAT_ID) FROM v_diagnosis WHERE icd10 LIKE 'I50%'")
    assert r["count"] == expected
    assert len(r["matched_diagnoses"]) == 3


def test_medication_timeline_spots_sequential_switching():
    r = medication_timeline("Rogers, Jessica")
    statins = [x for x in r["same_class_repeats"] if x["class"] == "Statin"]
    assert statins and statins[0]["days_apart"] > 365, \
        "Rogers' statin orders span years; they must not read as concurrent"


def test_vocabulary_rejects_codes_absent_from_the_dataset():
    s = ScreeningSession()
    bad = s.define_criterion("x", "diabetes", "include", "diagnosis",
                             ["E11.22", "Z99.9"], "", "", "", "")
    assert "error" in bad and set(bad["rejected"]) == {"E11.22", "Z99.9"}


# ------------------------------------------------------------------- the floor
def test_floor_is_deterministic():
    """Identical on every call, or it is not a floor."""
    import json
    from prototype.floor import compute_floor
    assert json.dumps(compute_floor(), sort_keys=True) == \
           json.dumps(compute_floor(), sort_keys=True)


def test_floor_contains_the_drug_monitoring_mismatches(db):
    """Padilla's INR-on-a-DOAC surfaced in 1 of 5 runs before the floor existed."""
    from prototype.floor import compute_floor
    named = {p for f in compute_floor()
             if f["category"] == "drug monitoring mismatch" for p in f["patients"]}
    assert "Padilla, Elizabeth" in named


def test_floor_crisis_matches_the_staging_tool():
    from prototype.floor import compute_floor
    from prototype.panel import blood_pressure_staging
    floor_named = {p for f in compute_floor()
                   if f["category"] == "hypertensive crisis" for p in f["patients"]}
    assert floor_named == {r["patient"] for r in blood_pressure_staging()["crisis"]}


def test_floor_is_seeded_before_any_agent_runs():
    from prototype.panel import Findings, build_supervisor
    f = Findings()
    build_supervisor([], f)
    rows = f.all()
    assert rows and all(r["agent"] == "guaranteed" for r in rows)


# ------------------------------------------------------------- shared rules
def test_every_agent_carries_every_shared_rule():
    """The rules drifted when they lived inline; this is what stops that.

    They were copied by hand into each instruction, so SEVERITY_WORDS reached
    two specialists of three and the pre-visit brief -- written last --
    reintroduced an honorific bug already fixed in the panel agents.
    """
    from prototype import brief, panel
    from prototype.rules import BP_SEVERITY, NAMING, NO_STOP_DATES, NUMBERS

    instructions = {
        "integrity": panel.INTEGRITY_INSTRUCTION,
        "guideline": panel.GUIDELINE_INSTRUCTION,
        "followup": panel.FOLLOWUP_INSTRUCTION,
        "brief": brief.BRIEF_INSTRUCTION,
    }
    rules = {"naming": NAMING, "numbers": NUMBERS,
             "no stop dates": NO_STOP_DATES, "bp severity": BP_SEVERITY}
    missing = [f"{agent} is missing the {rule} rule"
               for agent, text in instructions.items()
               for rule, body in rules.items() if body.strip() not in text]
    assert not missing, missing


def test_shared_rules_appear_once_per_instruction():
    """A rule pasted twice is a rule someone will edit in one place."""
    from prototype import brief, panel
    from prototype.rules import NAMING
    marker = NAMING.strip().splitlines()[0]
    for text in (panel.INTEGRITY_INSTRUCTION, panel.GUIDELINE_INSTRUCTION,
                 panel.FOLLOWUP_INSTRUCTION, brief.BRIEF_INSTRUCTION):
        assert text.count(marker) == 1
