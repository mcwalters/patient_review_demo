"""Pre-visit brief: everything known about one patient, assembled deterministically.

This module contains no model. It gathers the facts -- who the patient is, what
they are on, what is outstanding, which recommendations they fall short of, and
which of their records cannot be trusted -- and returns them as structured
sections. A narrative layer can sit on top later; the pack is readable on its
own, and is the fallback when the model is unavailable.

Ordering is deliberate. Data-quality flags come first. If a patient's sodium is
impossible, the reader learns that before reading their labs, not after.
"""
from __future__ import annotations

from datetime import date, timedelta

from .guidelines import GUIDELINES, check_guideline
from .tools import AS_OF, ScreeningSession, connect

# An AWV is an annual benefit, and the notes themselves say "Follow-up in 12
# months". There are no scheduled appointments in this extract, so the next
# visit is DERIVED from the last one -- never presented as a booked date.
AWV_INTERVAL = timedelta(days=365)


def _due_status(last_contact: date) -> dict:
    due = last_contact + AWV_INTERVAL
    days = (AS_OF - due).days
    if days > 0:
        status, detail = "overdue", f"{days // 30} months overdue"
    elif days > -60:
        status, detail = "due soon", f"due in {abs(days)} days"
    else:
        status, detail = "not yet due", f"due in {abs(days) // 30} months"
    return {"last_awv": str(last_contact), "next_awv_due_derived": str(due),
            "status": status, "detail": detail,
            "basis": "last AWV + 12 months; this extract has no scheduled "
                     "appointments, so the date is derived, not booked"}


def build_brief_pack(patient: str, findings: list[dict] | None = None) -> dict:
    """Assemble everything known about one patient. Deterministic, no model.

    Args:
        patient: a PAT_ID or a name exactly as recorded.
        findings: optionally, the panel-review findings, so the brief can show
                  which of them named this patient.
    """
    con = connect()
    try:
        row = con.execute(
            "SELECT PAT_ID, PAT_NAME, PAT_AGE, SEX_NAME, BIRTH_DATE "
            "FROM patient WHERE PAT_ID = ? OR lower(PAT_NAME) = lower(?)",
            [patient, patient]).fetchone()
        if not row:
            return {"error": f"no patient matches {patient!r}"}
        pid, name, age, sex, dob = row

        last = con.execute("SELECT max(CONTACT_DATE) FROM pat_enc WHERE PAT_ID = ?",
                           [pid]).fetchone()[0]
        encounters = con.execute(
            "SELECT CONTACT_DATE, DEPARTMENT_NAME, VISIT_PROV_NAME, INSURANCE "
            "FROM pat_enc WHERE PAT_ID = ? ORDER BY CONTACT_DATE DESC", [pid]).fetchall()

        dx = con.execute(
            "SELECT DISTINCT icd10, dx_name FROM v_diagnosis WHERE PAT_ID = ? "
            "ORDER BY icd10", [pid]).fetchall()

        meds = con.execute("""
            SELECT DISTINCT SIMPLE_GENERIC_C_NAME, DISPLAY_NAME, START_DATE
            FROM order_med WHERE PAT_ID = ? ORDER BY START_DATE DESC""", [pid]).fetchall()
        by_class: dict[str, list] = {}
        for cls, agent, start in meds:
            by_class.setdefault(cls, []).append({"agent": agent, "started": str(start)})

        vit = con.execute("""
            SELECT systolic, diastolic, heart_rate, temperature, resp_rate,
                   weight_lb, bmi, RECORD_DATE
            FROM v_vitals WHERE PAT_ID = ? ORDER BY RECORD_DATE DESC LIMIT 1""",
            [pid]).fetchone()

        labs = con.execute("""
            SELECT COMPONENT_NAME, value, unit, REFERENCE_LOW, REFERENCE_HIGH,
                   is_abnormal, RESULT_DATE
            FROM (SELECT *, row_number() OVER (PARTITION BY COMPONENT_NAME
                           ORDER BY RESULT_DATE DESC) rn
                  FROM v_lab_result WHERE PAT_ID = ?) WHERE rn = 1
            ORDER BY is_abnormal DESC, COMPONENT_NAME""", [pid]).fetchall()

        pending = con.execute("""
            SELECT test_name, ORDER_DATE, date_diff('month', ORDER_DATE, DATE '2026-05-27')
            FROM v_lab_order WHERE PAT_ID = ? AND is_pending
            ORDER BY ORDER_DATE""", [pid]).fetchall()
    finally:
        con.close()

    # --- blood pressure, staged rather than eyeballed -------------------------
    bp = None
    if vit and vit[0] is not None:
        sys_, dia = vit[0], vit[1]
        pp = sys_ - dia
        if sys_ > 180 or dia > 120:
            stage = "hypertensive crisis"
        elif sys_ >= 140 or dia >= 90:
            stage = "stage 2"
        elif sys_ >= 130 or dia >= 80:
            stage = "stage 1"
        elif sys_ >= 120:
            stage = "elevated"
        else:
            stage = "normal"
        bp = {"reading": f"{sys_}/{dia}", "stage": stage, "pulse_pressure": pp,
              "thresholds": "ACC/AHA 2017",
              "implausible": pp <= 0 or pp < 20 or pp > 100}

    # --- which recommendations this patient falls short of ---------------------
    gaps = []
    for g in GUIDELINES:
        res = check_guideline(g["id"])
        if any(x["name"] == name for x in res.get("gaps", [])):
            gaps.append({"id": g["id"], "title": g["title"],
                         "recommendation": g["recommendation"],
                         "source": g["source"], "caveat": g.get("caveat", "")})

    # --- data quality FIRST ----------------------------------------------------
    flags = list(ScreeningSession().implausible_patients().get(pid, []))
    if bp and bp["implausible"]:
        flags.append(f"blood pressure {bp['reading']} has a pulse pressure of "
                     f"{bp['pulse_pressure']}, which is not physiologically possible")
    repeats = {c: v for c, v in by_class.items() if len(v) > 1}

    return {
        "patient": {"pat_id": pid, "name": name, "age": age, "sex": sex,
                    "birth_date": str(dob)},
        "visit": _due_status(last) | {"encounters_on_file": len(encounters),
                                      "history": [{"date": str(e[0]), "department": e[1],
                                                   "provider": e[2], "insurance": e[3]}
                                                  for e in encounters]},
        "data_quality": {
            "flags": flags,
            "medication_caveat":
                "This extract records no stop dates -- END_DATE and DISCON_TIME are "
                "empty and every order reads Active -- so the list below is what was "
                "ever STARTED, not what is currently taken. Same-class repeats are "
                "almost certainly sequential switches; check the start dates.",
            "same_class_repeats": [
                {"class": c, "n": len(v), "agents": v,
                 "days_apart": (date.fromisoformat(v[0]["started"])
                                - date.fromisoformat(v[-1]["started"])).days}
                for c, v in repeats.items()],
        },
        "conditions": [{"icd10": d[0], "name": d[1]} for d in dx],
        "medications": {"by_class": by_class, "distinct_agents":
                        len({a["agent"] for v in by_class.values() for a in v})},
        "vitals": ({"blood_pressure": bp, "heart_rate": vit[2], "temperature": vit[3],
                    "resp_rate": vit[4], "weight_lb": vit[5], "bmi": vit[6],
                    "recorded": str(vit[7])} if vit else None),
        "labs": {
            "abnormal": [{"analyte": l[0], "value": l[1], "unit": l[2],
                          "reference": f"{l[3]}-{l[4]}", "date": str(l[6])}
                         for l in labs if l[5]],
            "normal_count": sum(1 for l in labs if not l[5]),
        },
        "outstanding_orders": [{"test": o[0], "ordered": str(o[1]),
                                "months_open": o[2]} for o in pending],
        "care_gaps": gaps,
        "panel_findings": [
            {"finding_id": f.get("finding_id"), "agent": f.get("agent"),
             "severity": f.get("severity"), "headline": f.get("headline")}
            for f in (findings or []) if name in (f.get("patients") or [])
        ],
    }
