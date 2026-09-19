"""Deterministic tools the screening agent calls.

Design rule: the agent never writes SQL. It searches a fixed vocabulary and
registers criteria as structured values; these functions build and run every
query. A code the dataset does not contain cannot be registered, so a
hallucinated ICD-10 code cannot reach the database.

Three-state evaluation. A criterion is met, not_met, or UNKNOWN when the
patient has no data to judge it on. Unknown is never silently treated as a
pass: it routes the patient to review. Missing is not normal.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field

from datetime import date

import duckdb

from . import vocab

DB = vocab.DB


def connect() -> duckdb.DuckDBPyConnection:
    """Read-only connection, falling back to a copy if another process holds the lock."""
    try:
        return duckdb.connect(DB, read_only=True)
    except duckdb.IOException:
        tmp = os.path.join(tempfile.gettempdir(), "ehr_screening_copy.duckdb")
        shutil.copy2(DB, tmp)
        return duckdb.connect(tmp, read_only=True)


# Hard physiologic limits. Deterministic backstop to the LLM plausibility
# linter: a value outside these is not a rare finding, it is a bad record.
# The linter (prototype/preflight.py) reasons about *rates* and *combinations*,
# which need clinical knowledge; these are the bounds that never need judgment.
PHYSIOLOGIC_LIMITS = {
    "SpO2": (50, 100),                 # a saturation above 100% is impossible
    "Sodium": (110, 160),
    "CBC — Hemoglobin": (4, 22),
    "LDL Cholesterol": (10, 400),
    "Total Cholesterol": (50, 500),
    "HDL Cholesterol": (10, 150),
    "Triglycerides": (20, 1500),
    "HbA1c": (3, 20),
    "Potassium": (2.0, 8.0),
    "eGFR": (3, 150),
}

# The extract's snapshot date. Every "months open" / "how stale" calculation in
# the prototype measures from here, so the panel view and the patient brief can
# never disagree about how old something is.
AS_OF = date(2026, 5, 27)

KINDS = ("diagnosis", "medication", "lab", "vital", "demographic")
POLARITIES = ("include", "exclude")


@dataclass
class Criterion:
    criterion_id: str
    source_text: str
    polarity: str
    kind: str
    codes: list[str] = field(default_factory=list)
    field_name: str = ""
    min_value: float | None = None
    max_value: float | None = None
    rationale: str = ""
    offered: list[str] = field(default_factory=list)


def _num(s: str) -> float | None:
    s = (s or "").strip()
    if s in ("", "none", "null", "-"):
        return None
    return float(s)


class ScreeningSession:
    """Holds the criteria the agent has registered, and evaluates them."""

    def __init__(self) -> None:
        self.vocab = vocab.load()
        self.criteria: dict[str, Criterion] = {}
        self.rejected: list[dict] = []
        # What the search tools last offered, per kind. Lets us show a reviewer
        # which candidate codes the agent saw and chose NOT to use -- concept
        # under-expansion is invisible otherwise.
        self.offered: dict[str, list[str]] = {}

    # ---- vocabulary search (read-only, grounds the agent) -------------------
    @staticmethod
    def _relevant(query: str, text: str) -> bool:
        """Loose stem match: 'hypertension' should surface 'Hypertensive'."""
        t = text.lower()
        for w in query.lower().replace("-", " ").split():
            if len(w) < 4:
                continue
            if w in t or w[:6] in t or w.rstrip("s")[:7] in t:
                return True
        return False

    def search_diagnoses(self, query: str) -> dict:
        # Always return the whole vocabulary (only 30 rows). A naive matcher that
        # hides candidates is how a cohort silently loses patients -- surfacing
        # everything and marking likely hits is strictly safer.
        best = [c["icd10"] for c in self.vocab.conditions if self._relevant(query, c["name"])]
        return {"query": query, "likely_matches": best,
                "all_diagnoses_in_dataset": self.vocab.conditions,
                "note": "Every diagnosis in the dataset is listed. `likely_matches` is a "
                        "keyword hint only -- judge each entry yourself, and include every "
                        "code that is genuinely an instance, subtype or complication of the "
                        "concept."}

    def search_medications(self, query: str) -> dict:
        q = query.lower()
        hits = [m for m in self.vocab.med_classes
                if q in m["drug_class"].lower()
                or any(q in a.lower() for a in m["agents"])
                or any(w in m["drug_class"].lower() for w in q.split() if len(w) > 3)]
        best = [m["drug_class"] for m in self.vocab.med_classes
                if self._relevant(query, m["drug_class"] + " " + " ".join(m["agents"]))]
        return {"query": query, "likely_matches": best,
                "all_classes_in_dataset": self.vocab.med_classes,
                "note": "Every medication class in the dataset is listed with its agents. "
                        "`likely_matches` is a keyword hint only -- judge each class yourself."}

    def search_vitals(self, query: str) -> dict:
        """Vitals are stored as columns on v_vitals, one row per patient per date."""
        return {"query": query, "matches": [
            {"vital": v, "note": {"bmi": "body mass index",
                                  "systolic": "systolic blood pressure, mmHg",
                                  "diastolic": "diastolic blood pressure, mmHg",
                                  "weight_lb": "weight in pounds",
                                  "heart_rate": "beats per minute",
                                  "resp_rate": "breaths per minute",
                                  "temperature": "degrees F"}.get(v, "")}
            for v in self.vocab.vitals],
            "note": "Use the exact `vital` string as field_name. Names are lowercase."}

    def search_analytes(self, query: str) -> dict:
        q = query.lower()
        hits = [a for a in self.vocab.analytes
                if q in a["analyte"].lower()
                or any(w in a["analyte"].lower() for w in q.split() if len(w) > 2)]
        return {"query": query, "matches": hits or self.vocab.analytes,
                "note": "No match; returning all analytes." if not hits
                        else f"{len(hits)} of {len(self.vocab.analytes)} analytes matched."}

    # ---- criterion registration (validated) ---------------------------------
    def define_criterion(self, criterion_id: str, source_text: str, polarity: str,
                         kind: str, codes: list[str], field_name: str,
                         min_value: str, max_value: str, rationale: str) -> dict:
        if polarity not in POLARITIES:
            return {"error": f"polarity must be one of {POLARITIES}"}
        if kind not in KINDS:
            return {"error": f"kind must be one of {KINDS}"}

        # Reject anything not present in the dataset. This is the guardrail.
        bad: list[str] = []
        if kind == "diagnosis":
            bad = [c for c in codes if c not in self.vocab.icd10_codes]
        elif kind == "medication":
            bad = [c for c in codes if c not in self.vocab.class_names]
        elif kind == "lab" and field_name not in self.vocab.analyte_names:
            bad = [field_name]
        elif kind == "vital":
            # tolerate case/spacing differences rather than rejecting on cosmetics
            canon = {v.lower(): v for v in self.vocab.vitals}
            key = field_name.strip().lower().replace(" ", "_")
            if key in canon:
                field_name = canon[key]
            else:
                bad = [field_name]
        if bad:
            self.rejected.append({"criterion_id": criterion_id, "rejected": bad, "kind": kind})
            return {"error": f"not present in this dataset: {bad}. Search first and select "
                             f"only values the search tool returned.", "rejected": bad}

        try:
            lo, hi = _num(min_value), _num(max_value)
        except ValueError:
            return {"error": "min_value/max_value must be numeric strings or empty"}

        c = Criterion(criterion_id, source_text, polarity, kind, list(codes),
                      field_name, lo, hi, rationale)
        c.offered = list(self.offered.get(kind, []))
        self.criteria[criterion_id] = c
        res = self._evaluate(c)
        met = sum(1 for v in res.values() if v["status"] == "met")
        unk = sum(1 for v in res.values() if v["status"] == "unknown")
        return {"criterion_id": criterion_id, "registered": True,
                "patients_meeting": met, "patients_unknown": unk,
                "of_total": len(res)}

    # ---- evaluation ----------------------------------------------------------
    def _evaluate(self, c: Criterion) -> dict[str, dict]:
        """Return {pat_id: {status, evidence}} for all 100 patients."""
        con = connect()
        try:
            pats = [r[0] for r in con.execute("SELECT PAT_ID FROM patient ORDER BY 1").fetchall()]
            hits: dict[str, str] = {}
            has_data: set[str] = set(pats)  # kinds where absence == not_met

            if c.kind == "diagnosis":
                rows = con.execute(
                    "SELECT PAT_ID, string_agg(DISTINCT dx_name, '; ') FROM v_diagnosis "
                    "WHERE icd10 IN ({}) GROUP BY 1".format(",".join("?" * len(c.codes))),
                    c.codes).fetchall()
                hits = {r[0]: r[1] for r in rows}

            elif c.kind == "medication":
                rows = con.execute(
                    "SELECT PAT_ID, string_agg(DISTINCT DISPLAY_NAME, '; ') FROM v_medication "
                    "WHERE generic_class IN ({}) GROUP BY 1".format(",".join("?" * len(c.codes))),
                    c.codes).fetchall()
                hits = {r[0]: r[1] for r in rows}

            elif c.kind == "lab":
                rows = con.execute("""
                    SELECT PAT_ID, value, unit, RESULT_DATE FROM (
                      SELECT PAT_ID, value, unit, RESULT_DATE,
                             row_number() OVER (PARTITION BY PAT_ID ORDER BY RESULT_DATE DESC) rn
                      FROM v_lab_result WHERE COMPONENT_NAME = ?) WHERE rn = 1
                """, [c.field_name]).fetchall()
                has_data = {r[0] for r in rows}          # no result => unknown
                for pid, val, unit, dt in rows:
                    if val is None:
                        has_data.discard(pid); continue
                    if (c.min_value is None or val >= c.min_value) and \
                       (c.max_value is None or val <= c.max_value):
                        hits[pid] = f"{c.field_name} {val} {unit or ''} on {dt}"

            elif c.kind == "vital":
                rows = con.execute(f"""
                    SELECT PAT_ID, {c.field_name}, RECORD_DATE FROM (
                      SELECT PAT_ID, {c.field_name}, RECORD_DATE,
                             row_number() OVER (PARTITION BY PAT_ID ORDER BY RECORD_DATE DESC) rn
                      FROM v_vitals) WHERE rn = 1
                """).fetchall()
                has_data = {r[0] for r in rows if r[1] is not None}
                for pid, val, dt in rows:
                    if val is None:
                        continue
                    if (c.min_value is None or val >= c.min_value) and \
                       (c.max_value is None or val <= c.max_value):
                        hits[pid] = f"{c.field_name} {val} on {dt}"

            elif c.kind == "demographic":
                col = {"pat_age": "PAT_AGE", "age": "PAT_AGE", "sex_name": "SEX_NAME",
                       "sex": "SEX_NAME", "gender": "SEX_NAME",
                       "insurance": "INSURANCE"}.get(c.field_name.strip().lower(), c.field_name)
                rows = con.execute(f"SELECT PAT_ID, {col} FROM patient").fetchall()
                has_data = {r[0] for r in rows if r[1] is not None}
                for pid, val in rows:
                    if val is None:
                        continue
                    if isinstance(val, str):
                        if not c.codes or val in c.codes:
                            hits[pid] = f"{col}={val}"
                    else:
                        if (c.min_value is None or val >= c.min_value) and \
                           (c.max_value is None or val <= c.max_value):
                            hits[pid] = f"{col}={val}"
        finally:
            con.close()

        out = {}
        for p in pats:
            if p in hits:
                out[p] = {"status": "met", "evidence": hits[p]}
            elif p in has_data:
                out[p] = {"status": "not_met", "evidence": ""}
            else:
                out[p] = {"status": "unknown", "evidence": "no data on file"}
        return out

    def implausible_patients(self) -> dict[str, list[str]]:
        """Patients holding at least one physiologically impossible lab value."""
        con = connect()
        try:
            out: dict[str, list[str]] = {}
            for analyte, (lo, hi) in PHYSIOLOGIC_LIMITS.items():
                for pid, val in con.execute(
                    "SELECT PAT_ID, value FROM v_lab_result "
                    "WHERE COMPONENT_NAME = ? AND (value < ? OR value > ?)",
                    [analyte, lo, hi]).fetchall():
                    out.setdefault(pid, []).append(f"{analyte}={val} (limit {lo}-{hi})")
        finally:
            con.close()
        return out

    def run_screening(self) -> dict:
        """Combine every registered criterion into a cohort. Deterministic."""
        if not self.criteria:
            return {"error": "no criteria registered yet"}
        per = {cid: self._evaluate(c) for cid, c in self.criteria.items()}
        con = connect()
        try:
            names = dict(con.execute("SELECT PAT_ID, PAT_NAME FROM patient").fetchall())
            ages = dict(con.execute("SELECT PAT_ID, PAT_AGE FROM patient").fetchall())
        finally:
            con.close()

        bad_data = self.implausible_patients()
        eligible, excluded, review = [], [], []
        for pid in sorted(names):
            detail, verdict = {}, "eligible"
            for cid, c in self.criteria.items():
                st = per[cid][pid]["status"]
                if c.polarity == "include":
                    ok = st == "met"
                else:
                    ok = st == "not_met"
                detail[cid] = {"status": st, "passes": ok,
                               "evidence": per[cid][pid]["evidence"],
                               "source_text": c.source_text, "polarity": c.polarity}
                if st == "unknown":
                    verdict = "needs_review" if verdict == "eligible" else verdict
                elif not ok:
                    verdict = "excluded"
            row = {"pat_id": pid, "name": names[pid], "age": ages[pid],
                   "verdict": verdict, "criteria": detail,
                   "data_quality_flags": bad_data.get(pid, [])}
            {"eligible": eligible, "excluded": excluded, "needs_review": review}[verdict].append(row)

        # Per-criterion impact, and a deterministic flag for exclusions that remove
        # an unusually large share of the panel. An over-broad exclusion wrongly
        # denies people access, so it gets surfaced for human review rather than
        # trusted. This catches concept over-expansion without needing the model
        # to notice its own mistake.
        impact, warnings = {}, []
        n = len(names)
        for cid, c in self.criteria.items():
            st = [per[cid][p]["status"] for p in names]
            met = sum(1 for x in st if x == "met")
            unknown = sum(1 for x in st if x == "unknown")
            removed = met if c.polarity == "exclude" else n - met
            impact[cid] = {"source_text": c.source_text, "polarity": c.polarity,
                           "kind": c.kind, "codes": c.codes or c.field_name,
                           "met": met, "unknown": unknown, "removed_from_cohort": removed,
                           "rationale": c.rationale}
            # ICD-10 is hierarchical: codes sharing a 3-character category are
            # subtypes of one condition. Using some but not all of a category is
            # the signature of the fragmentation bug -- E11.9 without E11.51 and
            # E11.65 loses 21 of 28 diabetics. Precise, and it does not fire on
            # correct decisions the way a bare "unused candidate" check does.
            if c.kind == "diagnosis" and c.codes:
                cats = {x.split(".")[0] for x in c.codes}
                siblings = sorted({x for x in self.vocab.icd10_codes
                                   if x.split(".")[0] in cats and x not in c.codes})
                impact[cid]["unused_siblings"] = siblings
                if siblings:
                    extra = con_patients = sum(
                        1 for d in self.vocab.conditions if d["icd10"] in siblings)
                    warnings.append({
                        "criterion_id": cid, "severity": "high",
                        "message": f"'{c.source_text}' uses {sorted(c.codes)} but leaves "
                                   f"{siblings} unused from the same ICD-10 category. These are "
                                   f"subtypes of the same condition; omitting them usually means "
                                   f"a silently smaller cohort.",
                        "rationale": c.rationale})
            if c.polarity == "exclude" and met > 0.15 * n:
                warnings.append({
                    "criterion_id": cid, "severity": "high",
                    "message": f"Exclusion '{c.source_text}' removes {met} of {n} patients "
                               f"({met*100//n}%). Verify the codes are all truly part of this "
                               f"concept before acting: {c.codes or c.field_name}",
                    "rationale": c.rationale})
            if unknown > 0.30 * n:
                warnings.append({
                    "criterion_id": cid, "severity": "medium",
                    "message": f"'{c.source_text}' has no data for {unknown} of {n} patients. "
                               f"They are routed to review, not passed.",
                    "rationale": c.rationale})

        # A cohort built partly on impossible records should say so.
        tainted = [r for r in eligible + review if r["data_quality_flags"]]
        if tainted:
            warnings.append({
                "criterion_id": "_data_quality", "severity": "high",
                "message": f"{len(tainted)} patient(s) in the eligible/review set hold a "
                           f"physiologically impossible lab value. Their eligibility rests "
                           f"partly on records that cannot be correct: "
                           f"{[r['name'] for r in tainted][:5]}",
                "rationale": "Deterministic physiologic-range check, independent of the model."})

        return {"eligible": eligible, "needs_review": review, "excluded": excluded,
                "counts": {"eligible": len(eligible), "needs_review": len(review),
                           "excluded": len(excluded), "total": n,
                           "with_data_quality_flags": len(tainted)},
                "impact": impact, "warnings": warnings,
                "criteria": {cid: vars(c) for cid, c in self.criteria.items()}}
