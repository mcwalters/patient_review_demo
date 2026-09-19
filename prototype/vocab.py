"""The controlled vocabulary actually present in ehr.duckdb.

The LLM never invents an ICD-10 code, drug class or analyte name. It selects
from these lists. A code that is not in the data cannot be selected, so a
hallucinated code cannot reach the generated SQL.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

import duckdb

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ehr.duckdb")


@dataclass
class Vocabulary:
    conditions: list[dict] = field(default_factory=list)   # icd10, name, patients
    med_classes: list[dict] = field(default_factory=list)  # drug_class, agents, patients
    analytes: list[dict] = field(default_factory=list)     # analyte, unit, ref_low, ref_high, n
    vitals: list[str] = field(default_factory=list)
    demographics: list[str] = field(default_factory=list)

    def as_prompt_block(self) -> str:
        """Render for injection into a prompt. Compact but complete."""
        lines = ["## DIAGNOSES (icd10 | name | patients)"]
        for c in self.conditions:
            lines.append(f"{c['icd10']} | {c['name']} | {c['patients']}")
        lines.append("\n## MEDICATION CLASSES (class | patients | agents)")
        for m in self.med_classes:
            lines.append(f"{m['drug_class']} | {m['patients']} | {', '.join(m['agents'])}")
        lines.append("\n## LAB ANALYTES (name | unit | reference range | n results)")
        for a in self.analytes:
            rng = f"{a['ref_low']}-{a['ref_high']}" if a["ref_low"] is not None else "n/a"
            lines.append(f"{a['analyte']} | {a['unit']} | {rng} | {a['n']}")
        lines.append("\n## VITALS (one row per patient per encounter date)")
        lines.append(", ".join(self.vitals))
        lines.append("\n## DEMOGRAPHICS")
        lines.append(", ".join(self.demographics))
        return "\n".join(lines)

    # Membership tests used to validate LLM output before it reaches SQL.
    @property
    def icd10_codes(self) -> set[str]:
        return {c["icd10"] for c in self.conditions}

    @property
    def class_names(self) -> set[str]:
        return {m["drug_class"] for m in self.med_classes}

    @property
    def analyte_names(self) -> set[str]:
        return {a["analyte"] for a in self.analytes}


@lru_cache(maxsize=1)
def load(db: str = DB) -> Vocabulary:
    con = duckdb.connect(db, read_only=True)
    try:
        conditions = [
            {"icd10": r[0], "name": r[1], "patients": r[2]}
            for r in con.execute("""
                SELECT icd10, any_value(dx_name), count(DISTINCT PAT_ID)
                FROM v_diagnosis GROUP BY icd10 ORDER BY 3 DESC, 1
            """).fetchall()
        ]
        med_classes = [
            {"drug_class": r[0], "patients": r[1], "agents": r[2]}
            for r in con.execute("""
                SELECT generic_class, count(DISTINCT PAT_ID),
                       list_sort(list(DISTINCT DISPLAY_NAME))
                FROM v_medication GROUP BY 1 ORDER BY 2 DESC, 1
            """).fetchall()
        ]
        analytes = [
            {"analyte": r[0], "unit": r[1], "ref_low": r[2], "ref_high": r[3], "n": r[4]}
            for r in con.execute("""
                SELECT COMPONENT_NAME, any_value(unit),
                       any_value(REFERENCE_LOW), any_value(REFERENCE_HIGH), count(*)
                FROM v_lab_result GROUP BY 1 ORDER BY 1
            """).fetchall()
        ]
        vitals = [c[0] for c in con.execute("DESCRIBE v_vitals").fetchall()
                  if c[0] not in ("PAT_ID", "PAT_MRN_ID", "RECORD_DATE")]
        demographics = ["PAT_AGE (integer years)", "SEX_NAME (Male/Female)",
                        "BIRTH_DATE", "INSURANCE (medicare/commercial/medicaid)"]
    finally:
        con.close()
    return Vocabulary(conditions, med_classes, analytes, vitals, demographics)


if __name__ == "__main__":
    v = load()
    print(f"{len(v.conditions)} diagnoses, {len(v.med_classes)} drug classes, "
          f"{len(v.analytes)} analytes, {len(v.vitals)} vitals")
    block = v.as_prompt_block()
    print(f"\nprompt block: {len(block)} chars, {len(block.split(chr(10)))} lines\n")
    print(block[:900] + "\n...")
