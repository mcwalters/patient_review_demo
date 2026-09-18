# fake_ehr — synthetic EHR demo dataset

A synthetic Epic Clarity–style extract built around the **Annual Wellness Visit (AWV)**,
loaded into a single-file DuckDB database for local analytics.

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python build_db.py      # writes ehr.duckdb
```

`ehr.duckdb` is derived and gitignored — rebuild it any time from `data/`.

---

## What we received

14 CSVs, **4,780 rows, 1.1 MB**. 100 patients, 153 encounters, 20 providers,
30 distinct ICD-10 codes, 54 medications, 35 lab analytes.

Every patient has 1 or 2 AWV encounters (47 have one, 53 have two). The clinical
detail hangs off those encounters.

| CSV | Rows | Grain | What it is |
|---|---|---|---|
| `patient` | 100 | patient | demographics, PCP |
| `provider` | 20 | provider | NPI, specialty, department |
| `pat_enc` | 153 | encounter | the AWV itself: date, dept, location, insurance |
| `hno_info` | 153 | encounter | one signed progress note per encounter, free text |
| `pat_enc_dx` | 355 | encounter × dx | diagnoses coded at the visit |
| `problem_list` | 229 | patient × problem | longitudinal problem list, active/resolved |
| `medical_hx` | 355 | patient × dx | past medical history |
| `surgical_hx` | 37 | patient | surgical history + procedure flags |
| `order_med` | 522 | med order | prescriptions written |
| `rxnorm_codes` | 54 | medication | RxNorm lookup |
| `ip_frequency` | 8 | frequency | dosing frequency lookup |
| `ip_flwsht_meas` | 918 | encounter × measure | vitals, long-and-thin (6 per encounter) |
| `order_proc_awv` | 588 | lab order | labs ordered **at** the AWV, result inline |
| `order_results` | 1,288 | observation | **longitudinal** lab history |

### The two lab tables are separate on purpose

They describe different parts of the care journey and are **not** a parent/child pair:

|  | `order_proc_awv` | `order_results` |
|---|---|---|
| Window | encounter −3 to 0 days | encounter −727 to +345 days |
| Answers | "what did we order at this visit, and what came back?" | "what is this patient's lab trend?" |
| Result data | inline on the row | inline + reference ranges |
| Pending | 120 of 588 unresulted | n/a |

`ORDER_PROC_ID` values are disjoint between them (0 of 1,288 match), and where
the same encounter+analyte appears in both, the values and dates disagree. So
they are kept independent, each keyed to the **encounter**, never to each other.

---

## How it's represented in DuckDB

**14 tables** — one per CSV, loaded verbatim with full-file type inference
(`sample_size=-1`), so dates land as `DATE`/`TIMESTAMP` rather than text.
Columns that are 100% empty in the source are explicitly typed in `FORCE_TYPES`
so they don't silently become `VARCHAR`.

**6 views** — the analyst-facing layer:

| View | Rows | Purpose |
|---|---|---|
| `v_encounter` | 153 | encounter + patient + provider + primary dx, one row per visit |
| `v_vitals` | 153 | flowsheet pivoted wide; BP split into `systolic`/`diastolic` |
| `v_lab_order` | 588 | AWV labs with their inline result; `is_pending`, `is_abnormal` |
| `v_lab_result` | 1,288 | longitudinal labs with reference ranges and `is_abnormal` |
| `v_medication` | 522 | med orders + RxNorm + daily frequency |
| `v_diagnosis` | 939 | the three dx sources unioned, keyed on ICD-10 |

### Why the views exist

Four quirks in the source data make the raw tables awkward to query directly:

1. **BP is a string.** `ip_flwsht_meas.MEAS_VALUE` holds `"150/99"` and
   `MEAS_VALUE_NUM` is null for all 153 BP rows. `v_vitals` splits it.
2. **Vitals are long-and-thin.** Six rows per encounter. `v_vitals` pivots to
   one row per patient-date.
3. **`DX_ID` is row-random.** 355/355 distinct in `pat_enc_dx`, 229/229 in
   `problem_list`, cross-table overlap of 1. It is *not* a dictionary key —
   `v_diagnosis` unions the three sources on `ICD10_CODE`, the only stable
   diagnosis identifier.
4. **`GPI` is row-random too** — 522 distinct values for 54 drugs. Use
   `MEDICATION_ID` → `RXNORM_CODE` (100% coverage), as `v_medication` does.

---

## Known limits of the synthetic data

Worth knowing before you build a demo on it:

- **Vitals don't change between visits.** For all 53 patients with two
  encounters, systolic, diastolic, temperature, weight, and BMI are identical
  across both. Only heart rate varies. **No vitals-trending demo.** Labs *do*
  vary, with 4–24 distinct dates per patient — trend on those instead.
- **Low variety in categoricals.** Every encounter is a completed Office Visit,
  every med order is Active, every note is a signed Progress Note. Limits any
  filter/status demo.
- **`PAT_AGE` is stale** for 4 of 100 patients against `BIRTH_DATE`. Compute age
  from the birth date.
- **`RESULT_VALUE_TEXT` duplicates `ORD_NUM_VALUE`** in all 1,288 rows.
- ~30 columns are entirely empty (`DEATH_DATE`, `END_DATE`, `COMMENTS`, …).

### What's clean

All 19 other foreign-key relationships resolve with zero orphans. Abnormal flags
agree with reference ranges in 100% of `order_results` rows. RxNorm coverage is
100%. Every encounter has exactly one note and one full set of vitals.
