# fake_ehr — synthetic EHR demo dataset

A synthetic Epic Clarity–style extract built around the **Annual Wellness Visit (AWV)**,
loaded into a single-file DuckDB database for local analytics.

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python build_db.py      # writes ehr.duckdb
```

`ehr.duckdb` is derived and gitignored — rebuild it any time from `data/`.

## Prototype

[`prototype/`](prototype/README.md) holds a working GenAI prototype built on this
data: **eligibility screening from a free-text protocol**. An ADK agent running
Gemini 2.5 Pro on Vertex AI (application default credentials, no API key) reads
criteria as written, grounds every clinical concept in the dataset's own
vocabulary, and returns the cohort with per-patient evidence. A pre-flight
plausibility linter audits the extract first.

```bash
streamlit run prototype/app.py
```

The agent never writes SQL — it selects from a fixed vocabulary and deterministic
code runs every query, so a hallucinated ICD-10 code cannot reach the database.
See [`prototype/README.md`](prototype/README.md) for the architecture, the safety
properties, and the two ways it has been observed to get concept expansion wrong.

Thirteen worked examples live in [`demo_queries.sql`](demo_queries.sql) — panel
snapshot, chronic-condition registry, care gaps (uncontrolled hypertension,
heart failure missing guideline-directed therapy, AFib without anticoagulation),
lab trajectories, outstanding orders, polypharmacy, note search, and two on the
notes themselves (an extraction audit and the template distribution):

```bash
duckdb ehr.duckdb -f demo_queries.sql
```

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
| `v_note_extract` | 153 | progress notes parsed into structured slots |

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

### Progress notes are generated from the structured data

`hno_info.NOTE_TEXT` looks like free text but is templated prose in four styles
(`Annual Wellness Visit` ×55, `Annual preventive visit` ×41, `AWV encounter` ×33,
`Progress Note — AWV` ×24), each with the same six slots. `v_note_extract`
parses them into `note_style`, `conditions[]`, `medications[]`, `systolic`,
`diastolic`, `bmi` and `followup_months`.

Query 12 scores that extraction against the structured tables and every field
comes back at **100%** — the notes agree exactly with `ip_flwsht_meas`,
`pat_enc_dx` and `order_med`, because they were rendered from them.

So the notes are useful for demonstrating extraction *mechanics* against a known
answer key, but they contain **no information that isn't already in a column**.
There is no negation, hedging, social history, symptom or exam narrative to
find. `followup_months` is always 12, and note style is unrelated to the
authoring service.

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
- **Provider attribution is random.** See below.
- **Provider name fields carry titles and credentials.** Four of 20: `Peter
  Thomas DVM` (a veterinary doctorate on someone `PROV_TYPE` says is a PA),
  `Miss Adriana Flores`, `Dr. Joseph Costa`, `Dean Washington Jr.` Patient names
  are clean, so the two tables came from different generator calls. The provider
  name is denormalized into five other tables — 155 rows of
  `order_proc_awv.ORD_PROV_NAME`, 41 each of `pat_enc.VISIT_PROV_NAME` and
  `hno_info.AUTHOR_PROV_NAME`, 33 of `pat_enc.PROV_NAME_WID`, 22 of
  `patient.CUR_PCP_PROV_NAME` — so 292 rows carry a dirty string. **Join on
  `PROV_ID`, never on `PROV_NAME`**, and read credentials from `PROV_TYPE`
  rather than parsing the name.
- **ICD-10 codes fragment across near-duplicates.** Grouping on the raw code
  silently halves most cohorts. Roll up to condition families first.

  | family | true patients | largest single code | undercount |
  |---|---|---|---|
  | Lipid disorder | 50 | `E78.00` (19) | −31 |
  | Hypertension | 46 | `I11.9` (25) | −21 |
  | Type 2 diabetes | 28 | `E11.51` (12) | −16 |
  | GERD | 27 | `K21.9` (17) | −10 |
  | Anxiety | 18 | `F41.9` (10) | −8 |
  | Depression | 13 | `F32.9` (8) | −5 |
  | Atrial fibrillation | 9 | `I48.19` (4) | −5 |

- **Age and insurance are incoherent with the visit type.** Median age 43.5,
  range 18–81, and only **19 of 100** patients are 65+ — yet all 153 encounters
  are billed as Annual Wellness Visits, a Medicare benefit. `INSURANCE` is
  unrelated to age: "medicare" covers 51 patients aged **18–75**. Do not build
  anything that depends on age-based screening thresholds or Medicare
  eligibility.
- **Medications are 100% coherent with their indication.** Across nine classes
  tested (statin→lipids, SSRI→depression/anxiety, DOAC→AFib, PPI→GERD,
  SGLT2i→T2DM/HF, and others) every patient on the drug carries the matching
  diagnosis. The generator assigned drugs from conditions, so **"on a drug with
  no indication" returns zero rows** — overtreatment and wrong-drug detection
  have no material here. Undertreatment does (24 of 28 diabetics are not on a
  statin), as does within-class duplication (20 patient-class pairs, including
  three patients on three SSRIs and one on three DOACs).
- **Some prescribing rates are implausible.** PCSK9i appears in 16 of 100
  patients, half of them not on a statin; real-world use is 1–2% of a lipid
  population and near-always statin-refractory. Useful for making a mechanism
  legible, not for citing as epidemiology.
- **There is almost no coverage data, and `COVERAGE_ID` is not a coverage key.**
  The entire footprint is two columns on `pat_enc`. `INSURANCE` holds three
  lowercase payer categories — `medicare` (82 encounters / 51 patients),
  `commercial` (43 / 31), `medicaid` (28 / 18) — with no nulls and no casing
  variants, but it is a *category*, not a plan. `COVERAGE_ID` has 153 distinct
  values over 153 encounters and **no patient shares one across their two
  visits**, so it is a row id, not a policy.

  Absent entirely: plan or product name, group number, member id, subscriber,
  effective and term dates, eligibility, benefit tier, copay or deductible,
  secondary payer or COB, plan type (HMO/PPO/EPO), line of business, contract or
  risk arrangement, Medicare Advantage vs FFS, and attribution.

  Payer is stable per patient (53 of 53 keep the same one across both visits)
  but is assigned independently of age — **45 of 81 under-65 patients are on
  medicare, and 13 of 19 patients aged 65+ are not**; the medicare cohort has
  the *youngest* median age at 37. Payer is also uncorrelated with diagnosis
  count, medication count and labs ordered.

  So anything financial or contractual is out of reach: value-based care and
  risk contracts, Medicare Advantage or Stars, prior authorisation and
  formulary, cost or reimbursement modelling, coverage-gap and churn analysis,
  and payer-mix segmentation of any clinical finding.
- **Note phrasing is template assignment, not clinical fact.** See below.

### Provider attribution is random

The 20 providers (10 Physician, 6 NP, 4 PA) span eight specialties, but nothing
about who saw whom holds together:

- **Specialty does not match home department** for 19 of 20 — a Cardiologist
  staffing Endocrinology, Family Medicine in the GI Clinic.
- **Visit department does not match provider specialty** — Neurology running
  Annual Wellness Visits in the OB/GYN Clinic.
- **Only 9 of 153 visits** are with the patient's own `CUR_PCP_PROV_ID`.
- Only 4 of 20 providers are primary care at all, yet all 153 visits are AWVs.
- **NPIs are random 10-digit numbers.** Just 1 of 20 passes the standard Luhn
  check with the `80840` prefix, which is chance. There are no NUCC taxonomy
  codes, licence numbers or board certifications anywhere in the dataset.

Note authorship itself is consistent — the note author is the visit provider in
153 of 153, and `AUTHOR_SERVICE` equals `provider.SPECIALTY` in 153 of 153 — so
the randomness is in the *assignment*, not the bookkeeping.

Anything that depends on provider routing, panel attribution, referral logic or
care-team structure has no signal to work with here.

### Don't build cohorts on note prose

The four note styles are randomly assigned. Template choice predicts nothing —
chi-square against department, location, insurance, sex, authoring service,
provider specialty, primary diagnosis and month of visit is null in every case
(p = 0.12 to 0.93), and age, BP, BMI, condition count, medication count, labs
ordered and note length all differ by under 0.5 SD between templates. Of the 53
patients with two encounters, only 11 drew the same template twice — below the
~14 expected by chance, so it is redrawn per note rather than sticking to a
patient or provider.

Each template recites a fixed closing, with **no variation within the template**:

| style | notes | patients | lab phrasing | also states |
|---|---|---|---|---|
| `Annual Wellness Visit` | 55 | 50 | preventive labs ordered | medication adherence education |
| `Annual preventive visit` | 41 | 38 | monitoring labs ordered | reports adherence, lifestyle counselling |
| `AWV encounter` | 33 | 31 | labs reviewed, plan updated | follow-up in 12 months |
| `Progress Note — AWV` | 24 | 23 | lab orders placed | preventive screening gaps addressed |

So a query like *"which patients received lifestyle counselling?"* returns
exactly the 38 patients who drew template B on at least one note — an artifact
of template assignment, not a clinical finding. The same holds for "follow-up
documented" (template C) and "screening gaps addressed" (template D). **Any
cohort built on note phrasing is really a cohort of template assignment.**

Note the notes/patients gap: a patient with two encounters can draw two
different templates, so the same person may be "counselled" in one note and not
the other.

The six parsed slots in `v_note_extract` are trustworthy — but only because they
restate the structured tables, which is what query 12 verifies.

### What's clean

All 19 other foreign-key relationships resolve with zero orphans. Abnormal flags
agree with reference ranges in 100% of `order_results` rows. RxNorm coverage is
100%. Every encounter has exactly one note and one full set of vitals.
