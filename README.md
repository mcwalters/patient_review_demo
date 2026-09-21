# fake_ehr — a panel manager's worklist over a synthetic EHR

A Streamlit demo in which agents rank a 100-patient panel by who needs
attention this week. The user is the population-health nurse who works the
list between visits; the product's job is **prioritisation, not retrieval** —
where to point scarce attention for the most urgent clinical need.

Gemini 2.5 Pro on Vertex AI via application default credentials. No model
writes SQL, and no model computes what code can compute. The data is a
synthetic Epic Clarity–style extract (`data/`, 14 CSVs) loaded into a
single-file DuckDB database.

## Run it

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
gcloud auth application-default login          # project accorded-lake, us-west1
./.venv/bin/python build_db.py                 # writes ehr.duckdb: 14 tables, 7 views
./.venv/bin/python -m streamlit run prototype/app.py \
    --server.port 8501 --server.headless true --server.fileWatcherType none
```

`ehr.duckdb` is derived and gitignored — rebuild it, never patch it. The
`--server.fileWatcherType none` flag matters: with the watcher on, editing any
file under `prototype/` during an agent run cancels the run.

The app opens on a **saved panel review**, so the first screen renders
instantly rather than after a three-minute agent run. The *Run panel review*
button is still there for anyone who wants to watch it work end to end.

## What is on screen

Six sections, reached from the control at the top. A patient name anywhere in
the app is a link to that patient's brief.

| Section | What it shows | Model involved? |
|---|---|---|
| **Panel review** | The supervisor's ranked shortlist (capped at twelve), its report with every number cited to a finding id, and the findings store beneath it. A free-text steer ("I am running a diabetes clinic on Thursday") changes what the specialists are briefed to look for, not just how the answer is worded. | Yes — supervisor, three specialists, an extractor |
| **Patient brief** | Everything on file for one patient, assembled deterministically, with a narrative on top and a reconciliation of the brief against the clinician's own note. | Yes — one writer, one reconciler with no tools |
| **Priority score** | A transparent per-patient score (burden / instability / neglect) that decomposes into named contributions a clinician can disagree with line by line. Compared against the review's ranking as convergent validity. | No — a model wrote the weights once, offline, into `score_weights.json`; code applies them |
| **Pre-flight data audit** | Clinical plausibility of the extract, run before any patient logic. Splits *impossible* values (a defect whatever the population) from *population-dependent* rates (a question for whoever supplied the data). | Yes — an auditing agent, cached |
| **What the model may select** | The controlled vocabulary — 30 diagnoses, 42 drug classes, 36 analytes — that every agent must choose from. A code absent from the dataset is refused at registration. | No |
| **Guidelines used** | The four-recommendation guideline pack `guideline_concordance` checks against, with its disclaimer and the deterministic population SQL behind each. | No |

An earlier protocol-screening view is off the nav but still reachable at
`?view=Screen%20a%20protocol`; it is not part of the presented product.

## How it works

```
Pre-flight audit ──────────────────────────────┐   (runs first, judges the extract)
                                               ▼
Supervisor ── briefs all three at once ──► data_integrity · guideline_concordance · followup
                                               │  (concurrent; they cannot read each other)
                                               ▼
                                     Finding extractor: prose → typed rows
                                               ▼
Floor (9 computed findings) ──────────► Findings store ──► Pre-visit brief ──► Note reconciliation
                                               │
                              DuckDB · tools.py builds every query · 0 model-written SQL
```

- **The floor.** Nine findings whose absence would harm someone — drug
  monitoring mismatches, hypertensive crisis, stale orders with a live
  indication, HFrEF therapy gaps — are computed and seeded into the store
  *before* any agent runs. The review refuses to start if the count is wrong.
  A five-run eval showed agents reliably surface what a deterministic sweep
  backs and unreliably surface what competes for a reporting slot; the floor
  removes the choice.
- **Structured hand-offs.** Findings travel between agents as typed rows, not
  prose. A finding naming a patient who is not on the roster is refused; one
  whose evidence discusses a patient it does not list is flagged; the same
  problem worded twice is one finding.
- **Code checks the report.** Any high-severity finding the supervisor's
  narrative fails to cite is raised above the report. An empty report is
  announced, not swallowed.
- **Notes are data, never instructions.** The reconciliation agent holds no
  tools and is bound to a fixed output schema, so a note cannot make it act. A
  planted prompt-injection note (`fixtures.py`) is reported as a high-severity
  conflict rather than obeyed.
- **Every control ships with a fault that trips it** (`faults.py`), because a
  control nobody has watched fail is not a control.

The full architecture, the controls and their failure modes, and the A/B
against a single agent are in [`prototype/README.md`](prototype/README.md).

## The caches, and when they go stale

Three artifacts are saved to disk so the demo never waits on a model: the
panel review (`prototype/panel_cache.json`), the pre-visit briefs
(`prototype/brief_cache.json`) and the audit (`prototype/preflight_findings.json`).
Each is real output from a real run, not a fixture.

The review and the briefs carry a **fingerprint** of the database plus the
eight modules their behaviour depends on — `panel.py`, `floor.py`, `tools.py`,
`brief.py`, `reconcile.py`, `guidelines.py`, `vocab.py`, `rules.py`. Edit any
of them and the saved review is shown with a *stale* warning and the briefs are
dropped, because a cache that is merely stale looks exactly like a current
result. Regenerate with:

```bash
./.venv/bin/python -m prototype.panel_cache     # ~3 min, ~$0.15–0.30
./.venv/bin/python -m prototype.preflight       # the data audit
```

Briefs regenerate themselves on the next click, about forty seconds each.

## Verifying it

```bash
./.venv/bin/python -m pytest tests/ -q          # 70 tests, ~7s, no model
./.venv/bin/python evals/stability.py 5         # ~25 min, ~$1.40
./.venv/bin/python -m prototype.score           # the priority score, top ten
```

The tests re-derive every tool's answer in independent SQL and pin every
figure quoted on the slides — the note audit, the lab-table scope, the
blood-pressure correlation, the statin gap. The stability eval runs the review
N times and reports ten invariants, coverage of independently verified facts,
and how much the shortlist and its order move between runs (Jaccard and
Spearman).

Defects that are diagnosed and deliberately not fixed, each with its cost, are
in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md), alongside the behaviour that looks
wrong and is not.

## Repository map

| Path | What it is |
|---|---|
| `prototype/` | The app and the agents. See its [README](prototype/README.md). |
| `build_db.py` | Loads `data/*.csv` into `ehr.duckdb` and defines the views. |
| `tests/test_tools.py` | The regression suite. |
| `evals/stability.py` | The repeated-run eval; `stability_results.json` and `steer_test.json` are its last outputs. |
| `Panel Review — take-home reasoning.pdf` | The four presentation slides, exported from the published deck on 2026-09-21. The tests pin every figure on them. |
| `KNOWN_ISSUES.md` | Diagnosed, reproducible, not fixed. |
| `CLAUDE.md` | Operational notes that are easy to get wrong — the file watcher, the database lock, why `curl` cannot verify the UI. |

---

# The data

Everything below is a measured property of the extract. Each figure is pinned
by a test in `tests/test_tools.py`, and several of them are what the product's
design follows from.

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

### The two lab tables are different datasets

These are Epic Clarity names, and in Clarity `ORDER_RESULTS` is a child of
`ORDER_PROC`, so the join is expected to work. It does not, and the reason is
scope rather than corruption:

|  | `order_proc_awv` | `order_results` |
|---|---|---|
| Window | encounter −3 to 0 days | encounter −727 to +345 days |
| Answers | "what did we order at this visit, and what came back?" | "what is this patient's lab trend?" |
| Result data | inline on the row | inline + reference ranges |
| Pending | 120 of 588 unresulted (74 actionable) | n/a |

Only 127 of the 1,288 results fall within two months of the visit; the parent
orders for the rest were never in the extract. `ORDER_PROC_ID` values are
disjoint (0 of 1,288 match). The fallback join a reasonable person then reaches
for — patient plus analyte — returns rows, just the wrong ones: it falsely
resolves **85 of the 120** pending orders against a result from a different
year. So the tables are kept independent, each keyed to the encounter, and
"never resulted" is read from a single row.

Of the 120 unresulted orders, **34 are superseded** by a later result for the
same analyte and 12 monitor a condition the patient does not carry. **74 are
genuinely outstanding.** `prototype.panel._pending_orders()` does this triage
deterministically.

## How it is represented in DuckDB

**14 tables** — one per CSV, loaded verbatim with full-file type inference
(`sample_size=-1`), so dates land as `DATE`/`TIMESTAMP` rather than text.
Columns that are 100% empty in the source are explicitly typed in `FORCE_TYPES`
so they don't silently become `VARCHAR`.

**Views** — the analyst-facing layer:

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

Across 153 notes, **zero** diagnoses and **zero** medications appear that are
not already rows in the tables, and every parsed field agrees with
`ip_flwsht_meas`, `pat_enc_dx` and `order_med` at 100% — because the notes were
rendered from them. Strip the structured values and what remains is exactly
four fixed sentences, one per template, with no variation. There is no
negation, hedging, social history, symptom or exam narrative to find.

So there is nothing to *extract*. What the four sentences do carry is a claim
that care was delivered — "labs ordered per guideline intervals" — which no
column records. That is why the product uses the notes as a **check on** the
brief rather than a source for it: the reconciler finds three encounters where
the note claims labs were ordered and not one ever resulted.

## Known limits of the synthetic data

Worth knowing before you build a demo on it:

- **Systolic and diastolic do not co-vary, so many blood pressures are not
  physiologically possible.** They correlate at r = −0.37 where real pressures
  run +0.5 to +0.7, and the pulse-pressure distribution is flat from 20 to 119
  rather than peaked — so this is the shape of the whole column, not entry
  error in a tail of it. How the generator produced that is not knowable from
  the data. Pulse pressure (systolic − diastolic) should sit roughly between 20
  and 100 mmHg. **27 of 100 patients** have a latest reading outside that, and
  `Cervantes, Stephen` reads **108/111** — diastolic above systolic, which
  cannot occur.

  | reading | pulse pressure | |
  |---|---|---|
  | `Cervantes, Stephen` 108/111 | **−3** | impossible |
  | `Bender, Jessica` 111/104 | 7 | not a blood pressure |
  | `Ware, Cassandra` 180/61 | 119 | implausibly wide |

  This undermines any query keyed on a BP threshold. Screen on pulse pressure
  before treating a BP as usable. `prototype.panel.blood_pressure_staging()`
  does this and returns implausible readings separately from staged ones.

  Genuine severity is also rarer than it looks: only **5 of 100** patients reach
  hypertensive crisis (>180 or >120) on ACC/AHA 2017 categories.
- **Prescribing is statistically independent of provider specialty**, so
  scope-of-practice analysis is not supportable. A chi-square across specialty ×
  drug class gives **χ²/df = 0.94**, where 1.0 is what random assignment
  produces. Statins are spread across all seven specialties; SSRIs are
  prescribed more often by Internal Medicine and Cardiology than by Psychiatry;
  one PA prescribes 32 of the 42 drug classes. Any "provider prescribing outside
  their scope" report built on this would be confident, specific and fabricated.
  Use the provider fields for routing — *who do I chase about this order* — not
  for judgement.
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
  silently halves most cohorts. Roll up to the 3-character category first, as
  `v_diagnosis` consumers and `score.py` do.

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
  have no material here. Undertreatment does (25 of 28 diabetics are not on a
  statin).
- **No medication is ever recorded as stopped, so you cannot tell what a patient
  is currently taking.** `START_DATE` is populated on all 522 orders, but
  `END_DATE` and `DISCON_TIME` are **100% null** and `ORDER_STATUS_C_NAME` is
  `Active` on every row. A switch and a combination are therefore
  indistinguishable without reading the dates.

  This matters because the naive read is alarming and wrong. Twenty patients
  hold two or three agents of one class, which looks like dangerous duplicate
  therapy — one patient on three DOACs, two on three SSRIs. But the start dates
  within a duplicated class are **113 to 1376 days apart, mean 814 days**. The
  three DOACs began in March 2023, July 2024 and December 2025. That is
  sequential switching, which is ordinary care; it only reads as triple
  anticoagulation because nothing was ever closed out.

  The real defect is the missing discontinuation data, not the patients. Claims
  of the form "this patient is on X" are unreliable here. Claims of the form
  "this patient has never had X" survive, which is why the guideline gaps are
  findings and duplicate-therapy claims are not.
- **Some prescribing rates are implausible.** PCSK9i appears in 16 of 100
  patients, half of them not on a statin; real-world use is 1–2% of a lipid
  population and near-always statin-refractory. Absurd for general practice,
  ordinary in a refractory-lipid clinic, and nothing in the extract says which
  this is — so the audit reports it as a question about the population, not a
  defect.
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
restate the structured tables, which is what
`test_the_notes_hold_no_clinical_fact_the_tables_do_not` verifies.

### What's clean

All 19 other foreign-key relationships resolve with zero orphans. Abnormal flags
agree with reference ranges in 100% of `order_results` rows. RxNorm coverage is
100%. Every encounter has exactly one note and one full set of vitals.
