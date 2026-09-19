# Eligibility screening from a free-text protocol

A care program or study defines who qualifies in prose. Today somebody reads
that and hand-builds a query, or works a chart list. This takes the criteria as
written, grounds every clinical concept in the dataset's own vocabulary, and
returns the qualifying patients with per-patient evidence and exclusion reasons.

Gemini 2.5 Pro on Vertex AI, via **application default credentials** — no API
key to manage.

```bash
pip install -r ../requirements.txt
gcloud auth application-default login      # if ADC is not already set up
streamlit run app.py
```

---

## Why an LLM is load-bearing

The criteria are arbitrary prose, so there is no fixed schema to compile
against. Three things have to happen that no rules table can cover:

1. **Parse criteria nobody wrote a parser for.** The protocol is free text and
   unknown in advance.
2. **Expand clinical concepts.** This dataset splits conditions across
   near-duplicate ICD-10 codes. "Diabetes" is `E11.9` **and** `E11.51` **and**
   `E11.65` — 7 patients versus 28. "Hypertension" is `I10` **and** `I11.9` —
   21 versus 46. On the sample protocol, getting that expansion right moved the
   cohort from **7 eligible to 12**.
3. **Judge plausibility.** The pre-flight linter recognises that a PCSK9i rate
   of 16% or an SpO2 of 126% is not credible. That is medical knowledge; the
   dataset contains no formulary, interaction table or guideline library.

## Architecture

```
protocol (free text)
   │
   ▼
ADK agent ── search_diagnoses / search_medications / search_analytes / search_vitals
   │              └─ returns the FULL dataset vocabulary, with keyword hints
   │
   ├── define_criterion(...)   structured values only, validated against the vocabulary
   │
   ▼
tools.py ── compiles and runs every query ── DuckDB
   │
   ▼
cohort: eligible / needs review / excluded, with evidence per criterion
```

**The agent never writes SQL.** It selects from a fixed vocabulary and registers
criteria as structured values; `tools.py` builds every query. A code the dataset
does not contain is rejected at registration, so a hallucinated ICD-10 code
cannot reach the database.

| File | Role |
|---|---|
| `vocab.py` | The controlled vocabulary present in the data — 30 diagnoses, 42 drug classes, 36 analytes, 7 vitals |
| `tools.py` | Every tool the agents call. All SQL lives here. Validation, three-state evaluation, physiologic backstop |
| `screener.py` | The eligibility agent |
| `preflight.py` | The clinical plausibility linter |
| `app.py` | Streamlit demo |

## Styling

The UI uses the Qualified Health brand, extracted from `PPT Template.pptx`
rather than eyeballed — `prototype/theme.py` and `.streamlit/config.toml` carry
the values from the deck's own `ppt/theme/theme1.xml`:

| Token | Value | Used for |
|---|---|---|
| `accent1` | `#0A3D63` | navy — headings, buttons, primary |
| `accent2` | `#058BE8` | bright blue — links, hover |
| — | `#518AE1` | the lighter blue of slide titles |
| `dk2` | `#5C6C80` | slate body text |
| `accent4` | `#C1CFD8` | borders and rules |
| `lt2` | `#F9FCFF` | page background |
| `accent6` | `#ED8D02` | orange — medium-severity warnings |
| *added* | `#B42318` | red for high-severity findings |

Type is Helvetica Neue (Arial fallback), as used on the slides. The header
reproduces the deck's two-tone title pattern — light-blue phrase, thin rule,
navy phrase — and the footer reproduces the content-slide furniture with the
logo lifted from `ppt/media/`.

**The red is an addition.** The template has no red at all; a clinical safety UI
needs one, so it is introduced only for high-severity findings and used nowhere
else.

## A correction worth reading

An early finding of this prototype was "clinically dangerous triple
anticoagulation — one patient on three DOACs at once." It was wrong, and it took
a direct question to catch it: *is there timing information that would show
these are sequential rather than concurrent?*

There is. `START_DATE` is populated on all 522 orders. `END_DATE` and
`DISCON_TIME` are entirely null and every order reads `Active`. Those three
DOACs started in March 2023, July 2024 and December 2025 — nearly three years
apart. It is a switch, recorded by a system that never closes anything out.

The corrected finding is both more accurate and more damning: **this extract
cannot tell you what any patient is currently taking.** `cohort_statistic(
"duplicate_therapy")` now returns the start-date spread and a warning, there is
a `medication_timeline` tool, and the data-integrity agent is told to check the
dates before calling anything duplicate therapy.

The lesson generalises past this dataset: an alarming finding that no one
questions is the most dangerous output the system can produce, because its
alarm is what stops people checking it.

## Safety properties

**Hallucinated codes cannot reach SQL.** Registration rejects any value absent
from the dataset and tells the agent to search again.

**Missing is not normal.** Every criterion evaluates to met, not_met, or
**unknown**. Unknown routes the patient to review — it is never silently a pass.
On the sample protocol 46 of 100 patients have no eGFR, and 9 land in review
rather than being quietly cleared.

**The plan is reviewed before the cohort.** The UI shows the criteria, the codes
each was bound to, and the agent's rationale *above* the patient list. An
over-broad exclusion is invisible in a list of patients and obvious in the plan.

**Numbers come from tools, never from model prose.** In testing, the linter's
qualitative judgments were all correct but 2 of ~15 quantitative claims drifted
(it reported 8 digoxin patients where there are 6). Anything numeric shown to a
user is computed, not generated.

**Deterministic warnings the model cannot suppress:**
- an exclusion removing >15% of the panel (over-expansion wrongly denies care)
- a diagnosis criterion using part of an ICD-10 3-character category but not all
  of it (the fragmentation bug)
- a criterion with no data for >30% of patients
- any eligible patient holding a physiologically impossible lab value

## Where it can be wrong

Both failure modes below were found by running it, not by planning:

- **Over-expanding an exclusion.** The agent bound "heart failure" to the three
  `I50` codes *plus* `I11.9` (Hypertensive Heart Disease), which is not heart
  failure. As an exclusion that wrongly denied 25 patients. Mitigated by the
  >15% warning and by plan review — not eliminated.
- **Under-expanding an inclusion.** Correcting the above caused the opposite
  error: hypertension bound to `I10` alone, losing 25 patients. Mitigated by the
  ICD-10 sibling warning.

The honest summary: concept expansion is the model's job, it gets it wrong in
both directions, and the design surfaces the judgment for human approval rather
than trusting it. A reviewer approving the plan is the control, not a
disclaimer.
