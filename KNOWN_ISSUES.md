# Known issues

Diagnosed, reproducible, and not fixed. Each one says what it would take, so
the cost of the fix is on the record next to the defect.

---

## The floor rates a whole category high without grading inside it

`_stale_orders_with_indication` in `prototype/floor.py` assigns
`severity: "high"` to every stale order where the patient still has the
condition the test monitors. There are eight, and they are not alike:

```
11mo  Black, Tyler         Urine Albumin/Creat Ratio   kidney damage screen in diabetes
11mo  Taylor, Jonathan     Fasting Glucose             glycaemic control
11mo  Rogers, Jessica      LDL Cholesterol
 9mo  Williams, Heather    Triglycerides
 8mo  Mcdaniel, Dana       INR / PT
 6mo  Vargas, Kimberly     Total Cholesterol
 6mo  Mckinney, Scott      Triglycerides               routine lipid monitoring
 6mo  Juarez, Rebecca      Triglycerides               routine lipid monitoring
```

A six-month-overdue triglycerides is filed at the same severity as an
eleven-month-overdue kidney screen in a diabetic. The supervisor then does the
grading the floor refused to do, bundles the tail as non-urgent, and cites the
grouped row — which is correct behaviour, and which `uncited_high_severity`
reports as a finding that failed to reach the narrative.

So the visible symptom is a warning above the report blaming the supervisor for
a judgement the floor got wrong. The banner has been reworded to describe the
disagreement rather than assign fault; the underlying grading has not changed.

**Root cause.** The classification already exists, as prose, in the followup
agent's instruction (`prototype/panel.py`, the FOLLOWUP_INSTRUCTION): a test is
high-stakes if it is "a potassium on an ACE inhibitor plus a diuretic, a kidney
screen in diabetes, a level for a narrow-therapeutic-index drug". That is a
lookup table written as a sentence, left in a prompt, in a codebase whose
governing rule is that no model computes what code can compute. The floor
cannot apply it because the floor cannot read a prompt.

**The fix.**

1. Move the classification into code: organ-damage screening against an active
   condition, glycaemic control in diabetes, narrow-therapeutic-index drug
   levels, electrolytes on interacting drugs. Everything else, including all
   four lipid analytes, is routine.
2. Split the floor finding in two — high for the first group, medium for the
   rest. `EXPECTED_FLOOR` goes 9 to 10.
3. Return the flag from `_pending_orders` and delete the prose version from
   FOLLOWUP_INSTRUCTION, or the prompt and the floor will disagree.
4. Downstream: one hardcoded `9` in `tests/test_tools.py`, two mentions of
   "nine findings" in the slides, regenerate `prototype/panel_cache.json`
   (~4 min), re-run `evals/stability.py 5` (~25 min, ~$1.40).

**Why it is not done.** Step 4. Slide 4's numbers — Jaccard 0.90, eight
invariants holding 5 of 5, $0.28 a run — were all measured against the
nine-finding floor, and the floor is roughly a third of a run's findings.
Regrading it invalidates them, and re-measuring the night before a demo means
presenting whatever comes back with no time to investigate a regression.

---

## The floor tells you to chase an order it also says should not exist

Mcdaniel, Dana's INR / PT appears twice in the same floor:

- `drug monitoring mismatch` — "INR / PT ordered for a patient not on the drug
  it monitors". The patient is on a DOAC; an INR is the wrong test.
- `stale order with live indication` — eight months open, chase it.

Both are technically true: the condition is live, the test is wrong for the
drug. Presented side by side they are incoherent, and a nurse acting on the
second would be chasing a result that means nothing. The categories need to
suppress each other — an order already flagged as wrongly placed should not
also appear as one to follow up.

---

## The pre-flight audit has no floor and nothing measures it across runs

The panel review is guaranteed nine computed findings and measured over five
runs. The audit has neither, and it shows: `Digoxin levels present without a
prescription` was in the cache before `check_physiologic_limits` was added and
has been absent from every run since. It was noticed only because an old cache
happened to be diffed against a new one.

Four audit runs today were judged by reading them, one at a time, which is the
failure `evals/stability.py` exists to prevent in the other half of the system.
It needs the same treatment: a small guaranteed set of findings that must
always appear, and a repeated-run harness.

---

## Nothing validates the ranking

The largest uncovered hazard in the system, and the one with no control at all.

The product is an ordering. A panel manager works down from the top and stops
at about fifteen, so being on the list at position 3 and position 30 are
different outcomes, and nothing here checks that the order is right. Every
control in `prototype/` verifies that a *fact* is true. None verifies that a
*priority* is.

The floor's blanket grading, above, is what this gap looks like in practice: a
whole category rated high, wrong, for weeks, noticed only because it collided
with a different control.

The temptation is to answer "a clinician-labelled gold set" and stop. That is
expensive, slow, and lets the problem sit until someone else solves it. Most of
this is reachable without a clinician.

### Without a clinician

**1. Derive severity instead of asserting it.** Severity is currently a
hardcoded string per floor category. A rating nobody can trace to a rule cannot
be audited, and cannot be argued with. Replace it with an explicit rubric --
consequence class of the test, how far overdue, whether the condition is active
-- so that a rating is reproducible and *contestable*. A clinician disagreeing
with one rubric is a far better conversation than a clinician disagreeing with
thirty individual ratings.

**2. Dominance tests.** If patient A carries every finding patient B carries,
plus one more of equal or greater severity, A must not rank below B. A real
safety property, checkable with no ground truth, and the standard approach for
systems where no gold answer exists. With the rubric from (1) it catches the
floor bug directly.

**3. Threshold sensitivity.** Move a systolic from 179 to 181 and the ranking
must move in the expected direction. Confirms the order responds to clinical
facts rather than to phrasing.

**4. Rank stability.** DONE. `evals/stability.py` now reports pairwise Spearman
correlation over the shortlist ordering alongside Jaccard. Jaccard only ever
measured whether the same *people* came back; two runs can agree completely on
the set and disagree on who to see first. Order stability is not order
correctness, but it is a precondition for it.

### The cheap clinician ask

Not a gold set. **Pairwise preference**: show a clinician 25 pairs of patients
and ask which needs attention first. Fifteen minutes yields a partial order to
measure agreement against, and people are far more reliable comparing two
things than ranking a hundred. Second option, also cheap: have them pick their
own fifteen from the panel and compare the overlap with the system's twelve.

Only after those is a labelled gold set worth the cost, and by then it would be
measuring something the cheaper instruments had already pointed at.

---

## Known-good behaviour that looks like a bug

Recorded so nobody "fixes" it.

- **The reconciliation control reports no factual conflicts.** The notes are
  generated from the same tables the brief is built from — 153 of 153 fields
  agree — so there is nothing to disagree about. It finds three care-delivery
  conflicts, which is the real result. See the docstring in `reconcile.py`.
- **The PCSK9 prescribing rate is not flagged as a defect.** 16% is absurd for
  general practice and ordinary in a refractory-lipid clinic, and nothing in
  the extract says which this is. It is reported as a question for whoever
  supplied the data.
