"""Tier 2: how stable is the panel review across repeated runs?

Regression checks prove a run does not do the wrong thing. They say nothing
about whether the same panel produces the same answer twice, which is the
question a clinician actually asks: does it find everything, every time?

This runs the review N times and measures three things.

  invariants     the nine properties that must hold on every run
  coverage       for facts we independently know to be true, how many runs
                 surfaced them -- the closest thing to a recall measure that
                 is available without exhaustive clinical review
  stability      how much the surfaced patient set moves between runs,
                 and separately how much the ORDER of it moves

    ./.venv/bin/python evals/stability.py 5
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prototype.floor import EXPECTED_FLOOR  # noqa: E402
from prototype.panel import (  # noqa: E402
    Findings, review, uncited_high_severity)

GOAL = "Who on this panel needs my attention this week? I can review about a dozen."
OUT = Path(__file__).parent / "stability_results.json"

# Facts verified directly against the database earlier in development. Each is
# true of this extract regardless of what any run says, so "did this run surface
# it?" is a fair coverage question rather than a matter of taste.
CANARIES = {
    "Padilla INR on a DOAC": lambda f: "Padilla, Elizabeth" in f["patients"]
        and re.search(r"inr", f["headline"] + f["evidence"], re.I),
    "Schwartz 11-month potassium": lambda f: "Schwartz, Mary" in f["patients"]
        and re.search(r"potassium", f["headline"] + f["evidence"], re.I),
    "Stein missing HFrEF therapy": lambda f: "Stein, Larry" in f["patients"]
        and re.search(r"hf|heart failure|beta.?block|sglt|ace|arb|arni",
                      f["headline"] + f["evidence"], re.I),
    "digoxin level, nobody on digoxin": lambda f: re.search(
        r"digoxin", f["headline"] + f["evidence"], re.I),
    "physiologically impossible labs": lambda f: re.search(
        r"impossible|implausible", f["headline"], re.I),
    "no medication stop dates": lambda f: re.search(
        r"discontinuation|never marked as stopped|stop date",
        f["headline"] + f["evidence"], re.I),
}

_ROSTER = Findings._roster()

# Each takes (findings, report). The report is here because "did a high-severity
# finding actually reach the narrative" is not answerable from the store alone,
# and it was the control this harness could not see: main() discarded the report
# and kept only the findings, so uncited_high_severity had never been measured
# across runs even though the shortlist caps at twelve patients out of a hundred
# and the supervisor picks the cut itself.
INVARIANTS = {
    "no invented BP severity": lambda fs, report: not any(
        re.search(r"severe hypertension", f["headline"] + f["evidence"], re.I) for f in fs),
    # The exculpation is looked for in the headline AND the evidence, because
    # it can be in either and this once penalised the right answer: a finding
    # headed "Medication discontinuation data is missing, creating appearance
    # of duplicate therapy" tripped the check that exists to produce exactly
    # that framing. Same mis-specification as the duplicate-headline invariant
    # in c33170e -- reading one field when the claim can live in two.
    "no concurrent-duplicate-therapy claim": lambda fs, report: not any(
        re.search(r"(duplicate|triple|concurrent) therapy", f["headline"], re.I)
        and not re.search(r"sequential|not concurrent|appears? as|apparent|artifact|"
                          r"stop date|discontinu",
                          f["headline"] + " " + f.get("evidence", ""), re.I)
        for f in fs),
    # Matches the dedup key in Findings._key -- headline AND patients. Checking
    # the headline alone was stricter than the rule it polices and failed a
    # correct run: "INR ordered for a patient not on the drug it monitors" is
    # one finding per patient, and three people had an open triglycerides order.
    "no duplicate findings": lambda fs, report: len(
        {(f["headline"].strip().lower(), frozenset(f.get("patients") or []))
         for f in fs}) == len(fs),
    "followup reports at most 9": lambda fs, report: sum(
        1 for f in fs if f["agent"] == "followup") <= 9,
    "every finding names its agent": lambda fs, report: all(f.get("agent") for f in fs),
    "every finding has evidence": lambda fs, report: all(
        f.get("evidence", "").strip() for f in fs),
    "every named patient is a real patient": lambda fs, report: all(
        p in _ROSTER for f in fs for p in (f.get("patients") or [])),
    "the floor is intact": lambda fs, report: sum(
        1 for f in fs if f["agent"] == "guaranteed") == EXPECTED_FLOOR,
    "every high-severity finding reaches the report": lambda fs, report: not
        uncited_high_severity(report, fs),
}


def shortlist_order(report: str, patients: list[str]) -> list[str]:
    """The patients of the shortlist, in the order the report raises them.

    The shortlist is numbered 1..12 in priority order, so first appearance is
    the rank. Prose, not structure, because the ranking only exists in prose --
    which is itself worth knowing.
    """
    seen = [(report.find(p), p) for p in patients]
    return [p for at, p in sorted(seen) if at >= 0]


def spearman(a: list[str], b: list[str]) -> float | None:
    """Rank correlation between two orderings, over the names common to both.

    Jaccard says whether the same people come back. It says nothing about
    whether they come back in the same order, and the product is an ordering:
    the nurse works down from the top and stops. Two runs can agree perfectly
    on the set and disagree on who to see first.

    No scipy -- Pearson on the ranks, which is what Spearman is.
    """
    common = [p for p in a if p in b]
    n = len(common)
    if n < 3:
        return None
    ra = {p: i for i, p in enumerate(a)}
    rb = {p: i for i, p in enumerate(b)}
    xs = [ra[p] for p in common]
    ys = [rb[p] for p in common]
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return round(num / den, 3) if den else None


def main(n: int) -> None:
    runs = []
    for i in range(1, n + 1):
        t0 = time.time()
        print(f"run {i}/{n} …", flush=True)
        report, trace, findings, usage = review(GOAL, verbose=False)
        patients = sorted({p for f in findings for p in (f.get("patients") or [])})
        runs.append({
            "run": i, "seconds": round(time.time() - t0, 1),
            "findings": len(findings), "tool_calls": len(trace),
            "patients": patients, "usage": usage,
            "invariants": {k: bool(fn(findings, report))
                           for k, fn in INVARIANTS.items()},
            "uncited_high": [f["finding_id"]
                             for f in uncited_high_severity(report, findings)],
            "canaries": {k: any(fn(f) for f in findings) for k, fn in CANARIES.items()},
            "shortlist_order": shortlist_order(report, patients),
            "report": report,
        })
        print(f"   {len(findings)} findings, {len(patients)} patients, "
              f"${usage.get('usd', 0):.2f}, {runs[-1]['seconds']:.0f}s", flush=True)

    print("\n" + "=" * 74)
    print(f"STABILITY OVER {n} RUNS")
    print("=" * 74)

    print("\nINVARIANTS (must hold every run)")
    for k in INVARIANTS:
        held = sum(r["invariants"][k] for r in runs)
        print(f"  {'PASS' if held == n else 'FAIL'}  {held}/{n}  {k}")

    missed = [(r["run"], r["uncited_high"]) for r in runs if r["uncited_high"]]
    if missed:
        print("\n  high-severity findings that never reached the narrative")
        for run_no, ids in missed:
            print(f"    run {run_no}: {', '.join(ids)}")

    print("\nCOVERAGE of independently verified facts")
    for k in CANARIES:
        hit = sum(r["canaries"][k] for r in runs)
        bar = "#" * hit + "." * (n - hit)
        print(f"  {hit}/{n}  {bar}  {k}")

    print("\nPATIENT-LEVEL STABILITY")
    seen = Counter(p for r in runs for p in r["patients"])
    always = [p for p, c in seen.items() if c == n]
    once = [p for p, c in seen.items() if c == 1]
    print(f"  distinct patients surfaced across all runs : {len(seen)}")
    print(f"  surfaced in every run                      : {len(always)}")
    print(f"  surfaced in exactly one run                : {len(once)}")
    if n > 1:
        jac = [len(set(a["patients"]) & set(b["patients"]))
               / len(set(a["patients"]) | set(b["patients"]))
               for a, b in combinations(runs, 2)]
        print(f"  pairwise Jaccard similarity                : "
              f"{statistics.mean(jac):.2f} (min {min(jac):.2f}, max {max(jac):.2f})")
        rho = [r for r in (spearman(a["shortlist_order"], b["shortlist_order"])
                           for a, b in combinations(runs, 2)) if r is not None]
        if rho:
            print(f"  pairwise rank correlation (Spearman)       : "
                  f"{statistics.mean(rho):.2f} (min {min(rho):.2f}, max {max(rho):.2f})")
            print(f"    same people in a different order is still a different "
                  f"worklist: the nurse works down from the top and stops.")
    print(f"\n  always surfaced: {', '.join(sorted(always)) or 'none'}")
    print(f"  one run only   : {', '.join(sorted(once)) or 'none'}")

    print("\nCOST AND LATENCY")
    tot = sum(r["usage"].get("usd", 0) for r in runs)
    print(f"  per run: ${tot / n:.2f} avg  |  "
          f"{statistics.mean(r['seconds'] for r in runs):.0f}s avg  |  "
          f"{statistics.mean(r['usage'].get('model_calls', 0) for r in runs):.0f} model calls")
    print(f"  findings per run: {statistics.mean(r['findings'] for r in runs):.1f} "
          f"(min {min(r['findings'] for r in runs)}, max {max(r['findings'] for r in runs)})")
    print(f"  total for this eval: ${tot:.2f}")

    OUT.write_text(json.dumps(runs, indent=1))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
