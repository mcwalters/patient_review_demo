"""Deliberate faults, for proving the findings-store controls actually fire.

`reconcile.py` already argues the case and ships `fixtures.py` so the
note-reconciliation control can be watched catching something:

    A safety check nobody has ever seen trigger is not a safety check.

The three controls added to `Findings` were held to a lower bar than that. Two
of them have never fired outside a unit test, and one of the two -- the refusal
path through `_consult` -- had a real bug in it precisely because nothing had
ever executed it end to end: the return value of `record()` was discarded, so a
refused row was still counted as filed.

Nothing here is data. Each fault injects one specific bad input into a live run
so the control that exists to catch it can be seen doing so.
"""
from __future__ import annotations

from typing import Callable

# Each fault: what it plants, and the control it should trip.
FAULTS: dict[str, tuple[str, str]] = {
    "unknown-patient": (
        "a finding naming a patient who is not in the patient table",
        "Findings.record refuses the row and names the unrecognised patient",
    ),
    "drop-high-finding": (
        "a high-severity finding the report is made to omit",
        "uncited_high_severity raises it above the narrative",
    ),
    "starve-floor": (
        "a data layer that answers with only part of the floor",
        "seed_floor refuses to start a review that would look like a quiet week",
    ),
}


def inject_unknown_patient(findings) -> dict:
    """Record a finding for somebody who does not exist.

    Mimics the thing the control is for: a specialist naming a patient the
    roster has never heard of, which the UI would otherwise turn into a link to
    that person's brief.
    """
    return findings.record(
        "followup",
        headline="Overdue INR for a patient who is not in this panel",
        patients=["Nobody, Fictional"],
        category="drug monitoring mismatch",
        severity="high",
        evidence="planted by prototype.faults -- not a real finding",
        recommended_action="none; this row should never have been stored")


def drop_high_finding(report: str, findings: list[dict]) -> tuple[str, str]:
    """Strip one high-severity citation out of a report.

    The supervisor caps its shortlist at twelve of a hundred and picks the cut
    itself, so a high finding going unmentioned is an ordinary outcome rather
    than an exotic one. This makes it happen on demand.

    Removing one citation is not enough to hide anything: the floor files an
    HFrEF gap per drug, so striking F09 leaves F07 and F08 citing the same
    category about the same people and the finding is still, correctly, not
    missing. To make content actually vanish the fault has to remove every
    citation that would cover it.

    Returns (report with the citations removed, the finding id targeted).
    """
    highs = [f for f in findings if f.get("severity") == "high"]
    if not highs:
        return report, ""
    victim = highs[-1]
    cat = victim.get("category")
    doomed = {victim["finding_id"]} | {
        f["finding_id"] for f in findings
        if cat and cat != "other" and f.get("category") == cat}
    for fid in doomed:
        report = report.replace(fid, "[redacted by prototype.faults]")
    return report, victim["finding_id"]


def starve_floor(monkeypatch_target, keep: int = 4) -> Callable:
    """Return a compute_floor that answers with only part of the floor.

    Stands in for the failure the tripwire exists for: a query that silently
    returns fewer rows than it should, which is indistinguishable in the output
    from a panel with nothing wrong.
    """
    real = monkeypatch_target

    def half_answer():
        return real()[:keep]

    return half_answer
