"""The last panel review, saved so the demo does not open on a spinner.

A run measures 311s average over the five-run eval. In a 45-minute session that
is a seventh of the time watching nothing, and it is the first thing anyone
sees. `preflight.py` already solved this for the audit; this is the same move
for the review.

The saved run is real output from a real run, not a fixture -- regenerate it
with `python -m prototype.panel_cache` and it is whatever the agents produced
that time, refusals and uncited findings included. The live button stays, so
the run can still be watched end to end by anyone who wants to see it work.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

CACHE = Path(__file__).parent / "panel_cache.json"


def save(goal: str, result: tuple, rejected: list[dict] | None = None) -> Path:
    """Persist one completed review. `result` is review_async's 4-tuple."""
    report, trace, findings, usage = result
    CACHE.write_text(json.dumps({
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "goal": goal,
        "report": report,
        "trace": trace,
        "findings": findings,
        "usage": usage,
        "rejected": rejected or [],
    }, indent=1))
    return CACHE


def load() -> dict | None:
    """The saved run, or None. Never raises -- a bad cache just means no cache.

    A demo that dies on its own cache file is worse than one that takes four
    minutes, so anything unreadable here is treated as absent.
    """
    if not CACHE.exists():
        return None
    try:
        blob = json.loads(CACHE.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not all(k in blob for k in ("report", "trace", "findings", "usage")):
        return None
    return blob


def as_session_value(blob: dict) -> tuple:
    """Shape the saved run the way the UI unpacks a live one."""
    return blob["report"], blob["trace"], blob["findings"], blob["usage"]


def age_phrase(blob: dict) -> str:
    """"3 minutes ago" / "2 days ago", for the banner over a saved run."""
    try:
        when = datetime.fromisoformat(blob["saved_at"])
    except (KeyError, ValueError):
        return "at an unknown time"
    secs = (datetime.now(timezone.utc) - when).total_seconds()
    for limit, div, unit in ((3600, 60, "minute"), (86400, 3600, "hour"),
                             (float("inf"), 86400, "day")):
        if secs < limit:
            n = max(1, int(secs // div))
            return f"{n} {unit}{'s' if n != 1 else ''} ago"
    return "a while ago"


def main() -> None:
    """Run a review and save it. This is what regenerates the demo's opening screen."""
    from .panel import Findings, review_async
    import asyncio

    goal = "Anything that needs attention this week."
    findings = Findings()
    print(f"running a panel review to cache … (expect ~5 minutes)")
    result = asyncio.run(review_async(goal, verbose=True, findings=findings))
    path = save(goal, result, findings.rejected)
    report, trace, rows, usage = result
    print(f"\nsaved to {path}")
    print(f"  {len(rows)} findings, {len(trace)} tool calls, "
          f"${usage.get('usd', 0):.2f}, {usage.get('wall_clock_seconds', 0)}s")


if __name__ == "__main__":
    main()
