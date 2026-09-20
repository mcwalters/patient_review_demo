"""The last panel review, saved so the demo does not open on a spinner.

A run measures about 160s with the specialists consulted concurrently. That is
still minutes of a 45-minute session spent watching nothing, and it is the
first thing anyone sees. `preflight.py` already solved this for the audit; this is the same move
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

from .tools import artifact_fingerprint

CACHE = Path(__file__).parent / "panel_cache.json"
BRIEFS = Path(__file__).parent / "brief_cache.json"


def save(goal: str, result: tuple, rejected: list[dict] | None = None) -> Path:
    """Persist one completed review. `result` is review_async's 4-tuple."""
    report, trace, findings, usage = result
    CACHE.write_text(json.dumps({
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": artifact_fingerprint(),
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


def is_stale(blob: dict) -> bool:
    """Was this artifact built from a different database or different code?

    An artifact with no fingerprint predates the check and is treated as stale,
    because "we do not know" and "it is current" are not the same answer.
    """
    return blob.get("fingerprint") != artifact_fingerprint()


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


# ------------------------------------------------------------------ briefs
# A brief is a model call plus a reconciliation pass -- about forty seconds
# warm, and noticeably longer on a cold server.
# It was cached in st.session_state, which is per browser session: restart the
# server or open a second tab and every patient costs that again.
# On a fixed extract the answer never changes, so it belongs on disk.


def load_briefs() -> dict[str, tuple]:
    """Every saved brief. Unreadable or malformed means empty, never an error."""
    if not BRIEFS.exists():
        return {}
    try:
        blob = json.loads(BRIEFS.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(blob, dict):
        return {}
    # Briefs built from other code are dropped rather than shown. Unlike the
    # panel review there is no cost to regenerating one -- it is under a minute
    # and only the patient being looked at -- so silently serving an old brief
    # buys nothing and risks showing a stale clinical summary.
    now = artifact_fingerprint()
    return {k: (v.get("narrative", ""), v.get("pack", {}))
            for k, v in blob.items()
            if isinstance(v, dict) and v.get("fingerprint") == now}


def save_brief(patient: str, narrative: str, pack: dict) -> None:
    """Add one brief to the file, keeping whatever is already there.

    A brief that failed is not cached -- pack carries an "error" key and the
    next click should try again rather than replay the failure for ever.
    """
    if not narrative.strip() or "error" in pack:
        return
    blob = {}
    if BRIEFS.exists():
        try:
            blob = json.loads(BRIEFS.read_text())
        except (json.JSONDecodeError, OSError):
            blob = {}
        # Same guard as load_briefs. Without it, writing over a corrupt file
        # raised instead of replacing it -- the read path tolerated the wreckage
        # and the write path fell over on it.
        if not isinstance(blob, dict):
            blob = {}
    blob[patient] = {"narrative": narrative, "pack": pack,
                     "fingerprint": artifact_fingerprint(),
                     "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    BRIEFS.write_text(json.dumps(blob, indent=1, default=str))


def main() -> None:
    """Run a review and save it. This is what regenerates the demo's opening screen."""
    from .panel import Findings, review_async
    import asyncio

    goal = "Anything that needs attention this week."
    findings = Findings()
    print("running a panel review to cache … (two to three minutes)")
    result = asyncio.run(review_async(goal, verbose=True, findings=findings))
    path = save(goal, result, findings.rejected)
    report, trace, rows, usage = result
    print(f"\nsaved to {path}")
    print(f"  {len(rows)} findings, {len(trace)} tool calls, "
          f"${usage.get('usd', 0):.2f}, {usage.get('wall_clock_seconds', 0)}s")


if __name__ == "__main__":
    main()
