"""Reconcile a pre-visit brief against what the clinician actually wrote.

A brief is assembled from structured tables. The note is what a human recorded
at the visit. When those disagree, the brief is the one that is wrong, and
acting on it is how a patient gets told something untrue about their own care.
So this is a CONTROL, not a discovery feature: its job is to fail loudly, not
to find new things.

On this extract it finds nothing, and that is the correct result rather than a
disappointing one -- the notes are generated from the same tables the brief is
built from (153/153 field agreement, see the repo README), so there is nothing
to disagree about. A control that reports no conflicts on data with no conflicts
is working.

Which is exactly why it ships with `fixtures.py`: a handful of deliberately
altered notes, clearly labelled as fabricated, so the control can be shown
firing. A safety check nobody has ever seen trigger is not a safety check.
"""
from __future__ import annotations

import asyncio
import json
from typing import Literal

from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent

from .panel import MODEL, _run_agent
from .tools import connect


class Conflict(BaseModel):
    field: str = Field(description="what disagrees: a medication, a condition, a vital")
    brief_says: str
    note_says: str
    severity: Literal["high", "medium", "low"] = Field(
        description="high if acting on the brief could harm; low if cosmetic")
    why_it_matters: str


class Reconciliation(BaseModel):
    conflicts: list[Conflict]
    checked: list[str] = Field(
        description="what you compared, so a reader knows the scope of the check")


INSTRUCTION = """\
You compare a pre-visit brief, assembled from structured data, against the
progress note a clinician wrote at the patient's last visit.

Report only genuine disagreements about fact. Specifically:
  - a medication the note records that the brief does not, or the reverse
  - a condition named in one and absent from the other
  - a vital sign or value that differs between them
  - anything the note states about the patient's care that the brief contradicts

Do NOT report:
  - detail present in one and simply absent from the other, unless the absence
    changes what a clinician would do
  - differences of wording, ordering or rounding for the same underlying fact
  - the brief being more complete than the note; the brief draws on the whole
    record, the note describes one visit

The brief is the artefact under test. When they disagree, say what the note
says and treat the brief as the thing that may be wrong.

If there are no conflicts, return an empty list and say in `checked` what you
compared. An empty result is a real result; do not manufacture a finding to
appear thorough.
"""


def _notes_for(pat_id: str) -> list[dict]:
    con = connect()
    try:
        return [{"date": str(r[0]), "author": r[1], "text": r[2]} for r in con.execute(
            "SELECT n.ENTRY_TIME, n.AUTHOR_PROV_NAME, n.NOTE_TEXT FROM hno_info n "
            "WHERE n.PAT_ID = ? ORDER BY n.ENTRY_TIME DESC", [pat_id]).fetchall()]
    finally:
        con.close()


async def reconcile_async(pack: dict, narrative: str,
                          notes: list[dict] | None = None) -> dict:
    """Check a brief against the patient's notes. Returns the parsed result."""
    if "error" in pack:
        return {"error": pack["error"]}
    notes = notes if notes is not None else _notes_for(pack["patient"]["pat_id"])
    if not notes:
        return {"conflicts": [], "checked": ["no progress notes on file"],
                "note_count": 0}

    payload = {
        "brief_narrative": narrative,
        "brief_structured": {k: pack[k] for k in
                             ("conditions", "medications", "vitals", "labs")},
        "clinician_notes": notes,
    }
    agent = LlmAgent(name="note_reconciliation", model=MODEL,
                     instruction=INSTRUCTION, output_schema=Reconciliation)
    raw = await _run_agent(agent, json.dumps(payload, indent=1, default=str), "reconcile")
    try:
        parsed = Reconciliation.model_validate_json(raw)
        return {"conflicts": [c.model_dump() for c in parsed.conflicts],
                "checked": parsed.checked, "note_count": len(notes)}
    except Exception as exc:
        return {"error": f"could not parse reconciliation ({type(exc).__name__})",
                "note_count": len(notes)}


def reconcile(pack: dict, narrative: str, notes: list[dict] | None = None) -> dict:
    return asyncio.run(reconcile_async(pack, narrative, notes))
