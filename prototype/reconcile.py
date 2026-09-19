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

Check two different things.

FIRST, disagreements about fact:
  - a medication the note records that the brief does not, or the reverse
  - a condition named in one and absent from the other
  - a vital sign or value that differs between them
  - anything the note states about the patient's care that the brief contradicts

SECOND, claims about care delivered that the record does not bear out. Each note
carries orders_placed_at_this_encounter with the outcome of every order from
that visit. Where the note asserts that care happened -- "labs ordered per
condition-specific guidelines", "monitoring labs ordered", "lab orders placed
based on guideline intervals" -- check whether it completed. If every order
from that encounter reads "never returned", the note is technically true and
practically misleading: a clinician reading it concludes the monitoring was
handled when nothing came back. Report that, naming the tests, with the field
"care claimed vs delivered".

ONLY when EVERY order from that encounter went unresulted. If some returned and
some did not, do NOT report it. A partial gap is ordinary -- 73 of the 120
encounters that mention labs have at least one order outstanding, so flagging
those would put the same warning on most briefs and train the reader to skip
it. The outstanding orders are already listed in the brief on their own merits.
What is worth interrupting someone for is the note whose account of care is
wholly unsupported: it said labs were ordered, and not one came back.

That second category is not a contradiction in the note's facts. It is a gap
between what the note says was done and what the record shows happening, and it
is the thing most worth catching here.

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
    """Each note with the orders placed at that same encounter and their outcome.

    The note's FACTS always match the tables in this extract. Its CLAIMS about
    care delivered do not always hold -- three encounters assert "labs ordered
    per guidelines" where every order placed is still unresulted. The model
    cannot see that unless the outcome travels with the note, so it does.
    """
    con = connect()
    try:
        rows = con.execute(
            "SELECT n.PAT_ENC_CSN_ID, n.ENTRY_TIME, n.AUTHOR_PROV_NAME, n.NOTE_TEXT "
            "FROM hno_info n WHERE n.PAT_ID = ? ORDER BY n.ENTRY_TIME DESC",
            [pat_id]).fetchall()
        out = []
        for csn, when, author, text in rows:
            orders = con.execute(
                "SELECT test_name, is_pending FROM v_lab_order WHERE PAT_ENC_CSN_ID = ?",
                [csn]).fetchall()
            out.append({
                "date": str(when), "author": author, "text": text,
                "orders_placed_at_this_encounter": [
                    {"test": t, "result": "never returned" if p else "resulted"}
                    for t, p in orders],
            })
        return out
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
