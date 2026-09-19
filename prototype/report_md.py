"""Turning the supervisor's prose into the markdown the UI actually renders.

Pure string functions, kept out of app.py so they can be tested without
standing up a Streamlit runtime.
"""
from __future__ import annotations

import re
from urllib.parse import quote

FINDINGS_ANCHOR = "findings-the-specialists-recorded"


def link_citations(markdown: str, anchor: str = FINDINGS_ANCHOR) -> str:
    """Turn [F03] and [F03, F05] in the supervisor's prose into anchor links.

    The brackets are kept inside the link text (escaped) so the citation still
    reads as [F03] rather than losing its brackets to markdown link syntax.
    """
    def repl(match: re.Match) -> str:
        ids = re.findall(r"F\d+", match.group(0))
        return " ".join(f"[\\[{i}\\]](#{anchor})" for i in ids)

    return re.sub(r"\[F\d+(?:\s*,\s*F\d+)*\]", repl, markdown)


# The supervisor writes each shortlist entry's action as a nested bullet under
# the patient's numbered line. Markdown turns that into a second-level list, and
# a second-level list inside a loose ordered list renders as its own paragraph:
# a bullet glyph, a full line of leading indent and a blank line above and below.
# Twelve patients became twelve pairs of widely separated lines. Folding the
# action up into its parent item keeps it a hard line break inside the same list
# item -- attached to the patient, no bullet, no paragraph gap.
_ACTION_BULLET = re.compile(
    r"^(\s{2,})[*+-]\s+\*\*\s*(?:Recommended\s+)?Actions?\s*:?\s*\*\*\s*:?\s*(.*)$",
    re.IGNORECASE)


def fold_actions(markdown: str) -> str:
    """Fold "* **Action**: ..." sub-bullets into the line above them.

    Defensive rather than authoritative: the supervisor is asked for this shape
    directly, and this catches the runs where it reaches for a sub-bullet anyway.
    """
    out: list[str] = []
    for line in markdown.split("\n"):
        m = _ACTION_BULLET.match(line)
        if not m or not any(l.strip() for l in out):
            out.append(line)
            continue
        indent, rest = m.group(1), m.group(2).strip()
        while out and not out[-1].strip():      # drop the blank line between them
            out.pop()
        out[-1] = out[-1].rstrip() + "  "       # two spaces == hard line break
        out.append(f"{indent}**Do** {rest}")
    return "\n".join(out)


def link_patients(markdown: str, names: list[str]) -> str:
    """Turn patient names in the prose into links to their brief.

    One pass over an alternation of all the names, longest first, so a name is
    never re-scanned inside a URL this function just inserted and a short name
    cannot match inside a longer one.
    """
    if not names:
        return markdown
    # Idempotent: skip a name already used as link TEXT (preceded by "[") or
    # already sitting inside a link URL (preceded by "="). Without the "=" case
    # a second application nests the link inside its own href.
    pattern = re.compile(
        r"(?<![\[=])(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
        + r")(?!\]\()")
    # The URL must be percent-encoded. A patient name holds a comma and a space,
    # and a markdown link whose target contains a raw space is not parsed as a
    # link at all -- it renders as the literal "[Name](?patient=Name)".
    return pattern.sub(
        lambda m: f"[{m.group(0)}](?patient={quote(m.group(0))})", markdown)
