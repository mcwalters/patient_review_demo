"""Qualified Health brand styling, lifted from the PPT template's theme1.xml.

Palette and fonts are the deck's, not invented. The one addition is a danger
red: the template has no red, and a clinical safety UI needs one, so #B42318
is introduced for high-severity findings and kept away from everything else.
"""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / "assets"

NAVY       = "#0A3D63"   # accent1
BLUE       = "#058BE8"   # accent2 / hyperlink
TITLE_BLUE = "#518AE1"   # the lighter blue used in slide titles
SLATE      = "#5C6C80"   # dk2, body text
MIST       = "#C1CFD8"   # accent4, borders
PAGE       = "#F9FCFF"   # lt2, page background
INDIGO     = "#635BFF"   # accent5
ORANGE     = "#ED8D02"   # accent6
DANGER     = "#B42318"   # added -- the template has no red


def _b64(name: str) -> str:
    return base64.b64encode((ASSETS / name).read_bytes()).decode()


CSS = f"""
<style>
  html, body, [class*="st-"], button, input, textarea, select {{
      font-family: "Helvetica Neue", Arial, sans-serif;
  }}
  /* Streamlit draws its icons with the Material Symbols ligature font, fetched
     from a CDN. When that fetch fails -- offline, blocked, or slow conference
     wifi -- the browser falls back to a text face and renders the ligature NAME
     as literal text ("keyboard_arrow_right") on top of the label. Suppress the
     glyph text and draw the chevron in CSS instead, so the UI does not depend
     on a font download. */
  [data-testid="stIconMaterial"] {{
      font-size: 0 !important; line-height: 1; color: transparent;
  }}
  [data-testid="stIconMaterial"]::before {{
      content: "\203A";                      /* single right-pointing chevron */
      font-family: "Helvetica Neue", Arial, sans-serif;
      font-size: 1.05rem; font-weight: 600; color: {SLATE};
      display: inline-block; transition: transform .15s ease;
  }}
  details[open] [data-testid="stIconMaterial"]::before {{
      transform: rotate(90deg);
  }}
  .stApp {{ background: {PAGE}; }}
  .block-container {{ padding-top: 2.6rem; max-width: 1400px; }}

  /* Deck title pattern: light-blue phrase, thin rule, navy phrase */
  .qh-title {{
      font-size: 1.85rem; font-weight: 700; letter-spacing: -.015em;
      line-height: 1.22; margin: 0 0 .2rem 0;
  }}
  .qh-title .a {{ color: {TITLE_BLUE}; white-space: nowrap; }}
  .qh-title .bar {{
      color: {MIST}; font-weight: 300; padding: 0 .4rem;
  }}
  .qh-title .b {{ color: {NAVY}; }}
  .qh-sub {{ color: {SLATE}; font-size: .86rem; margin-bottom: 1.1rem; }}

  h2, h3 {{ color: {NAVY}; letter-spacing: -.01em; }}
  h3 {{ font-size: 1.12rem; margin-top: 1.5rem; }}

  /* Tabs: navy underline, no pill chrome */
  .stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {MIST}; }}
  .stTabs [data-baseweb="tab"] {{
      padding: .35rem 0; color: {SLATE}; font-weight: 600; font-size: .92rem;
  }}
  .stTabs [aria-selected="true"] {{ color: {NAVY}; }}
  .stTabs [data-baseweb="tab-highlight"] {{ background: {NAVY}; height: 2px; }}

  /* Metrics as bordered cards, like a deck stat row */
  [data-testid="stMetric"] {{
      background: #FFFFFF; border: 1px solid {MIST}; border-radius: 6px;
      padding: .75rem .9rem;
  }}
  [data-testid="stMetricLabel"] p {{
      color: {SLATE}; font-size: .74rem; font-weight: 600;
      text-transform: uppercase; letter-spacing: .06em;
  }}
  [data-testid="stMetricValue"] {{ color: {NAVY}; font-weight: 700; }}

  /* Alerts: left rule in the severity colour, quiet fill */
  [data-testid="stAlert"] {{ border-radius: 4px; border: none; padding: .7rem .9rem; }}
  div[data-testid="stAlert"][class*="error"], .stAlert[data-baseweb="notification"] {{
      border-left: 3px solid {DANGER};
  }}

  .stButton > button {{
      background: {NAVY}; color: #fff; border: none; border-radius: 4px;
      font-weight: 600; padding: .45rem 1.3rem;
  }}
  .stButton > button:hover {{ background: {BLUE}; color: #fff; }}

  [data-testid="stDataFrame"] {{ border: 1px solid {MIST}; border-radius: 6px; }}
  .stTextArea textarea {{
      border: 1px solid {MIST}; border-radius: 6px; font-size: .9rem;
      background: #FFFFFF;
  }}

  /* Footer, matching the content-slide furniture */
  .qh-footer {{
      display: flex; align-items: center; justify-content: space-between;
      border-top: 1px solid {MIST}; margin-top: 2.5rem; padding-top: .8rem;
  }}
  .qh-footer img {{ height: 20px; }}
  .qh-footer span {{ color: {SLATE}; font-size: .72rem; }}
</style>
"""


def apply() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def title(lead: str, rest: str, subtitle: str = "") -> None:
    """The deck's two-tone title: light-blue phrase | navy phrase."""
    st.markdown(
        f'<div class="qh-title"><span class="a">{lead}</span>'
        f'<span class="bar">|</span><span class="b">{rest}</span></div>'
        + (f'<div class="qh-sub">{subtitle}</div>' if subtitle else ""),
        unsafe_allow_html=True)


def footer(note: str = "Synthetic data — no PHI. Prototype for discussion, not clinical use.") -> None:
    st.markdown(
        f'<div class="qh-footer">'
        f'<img src="data:image/png;base64,{_b64("qualified-health-logo.png")}"/>'
        f'<span>{note}</span></div>',
        unsafe_allow_html=True)
