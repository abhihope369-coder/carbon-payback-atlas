"""Visual system: colours, Plotly theme, CSS and small HTML components."""
from __future__ import annotations

import plotly.graph_objects as go

# Surfaces & ink
BG = "#0b1426"
SURFACE = "#111c33"
SURFACE_2 = "#16233f"
BORDER = "#22314f"
INK = "#e8eef8"
INK_2 = "#aab6cc"
MUTED = "#7c8aa5"
GRID = "#1f2c47"
ACCENT = "#2dd4bf"   # UI chrome only (teal)
ACCENT_2 = "#4ade80"  # UI chrome only (green)

# Categorical series, fixed order. Validated (scripts/validate_palette.js, dark,
# surface #111c33): all six checks pass, worst adjacent CVD dE 8.4.
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]

# Sequential (single hue, blue ramp) for magnitude, e.g. the choropleth.
SEQ_BLUE = ["#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5",
            "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"]

# Generation sources in palette order so stacked neighbours are validated pairs.
MIX = [
    ("hydro_share_elec", "Hydro"), ("coal_share_elec", "Coal"), ("wind_share_elec", "Wind"),
    ("solar_share_elec", "Solar"), ("gas_share_elec", "Gas"), ("bio_other", "Bioenergy & other renewables"),
    ("nuclear_share_elec", "Nuclear"), ("oil_share_elec", "Oil"),
]

FONT = "IBM Plex Sans, Inter, system-ui, sans-serif"
MONO = "IBM Plex Mono, ui-monospace, monospace"


def style(fig: go.Figure, height: int = 380, title: str | None = None,
          y_title: str | None = None, x_title: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, template="none", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_2, size=13),
        title=dict(text=title, font=dict(size=15, color=INK), x=0.01, xanchor="left") if title else None,
        margin=dict(l=56, r=24, t=56 if title else 24, b=(118 if x_title else 96) if legend else 48),
        hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=BORDER, font=dict(family=FONT, color=INK)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.22 if x_title else -0.16, xanchor="left", x=0,
                    font=dict(color=INK_2), bgcolor="rgba(0,0,0,0)", traceorder="normal"),
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER, tickcolor=BORDER, tickfont=dict(color=MUTED),
                     title=dict(text=x_title, font=dict(color=MUTED)), zeroline=False)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, linecolor="rgba(0,0,0,0)", tickfont=dict(color=MUTED),
                     title=dict(text=y_title, font=dict(color=MUTED)), zeroline=False)
    return fig


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{ font-family: {FONT}; }}
.stApp {{ background: radial-gradient(1200px 600px at 85% -10%, rgba(45,212,191,0.07), transparent 60%),
                       radial-gradient(900px 500px at -10% 110%, rgba(74,222,128,0.05), transparent 60%), {BG}; }}
section[data-testid="stSidebar"] {{ background: #0a1222; border-right: 1px solid {BORDER}; }}
.block-container {{ padding-top: 4.2rem; max-width: 1320px; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
.cpa-eyebrow {{ color: {ACCENT}; font: 600 12px/1 {MONO}; letter-spacing: .14em; text-transform: uppercase; }}
.cpa-title {{ color: {INK}; font-size: 34px; font-weight: 700; margin: 6px 0 4px; letter-spacing: -0.02em; }}
.cpa-sub {{ color: {INK_2}; font-size: 15.5px; max-width: 880px; line-height: 1.55; }}
.cpa-card {{ background: linear-gradient(180deg, {SURFACE_2}, {SURFACE}); border: 1px solid {BORDER};
             border-radius: 12px; padding: 16px 18px; height: 100%; }}
.cpa-card .lbl {{ color: {MUTED}; font: 500 11.5px/1.3 {MONO}; letter-spacing: .08em; text-transform: uppercase; }}
.cpa-card .val {{ color: {INK}; font: 600 28px/1.15 {FONT}; margin-top: 8px; letter-spacing: -0.01em; }}
.cpa-card .unit {{ color: {INK_2}; font-size: 14px; font-weight: 500; margin-left: 4px; }}
.cpa-card .note {{ color: {INK_2}; font-size: 12.5px; margin-top: 6px; line-height: 1.4; }}
.cpa-card.accent {{ border-color: rgba(45,212,191,0.45); box-shadow: 0 0 0 1px rgba(45,212,191,0.10) inset; }}
.cpa-pill {{ display:inline-block; padding: 2px 9px; border-radius: 999px; font: 500 11.5px/1.6 {MONO};
             border: 1px solid {BORDER}; color: {INK_2}; margin-right: 6px; }}
.cpa-pill.real {{ color: {ACCENT}; border-color: rgba(45,212,191,0.45); }}
.cpa-pill.hypo {{ color: #f5b453; border-color: rgba(245,180,83,0.45); }}
.cpa-pill.model {{ color: #8fb8f2; border-color: rgba(57,135,229,0.5); }}
.cpa-flow {{ display:flex; gap:8px; flex-wrap:wrap; align-items:stretch; margin: 6px 0 4px; }}
.cpa-step {{ flex:1 1 140px; background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px; padding:10px 12px; }}
.cpa-step b {{ color:{INK}; font-size:13.5px; }} .cpa-step span {{ color:{MUTED}; font-size:12px; display:block; margin-top:3px; }}
.cpa-callout {{ background: rgba(45,212,191,0.06); border:1px solid rgba(45,212,191,0.30); border-left: 3px solid {ACCENT};
                border-radius: 8px; padding: 12px 14px; color: {INK_2}; font-size: 14px; line-height: 1.5; }}
.cpa-callout.warn {{ background: rgba(245,180,83,0.06); border-color: rgba(245,180,83,0.35); border-left-color: #f5b453; }}
.cpa-callout b {{ color: {INK}; }}
.cpa-brand {{ font: 700 17px/1.2 {FONT}; color: {INK}; }} .cpa-brand span {{ color:{ACCENT}; }}
div[data-testid="stMetric"] {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px; padding:10px 14px; }}
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 8px; }}
</style>
"""


def card(label: str, value: str, unit: str = "", note: str = "", accent: bool = False) -> str:
    cls = "cpa-card accent" if accent else "cpa-card"
    return (f'<div class="{cls}"><div class="lbl">{label}</div>'
            f'<div class="val">{value}<span class="unit">{unit}</span></div>'
            f'<div class="note">{note}</div></div>')


def pill(text: str, kind: str = "") -> str:
    return f'<span class="cpa-pill {kind}">{text}</span>'


def header(eyebrow: str, title: str, sub: str = "") -> str:
    return (f'<div class="cpa-eyebrow">{eyebrow}</div><div class="cpa-title">{title}</div>'
            f'<div class="cpa-sub">{sub}</div>')


def callout(html: str, warn: bool = False) -> str:
    return f'<div class="cpa-callout{" warn" if warn else ""}">{html}</div>'
