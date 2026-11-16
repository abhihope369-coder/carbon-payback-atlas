"""Carbon Payback Atlas - Streamlit application.

Run:  streamlit run app.py
Data/model artifacts are produced by scripts/build_data.py and scripts/train_forecast.py.
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import pipeline as pl
from src import report, ui
from src.data_processing import TARGET
from src.forecasting import MODEL_LABELS
from src.lifecycle import (VehicleAssumptions, additional_mfg_kg, ev_annual_kg, lifecycle_totals,
                           payback_summary, payback_table, petrol_annual_kg)
from src.optimization import explain
from src.sensitivity import tornado
from src.scenarios import backtest_paths, build_path, observed_series, path_metadata

st.set_page_config(page_title="Carbon Payback Atlas", page_icon="🌍", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(ui.CSS, unsafe_allow_html=True)

BLUE, ORANGE = ui.SERIES[0], ui.SERIES[1]  # blue = forecast-aware, orange = current-data (app-wide)


@st.cache_resource(show_spinner="Loading data and model outputs...")
def data():
    return pl.load_all()


D = data()
FC = D["forecasts"]
PANEL = D["panel"]
MET = D["metrics"]
FC_COUNTRIES = sorted(FC["country"].unique())
A0 = VehicleAssumptions()
A_KEYS = list(A0.to_dict())
PAGES = ["Overview", "Global Atlas", "AI Forecast", "Fleet Lab", "Payback Engine",
         "Deployment Optimizer", "Strategy Report"]

DEFAULTS = {
    **{f"a_{k}": v for k, v in A0.to_dict().items()},
    "page": "Overview", "focus": "Canada", "deploy_year": 2026, "fleet_n": 100, "cost_per_ev": 50_000,
    "budget": 10_000_000, "target_year": 2035, "deploy_range": (2026, 2030), "max_per_year": 80,
    "cost_decline": 0.0, "cands": pl.DEFAULT_CANDIDATES.copy(), "bt_origin": pl.DEFAULT_BACKTEST_ORIGIN,
    "atlas_compare": ["Canada", "United States", "Germany", "India", "Poland"], "color_map": {},
    "tour": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
    elif k not in ("cands", "tour", "color_map"):
        st.session_state[k] = st.session_state[k]  # keep widget values when switching pages

CAND_COLS = ["country", "fleet_available", "min_vehicles", "cost_per_ev", "annual_km"]


def A() -> VehicleAssumptions:
    return VehicleAssumptions(**{k: st.session_state[f"a_{k}"] for k in A_KEYS})


def deploy_years() -> list[int]:
    lo, hi = st.session_state["deploy_range"]
    return list(range(int(lo), int(hi) + 1))


def clean_cands(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    issues = []
    df = df.copy()
    if "min_vehicles" not in df:
        df["min_vehicles"] = 0
    df["min_vehicles"] = pd.to_numeric(df["min_vehicles"], errors="coerce").fillna(0)
    df = df.dropna(subset=["country"])
    bad = df[~df["country"].isin(FC_COUNTRIES)]["country"].tolist()
    if bad:
        issues.append(f"No forecast available for: {', '.join(map(str, bad))} (excluded).")
    df = df[df["country"].isin(FC_COUNTRIES)]
    for c in ["fleet_available", "cost_per_ev", "annual_km"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=CAND_COLS)
    df = df[(df["fleet_available"] >= 0) & (df["cost_per_ev"] > 0) & (df["annual_km"] > 0)
            & (df["min_vehicles"] >= 0)]
    if len(df) < n0:
        issues.append(f"{n0 - len(df)} row(s) with missing or invalid values excluded.")
    if df["country"].duplicated().any():
        issues.append("Duplicate regions found; kept the first row for each.")
        df = df.drop_duplicates("country")
    over = df["min_vehicles"] > df["fleet_available"]
    if over.any():
        issues.append(f"Minimum EVs capped at the fleet size for: {', '.join(df.loc[over, 'country'])}.")
        df.loc[over, "min_vehicles"] = df.loc[over, "fleet_available"]
    df["fleet_available"] = df["fleet_available"].astype(int)
    df["min_vehicles"] = df["min_vehicles"].astype(int)
    return df[CAND_COLS].reset_index(drop=True), issues


@st.cache_data(show_spinner="Optimizing three strategies...")
def run_compare(cands: pd.DataFrame, years: tuple, target: int, a: dict, budget: float,
                mpy: int, decline: float):
    return pl.compare_strategies(D, cands, list(years), target, VehicleAssumptions(**a), budget,
                                 mpy or None, decline)


@st.cache_data(show_spinner="Running historical backtest...")
def run_backtest(cands: pd.DataFrame, a: dict, budget: float, mpy: int, decline: float, origin: int):
    return pl.run_backtest(D, cands, VehicleAssumptions(**a), budget, mpy or None, decline, origin)


def current_runs():
    cands, issues = clean_cands(st.session_state["cands"])
    a = A().to_dict()
    args = (int(st.session_state["budget"]), int(st.session_state["max_per_year"]),
            float(st.session_state["cost_decline"]))
    cmp = run_compare(cands, tuple(deploy_years()), int(st.session_state["target_year"]), a,
                      float(args[0]), args[1], args[2]) if len(cands) else None
    bt = (run_backtest(cands, a, float(args[0]), args[1], args[2], int(st.session_state["bt_origin"]))
          if len(cands) else {"usable": []})
    return cands, issues, cmp, bt


# ------------------------------------------------------------------ guided demo tour
_T = MET["phases"]["test"]["metrics"]
_SKILL = (1 - _T[MET["chosen_model"]]["mae"] / _T["persistence"]["mae"]) * 100
_NSTUDY = D["study_summary"]["overall"]["n_scenarios"] if D["study_summary"] else 0
TOUR = [
    ("Overview", "The problem", "A basic calculator uses today's grid. The Atlas forecasts each grid, prices the "
     "manufacturing debt, optimizes the budget and tests itself against history."),
    ("Global Atlas", "Real data", f"{PANEL['country'].nunique()} countries of real Ember data. Compare grids: Poland and India are dirty "
     "today; the question is how fast they are cleaning up."),
    ("AI Forecast", "The AI", f"A {MODEL_LABELS[MET['chosen_model']]} forecasts 10 years ahead with an 80% band. On unseen 2020+ data "
     f"it has {_SKILL:.0f}% lower error than assuming today's grid persists."),
    ("Payback Engine", "Carbon payback", "The EV starts in carbon debt from its battery, then pays it back. "
     "The tornado shows which assumption matters most; 'Show the math' proves every number."),
    ("Deployment Optimizer", "The decision", "A MILP allocates the budget. Three strategies, manager overrides "
     f"with their price in tonnes, and a {_NSTUDY}-scenario historical test of whether forecasting helps."),
    ("Strategy Report", "The deliverable", "A decision-ready report and CSVs, built from the numbers above."),
]


def tour_go(i):
    st.session_state["tour"] = i
    if i is not None:
        st.session_state["page"] = TOUR[i][0]


def tour_banner():
    i = st.session_state.get("tour")
    if i is None or TOUR[i][0] != st.session_state["page"]:
        return
    page_, title, text = TOUR[i]
    st.markdown(ui.callout(f"<b>Guided demo · step {i + 1} of {len(TOUR)} · {title}.</b> {text}"),
                unsafe_allow_html=True)
    c = st.columns([1, 1, 1, 6])
    if i > 0:
        c[0].button("← Back", on_click=tour_go, args=(i - 1,), key=f"tb{i}")
    if i < len(TOUR) - 1:
        c[1].button("Next →", on_click=tour_go, args=(i + 1,), type="primary", key=f"tn{i}")
    c[2].button("End demo", on_click=tour_go, args=(None,), key=f"te{i}")
    st.write("")


def entity_colors(names: list[str]) -> dict:
    """Stable colour per entity: a country keeps its slot while it stays selected."""
    cmap = {k: v for k, v in st.session_state["color_map"].items() if k in names}
    for n in names:
        if n not in cmap:
            free = [c for c in ui.SERIES if c not in cmap.values()]
            cmap[n] = free[0] if free else ui.MUTED
    st.session_state["color_map"] = cmap
    return cmap


def fmt_t(kg: float) -> str:
    return f"{kg / 1000:,.1f}"


def payback_for(country: str, year: int, a: VehicleAssumptions, kind: str = "forecast"):
    path = build_path(PANEL, FC, country, kind=kind)
    t = payback_table(a, year, path)
    return t, payback_summary(t, a)


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "topojsonURL": "./app/static/"})


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown('<div class="cpa-brand">Carbon <span>Payback</span> Atlas</div>'
                '<div style="color:#7c8aa5;font-size:12.5px;margin:4px 0 14px">'
                'The right climate investment. In the right place. At the right time.</div>',
                unsafe_allow_html=True)
    page = st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    st.selectbox("Focus region", FC_COUNTRIES, key="focus",
                 help="Used by Overview, AI Forecast, Fleet Lab and Payback Engine.")
    st.markdown(
        f'<div style="color:#7c8aa5;font-size:11.5px;line-height:1.5;margin-top:18px">'
        f'Grid data: Ember via Our World in Data (CC BY 4.0), accessed {D["manifest"]["date_accessed"]}.<br>'
        f'{ui.pill("REAL", "real")} sourced data<br>{ui.pill("MODEL", "model")} model output<br>'
        f'{ui.pill("HYPOTHETICAL", "hypo")} scenario input</div>', unsafe_allow_html=True)

focus = st.session_state["focus"]
a = A()


# ================================================================== OVERVIEW
def page_overview():
    st.markdown(ui.header("AI climate investment intelligence", "Carbon Payback Atlas",
                          "Where and when should a fleet spend a limited budget on EVs? The Atlas "
                          "forecasts how clean each grid will become, prices the battery's carbon debt "
                          "year by year, allocates the budget with an optimizer, and tests itself "
                          "against history."), unsafe_allow_html=True)
    st.write("")
    b1, b2, _ = st.columns([1.3, 1.2, 4])
    b1.button("▶ Start guided demo", type="primary", on_click=tour_go, args=(0,), width="stretch")
    b2.button("Jump to optimizer", on_click=lambda: st.session_state.update(page="Deployment Optimizer"),
              width="stretch")
    st.write("")

    year = int(st.session_state["deploy_year"])
    meta = path_metadata(FC, focus)
    f = FC[FC["country"] == focus].set_index("year")
    t_f, pb = payback_for(focus, year, a)
    _, pb_c = payback_for(focus, year, a, "current")
    _, pb_lo = payback_for(focus, year, a, "lo80")
    _, pb_hi = payback_for(focus, year, a, "hi80")
    cands, _, cmp, bt = current_runs()
    test = MET["phases"]["test"]["metrics"]
    chosen = MET["chosen_model"]
    ss = D["study_summary"]

    # ---- the "before vs after" contrast
    left, right = st.columns(2)
    pbc_txt = f"{pb_c['payback_years_fractional']:.1f} years" if pb_c["payback_year"] else "no payback"
    pb_txt = f"{pb['payback_years_fractional']:.1f} years" if pb["payback_year"] else "no payback"
    left.markdown(f"""<div class="cpa-card"><div class="lbl">A basic carbon calculator says</div>
      <div class="note" style="font-size:14.5px;line-height:1.7;margin-top:10px">
      Uses {focus}'s grid <b>today</b> ({meta['latest_observed']:,.0f} g/kWh) for every future year.<br>
      One EV in {year}: payback <b>{pbc_txt}</b>, lifetime net <b>{fmt_t(pb_c['lifetime_net_kg'])} t</b>.<br>
      One number, no uncertainty, no budget decision, never checked against reality.</div></div>""",
                  unsafe_allow_html=True)
    if cmp:
        B = cmp["B"]
        n_reg = int((B["allocation"].groupby("country")["vehicles"].sum() > 0).sum())
        cpt = f"${B['cost_per_t']:,.0f}/t" if B.get("cost_per_t") else "n/a"
        study_line = (f"Tested on <b>{ss['overall']['n_scenarios']} historical scenarios</b> (2010-2015 "
                      f"origins) to measure what forecasting is really worth." if ss else "")
        right.markdown(f"""<div class="cpa-card accent"><div class="lbl">Carbon Payback Atlas says</div>
          <div class="note" style="font-size:14.5px;line-height:1.7;margin-top:10px">
          Forecasts {focus}'s grid to {meta['last_forecast_year']} with an 80% band (AI, validated on unseen years).<br>
          One EV in {year}: payback <b>{pb_txt}</b>, lifetime net <b>{fmt_t(pb['lifetime_net_kg'])} t</b>
          (range {fmt_t(pb_hi['lifetime_net_kg'])} to {fmt_t(pb_lo['lifetime_net_kg'])} t).<br>
          ${st.session_state['budget'] / 1e6:,.1f}M allocated across {n_reg} regions: <b>{B['planned_net_t']:,.0f} t
          CO₂e</b> by {st.session_state['target_year']} (<b>{cpt}</b> of gross EV spend).<br>{study_line}</div></div>""",
                       unsafe_allow_html=True)
    st.write("")

    c = st.columns(4)
    y5 = meta["origin_year"] + 5
    c[0].markdown(ui.card(f"{focus} grid today", f"{meta['latest_observed']:,.0f}", "g CO₂e/kWh",
                          f"{ui.pill('REAL', 'real')}Observed {meta['origin_year']}"), unsafe_allow_html=True)
    c[1].markdown(ui.card(f"Forecast {y5}", f"{f.loc[y5, 'forecast']:,.0f}", "g CO₂e/kWh",
                          f"{ui.pill('MODEL', 'model')}80% band {f.loc[y5, 'lo80']:,.0f}-{f.loc[y5, 'hi80']:,.0f}"),
                  unsafe_allow_html=True)
    if cmp:
        c[2].markdown(ui.card("Budget per tonne", f"${cmp['B']['cost_per_t']:,.0f}" if cmp['B'].get('cost_per_t')
                              else "n/a", "per t CO₂e", f"{ui.pill('MODEL', 'model')}Gross EV spend ÷ net t by "
                              f"{st.session_state['target_year']}; not net of fuel savings"), unsafe_allow_html=True)
    c[3].markdown(ui.card("Forecast skill, unseen 2020+", f"{(1 - test[chosen]['mae'] / test['persistence']['mae']) * 100:.0f}%",
                          "lower error", f"MAE {test[chosen]['mae']:.1f} vs {test['persistence']['mae']:.1f} g/kWh "
                          "for 'grid stays the same'"), unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.5, 1])
    with left:
        show(forecast_figure(focus, height=340, title=f"{focus}: observed and forecast grid intensity"))
    with right:
        if ss:
            o = ss["overall"]
            st.markdown(ui.callout(
                f"<b>What history taught us (honest result).</b><br>Across {o['n_scenarios']} plans made as of "
                f"2010-2015 and scored on the real grids that followed: forecasting changed the plan in only "
                f"{(1 - o['identical_plans'] / o['n_scenarios']) * 100:.0f}% of cases, and even perfect "
                f"foresight beat the simple plan by just {o['mean_perfect_gain_pct']:.2f}%. <b>Where</b> to "
                f"invest is driven by today's grid gap. Where forecasting does help is <b>how much</b>: the "
                f"simple method under-estimated real savings by {abs(o['mean_est_err_current_pct']):.1f}% on "
                f"average; the forecast estimate was off by {o['mean_est_err_forecast_pct']:+.1f}% and closer "
                f"in {o['forecast_estimate_closer_share']:.0%} of cases."), unsafe_allow_html=True)
        st.write("")
        st.markdown(ui.callout(
            f"{ui.pill('REAL', 'real')}{ui.pill('MODEL', 'model')}{ui.pill('HYPOTHETICAL', 'hypo')}<br>"
            "Every number is computed from the dataset, the trained model, or an editable assumption, and "
            "each is labelled so judges can tell which is which."), unsafe_allow_html=True)


# ================================================================== ATLAS
def page_atlas():
    st.markdown(ui.header("Module 1", "Global Atlas",
                          "Lifecycle carbon intensity of electricity generation by country (Ember via "
                          "Our World in Data). Latest available year per country."), unsafe_allow_html=True)
    latest = PANEL.sort_values("year").groupby("country").tail(1)
    fig = go.Figure(go.Choropleth(
        locations=latest["iso_code"], z=latest[TARGET], text=latest["country"],
        customdata=latest["year"], colorscale=[[i / 11, c] for i, c in enumerate(ui.SEQ_BLUE)],
        marker_line_color=ui.SURFACE, marker_line_width=0.5,
        colorbar=dict(title=dict(text="g CO₂e/kWh", font=dict(color=ui.MUTED)), tickfont=dict(color=ui.MUTED),
                      thickness=10, len=0.75, outlinewidth=0),
        hovertemplate="<b>%{text}</b><br>%{z:,.0f} g CO₂e/kWh (%{customdata})<extra></extra>"))
    fig.update_geos(bgcolor=ui.SURFACE, showframe=False, showcoastlines=False, projection_type="natural earth",
                    landcolor=ui.SURFACE_2, showland=True, showcountries=False, lataxis_range=[-58, 85])
    show(ui.style(fig, height=470, legend=False))
    st.caption(f"{latest['country'].nunique()} countries shown. Lighter = more carbon-intensive electricity.")

    st.multiselect("Compare regions (up to 8)", sorted(PANEL["country"].unique()), key="atlas_compare",
                   max_selections=8)
    names = st.session_state["atlas_compare"]
    cmap = entity_colors(names)
    col1, col2 = st.columns([1.4, 1])
    with col1:
        fig = go.Figure()
        for n in names:
            s = observed_series(PANEL, n)
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=n, mode="lines",
                                     line=dict(color=cmap[n], width=2),
                                     hovertemplate=f"{n}: %{{y:,.0f}} g/kWh<extra></extra>"))
        fig.update_layout(hovermode="x unified")
        show(ui.style(fig, 400, "Historical carbon intensity", "g CO₂e/kWh"))
    with col2:
        rows = []
        for n in names:
            s = observed_series(PANEL, n)
            if len(s):
                y0 = s.index[0]
                rows.append({"Region": n, "Latest year": int(s.index[-1]), "g/kWh": round(s.iloc[-1]),
                             f"Change since {y0}": f"{(s.iloc[-1] / s.iloc[0] - 1) * 100:+.0f}%"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    g = PANEL[PANEL["country"] == focus].sort_values("year").copy()
    g["bio_other"] = g["biofuel_share_elec"].fillna(0) + g["other_renewables_share_elec"].fillna(0)
    fig = go.Figure()
    for i, (col, label) in enumerate(ui.MIX):
        fig.add_trace(go.Scatter(x=g["year"], y=g[col], name=label, stackgroup="mix", mode="lines",
                                 line=dict(width=0.6, color=ui.SURFACE), fillcolor=ui.SERIES[i],
                                 hovertemplate=f"{label}: %{{y:.1f}}%<extra></extra>"))
    fig.update_layout(hovermode="x unified")
    show(ui.style(fig, 400, f"{focus}: electricity generation mix", "% of generation"))
    st.caption("Shares as reported; missing source values are shown as gaps, not filled. "
               "Change the focus region in the sidebar.")


# ================================================================== FORECAST
def forecast_figure(country: str, height: int = 430, title: str | None = None) -> go.Figure:
    obs = observed_series(PANEL, country)
    f = FC[FC["country"] == country].sort_values("year")
    origin = int(f["origin_year"].iloc[0])
    obs = obs[obs.index <= origin]
    x_band = [origin] + f["year"].tolist()
    lo = [obs.iloc[-1]] + f["lo80"].tolist()
    hi = [obs.iloc[-1]] + f["hi80"].tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_band + x_band[::-1], y=hi + lo[::-1], fill="toself",
                             fillcolor="rgba(57,135,229,0.16)", line=dict(width=0), name="80% interval",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=obs.index, y=obs.values, name="Observed", mode="lines",
                             line=dict(color=BLUE, width=2),
                             hovertemplate="Observed %{x}: %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[origin] + f["year"].tolist(), y=[obs.iloc[-1]] + f["forecast"].tolist(),
                             name="Forecast (model)", mode="lines+markers",
                             line=dict(color=BLUE, width=2, dash="dash"), marker=dict(size=8, color=BLUE,
                                                                                    line=dict(color=ui.SURFACE, width=2)),
                             hovertemplate="Forecast %{x}: %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[origin, f["year"].max()], y=[obs.iloc[-1]] * 2, name="Current-data assumption",
                             mode="lines", line=dict(color=ORANGE, width=2, dash="dot"),
                             hovertemplate="Flat at %{y:,.0f}<extra></extra>"))
    fig.add_vline(x=origin, line=dict(color=ui.MUTED, width=1, dash="dot"))
    fig.add_annotation(x=origin, y=1.02, yref="paper", text=f"forecast origin {origin}", showarrow=False,
                       font=dict(color=ui.MUTED, size=11), xanchor="right")
    fig.update_layout(hovermode="x unified")
    return ui.style(fig, height, title, "g CO₂e/kWh")


def page_forecast():
    st.markdown(ui.header("Module 2 · Engine 1", "Grid Future AI",
                          "A Random Forest trained on 2000-2025 data from 104 grids predicts the ratio of "
                          "future to current intensity for 1 to 10 years ahead, using intensity lags, "
                          "recent trend and generation mix."), unsafe_allow_html=True)
    meta = path_metadata(FC, focus)
    show(forecast_figure(focus, title=f"{focus}: observed (solid) vs forecast (dashed) carbon intensity"))
    st.caption(f"Forecast covers {meta['origin_year'] + 1}-{meta['last_forecast_year']}. Beyond "
               f"{meta['last_forecast_year']}, lifecycle calculations hold intensity flat at the last "
               "forecast value (an explicit assumption, not a model output).")
    f = FC[FC["country"] == focus][["year", "forecast", "lo80", "hi80"]].round(0)
    with st.expander("Forecast values table"):
        st.dataframe(f.rename(columns={"year": "Year", "forecast": "Central g/kWh", "lo80": "Low (10th pct)",
                                       "hi80": "High (90th pct)"}), hide_index=True, width="stretch")

    st.subheader("Validation (temporal splits, no leakage)")
    test = MET["phases"]["test"]
    sel = MET["phases"]["select"]
    chosen = MET["chosen_model"]
    c = st.columns(4)
    c[0].markdown(ui.card("Selected model", MODEL_LABELS[chosen], "", "Lowest MAE on 2015-2019 validation"),
                  unsafe_allow_html=True)
    c[1].markdown(ui.card("Test MAE (2020+)", f"{test['metrics'][chosen]['mae']:.1f}", "g/kWh",
                          f"n = {test['metrics'][chosen]['n']:,} country-horizon pairs"), unsafe_allow_html=True)
    c[2].markdown(ui.card("Persistence MAE", f"{test['metrics']['persistence']['mae']:.1f}", "g/kWh",
                          "Baseline: next years equal latest year"), unsafe_allow_html=True)
    c[3].markdown(ui.card("80% interval coverage", f"{test['interval_80_coverage_on_test']:.0%}", "",
                          "Calibrated on 2015-2019, measured on 2020+. Slightly under nominal 80%."),
                  unsafe_allow_html=True)
    st.write("")
    rows = []
    for m in ["ridge", "random_forest", "gradient_boosting", "persistence", "linear_trend"]:
        rows.append({"Model": MODEL_LABELS.get(m, m.replace("_", " ").capitalize() + " (baseline)"),
                     "Validation MAE 2015-19": round(sel["metrics"][m]["mae"], 1),
                     "Test MAE 2020+": round(test["metrics"][m]["mae"], 1),
                     "Test RMSE": round(test["metrics"][m]["rmse"], 1),
                     "Test R² (levels)": round(test["metrics"][m]["r2_levels"], 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("R² on intensity levels is high partly because countries differ greatly in level; MAE "
               "against the persistence baseline is the more meaningful comparison. The linear-trend "
               "baseline extrapolates recent changes and fails badly on volatile grids.")

    col1, col2 = st.columns(2)
    with col1:
        ph_m = pd.DataFrame(test["per_horizon"][chosen])
        ph_p = pd.DataFrame(test["per_horizon"]["persistence"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ph_m["h"], y=ph_m["mae"], name=MODEL_LABELS[chosen], mode="lines+markers",
                                 line=dict(color=BLUE, width=2), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=ph_p["h"], y=ph_p["mae"], name="Persistence baseline", mode="lines+markers",
                                 line=dict(color=ORANGE, width=2), marker=dict(size=8)))
        fig.update_layout(hovermode="x unified")
        show(ui.style(fig, 360, "Test error by forecast horizon", "MAE g CO₂e/kWh", "Years ahead"))
        h1m, h1p = ph_m.iloc[0]["mae"], ph_p.iloc[0]["mae"]
        if h1m > h1p:
            st.caption(f"At 1 year ahead the model ({h1m:.1f}) is slightly worse than persistence "
                       f"({h1p:.1f}); its advantage appears from 2 years onward and grows with horizon.")
    with col2:
        imp = pd.Series(MET.get("feature_importance", {})).sort_values()
        labels = {"h": "Horizon (years ahead)", "log_i0": "Current intensity (log)", "d1": "1-yr change",
                  "d2": "Prior 1-yr change", "d3": "3-yr change", "coal": "Coal share", "gas": "Gas share",
                  "oil": "Oil share", "nuclear": "Nuclear share", "hydro": "Hydro share",
                  "wind_solar": "Wind + solar share", "low_carbon_chg3": "3-yr change in low-carbon share"}
        fig = go.Figure(go.Bar(x=imp.values, y=[labels.get(i, i) for i in imp.index], orientation="h",
                               marker=dict(color=BLUE, cornerradius=4),
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        show(ui.style(fig, 360, "Feature importance (impurity-based)", None, None, legend=False))

    st.subheader(f"Retro-check: what a 2014 model predicted for {focus}")
    try:
        p = backtest_paths(PANEL, D["backtests"][pl.DEFAULT_BACKTEST_ORIGIN], focus, pl.DEFAULT_BACKTEST_ORIGIN)
        obs = observed_series(PANEL, focus)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=obs.index, y=obs.values, name="Actual", mode="lines",
                                 line=dict(color=ui.INK_2, width=2)))
        fig.add_trace(go.Scatter(x=[2014] + list(p["forecast"].index),
                                 y=[obs.loc[2014]] + list(p["forecast"].values),
                                 name="Forecast made in 2014", mode="lines+markers",
                                 line=dict(color=BLUE, width=2, dash="dash"), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=[2014, 2024], y=[obs.loc[2014]] * 2, name="Current-data (2014 value)",
                                 mode="lines", line=dict(color=ORANGE, width=2, dash="dot")))
        fig.update_layout(hovermode="x unified")
        show(ui.style(fig, 340, None, "g CO₂e/kWh"))
        st.caption("This model was trained only on targets up to 2014, so 2015-2024 is genuinely unseen.")
    except KeyError:
        st.info(f"No 2014 retro forecast is available for {focus} (insufficient data before 2014).")

    st.markdown(ui.callout(
        "<b>Limitations.</b> The model learns from 2000-2025 patterns across 104 grids. It cannot "
        "anticipate new policies, plant closures or demand shocks. Forecasts are estimates, not "
        "predictions of policy outcomes; intervals are empirical (from past errors) and covered "
        f"{test['interval_80_coverage_on_test']:.0%} of 2020+ outcomes versus a nominal 80%.", warn=True),
        unsafe_allow_html=True)


# ================================================================== FLEET LAB
SOURCES = {
    "petrol_kg_per_l": "UK DESNZ 2025 conversion factors: 2.069 (tailpipe) + 0.581 (well-to-tank) kg CO₂e/L. "
                       "Well-to-wheel to match Ember's lifecycle grid intensity.",
    "battery_kg_per_kwh": "ICCT (2021) global LCA: 60 kg CO₂e/kWh for Europe/US batteries (68 for China/India).",
    "petrol_vehicle_mfg_t": "Argonne GREET: 8-12 t CO₂e vehicle-cycle for current powertrains (low end used).",
    "ev_vehicle_mfg_excl_battery_t": "ICCT (2021): non-battery manufacturing is similar across powertrains.",
}


def reset_defaults():
    for k, v in DEFAULTS.items():
        if k.startswith("a_") or k in ("deploy_year", "fleet_n", "cost_per_ev"):
            st.session_state[k] = v


def page_fleet():
    st.markdown(ui.header("Module 3", "Fleet Lab",
                          f"Configure the hypothetical MetroFleet transition in <b>{focus}</b>. Every "
                          "page uses these assumptions. Sourced defaults are marked; everything "
                          "else is an editable scenario input."), unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Usage** {ui.pill('HYPOTHETICAL', 'hypo')}", unsafe_allow_html=True)
        st.number_input("Annual driving distance (km)", 1000, 100_000, step=1000, key="a_annual_km")
        st.number_input("Vehicle lifetime (years)", 3, 25, step=1, key="a_lifetime_years")
        st.markdown(f"**Petrol vehicle**", unsafe_allow_html=True)
        st.number_input("Fuel consumption (L/100 km)", 2.0, 30.0, step=0.5, key="a_petrol_l_per_100km")
        st.number_input("Petrol emissions, well-to-wheel (kg CO₂e/L)", 1.0, 4.0, step=0.05,
                        key="a_petrol_kg_per_l", help=SOURCES["petrol_kg_per_l"])
    with c2:
        st.markdown(f"**Electric vehicle**", unsafe_allow_html=True)
        st.number_input("Electricity use at the wall, incl. charging losses (kWh/km)", 0.08, 0.6, step=0.01,
                        key="a_ev_kwh_per_km")
        st.number_input("Extra grid losses on top of generation intensity (%)", 0.0, 20.0, step=0.5,
                        key="a_grid_loss_pct", help="0% treats intensity per kWh generated as per kWh delivered.")
        st.markdown(f"**Manufacturing** {ui.pill('SOURCED', 'real')}", unsafe_allow_html=True)
        st.number_input("Battery size (kWh)", 10.0, 200.0, step=5.0, key="a_battery_kwh")
        st.number_input("Battery manufacturing (kg CO₂e/kWh)", 20.0, 200.0, step=5.0, key="a_battery_kg_per_kwh",
                        help=SOURCES["battery_kg_per_kwh"])
    with c3:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.number_input("Petrol vehicle manufacturing (t CO₂e)", 2.0, 30.0, step=0.5, key="a_petrol_vehicle_mfg_t",
                        help=SOURCES["petrol_vehicle_mfg_t"])
        st.number_input("EV manufacturing excluding battery (t CO₂e)", 2.0, 30.0, step=0.5,
                        key="a_ev_vehicle_mfg_excl_battery_t", help=SOURCES["ev_vehicle_mfg_excl_battery_t"])
        st.radio("Replacement scenario", ["new_vs_new", "early_retirement"], key="a_replacement_mode",
                 format_func=lambda v: {"new_vs_new": "Buying new anyway (EV vs new petrol car)",
                                        "early_retirement": "Retiring a working petrol car early"}[v])
    st.markdown(f"**Deployment** {ui.pill('HYPOTHETICAL', 'hypo')}", unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    d1.markdown(f"Region: **{focus}**  \n<span style='color:#7c8aa5;font-size:12px'>change in sidebar</span>",
                unsafe_allow_html=True)
    d2.number_input("Deployment year", 2026, 2035, step=1, key="deploy_year")
    d3.number_input("EVs to deploy", 1, 100_000, step=10, key="fleet_n")
    d4.number_input("Cost per replacement EV ($)", 1_000, 500_000, step=1_000, key="cost_per_ev")
    st.button("Reset to documented defaults", on_click=reset_defaults)

    a = A()
    year = int(st.session_state["deploy_year"])
    n = int(st.session_state["fleet_n"])
    t, pb = payback_for(focus, year, a)
    st.divider()
    c = st.columns(5)
    c[0].markdown(ui.card("Petrol, per vehicle-year", fmt_t(petrol_annual_kg(a)), "t CO₂e", "Well-to-wheel"),
                  unsafe_allow_html=True)
    c[1].markdown(ui.card(f"EV charging, {year}", fmt_t(t['ev_kg'].iloc[0]), "t CO₂e",
                          f"Grid {t['grid_g_per_kwh'].iloc[0]:,.0f} g/kWh (forecast)"), unsafe_allow_html=True)
    c[2].markdown(ui.card("Extra manufacturing (M)", fmt_t(additional_mfg_kg(a)), "t CO₂e",
                          "Per EV, paid back by operation"), unsafe_allow_html=True)
    c[3].markdown(ui.card("Payback", f"{pb['payback_years_fractional']:.1f}" if pb["payback_year"] else "None",
                          "years" if pb["payback_year"] else "", "Within lifetime" if pb["payback_year"]
                          else "No payback within lifetime"), unsafe_allow_html=True)
    c[4].markdown(ui.card(f"Fleet of {n:,}", f"${n * st.session_state['cost_per_ev'] / 1e6:,.2f}M", "",
                          f"Lifetime net {t['cumulative_net_kg'].iloc[-1] * n / 1e6:,.1f} kt CO₂e"),
                  unsafe_allow_html=True)


# ================================================================== PAYBACK
def page_payback():
    year = int(st.session_state["deploy_year"])
    n = int(st.session_state["fleet_n"])
    st.markdown(ui.header("Module 4 · Engine 2", "Carbon Payback Engine",
                          f"One EV deployed in <b>{focus}</b> in <b>{year}</b>, compared with a petrol car "
                          "over the vehicle's lifetime. Charging emissions change every year with the "
                          "forecast grid."), unsafe_allow_html=True)
    tabs = {k: payback_for(focus, year, a, k) for k in ["forecast", "lo80", "hi80", "current"]}
    t, pb = tabs["forecast"]
    tc, pbc = tabs["current"]

    c = st.columns(4)
    c[0].markdown(ui.card("Payback (forecast grid)", f"{pb['payback_years_fractional']:.1f}" if pb["payback_year"]
                          else "None", "years" if pb["payback_year"] else "",
                          f"Breakeven during {pb['payback_year']}" if pb["payback_year"] else
                          "Never within the modelled lifetime", accent=True), unsafe_allow_html=True)
    c[1].markdown(ui.card("Payback (current-data grid)", f"{pbc['payback_years_fractional']:.1f}"
                          if pbc["payback_year"] else "None", "years" if pbc["payback_year"] else "",
                          "Grid frozen at latest observed value"), unsafe_allow_html=True)
    c[2].markdown(ui.card("Lifetime net savings / EV", fmt_t(pb["lifetime_net_kg"]), "t CO₂e",
                          f"80% band {fmt_t(tabs['hi80'][1]['lifetime_net_kg'])} to "
                          f"{fmt_t(tabs['lo80'][1]['lifetime_net_kg'])}"), unsafe_allow_html=True)
    c[3].markdown(ui.card(f"Fleet of {n:,} EVs", f"{pb['lifetime_net_kg'] * n / 1e6:,.1f}", "kt CO₂e",
                          "Lifetime net, forecast grid"), unsafe_allow_html=True)
    st.write("")

    fig = go.Figure()
    xs = t["year"].tolist()
    lo, hi = tabs["lo80"][0]["cumulative_net_kg"] / 1000, tabs["hi80"][0]["cumulative_net_kg"] / 1000
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=list(lo) + list(hi[::-1]), fill="toself",
                             fillcolor="rgba(57,135,229,0.16)", line=dict(width=0), name="80% forecast band",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=t["cumulative_net_kg"] / 1000, name="Forecast-aware", mode="lines+markers",
                             line=dict(color=BLUE, width=2), marker=dict(size=8, line=dict(color=ui.SURFACE, width=2)),
                             hovertemplate="%{x}: %{y:,.1f} t<extra>Forecast</extra>"))
    fig.add_trace(go.Scatter(x=xs, y=tc["cumulative_net_kg"] / 1000, name="Current-data", mode="lines",
                             line=dict(color=ORANGE, width=2, dash="dot"),
                             hovertemplate="%{x}: %{y:,.1f} t<extra>Current-data</extra>"))
    fig.add_hline(y=0, line=dict(color=ui.MUTED, width=1))
    if pb["payback_year"]:
        fig.add_annotation(x=year - 1 + pb["payback_years_fractional"], y=0,
                           text=f"payback {pb['payback_years_fractional']:.1f} yrs",
                           showarrow=True, arrowcolor=ui.MUTED, ax=40, ay=-40, font=dict(color=ui.INK, size=12))
    fig.add_annotation(x=xs[0], y=-additional_mfg_kg(a) / 1000, text="starts at −M (extra manufacturing)",
                       showarrow=False, yshift=-14, xanchor="left", font=dict(color=ui.MUTED, size=11))
    fig.update_layout(hovermode="x unified")
    show(ui.style(fig, 420, "Cumulative net emissions savings per EV (end of each year)", "t CO₂e",
                  "Operating year (calendar)"))
    st.caption("Values are cumulative at the end of each operating year; payback time is interpolated "
               "within the breakeven year.")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(x=xs, y=t["saving_kg"] / 1000, marker=dict(color=BLUE, cornerradius=4),
                               customdata=t["grid_g_per_kwh"],
                               hovertemplate="%{x}: %{y:,.2f} t saved (grid %{customdata:,.0f} g/kWh)<extra></extra>"))
        show(ui.style(fig, 360, "Annual operational savings per EV (forecast grid)", "t CO₂e", legend=False))
    with col2:
        lt = lifecycle_totals(t, a)
        comps = ["Vehicle manufacturing", "Battery manufacturing", "Fuel / electricity (well-to-wheel)"]
        fig = go.Figure()
        for i, comp in enumerate(comps):
            fig.add_trace(go.Bar(x=["Petrol", "Electric"], y=[lt["Petrol"][comp] / 1000, lt["Electric"][comp] / 1000],
                                 name=comp, marker=dict(color=ui.SERIES[i], line=dict(color=ui.SURFACE, width=2)),
                                 hovertemplate=f"{comp}: %{{y:,.1f}} t<extra></extra>"))
        fig.update_layout(barmode="stack", bargap=0.45)
        show(ui.style(fig, 360, f"Lifecycle emissions per vehicle, {a.lifetime_years} years", "t CO₂e"))

    st.subheader("Which assumption matters most?")
    path_f = build_path(PANEL, FC, focus, kind="forecast")
    base, tor = tornado(a, year, path_f, 20.0)
    col1, col2 = st.columns([1.35, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(y=tor["parameter"], x=tor["low"] - base, base=base, orientation="h",
                             name="Input −20%", marker=dict(color=ORANGE, cornerradius=4),
                             customdata=tor["low"], hovertemplate="%{y} −20%: %{customdata:,.1f} t<extra></extra>"))
        fig.add_trace(go.Bar(y=tor["parameter"], x=tor["high"] - base, base=base, orientation="h",
                             name="Input +20%", marker=dict(color=BLUE, cornerradius=4),
                             customdata=tor["high"], hovertemplate="%{y} +20%: %{customdata:,.1f} t<extra></extra>"))
        fig.add_vline(x=base, line=dict(color=ui.MUTED, width=1, dash="dot"))
        fig.update_layout(barmode="overlay", bargap=0.35)
        show(ui.style(fig, 400, f"Lifetime net savings per EV (base {base:,.1f} t): each input ±20%",
                      None, "t CO₂e"))
    with col2:
        top = tor.iloc[-1]
        bottom = tor.iloc[0]
        g_row = tor[tor["parameter"].str.startswith("Grid")].iloc[0]
        st.markdown(ui.callout(
            f"<b>Biggest lever: {top['parameter'].lower()}</b> (swing {top['swing']:,.1f} t per EV between "
            f"−20% and +20%). Least sensitive: {bottom['parameter'].lower()} ({bottom['swing']:,.1f} t).<br><br>"
            f"A ±20% error in the <b>grid forecast</b> moves the answer by {g_row['swing']:,.1f} t "
            f"({g_row['swing'] / abs(base) * 100 if base else 0:.0f}% of the base), ranking "
            f"<b>{len(tor) - list(tor['parameter']).index(g_row['parameter'])} of {len(tor)}</b> for {focus}. "
            f"The grid matters more on carbon-intensive grids; switch the focus region to compare."),
            unsafe_allow_html=True)

    with st.expander("Show the math (verify any number on this page)"):
        r0 = t.iloc[0]
        f_kgkm = a.petrol_l_per_100km / 100 * a.petrol_kg_per_l
        kwh = a.annual_km * a.ev_kwh_per_km * (1 + a.grid_loss_pct / 100)
        m_kg = additional_mfg_kg(a)
        st.markdown(f"""
**Petrol, per year (well-to-wheel):** E_petrol = D × (L/100 km ÷ 100) × kg/L = {a.annual_km:,.0f} × ({a.petrol_l_per_100km} ÷ 100) × {a.petrol_kg_per_l} = **{petrol_annual_kg(a):,.0f} kg**

**EV, year 1 ({int(r0['year'])}):** E_EV = D × kWh/km × (1 + grid loss) × G = {a.annual_km:,.0f} × {a.ev_kwh_per_km} × {1 + a.grid_loss_pct / 100:.3f} × {r0['grid_g_per_kwh']:,.1f} g ÷ 1000 = {kwh:,.0f} kWh × {r0['grid_g_per_kwh'] / 1000:.4f} = **{r0['ev_kg']:,.0f} kg**

**Saving, year 1:** S = {petrol_annual_kg(a):,.0f} − {r0['ev_kg']:,.0f} = **{r0['saving_kg']:,.0f} kg**

**Extra manufacturing (M):** {"battery " + f"{a.battery_kwh:g} kWh × {a.battery_kg_per_kwh:g} kg/kWh + EV body {a.ev_vehicle_mfg_excl_battery_t:g} t − petrol car {a.petrol_vehicle_mfg_t:g} t" if a.replacement_mode == "new_vs_new" else "full EV manufacturing (early retirement): body " + f"{a.ev_vehicle_mfg_excl_battery_t:g} t + battery {a.battery_kwh:g} × {a.battery_kg_per_kwh:g} kg"} = **{m_kg:,.0f} kg**

**Cumulative net after year k:** N_k = Σ S_y − M. After year 1: {r0['saving_kg']:,.0f} − {m_kg:,.0f} = **{r0['cumulative_net_kg']:,.0f} kg**.
Payback = first year with N_k ≥ 0, interpolated: **{f"{pb['payback_years_fractional']:.2f} years" if pb["payback_year"] else "no payback within the lifetime"}**""")
        st.dataframe(t.assign(**{c: t[c].round(1) for c in ["grid_g_per_kwh", "petrol_kg", "ev_kg", "saving_kg",
                                                              "cumulative_net_kg"]}),
                     hide_index=True, width="stretch")
        st.caption("Grid values for years after the forecast horizon are held at the last forecast value.")

    st.subheader("Payback across candidate regions")
    cands, _ = clean_cands(st.session_state["cands"])
    rows = []
    for r in cands.itertuples():
        ar = replace(a, annual_km=float(r.annual_km))
        _, p_f = payback_for(r.country, year, ar, "forecast")
        _, p_c = payback_for(r.country, year, ar, "current")
        rows.append({"Region": r.country, "Payback forecast (yrs)": p_f["payback_years_fractional"],
                     "Payback current-data (yrs)": p_c["payback_years_fractional"],
                     "Lifetime net, forecast (t)": round(p_f["lifetime_net_kg"] / 1000, 1),
                     "Lifetime net, current-data (t)": round(p_c["lifetime_net_kg"] / 1000, 1)})
    df = pd.DataFrame(rows).sort_values("Lifetime net, forecast (t)", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Region"], y=df["Lifetime net, forecast (t)"], name="Forecast-aware",
                         marker=dict(color=BLUE, cornerradius=4)))
    fig.add_trace(go.Bar(x=df["Region"], y=df["Lifetime net, current-data (t)"], name="Current-data",
                         marker=dict(color=ORANGE, cornerradius=4)))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    show(ui.style(fig, 360, f"Lifetime net savings per EV deployed in {year}", "t CO₂e"))
    st.dataframe(df.round(2), hide_index=True, width="stretch")
    st.caption("Regions come from the Deployment Optimizer candidate table (each with its own annual km). "
               "An empty payback cell means no payback within the vehicle lifetime.")


# ================================================================== OPTIMIZER
def _apply_editor():
    ed = st.session_state["cands_editor"]
    df = st.session_state["cands"].copy()
    for idx, ch in ed.get("edited_rows", {}).items():
        for col, v in ch.items():
            df.at[int(idx), col] = v
    if ed.get("added_rows"):
        df = pd.concat([df, pd.DataFrame(ed["added_rows"])], ignore_index=True)
    if ed.get("deleted_rows"):
        df = df.drop(index=[i for i in ed["deleted_rows"] if i in df.index])
    st.session_state["cands"] = df.reset_index(drop=True)


def _load_upload():
    up = st.session_state.get("cands_upload")
    if up is None:
        return
    try:
        df = pd.read_csv(up)
        df.columns = [c.strip().lower() for c in df.columns]
        missing = [c for c in ["country", "fleet_available", "cost_per_ev", "annual_km"] if c not in df]
        if missing:
            st.session_state["upload_msg"] = ("error", f"CSV is missing columns: {', '.join(missing)}")
            return
        if "min_vehicles" not in df:
            df["min_vehicles"] = 0
        st.session_state["cands"] = df[CAND_COLS].reset_index(drop=True)
        st.session_state["upload_msg"] = ("ok", f"Loaded {len(df)} regions from {up.name}.")
    except Exception as e:  # malformed file: report, never crash the app
        st.session_state["upload_msg"] = ("error", f"Could not read CSV: {e}")


STRAT = {"B": ("B · Forecast-aware", BLUE), "A": ("A · Current-data", ORANGE), "C": ("C · Robust (max-min)", ui.SERIES[2])}
SCEN_LABEL = {"current": "Grid frozen", "hi80": "Slow decarb. (90th pct)", "forecast": "Forecast central",
              "lo80": "Fast decarb. (10th pct)"}


def page_optimizer():
    st.markdown(ui.header("Module 5 · Engine 3", "Deployment Optimizer",
                          "Mixed-integer linear program (SciPy HiGHS): choose whole EVs per region and "
                          "year to maximize estimated net emissions savings by the target year, within the "
                          "budget, regional fleet sizes, yearly capacity and any manager overrides."),
                unsafe_allow_html=True)
    st.write("")
    c = st.columns([1.2, 1, 1.4, 1, 1])
    c[0].number_input("Budget ($)", 100_000, 1_000_000_000, step=500_000, key="budget", format="%d")
    c[1].number_input("Target year", 2027, 2045, step=1, key="target_year")
    c[2].slider("Deployment years", 2026, 2035, key="deploy_range")
    c[3].number_input("Max EVs per year (0 = none)", 0, 100_000, step=10, key="max_per_year")
    c[4].number_input("EV cost decline per year (%)", 0.0, 20.0, step=0.5, key="cost_decline")

    st.markdown(f"**Candidate regions** {ui.pill('HYPOTHETICAL fleet and cost', 'hypo')} "
                f"{ui.pill('REAL grid data', 'real')}", unsafe_allow_html=True)
    u1, u2 = st.columns([3, 1])
    with u1:
        st.file_uploader("Upload your own fleet (CSV)", type=["csv"], key="cands_upload", on_change=_load_upload,
                         help="Columns: country, fleet_available, min_vehicles (optional), cost_per_ev, annual_km")
    with u2:
        st.write("")
        st.download_button("Download CSV template", pl.DEFAULT_CANDIDATES.to_csv(index=False),
                           "fleet_template.csv", "text/csv", width="stretch")
    msg = st.session_state.pop("upload_msg", None)
    if msg:
        (st.success if msg[0] == "ok" else st.error)(msg[1])
    st.data_editor(st.session_state["cands"], key="cands_editor", on_change=_apply_editor,
                   num_rows="dynamic", width="stretch", hide_index=True,
                   column_config={
                       "country": st.column_config.SelectboxColumn("Region", options=FC_COUNTRIES, required=True),
                       "fleet_available": st.column_config.NumberColumn("Petrol vehicles replaceable", min_value=0,
                                                                        step=1),
                       "min_vehicles": st.column_config.NumberColumn(
                           "Manager minimum (override)", min_value=0, step=1,
                           help="Human-in-the-loop: force at least this many EVs here (equity, politics, "
                                "depot readiness). The app shows what the override costs in tonnes."),
                       "cost_per_ev": st.column_config.NumberColumn("Cost per EV ($)", min_value=1, step=1000,
                                                                    format="dollar"),
                       "annual_km": st.column_config.NumberColumn("Annual km per vehicle", min_value=1, step=500),
                   })
    cands, issues, cmp, bt = current_runs()
    for i in issues:
        st.warning(i)
    if cmp is None:
        st.error("Add at least one valid candidate region.")
        return
    if st.session_state["target_year"] < deploy_years()[0]:
        st.warning("Target year is before the first deployment year, so no savings can be counted.")
    R, sc, target = cmp, cmp["scores"], st.session_state["target_year"]
    B_ = R["B"]
    if B_.get("infeasible"):
        st.error("The manager minimums cannot all be met within the budget and yearly capacity. "
                 "Lower an override or raise the budget.")
        return

    st.subheader("Recommended plan: forecast-aware strategy")
    c = st.columns(4)
    c[0].markdown(ui.card(f"Net savings by {target}", f"{B_['planned_net_t']:,.0f}", "t CO₂e",
                          f"80% band {sc['B']['hi80']:,.0f} to {sc['B']['lo80']:,.0f} t", accent=True),
                  unsafe_allow_html=True)
    c[1].markdown(ui.card("EVs deployed", f"{B_['vehicles']:,}", "",
                          f"of {int(cands['fleet_available'].sum()):,} replaceable"), unsafe_allow_html=True)
    c[2].markdown(ui.card("Budget per tonne", f"${B_['cost_per_t']:,.0f}" if B_.get("cost_per_t") else "n/a",
                          "per t CO₂e", f"Spend ${B_['total_spend'] / 1e6:,.2f}M (gross EV cost; excludes fuel "
                          f"savings and the petrol car not bought)"), unsafe_allow_html=True)
    c[3].markdown(ui.card("Solver", "Optimal" if B_["success"] else "Check", "", str(B_["status"])[:60]),
                  unsafe_allow_html=True)
    if R["mins"]:
        U = R["unconstrained_B"]
        price = U["planned_net_t"] - B_["planned_net_t"]
        st.write("")
        st.markdown(ui.callout(
            f"<b>Human-in-the-loop overrides active:</b> {', '.join(f'{k} ≥ {v}' for k, v in R['mins'].items())}. "
            f"Without them the optimizer would save {U['planned_net_t']:,.0f} t. "
            f"<b>Price of your overrides: {price:,.0f} t CO₂e</b> "
            f"({price / U['planned_net_t'] * 100 if U['planned_net_t'] else 0:.1f}% of the optimum). "
            "The decision stays yours; the cost is made visible.", warn=price > 0), unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns([1.25, 1])
    with col1:
        alloc = B_["allocation"]
        piv = alloc.pivot_table(index="country", columns="year", values="vehicles", aggfunc="sum", fill_value=0)
        piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]
        fig = go.Figure(go.Heatmap(z=piv.values, x=[str(x) for x in piv.columns], y=piv.index,
                                   colorscale=[[0, ui.SURFACE_2], [1e-6, ui.SEQ_BLUE[1]], [1, ui.SEQ_BLUE[5]]],
                                   xgap=2, ygap=2, text=np.where(piv.values > 0, piv.values.astype(str), ""),
                                   texttemplate="%{text}", textfont=dict(color=ui.INK), showscale=False,
                                   hovertemplate="%{y} %{x}: %{z} EVs<extra></extra>"))
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        show(ui.style(fig, 360, "Recommended allocation: EVs by region and year", legend=False))
    with col2:
        st.markdown("**Why this allocation** (generated from the solved numbers)")
        for line in explain(B_):
            st.markdown(f"- {line}")

    st.subheader("Three strategies, four possible futures")
    st.markdown(ui.callout(
        "All strategies get the <b>same budget, candidates, assumptions, overrides and target year</b>; only "
        "the grid assumption used for planning differs. <b>C</b> maximizes the <i>worst</i> outcome across all "
        "four futures. Each plan is optimal under its own assumption by construction, so the "
        "<b>regret</b> row (tonnes lost versus the best plan in the worst case) and the historical test below "
        "are the fair comparisons."), unsafe_allow_html=True)
    st.write("")
    regions = lambda r: ", ".join(r["allocation"].query("vehicles > 0").groupby("country")["vehicles"].sum()
                                  .sort_values(ascending=False).index)
    rows = {"Metric": ["Vehicles", "Budget per tonne ($/t gross, scored on forecast central)", "Regions"]
            + [f"Net t: {SCEN_LABEL[k]}" for k in pl.SCENARIOS] + ["Worst-case regret (t)"]}
    for k in ["A", "B", "C"]:
        r = R[k]
        rows[STRAT[k][0]] = ([r["vehicles"], f"{r['total_spend'] / sc[k]['forecast']:,.0f}" if sc[k]["forecast"] > 0
                              else "n/a", regions(r)]
                             + [f"{sc[k][s]:,.0f}" for s in pl.SCENARIOS]
                             + [f"{max(R['regret'][k].values()):,.0f}"])
    st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True, width="stretch")
    fig = go.Figure()
    for k in ["B", "A", "C"]:
        fig.add_trace(go.Bar(x=[SCEN_LABEL[s] for s in pl.SCENARIOS], y=[sc[k][s] for s in pl.SCENARIOS],
                             name=STRAT[k][0], marker=dict(color=STRAT[k][1], cornerradius=4,
                                                           line=dict(color=ui.SURFACE, width=2)),
                             hovertemplate="%{x}: %{y:,.0f} t<extra>" + STRAT[k][0] + "</extra>"))
    fig.update_layout(barmode="group", bargap=0.3)
    show(ui.style(fig, 340, "Each plan scored under each grid future", f"Net t CO₂e by {target}"))
    same_ab = (R["A"]["allocation"]["vehicles"].to_numpy() == B_["allocation"]["vehicles"].to_numpy()).all()
    if same_ab:
        st.info("Strategies A and B chose the identical allocation here: the forecast did not change the decision.")

    st.subheader("Historical decision backtest")
    st.selectbox("Plan as of", pl.BACKTEST_ORIGINS, key="bt_origin",
                 help="The forecast comes from a model trained only on data up to this year.")
    if not bt.get("usable"):
        st.info("No candidate region has enough history for this backtest origin.")
    else:
        r = bt["realized_t"]
        o = bt["origin"]
        diff = r["forecast"] - r["current"]
        st.markdown(ui.callout(
            f"Planned <b>as of {o}</b> (deployments {bt['deploy_years'][0]}-{bt['deploy_years'][-1]}, savings "
            f"counted to {bt['target_year']}), then scored on the <b>actual</b> grids. Forecast-aware realized "
            f"<b>{r['forecast']:,.0f} t</b> vs <b>{r['current']:,.0f} t</b> for current-data "
            f"({diff / r['current'] * 100 if r['current'] else 0:+.2f}%); perfect foresight {r['actual']:,.0f} t. "
            + ("Forecasting improved this decision." if diff > 1e-6 else
               "Forecasting did <b>not</b> improve this decision." if diff < -1e-6 else
               "Both plans performed identically."), warn=diff < -1e-6), unsafe_allow_html=True)
        st.write("")
        col1, col2 = st.columns([1, 1.2])
        with col1:
            fig = go.Figure(go.Bar(
                x=["Current-data plan", "Forecast-aware plan", "Perfect foresight"],
                y=[r["current"], r["forecast"], r["actual"]],
                marker=dict(color=[ORANGE, BLUE, ui.MUTED], cornerradius=4),
                text=[f"{v:,.0f} t" for v in [r["current"], r["forecast"], r["actual"]]],
                textposition="outside", textfont=dict(color=ui.INK_2),
                hovertemplate="%{x}: %{y:,.0f} t<extra></extra>"))
            fig.update_yaxes(range=[0, max(r.values()) * 1.15])
            show(ui.style(fig, 320, f"Realized net savings {o + 1}-{bt['target_year']}", "t CO₂e", legend=False))
        with col2:
            rows = []
            for k, lbl in [("current", "Current-data"), ("forecast", "Forecast-aware"), ("actual", "Perfect foresight")]:
                al = bt["results"][k]["allocation"].query("vehicles > 0")
                for (cty, yr), v in al.groupby(["country", "year"])["vehicles"].sum().items():
                    rows.append({"Plan": lbl, "Region": cty, "Year": yr, "EVs": int(v)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=320)
        if bt["skipped"]:
            st.caption(f"Excluded (insufficient data for this origin): {', '.join(bt['skipped'])}.")

    study_section()

    with st.expander("Per-vehicle value table used by the optimizer (forecast-aware)"):
        vt = R["value_tables"]["forecast"].copy()
        vt["net_t_per_vehicle"] = vt["net_kg_per_vehicle"] / 1000
        st.dataframe(vt[["country", "year", "years_counted", "avg_grid_g_per_kwh", "net_t_per_vehicle",
                         "cost_per_ev", "net_t_per_100k"]].round(2), hide_index=True, width="stretch")


def study_section():
    ss, df = D["study_summary"], D["study"]
    if not ss:
        return
    o = ss["overall"]
    des = ss["design"]
    st.subheader(f"Does forecasting actually help? {o['n_scenarios']} historical scenarios")
    st.caption(f"Origins {des['origins'][0]}-{des['origins'][-1]}, {des['sets_per_origin']} random sets of "
               f"{des['regions_per_set']} grids each, ${des['budget'] / 1e6:.0f}M budget, a separate model "
               "trained for each origin on data up to that year only. Run: python scripts/backtest_study.py")
    c = st.columns(4)
    c[0].markdown(ui.card("Plans identical", f"{o['identical_plans'] / o['n_scenarios'] * 100:.0f}%", "",
                          "Forecast did not change where to invest"), unsafe_allow_html=True)
    c[1].markdown(ui.card("Forecast better / worse", f"{o['forecast_wins']} / {o['current_wins']}", "",
                          f"Mean effect {o['mean_diff_pct_all']:+.2f}% realized savings"), unsafe_allow_html=True)
    c[2].markdown(ui.card("Perfect-foresight ceiling", f"+{o['mean_perfect_gain_pct']:.2f}%", "",
                          "Max possible gain from knowing the future grid"), unsafe_allow_html=True)
    c[3].markdown(ui.card("Savings estimate bias", f"{o['mean_est_err_forecast_pct']:+.1f}%", "",
                          f"Forecast estimate vs {o['mean_est_err_current_pct']:+.1f}% for current-data "
                          f"(closer in {o['forecast_estimate_closer_share']:.0%} of cases)", accent=True),
                  unsafe_allow_html=True)
    st.write("")
    by = pd.DataFrame(ss["by_origin"]).T.reset_index().rename(columns={"index": "origin"})
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=by["origin"], y=by["mean_est_err_forecast_pct"].astype(float), name="Forecast-aware",
                             marker=dict(color=BLUE, cornerradius=4)))
        fig.add_trace(go.Bar(x=by["origin"], y=by["mean_est_err_current_pct"].astype(float), name="Current-data",
                             marker=dict(color=ORANGE, cornerradius=4)))
        fig.add_hline(y=0, line=dict(color=ui.MUTED, width=1))
        fig.update_layout(barmode="group", bargap=0.3)
        show(ui.style(fig, 330, "Promised vs delivered savings, by year the plan was made", "Mean % error"))
    with col2:
        d2 = df[~df["identical_plans"]]
        fig = go.Figure(go.Histogram(x=d2["diff_pct"], nbinsx=30, marker=dict(color=BLUE, line=dict(color=ui.SURFACE, width=1)),
                                     hovertemplate="%{x}: %{y} scenarios<extra></extra>"))
        fig.add_vline(x=0, line=dict(color=ui.MUTED, width=1, dash="dot"))
        show(ui.style(fig, 330, f"Realized gain when plans differ (n={len(d2)})", "Scenarios",
                      "Forecast-aware vs current-data, %", legend=False))
    st.markdown(ui.callout(
        f"<b>Finding.</b> For choosing <i>where</i> to deploy, today's grid gap dominates: even perfect knowledge of "
        f"future grids would have improved realized savings by only {o['mean_perfect_gain_pct']:.2f}% on average, "
        f"and the forecast captured {o['share_of_gap_closed_pct']:.0f}% of that small gap. The forecast's real "
        f"value is the <i>size</i> of the promise: plans based on today's grid systematically under-promised "
        f"({o['mean_est_err_current_pct']:+.1f}%), while forecast-based estimates were nearly unbiased "
        f"({o['mean_est_err_forecast_pct']:+.1f}%). Budget holders get a number they can defend."),
        unsafe_allow_html=True)


# ================================================================== REPORT
def page_report():
    st.markdown(ui.header("Module 6", "Strategy Report",
                          "A decision-maker summary built from the current scenario. Download it as "
                          "Markdown, or download the underlying tables as CSV."), unsafe_allow_html=True)
    cands, _, cmp, bt = current_runs()
    if cmp is None:
        st.error("No valid candidate regions; configure them in Deployment Optimizer.")
        return
    md = report.build_markdown(cmp=cmp, bt=bt, metrics=MET, manifest=D["manifest"], a=A(),
                               budget=float(st.session_state["budget"]), target_year=int(st.session_state["target_year"]),
                               deploy_years=deploy_years(), max_per_year=int(st.session_state["max_per_year"]) or None,
                               cost_decline=float(st.session_state["cost_decline"]), cands=cands,
                               study=D["study_summary"])
    c = st.columns(4)
    c[0].download_button("Report (.md)", md, "carbon_payback_atlas_report.md", "text/markdown",
                         width="stretch")
    alloc = cmp["B"]["allocation"].query("vehicles > 0")
    c[1].download_button("Allocation (.csv)", alloc.to_csv(index=False), "recommended_allocation.csv", "text/csv",
                         width="stretch")
    c[2].download_button("Value table (.csv)", cmp["value_tables"]["forecast"].to_csv(index=False),
                         "value_table_forecast.csv", "text/csv", width="stretch")
    c[3].download_button("Validation metrics (.json)", json.dumps(MET, indent=2), "validation_metrics.json",
                         "application/json", width="stretch")
    st.caption("PDF export is not built in: open the Markdown in any editor or print this page to PDF.")
    st.divider()
    st.markdown(md)


tour_banner()
{"Overview": page_overview, "Global Atlas": page_atlas, "AI Forecast": page_forecast,
 "Fleet Lab": page_fleet, "Payback Engine": page_payback, "Deployment Optimizer": page_optimizer,
 "Strategy Report": page_report}[page]()
