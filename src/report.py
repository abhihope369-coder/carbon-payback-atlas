"""Strategy report generation (Markdown + CSV) from actual computed results."""
from __future__ import annotations

from datetime import date

import pandas as pd

from .lifecycle import VehicleAssumptions
from .optimization import explain


def _alloc_table(alloc: pd.DataFrame) -> str:
    a = alloc[alloc["vehicles"] > 0]
    if a.empty:
        return "_No vehicles allocated._\n"
    piv = a.pivot_table(index="country", columns="year", values="vehicles", aggfunc="sum", fill_value=0)
    cols = [str(c) for c in piv.columns]
    out = "| Region | " + " | ".join(cols) + " | Total |\n|---|" + "---:|" * (len(cols) + 1) + "\n"
    for r, row in piv.iterrows():
        out += f"| {r} | " + " | ".join(str(int(v)) for v in row) + f" | {int(row.sum())} |\n"
    return out


def build_markdown(*, cmp: dict, bt: dict, metrics: dict, manifest: dict, a: VehicleAssumptions,
                   budget: float, target_year: int, deploy_years: list[int], max_per_year,
                   cost_decline: float, cands: pd.DataFrame, study: dict | None = None) -> str:
    A, B, C, sc = cmp["A"], cmp["B"], cmp["C"], cmp["scores"]
    cpt = lambda r: f"${r['cost_per_t']:,.0f}/t" if r.get("cost_per_t") else "n/a"
    test = metrics["phases"]["test"]["metrics"]
    chosen = metrics["chosen_model"]
    lines = [
        "# Carbon Payback Atlas: Strategy Report",
        f"_Generated {date.today().isoformat()}. Fleet, cost and budget figures are hypothetical "
        "scenario inputs; grid data are real (Ember via Our World in Data)._",
        "",
        "## 1. Recommended deployment strategy (forecast-aware)",
        f"- Budget: ${budget:,.0f}; spent ${B['total_spend']:,.0f}; remaining ${B['remaining_budget']:,.0f}",
        f"- Vehicles replaced: {B['vehicles']}",
        f"- Estimated net emissions reduction by {target_year}: **{B['planned_net_t']:,.0f} t CO2e** "
        f"(80% forecast band: {sc['B']['hi80']:,.0f} to {sc['B']['lo80']:,.0f} t)",
        f"- Budget per tonne: {cpt(B)} (gross EV spend / net tonnes by the target year; this is not a net "
        "abatement cost, because it excludes fuel savings and the petrol car that would otherwise be bought)",
        f"- Deployment years considered: {deploy_years[0]}-{deploy_years[-1]}; max per year: "
        f"{max_per_year or 'no limit'}; annual EV cost decline: {cost_decline}%",
        "",
        _alloc_table(B["allocation"]),
        "### Why this allocation",
        *[f"- {s}" for s in explain(B)],
        "",
        "## 2. Strategy comparison (same budget, candidates, assumptions, overrides, target year)",
        "| Metric | A: current-data | B: forecast-aware | C: robust (max-min) |",
        "|---|---:|---:|---:|",
        f"| Vehicles | {A['vehicles']} | {B['vehicles']} | {C['vehicles']} |",
        "| Budget per tonne (gross, scored on forecast central) | "
        + " | ".join(f"${x['total_spend'] / sc[k]['forecast']:,.0f}/t" if sc[k]["forecast"] > 0 else "n/a"
                     for k, x in [("A", A), ("B", B), ("C", C)]) + " |",
        *[f"| Net t CO2e: {lbl} | {sc['A'][k]:,.0f} | {sc['B'][k]:,.0f} | {sc['C'][k]:,.0f} |"
          for k, lbl in [("current", "grid frozen at today"), ("hi80", "slow decarbonization (90th pct)"),
                         ("forecast", "forecast central"), ("lo80", "fast decarbonization (10th pct)")]],
        f"| Worst-case regret (t) | {max(cmp['regret']['A'].values()):,.0f} | "
        f"{max(cmp['regret']['B'].values()):,.0f} | {max(cmp['regret']['C'].values()):,.0f} |",
        "",
        "Note: each plan is optimal under its own planning assumption by construction. The "
        "historical tests below are the fair comparison.",
        "",
        "Strategy A allocation:", "", _alloc_table(A["allocation"]),
        "Strategy C allocation:", "", _alloc_table(C["allocation"]),
    ]
    if cmp.get("mins"):
        U = cmp["unconstrained_B"]
        lines += ["", "### Manager overrides (human-in-the-loop)",
                  "- Minimums: " + ", ".join(f"{k} >= {v}" for k, v in cmp["mins"].items()),
                  f"- Unconstrained optimum: {U['planned_net_t']:,.0f} t; with overrides: {B['planned_net_t']:,.0f} t; "
                  f"price of overrides: {U['planned_net_t'] - B['planned_net_t']:,.0f} t", ""]
    if bt.get("usable"):
        r = bt["realized_t"]
        diff = r["forecast"] - r["current"]
        verdict = ("outperformed" if diff > 0 else "underperformed" if diff < 0 else "matched")
        lines += [
            f"## 3. Historical decision backtest (plan as of {bt['origin']}, scored on actual "
            f"{bt['origin'] + 1}-{bt['target_year']} grids)",
            f"- Current-data plan realized: {r['current']:,.0f} t CO2e",
            f"- Forecast-aware plan (model trained only on targets <= {bt['origin']}) realized: {r['forecast']:,.0f} t CO2e",
            f"- Perfect-foresight upper bound: {r['actual']:,.0f} t CO2e",
            f"- Result: the forecast-aware plan **{verdict}** the current-data plan by {abs(diff):,.0f} t "
            f"({(diff / r['current'] * 100) if r['current'] else 0:+.1f}%).",
            f"- Regions excluded from backtest (insufficient data): {', '.join(bt['skipped']) or 'none'}",
            "",
        ]
    if study:
        o = study["overall"]
        lines += [
            f"### Multi-origin study ({o['n_scenarios']} scenarios, origins "
            f"{study['design']['origins'][0]}-{study['design']['origins'][-1]})",
            f"- Plans identical: {o['identical_plans']} ({o['identical_plans'] / o['n_scenarios'] * 100:.0f}%); "
            f"forecast better {o['forecast_wins']}, current-data better {o['current_wins']}",
            f"- Mean realized effect of forecasting: {o['mean_diff_pct_all']:+.2f}%; perfect-foresight ceiling "
            f"{o['mean_perfect_gain_pct']:+.2f}%",
            f"- Savings estimate bias: current-data {o['mean_est_err_current_pct']:+.1f}%, forecast-aware "
            f"{o['mean_est_err_forecast_pct']:+.1f}% (forecast closer in {o['forecast_estimate_closer_share']:.0%})",
            "- Interpretation: forecasting rarely changes WHERE to invest here, but makes the size of the "
            "promised savings more accurate.", ""]
    lines += [
        "## 4. Forecast model (Grid Future AI)",
        f"- Model: {chosen}, direct multi-horizon (1-{metrics['horizon_years']} years), "
        "target = log ratio of future to current intensity",
        f"- Held-out test (target years >= 2020, n={test[chosen]['n']}): MAE {test[chosen]['mae']:.1f} g/kWh, "
        f"RMSE {test[chosen]['rmse']:.1f} g/kWh",
        f"- Persistence baseline: MAE {test['persistence']['mae']:.1f}; linear-trend baseline: "
        f"MAE {test['linear_trend']['mae']:.1f}",
        f"- 80% interval empirical coverage on test: "
        f"{metrics['phases']['test']['interval_80_coverage_on_test']:.1%} (nominal 80%)",
        "- Beyond the 10-year horizon, intensity is held flat at the last forecast value (assumption).",
        "",
        "## 5. Vehicle and lifecycle assumptions",
        "| Assumption | Value |", "|---|---:|",
        *[f"| {k} | {v} |" for k, v in a.to_dict().items()],
        "",
        "Candidate regions:", "",
        "| Region | Fleet available | Manager minimum | Cost per EV ($) | Annual km |", "|---|---:|---:|---:|---:|",
        *[f"| {r.country} | {int(r.fleet_available)} | {int(r.min_vehicles)} | {r.cost_per_ev:,.0f} | {r.annual_km:,.0f} |"
          for r in cands.itertuples()],
        "",
        "## 6. Methodology and limitations",
        "- Grid intensity: Ember lifecycle g CO2e/kWh of generation (not consumption; imports ignored).",
        "- Petrol: well-to-wheel factor (UK DESNZ 2025: 2.069 tailpipe + 0.581 upstream kg CO2e/L) "
        "so both sides include upstream emissions.",
        "- Battery: 60 kg CO2e/kWh (ICCT 2021). Non-battery manufacturing treated as equal for "
        "both powertrains (ICCT 2021); battery counted once.",
        "- Excluded: maintenance, end-of-life, recycling credits, infrastructure.",
        "- Forecasts reflect historical patterns (2000-2025) and cannot anticipate policy shocks.",
        "- Results are estimates for decision support, not causal claims.",
        "",
        "## 7. Data sources",
        f"- {manifest['source_name']}: {manifest['repository']} (accessed {manifest['date_accessed']}, "
        f"{manifest['licence']})",
        "- UK DESNZ Greenhouse gas reporting: conversion factors 2025",
        "- ICCT (2021) A global comparison of the life-cycle GHG emissions of combustion engine "
        "and electric passenger cars",
        "- Argonne National Laboratory GREET (vehicle manufacturing range)",
    ]
    return "\n".join(lines)
