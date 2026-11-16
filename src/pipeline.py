"""Glue between the three engines: used by the app, the report and the backtest study."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import optimization as opt
from .data_processing import PROCESSED_DIR, load_manifest, load_panel
from .lifecycle import VehicleAssumptions
from .scenarios import backtest_paths, build_path

ROOT = Path(__file__).resolve().parents[1]
BACKTEST_ORIGINS = [2010, 2011, 2012, 2013, 2014, 2015]
DEFAULT_BACKTEST_ORIGIN = 2014
LAST_SCORED_YEAR = 2024  # last year with near-complete actual coverage
SCENARIOS = ["current", "hi80", "forecast", "lo80"]  # grid futures used for scoring


def backtest_window(origin: int) -> tuple[list[int], int]:
    """Deploy in the 3 years after the origin; count savings up to origin+10 (max 2024)."""
    return [origin + 1, origin + 2, origin + 3], min(origin + 10, LAST_SCORED_YEAR)


def load_all() -> dict:
    bt = {}
    for o in BACKTEST_ORIGINS:
        p = PROCESSED_DIR / f"backtest_forecasts_{o}.csv"
        if p.exists():
            bt[o] = pd.read_csv(p)
    study_p = ROOT / "outputs" / "backtest_study.csv"
    return {
        "panel": load_panel(),
        "forecasts": pd.read_csv(PROCESSED_DIR / "forecasts.csv"),
        "backtests": bt,
        "metrics": json.loads((ROOT / "outputs" / "validation_metrics.json").read_text()),
        "manifest": load_manifest(),
        "test_predictions": pd.read_csv(ROOT / "outputs" / "test_predictions.csv"),
        "study": pd.read_csv(study_p) if study_p.exists() else None,
        "study_summary": (json.loads((ROOT / "outputs" / "backtest_study_summary.json").read_text())
                          if (ROOT / "outputs" / "backtest_study_summary.json").exists() else None),
    }


def _mins(cands: pd.DataFrame) -> dict:
    if "min_vehicles" not in cands:
        return {}
    return {r.country: int(r.min_vehicles) for r in cands.itertuples() if int(r.min_vehicles or 0) > 0}


def compare_strategies(d: dict, cands: pd.DataFrame, deploy_years: list[int], target_year: int,
                       a: VehicleAssumptions, budget: float, max_per_year: int | None,
                       cost_decline_pct: float) -> dict:
    """Three strategies with identical inputs; only the planning assumption differs.

    A  current-data : grid frozen at the latest observed value
    B  forecast-aware: Grid Future AI central forecast
    C  robust        : max-min over {frozen, slow, central, fast} grid futures
    Each plan is then scored under every scenario (regret table). Under its own
    scenario a plan is optimal by construction; the backtest is the fair test.
    Manager overrides (min_vehicles) apply to all strategies; the unconstrained
    optimum is also solved so the 'price' of the overrides can be shown.
    """
    paths = {k: {c: build_path(d["panel"], d["forecasts"], c, kind=k) for c in cands["country"]}
             for k in SCENARIOS}
    vts = {k: opt.value_table(cands, deploy_years, target_year, a, paths[k], cost_decline_pct)
           for k in SCENARIOS}
    mins = _mins(cands)
    res = {
        "A": opt.optimize(vts["current"], budget, max_per_year, mins),
        "B": opt.optimize(vts["forecast"], budget, max_per_year, mins),
        "C": opt.optimize_robust(vts, budget, max_per_year, mins),
    }
    scores = {s: {k: opt.evaluate(r["allocation"], vts[k]) for k in SCENARIOS} for s, r in res.items()}
    best = {k: max(scores[s][k] for s in scores) for k in SCENARIOS}
    regret = {s: {k: best[k] - scores[s][k] for k in SCENARIOS} for s in scores}
    unconstrained = opt.optimize(vts["forecast"], budget, max_per_year) if mins else None
    return {**res, "scores": scores, "regret": regret, "value_tables": vts, "paths": paths,
            "mins": mins, "unconstrained_B": unconstrained}


def run_backtest(d: dict, cands: pd.DataFrame, a: VehicleAssumptions, budget: float,
                 max_per_year: int | None, cost_decline_pct: float,
                 origin: int = DEFAULT_BACKTEST_ORIGIN) -> dict:
    """Plan 'as of origin' and score both strategies on what actually happened afterwards."""
    deploy, target = backtest_window(origin)
    bt = d["backtests"].get(origin)
    usable, skipped = [], []
    paths = {"current": {}, "forecast": {}, "actual": {}}
    for c in cands["country"]:
        try:
            p = backtest_paths(d["panel"], bt, c, origin)
        except (KeyError, TypeError):
            skipped.append(c)
            continue
        if not set(range(origin + 1, target + 1)) <= set(p["actual"].index):
            skipped.append(c)
            continue
        for k in paths:
            paths[k][c] = p[k]
        usable.append(c)
    cb = cands[cands["country"].isin(usable)]
    if cb.empty:
        return {"usable": [], "skipped": skipped, "origin": origin}
    vts = {k: opt.value_table(cb, deploy, target, a, paths[k], cost_decline_pct) for k in paths}
    mins = _mins(cb)
    res = {k: opt.optimize(vts[k], budget, max_per_year, mins) for k in paths}
    realized = {k: opt.evaluate(res[k]["allocation"], vts["actual"]) for k in res}
    return {"usable": usable, "skipped": skipped, "results": res, "realized_t": realized,
            "value_tables": vts, "paths": paths, "origin": origin, "deploy_years": deploy,
            "target_year": target}


DEFAULT_CANDIDATES = pd.DataFrame({
    "country": ["Canada", "United States", "Germany", "Poland", "China", "India",
                "South Africa", "Australia"],
    "fleet_available": [50] * 8,
    "min_vehicles": [0] * 8,
    "cost_per_ev": [50_000] * 8,
    "annual_km": [15_000] * 8,
})
