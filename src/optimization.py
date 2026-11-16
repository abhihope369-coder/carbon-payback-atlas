"""Deployment Optimizer: budget-constrained allocation of EVs across regions and years.

Mixed-integer linear program (solved with SciPy's HiGHS-based `milp`):

  decision   x[r, y] = whole EVs deployed in region r in year y   (integer >= 0)
  maximize   sum_{r,y} N[r, y] * x[r, y]
  subject to sum_{r,y} cost[r, y] * x[r, y] <= budget
             sum_y x[r, y] <= fleet_available[r]          (per region)
             sum_r x[r, y] <= max_per_year                (optional, per year)

N[r, y] is the per-vehicle net savings by the target year: operational savings
from year y to min(target, y + lifetime - 1), minus the additional manufacturing
emissions M. Options with N <= 0 are bounded to zero (never selected).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .lifecycle import VehicleAssumptions, additional_mfg_kg, ev_annual_kg, petrol_annual_kg


def value_table(candidates: pd.DataFrame, deploy_years: list[int], target_year: int,
                base: VehicleAssumptions, paths: dict[str, pd.Series],
                cost_decline_pct: float = 0.0) -> pd.DataFrame:
    """Per-vehicle economics for every (region, deployment year) option.

    candidates columns: country, fleet_available, cost_per_ev, annual_km
    paths: country -> grid intensity path (year -> g/kWh) for THIS strategy.
    """
    rows = []
    y0 = min(deploy_years)
    for _, c in candidates.iterrows():
        a = replace(base, annual_km=float(c["annual_km"]))
        m = additional_mfg_kg(a)
        e_pet = petrol_annual_kg(a)
        path = paths[c["country"]]
        for y in deploy_years:
            if y > target_year:
                continue
            last = min(target_year, y + a.lifetime_years - 1)
            yrs = list(range(y, last + 1))
            g = path.loc[yrs].to_numpy(dtype=float)
            s = e_pet - ev_annual_kg(a, g)
            cost = float(c["cost_per_ev"]) * (1 - cost_decline_pct / 100.0) ** (y - y0)
            rows.append({
                "country": c["country"], "year": int(y), "years_counted": len(yrs),
                "avg_grid_g_per_kwh": float(g.mean()), "operational_saving_kg": float(s.sum()),
                "additional_mfg_kg": m, "net_kg_per_vehicle": float(s.sum() - m),
                "cost_per_ev": cost, "fleet_available": int(c["fleet_available"]),
            })
    vt = pd.DataFrame(rows)
    vt["net_t_per_100k"] = vt["net_kg_per_vehicle"] / 1000 / vt["cost_per_ev"] * 100_000
    return vt


def _constraints(vt: pd.DataFrame, budget: float, max_per_year: int | None,
                 min_by_region: dict | None, n_extra: int = 0):
    """Shared constraint rows. `n_extra` zero-columns are appended (e.g. robust t)."""
    pad = lambda M: np.hstack([M, np.zeros((M.shape[0], n_extra))]) if n_extra else M
    cons = [LinearConstraint(pad(vt["cost_per_ev"].to_numpy()[None, :]), -np.inf, budget)]
    regions = vt["country"].unique()
    A_r = np.array([(vt["country"] == r).to_numpy(dtype=float) for r in regions])
    fleet = vt.groupby("country")["fleet_available"].first().loc[regions].to_numpy(dtype=float)
    mins = np.array([float((min_by_region or {}).get(r, 0) or 0) for r in regions])
    cons.append(LinearConstraint(pad(A_r), np.minimum(mins, fleet), fleet))
    if max_per_year:
        years = sorted(vt["year"].unique())
        A_y = np.array([(vt["year"] == y).to_numpy(dtype=float) for y in years])
        cons.append(LinearConstraint(pad(A_y), -np.inf, float(max_per_year)))
    return cons, mins


def _upper_bounds(vt: pd.DataFrame, positive: np.ndarray, min_by_region: dict | None) -> np.ndarray:
    """Never select an option with no net benefit, unless a manager override forces that region."""
    forced = vt["country"].map(lambda r: float((min_by_region or {}).get(r, 0) or 0) > 0).to_numpy()
    return np.where(positive | forced, vt["fleet_available"], 0).astype(float)


def _package(vt: pd.DataFrame, x, res, budget: float) -> dict:
    alloc = vt.assign(vehicles=x)
    alloc["spend"] = alloc["vehicles"] * alloc["cost_per_ev"]
    alloc["net_kg"] = alloc["vehicles"] * alloc["net_kg_per_vehicle"]
    spend = float(alloc["spend"].sum())
    assert spend <= budget + 1e-6, "Budget constraint violated"
    net_t = float(alloc["net_kg"].sum() / 1000)
    return {
        "status": res.message, "success": bool(res.success),
        "allocation": alloc,
        "total_spend": spend, "remaining_budget": float(budget - spend),
        "vehicles": int(alloc["vehicles"].sum()),
        "planned_net_t": net_t,
        "cost_per_t": spend / net_t if net_t > 0 else None,
    }


def optimize(vt: pd.DataFrame, budget: float, max_per_year: int | None = None,
             min_by_region: dict | None = None, time_limit: float = 20.0) -> dict:
    """Maximize net savings. `min_by_region` = manager overrides (minimum EVs per region)."""
    n = len(vt)
    if n == 0:
        return {"status": "no options", "success": False, "allocation": vt.assign(vehicles=0)}
    c = -vt["net_kg_per_vehicle"].to_numpy()
    cons, _ = _constraints(vt, budget, max_per_year, min_by_region)
    ub = _upper_bounds(vt, vt["net_kg_per_vehicle"].to_numpy() > 0, min_by_region)
    res = milp(c=c, constraints=cons, integrality=np.ones(n), bounds=Bounds(np.zeros(n), ub),
               options={"time_limit": time_limit})
    if res.x is None:
        return {"status": res.message, "success": False, "infeasible": True,
                "allocation": vt.assign(vehicles=0, spend=0.0, net_kg=0.0), "total_spend": 0.0,
                "remaining_budget": budget, "vehicles": 0, "planned_net_t": 0.0, "cost_per_t": None}
    return _package(vt, np.round(res.x).astype(int), res, budget)


def optimize_robust(vts: dict[str, pd.DataFrame], budget: float, max_per_year: int | None = None,
                    min_by_region: dict | None = None, time_limit: float = 20.0) -> dict:
    """Max-min robust plan: maximize the WORST net savings across grid scenarios.

    Variables [x_1..x_n, t]; maximize t subject to t <= sum_i N_s[i] x_i for every scenario s.
    All value tables must list the same options in the same order.
    """
    keys = list(vts)
    base = vts[keys[0]]
    n = len(base)
    N = np.array([vts[k]["net_kg_per_vehicle"].to_numpy() for k in keys])  # scenarios x n
    cons, _ = _constraints(base, budget, max_per_year, min_by_region, n_extra=1)
    cons.append(LinearConstraint(np.hstack([-N, np.ones((len(keys), 1))]), -np.inf, 0.0))
    c = np.zeros(n + 1)
    c[-1] = -1.0
    ub = np.append(_upper_bounds(base, (N > 0).any(axis=0), min_by_region), np.inf)
    lb = np.append(np.zeros(n), -np.inf)
    integ = np.append(np.ones(n), 0)
    res = milp(c=c, constraints=cons, integrality=integ, bounds=Bounds(lb, ub),
               options={"time_limit": time_limit})
    if res.x is None:
        return {"status": res.message, "success": False, "infeasible": True,
                "allocation": base.assign(vehicles=0, spend=0.0, net_kg=0.0), "total_spend": 0.0,
                "remaining_budget": budget, "vehicles": 0, "planned_net_t": 0.0, "cost_per_t": None}
    out = _package(base, np.round(res.x[:n]).astype(int), res, budget)
    out["worst_case_t"] = float(res.x[-1] / 1000)
    return out


def evaluate(allocation: pd.DataFrame, vt_eval: pd.DataFrame) -> float:
    """Score an allocation under a (possibly different) value table, in t CO2e."""
    m = allocation[["country", "year", "vehicles"]].merge(
        vt_eval[["country", "year", "net_kg_per_vehicle"]], on=["country", "year"], how="left")
    return float((m["vehicles"] * m["net_kg_per_vehicle"]).sum() / 1000)


def explain(result: dict) -> list[str]:
    """Plain-language reasons generated from the solved numbers (no canned text)."""
    a = result["allocation"]
    chosen = a[a["vehicles"] > 0].sort_values("net_t_per_100k", ascending=False)
    if chosen.empty:
        return ["No option produced positive net savings within budget, so nothing was deployed."]
    lines = []
    # Same yardstick for chosen and skipped regions: best value among ALL of a region's options.
    best_any = a.groupby("country")["net_t_per_100k"].max()
    by_r = chosen.groupby("country").agg(vehicles=("vehicles", "sum"),
                                         net_kg=("net_kg", "sum"),
                                         grid=("avg_grid_g_per_kwh", "mean"))
    by_r["best"] = best_any.loc[by_r.index]
    by_r = by_r.sort_values("best", ascending=False)
    for r, row in by_r.iterrows():
        lines.append(f"{r}: {int(row.vehicles)} EVs, {row.net_kg / 1000:,.0f} t CO2e net by the target "
                     f"year. Average grid over its counted years {row.grid:,.0f} g/kWh; its best "
                     f"option yields {row.best:,.1f} t per $100k.")
    skipped = best_any[~best_any.index.isin(by_r.index)].sort_values(ascending=False)
    if len(skipped):
        weakest = by_r["best"].min()
        if skipped.iloc[0] < weakest:
            lines.append(f"Not selected: {', '.join(skipped.index)}. The strongest of these "
                         f"({skipped.index[0]}, {skipped.iloc[0]:,.1f} t per $100k) is below the weakest "
                         f"selected region ({weakest:,.1f}), so budget went elsewhere.")
        else:
            lines.append(f"Not selected: {', '.join(skipped.index)}. {skipped.index[0]} has a strong "
                         f"best-case option ({skipped.iloc[0]:,.1f} t per $100k), but with the per-year "
                         f"and per-region limits the solver found a higher total without it.")
    years = chosen.groupby("year")["vehicles"].sum()
    lines.append("Timing: " + ", ".join(f"{y}: {v}" for y, v in years.items())
                 + ". Earlier deployment counts more years of savings before the target year.")
    return lines
