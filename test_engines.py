import itertools
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lifecycle import (VehicleAssumptions, additional_mfg_kg, ev_annual_kg,  # noqa: E402
                           payback_summary, payback_table, petrol_annual_kg)
from src.optimization import optimize, value_table  # noqa: E402


def test_spec_example_ev_charging():
    # Spec 5.5: 15,000 km x 0.18 kWh/km x 0.4 kg/kWh = 1,080 kg
    a = VehicleAssumptions(annual_km=15_000, ev_kwh_per_km=0.18)
    assert ev_annual_kg(a, 400) == pytest.approx(1080.0)


def test_petrol_well_to_wheel():
    a = VehicleAssumptions(annual_km=15_000, petrol_l_per_100km=8, petrol_kg_per_l=2.65)
    assert petrol_annual_kg(a) == pytest.approx(15_000 * 0.08 * 2.65)


def test_battery_not_double_counted():
    a = VehicleAssumptions()
    # new vs new: only the battery differs when glider emissions are equal
    assert additional_mfg_kg(a) == pytest.approx(a.battery_kwh * a.battery_kg_per_kwh)
    b = replace(a, replacement_mode="early_retirement")
    assert additional_mfg_kg(b) == pytest.approx(8000 + 3600)


def test_payback_found_and_interpolated():
    a = VehicleAssumptions()
    path = pd.Series(400.0, index=range(2026, 2040))
    t = payback_table(a, 2026, path)
    s = payback_summary(t, a)
    annual = petrol_annual_kg(a) - ev_annual_kg(a, 400)
    assert s["payback_years_fractional"] == pytest.approx(additional_mfg_kg(a) / annual)
    assert t["cumulative_net_kg"].iloc[0] == pytest.approx(annual - additional_mfg_kg(a))


def test_no_payback_is_reported():
    # Extremely dirty grid: EV emits more per year than petrol -> never pays back
    a = VehicleAssumptions()
    path = pd.Series(1500.0, index=range(2026, 2040))
    s = payback_summary(payback_table(a, 2026, path), a)
    assert s["payback_year"] is None and s["lifetime_net_kg"] < 0


def _toy():
    cands = pd.DataFrame({"country": ["A", "B", "C"], "fleet_available": [3, 4, 2],
                          "cost_per_ev": [50_000, 40_000, 60_000], "annual_km": [15_000, 20_000, 10_000]})
    paths = {"A": pd.Series(100.0, index=range(2020, 2060)),
             "B": pd.Series(np.linspace(700, 200, 40), index=range(2020, 2060)),
             "C": pd.Series(300.0, index=range(2020, 2060))}
    return cands, paths


def test_milp_matches_brute_force_and_respects_budget():
    cands, paths = _toy()
    a = VehicleAssumptions()
    vt = value_table(cands, [2026, 2027], 2032, a, paths, cost_decline_pct=5)
    budget = 230_000
    res = optimize(vt, budget, max_per_year=3)
    assert res["total_spend"] <= budget
    # brute force over all integer allocations
    best = -np.inf
    ranges = [range(0, int(f) + 1) for f in vt["fleet_available"]]
    for x in itertools.product(*ranges):
        x = np.array(x)
        if (x * vt["cost_per_ev"]).sum() > budget:
            continue
        if any(x[(vt["country"] == r).to_numpy()].sum() > f
               for r, f in zip(cands["country"], cands["fleet_available"])):
            continue
        if any(x[(vt["year"] == y).to_numpy()].sum() > 3 for y in [2026, 2027]):
            continue
        best = max(best, float((x * np.maximum(vt["net_kg_per_vehicle"], -np.inf)).sum()))
    assert res["planned_net_t"] * 1000 == pytest.approx(best, rel=1e-9)


def test_negative_options_never_selected():
    cands, paths = _toy()
    a = VehicleAssumptions()
    vt = value_table(cands, [2026], 2026, a, paths)  # 1 year of savings < manufacturing
    assert (vt["net_kg_per_vehicle"] < 0).all()
    res = optimize(vt, 1e9)
    assert res["vehicles"] == 0


def test_manager_minimums_are_honoured_and_cost_something():
    from src.optimization import optimize
    cands, paths = _toy()
    a = VehicleAssumptions()
    vt = value_table(cands, [2026, 2027], 2035, a, paths)
    free = optimize(vt, 250_000)
    forced = optimize(vt, 250_000, min_by_region={"C": 2})
    assert forced["allocation"].query("country == 'C'")["vehicles"].sum() >= 2
    assert forced["planned_net_t"] <= free["planned_net_t"] + 1e-9


def test_infeasible_minimums_reported():
    from src.optimization import optimize
    cands, paths = _toy()
    vt = value_table(cands, [2026], 2035, VehicleAssumptions(), paths)
    res = optimize(vt, 60_000, min_by_region={"A": 2, "B": 2})
    assert res.get("infeasible") and res["vehicles"] == 0


def test_robust_plan_maximizes_worst_case():
    from src.optimization import evaluate, optimize, optimize_robust
    cands, paths = _toy()
    a = VehicleAssumptions()
    scen = {"low": {k: p * 0.6 for k, p in paths.items()}, "high": {k: p * 1.4 for k, p in paths.items()}}
    vts = {k: value_table(cands, [2026, 2027], 2032, a, p) for k, p in scen.items()}
    rob = optimize_robust(vts, 230_000, max_per_year=3)
    worst = lambda alloc: min(evaluate(alloc, vt) for vt in vts.values())
    assert rob["worst_case_t"] == pytest.approx(worst(rob["allocation"]), rel=1e-9)
    for vt in vts.values():  # no single-scenario plan has a better worst case
        assert worst(optimize(vt, 230_000, max_per_year=3)["allocation"]) <= rob["worst_case_t"] + 1e-9


def test_tornado_grid_direction():
    from src.sensitivity import tornado
    path = pd.Series(400.0, index=range(2020, 2060))
    base, df = tornado(VehicleAssumptions(), 2026, path)
    g = df[df["parameter"].str.startswith("Grid")].iloc[0]
    assert g["high"] < base < g["low"]  # dirtier grid -> fewer savings
