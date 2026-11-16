"""Step 3: does forecasting actually improve fleet decisions? A historical study.

For each origin year O in 2010-2015:
  1. Train the selected model type ONLY on rows whose target year <= O.
  2. Forecast every grid from O for 1-10 years ahead (saved for the app).
  3. Draw random candidate sets of 6 regions; plan with (a) intensity frozen at O,
     (b) the O-era forecast, (c) perfect foresight; score all three on the ACTUAL
     grids for O+1 .. min(O+10, 2024).

Fixed default scenario inputs (budget, fleet, costs) so only the grid assumption varies.
Usage: python scripts/backtest_study.py [n_sets_per_origin]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import forecasting as fc  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src.data_processing import PROCESSED_DIR, load_panel  # noqa: E402
from src.lifecycle import VehicleAssumptions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SET_SIZE = 6
BUDGET, FLEET, COST, KM, MPY = 10_000_000, 50, 50_000, 15_000, 80


def main(n_sets: int = 150):
    panel = load_panel()
    uni = panel[panel["in_model_universe"]]
    chosen = json.loads((ROOT / "outputs" / "validation_metrics.json").read_text())["chosen_model"]
    sup = fc.complete(fc.supervised_rows(uni))
    a = VehicleAssumptions()
    rng = np.random.default_rng(2026)
    rows = []
    for o in pl.BACKTEST_ORIGINS:
        tr = sup[sup["target_year"] <= o]
        model = fc.make_model(chosen).fit(tr[fc.FEATURES], tr["y"])
        bt = fc.forecast_from_latest(model, uni, origin_year=o)
        bt.to_csv(PROCESSED_DIR / f"backtest_forecasts_{o}.csv", index=False)
        d = {"panel": panel, "backtests": {o: bt}}
        deploy, target = pl.backtest_window(o)
        eligible = []
        for c in sorted(bt["country"].unique()):
            yrs = set(panel.loc[panel["country"] == c, "year"])
            if set(range(o, target + 1)) <= yrs:
                eligible.append(c)
        print(f"origin {o}: trained on {len(tr):,} rows, {len(eligible)} eligible grids, "
              f"deploy {deploy}, scored to {target}")
        for i in range(n_sets):
            regions = list(rng.choice(eligible, SET_SIZE, replace=False))
            cands = pd.DataFrame({"country": regions, "fleet_available": FLEET, "min_vehicles": 0,
                                  "cost_per_ev": COST, "annual_km": KM})
            r = pl.run_backtest(d, cands, a, BUDGET, MPY, 0.0, origin=o)
            if not r["usable"]:
                continue
            z = r["realized_t"]
            same = bool((r["results"]["current"]["allocation"]["vehicles"].to_numpy()
                         == r["results"]["forecast"]["allocation"]["vehicles"].to_numpy()).all())
            rows.append({"origin": o, "set_id": i, "regions": "|".join(sorted(regions)),
                         "current_t": z["current"], "forecast_t": z["forecast"], "perfect_t": z["actual"],
                         "current_planned_t": r["results"]["current"]["planned_net_t"],
                         "forecast_planned_t": r["results"]["forecast"]["planned_net_t"],
                         "identical_plans": same})
    df = pd.DataFrame(rows)
    df["diff_t"] = df["forecast_t"] - df["current_t"]
    df["diff_pct"] = df["diff_t"] / df["current_t"] * 100
    gap = df["perfect_t"] - df["current_t"]
    df["gap_closed_pct"] = np.where(gap > 1e-6, df["diff_t"] / gap.where(gap > 1e-6) * 100, np.nan)
    # Estimate accuracy: how far each plan's promised savings were from what it actually delivered
    df["current_est_err_pct"] = (df["current_planned_t"] - df["current_t"]) / df["current_t"] * 100
    df["forecast_est_err_pct"] = (df["forecast_planned_t"] - df["forecast_t"]) / df["forecast_t"] * 100
    df["perfect_gain_pct"] = (df["perfect_t"] - df["current_t"]) / df["current_t"] * 100
    df.to_csv(ROOT / "outputs" / "backtest_study.csv", index=False)

    def summarize(g: pd.DataFrame) -> dict:
        diff = g[~g["identical_plans"]]
        return {
            "n_scenarios": int(len(g)),
            "identical_plans": int(g["identical_plans"].sum()),
            "forecast_wins": int((g["diff_t"] > 1e-6).sum()),
            "current_wins": int((g["diff_t"] < -1e-6).sum()),
            "mean_diff_pct_all": float(g["diff_pct"].mean()),
            "mean_diff_pct_when_plans_differ": float(diff["diff_pct"].mean()) if len(diff) else None,
            "median_diff_pct_when_plans_differ": float(diff["diff_pct"].median()) if len(diff) else None,
            "total_current_t": float(g["current_t"].sum()), "total_forecast_t": float(g["forecast_t"].sum()),
            "total_perfect_t": float(g["perfect_t"].sum()),
            "mean_abs_est_err_current_pct": float(g["current_est_err_pct"].abs().mean()),
            "mean_abs_est_err_forecast_pct": float(g["forecast_est_err_pct"].abs().mean()),
            "mean_est_err_current_pct": float(g["current_est_err_pct"].mean()),
            "mean_est_err_forecast_pct": float(g["forecast_est_err_pct"].mean()),
            "forecast_estimate_closer_share": float((g["forecast_est_err_pct"].abs()
                                                     < g["current_est_err_pct"].abs()).mean()),
            "mean_perfect_gain_pct": float(g["perfect_gain_pct"].mean()),
            "share_of_gap_closed_pct": float((g["forecast_t"].sum() - g["current_t"].sum())
                                             / max(g["perfect_t"].sum() - g["current_t"].sum(), 1e-9) * 100),
        }

    summary = {"overall": summarize(df), "by_origin": {str(o): summarize(g) for o, g in df.groupby("origin")},
               "design": {"origins": pl.BACKTEST_ORIGINS, "sets_per_origin": n_sets, "regions_per_set": SET_SIZE,
                          "budget": BUDGET, "fleet_per_region": FLEET, "cost_per_ev": COST,
                          "annual_km": KM, "max_per_year": MPY, "model": chosen}}
    (ROOT / "outputs" / "backtest_study_summary.json").write_text(json.dumps(summary, indent=2))
    s = summary["overall"]
    print(f"\n{s['n_scenarios']} scenarios: forecast better {s['forecast_wins']}, current better "
          f"{s['current_wins']}, identical plans {s['identical_plans']}")
    print(f"Mean diff (all) {s['mean_diff_pct_all']:+.2f}%  | when plans differ: mean "
          f"{s['mean_diff_pct_when_plans_differ']:+.2f}%, median {s['median_diff_pct_when_plans_differ']:+.2f}%")
    print(f"Share of gap to perfect foresight closed: {s['share_of_gap_closed_pct']:.1f}%")
    print(f"Perfect foresight beats current-data by {s['mean_perfect_gain_pct']:.2f}% on average")
    print(f"Estimate error |mean|: current {s['mean_abs_est_err_current_pct']:.1f}% (bias "
          f"{s['mean_est_err_current_pct']:+.1f}%), forecast {s['mean_abs_est_err_forecast_pct']:.1f}% (bias "
          f"{s['mean_est_err_forecast_pct']:+.1f}%), forecast closer in {s['forecast_estimate_closer_share']:.0%}")
    for o, g in summary["by_origin"].items():
        print(f"  {o}: wins {g['forecast_wins']}/{g['current_wins']} (F/C), mean {g['mean_diff_pct_all']:+.2f}%")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
