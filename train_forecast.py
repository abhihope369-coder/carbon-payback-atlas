"""Step 2: train, validate and save Grid Future AI.

Phases (all split by target year, see src/validation.py):
  1. SELECT  train <= 2014, validate 2015-2019 -> choose model by MAE
  2. TEST    train <= 2019, test >= 2020        -> honest held-out report
  3. BACKTEST model = selection-phase model (trained <= 2014), used by the
     optimizer's historical decision backtest (forecasts from origin 2014)
  4. PRODUCTION  train on all rows               -> forecasts from latest year

Usage: python scripts/train_forecast.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import forecasting as fc  # noqa: E402
from src import validation as va  # noqa: E402
from src.data_processing import PROCESSED_DIR, load_panel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["ridge", "random_forest", "gradient_boosting"]
BASELINES = ["persistence", "linear_trend"]
BACKTEST_ORIGIN = va.SELECT_TRAIN_END  # 2014


def fit_predict(name, tr, te):
    if name in BASELINES:
        return None, fc.baseline_predict(name, te)
    m = fc.make_model(name)
    m.fit(tr[fc.FEATURES], tr["y"])
    return m, m.predict(te[fc.FEATURES])


def main():
    panel = load_panel()
    uni = panel[panel["in_model_universe"]]
    sup_all = fc.supervised_rows(uni)
    sup = fc.complete(sup_all)
    dropped = len(sup_all) - len(sup)
    print(f"Supervised rows: {len(sup):,} (dropped {dropped:,} with missing features)")

    report = {"n_rows_total": int(len(sup)), "n_rows_dropped_missing_features": int(dropped),
              "horizon_years": fc.HORIZON, "features": fc.FEATURES, "phases": {}}

    # ---------- Phase 1: model selection ----------
    tr, va_df = va.split(sup, "select")
    sel = {}
    sel_models = {}
    va_df = va_df.copy()
    for name in MODELS + BASELINES:
        m, p = fit_predict(name, tr, va_df)
        va_df[f"p_{name}"] = p
        sel_models[name] = m
        sel[name] = va.level_metrics(va_df["i0"].to_numpy(), va_df["y"].to_numpy(), p)
    chosen = min(MODELS, key=lambda n: sel[n]["mae"])
    report["phases"]["select"] = {"train_targets": f"<= {va.SELECT_TRAIN_END}",
                                  "validation_targets": f"{va.SELECT_VAL[0]}-{va.SELECT_VAL[1]}",
                                  "n_train": int(len(tr)), "n_val": int(len(va_df)), "metrics": sel}
    report["chosen_model"] = chosen
    print("\nSELECTION (validation 2015-2019, MAE g/kWh):")
    for n, m in sel.items():
        print(f"  {n:18s} MAE={m['mae']:6.1f} RMSE={m['rmse']:6.1f}")
    print(f"Chosen: {chosen}")

    # Interval calibration from validation residuals of the chosen model
    q_val = va.residual_quantiles(va_df, f"p_{chosen}")

    # ---------- Phase 2: held-out test ----------
    tr2, te = va.split(sup, "test")
    te = te.copy()
    test = {}
    for name in MODELS + BASELINES:
        _, p = fit_predict(name, tr2, te)
        te[f"p_{name}"] = p
        test[name] = va.level_metrics(te["i0"].to_numpy(), te["y"].to_numpy(), p)
    cov = va.interval_coverage(te, f"p_{chosen}", q_val)
    ph = {name: va.per_horizon(te, f"p_{name}").to_dict(orient="records")
          for name in [chosen] + BASELINES}
    report["phases"]["test"] = {"train_targets": f"<= {va.TEST_TRAIN_END}",
                                "test_targets": f">= {va.TEST_START}",
                                "n_train": int(len(tr2)), "n_test": int(len(te)),
                                "metrics": test, "per_horizon": ph,
                                "interval_80_coverage_on_test": cov}
    print("\nHELD-OUT TEST (targets >= 2020, MAE g/kWh):")
    for n, m in test.items():
        print(f"  {n:18s} MAE={m['mae']:6.1f} RMSE={m['rmse']:6.1f} R2={m['r2_levels']:.3f}")
    print(f"80% interval (calibrated on 2015-2019) coverage on test: {cov:.1%}")

    te_out = te[["country", "iso_code", "origin_year", "h", "target_year", "i0", "i_target",
                 f"p_{chosen}", "p_persistence", "p_linear_trend"]].copy()
    te_out["pred_model"] = te_out["i0"] * np.exp(te_out[f"p_{chosen}"])
    te_out.to_csv(ROOT / "outputs" / "test_predictions.csv", index=False)

    # ---------- Phase 3: backtest forecasts (model trained <= 2014) ----------
    bt_model = sel_models[chosen]
    bt = fc.forecast_from_latest(bt_model, uni, origin_year=BACKTEST_ORIGIN)
    bt.to_csv(PROCESSED_DIR / f"backtest_forecasts_{BACKTEST_ORIGIN}.csv", index=False)
    report["backtest_origin_year"] = BACKTEST_ORIGIN

    # ---------- Phase 4: production ----------
    prod = fc.make_model(chosen)
    prod.fit(sup[fc.FEATURES], sup["y"])
    fcst = fc.forecast_from_latest(prod, uni)
    # Production intervals: residual quantiles pooled over validation + test
    pooled = pd.concat([
        va_df[["h", "y", f"p_{chosen}"]].rename(columns={f"p_{chosen}": "p"}),
        te[["h", "y", f"p_{chosen}"]].rename(columns={f"p_{chosen}": "p"}),
    ])
    q_prod = va.residual_quantiles(pooled, "p")
    fcst = fcst.merge(q_prod, on="h", how="left")
    fcst["lo80"] = fcst["i0"] * np.exp(fcst["log_ratio"] + fcst["q_lo"])
    fcst["hi80"] = fcst["i0"] * np.exp(fcst["log_ratio"] + fcst["q_hi"])
    fcst.to_csv(PROCESSED_DIR / "forecasts.csv", index=False)
    report["interval_quantiles_production"] = q_prod.to_dict(orient="records")
    report["n_countries_forecast"] = int(fcst["country"].nunique())
    report["forecast_origin_years"] = {str(k): int(v) for k, v in
                                       fcst.groupby("origin_year")["country"].nunique().items()}

    imp = fc.feature_importance(prod)
    if imp is not None:
        report["feature_importance"] = {k: float(v) for k, v in imp.items()}

    joblib.dump({"model": prod, "name": chosen, "features": fc.FEATURES, "horizon": fc.HORIZON},
                ROOT / "models" / "grid_forecast_model.joblib", compress=3)
    (ROOT / "outputs" / "validation_metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved model ({chosen}), forecasts for {report['n_countries_forecast']} countries.")
    print("Origins:", report["forecast_origin_years"])


if __name__ == "__main__":
    main()
