"""Grid Future AI: direct multi-horizon forecasting of electricity carbon intensity.

Design
------
For each country and origin year t we build features from information known at
t (intensity lags, recent trend, generation mix). The model predicts the log
ratio  y = ln(I[t+h] / I[t])  for horizons h = 1..H, with h as an input feature.

Why a log ratio: tree models cannot extrapolate beyond the levels they were
trained on, and the relative change is comparable across clean and dirty grids.
Why direct (not recursive): a recursive model would need future generation-mix
values, which we do not have. Each horizon is predicted from data known at t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_processing import TARGET

HORIZON = 10  # years ahead the model forecasts directly

FEATURES = [
    "h",                 # forecast horizon in years
    "log_i0",            # ln intensity at origin year t
    "d1",                # ln I[t] - ln I[t-1]
    "d2",                # ln I[t-1] - ln I[t-2]
    "d3",                # ln I[t] - ln I[t-3]  (3-year change)
    "coal", "gas", "oil", "nuclear", "hydro", "wind_solar",
    "low_carbon_chg3",   # change in low-carbon share over 3 years (pct points)
]

MODEL_LABELS = {
    "ridge": "Ridge regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


def origin_features(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (country, origin year) with features known at that year."""
    rows = []
    for country, g in panel.groupby("country"):
        g = g.sort_values("year").set_index("year")
        li = np.log(g[TARGET])
        lc = g["low_carbon_share_elec"]
        f = pd.DataFrame({
            "country": country,
            "iso_code": g["iso_code"],
            "origin_year": g.index,
            "i0": g[TARGET],
            "log_i0": li,
            "d1": li - li.shift(1),
            "d2": li.shift(1) - li.shift(2),
            "d3": li - li.shift(3),
            "coal": g["coal_share_elec"],
            "gas": g["gas_share_elec"],
            "oil": g["oil_share_elec"],
            "nuclear": g["nuclear_share_elec"],
            "hydro": g["hydro_share_elec"],
            "wind_solar": g["wind_share_elec"] + g["solar_share_elec"],
            "low_carbon_chg3": lc - lc.shift(3),
        })
        # Only keep lags that come from consecutive years (no gap jumping).
        yrs = pd.Series(g.index, index=g.index)
        consecutive = (yrs - yrs.shift(3)) == 3
        f = f[consecutive.values]
        rows.append(f.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def supervised_rows(panel: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Pair each origin row with its observed future value for h = 1..horizon."""
    feats = origin_features(panel)
    actual = panel[["country", "year", TARGET]].rename(columns={"year": "target_year", TARGET: "i_target"})
    out = []
    for h in range(1, horizon + 1):
        f = feats.copy()
        f["h"] = h
        f["target_year"] = f["origin_year"] + h
        out.append(f)
    sup = pd.concat(out, ignore_index=True).merge(actual, on=["country", "target_year"], how="inner")
    sup["y"] = np.log(sup["i_target"]) - sup["log_i0"]
    return sup


def make_model(name: str):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=400, min_samples_leaf=5, max_features=0.6,
                                     random_state=42, n_jobs=-1)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(n_estimators=400, learning_rate=0.03, max_depth=3,
                                         subsample=0.8, random_state=42)
    raise ValueError(name)


def baseline_predict(name: str, X: pd.DataFrame) -> np.ndarray:
    """Baselines in log-ratio space.

    persistence: future intensity equals the latest observed value (ratio = 1).
    linear_trend: continue the country's average log change of the last 3 years.
    """
    if name == "persistence":
        return np.zeros(len(X))
    if name == "linear_trend":
        return (X["d3"].to_numpy() / 3.0) * X["h"].to_numpy()
    raise ValueError(name)


def complete(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any missing feature (reported by the training script)."""
    return df.dropna(subset=[f for f in FEATURES if f in df.columns])


def forecast_from_latest(model, panel: pd.DataFrame, horizon: int = HORIZON,
                         origin_year: int | None = None) -> pd.DataFrame:
    """Forecast h = 1..horizon from each country's latest year (or a fixed origin)."""
    feats = complete(origin_features(panel))
    if origin_year is None:
        feats = feats.sort_values("origin_year").groupby("country").tail(1)
    else:
        feats = feats[feats["origin_year"] == origin_year]
    rows = []
    for h in range(1, horizon + 1):
        f = feats.copy()
        f["h"] = h
        f["year"] = f["origin_year"] + h
        f["log_ratio"] = model.predict(f[FEATURES])
        f["forecast"] = f["i0"] * np.exp(f["log_ratio"])
        rows.append(f[["country", "iso_code", "origin_year", "h", "year", "i0", "log_ratio", "forecast"]])
    return pd.concat(rows, ignore_index=True)


def feature_importance(model) -> pd.Series | None:
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return None
