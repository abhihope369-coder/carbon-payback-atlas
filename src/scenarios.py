"""Grid intensity paths used by the lifecycle engine and optimizer.

A path maps calendar year -> g CO2e/kWh for one country:
  observed years            -> Ember/OWID historical value
  origin+1 .. origin+H      -> Grid Future AI forecast (central / lo80 / hi80)
  beyond the forecast range -> held flat at the last forecast value
                               (explicit assumption, flagged in the UI)
"current" strategy: every future year uses the latest observed value.
"""
from __future__ import annotations

import pandas as pd

from .data_processing import TARGET

PATH_END_YEAR = 2060


def observed_series(panel: pd.DataFrame, country: str) -> pd.Series:
    g = panel[panel["country"] == country].set_index("year")[TARGET].dropna()
    return g.sort_index()


def build_path(panel: pd.DataFrame, forecasts: pd.DataFrame, country: str,
               kind: str = "forecast", end_year: int = PATH_END_YEAR) -> pd.Series:
    """kind: 'forecast' | 'lo80' | 'hi80' | 'current'."""
    obs = observed_series(panel, country)
    f = forecasts[forecasts["country"] == country].sort_values("year")
    if f.empty:
        raise KeyError(f"No forecast for {country}")
    origin = int(f["origin_year"].iloc[0])
    obs = obs[obs.index <= origin]
    base = float(obs.loc[origin])
    years = range(origin + 1, end_year + 1)
    if kind == "current":
        fut = pd.Series(base, index=list(years))
    else:
        col = {"forecast": "forecast", "lo80": "lo80", "hi80": "hi80"}[kind]
        fs = f.set_index("year")[col]
        last = float(fs.iloc[-1])
        fut = pd.Series([float(fs.get(y, last)) for y in years], index=list(years))
    return pd.concat([obs, fut]).astype(float)


def path_metadata(forecasts: pd.DataFrame, country: str) -> dict:
    f = forecasts[forecasts["country"] == country]
    origin = int(f["origin_year"].iloc[0])
    return {"origin_year": origin, "last_forecast_year": int(f["year"].max()),
            "latest_observed": float(f["i0"].iloc[0])}


def backtest_paths(panel: pd.DataFrame, bt_forecasts: pd.DataFrame, country: str,
                   origin: int) -> dict[str, pd.Series]:
    """Paths for the historical decision backtest.

    'current'  : intensity frozen at the origin year (what a planner knew)
    'forecast' : forecast from a model trained only on targets <= origin
    'actual'   : what really happened afterwards (used only for scoring)
    """
    obs = observed_series(panel, country)
    f = bt_forecasts[bt_forecasts["country"] == country].set_index("year")["forecast"]
    if f.empty or origin not in obs.index:
        raise KeyError(country)
    base = float(obs.loc[origin])
    years = list(f.index)
    return {
        "current": pd.Series(base, index=years),
        "forecast": f.astype(float),
        "actual": obs[obs.index > origin],
    }
