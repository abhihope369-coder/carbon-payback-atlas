"""One-at-a-time sensitivity (tornado) for lifetime net savings of one EV."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from .lifecycle import VehicleAssumptions, payback_summary, payback_table

PARAMS = [
    ("annual_km", "Annual driving distance"),
    ("lifetime_years", "Vehicle lifetime"),
    ("petrol_l_per_100km", "Petrol fuel consumption"),
    ("petrol_kg_per_l", "Petrol emission factor"),
    ("ev_kwh_per_km", "EV electricity use"),
    ("battery_kwh", "Battery size"),
    ("battery_kg_per_kwh", "Battery manufacturing intensity"),
    ("grid", "Grid carbon intensity (whole path)"),
]


def _net(a: VehicleAssumptions, year: int, path: pd.Series) -> float:
    return payback_summary(payback_table(a, year, path), a)["lifetime_net_kg"] / 1000


def tornado(a: VehicleAssumptions, year: int, path: pd.Series, pct: float = 20.0) -> tuple[float, pd.DataFrame]:
    """Lifetime net t CO2e per EV when each input moves -pct% / +pct% on its own."""
    base = _net(a, year, path)
    rows = []
    for key, label in PARAMS:
        vals = {}
        for sign in (-1, 1):
            f = 1 + sign * pct / 100
            if key == "grid":
                vals[sign] = _net(a, year, path * f)
            else:
                v = getattr(a, key) * f
                if key == "lifetime_years":
                    v = max(1, int(round(v)))
                vals[sign] = _net(replace(a, **{key: v}), year, path)
        rows.append({"parameter": label, "low": vals[-1], "high": vals[1],
                     "swing": abs(vals[1] - vals[-1])})
    df = pd.DataFrame(rows).sort_values("swing")
    return base, df
