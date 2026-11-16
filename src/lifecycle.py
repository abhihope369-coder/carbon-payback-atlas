"""Carbon Payback Engine: lifecycle emissions of a petrol vs electric vehicle.

System boundary (first prototype)
  Included: vehicle manufacturing, battery manufacturing, petrol well-to-wheel
            (combustion + upstream fuel production), EV charging electricity
            using Ember LIFECYCLE grid intensity (so both sides include upstream).
  Excluded: maintenance, end-of-life / recycling credits, road infrastructure,
            imports/exports of electricity (intensity is generation-based).

Units: distances km, energy kWh, emissions kg CO2e unless stated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class VehicleAssumptions:
    annual_km: float = 15_000          # hypothetical fleet usage (spec example)
    lifetime_years: int = 12           # hypothetical assumption
    petrol_l_per_100km: float = 8.0    # hypothetical fleet sedan
    petrol_kg_per_l: float = 2.65      # DESNZ 2025: 2.069 TTW + 0.581 WTT (well-to-wheel)
    ev_kwh_per_km: float = 0.18        # at the wall, incl. charging losses (spec example)
    grid_loss_pct: float = 0.0         # optional T&D losses on top of generation intensity
    petrol_vehicle_mfg_t: float = 8.0  # Argonne GREET: 8-12 t for current powertrains
    ev_vehicle_mfg_excl_battery_t: float = 8.0  # ICCT 2021: vehicle mfg ~powertrain-agnostic
    battery_kwh: float = 60.0          # hypothetical pack size
    battery_kg_per_kwh: float = 60.0   # ICCT 2021 (Europe/US): 60 kg CO2e/kWh
    replacement_mode: str = "new_vs_new"  # or "early_retirement"

    def to_dict(self) -> dict:
        return asdict(self)


def petrol_annual_kg(a: VehicleAssumptions) -> float:
    """E_petrol = D x F, with F = L/km x kg/L (well-to-wheel)."""
    f_kg_per_km = a.petrol_l_per_100km / 100.0 * a.petrol_kg_per_l
    return a.annual_km * f_kg_per_km


def ev_annual_kg(a: VehicleAssumptions, grid_g_per_kwh: float | np.ndarray) -> float | np.ndarray:
    """E_EV = D x C x G   (G converted from g/kWh to kg/kWh)."""
    kwh = a.annual_km * a.ev_kwh_per_km * (1 + a.grid_loss_pct / 100.0)
    return kwh * np.asarray(grid_g_per_kwh) / 1000.0


def battery_mfg_kg(a: VehicleAssumptions) -> float:
    return a.battery_kwh * a.battery_kg_per_kwh


def ev_mfg_kg(a: VehicleAssumptions) -> float:
    """Total EV manufacturing = vehicle excl. battery + battery (counted once)."""
    return a.ev_vehicle_mfg_excl_battery_t * 1000 + battery_mfg_kg(a)


def additional_mfg_kg(a: VehicleAssumptions) -> float:
    """M, the upfront emissions the EV must pay back.

    new_vs_new: the fleet buys a new vehicle either way -> M = M_EV - M_petrol.
    early_retirement: a working petrol car is retired early -> M = M_EV
                      (no petrol car is built in the counterfactual).
    """
    if a.replacement_mode == "early_retirement":
        return ev_mfg_kg(a)
    return ev_mfg_kg(a) - a.petrol_vehicle_mfg_t * 1000


def payback_table(a: VehicleAssumptions, deploy_year: int, intensity: pd.Series) -> pd.DataFrame:
    """Year-by-year savings for one vehicle deployed in `deploy_year`.

    `intensity` maps calendar year -> grid g CO2e/kWh and must cover every
    operating year. N_t = sum_{y<=t} S_y - M.
    """
    years = np.arange(deploy_year, deploy_year + a.lifetime_years)
    missing = [y for y in years if y not in intensity.index]
    if missing:
        raise ValueError(f"No grid intensity for years {missing}")
    g = intensity.loc[years].to_numpy(dtype=float)
    e_petrol = np.full(len(years), petrol_annual_kg(a))
    e_ev = ev_annual_kg(a, g)
    s = e_petrol - e_ev
    m = additional_mfg_kg(a)
    return pd.DataFrame({
        "year": years,
        "operating_year": np.arange(1, len(years) + 1),
        "grid_g_per_kwh": g,
        "petrol_kg": e_petrol,
        "ev_kg": e_ev,
        "saving_kg": s,
        "cumulative_net_kg": np.cumsum(s) - m,
    })


def payback_summary(tbl: pd.DataFrame, a: VehicleAssumptions) -> dict:
    """Payback year and interpolated payback time; None if never within lifetime."""
    m = additional_mfg_kg(a)
    cum = tbl["cumulative_net_kg"].to_numpy()
    s = tbl["saving_kg"].to_numpy()
    out = {
        "additional_mfg_kg": m,
        "lifetime_net_kg": float(cum[-1]),
        "payback_year": None,
        "payback_years_fractional": None,
    }
    if m <= 0:
        out["payback_year"] = int(tbl["year"].iloc[0])
        out["payback_years_fractional"] = 0.0
        return out
    hit = np.where(cum >= 0)[0]
    if len(hit):
        k = int(hit[0])
        prev = -m if k == 0 else cum[k - 1]
        frac = k + (-prev / s[k]) if s[k] > 0 else k + 1
        out["payback_year"] = int(tbl["year"].iloc[k])
        out["payback_years_fractional"] = float(frac)
    return out


def lifecycle_totals(tbl: pd.DataFrame, a: VehicleAssumptions) -> dict:
    """Whole-life totals for the stacked comparison chart (kg CO2e)."""
    petrol_mfg = a.petrol_vehicle_mfg_t * 1000 if a.replacement_mode == "new_vs_new" else 0.0
    return {
        "Petrol": {"Vehicle manufacturing": petrol_mfg, "Battery manufacturing": 0.0,
                   "Fuel / electricity (well-to-wheel)": float(tbl["petrol_kg"].sum())},
        "Electric": {"Vehicle manufacturing": a.ev_vehicle_mfg_excl_battery_t * 1000,
                     "Battery manufacturing": battery_mfg_kg(a),
                     "Fuel / electricity (well-to-wheel)": float(tbl["ev_kg"].sum())},
    }
