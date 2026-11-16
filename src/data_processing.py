"""Data loading and cleaning for Carbon Payback Atlas.

Source: Our World in Data energy dataset (https://github.com/owid/energy-data),
licence CC BY 4.0. The electricity carbon-intensity column is Ember's lifecycle
carbon intensity of electricity generation (g CO2e/kWh).
"""
from __future__ import annotations

import json
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

OWID_URL = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
OWID_CODEBOOK_URL = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-codebook.csv"
RAW_FILE = RAW_DIR / "owid-energy-data.csv"
PANEL_FILE = PROCESSED_DIR / "panel.csv"
META_FILE = PROCESSED_DIR / "data_manifest.json"

TARGET = "carbon_intensity_elec"  # g CO2e/kWh, lifecycle, Ember via OWID

# Generation-mix columns kept for the Atlas and as model features (% of generation).
MIX_COLS = [
    "coal_share_elec", "gas_share_elec", "oil_share_elec", "nuclear_share_elec",
    "hydro_share_elec", "wind_share_elec", "solar_share_elec", "biofuel_share_elec",
    "other_renewables_share_elec", "low_carbon_share_elec", "fossil_share_elec",
    "renewables_share_elec",
]
KEEP_COLS = ["country", "iso_code", "year", TARGET, "electricity_generation"] + MIX_COLS

# Modelling universe rules (documented in README / app methodology).
FIRST_YEAR = 2000          # Ember global coverage starts in 2000
MIN_GENERATION_TWH = 10.0  # exclude very small systems whose series are noisy
MIN_YEARS = 15             # need enough history for lags + validation


def download_raw(force: bool = False) -> Path:
    """Download the OWID energy CSV (and codebook) into data/raw."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if force or not RAW_FILE.exists():
        urllib.request.urlretrieve(OWID_URL, RAW_FILE)
        urllib.request.urlretrieve(OWID_CODEBOOK_URL, RAW_DIR / "owid-energy-codebook.csv")
    return RAW_FILE


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df[KEEP_COLS].copy()


def clean_panel(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep real countries (ISO-3 codes, no OWID aggregates) with usable history.

    Returns the panel and a dict describing every filtering step, so nothing is
    dropped silently.
    """
    log: dict = {}
    is_country = df["iso_code"].notna() & ~df["iso_code"].astype(str).str.startswith("OWID")
    df = df[is_country & (df["year"] >= FIRST_YEAR)].copy()
    log["countries_with_iso_code"] = int(df["country"].nunique())

    df = df.dropna(subset=[TARGET])
    log["countries_with_any_intensity"] = int(df["country"].nunique())

    latest = df.sort_values("year").groupby("country").tail(1).set_index("country")
    big = latest.index[latest["electricity_generation"] >= MIN_GENERATION_TWH]
    counts = df.groupby("country")["year"].count()
    long_enough = counts.index[counts >= MIN_YEARS]
    keep = sorted(set(big) & set(long_enough))
    log["min_generation_twh_latest_year"] = MIN_GENERATION_TWH
    log["min_years_of_history"] = MIN_YEARS
    log["countries_in_model_universe"] = len(keep)

    # Flag gaps (missing years inside a country's series). We do not interpolate.
    gaps = {}
    for c, g in df[df["country"].isin(keep)].groupby("country"):
        yrs = g["year"].to_numpy()
        expected = np.arange(yrs.min(), yrs.max() + 1)
        missing = sorted(set(expected) - set(yrs))
        if missing:
            gaps[c] = [int(y) for y in missing]
    log["countries_with_internal_gaps"] = gaps

    df["in_model_universe"] = df["country"].isin(keep)
    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    return df, log


def build(force_download: bool = False) -> pd.DataFrame:
    path = download_raw(force=force_download)
    raw = load_raw(path)
    panel, log = clean_panel(raw)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_FILE, index=False)
    manifest = {
        "source_name": "Our World in Data - Energy dataset (electricity data from Ember)",
        "download_url": OWID_URL,
        "repository": "https://github.com/owid/energy-data",
        "licence": "CC BY 4.0 (Our World in Data); underlying Ember data CC BY 4.0",
        "date_accessed": date.today().isoformat(),
        "target_variable": TARGET,
        "target_units": "g CO2e per kWh (lifecycle emissions of electricity generation)",
        "year_range": [int(panel["year"].min()), int(panel["year"].max())],
        "filtering": log,
    }
    META_FILE.write_text(json.dumps(manifest, indent=2))
    return panel


def load_panel() -> pd.DataFrame:
    return pd.read_csv(PANEL_FILE)


def load_manifest() -> dict:
    return json.loads(META_FILE.read_text())
