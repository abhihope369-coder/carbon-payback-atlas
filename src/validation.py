"""Temporal validation utilities for Grid Future AI.

All splits are by TARGET year (the year being predicted), so no information
from a test year can leak into training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SELECT_TRAIN_END = 2014   # selection phase: train on targets <= 2014
SELECT_VAL = (2015, 2019)  # validate on targets 2015-2019 to choose the model
TEST_TRAIN_END = 2019     # test phase: train on targets <= 2019
TEST_START = 2020         # report on targets >= 2020 (held out)


def level_metrics(i0: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Metrics in g CO2e/kWh, computed on intensity levels."""
    actual = i0 * np.exp(y_true)
    pred = i0 * np.exp(y_pred)
    err = pred - actual
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "n": int(len(actual)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2_levels": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "mape_pct": float(np.mean(np.abs(err) / actual) * 100),
    }


def split(sup: pd.DataFrame, phase: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if phase == "select":
        tr = sup[sup["target_year"] <= SELECT_TRAIN_END]
        te = sup[sup["target_year"].between(*SELECT_VAL)]
    elif phase == "test":
        tr = sup[sup["target_year"] <= TEST_TRAIN_END]
        te = sup[sup["target_year"] >= TEST_START]
    else:
        raise ValueError(phase)
    return tr, te


def per_horizon(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    out = []
    for h, g in df.groupby("h"):
        m = level_metrics(g["i0"].to_numpy(), g["y"].to_numpy(), g[pred_col].to_numpy())
        m["h"] = int(h)
        out.append(m)
    return pd.DataFrame(out)


def residual_quantiles(df: pd.DataFrame, pred_col: str, lo: float = 0.10, hi: float = 0.90) -> pd.DataFrame:
    """Empirical log-space residual quantiles per horizon (for 80% intervals)."""
    r = df.assign(res=df["y"] - df[pred_col])
    q = r.groupby("h")["res"].quantile([lo, hi]).unstack()
    q.columns = ["q_lo", "q_hi"]
    return q.reset_index()


def interval_coverage(df: pd.DataFrame, pred_col: str, q: pd.DataFrame) -> float:
    m = df.merge(q, on="h", how="left")
    inside = (m["y"] >= m[pred_col] + m["q_lo"]) & (m["y"] <= m[pred_col] + m["q_hi"])
    return float(inside.mean())
