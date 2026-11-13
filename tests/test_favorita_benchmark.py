"""Focused correctness tests for the Favorita forecasting benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from favorita_benchmark import add_features, baseline_prediction, bias, wape


def synthetic_panel() -> pd.DataFrame:
    weeks = pd.date_range("2020-01-05", periods=60, freq="W-SUN")
    rows = []
    for store in (1, 2):
        for family in ("A", "B"):
            for index, week in enumerate(weeks):
                rows.append({
                    "week": week,
                    "store_nbr": store,
                    "family": family,
                    "sales": float(index + store),
                    "promo_items": index % 3,
                    "transactions": float(100 + index),
                    "holiday_days": 0,
                    "oil_price": 50.0,
                    "city": "City",
                    "state": "State",
                    "type": "A",
                    "cluster": 1,
                })
    return pd.DataFrame(rows).sort_values(["store_nbr", "family", "week"]).reset_index(drop=True)


def test_target_features_are_shifted() -> None:
    panel = synthetic_panel()
    original = add_features(panel)
    changed = panel.copy()
    target_row = changed.index[
        (changed["store_nbr"] == 1) & (changed["family"] == "A") & (changed["week"] == pd.Timestamp("2020-07-26"))
    ][0]
    changed.loc[target_row, "sales"] = 999999
    rebuilt = add_features(changed)

    feature_columns = [column for column in original if column.startswith("sales_")] + ["expanding_mean"]
    assert np.allclose(
        original.loc[target_row, feature_columns].to_numpy(dtype=float),
        rebuilt.loc[target_row, feature_columns].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_seasonal_baseline_uses_documented_fallback() -> None:
    frame = pd.DataFrame({
        "sales_lag_52": [np.nan, 12.0],
        "sales_mean_13": [8.0, 10.0],
        "expanding_mean": [6.0, 7.0],
    })
    assert baseline_prediction(frame, "sales_lag_52").tolist() == [8.0, 12.0]


def test_wape_and_bias() -> None:
    actual = np.array([10.0, 20.0])
    predicted = np.array([12.0, 18.0])
    assert np.isclose(wape(actual, predicted), 4 / 30 * 100)
    assert np.isclose(bias(actual, predicted), 0.0)
