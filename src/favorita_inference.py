"""One-week-ahead inference from an uploaded Favorita demand plan."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from favorita_benchmark import GROUPS, MODEL_OUT, ROOT, add_features


WEEKLY_PATH = ROOT / "data" / "processed" / "favorita_weekly.parquet"
REQUIRED = ["week", "store_nbr", "family", "promo_items", "holiday_days", "oil_price"]


def sample_plan() -> pd.DataFrame:
    history = pd.read_parquet(WEEKLY_PATH)
    next_week = pd.Timestamp(history["week"].max()) + pd.Timedelta(days=7)
    latest = history.sort_values("week").groupby(GROUPS, as_index=False).tail(1)
    sample = latest.head(20)[["store_nbr", "family", "oil_price"]].copy()
    sample.insert(0, "week", next_week.date().isoformat())
    sample["promo_items"] = 0
    sample["holiday_days"] = 0
    return sample[REQUIRED]


def forecast_plan(upload) -> pd.DataFrame:
    plan = pd.read_csv(upload)
    missing = [column for column in REQUIRED if column not in plan]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    plan = plan[REQUIRED].copy()
    plan["week"] = pd.to_datetime(plan["week"], errors="raise")
    if plan.duplicated(["week", *GROUPS]).any():
        raise ValueError("Each week/store/family combination must be unique.")
    if plan["week"].nunique() != 1:
        raise ValueError("This production artifact supports one forecast week per upload.")

    history = pd.read_parquet(WEEKLY_PATH)
    history["week"] = pd.to_datetime(history["week"])
    expected_week = history["week"].max() + pd.Timedelta(days=7)
    if plan["week"].iloc[0] != expected_week:
        raise ValueError(f"Forecast week must be {expected_week.date().isoformat()}.")

    valid_stores = set(history["store_nbr"].unique())
    valid_families = set(history["family"].unique())
    invalid_stores = sorted(set(plan["store_nbr"]) - valid_stores)
    invalid_families = sorted(set(plan["family"]) - valid_families)
    if invalid_stores or invalid_families:
        raise ValueError(f"Unknown stores={invalid_stores}; unknown families={invalid_families}")

    metadata = history.sort_values("week").groupby(GROUPS, as_index=False).tail(1)[
        [*GROUPS, "city", "state", "type", "cluster", "transactions"]
    ]
    future = plan.merge(metadata, on=GROUPS, how="left")
    future["sales"] = np.nan
    combined = pd.concat([history, future[history.columns]], ignore_index=True)
    featured = add_features(combined)
    future_features = featured[
        (featured["week"] == expected_week)
        & featured.set_index(GROUPS).index.isin(plan.set_index(GROUPS).index)
    ].copy()

    artifact = joblib.load(MODEL_OUT)
    prediction = np.maximum(0, artifact["model"].predict(future_features[artifact["features"]]))
    result = future_features[["week", "store_nbr", "family", "promo_items", "holiday_days"]].copy()
    result["forecast_units"] = prediction
    return result.sort_values(["store_nbr", "family"]).reset_index(drop=True)
