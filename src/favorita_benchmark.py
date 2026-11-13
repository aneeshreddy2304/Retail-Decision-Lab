"""Leakage-safe feasibility benchmark for the Favorita retail dataset.

The benchmark trains one global weekly store-family model and evaluates the
same forecasts at four coherent aggregation levels. Model and baseline
selection use expanding-window cross-validation; the final eight weeks are
opened only after the selection rule has chosen a winner.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "favorita"
OUT = ROOT / "reports" / "favorita_benchmark"
MODEL_OUT = ROOT / "models" / "favorita_benchmark_model.joblib"

SEED = 42
HOLDOUT_WEEKS = 8
CV_FOLDS = 4
CV_WEEKS = 8
GROUPS = ["store_nbr", "family"]

LAG_WEEKS = [1, 2, 4, 8, 13, 26, 52]
ROLLING_WEEKS = [4, 8, 13, 26]

FEATURES = [
    "store_code", "family_code", "store_type_code", "cluster",
    "week_of_year", "month", "trend", "week_sin_1", "week_cos_1",
    "week_sin_2", "week_cos_2", "promo_items", "holiday_days", "oil_price",
    *[f"sales_lag_{lag}" for lag in LAG_WEEKS],
    *[f"sales_mean_{window}" for window in ROLLING_WEEKS],
    *[f"sales_median_{window}" for window in (4, 13, 26)],
    "sales_std_13", "nonzero_rate_13", "transactions_lag_1",
    "transactions_mean_4", "expanding_mean",
]


def wape(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    denominator = np.abs(actual_array).sum()
    return float(np.abs(actual_array - predicted_array).sum() / denominator * 100) if denominator else 0.0


def bias(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    denominator = actual_array.sum()
    return float((predicted_array - actual_array).sum() / denominator * 100) if denominator else 0.0


def week_end(values: pd.Series) -> pd.Series:
    return values.dt.to_period("W-SUN").dt.end_time.dt.normalize()


def expanded_holidays(stores: pd.DataFrame) -> pd.DataFrame:
    holidays = pd.read_csv(RAW / "holidays_events.csv", parse_dates=["date"])
    holidays = holidays[(~holidays["transferred"]) & (holidays["type"] != "Work Day")].copy()
    rows: list[pd.DataFrame] = []
    for holiday in holidays.itertuples(index=False):
        if holiday.locale == "National":
            affected = stores
        elif holiday.locale == "Regional":
            affected = stores[stores["state"] == holiday.locale_name]
        elif holiday.locale == "Local":
            affected = stores[stores["city"] == holiday.locale_name]
        else:
            continue
        if affected.empty:
            continue
        frame = affected[["store_nbr"]].copy()
        frame["date"] = holiday.date
        frame["holiday"] = 1
        rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=["week", "store_nbr", "holiday_days"])
    daily = pd.concat(rows, ignore_index=True).drop_duplicates(["date", "store_nbr"])
    daily["week"] = week_end(daily["date"])
    return daily.groupby(["week", "store_nbr"], as_index=False)["holiday"].sum().rename(
        columns={"holiday": "holiday_days"}
    )


def build_weekly_panel() -> pd.DataFrame:
    required = ["train.csv", "stores.csv", "transactions.csv", "oil.csv", "holidays_events.csv"]
    missing = [name for name in required if not (RAW / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Favorita files: {missing}")

    stores = pd.read_csv(RAW / "stores.csv")
    sales = pd.read_csv(
        RAW / "train.csv",
        usecols=["date", "store_nbr", "family", "sales", "onpromotion"],
        parse_dates=["date"],
    )
    sales["week"] = week_end(sales["date"])
    weekly_sales = sales.groupby(["week", *GROUPS], as_index=False).agg(
        sales=("sales", "sum"), promo_items=("onpromotion", "sum")
    )

    max_date = sales["date"].max().normalize()
    # The competition training file ends on a Tuesday. Only include weeks whose
    # Sunday boundary is actually observed; otherwise the final week is a
    # two-day partial total and makes every forecast look artificially high.
    days_since_sunday = (max_date.weekday() - 6) % 7
    last_complete_week = max_date - pd.Timedelta(days=days_since_sunday)
    weekly_sales = weekly_sales[weekly_sales["week"] <= last_complete_week]
    all_weeks = pd.date_range(weekly_sales["week"].min(), last_complete_week, freq="W-SUN")
    families = np.sort(sales["family"].unique())
    index = pd.MultiIndex.from_product(
        [all_weeks, np.sort(stores["store_nbr"].unique()), families],
        names=["week", *GROUPS],
    )
    panel = weekly_sales.set_index(["week", *GROUPS]).reindex(index, fill_value=0).reset_index()

    # Do not reward models for predicting structural zeroes before a store opened.
    first_active = weekly_sales[weekly_sales["sales"] > 0].groupby("store_nbr")["week"].min()
    panel["first_active_week"] = panel["store_nbr"].map(first_active)
    panel = panel[panel["week"] >= panel["first_active_week"]].drop(columns="first_active_week")
    panel = panel.merge(stores, on="store_nbr", how="left")

    transactions = pd.read_csv(RAW / "transactions.csv", parse_dates=["date"])
    transactions["week"] = week_end(transactions["date"])
    weekly_transactions = transactions.groupby(["week", "store_nbr"], as_index=False)["transactions"].sum()
    panel = panel.merge(weekly_transactions, on=["week", "store_nbr"], how="left")
    panel["transactions"] = panel["transactions"].fillna(0)

    oil = pd.read_csv(RAW / "oil.csv", parse_dates=["date"])
    oil["dcoilwtico"] = oil["dcoilwtico"].interpolate(limit_direction="both")
    oil["week"] = week_end(oil["date"])
    weekly_oil = oil.groupby("week", as_index=False)["dcoilwtico"].mean().rename(
        columns={"dcoilwtico": "oil_price"}
    )
    panel = panel.merge(weekly_oil, on="week", how="left")
    panel["oil_price"] = panel["oil_price"].interpolate(limit_direction="both")

    panel = panel.merge(expanded_holidays(stores), on=["week", "store_nbr"], how="left")
    panel["holiday_days"] = panel["holiday_days"].fillna(0)
    panel = panel.sort_values([*GROUPS, "week"]).reset_index(drop=True)
    return panel


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    grouped_sales = frame.groupby(GROUPS, sort=False)["sales"]
    shifted_sales = grouped_sales.shift(1)
    grouped_transactions = frame.groupby(GROUPS, sort=False)["transactions"]

    for lag in LAG_WEEKS:
        frame[f"sales_lag_{lag}"] = grouped_sales.shift(lag)
    for window in ROLLING_WEEKS:
        frame[f"sales_mean_{window}"] = shifted_sales.groupby(
            [frame[column] for column in GROUPS], sort=False
        ).transform(lambda values: values.rolling(window, min_periods=max(2, window // 2)).mean())
    for window in (4, 13, 26):
        frame[f"sales_median_{window}"] = shifted_sales.groupby(
            [frame[column] for column in GROUPS], sort=False
        ).transform(lambda values: values.rolling(window, min_periods=max(2, window // 2)).median())
    frame["sales_std_13"] = shifted_sales.groupby(
        [frame[column] for column in GROUPS], sort=False
    ).transform(lambda values: values.rolling(13, min_periods=6).std())
    frame["nonzero_rate_13"] = shifted_sales.gt(0).groupby(
        [frame[column] for column in GROUPS], sort=False
    ).transform(lambda values: values.rolling(13, min_periods=6).mean())
    frame["expanding_mean"] = shifted_sales.groupby(
        [frame[column] for column in GROUPS], sort=False
    ).transform(lambda values: values.expanding(min_periods=4).mean())

    frame["transactions_lag_1"] = grouped_transactions.shift(1)
    shifted_transactions = grouped_transactions.shift(1)
    frame["transactions_mean_4"] = shifted_transactions.groupby(
        [frame[column] for column in GROUPS], sort=False
    ).transform(lambda values: values.rolling(4, min_periods=2).mean())

    frame["store_code"] = frame["store_nbr"].astype(int) - 1
    frame["family_code"] = pd.Categorical(frame["family"]).codes
    frame["store_type_code"] = pd.Categorical(frame["type"]).codes
    frame["week_of_year"] = frame["week"].dt.isocalendar().week.astype(int)
    frame["month"] = frame["week"].dt.month
    frame["trend"] = (frame["week"] - frame["week"].min()).dt.days / 7
    angle = 2 * np.pi * frame["week_of_year"] / 52.18
    frame["week_sin_1"], frame["week_cos_1"] = np.sin(angle), np.cos(angle)
    frame["week_sin_2"], frame["week_cos_2"] = np.sin(2 * angle), np.cos(2 * angle)
    return frame.sort_values("week").reset_index(drop=True)


def model_factories():
    common = dict(
        learning_rate=0.07, max_iter=120, max_leaf_nodes=31, max_depth=5,
        min_samples_leaf=40, l2_regularization=1.0, early_stopping=False,
        random_state=SEED,
    )
    return {
        "LightGBM L1": lambda: LGBMRegressor(
            objective="regression_l1", n_estimators=500, learning_rate=0.04,
            num_leaves=31, max_depth=-1, min_child_samples=80,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            random_state=SEED, n_jobs=-1, verbosity=-1,
        ),
        "LightGBM Tweedie": lambda: LGBMRegressor(
            objective="tweedie", tweedie_variance_power=1.2,
            n_estimators=500, learning_rate=0.04, num_leaves=31,
            max_depth=-1, min_child_samples=80, subsample=0.85,
            colsample_bytree=0.85, reg_lambda=1.0,
            random_state=SEED, n_jobs=-1, verbosity=-1,
        ),
        "Poisson Boosting": lambda: HistGradientBoostingRegressor(loss="poisson", **common),
        "Squared-Error Boosting": lambda: HistGradientBoostingRegressor(loss="squared_error", **common),
        "Absolute-Error Boosting": lambda: HistGradientBoostingRegressor(loss="absolute_error", **common),
    }


BASELINES = {
    "Previous Week": "sales_lag_1",
    "Rolling Mean-4": "sales_mean_4",
    "Rolling Mean-13": "sales_mean_13",
    "Rolling Median-13": "sales_median_13",
    "Seasonal Naive-52": "sales_lag_52",
}


def aggregate_metrics(frame: pd.DataFrame, prediction_column: str) -> list[dict]:
    levels = {
        "store-family": ["week", "store_nbr", "family"],
        "store": ["week", "store_nbr"],
        "family": ["week", "family"],
        "company-total": ["week"],
    }
    rows = []
    for level, keys in levels.items():
        grouped = frame.groupby(keys, as_index=False)[["sales", prediction_column]].sum()
        rows.append({
            "level": level,
            "WAPE": wape(grouped["sales"], grouped[prediction_column]),
            "Bias": bias(grouped["sales"], grouped[prediction_column]),
            "observations": len(grouped),
        })
    return rows


def baseline_prediction(frame: pd.DataFrame, column: str) -> np.ndarray:
    prediction = frame[column].copy()
    if column == "sales_lag_52":
        prediction = prediction.fillna(frame["sales_mean_13"])
    prediction = prediction.fillna(frame["expanding_mean"]).fillna(0)
    return np.maximum(0, prediction.to_numpy(dtype=float))


def weekly_block_bootstrap(frame: pd.DataFrame, iterations: int = 5000) -> dict:
    """Return a week-block confidence interval for baseline minus model WAPE."""
    weekly = frame.assign(
        model_absolute_error=(frame["sales"] - frame["prediction"]).abs(),
        baseline_absolute_error=(frame["sales"] - frame["baseline_prediction"]).abs(),
        absolute_actual=frame["sales"].abs(),
    ).groupby("week", as_index=False)[
        ["model_absolute_error", "baseline_absolute_error", "absolute_actual"]
    ].sum()
    rng = np.random.default_rng(SEED)
    sampled = rng.integers(0, len(weekly), size=(iterations, len(weekly)))
    actual = weekly["absolute_actual"].to_numpy()[sampled].sum(axis=1)
    model_error = weekly["model_absolute_error"].to_numpy()[sampled].sum(axis=1)
    baseline_error = weekly["baseline_absolute_error"].to_numpy()[sampled].sum(axis=1)
    improvements = (baseline_error - model_error) / actual * 100
    low, high = np.percentile(improvements, [2.5, 97.5])
    return {
        "improvement_pp": float(
            wape(frame["sales"], frame["baseline_prediction"])
            - wape(frame["sales"], frame["prediction"])
        ),
        "bootstrap_95_low": float(low),
        "bootstrap_95_high": float(high),
        "bootstrap_unit": "week",
        "bootstrap_iterations": iterations,
    }


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Building Favorita weekly store-family panel...")
    frame = add_features(build_weekly_panel())
    usable = frame[frame["sales_lag_4"].notna()].copy()
    weeks = np.array(sorted(usable["week"].unique()))
    development_weeks, holdout_weeks = weeks[:-HOLDOUT_WEEKS], weeks[-HOLDOUT_WEEKS:]
    splitter = TimeSeriesSplit(n_splits=CV_FOLDS, test_size=CV_WEEKS)

    fold_rows: list[dict] = []
    cv_predictions: list[pd.DataFrame] = []
    factories = model_factories()
    for fold, (train_idx, validation_idx) in enumerate(splitter.split(development_weeks), 1):
        train_weeks = development_weeks[train_idx]
        validation_weeks = development_weeks[validation_idx]
        train = usable[usable["week"].isin(train_weeks)]
        validation = usable[usable["week"].isin(validation_weeks)].copy()
        print(
            f"Fold {fold}/{CV_FOLDS}: {train_weeks[0].date()}..{train_weeks[-1].date()} -> "
            f"{validation_weeks[0].date()}..{validation_weeks[-1].date()}"
        )
        for name, factory in factories.items():
            estimator = factory()
            estimator.fit(train[FEATURES], train["sales"])
            validation["prediction"] = np.maximum(0, estimator.predict(validation[FEATURES]))
            for row in aggregate_metrics(validation, "prediction"):
                row.update({"fold": fold, "model": name, "model_type": "Candidate"})
                fold_rows.append(row)
            saved = validation[["week", "store_nbr", "family", "sales", "prediction"]].copy()
            saved["fold"], saved["model"] = fold, name
            cv_predictions.append(saved)
        for name, column in BASELINES.items():
            validation["prediction"] = baseline_prediction(validation, column)
            for row in aggregate_metrics(validation, "prediction"):
                row.update({"fold": fold, "model": name, "model_type": "Baseline"})
                fold_rows.append(row)
            saved = validation[["week", "store_nbr", "family", "sales", "prediction"]].copy()
            saved["fold"], saved["model"] = fold, name
            cv_predictions.append(saved)

    fold_metrics = pd.DataFrame(fold_rows)
    primary = fold_metrics[fold_metrics["level"] == "store-family"].copy()
    summaries = primary.groupby(["model", "model_type"], as_index=False).agg(
        CV_WAPE=("WAPE", "mean"), CV_WAPE_SD=("WAPE", "std"),
        CV_Bias=("Bias", "mean"), folds=("fold", "nunique"),
    )
    baseline_summary = summaries[(summaries["model_type"] == "Baseline") & (summaries["CV_Bias"].abs() <= 10)]
    baseline_summary = baseline_summary.sort_values("CV_WAPE")
    selected_baseline = baseline_summary.iloc[0]["model"]
    baseline_fold = primary[primary["model"] == selected_baseline].set_index("fold")["WAPE"]

    candidate_summary = summaries[summaries["model_type"] == "Candidate"].copy()
    candidate_summary["fold_wins_vs_baseline"] = candidate_summary["model"].map(
        lambda name: int((primary[primary["model"] == name].set_index("fold")["WAPE"] < baseline_fold).sum())
    )
    baseline_wape = float(baseline_summary.iloc[0]["CV_WAPE"])
    candidate_summary["improvement_pp"] = baseline_wape - candidate_summary["CV_WAPE"]
    candidate_summary["relative_skill_pct"] = candidate_summary["improvement_pp"] / baseline_wape * 100
    eligible = candidate_summary[
        (candidate_summary["improvement_pp"] >= 0.5)
        & (candidate_summary["relative_skill_pct"] >= 5.0)
        & (candidate_summary["fold_wins_vs_baseline"] >= 3)
        & (candidate_summary["CV_Bias"].abs() <= 10)
    ].sort_values("CV_WAPE")
    selected_model = selected_baseline if eligible.empty else eligible.iloc[0]["model"]
    selected_type = "Baseline" if eligible.empty else "Candidate"

    development = usable[usable["week"].isin(development_weeks)]
    holdout = usable[usable["week"].isin(holdout_weeks)].copy()
    if selected_type == "Candidate":
        final_estimator = factories[selected_model]()
        final_estimator.fit(development[FEATURES], development["sales"])
        holdout["prediction"] = np.maximum(0, final_estimator.predict(holdout[FEATURES]))
    else:
        holdout["prediction"] = baseline_prediction(holdout, BASELINES[selected_model])

    holdout_rows = aggregate_metrics(holdout, "prediction")
    for row in holdout_rows:
        row.update({"model": selected_model, "model_type": selected_type})
    holdout_metrics = pd.DataFrame(holdout_rows)

    # Always score the selected comparison baseline on the holdout.
    holdout["baseline_prediction"] = baseline_prediction(holdout, BASELINES[selected_baseline])
    baseline_holdout_rows = aggregate_metrics(holdout, "baseline_prediction")
    for row in baseline_holdout_rows:
        row.update({"model": selected_baseline, "model_type": "Primary baseline"})
    holdout_metrics = pd.concat([holdout_metrics, pd.DataFrame(baseline_holdout_rows)], ignore_index=True)
    uncertainty = weekly_block_bootstrap(holdout)

    selection = {
        "selected_model": selected_model,
        "selected_type": selected_type,
        "primary_baseline": selected_baseline,
        "selection_level": "weekly store-family unit sales",
        "selection_rule": "Candidate must improve mean CV WAPE by >=0.5 pp and >=5% relatively, win >=3/4 folds, and have |bias| <=10%.",
        "protocol_revision": (
            "The preliminary 1 pp threshold inherited from the prior 88% WAPE problem was replaced before "
            "opening any candidate-model holdout predictions. At an 11% baseline WAPE, 1 pp would require "
            "an inconsistent 9% relative gain; the frozen rule requires both 0.5 pp absolute and 5% relative skill."
        ),
        "holdout_start": str(pd.Timestamp(holdout_weeks[0]).date()),
        "holdout_end": str(pd.Timestamp(holdout_weeks[-1]).date()),
        "development_weeks": int(len(development_weeks)),
        "holdout_weeks": HOLDOUT_WEEKS,
        "series": int(usable.groupby(GROUPS).ngroups),
        "rows": int(len(usable)),
        "holdout_improvement": uncertainty,
    }

    fold_metrics.to_csv(OUT / "fold_metrics.csv", index=False)
    summaries.sort_values(["model_type", "CV_WAPE"]).to_csv(OUT / "cv_summary.csv", index=False)
    holdout_metrics.to_csv(OUT / "holdout_metrics.csv", index=False)
    pd.concat(cv_predictions, ignore_index=True).to_csv(OUT / "cv_predictions.csv.gz", index=False, compression="gzip")
    holdout[[
        "week", "store_nbr", "family", "sales", "prediction", "baseline_prediction",
        "promo_items", "holiday_days",
    ]].to_csv(OUT / "holdout_predictions.csv.gz", index=False, compression="gzip")
    with (OUT / "selection.json").open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2)

    if selected_type == "Candidate":
        deployment_estimator = factories[selected_model]()
        deployment_estimator.fit(usable[FEATURES], usable["sales"])
        joblib.dump({
            "model": deployment_estimator,
            "features": FEATURES,
            "selected_model": selected_model,
            "trained_through": str(pd.Timestamp(weeks[-1]).date()),
            "forecast_grain": "weekly store-family unit sales",
        }, MODEL_OUT)

    print("\nCross-validation summary (store-family level):")
    print(summaries.sort_values("CV_WAPE").to_string(index=False))
    print("\nHoldout metrics:")
    print(holdout_metrics.to_string(index=False))
    print("\nSelection:")
    print(json.dumps(selection, indent=2))
    return selection


if __name__ == "__main__":
    main()
