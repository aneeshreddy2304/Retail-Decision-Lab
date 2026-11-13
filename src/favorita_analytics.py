"""Classification, promotion inference, and SQL artifacts for Favorita."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score, brier_score_loss, precision_score, recall_score,
    roc_auc_score,
)

from favorita_benchmark import FEATURES, OUT, ROOT, SEED, add_features


WEEKLY_PATH = ROOT / "data" / "processed" / "favorita_weekly.parquet"
ANALYTICS = ROOT / "reports" / "favorita_analytics"
SQL_OUT = ROOT / "data" / "favorita_sql"
CLASSIFIER_PATH = ROOT / "models" / "favorita_demand_spike_classifier.joblib"


def promotion_analysis(frame: pd.DataFrame) -> None:
    ANALYTICS.mkdir(parents=True, exist_ok=True)
    work = frame.copy().sort_values(["store_nbr", "family", "week"])
    work["promoted"] = work["promo_items"].gt(0)
    # Compare demand with a past-only local level to reduce store/family scale confounding.
    work["expected_units"] = work.groupby(["store_nbr", "family"])["sales"].transform(
        lambda values: values.shift(1).rolling(13, min_periods=6).median()
    )
    work = work[work["expected_units"].gt(0)].copy()
    work["relative_demand"] = work["sales"] / work["expected_units"]
    work["relative_demand"] = work["relative_demand"].clip(upper=10)

    promoted = work[work["promoted"]]["relative_demand"]
    unpromoted = work[~work["promoted"]]["relative_demand"]
    observed_difference = float(promoted.mean() - unpromoted.mean())

    weekly = work.groupby(["week", "promoted"], as_index=False)["relative_demand"].mean()
    pivot = weekly.pivot(index="week", columns="promoted", values="relative_demand").dropna()
    weekly_differences = (pivot[True] - pivot[False]).to_numpy()
    rng = np.random.default_rng(SEED)
    samples = rng.choice(weekly_differences, size=(5000, len(weekly_differences)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])

    family = work.groupby(["family", "promoted"], as_index=False).agg(
        mean_relative_demand=("relative_demand", "mean"), observations=("sales", "size")
    )
    family = family.pivot(index="family", columns="promoted", values=["mean_relative_demand", "observations"])
    family.columns = [f"{metric}_{'promoted' if status else 'not_promoted'}" for metric, status in family.columns]
    family = family.reset_index()
    family["observed_relative_difference"] = (
        family["mean_relative_demand_promoted"] - family["mean_relative_demand_not_promoted"]
    )
    family.sort_values("observed_relative_difference", ascending=False).to_csv(
        ANALYTICS / "promotion_by_family.csv", index=False
    )

    results = {
        "method": "Past-only 13-week median normalization with week-block bootstrap",
        "promoted_observations": int(work["promoted"].sum()),
        "not_promoted_observations": int((~work["promoted"]).sum()),
        "mean_relative_demand_promoted": float(promoted.mean()),
        "mean_relative_demand_not_promoted": float(unpromoted.mean()),
        "observed_relative_difference": observed_difference,
        "bootstrap_95_low": float(low),
        "bootstrap_95_high": float(high),
        "causal_claim": False,
    }
    (ANALYTICS / "promotion_inference.json").write_text(json.dumps(results, indent=2))


def train_classifier(frame: pd.DataFrame) -> None:
    ANALYTICS.mkdir(parents=True, exist_ok=True)
    featured = add_features(frame)
    trailing_threshold = featured.groupby(["store_nbr", "family"])["sales"].transform(
        lambda values: values.shift(1).rolling(26, min_periods=13).quantile(0.75)
    )
    featured["demand_spike"] = (featured["sales"] > trailing_threshold).astype(int)
    model_data = featured[trailing_threshold.notna()].dropna(subset=["sales_lag_4"]).copy()
    weeks = np.array(sorted(model_data["week"].unique()))
    cutoff = weeks[int(len(weeks) * 0.8)]
    train = model_data[model_data["week"] < cutoff]
    test = model_data[model_data["week"] >= cutoff]

    model = LGBMClassifier(
        objective="binary", n_estimators=350, learning_rate=0.04, num_leaves=31,
        min_child_samples=80, colsample_bytree=0.85, reg_lambda=1.0,
        class_weight="balanced", random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    model.fit(train[FEATURES], train["demand_spike"])
    probability = model.predict_proba(test[FEATURES])[:, 1]
    prediction = probability >= 0.5
    metrics = {
        "target": "Weekly demand exceeds its prior 26-week 75th percentile",
        "roc_auc": float(roc_auc_score(test["demand_spike"], probability)),
        "pr_auc": float(average_precision_score(test["demand_spike"], probability)),
        "precision": float(precision_score(test["demand_spike"], prediction, zero_division=0)),
        "recall": float(recall_score(test["demand_spike"], prediction, zero_division=0)),
        "brier_score": float(brier_score_loss(test["demand_spike"], probability)),
        "positive_rate": float(test["demand_spike"].mean()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_start": str(pd.Timestamp(cutoff).date()),
    }
    (ANALYTICS / "classification_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(ANALYTICS / "classification_importance.csv", index=False)
    joblib.dump({"model": model, "features": FEATURES, "metrics": metrics}, CLASSIFIER_PATH)


def build_sql_marts(frame: pd.DataFrame) -> None:
    SQL_OUT.mkdir(parents=True, exist_ok=True)
    database = SQL_OUT / "favorita_analytics.db"
    with sqlite3.connect(database) as connection:
        frame.to_sql("weekly_demand", connection, if_exists="replace", index=False)
        queries = {
            "family_performance": """
                SELECT family, SUM(sales) AS units, SUM(promo_items) AS promoted_items,
                       AVG(sales) AS average_weekly_units,
                       RANK() OVER (ORDER BY SUM(sales) DESC) AS demand_rank
                FROM weekly_demand GROUP BY family ORDER BY units DESC
            """,
            "store_performance": """
                SELECT store_nbr, city, state, type, SUM(sales) AS units,
                       SUM(transactions) / COUNT(DISTINCT family) AS transactions,
                       RANK() OVER (ORDER BY SUM(sales) DESC) AS demand_rank
                FROM weekly_demand GROUP BY store_nbr, city, state, type ORDER BY units DESC
            """,
            "promotion_performance": """
                WITH tagged AS (
                    SELECT *, CASE WHEN promo_items > 0 THEN 'Promoted' ELSE 'Not promoted' END AS status
                    FROM weekly_demand
                )
                SELECT status, COUNT(*) AS observations, AVG(sales) AS average_units,
                       SUM(sales) AS total_units FROM tagged GROUP BY status
            """,
            "rolling_company_demand": """
                WITH company AS (
                    SELECT week, SUM(sales) AS units FROM weekly_demand GROUP BY week
                )
                SELECT week, units,
                       AVG(units) OVER (ORDER BY week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS four_week_average
                FROM company ORDER BY week
            """,
        }
        for name, query in queries.items():
            pd.read_sql_query(query, connection).to_csv(SQL_OUT / f"{name}.csv", index=False)
        (SQL_OUT / "queries.sql").write_text("\n\n".join(query.strip() + ";" for query in queries.values()))


def main() -> None:
    frame = pd.read_parquet(WEEKLY_PATH)
    frame["week"] = pd.to_datetime(frame["week"])
    promotion_analysis(frame)
    train_classifier(frame)
    build_sql_marts(frame)
    print("Favorita analytics artifacts generated.")


if __name__ == "__main__":
    main()
