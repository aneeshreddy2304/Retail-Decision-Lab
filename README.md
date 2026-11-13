# Retail Decision Lab

Retail Decision Lab is an end-to-end retail demand forecasting and decision-science application built on the Corporación Favorita store-sales dataset. It forecasts weekly unit demand across 54 stores and 33 product families, evaluates predictions across the retail hierarchy, studies promotion associations, flags demand spikes, and exposes the results through a Streamlit dashboard.

## Verified results

The forecasting experiment uses four expanding-window validation folds followed by an untouched eight-week holdout.

| Decision level | LightGBM Tweedie WAPE | Rolling Mean-4 WAPE |
|---|---:|---:|
| Store × product family | **7.25%** | 8.63% |
| Store | **4.78%** | 6.35% |
| Product family | **3.30%** | 5.65% |
| Company total | **1.90%** | 4.53% |

- Holdout bias: **1.49%**
- Store-family improvement: **1.38 percentage points**
- 95% week-block bootstrap interval: **0.66 to 2.03 points**
- Demand-spike classifier: **0.743 ROC-AUC**, **0.570 PR-AUC**

## Product capabilities

- Forecast monitor with holdout uncertainty and hierarchical accuracy
- Executive demand overview across stores, states, and product families
- Store-family demand explorer and seasonality views
- Baseline and ML comparison using rolling-origin validation
- Promotion intelligence normalized against past-only local demand levels
- Time-split demand-spike classification
- SQLite marts using CTEs, ranks, and window functions
- Upload workflow for next-week promotion and calendar plans
- Persisted LightGBM forecasting and classification artifacts
- Docker-ready Streamlit application

## Methodology

Daily observations are aggregated into complete Sunday-ending weeks. Structural zeroes before a store becomes active are excluded. All demand, transaction, rolling, and expanding features are shifted so they use only earlier weeks. Current promotion and holiday variables are allowed because they represent known planning inputs.

Candidates include LightGBM Tweedie and L1 models, histogram gradient boosting with multiple losses, and rolling/seasonal baselines. A candidate must improve mean CV WAPE by at least 0.5 points and 5% relatively, win at least three of four folds, and remain within ±10% bias.

The promotion analysis is observational. It normalizes demand against each store-family’s prior 13-week median and uses a week-block bootstrap, but it does not claim that promotions caused the difference.

## Data

Place the official Kaggle files under `data/raw/favorita/`:

- `train.csv`
- `test.csv`
- `stores.csv`
- `transactions.csv`
- `oil.csv`
- `holidays_events.csv`

Raw competition data is intentionally ignored by Git.

## Run locally

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_favorita_pipeline.py
streamlit run app/main.py
```

The full pipeline rebuilds the weekly panel, forecasting benchmark, model artifacts, uncertainty estimates, promotion analysis, demand-spike classifier, and SQL marts.

## Upload contract

The application accepts a one-week planning CSV with:

```text
week,store_nbr,family,promo_items,holiday_days,oil_price
```

The forecast combines those known future inputs with lagged historical demand and transactions. The page provides a downloadable sample plan.

## Repository map

```text
app/                         Streamlit interface
src/favorita_benchmark.py    Forecasting experiment and model selection
src/favorita_analytics.py    Promotion, classification, and SQL artifacts
src/favorita_inference.py    Upload-based next-week inference
run_favorita_pipeline.py     End-to-end reproducible runner
reports/favorita_benchmark/  CV, holdout, predictions, and selection evidence
reports/favorita_analytics/  Inference and classification evidence
data/favorita_sql/           SQLite database, SQL, and exported marts
models/                      Persisted production artifacts and model card
tests/                       Leakage and metric correctness tests
```

## Limitations

- Public competition data demonstrates methodology, not a deployed commercial system.
- The primary forecast is weekly unit demand, not revenue or profit.
- The final holdout contains eight weeks; uncertainty is reported explicitly.
- Accuracy at different aggregation levels supports different decisions and should not be compared as if it were the same task.
