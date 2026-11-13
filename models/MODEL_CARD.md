# Retail Decision Lab Forecast Model Card

## Forecast contract

- Target: weekly unit sales
- Horizon: one week ahead
- Grain: store × product family × week
- Series: 1,782
- Stores: 54
- Product families: 33
- Holdout: 2017-06-25 through 2017-08-13
- Output constraint: non-negative predictions

## Selected model

LightGBM with Tweedie loss was selected using four expanding-window validation folds. It qualified by improving mean CV WAPE by at least 0.5 percentage points and 5% relatively, winning at least three folds, and remaining within ±10% aggregate bias.

### Untouched holdout

| Level | LightGBM WAPE | Baseline WAPE |
|---|---:|---:|
| Store-family | 7.25% | 8.63% |
| Store | 4.78% | 6.35% |
| Family | 3.30% | 5.65% |
| Company total | 1.90% | 4.53% |

- Model bias: 1.49%
- Store-family improvement: 1.38 percentage points
- 95% week-block bootstrap interval: 0.66 to 2.03 points

## Inputs

- Store and product-family identifiers
- Store type and cluster
- Calendar seasonality and trend
- Known promotion counts and holiday days
- Oil price
- Shifted unit-sales lags and rolling summaries
- Shifted store transactions

## Leakage controls

- Incomplete final calendar weeks are excluded.
- Structural pre-opening zeroes are excluded.
- Demand and transaction-derived predictors are shifted by at least one week.
- Folds and holdout are chronological.
- Model selection occurs before holdout scoring.
- The final deployment artifact is refit only after evaluation is complete.

## Intended use

This model supports portfolio-scale weekly demand planning and demonstrates global forecasting across many related retail series. It is appropriate for scenario analysis, prioritization, and exception monitoring.

## Limitations

- It is not a replenishment optimizer and does not model inventory availability or lost sales.
- The holdout contains eight weeks.
- Future promotions and calendar inputs must be supplied accurately.
- Public data does not establish production performance for another retailer.

## Reproduction

```bash
python run_favorita_pipeline.py
```
