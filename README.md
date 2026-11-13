<p align="center"><img src="docs/assets/banner.svg" alt="Retail Decision Lab — Turn weekly demand signals into planning decisions." width="100%"></p>

<p align="center"><img src="docs/assets/badges/python.svg" alt="python" height="28"> <img src="docs/assets/badges/streamlit.svg" alt="streamlit" height="28"> <img src="docs/assets/badges/pandas.svg" alt="pandas" height="28"> <img src="docs/assets/badges/scikit-learn.svg" alt="scikit-learn" height="28"> <img src="docs/assets/badges/sqlite.svg" alt="sqlite" height="28"> <img src="docs/assets/badges/docker.svg" alt="docker" height="28"></p>

<p align="center"><strong>A Streamlit analytics workspace for demand forecasting, promotion analysis, SQL exploration, and one-week retail planning.</strong></p>

## Overview

A Streamlit analytics workspace for demand forecasting, promotion analysis, SQL exploration, and one-week retail planning.

### Why it matters

Retail Decision Lab connects model development to the questions a planner actually asks—what demand is changing, where it is changing, and how to prepare the next-week plan.

## Key features

- Forecast monitoring and executive demand views
- Store and product-family exploration
- Promotion and demand-spike analysis
- Upload-based next-week planning workflow
- SQLite marts for reusable business queries

## Architecture

```mermaid
flowchart LR
  A[Kaggle Favorita files] --> B[Weekly panel]
  B --> C[Feature pipeline]
  C --> D[Forecast + classification models]
  B --> E[SQLite marts]
  D --> F[Streamlit workspace]
  E --> F
  G[Planning CSV] --> F
```

## Technology stack

**python** · **streamlit** · **pandas** · **scikit learn** · **sqlite** · **docker**

## Getting started

Create a Python 3.13 environment, install `requirements.txt`, and place the official Favorita files under `data/raw/favorita/`. Run `python run_favorita_pipeline.py`, then `streamlit run app/main.py`.
