import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from common import init_page, render_chart, section_card


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports" / "favorita_analytics"
init_page()
section_card(
    "Demand-spike classification",
    "A time-split LightGBM classifier flags store–family weeks likely to exceed their own recent high-demand threshold.",
)

metrics = json.loads((OUTPUT / "classification_metrics.json").read_text())
importance = pd.read_csv(OUTPUT / "classification_importance.csv").head(15)

c1, c2, c3, c4 = st.columns(4)
c1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
c2.metric("PR-AUC", f"{metrics['pr_auc']:.3f}", f"{metrics['positive_rate']:.1%} event rate")
c3.metric("Precision", f"{metrics['precision']:.3f}")
c4.metric("Recall", f"{metrics['recall']:.3f}")

fig = px.bar(
    importance.sort_values("importance"), x="importance", y="feature", orientation="h",
)
render_chart(fig, "Most-used classification features", height=520)

st.markdown("### Evaluation design")
st.write(
    f"A spike is defined as weekly demand above the store–family’s prior 26-week 75th percentile. "
    f"The model trained on {metrics['train_rows']:,} earlier observations and was evaluated on "
    f"{metrics['test_rows']:,} later observations beginning {metrics['test_start']}."
)
st.caption(
    "The threshold uses past weeks only. Predictions use lagged demand, known promotions, holidays, store metadata and calendar features."
)
