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
    "Promotion intelligence",
    "Observed demand relative to each store–family’s own past-only 13-week level, with uncertainty and explicit causal limits.",
)

results = json.loads((OUTPUT / "promotion_inference.json").read_text())
families = pd.read_csv(OUTPUT / "promotion_by_family.csv")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Promoted observations", f"{results['promoted_observations']:,}")
c2.metric("Observed difference", f"{results['observed_relative_difference']:.1%}", "Relative to past local level")
c3.metric("95% interval low", f"{results['bootstrap_95_low']:.1%}")
c4.metric("95% interval high", f"{results['bootstrap_95_high']:.1%}")

st.warning(
    "This is an adjusted observational comparison, not a causal promotion-lift estimate. "
    "Campaign selection, timing and unmeasured factors can still confound the result."
)

overall = pd.DataFrame({
    "status": ["Not promoted", "Promoted"],
    "relative_demand": [
        results["mean_relative_demand_not_promoted"],
        results["mean_relative_demand_promoted"],
    ],
})
fig = px.bar(overall, x="status", y="relative_demand")
render_chart(fig, "Mean demand relative to the prior 13-week median", height=370)

eligible = families[
    (families["observations_promoted"] >= 100)
    & (families["observations_not_promoted"] >= 100)
].copy()
top = eligible.nlargest(15, "observed_relative_difference").sort_values("observed_relative_difference")
family_fig = px.bar(top, x="observed_relative_difference", y="family", orientation="h")
family_fig.update_xaxes(tickformat=".0%")
render_chart(family_fig, "Largest adjusted differences with adequate observations", height=500)

st.markdown("### Product-family evidence")
st.dataframe(
    eligible.sort_values("observed_relative_difference", ascending=False),
    width="stretch", hide_index=True,
    column_config={
        "mean_relative_demand_not_promoted": st.column_config.NumberColumn("Non-promoted index", format="%.2f"),
        "mean_relative_demand_promoted": st.column_config.NumberColumn("Promoted index", format="%.2f"),
        "observations_not_promoted": st.column_config.NumberColumn("Non-promoted n", format="%,.0f"),
        "observations_promoted": st.column_config.NumberColumn("Promoted n", format="%,.0f"),
        "observed_relative_difference": st.column_config.NumberColumn("Observed difference", format="%.1f%%"),
    },
)
