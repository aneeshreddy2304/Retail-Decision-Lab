import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

from common import init_page, render_chart, section_card

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from favorita_inference import forecast_plan, sample_plan


init_page()
section_card(
    "Upload and forecast",
    "Submit one week of known promotion, holiday and oil-price inputs for selected store–family combinations.",
)

st.markdown(
    "The saved production model combines your future plan with the project’s historical lag and rolling features. "
    "It supports the next Sunday-ending week only; it does not silently use future sales or transactions."
)

sample = sample_plan()
st.download_button(
    "Download sample plan",
    sample.to_csv(index=False).encode("utf-8"),
    file_name="favorita_weekly_plan.csv",
    mime="text/csv",
)
with st.expander("Required columns"):
    st.code("week, store_nbr, family, promo_items, holiday_days, oil_price")

uploaded = st.file_uploader("Upload weekly demand plan", type=["csv"])
if uploaded is not None:
    try:
        result = forecast_plan(uploaded)
        st.success("Forecast generated successfully.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Forecast week", result["week"].dt.date.iloc[0].isoformat())
        c2.metric("Store–family rows", f"{len(result):,}")
        c3.metric("Forecast units", f"{result['forecast_units'].sum():,.0f}")

        chart = result.groupby("family", as_index=False)["forecast_units"].sum().nlargest(15, "forecast_units")
        fig = px.bar(chart.sort_values("forecast_units"), x="forecast_units", y="family", orientation="h")
        render_chart(fig, "Forecast demand by product family", height=460)
        st.dataframe(result, width="stretch", hide_index=True)
        st.download_button(
            "Download forecast",
            result.to_csv(index=False).encode("utf-8"),
            file_name="retail_decision_lab_forecast.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(str(exc))
