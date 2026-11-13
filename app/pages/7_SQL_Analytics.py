from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from common import init_page, render_chart, section_card


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "favorita_sql"
init_page()
section_card(
    "SQL analytics",
    "SQLite marts built from weekly demand using CTEs, aggregation, ranking and rolling-window functions.",
)

families = pd.read_csv(OUTPUT / "family_performance.csv")
stores = pd.read_csv(OUTPUT / "store_performance.csv")
promotions = pd.read_csv(OUTPUT / "promotion_performance.csv")
rolling = pd.read_csv(OUTPUT / "rolling_company_demand.csv", parse_dates=["week"])

left, right = st.columns(2, gap="large")
with left:
    fig = px.bar(families.head(12).sort_values("units"), x="units", y="family", orientation="h")
    render_chart(fig, "Top product families from SQL mart", height=430)
with right:
    fig = px.bar(stores.head(12), x="store_nbr", y="units", color="type")
    render_chart(fig, "Top stores from SQL mart", height=430)

rolling_fig = px.line(rolling, x="week", y=["units", "four_week_average"])
render_chart(rolling_fig, "Company demand with SQL four-week window", height=390)

st.markdown("### Promotion aggregation")
st.dataframe(promotions, width="stretch", hide_index=True)
with st.expander("View SQL implementation"):
    st.code((OUTPUT / "queries.sql").read_text(), language="sql")
