import plotly.express as px
import streamlit as st

from common import favorita_filters, init_page, load_favorita_weekly, render_chart, section_card


init_page()
df = load_favorita_weekly()
filtered = favorita_filters(df)
section_card(
    "Executive overview",
    "A concise view of unit demand, store coverage, product mix and promotion activity across the selected period.",
)

store_weeks = filtered[["week", "store_nbr", "transactions"]].drop_duplicates()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Units sold", f"{filtered['sales'].sum():,.0f}")
c2.metric("Transactions", f"{store_weeks['transactions'].sum():,.0f}")
c3.metric("Active stores", f"{filtered['store_nbr'].nunique()}")
c4.metric("Promoted observations", f"{(filtered['promo_items'] > 0).mean():.1%}")

weekly = filtered.groupby("week", as_index=False)["sales"].sum()
weekly["four_week_average"] = weekly["sales"].rolling(4, min_periods=1).mean()
trend = px.line(weekly, x="week", y=["sales", "four_week_average"])
trend.for_each_trace(lambda trace: trace.update(
    name="Weekly units" if trace.name == "sales" else "Four-week average"
))
render_chart(trend, "Company-wide unit demand", height=390)

family = filtered.groupby("family", as_index=False)["sales"].sum().nlargest(12, "sales")
state = filtered.groupby("state", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
left, right = st.columns(2, gap="large")
with left:
    family_fig = px.bar(family.sort_values("sales"), x="sales", y="family", orientation="h")
    render_chart(family_fig, "Top product families", height=430)
with right:
    state_fig = px.bar(state, x="state", y="sales")
    render_chart(state_fig, "Demand by state", height=430)
