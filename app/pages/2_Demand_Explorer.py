import plotly.express as px
import streamlit as st

from common import favorita_filters, init_page, load_favorita_weekly, render_chart, section_card


init_page()
df = load_favorita_weekly()
filtered = favorita_filters(df)
section_card(
    "Demand explorer",
    "Inspect seasonality, product-family mix and store-level demand without conflating aggregation levels.",
)

monthly = filtered.assign(month=filtered["week"].dt.to_period("M").dt.to_timestamp())
top_families = monthly.groupby("family")["sales"].sum().nlargest(8).index
monthly = monthly[monthly["family"].isin(top_families)].groupby(
    ["month", "family"], as_index=False
)["sales"].sum()
fig = px.line(monthly, x="month", y="sales", color="family")
render_chart(fig, "Monthly demand for leading product families", height=440)

family_store = filtered.groupby(["store_nbr", "family"], as_index=False)["sales"].sum()
top_stores = family_store.groupby("store_nbr")["sales"].sum().nlargest(12).index
top_family_set = family_store.groupby("family")["sales"].sum().nlargest(10).index
heat = family_store[
    family_store["store_nbr"].isin(top_stores) & family_store["family"].isin(top_family_set)
]
heatmap = px.density_heatmap(
    heat, x="store_nbr", y="family", z="sales", histfunc="sum",
    color_continuous_scale=[[0, "#f1f3ef"], [1, "#35463a"]],
)
render_chart(heatmap, "Demand concentration: top stores and families", height=500)

summary = filtered.groupby("family", as_index=False).agg(
    total_units=("sales", "sum"),
    average_weekly_units=("sales", "mean"),
    active_stores=("store_nbr", "nunique"),
    promotion_exposure=("promo_items", lambda values: (values > 0).mean()),
).sort_values("total_units", ascending=False)
summary["promotion_exposure"] *= 100
st.markdown("### Product-family summary")
st.dataframe(
    summary, width="stretch", hide_index=True,
    column_config={
        "total_units": st.column_config.NumberColumn(format="%,.0f"),
        "average_weekly_units": st.column_config.NumberColumn(format="%,.1f"),
        "promotion_exposure": st.column_config.NumberColumn(format="%.1f%%"),
    },
)
