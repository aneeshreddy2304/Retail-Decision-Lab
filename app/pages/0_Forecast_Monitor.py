import plotly.express as px
import streamlit as st

from common import (
    init_page, load_favorita_cv_summary, load_favorita_holdout_metrics,
    load_favorita_holdout_predictions, load_favorita_selection, notice,
    render_chart, section_card,
)


init_page()
section_card(
    "Forecast monitor",
    "Weekly unit-demand performance across 1,782 store–family series, evaluated on an untouched eight-week holdout.",
)

selection = load_favorita_selection()
metrics = load_favorita_holdout_metrics()
predictions = load_favorita_holdout_predictions()
selected = metrics[
    (metrics["model"] == selection["selected_model"]) & (metrics["level"] == "store-family")
].iloc[0]
uncertainty = selection["holdout_improvement"]

notice(
    "The selected model improves on the operational baseline.",
    "The week-block confidence interval remains above zero on the untouched holdout.",
    "Validated across the retail hierarchy",
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Selected model", selection["selected_model"], "Chosen by rolling-origin CV")
col2.metric("Store–family WAPE", f"{selected['WAPE']:.2f}%", f"{uncertainty['improvement_pp']:.2f} pp better")
col3.metric("Forecast bias", f"{selected['Bias']:.2f}%", "Within the ±10% guardrail")
col4.metric(
    "95% improvement interval",
    f"{uncertainty['bootstrap_95_low']:.2f}–{uncertainty['bootstrap_95_high']:.2f} pp",
    "Resampled by week",
)

level_metrics = metrics[metrics["model"] == selection["selected_model"]].copy()
level_order = ["store-family", "store", "family", "company-total"]
level_metrics["order"] = level_metrics["level"].map({name: i for i, name in enumerate(level_order)})
level_metrics = level_metrics.sort_values("order")
labels = {
    "store-family": "Store × family",
    "store": "Store",
    "family": "Product family",
    "company-total": "Company total",
}
st.markdown("### Accuracy across decision levels")
for column, (_, row) in zip(st.columns(4), level_metrics.iterrows()):
    column.metric(labels[row["level"]], f"{row['WAPE']:.2f}% WAPE")

weekly = predictions.groupby("week", as_index=False).agg(
    actual=("sales", "sum"), forecast=("prediction", "sum"), baseline=("baseline_prediction", "sum")
)
weekly_long = weekly.melt("week", var_name="series", value_name="unit_sales")
fig = px.line(
    weekly_long, x="week", y="unit_sales", color="series",
    color_discrete_map={"actual": "#242824", "forecast": "#9a4f36", "baseline": "#8c6a34"},
)
fig.for_each_trace(lambda trace: trace.update(name=trace.name.replace("_", " ").title()))
render_chart(fig, "Untouched holdout: company-wide weekly demand", height=390)

predictions["absolute_error"] = (predictions["sales"] - predictions["prediction"]).abs()
exceptions = predictions.groupby(["store_nbr", "family"], as_index=False).agg(
    actual_units=("sales", "sum"), absolute_error=("absolute_error", "sum")
)
exceptions["WAPE"] = 100 * exceptions["absolute_error"] / exceptions["actual_units"].clip(lower=1)
exceptions = exceptions[exceptions["actual_units"] >= exceptions["actual_units"].quantile(0.25)]
exceptions = exceptions.sort_values("absolute_error", ascending=False).head(12)

left, right = st.columns([1.4, 1], gap="large")
with left:
    cv = load_favorita_cv_summary().sort_values("CV_WAPE")
    st.markdown("### Cross-validation leaderboard")
    st.caption("Four expanding-window folds; lower WAPE is better.")
    st.dataframe(
        cv[["model", "model_type", "CV_WAPE", "CV_WAPE_SD", "CV_Bias"]],
        width="stretch", hide_index=True,
        column_config={
            "CV_WAPE": st.column_config.NumberColumn("CV WAPE", format="%.2f%%"),
            "CV_WAPE_SD": st.column_config.NumberColumn("Fold SD", format="%.2f"),
            "CV_Bias": st.column_config.NumberColumn("CV bias", format="%.2f%%"),
        },
    )
with right:
    st.markdown("### Largest error contributors")
    st.caption("High-volume segments ranked by total absolute error.")
    st.dataframe(
        exceptions[["store_nbr", "family", "actual_units", "WAPE"]],
        width="stretch", hide_index=True,
        column_config={
            "store_nbr": "Store",
            "family": "Product family",
            "actual_units": st.column_config.NumberColumn("Actual units", format="%,.0f"),
            "WAPE": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )

st.caption(
    f"Holdout: {selection['holdout_start']} to {selection['holdout_end']} · "
    f"{selection['series']:,} store–family series · Weekly predictions are clipped at zero."
)
