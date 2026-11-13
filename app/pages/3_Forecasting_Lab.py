import plotly.express as px
import streamlit as st

from common import (
    init_page, load_favorita_cv_summary, load_favorita_fold_metrics,
    load_favorita_holdout_predictions, load_favorita_selection, notice,
    render_chart, section_card,
)


init_page()
selection = load_favorita_selection()
summary = load_favorita_cv_summary().sort_values("CV_WAPE")
folds = load_favorita_fold_metrics()
predictions = load_favorita_holdout_predictions()

section_card(
    "Forecast evaluation",
    "Rolling-origin evidence, baseline skill and holdout diagnostics for weekly store–family unit demand.",
)
notice(
    f"{selection['selected_model']} passed the frozen selection gate.",
    "It improved validation WAPE, won three of four folds, remained inside the bias guardrail and improved the untouched holdout.",
    "Baseline: Rolling Mean-4",
)

selected_cv = summary[summary["model"] == selection["selected_model"]].iloc[0]
baseline_cv = summary[summary["model"] == selection["primary_baseline"]].iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Selected model", selection["selected_model"])
c2.metric("Mean CV WAPE", f"{selected_cv['CV_WAPE']:.2f}%", f"{baseline_cv['CV_WAPE'] - selected_cv['CV_WAPE']:.2f} pp")
c3.metric("Mean CV bias", f"{selected_cv['CV_Bias']:.2f}%")
c4.metric("Validation folds", "4", "Eight weeks per fold")

st.markdown("### Candidate and baseline comparison")
st.dataframe(
    summary[["model", "model_type", "CV_WAPE", "CV_WAPE_SD", "CV_Bias", "folds"]],
    width="stretch", hide_index=True,
    column_config={
        "CV_WAPE": st.column_config.NumberColumn("CV WAPE", format="%.2f%%"),
        "CV_WAPE_SD": st.column_config.NumberColumn("Fold SD", format="%.2f"),
        "CV_Bias": st.column_config.NumberColumn("CV bias", format="%.2f%%"),
    },
)

fold_view = folds[
    (folds["level"] == "store-family")
    & folds["model"].isin([selection["selected_model"], selection["primary_baseline"]])
]
fold_chart = px.line(fold_view, x="fold", y="WAPE", color="model", markers=True)
render_chart(fold_chart, "WAPE by rolling-origin fold", height=360)

st.markdown("### Store–family holdout diagnostic")
f1, f2 = st.columns(2)
stores = sorted(predictions["store_nbr"].unique())
families = sorted(predictions["family"].unique())
store = f1.selectbox("Store", stores)
family = f2.selectbox("Product family", families)
segment = predictions[(predictions["store_nbr"] == store) & (predictions["family"] == family)].copy()
segment["absolute_error"] = (segment["sales"] - segment["prediction"]).abs()
segment_wape = 100 * segment["absolute_error"].sum() / max(segment["sales"].sum(), 1)

m1, m2, m3 = st.columns(3)
m1.metric("Segment WAPE", f"{segment_wape:.1f}%")
m2.metric("Actual units", f"{segment['sales'].sum():,.0f}")
m3.metric("Forecast units", f"{segment['prediction'].sum():,.0f}")

long = segment.melt(
    id_vars="week", value_vars=["sales", "prediction", "baseline_prediction"],
    var_name="series", value_name="units",
)
segment_chart = px.line(long, x="week", y="units", color="series", markers=True)
segment_chart.for_each_trace(lambda trace: trace.update(
    name={"sales": "Actual", "prediction": "LightGBM", "baseline_prediction": "Baseline"}.get(trace.name, trace.name)
))
render_chart(segment_chart, f"Store {store} · {family}", height=410)
