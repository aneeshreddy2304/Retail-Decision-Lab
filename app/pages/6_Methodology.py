import streamlit as st

from common import init_page, load_favorita_selection, section_card


init_page()
selection = load_favorita_selection()
uncertainty = selection["holdout_improvement"]
section_card(
    "Methodology",
    "A reproducible global forecasting workflow with complete weekly panels, rolling-origin validation and an untouched holdout.",
)

st.markdown("### Forecast contract")
for column, (label, value) in zip(st.columns(4), [
    ("Target", "Weekly unit sales"),
    ("Grain", "Store × product family"),
    ("Series", f"{selection['series']:,}"),
    ("Holdout", f"{selection['holdout_weeks']} weeks"),
]):
    column.metric(label, value)

st.caption(
    f"Untouched holdout: {selection['holdout_start']} to {selection['holdout_end']} · "
    "Promotions and calendar events are treated as known future inputs; transactions are lagged."
)

st.markdown(
    f"""
    ### Evaluation workflow

    1. Aggregate daily sales into complete Sunday-ending store–family weeks.
    2. Exclude structural zeroes before each store first became active.
    3. Generate lags and rolling statistics exclusively from earlier weeks.
    4. Use current promotion and holiday schedules because they are known before the forecast week.
    5. Compare candidates on four expanding-window folds of eight weeks each.
    6. Select at the store–family level using WAPE with a ±10% bias guardrail.
    7. Open the final eight-week holdout only after model selection.
    8. Quantify improvement uncertainty by resampling complete weeks.

    ### Models evaluated

    - LightGBM with Tweedie loss
    - LightGBM with absolute-error loss
    - Histogram gradient boosting with squared, absolute and Poisson losses
    - Previous-week, seasonal-naive, rolling-mean and rolling-median baselines

    ### Selection rule

    {selection['selection_rule']}

    **Selected:** {selection['selected_model']}  
    **Primary baseline:** {selection['primary_baseline']}

    ### Final evidence

    The selected model improved store–family holdout WAPE by
    **{uncertainty['improvement_pp']:.2f} percentage points**. The 95% week-block
    interval was **{uncertainty['bootstrap_95_low']:.2f} to
    {uncertainty['bootstrap_95_high']:.2f} points**.

    ### Limitations

    - This is a weekly unit-demand forecast, not a revenue or profit forecast.
    - Promotion comparisons are observational and must not be interpreted causally.
    - Only eight weeks are available in the final holdout; uncertainty is reported explicitly.
    - Public competition data demonstrates methodology rather than a deployed commercial system.
    - Forecast accuracy naturally improves at higher aggregation levels; those values describe different decisions.
    """
)

with st.expander("Protocol revision disclosure"):
    st.write(selection["protocol_revision"])
