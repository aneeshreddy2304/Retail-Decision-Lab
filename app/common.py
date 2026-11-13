import html
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
FAVORITA_WEEKLY_PATH = ROOT_DIR / "data" / "processed" / "favorita_weekly.parquet"
FAVORITA_REPORT_DIR = ROOT_DIR / "reports" / "favorita_benchmark"
from styles import (
    apply_brand_style,
    inline_section,
    notice,
    render_brand_header,
    section_card,
)


pio.templates["retail_decision_lab"] = go.layout.Template(
    layout={
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "font": {
            "family": '-apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif',
            "color": "#242824",
            "size": 12,
        },
        "colorway": ["#35463a", "#9a4f36", "#8c6a34", "#6d7d70", "#667178"],
        "title": {"font": {"size": 15, "color": "#242824"}, "x": 0.02},
        "margin": {"l": 50, "r": 25, "t": 55, "b": 45},
        "xaxis": {"gridcolor": "#e3e6e1", "linecolor": "#cfd3cd", "zeroline": False},
        "yaxis": {"gridcolor": "#e3e6e1", "linecolor": "#cfd3cd", "zeroline": False},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    }
)
pio.templates.default = "retail_decision_lab"
px.defaults.template = "retail_decision_lab"


@st.cache_data
def load_favorita_weekly():
    df = pd.read_parquet(FAVORITA_WEEKLY_PATH)
    df["week"] = pd.to_datetime(df["week"])
    return df


@st.cache_data
def load_favorita_selection():
    return json.loads((FAVORITA_REPORT_DIR / "selection.json").read_text())


@st.cache_data
def load_favorita_holdout_metrics():
    return pd.read_csv(FAVORITA_REPORT_DIR / "holdout_metrics.csv")


@st.cache_data
def load_favorita_cv_summary():
    return pd.read_csv(FAVORITA_REPORT_DIR / "cv_summary.csv")


@st.cache_data
def load_favorita_fold_metrics():
    return pd.read_csv(FAVORITA_REPORT_DIR / "fold_metrics.csv")


@st.cache_data
def load_favorita_holdout_predictions():
    df = pd.read_csv(FAVORITA_REPORT_DIR / "holdout_predictions.csv.gz", parse_dates=["week"])
    return df


def favorita_filters(df):
    with st.sidebar:
        st.header("Demand filters")
        stores = sorted(df["store_nbr"].dropna().astype(int).unique())
        families = sorted(df["family"].dropna().unique())
        selected_stores = st.multiselect("Stores", stores, default=stores)
        selected_families = st.multiselect("Product families", families, default=families)
        date_range = st.date_input(
            "Week range",
            value=(df["week"].min().date(), df["week"].max().date()),
            min_value=df["week"].min().date(),
            max_value=df["week"].max().date(),
        )
    filtered = df[
        df["store_nbr"].isin(selected_stores) & df["family"].isin(selected_families)
    ].copy()
    if len(date_range) == 2:
        start, end = map(pd.Timestamp, date_range)
        filtered = filtered[(filtered["week"] >= start) & (filtered["week"] <= end)]
    return filtered


def init_page():
    apply_brand_style()
    render_brand_header()


def render_chart(fig, title, *, height=None):
    """Render a Plotly figure with a separate, non-overlapping title row."""
    st.markdown(
        f'<div class="rdl-chart-title">{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )
    layout = {
        "title": None,
        "legend": {
            "title_text": "",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        "margin": {"l": 50, "r": 25, "t": 46, "b": 45},
    }
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    st.plotly_chart(fig, width="stretch")


