import html
import json
from pathlib import Path

import streamlit as st

FAVORITA_SELECTION_PATH = (
    Path(__file__).resolve().parents[1] / "reports" / "favorita_benchmark" / "selection.json"
)


def apply_brand_style():
    st.markdown(
        """
        <style>
        :root {
            --rdl-bg: #f7f7f5;
            --rdl-surface: #ffffff;
            --rdl-sidebar: #eceeea;
            --rdl-ink: #242824;
            --rdl-muted: #626962;
            --rdl-line: #cfd3cd;
            --rdl-green: #35463a;
            --rdl-amber: #8c6a34;
            --rdl-amber-soft: #f0ebe1;
        }

        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
        }

        .stApp { background: var(--rdl-bg); color: var(--rdl-ink); }

        header[data-testid="stHeader"] {
            background: var(--rdl-bg);
            border-bottom: 1px solid var(--rdl-line);
        }

        .block-container {
            max-width: 1360px;
            padding-top: 1.15rem;
            padding-bottom: 3rem;
        }

        section[data-testid="stSidebar"] {
            width: 244px !important;
            min-width: 244px !important;
            background: var(--rdl-sidebar);
            border-right: 1px solid var(--rdl-line);
        }

        section[data-testid="stSidebar"] > div:first-child { width: 244px !important; }

        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p { color: var(--rdl-ink) !important; }

        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
            min-height: 42px;
            border-radius: 0;
            border-left: 3px solid transparent;
            color: var(--rdl-muted);
            font-size: 0.82rem;
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
            background: #e2e6e0;
            color: var(--rdl-ink);
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: #e2e6e0;
            border-left-color: #526b59;
            color: var(--rdl-ink);
            font-weight: 650;
        }

        .rdl-system-bar {
            display: grid;
            grid-template-columns: 1fr auto auto;
            align-items: center;
            gap: 1.6rem;
            min-height: 36px;
            padding: 0 0.95rem;
            margin: -1.15rem -1rem 1.25rem;
            background: #202621;
            color: #e5e9e5;
            font-size: 0.7rem;
            letter-spacing: 0.035em;
        }

        .rdl-system-bar strong { font-weight: 650; }
        .rdl-system-bar strong span { margin-left: 0.45rem; color: #8ca184; }
        .rdl-system-bar .rdl-health { color: #bdd0b6; }

        .rdl-page-head {
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 2rem;
            padding: 0.15rem 0 1rem;
            border-bottom: 1px solid var(--rdl-line);
            margin-bottom: 1.05rem;
        }

        .rdl-page-kicker {
            margin-bottom: 0.25rem;
            color: var(--rdl-muted);
            font-size: 0.64rem;
            font-weight: 650;
            letter-spacing: 0.13em;
            text-transform: uppercase;
        }

        .rdl-page-title {
            margin: 0;
            color: var(--rdl-ink);
            font-size: clamp(1.45rem, 2.4vw, 2rem);
            font-weight: 560;
            letter-spacing: -0.025em;
            line-height: 1.1;
        }

        .rdl-page-copy {
            max-width: 620px;
            margin: 0;
            color: var(--rdl-muted);
            font-size: 0.82rem;
            line-height: 1.55;
            text-align: right;
        }

        .rdl-notice {
            display: grid;
            grid-template-columns: auto 1fr auto;
            align-items: center;
            gap: 0.8rem;
            margin: 0.75rem 0 1rem;
            padding: 0.72rem 0.9rem;
            border-left: 3px solid var(--rdl-amber);
            background: var(--rdl-amber-soft);
            color: var(--rdl-ink);
            font-size: 0.76rem;
        }

        .rdl-notice span { color: var(--rdl-muted); }
        .rdl-notice a { color: var(--rdl-ink); text-decoration: underline; }

        .rdl-section-card {
            padding: 0.9rem 0;
            border-top: 1px solid var(--rdl-line);
            border-bottom: 1px solid var(--rdl-line);
            margin: 0.8rem 0 1rem;
        }

        .rdl-section-title { color: var(--rdl-ink); font-size: 0.9rem; font-weight: 650; }

        .rdl-section-copy {
            max-width: 860px;
            margin-top: 0.25rem;
            color: var(--rdl-muted);
            font-size: 0.78rem;
            line-height: 1.55;
        }

        .rdl-chart-title {
            margin: 1.15rem 0 0.2rem;
            color: var(--rdl-ink);
            font-size: 0.95rem;
            font-weight: 650;
            letter-spacing: -0.01em;
            line-height: 1.35;
        }

        div[data-testid="stMetric"] {
            min-height: 104px;
            padding: 0.8rem 0.9rem;
            border: 1px solid var(--rdl-line);
            border-radius: 0;
            background: #ffffff;
            box-shadow: none;
        }

        div[data-testid="stMetric"] label {
            color: var(--rdl-muted) !important;
            font-size: 0.73rem !important;
            font-weight: 500 !important;
        }

        div[data-testid="stMetricValue"] {
            color: var(--rdl-ink) !important;
            font-size: 1.35rem !important;
            font-weight: 540 !important;
            letter-spacing: -0.025em;
        }

        div[data-testid="stMetricDelta"] { font-size: 0.7rem !important; }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--rdl-line) !important;
            border-radius: 0 !important;
            background: #ffffff;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            border-bottom: 1px solid var(--rdl-line);
            background: transparent;
        }

        .stTabs [data-baseweb="tab"] {
            min-height: 42px;
            padding: 0.55rem 0.95rem;
            border-radius: 0;
            color: var(--rdl-muted);
            font-size: 0.78rem;
            font-weight: 500;
        }

        .stTabs [aria-selected="true"] {
            color: var(--rdl-ink) !important;
            background: transparent !important;
            border-bottom: 2px solid var(--rdl-green) !important;
        }

        div[data-testid="stDataFrame"] {
            overflow: hidden;
            border: 1px solid var(--rdl-line);
            border-radius: 0;
            box-shadow: none;
        }

        .stButton button, .stDownloadButton button {
            min-height: 40px;
            border: 0 !important;
            border-radius: 0 !important;
            background: var(--rdl-green) !important;
            color: #ffffff !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            box-shadow: none !important;
        }

        .stButton button:hover, .stDownloadButton button:hover { background: #26342b !important; }

        div[data-baseweb="select"] > div,
        .stDateInput div[data-baseweb="input"] > div,
        .stTextInput div[data-baseweb="input"] > div { border-radius: 0 !important; }

        .stFileUploader {
            padding: 0.7rem;
            border: 1px dashed #aeb5ad;
            border-radius: 0;
            background: #f1f3ef;
        }

        .stAlert, details { border-radius: 0 !important; }
        details { border-color: var(--rdl-line) !important; }

        h1, h2, h3 { color: var(--rdl-ink); font-weight: 560; letter-spacing: -0.02em; }
        h2 { font-size: 1.15rem; }
        h3 { font-size: 0.95rem; }
        p, li { color: var(--rdl-ink); line-height: 1.55; }
        hr { border-color: var(--rdl-line); }

        [data-testid="stPlotlyChart"] {
            border: 1px solid var(--rdl-line);
            background: #ffffff;
        }

        @media (max-width: 760px) {
            .block-container { padding-left: 1rem; padding-right: 1rem; }
            .rdl-system-bar { grid-template-columns: 1fr auto; }
            .rdl-system-bar > span:first-of-type { display: none; }
            .rdl-page-head { display: block; }
            .rdl-page-copy { margin-top: 0.65rem; text-align: left; }
            .rdl-notice { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_header():
    best_model = "Model pending"
    if FAVORITA_SELECTION_PATH.exists():
        best_model = str(json.loads(FAVORITA_SELECTION_PATH.read_text())["selected_model"])

    st.markdown(
        f"""
        <div class="rdl-system-bar">
          <strong>RDL<span>/ Retail Decision Lab</span></strong>
          <span>Selected model: {html.escape(best_model)}</span>
          <span class="rdl-health">Pipeline healthy</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_card(title, copy):
    st.markdown(
        f"""
        <div class="rdl-page-head">
          <div>
            <div class="rdl-page-kicker">Retail Decision Lab</div>
            <h1 class="rdl-page-title">{html.escape(title)}</h1>
          </div>
          <p class="rdl-page-copy">{html.escape(copy)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inline_section(title, copy):
    st.markdown(
        f"""
        <div class="rdl-section-card">
          <div class="rdl-section-title">{html.escape(title)}</div>
          <div class="rdl-section-copy">{html.escape(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def notice(title, detail, action=None):
    action_html = f"<a>{html.escape(action)}</a>" if action else ""
    st.markdown(
        f"""
        <div class="rdl-notice">
          <strong>{html.escape(title)}</strong>
          <span>{html.escape(detail)}</span>
          {action_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
