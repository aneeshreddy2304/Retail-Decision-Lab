import streamlit as st


st.set_page_config(page_title="Retail Decision Lab", layout="wide")

navigation = st.navigation(
    {
        "Monitor": [
            st.Page("pages/0_Forecast_Monitor.py", title="Forecast monitor", default=True),
            st.Page("pages/1_Executive_Overview.py", title="Executive overview"),
        ],
        "Explore": [
            st.Page("pages/2_Demand_Explorer.py", title="Demand explorer"),
            st.Page("pages/3_Forecasting_Lab.py", title="Forecast evaluation"),
            st.Page("pages/5_Promotion_Intelligence.py", title="Promotion intelligence"),
        ],
        "Validate": [
            st.Page("pages/9_Logistic_Classification.py", title="Demand-spike classification"),
            st.Page("pages/7_SQL_Analytics.py", title="SQL analytics"),
        ],
        "Operate": [
            st.Page("pages/4_Client_Upload_Forecast.py", title="Upload and forecast"),
            st.Page("pages/6_Methodology.py", title="Methodology"),
        ],
    },
    position="sidebar",
)

navigation.run()
