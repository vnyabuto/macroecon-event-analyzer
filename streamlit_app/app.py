print("[app.py] v1.0 loaded")

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import datetime

# Fix import path so src/ can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_prophet import train_prophet_model
from src.model_xgboost import predict_sector_movements, get_feature_importance
from src.fetch_fred import fetch_series
import yfinance as yf

# --- HELPER FUNCTIONS ---
def load_fred_data(start_date="2010-01-01") -> pd.DataFrame:
    series_names = ["CPI", "Unemployment Rate", "Fed Funds Rate"]
    dfs = [fetch_series(name, start_date=start_date) for name in series_names]
    return pd.concat(dfs, axis=1).dropna()

def load_sector_data(start="2019-01-01") -> pd.DataFrame:
    tickers = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE"]
    try:
        raw = yf.download(tickers, start=start, group_by='ticker', auto_adjust=True)
        df = pd.DataFrame({t: raw[t]["Close"] for t in tickers if t in raw})
        df.index = pd.to_datetime(df.index)
        return df.dropna(how='all')
    except Exception as e:
        st.error(f"Error loading sector data: {e}")
        return pd.DataFrame()

def forecast_sector_trend(df: pd.DataFrame, sector: str):
    sector_df = df[[sector]].dropna()
    model, forecast = train_prophet_model(sector_df, target_column=sector)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
    fig.add_trace(go.Scatter(x=sector_df.index, y=sector_df[sector], name='Actual'))
    fig.update_layout(
        title=f"{sector} Forecast with Prophet",
        xaxis_title="Date",
        yaxis_title="Price"
    )
    return fig

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Macroeconomic Event Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Macroeconomic Event Analyzer")
st.markdown("""
Analyze how macroeconomic events such as CPI, Unemployment, and Fed Rate changes influence various stock market sectors using Machine Learning and Time Series Forecasting.
""")
with st.expander("📘 What does this app do?"):
    st.markdown("""
    Welcome to the **Macroeconomic Event Analyzer**! 🎯

    This interactive tool is built to help you understand how **macroeconomic events** — such as changes in inflation, unemployment, or interest rates — impact different **stock market sectors** (like Technology, Energy, or Finance).

    ### 🧠 Core Features

    🔹 **Sector ETF Analysis**  
    Track how various sectors represented by **SPDR ETFs** (e.g., XLK for Tech, XLE for Energy) respond to macroeconomic changes over time.

    🔹 **Macroeconomic Data Integration**  
    We pull key economic indicators like:
    - **CPI (Consumer Price Index)** – Inflation trends 📈  
    - **Unemployment Rate** – Labor market health 👷‍♀️  
    - **Federal Funds Rate** – Interest rate policy by the Fed 🏦

    🔹 **Model Choices**  
    Choose from two models depending on your goal:
    - 📊 **Prophet Forecast** – Predicts **future price trends** using Facebook Prophet (great for trend forecasting).
    - ⚡ **XGBoost Prediction** – Uses machine learning to classify **upward or downward movements** based on historical macro data.

    ### ✅ How to Use

    1. **Select Date Range**: Pick a start and end date to define the period of analysis.
    2. **Choose Sector(s)**: Select one or more ETFs to analyze.
    3. **Pick a Model**:
        - Use **Prophet** if you want to forecast future prices.
        - Use **XGBoost** if you want to see predicted movement direction.
    4. **Explore the Results**:
        - 📈 Visual forecasts or predictions
        - 🔥 Sector impact heatmaps
        - 🧮 Correlation and feature importance analysis
        - 💾 Export predictions to CSV

    ### 💡 Who is this for?

    This tool is ideal for:
    - Students learning finance or data science 📚
    - Retail investors and market watchers 💼
    - Analysts researching macro-sector relationships 📉
    - Curious minds exploring economics + AI 🤖

    **No coding required!** Just make your selections and the app will do the rest.

    """)


# --- SIDEBAR ---
st.sidebar.header("🔍 Filter Data")
start_date = st.sidebar.date_input("Start Date", datetime.date(2019, 1, 1))
end_date   = st.sidebar.date_input("End Date", datetime.date.today())

sectors = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE"]
selected_sectors = st.sidebar.multiselect("Select Sectors (ETFs)", sectors, default=sectors[:4])
model_type = st.sidebar.radio("Model Type", ["Prophet Forecast", "XGBoost Prediction"])

# --- LOAD & FILTER DATA ---
fred_df   = load_fred_data()
sector_df = load_sector_data()

fred_df   = fred_df.loc[start_date:end_date]
sector_df = sector_df.loc[start_date:end_date]

# --- KPI SECTION ---
st.subheader("📌 Key Macroeconomic Indicators")
k1, k2, k3 = st.columns(3)
latest = fred_df.iloc[-1]
k1.metric("CPI", f"{latest['CPI']:.2f}")
k2.metric("Unemployment Rate", f"{latest['Unemployment Rate']:.2f}%")
k3.metric("Fed Funds Rate", f"{latest['Fed Funds Rate']:.2f}%")

# --- TABS ---
tab1, tab2, tab3 = st.tabs([
    "📈 Forecast & Prediction",
    "🔥 Sector Impact Heatmap",
    "🔗 Correlation Analysis"
])

with tab1:
    st.subheader(f"Model Results ({model_type})")
    for sector in selected_sectors:
        st.markdown(f"### {sector} Sector")

        if model_type == "Prophet Forecast":
            try:
                fig = forecast_sector_trend(sector_df, sector)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error forecasting {sector}: {e}")
        else:  # XGBoost Prediction
            combined = pd.concat([sector_df[[sector]], fred_df], axis=1).dropna()
            accuracy, predictions, model = predict_sector_movements(combined, sector)

            if accuracy is None:
                st.warning(f"⚠️ Insufficient data or error for {sector}, skipping XGBoost.")
                continue

            st.subheader(f"{sector} – XGBoost Accuracy: {accuracy:.2%}")
            st.line_chart(predictions)

            # Feature importance
            features = combined.drop(columns=[sector]).columns.tolist()
            fig_imp = get_feature_importance(model, features)
            st.plotly_chart(fig_imp, use_container_width=True)

            # Results table + download
            results = pd.DataFrame({
                "Date":   predictions.index,
                "Prediction": predictions.map({1: "Up", 0: "Down"})
            }).set_index("Date")
            st.dataframe(results)
            csv = results.to_csv().encode("utf-8")
            st.download_button(
                "Download Predictions CSV",
                csv,
                file_name=f"{sector}_predictions.csv",
                mime="text/csv"
            )

with tab2:
    st.subheader("Heatmap: Macro Impact by Sector")
    pct = sector_df[selected_sectors].pct_change().rolling(3).mean().dropna()
    corr = pct.corrwith(
        fred_df["Fed Funds Rate"].pct_change().rolling(3).mean().dropna(),
        method='pearson'
    )
    heat = go.Figure(go.Heatmap(
        z=[corr.values], x=selected_sectors, y=["Fed Funds Rate"],
        colorscale="RdBu", zmin=-1, zmax=1
    ))
    st.plotly_chart(heat, use_container_width=True)

with tab3:
    st.subheader("Correlation Matrix")
    merged = pd.concat([fred_df, sector_df[selected_sectors]], axis=1).dropna()
    mat = merged.corr().round(2)
    fig = px.imshow(mat, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Made with ❤️ by Victor | Streamlit · Plotly · Prophet · XGBoost")
