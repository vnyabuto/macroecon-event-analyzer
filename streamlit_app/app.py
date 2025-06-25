import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import datetime

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_prophet import train_prophet_model
from src.model_xgboost import predict_sector_movements, get_feature_importance
from src.fetch_fred import fetch_series
import yfinance as yf

# --- HELPER FUNCTIONS ---
def load_fred_data(start_date="2010-01-01") -> pd.DataFrame:
    series_names = ["CPI", "Unemployment Rate", "Fed Funds Rate"]
    all_dfs = [fetch_series(name, start_date=start_date) for name in series_names]
    merged_df = pd.concat(all_dfs, axis=1).dropna()
    return merged_df


def load_sector_data(start="2019-01-01"):
    tickers = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE"]
    try:
        df = yf.download(tickers, start=start, group_by='ticker', auto_adjust=True)

        # Extract Adjusted Close prices
        adj_close = pd.DataFrame()
        for ticker in tickers:
            if (ticker in df.columns.get_level_values(0)) and ("Close" in df[ticker].columns):
                adj_close[ticker] = df[ticker]["Close"]
            else:
                print(f"Warning: '{ticker}' data not available or missing 'Close'.")

        adj_close.dropna(how='all', inplace=True)

        # ✅ Convert index to datetime to fix filtering issues
        adj_close.index = pd.to_datetime(adj_close.index)

        return adj_close

    except Exception as e:
        print("Error loading sector data:", e)
        return pd.DataFrame()

def forecast_sector_trend(df, sector):
    sector_df = df[[sector]].dropna()
    model, forecast = train_prophet_model(sector_df, target_column=sector)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
    fig.add_trace(go.Scatter(x=sector_df.index, y=sector_df[sector], name='Actual'))
    fig.update_layout(title=f"{sector} Forecast with Prophet", xaxis_title="Date", yaxis_title="Price")
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
end_date = st.sidebar.date_input("End Date", datetime.date.today())

sectors = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE"]
selected_sectors = st.sidebar.multiselect("Select Sectors (ETFs)", sectors, default=sectors[:4])
model_type = st.sidebar.radio("Model Type", ["Prophet Forecast", "XGBoost Prediction"])

# --- LOAD DATA ---
fred_df = load_fred_data()
sector_df = load_sector_data()

# --- FILTER ---
fred_df = fred_df[(fred_df.index >= pd.to_datetime(start_date)) & (fred_df.index <= pd.to_datetime(end_date))]
sector_df = sector_df[(sector_df.index >= pd.to_datetime(start_date)) & (sector_df.index <= pd.to_datetime(end_date))]

# --- KPIs ---
st.subheader("📌 Key Macroeconomic Indicators")
kpi1, kpi2, kpi3 = st.columns(3)

latest_date = fred_df.index[-1]
latest_data = fred_df.loc[latest_date]

kpi1.metric("CPI", f"{latest_data['CPI']:.2f}")
kpi2.metric("Unemployment Rate", f"{latest_data['Unemployment Rate']}%")
kpi3.metric("Fed Funds Rate", f"{latest_data['Fed Funds Rate']:.2f}%")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📈 Forecast & Prediction", "🔥 Sector Impact Heatmap", "🔗 Correlation Analysis"])

with tab1:
    st.subheader(f"Model Results ({model_type})")
    for sector in selected_sectors:
        st.markdown(f"### {sector} Sector")

        if model_type == "Prophet Forecast":
            fig = forecast_sector_trend(sector_df, sector)
            st.plotly_chart(fig, use_container_width=True)

        elif model_type == "XGBoost Prediction":
            # Combine macro and sector data
            combined_df = pd.concat([sector_df[[sector]], fred_df], axis=1).dropna()

            # Run XGBoost prediction
            accuracy, predictions, model = predict_sector_movements(combined_df, sector)

            # Show results
            st.subheader(f"{sector} Sector – XGBoost Prediction Accuracy: {accuracy:.2%}")
            st.line_chart(predictions)

            # Feature importance plot
            fig_importance = get_feature_importance(model, combined_df.drop(columns=[sector, "Target"],
                                                                            errors="ignore").columns.tolist())
            st.plotly_chart(fig_importance, use_container_width=True, key=f"importance_chart_{sector}")

            # Combine predictions with date index and optionally the actual target
            results_df = pd.DataFrame({
                "Date": predictions.index,
                "Predicted Direction": predictions.values,
            })
            results_df["Predicted Direction"] = results_df["Predicted Direction"].map({1: "Up", 0: "Down"})
            results_df.set_index("Date", inplace=True)

            # Display results in a table
            st.subheader("📋 Prediction Results")
            st.dataframe(results_df)

            # Provide download button
            csv = results_df.to_csv().encode("utf-8")
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=csv,
                file_name=f"{sector}_xgboost_predictions.csv",
                mime="text/csv",
            )

with tab2:
    st.subheader("Heatmap: Macroeconomic Event Impact by Sector")
    heat_df = sector_df[selected_sectors].pct_change().rolling(3).mean().dropna()
    heat_corr = heat_df.corrwith(fred_df["Fed Funds Rate"].pct_change().rolling(3).mean().dropna(), method='pearson')

    heatmap_fig = go.Figure(data=go.Heatmap(
        z=[heat_corr.values],
        x=selected_sectors,
        y=["Fed Funds Rate"],
        colorscale="RdBu",
        zmin=-1,
        zmax=1
    ))
    st.plotly_chart(heatmap_fig, use_container_width=True)

with tab3:
    st.subheader("Correlation Matrix")
    combined = pd.concat([fred_df, sector_df[selected_sectors]], axis=1).dropna()
    corr = combined.corr().round(2)
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1)
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")
st.caption("Made with ❤️ by Victor | Powered by Streamlit, Plotly, Prophet, and XGBoost")
