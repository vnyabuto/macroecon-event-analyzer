📊 Project Summary
Macroeconomic Event Analyzer is a data-driven dashboard application designed to analyze and forecast the impact of key macroeconomic indicators on various stock market sectors represented by Exchange-Traded Funds (ETFs). Built using Streamlit, the application leverages machine learning and time series forecasting techniques to help users visualize and interpret how macroeconomic changes influence market behavior over time.

🎯 Key Features
Macroeconomic Indicator Tracking
The app ingests real-time and historical data on critical macroeconomic indicators such as:

Consumer Price Index (CPI)

Unemployment Rate

Federal Funds Rate

Sector ETF Analysis
Analyzes sector performance using major Exchange-Traded Funds:

XLK – Technology

XLV – Health Care

XLF – Financials

XLE – Energy

XLI – Industrials

XLY – Consumer Discretionary

XLP – Consumer Staples

XLB – Materials

XLU – Utilities

XLRE – Real Estate

Modeling Approaches
Users can choose between two forecasting strategies:

Prophet Forecasting:
Uses Facebook’s Prophet model to forecast ETF prices based on historical trends.

Extreme Gradient Boosting (XGBoost) Prediction:
Trains a classifier to predict directional movements (up or down) of sector ETFs based on macroeconomic features.

Visual Output

Line charts for forecasted sector trends

Feature importance plots for XGBoost model

Sector impact heatmaps

Correlation matrix of macro variables and ETF performance

User Interface Highlights

Intuitive filters for start/end date, sector selection, and model type

Plots powered by Plotly for interactivity

Real-time feedback on prediction accuracy

Ability to export predictions




🔍 streamlit_app/ — Core Application Files
app.py
Main Streamlit app script. Handles:

User interface (date filter, sector & model selection)

Model execution (Prophet or XGBoost)

Visualization rendering (charts, heatmaps, tables)

model_prophet.py
Implements:

Prophet time series forecasting model

Sector-specific predictions

Trend and seasonality visualization

model_xgboost.py
Implements:

XGBoost classification model to predict sector movements (up/down)

Feature importance plotting

Model evaluation (accuracy)

utils.py
Contains helper functions including:

Fetching sector ETF data via yfinance

Downloading macroeconomic indicators (e.g., CPI, Unemployment Rate, Fed Funds Rate) using fredapi

Preprocessing and data cleaning utilities

📦 requirements.txt
A list of Python libraries used in the project. Install using:

bash
Copy
Edit
pip install -r requirements.txt
Key packages:

streamlit

yfinance

prophet

xgboost

scikit-learn

pandas

plotly

fredapi

📘 README.md
This file — provides documentation, setup instructions, model overviews, and usage guidance.