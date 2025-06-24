import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

def plot_prophet_forecast(forecast, model):
    """
    Plots Prophet forecast using its built-in plot function.
    Returns the matplotlib figure so it can be shown in Streamlit.
    """
    fig = model.plot(forecast)
    plt.title("Prophet Forecast")
    plt.tight_layout()
    return fig

def plot_sector_heatmap(df):
    """
    Plots a correlation heatmap between sectors and macro indicators.
    Returns the matplotlib figure so it can be used in Streamlit.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap: Macroeconomic Indicators vs Sector Performance")
    plt.tight_layout()
    return fig
