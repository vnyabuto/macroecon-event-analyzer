from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go

def train_prophet_model(df, target_column):
    """
    Trains a Prophet model on the specified target_column of the input DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'Date' as index and target_column as the series.
        target_column (str): The name of the column to forecast.

    Returns:
        model (Prophet): Trained Prophet model.
        forecast (pd.DataFrame): Forecast dataframe.
    """
    prophet_df = df[[target_column]].reset_index().rename(columns={"Date": "ds", target_column: "y"})

    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=30, freq='D')
    forecast = model.predict(future)

    return model, forecast

def forecast_sector_trend(df, sector_name, periods=30):
    """
    Wrapper to train and plot Prophet forecast for a given sector.

    Args:
        df (pd.DataFrame): DataFrame with 'Date' index and sector columns.
        sector_name (str): Sector column name to forecast.
        periods (int): Forecasting horizon in days.

    Returns:
        fig (plotly.graph_objects.Figure): Forecast visualization.
        forecast (pd.DataFrame): Forecast dataframe.
    """
    model, forecast = train_prophet_model(df, sector_name)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower', line=dict(dash='dot')))
    fig.update_layout(title=f"Forecast for {sector_name}", xaxis_title="Date", yaxis_title="Price")

    return fig, forecast
