import os
from dotenv import load_dotenv
from fredapi import Fred
import pandas as pd

# Load API key from .env
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

# Instantiate Fred API
fred = Fred(api_key=FRED_API_KEY)

# FRED series mapping
FRED_SERIES = {
    "CPI": "CPIAUCSL",               # Consumer Price Index
    "Unemployment Rate": "UNRATE",  # Unemployment Rate
    "Fed Funds Rate": "FEDFUNDS"    # Federal Funds Rate
}

def fetch_series(series_name: str, start_date="2010-01-01") -> pd.DataFrame:
    """
    Fetch a single FRED time series as a DataFrame.
    """
    if series_name not in FRED_SERIES:
        raise ValueError(f"Series '{series_name}' not recognized.")

    data = fred.get_series(FRED_SERIES[series_name], start_date)
    df = data.to_frame(name=series_name)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df

def fetch_fred_data(start_date="2010-01-01") -> pd.DataFrame:
    """
    Fetch all configured FRED time series and merge them.
    """
    all_series = []
    for name in FRED_SERIES:
        df = fetch_series(name, start_date)
        all_series.append(df)

    merged_df = pd.concat(all_series, axis=1)
    merged_df.dropna(inplace=True)
    return merged_df
