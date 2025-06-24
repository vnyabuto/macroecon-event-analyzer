import pandas as pd
import yfinance as yf


def load_sector_data(start="2019-01-01"):
    tickers = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE"]
    try:
        df = yf.download(tickers, start=start, group_by='ticker', auto_adjust=True)

        # Extract Close prices for each sector
        adj_close = pd.DataFrame()
        for ticker in tickers:
            if (ticker in df.columns.get_level_values(0)) and ("Close" in df[ticker].columns):
                adj_close[ticker] = df[ticker]["Close"]
            else:
                print(f"Warning: '{ticker}' data not available or missing 'Close'.")

        adj_close.dropna(how='all', inplace=True)
        return adj_close

    except Exception as e:
        print("Error loading sector data:", e)
        return pd.DataFrame()
