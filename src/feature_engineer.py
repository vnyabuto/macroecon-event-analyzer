import pandas as pd
from fetch_fred import fetch_fred_data
from fetch_yahoo import fetch_all_etfs

def prepare_features(start_date="2015-01-01", macro=None):
    # Load data
    macros = fetch_fred_data(start_date)
    market = fetch_all_etfs(start_date)


    # Resample market data to monthly returns
    monthly_returns = market.resample("M").last().pct_change().dropna()
    monthly_returns.index = monthly_returns.index.to_period("M")

    # Allign macro to monthly periods too
    macro.index = macro.index.to_period("M")

    # Merge datasets
    df = macro.merge(monthly_returns, left_index=True, right_index=True)

    return df

if __name__ == "__main__":
    df = prepare_features()
    print(df.tall())

