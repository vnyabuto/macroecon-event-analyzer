import yfinance as yf
import pandas as pd

# Dictionary of sector ETFs and their symbols
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE"
}


def fetch_sector_etf(etf_symbol, start_date="2015-01-01"):
    print(f"Fetching {etf_symbol}...")

    try:
        df = yf.download(etf_symbol, start=start_date, progress=False)

        if df.empty:
            print(f"[!] No data returned for {etf_symbol}")
            return None

        print(f"[i] Columns returned for {etf_symbol}: {df.columns.tolist()}")

        # Try to fetch 'Adj Close', fallback to 'Close'
        if "Adj Close" in df.columns:
            df = df[["Adj Close"]].rename(columns={"Adj Close": etf_symbol})
        elif "Close" in df.columns:
            print(f"[!] Using 'Close' as fallback for {etf_symbol}")
            df = df[["Close"]].rename(columns={"Close": etf_symbol})
        else:
            print(f"[!] Neither 'Adj Close' nor 'Close' found for {etf_symbol}")
            return None

        return df

    except Exception as e:
        print(f"[!] Error fetching {etf_symbol}: {e}")
        return None


def fetch_all_etfs(start_date="2015-01-01"):
    etf_data = []

    for sector, symbol in SECTOR_ETFS.items():
        df = fetch_sector_etf(symbol, start_date)
        if df is not None:
            etf_data.append(df)

    if not etf_data:
        raise ValueError("No ETF data could be fetched.")

    # Combine all into a single DataFrame on Date
    merged_df = pd.concat(etf_data, axis=1)
    merged_df.dropna(inplace=True)

    return merged_df


if __name__ == "__main__":
    df = fetch_all_etfs("2015-01-01")
    print("\n[✓] Combined ETF Data:")
    print(df.head())


