import yfinance as yf
import pandas as pd
import os

# ── Config ────────────────────────────────────────
TICKER = "INFY.NS"       # Infosys (change to TCS.NS, RELIANCE.NS etc.)
START  = "2020-01-01"
END    = "2024-12-31"
OUTPUT = "data/raw_stock_data.csv"
# ──────────────────────────────────────────────────

def fetch_stock_data(ticker, start, end):
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    return df

if __name__ == "__main__":
    df = fetch_stock_data(TICKER, START, END)
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"✅ Saved {len(df)} rows to {OUTPUT}")
    print(df.head())