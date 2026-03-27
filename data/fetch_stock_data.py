import yfinance as yf
import pandas as pd
import os
from config import TICKER, START_DATE, END_DATE, RAW_DATA_PATH, DATA_DIR

def fetch_stock_data(ticker, start, end):
    print(f"📡 Fetching data for {ticker} ({start} → {end})...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.reset_index(inplace=True)

    # Flatten MultiIndex columns produced by newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    print(f"   Downloaded {len(df)} rows, columns: {list(df.columns)}")
    return df

if __name__ == "__main__":
    df = fetch_stock_data(TICKER, START_DATE, END_DATE)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"✅ Saved {len(df)} rows to {RAW_DATA_PATH}")
    print(df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].head())