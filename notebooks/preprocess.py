import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# ── Config ────────────────────────────────────────
INPUT  = "data/raw_stock_data.csv"
OUTPUT = "data/processed_stock_data.csv"
# ──────────────────────────────────────────────────

def load_data(path):
    # Skip second row (ticker row: INFY.NS, INFY.NS ...)
    df = pd.read_csv(path, header=0, skiprows=[1])
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def check_missing(df):
    print("🔍 Missing values:")
    print(df.isnull().sum())
    df.dropna(inplace=True)
    return df

def add_features(df):
    # Moving averages
    df['MA_7']  = df['Close'].rolling(window=7).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()
    # Daily return
    df['Daily_Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    return df

def scale_data(df):
    scaler = MinMaxScaler()
    df['Close_Scaled'] = scaler.fit_transform(df[['Close']])
    return df, scaler

def plot_closing_price(df):
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(14, 5))
    plt.plot(df['Date'], df['Close'], label='Close Price')
    plt.plot(df['Date'], df['MA_7'],  label='7-Day MA', linestyle='--')
    plt.plot(df['Date'], df['MA_21'], label='21-Day MA', linestyle='--')
    plt.title('INFY.NS Closing Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/closing_price.png')
    print("✅ Plot saved to outputs/closing_price.png")

if __name__ == "__main__":
    df = load_data(INPUT)
    df = check_missing(df)
    df = add_features(df)
    df, scaler = scale_data(df)
    df.to_csv(OUTPUT, index=False)
    print(f"✅ Processed data saved: {len(df)} rows")
    print(df[['Date','Close','MA_7','MA_21','Daily_Return','Close_Scaled']].head())
    plot_closing_price(df)