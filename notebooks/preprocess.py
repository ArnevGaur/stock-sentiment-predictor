import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, SCALER_PATH, OUTPUTS_DIR, MODELS_DIR

def load_data(path):
    df = pd.read_csv(path, header=0)

    # Handle the extra ticker row yfinance sometimes writes (e.g. "INFY.NS, INFY.NS ...")
    if df.iloc[0].astype(str).str.contains(r'^[A-Z]+\.[A-Z]+$|^[A-Z]+$', regex=True).any():
        df = df.iloc[1:].reset_index(drop=True)

    # Normalise column names — yfinance can emit different orderings
    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for col in df.columns:
        lc = col.lower()
        if lc == 'date':        rename[col] = 'Date'
        elif lc == 'close':     rename[col] = 'Close'
        elif lc == 'open':      rename[col] = 'Open'
        elif lc == 'high':      rename[col] = 'High'
        elif lc == 'low':       rename[col] = 'Low'
        elif lc == 'volume':    rename[col] = 'Volume'
    df.rename(columns=rename, inplace=True)

    df['Date']  = pd.to_datetime(df['Date'])
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def check_missing(df):
    print("🔍 Missing values:")
    print(df.isnull().sum())
    df.dropna(subset=['Close'], inplace=True)
    return df

def add_features(df):
    df['MA_7']         = df['Close'].rolling(window=7).mean()
    df['MA_21']        = df['Close'].rolling(window=21).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    return df

def scale_and_save(df):
    """
    Fit the scaler on the FULL dataset and save it.
    evaluate.py and app.py must LOAD this scaler — never re-fit it —
    so that inverse_transform produces correct INR prices.
    """
    scaler = MinMaxScaler()
    df['Close_Scaled'] = scaler.fit_transform(df[['Close']])

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Scaler saved to {SCALER_PATH}")
    return df, scaler

def plot_closing_price(df):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    plt.figure(figsize=(14, 5))
    plt.plot(df['Date'], df['Close'], label='Close Price',   color='royalblue')
    plt.plot(df['Date'], df['MA_7'],  label='7-Day MA',      color='orange',  linestyle='--')
    plt.plot(df['Date'], df['MA_21'], label='21-Day MA',     color='green',   linestyle='--')
    plt.title('INFY.NS — Closing Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}/closing_price.png', dpi=150)
    print(f"✅ Plot saved to {OUTPUTS_DIR}/closing_price.png")

if __name__ == "__main__":
    df = load_data(RAW_DATA_PATH)
    df = check_missing(df)
    df = add_features(df)
    df, scaler = scale_and_save(df)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"✅ Processed data saved: {len(df)} rows → {PROCESSED_DATA_PATH}")
    print(df[['Date', 'Close', 'MA_7', 'MA_21', 'Daily_Return', 'Close_Scaled']].head())
    plot_closing_price(df)