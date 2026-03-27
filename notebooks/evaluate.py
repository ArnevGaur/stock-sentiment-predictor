import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'   # macOS M-series fix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

from config import (
    PROCESSED_DATA_PATH, MODEL_PATH, SCALER_PATH,
    SEQ_LENGTH, TEST_SPLIT, METRICS_PATH, OUTPUTS_DIR
)

def prepare_sequences(series, seq_length):
    X, y = [], []
    for i in range(seq_length, len(series)):
        X.append(series[i - seq_length:i])
        y.append(series[i])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # ── Load ──────────────────────────────────────────────────────────────────
    df    = pd.read_csv(PROCESSED_DATA_PATH)
    model = load_model(MODEL_PATH, compile=False)   # compile=False avoids macOS mutex crash

    # ── Load the PERSISTED scaler (never re-fit here — that would be a data leak) ──
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. "
            "Run lstm_model.py first so the scaler is fitted and saved."
        )
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ Scaler loaded from {SCALER_PATH}")

    # ── Scale using the loaded scaler (transform only, no fit) ────────────────
    scaled = scaler.transform(df[['Close']])

    X, y  = prepare_sequences(scaled, SEQ_LENGTH)
    split = int(len(X) * (1 - TEST_SPLIT))
    X_test, y_test = X[split:], y[split:]

    # ── Predict ───────────────────────────────────────────────────────────────
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual      = scaler.inverse_transform(y_test.reshape(-1, 1))

    # ── Metrics ───────────────────────────────────────────────────────────────
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae  = mean_absolute_error(actual, predictions)
    r2   = r2_score(actual, predictions)
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100

    print("\n📊 Model Evaluation Metrics:")
    print(f"   RMSE : {rmse:.2f}")
    print(f"   MAE  : {mae:.2f}")
    print(f"   R²   : {r2:.4f}")
    print(f"   MAPE : {mape:.2f}%")

    # ── Save metrics ──────────────────────────────────────────────────────────
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    pd.DataFrame([{
        'RMSE': round(rmse, 2),
        'MAE':  round(mae,  2),
        'R2':   round(r2,   4),
        'MAPE': round(mape, 2),
    }]).to_csv(METRICS_PATH, index=False)
    print(f"✅ Metrics saved to {METRICS_PATH}")

    # ── Plot: Actual vs Predicted ─────────────────────────────────────────────
    plt.figure(figsize=(14, 5))
    plt.plot(actual,      label='Actual Price',    color='royalblue')
    plt.plot(predictions, label='Predicted Price', color='orange', linestyle='--')
    plt.title('INFY.NS — Actual vs Predicted Stock Price')
    plt.xlabel('Days (test set)')
    plt.ylabel('Price (INR)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}/evaluation_plot.png', dpi=150)
    print(f"✅ Evaluation plot saved to {OUTPUTS_DIR}/evaluation_plot.png")

    # ── Plot: Error Distribution ──────────────────────────────────────────────
    errors = actual.flatten() - predictions.flatten()
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=30, color='steelblue', edgecolor='black')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error (INR)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}/error_distribution.png', dpi=150)
    print(f"✅ Error distribution saved to {OUTPUTS_DIR}/error_distribution.png")