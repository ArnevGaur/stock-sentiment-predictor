import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
import os

# ── Config ────────────────────────────────────────
INPUT      = "data/processed_stock_data.csv"
MODEL_PATH = "models/lstm_model.keras"
SEQ_LENGTH = 60
# ──────────────────────────────────────────────────

def prepare_sequences(series, seq_length):
    X, y = [], []
    for i in range(seq_length, len(series)):
        X.append(series[i-seq_length:i])
        y.append(series[i])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # ── Load data & model ─────────────────────────
    df    = pd.read_csv(INPUT)
    model = load_model(MODEL_PATH)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']])

    X, y  = prepare_sequences(scaled, SEQ_LENGTH)
    split = int(len(X) * 0.8)
    X_test, y_test = X[split:], y[split:]

    # ── Predict ───────────────────────────────────
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual      = scaler.inverse_transform(y_test.reshape(-1, 1))

    # ── Metrics ───────────────────────────────────
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae  = mean_absolute_error(actual, predictions)
    r2   = r2_score(actual, predictions)
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100

    print("📊 Model Evaluation Metrics:")
    print(f"   RMSE : {rmse:.2f}")
    print(f"   MAE  : {mae:.2f}")
    print(f"   R²   : {r2:.4f}")
    print(f"   MAPE : {mape:.2f}%")

    # ── Save metrics to CSV ───────────────────────
    metrics_df = pd.DataFrame([{
        'RMSE': round(rmse, 2),
        'MAE':  round(mae, 2),
        'R2':   round(r2, 4),
        'MAPE': round(mape, 2)
    }])
    metrics_df.to_csv("outputs/metrics.csv", index=False)
    print("✅ Metrics saved to outputs/metrics.csv")

    # ── Plot Actual vs Predicted ──────────────────
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(14, 5))
    plt.plot(actual,      label='Actual Price',    color='blue')
    plt.plot(predictions, label='Predicted Price', color='orange', linestyle='--')
    plt.title('INFY.NS — Actual vs Predicted Stock Price')
    plt.xlabel('Days')
    plt.ylabel('Price (INR)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/evaluation_plot.png')
    print("✅ Evaluation plot saved to outputs/evaluation_plot.png")

    # ── Plot Error Distribution ───────────────────
    errors = actual - predictions
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=30, color='steelblue', edgecolor='black')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error (INR)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('outputs/error_distribution.png')
    print("✅ Error distribution saved to outputs/error_distribution.png")