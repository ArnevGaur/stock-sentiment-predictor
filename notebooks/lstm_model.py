import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# ── Config ────────────────────────────────────────
INPUT      = "data/processed_stock_data.csv"
SEQ_LENGTH = 60   # use last 60 days to predict next day
TEST_SPLIT = 0.2
# ──────────────────────────────────────────────────

def prepare_sequences(series, seq_length):
    X, y = [], []
    for i in range(seq_length, len(series)):
        X.append(series[i-seq_length:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def build_model(seq_length):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model

if __name__ == "__main__":
    # ── Load & scale ──────────────────────────────
    df     = pd.read_csv(INPUT)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']])

    # ── Sequences ─────────────────────────────────
    X, y   = prepare_sequences(scaled, SEQ_LENGTH)
    split  = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"✅ Train: {X_train.shape} | Test: {X_test.shape}")

    # ── Build & Train ─────────────────────────────
    model = build_model(SEQ_LENGTH)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # ── Predict ───────────────────────────────────
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual      = scaler.inverse_transform(y_test.reshape(-1, 1))

    # ── Plot ──────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(14, 5))
    plt.plot(actual,      label='Actual Price',    color='blue')
    plt.plot(predictions, label='Predicted Price', color='orange')
    plt.title('INFY.NS — Actual vs Predicted (LSTM)')
    plt.xlabel('Days')
    plt.ylabel('Price (INR)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/lstm_predictions.png')
    print("✅ Plot saved to outputs/lstm_predictions.png")

    # ── Save model ────────────────────────────────
    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_model.h5")
    print("✅ Model saved to models/lstm_model.h5")

    # ── Loss curve ────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'],     label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/loss_curve.png')
    print("✅ Loss curve saved to outputs/loss_curve.png")