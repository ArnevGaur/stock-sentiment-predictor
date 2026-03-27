import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'   # macOS M-series fix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from config import (
    PROCESSED_DATA_PATH, MODEL_PATH, SCALER_PATH,
    SEQ_LENGTH, TEST_SPLIT, EPOCHS, BATCH_SIZE, PATIENCE,
    OUTPUTS_DIR, MODELS_DIR
)

def prepare_sequences(series, seq_length):
    X, y = [], []
    for i in range(seq_length, len(series)):
        X.append(series[i - seq_length:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def build_model(seq_length):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model

if __name__ == "__main__":
    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # ── Fit and persist scaler on full dataset ────────────────────────────────
    # This is the canonical scaler. evaluate.py and app.py load it — they must
    # NEVER re-fit, because doing so would change the min/max and corrupt
    # inverse_transform back to INR prices.
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']])
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Scaler saved to {SCALER_PATH}")

    # ── Sequences ─────────────────────────────────────────────────────────────
    X, y  = prepare_sequences(scaled, SEQ_LENGTH)
    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"✅ Train: {X_train.shape}  |  Test: {X_test.shape}")

    # ── Build & train ─────────────────────────────────────────────────────────
    model      = build_model(SEQ_LENGTH)
    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
    )

    # ── Predict on test set ───────────────────────────────────────────────────
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual      = scaler.inverse_transform(y_test.reshape(-1, 1))

    # ── Plots ─────────────────────────────────────────────────────────────────
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    plt.figure(figsize=(14, 5))
    plt.plot(actual,      label='Actual Price',    color='royalblue')
    plt.plot(predictions, label='Predicted Price', color='orange')
    plt.title('INFY.NS — Actual vs Predicted (LSTM Test Set)')
    plt.xlabel('Days')
    plt.ylabel('Price (INR)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}/lstm_predictions.png', dpi=150)
    print(f"✅ Prediction plot saved to {OUTPUTS_DIR}/lstm_predictions.png")

    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'],     label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}/loss_curve.png', dpi=150)
    print(f"✅ Loss curve saved to {OUTPUTS_DIR}/loss_curve.png")

    # ── Save model ────────────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")