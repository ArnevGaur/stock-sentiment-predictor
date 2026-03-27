# ── config.py — single source of truth ───────────────────────────────────────
# All scripts import from here. Never define these constants anywhere else.

TICKER     = "INFY.NS"
START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"

SEQ_LENGTH = 60          # days of history used per prediction window
TEST_SPLIT = 0.2         # fraction of sequences held out for testing
EPOCHS     = 50
BATCH_SIZE = 32
PATIENCE   = 10          # EarlyStopping patience (epochs without val_loss improvement)

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATA_PATH       = "data/raw_stock_data.csv"
PROCESSED_DATA_PATH = "data/processed_stock_data.csv"
SENTIMENT_PATH      = "data/sentiment_scores.csv"
MODEL_PATH          = "models/lstm_model.h5"
SCALER_PATH         = "models/scaler.pkl"   # persisted scaler — eliminates re-fit data leak
METRICS_PATH        = "outputs/metrics.csv"
OUTPUTS_DIR         = "outputs"
MODELS_DIR          = "models"
DATA_DIR            = "data"