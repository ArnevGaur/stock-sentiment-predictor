import os
# ── macOS M-series fix — must be set BEFORE any TensorFlow import ─────────────
os.environ['TF_CPP_MIN_LOG_LEVEL']                = '3'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from textblob import TextBlob
import yfinance as yf
from datetime import date, timedelta

from config import MODEL_PATH, SCALER_PATH, SEQ_LENGTH, TICKER, SENTIMENT_PATH

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
)

# ── Model & scaler loader ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    import tensorflow as tf
    model  = tf.keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# ── Stock data fetcher ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.reset_index(inplace=True)

    # Flatten MultiIndex columns (yfinance >= 0.2.x)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.columns = [c.strip() for c in df.columns]
    cols = [c for c in df.columns if c.lower() in ('date', 'close')]
    df   = df[cols].copy()
    df.rename(columns={c: c.title() for c in df.columns}, inplace=True)
    df['Date']  = pd.to_datetime(df['Date'])
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(inplace=True)
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ── Prediction helpers ────────────────────────────────────────────────────────
def run_predictions(df, model, scaler):
    scaled = scaler.transform(df[['Close']])   # transform only — never re-fit
    X = []
    for i in range(SEQ_LENGTH, len(scaled)):
        X.append(scaled[i - SEQ_LENGTH:i])
    X      = np.array(X)
    preds  = model.predict(X, verbose=0)
    preds  = scaler.inverse_transform(preds).flatten()
    actual = df['Close'].values[SEQ_LENGTH:]
    dates  = df['Date'].values[SEQ_LENGTH:]
    return preds, actual, dates

def predict_future(df, model, scaler, n_days=30):
    scaled = scaler.transform(df[['Close']])
    window = scaled[-SEQ_LENGTH:].tolist()
    future_preds = []
    for _ in range(n_days):
        x    = np.array(window[-SEQ_LENGTH:]).reshape(1, SEQ_LENGTH, 1)
        pred = model.predict(x, verbose=0)[0, 0]
        future_preds.append(pred)
        window.append([pred])
    future_prices = scaler.inverse_transform(
        np.array(future_preds).reshape(-1, 1)
    ).flatten()
    last_date    = pd.to_datetime(df['Date'].iloc[-1])
    future_dates = pd.bdate_range(last_date + timedelta(days=1), periods=n_days)
    return future_dates, future_prices

# ── Sentiment helpers ─────────────────────────────────────────────────────────
def score_text(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0.05:    return "🟢 Positive", score
    elif score < -0.05: return "🔴 Negative", score
    else:               return "🟡 Neutral",  score

@st.cache_data(show_spinner=False)
def load_sentiment_history():
    try:
        return pd.read_csv(SENTIMENT_PATH)
    except FileNotFoundError:
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 Stock Price Predictor — INFY.NS")
st.markdown("**LSTM + Sentiment Analysis** | Machine Learning Lab · CSE3231 · MUJ")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    start_date  = st.date_input("Start Date", value=date(2023, 1, 1), min_value=date(2020, 1, 1))
    end_date    = st.date_input("End Date",   value=date.today())
    future_days = st.slider("Future forecast (business days)", 5, 60, 30)
    show_future = st.toggle("Show future forecast", value=True)
    st.divider()
    st.header("🗞️ Sentiment")
    news_input = st.text_area(
        "Enter a news headline:",
        value="Infosys raises FY25 guidance after strong deal wins",
        height=80,
    )
    st.button("Analyze Sentiment", use_container_width=True)

if end_date <= start_date:
    st.error("⚠️ End date must be after start date.")
    st.stop()

# ── Guard: check files exist before loading ───────────────────────────────────
missing = []
if not os.path.exists(MODEL_PATH):  missing.append(f"`{MODEL_PATH}`")
if not os.path.exists(SCALER_PATH): missing.append(f"`{SCALER_PATH}`")
if missing:
    st.error(
        f"Required file(s) not found: {', '.join(missing)}\n\n"
        "Run **`lstm_model.py`** first to train the model and save the scaler."
    )
    st.stop()

# ── Load model + data ─────────────────────────────────────────────────────────
with st.spinner("Loading model..."):
    model, scaler = load_resources()

with st.spinner("Fetching stock data..."):
    fetch_start = start_date - timedelta(days=SEQ_LENGTH * 2)
    df_full     = fetch_data(TICKER, fetch_start, end_date)

if len(df_full) < SEQ_LENGTH + 10:
    st.error("⚠️ Not enough data — select a wider date range.")
    st.stop()

with st.spinner("Running LSTM predictions..."):
    preds, actual, dates = run_predictions(df_full, model, scaler)

# Filter to user's chosen range
mask        = pd.to_datetime(dates) >= pd.to_datetime(start_date)
dates_show  = pd.to_datetime(dates)[mask]
actual_show = actual[mask]
preds_show  = preds[mask]
is_past     = end_date < date.today()

# ── Metrics row ───────────────────────────────────────────────────────────────
st.subheader("📊 Model Performance Metrics")
if len(actual_show) > 1:
    rmse = np.sqrt(mean_squared_error(actual_show, preds_show))
    mae  = mean_absolute_error(actual_show, preds_show)
    r2   = r2_score(actual_show, preds_show)
    mape = np.mean(np.abs((actual_show - preds_show) / actual_show)) * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE", f"₹{rmse:.2f}",  help="Root Mean Squared Error")
    c2.metric("MAE",  f"₹{mae:.2f}",   help="Mean Absolute Error")
    c3.metric("R²",   f"{r2:.4f}",     help="1.0 = perfect. > 0.75 is good.")
    c4.metric("MAPE", f"{mape:.2f}%",  help="Under 10% is good")
else:
    st.info("Widen the date range to compute metrics.")

st.divider()

# ── Main chart ────────────────────────────────────────────────────────────────
st.subheader("📉 Actual vs Predicted Stock Price" if is_past else "📉 Predicted Stock Price")

fig = go.Figure()

if len(actual_show) > 0:
    fig.add_trace(go.Scatter(
        x=dates_show, y=np.round(actual_show, 2),
        name='Actual Price',
        line=dict(color='royalblue', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([dates_show, dates_show[::-1]]),
        y=np.concatenate([actual_show, preds_show[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 100, 100, 0.10)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Difference',
        hoverinfo='skip',
    ))

fig.add_trace(go.Scatter(
    x=dates_show, y=np.round(preds_show, 2),
    name='Predicted Price',
    line=dict(color='orange', width=2, dash='dash'),
))

if show_future:
    with st.spinner(f"Forecasting {future_days} days ahead..."):
        future_dates, future_prices = predict_future(df_full, model, scaler, future_days)
    fig.add_trace(go.Scatter(
        x=future_dates, y=np.round(future_prices, 2),
        name='Future Forecast',
        line=dict(color='mediumseagreen', width=2, dash='dot'),
    ))
    fig.add_vline(
        x=pd.Timestamp(date.today()).timestamp() * 1000,
        line_width=1, line_dash="dash", line_color="gray",
        annotation_text="Today", annotation_position="top right",
    )

fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Price (INR)',
    hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
    height=460,
    margin=dict(l=0, r=0, t=30, b=0),
)
st.plotly_chart(fig, use_container_width=True)

# ── Comparison table ──────────────────────────────────────────────────────────
if is_past and len(dates_show) > 0:
    st.subheader("🔍 Prediction vs Actual — Last 20 Days")
    diff_df = pd.DataFrame({
        'Date':           dates_show,
        'Actual (₹)':    np.round(actual_show, 2),
        'Predicted (₹)': np.round(preds_show,  2),
        'Error (₹)':     np.round(actual_show - preds_show, 2),
        'Error %':        np.round(
            np.abs((actual_show - preds_show) / actual_show) * 100, 2
        ),
    })
    st.dataframe(
        diff_df.tail(20).style.background_gradient(subset=['Error %'], cmap='RdYlGn_r'),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ── Sentiment ─────────────────────────────────────────────────────────────────
st.subheader("🗞️ News Sentiment Analysis")
col_l, col_r = st.columns([3, 2])

with col_l:
    label, score = score_text(news_input)
    st.info(f"**Headline:** {news_input}")
    m1, m2 = st.columns(2)
    m1.metric("Sentiment", label)
    m2.metric("Polarity",  f"{score:.4f}")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 4),
        number=dict(font=dict(size=28)),
        gauge=dict(
            axis=dict(range=[-1, 1], tickwidth=1),
            bar=dict(color="darkblue", thickness=0.25),
            steps=[
                dict(range=[-1.0, -0.05], color='rgba(220,80,80,0.25)'),
                dict(range=[-0.05, 0.05], color='rgba(220,200,80,0.25)'),
                dict(range=[ 0.05, 1.0 ], color='rgba(80,180,80,0.25)'),
            ],
            threshold=dict(
                line=dict(color="navy", width=3),
                thickness=0.8,
                value=score,
            ),
        ),
        title=dict(text="Polarity  (−1 very negative · +1 very positive)", font=dict(size=12)),
    ))
    gauge.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))
    st.plotly_chart(gauge, use_container_width=True)

with col_r:
    st.markdown("**Historical headlines**")
    hist = load_sentiment_history()
    if not hist.empty:
        for _, row in hist.iterrows():
            lbl, sc = score_text(row['headline'])
            st.caption(f"{row['date']}  ·  {lbl}  `{sc:.2f}`")
            st.markdown(f"<small>{row['headline']}</small>", unsafe_allow_html=True)
            st.divider()
    else:
        st.info("Run `sentiment.py` to populate headlines.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "INFY.NS · LSTM 64×2 + TextBlob sentiment · "
    "Trained Jan 2020 – Dec 2024 · "
    "CSE3231 ML Lab · Manipal University Jaipur"
)