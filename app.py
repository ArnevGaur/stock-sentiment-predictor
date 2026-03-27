import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from textblob import TextBlob
import yfinance as yf
from datetime import date, timedelta

# ── Page Config ───────────────────────────────────
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# ── Constants ─────────────────────────────────────
MODEL_PATH = "models/lstm_model.keras"
SEQ_LENGTH = 90
TICKER     = "INFY.NS"

# ── Load Model ────────────────────────────────────
@st.cache_resource
def load_lstm_model():
    return load_model(MODEL_PATH)

# ── Fetch Data ────────────────────────────────────
@st.cache_data
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.reset_index(inplace=True)
    df.columns = ['Date','Close','High','Low','Open','Volume'] if len(df.columns) == 6 else df.columns
    # flatten multi-index if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip().split('_')[0] for col in df.columns]
    df = df[['Date','Close']].dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# ── Predict ───────────────────────────────────────
def predict(df, model):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']])
    X = []
    for i in range(SEQ_LENGTH, len(scaled)):
        X.append(scaled[i-SEQ_LENGTH:i])
    X = np.array(X)
    preds = model.predict(X, verbose=0)
    preds = scaler.inverse_transform(preds)
    return preds.flatten(), df['Close'].values[SEQ_LENGTH:], df['Date'].values[SEQ_LENGTH:]

# ── Sentiment ─────────────────────────────────────
SAMPLE_HEADLINES = {
    "Positive": "Infosys raises revenue guidance on strong deal wins and beats estimates",
    "Negative": "Infosys warns of slowdown due to weak client demand and global uncertainty",
    "Neutral":  "Infosys reports quarterly earnings in line with market expectations"
}

def get_sentiment(text):
    blob = TextBlob(text)
    score = blob.sentiment.polarity
    if score > 0.1:   return "🟢 Positive", score
    elif score < -0.1: return "🔴 Negative", score
    else:              return "🟡 Neutral",  score

# ── UI ────────────────────────────────────────────
st.title("📈 Stock Price Predictor — INFY.NS")
st.markdown("**LSTM + Sentiment Analysis** | Machine Learning Lab Project")
st.divider()

# ── Sidebar ───────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    start_date = st.date_input("Start Date", value=date(2023, 1, 1), min_value=date(2020, 1, 1))
    end_date   = st.date_input("End Date",   value=date.today())
    st.divider()
    st.header("🗞️ Sentiment Analysis")
    news_input = st.text_area(
        "Enter a news headline:",
        value="Infosys raises FY25 guidance after strong deal wins"
    )
    analyze_btn = st.button("Analyze Sentiment", use_container_width=True)

if end_date <= start_date:
    st.error("⚠️ End date must be after start date!")
    st.stop()

# ── Load & Predict ────────────────────────────────
with st.spinner("Fetching stock data and running predictions..."):
    model = load_lstm_model()
    df    = fetch_data(TICKER, start_date - timedelta(days=SEQ_LENGTH * 2), end_date)

    if len(df) < SEQ_LENGTH + 10:
        st.error("⚠️ Not enough data! Please select a wider date range.")
        st.stop()

    predictions, actual, dates = predict(df, model)

# ── Filter to selected range ──────────────────────
mask        = (pd.to_datetime(dates) >= pd.to_datetime(start_date))
dates_show  = pd.to_datetime(dates)[mask]
actual_show = actual[mask]
preds_show  = predictions[mask]

is_past = end_date < date.today()

# ── Metrics Row ───────────────────────────────────
st.subheader("📊 Model Performance Metrics")
col1, col2, col3, col4 = st.columns(4)

if len(actual_show) > 0:
    rmse = np.sqrt(mean_squared_error(actual_show, preds_show))
    mae  = mean_absolute_error(actual_show, preds_show)
    r2   = r2_score(actual_show, preds_show)
    mape = np.mean(np.abs((actual_show - preds_show) / actual_show)) * 100

    col1.metric("RMSE",  f"₹{rmse:.2f}",  help="Root Mean Squared Error")
    col2.metric("MAE",   f"₹{mae:.2f}",   help="Mean Absolute Error")
    col3.metric("R²",    f"{r2:.4f}",     help="Coefficient of Determination")
    col4.metric("MAPE",  f"{mape:.2f}%",  help="Mean Absolute Percentage Error")

st.divider()

# ── Chart ─────────────────────────────────────────
st.subheader("📉 Actual vs Predicted Stock Price" if is_past else "📉 Predicted Stock Price")

fig = go.Figure()

if is_past or end_date == date.today():
    fig.add_trace(go.Scatter(
        x=dates_show, y=actual_show,
        name='Actual Price',
        line=dict(color='royalblue', width=2)
    ))
    # Difference shading
    fig.add_trace(go.Scatter(
        x=np.concatenate([dates_show, dates_show[::-1]]),
        y=np.concatenate([actual_show, preds_show[::-1]]),
        fill='toself',
        fillcolor='rgba(255,100,100,0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Difference'
    ))

fig.add_trace(go.Scatter(
    x=dates_show, y=preds_show,
    name='Predicted Price',
    line=dict(color='orange', width=2, dash='dash')
))

fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Price (INR)',
    hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
    height=450
)
st.plotly_chart(fig, use_container_width=True)

# ── Difference Table ──────────────────────────────
if is_past and len(dates_show) > 0:
    st.subheader("🔍 Prediction vs Actual Comparison")
    diff_df = pd.DataFrame({
        'Date':      dates_show,
        'Actual (₹)':    np.round(actual_show, 2),
        'Predicted (₹)': np.round(preds_show, 2),
        'Difference (₹)': np.round(actual_show - preds_show, 2),
        'Error %':   np.round(np.abs((actual_show - preds_show) / actual_show) * 100, 2)
    })
    st.dataframe(
        diff_df.tail(20).style.background_gradient(subset=['Error %'], cmap='RdYlGn_r'),
        use_container_width=True
    )

st.divider()

# ── Sentiment ─────────────────────────────────────
st.subheader("🗞️ News Sentiment Analysis")
col1, col2 = st.columns([2, 1])

with col1:
    if analyze_btn or True:
        label, score = get_sentiment(news_input)
        st.info(f"**Headline:** {news_input}")
        st.markdown(f"**Sentiment:** {label}")
        sent_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            gauge={
                'axis': {'range': [-1, 1]},
                'bar':  {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.1], 'color': 'rgba(255,0,0,0.3)'},
                    {'range': [-0.1, 0.1],'color': 'rgba(255,255,0,0.3)'},
                    {'range': [0.1, 1],   'color': 'rgba(0,255,0,0.3)'},
                ],
            },
            title={'text': "Polarity Score"}
        ))
        sent_fig.update_layout(height=250)
        st.plotly_chart(sent_fig, use_container_width=True)

with col2:
    st.markdown("**Sample Headlines:**")
    for tone, headline in SAMPLE_HEADLINES.items():
        lbl, sc = get_sentiment(headline)
        st.markdown(f"{lbl} `{sc:.2f}`")
        st.caption(headline)
        st.divider()