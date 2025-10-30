import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from sklearn.linear_model import LinearRegression
from datetime import datetime

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(
    page_title="RPStockInsight",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    /* Mobile responsive tweaks */
    @media (max-width: 768px) {
        h1 { font-size: 26px !important; }
        .ticker { font-size: 14px !important; }
        .block-container { padding-top: 0.5rem !important; }
    }
    /* Smooth ticker animation */
    .ticker {
        font-weight: bold;
        font-size: 18px;
        text-align: center;
        animation: fadeIn 1.5s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Header
# -----------------------------
st.markdown("<h1 style='text-align:center;'>📊 RPStockInsight</h1>", unsafe_allow_html=True)
st.caption("Mobile-friendly · Auto light/dark theme · Pinned live ticker (auto-refresh every 60 s)")

# -----------------------------
# Live Ticker (NIFTY + SENSEX)
# -----------------------------
def get_index_ticker(symbol):
    try:
        data = yf.download(symbol, period="1d", interval="1m", progress=False)
        if data.empty:
            return None
        last = data['Close'].iloc[-1]
        prev = data['Close'].iloc[-2]
        pct = ((last - prev) / prev) * 100
        arrow = "▲" if pct >= 0 else "▼"
        color = "#00C853" if pct >= 0 else "#D50000"
        return f"<span style='color:{color};'>{symbol}: {last:,.2f} {arrow} {pct:.2f}%</span>"
    except Exception:
        return None

nifty = get_index_ticker("^NSEI")
sensex = get_index_ticker("^BSESN")
ticker_html = f"<div class='ticker'>{nifty or ''} &nbsp;&nbsp; {sensex or ''}</div>"
st.markdown(ticker_html, unsafe_allow_html=True)

# -----------------------------
# Stock Selection
# -----------------------------
st.divider()
symbol = st.text_input("Enter NSE/BSE stock symbol (e.g., RELIANCE.NS, TCS.NS):", "RELIANCE.NS")
period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)
interval = st.selectbox("Interval", ["1d", "1wk"], index=0)

if st.button("🔍 Analyze"):
    with st.spinner("Fetching stock data..."):
        try:
            df = yf.download(symbol, period=period, interval=interval)
            if df.empty:
                st.error("⚠️ No data found. Please check the stock symbol or try another.")
            else:
                df['MA20'] = df['Close'].rolling(20).mean()
                df['MA50'] = df['Close'].rolling(50).mean()

                # Chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Candlestick'
                ))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="MA20", line=dict(color="blue", width=1)))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name="MA50", line=dict(color="orange", width=1)))
                fig.update_layout(height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                # Short-term vs Long-term Prediction
                st.subheader("📈 Stock Predictions")
                df['Days'] = np.arange(len(df))
                X = df[['Days']]
                y = df['Close']

                model = LinearRegression()
                model.fit(X, y)
                future_days = np.arange(len(df), len(df) + 30).reshape(-1, 1)
                future_preds = model.predict(future_days)
                next_price = float(future_preds[-1])

                st.success(f"Predicted price in 30 days ≈ ₹{next_price:,.2f}")

                # Sector insights placeholder
                st.subheader("🧭 Sector Insights")
                st.info("Sector-wise analysis coming soon... (Beta)")

                # Breakout detection
                st.subheader("🚀 Breakout Detection")
                recent_close = df['Close'].iloc[-1]
                recent_ma20 = df['MA20'].iloc[-1]
                recent_ma50 = df['MA50'].iloc[-1]
                if recent_close > recent_ma20 > recent_ma50:
                    st.success("Potential bullish breakout detected!")
                elif recent_close < recent_ma20 < recent_ma50:
                    st.error("Potential bearish breakout detected!")
                else:
                    st.info("No clear breakout pattern currently.")
        except Exception as e:
            st.error(f"⚠️ Error fetching data: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
col1, col2 = st.columns([2, 1])
col1.caption("© 2025 RPStockInsight · Built with ❤️ using Streamlit")
col2.caption(datetime.now().strftime("Updated on %Y-%m-%d %H:%M:%S"))
