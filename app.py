import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import json
import os
import time
import datetime

# ------------------------------
# Basic Config
# ------------------------------
st.set_page_config(page_title="RPStockInsight", layout="wide", initial_sidebar_state="expanded")

# Auto-refresh every 60 seconds
st_autorefresh = st.experimental_rerun if not hasattr(st, "autorefresh") else st.autorefresh
st_autorefresh(interval=60000, key="data_refresh")

# ------------------------------
# Load Admin Info (local only)
# ------------------------------
USER_FILE = "users.json"
if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({}, f)

with open(USER_FILE, "r") as f:
    users = json.load(f)

# ------------------------------
# Header
# ------------------------------
st.markdown("""
# 📊 RPStockInsight  
**Mobile-friendly · Auto light/dark theme · Live auto-refresh (every 60s)**  
""")

# ------------------------------
# Sidebar Controls
# ------------------------------
st.sidebar.header("Dashboard Settings")

symbol = st.sidebar.text_input("Enter Stock Symbol (NSE/BSE) e.g. TCS.NS or INFY.NS", "TCS.NS")
period = st.sidebar.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

# ------------------------------
# Admin Login Section
# ------------------------------
st.sidebar.subheader("Admin area (local test/auth)")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

admin_choice = st.sidebar.radio("Admin action", ["Login", "Register"], horizontal=True)

if admin_choice == "Register":
    new_user = st.sidebar.text_input("Set Admin Username")
    new_pass = st.sidebar.text_input("Set Password", type="password")
    if st.sidebar.button("Register"):
        users[new_user] = new_pass
        with open(USER_FILE, "w") as f:
            json.dump(users, f)
        st.sidebar.success("✅ Admin registered successfully!")

elif admin_choice == "Login":
    user = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if users.get(user) == password:
            st.session_state.logged_in = True
            st.sidebar.success("✅ Logged in successfully!")
        else:
            st.sidebar.error("❌ Invalid credentials")

# ------------------------------
# Live Market Index Ticker
# ------------------------------
def pinned_ticker_html(index_label="NIFTY 50", index_symbol="^NSEI"):
    try:
        ticker = yf.Ticker(index_symbol)
        data = ticker.history(period="1d", interval="1m")

        if data.empty:
            return f"<span style='color:gray;'>No data for {index_label}</span>"

        price = float(data['Close'][-1])
        prev = float(data['Close'][-2])
        pct = ((price - prev) / prev * 100) if prev != 0 else 0.0

        color = "green" if pct >= 0 else "red"
        arrow = "▲" if pct >= 0 else "▼"

        return f"<b>{index_label}</b>: {price:,.2f} <span style='color:{color};'>{arrow} {pct:.2f}%</span>"
    except Exception as e:
        return f"<span style='color:red;'>⚠️ Error fetching data: {e}</span>"

# ------------------------------
# Display Ticker (auto-refresh)
# ------------------------------
ticker_placeholder = st.empty()
index_html = pinned_ticker_html("NIFTY", "^NSEI")
ticker_placeholder.markdown(f"<div style='font-size:18px;text-align:center;padding:6px;'>{index_html}</div>", unsafe_allow_html=True)

# ------------------------------
# Fetch Stock Data
# ------------------------------
try:
    df = yf.download(symbol, period=period, interval=interval)
    if df.empty:
        st.warning("⚠️ No data found. Please check the stock symbol or period.")
    else:
        st.subheader(f"📈 {symbol} Stock Chart")
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                             open=df['Open'], high=df['High'],
                                             low=df['Low'], close=df['Close'])])
        fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=450)
        st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"❌ Error fetching stock data: {e}")

# ------------------------------
# Short-Term & Long-Term Predictions
# ------------------------------
st.header("🔮 Stock Predictions")

try:
    df = df.dropna()
    df["Days"] = np.arange(len(df))
    X = df[["Days"]]
    y = df["Close"]

    model = LinearRegression()
    model.fit(X, y)

    next_30 = model.predict([[len(df) + 30]])[0]
    st.success(f"Predicted price in 30 days ≈ ₹{next_30:,.2f}")
except Exception as e:
    st.warning(f"⚠️ Prediction error: {e}")

# ------------------------------
# Breakout Detection
# ------------------------------
st.header("🚀 Breakout Detection")
try:
    recent = df["Close"].tail(10)
    avg = recent.mean()
    last = recent.iloc[-1]

    if last > 1.05 * avg:
        st.success("🚀 Potential Breakout Detected! Price above recent average.")
    elif last < 0.95 * avg:
        st.error("📉 Possible Breakdown. Price below recent average.")
    else:
        st.info("📊 Stable trend. No breakout yet.")
except Exception as e:
    st.error(f"⚠️ Error analyzing breakout: {e}")

# ------------------------------
# Sector-wise Insights (placeholder)
# ------------------------------
st.header("🧭 Sector Insights")
st.info("Sector-wise analysis coming soon... (Beta)")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown(f"""
<div style='text-align:center; font-size:14px;'>
© 2025 RPStockInsight · Built with ❤️ using Streamlit  
<br>Auto-refreshed at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
</div>
""", unsafe_allow_html=True)
