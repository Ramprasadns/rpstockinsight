import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import json
import os
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Page Config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="RPStockInsight", layout="wide")

# -----------------------------------------------------------------------------
# Admin Setup (Local only)
# -----------------------------------------------------------------------------
USER_FILE = "users.json"
if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({}, f)

with open(USER_FILE, "r") as f:
    users = json.load(f)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown("""
# 📊 RPStockInsight  
**Mobile-friendly · Auto light/dark theme · Live ticker (auto-refresh every 60s)**
""")

# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.header("Dashboard Settings")

symbol = st.sidebar.text_input("Enter Stock Symbol (NSE/BSE) e.g. TCS.NS or INFY.NS", "TCS.NS")
period = st.sidebar.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

# -----------------------------------------------------------------------------
# Admin Login
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Auto-refresh every 60s
# -----------------------------------------------------------------------------
st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Live Marquee Ticker (scrolling banner)
# -----------------------------------------------------------------------------
def get_index_ticker(label, symbol):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d", interval="1m")
        if data.empty:
            return f"{label}: No data"
        price = data["Close"].iloc[-1]
        prev = data["Close"].iloc[-2] if len(data) > 1 else price
        pct = ((price - prev) / prev) * 100 if prev != 0 else 0
        arrow = "▲" if pct >= 0 else "▼"
        color = "limegreen" if pct >= 0 else "tomato"
        return f"<b>{label}</b>: {price:,.2f} <span style='color:{color}'>{arrow} {pct:.2f}%</span>"
    except:
        return f"{label}: Error"

# Combine multiple tickers
indexes = [
    ("NIFTY 50", "^NSEI"),
    ("SENSEX", "^BSESN"),
    ("BANK NIFTY", "^NSEBANK")
]

ticker_html = " | ".join([get_index_ticker(name, sym) for name, sym in indexes])

st.markdown(f"""
<style>
.marquee {{
  white-space: nowrap;
  overflow: hidden;
  box-sizing: border-box;
  animation: marquee 25s linear infinite;
  font-size: 18px;
  font-weight: 500;
  color: white;
  padding: 10px;
  background-color: #111;
  border-radius: 5px;
}}
@keyframes marquee {{
  0%   {{ text-indent: 100% }}
  100% {{ text-indent: -100% }}
}}
</style>
<div class="marquee">{ticker_html}</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Fetch Stock Data
# -----------------------------------------------------------------------------
try:
    df = yf.download(symbol, period=period, interval=interval)
    if df.empty:
        st.warning("⚠️ No data found. Please check the symbol or period.")
    else:
        st.subheader(f"📈 {symbol} Stock Chart")
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                             open=df['Open'],
                                             high=df['High'],
                                             low=df['Low'],
                                             close=df['Close'])])
        fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"❌ Error fetching stock data: {e}")

# -----------------------------------------------------------------------------
# Stock Predictions
# -----------------------------------------------------------------------------
st.header("🔮 Stock Predictions")

try:
    df = df.dropna()
    df["Days"] = np.arange(len(df))
    X = df[["Days"]]
    y = df["Close"]

    model = LinearRegression()
    model.fit(X, y)
    next_30 = model.predict([[len(df) + 30]])[0]
    next_90 = model.predict([[len(df) + 90]])[0]

    st.success(f"📆 Predicted price in 30 days ≈ ₹{next_30:,.2f}")
    st.info(f"🧭 Predicted price in 90 days ≈ ₹{next_90:,.2f}")
except Exception as e:
    st.warning(f"⚠️ Prediction error: {e}")

# -----------------------------------------------------------------------------
# Breakout Detection
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Sector Insights (Placeholder)
# -----------------------------------------------------------------------------
st.header("🧭 Sector Insights")
st.info("Sector-wise analysis coming soon… (Beta)")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(f"""
<div style='text-align:center;font-size:14px;'>
© 2025 RPStockInsight · Built with ❤️ using Streamlit  
<br>Updated on {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
</div>
""", unsafe_allow_html=True)
