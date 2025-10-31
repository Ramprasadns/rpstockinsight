"""
RP StockInsight v3.0 - Final app.py
- Auto-scrolling ticker (indices + popular stocks)
- Auto-refresh every 60s (streamlit-autorefresh)
- Secure login via st.secrets (Super Admin/Admin/User)
- st.rerun() used instead of deprecated experimental_rerun
- Mobile-friendly layout and responsive charts
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import json
from io import BytesIO
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# Autorefresh helper (install: streamlit-autorefresh)
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st.warning("Optional package 'streamlit-autorefresh' not installed. Auto-refresh disabled. See requirements.")
    def st_autorefresh(interval, limit, key):
        return None

# ---------------------------
# Configuration
# ---------------------------
DEFAULT_REFRESH = 60  # seconds
TICKER_INDICES = ["^NSEI", "^BSESN", "^NSEBANK"]  # NIFTY, SENSEX, BANKNIFTY
POPULAR_STOCKS = ["INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", "SBIN.NS", "TCS.NS", "ICICIBANK.NS", "HINDUNILVR.NS"]

st.set_page_config(page_title="RP StockInsight", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Helper utilities
# ---------------------------
@st.cache_data(ttl=30)
def fetch_ticker_price(symbol):
    """Fetch latest tick minute-resolution price (1d/1m). Returns (price, prev_price) or (None,None)."""
    try:
        df = yf.download(symbol, period="1d", interval="1m", progress=False, threads=False)
        if df is None or df.empty:
            return None, None
        close = df["Close"].dropna()
        if len(close) < 2:
            return None, None
        latest = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        return latest, prev
    except Exception:
        return None, None

@st.cache_data(ttl=300)
def fetch_history(symbol, period="6mo", interval="1d"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        return df
    except Exception:
        return pd.DataFrame()

def safe_fmt(v):
    try:
        return f"{v:.2f}"
    except Exception:
        return str(v)

# ---------------------------
# Authentication (Streamlit Secrets)
# ---------------------------
# Put credentials in Streamlit Cloud -> Settings -> Secrets (see README below)
SECRETS = st.secrets.get("users", {}) if hasattr(st, "secrets") else {}
# SECRETS expected structure:
# users = {"superadmin":{"password":"...","role":"superadmin"}, "admin":{...}, "user1":{...}}

def check_login(username, password):
    if not username or not password:
        return False
    if username in SECRETS:
        return SECRETS[username].get("password") == password
    return False

# Sidebar login
with st.sidebar:
    st.header("🔐 Login")
    in_user = st.text_input("Username", value="")
    in_pass = st.text_input("Password", type="password", value="")
    login_clicked = st.button("Login")

    if login_clicked:
        if check_login(in_user, in_pass):
            st.session_state.user = in_user
            st.session_state.role = SECRETS[in_user].get("role", "user")
            st.success(f"Logged in as {in_user} ({st.session_state.role})")
            # rerun to refresh main page as logged-in
            st.rerun()
        else:
            st.error("Invalid credentials. Update secrets.toml on Streamlit Cloud.")

# Quick display if logged in
role = st.session_state.get("role", None)
user = st.session_state.get("user", None)

# ---------------------------
# Top ticker bar (marquee)
# ---------------------------
# Auto refresh using streamlit-autorefresh (interval in ms)
refresh_ms = DEFAULT_REFRESH * 1000
if "refresh_rate" in st.secrets:
    try:
        refresh_ms = int(st.secrets["refresh_rate"]) * 1000
    except Exception:
        pass

# start auto-refresh; return value is number of reruns (unused)
st_autorefresh(interval=refresh_ms, limit=None, key="ticker_autorefresh")

# Build ticker text list (indices + popular shares)
ticker_list = TICKER_INDICES + POPULAR_STOCKS

ticker_items = []
for sym in ticker_list:
    price, prev = fetch_ticker_price(sym)
    if price is None:
        ticker_items.append(f"{sym}: N/A")
    else:
        delta = price - prev if prev is not None else 0.0
        delta_pct = (delta / prev * 100) if prev not in (None, 0) else 0.0
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else "→"
        color = "green" if delta > 0 else "red" if delta < 0 else "gray"
        ticker_items.append(f"<span style='color:{color}; font-weight:600'>{sym}: {safe_fmt(price)} {arrow} {safe_fmt(delta_pct)}%</span>")

# Marquee HTML (simple and works mobile; Streamlit allows simple HTML in markdown)
marquee_html = f"""
<div style="width:100%; overflow:hidden; white-space:nowrap; background:var(--bg-color, #f8f9fa); padding:6px 0;">
  <div style="display:inline-block; padding-left:100%; animation: marquee 25s linear infinite;">
    {' &nbsp; • &nbsp; '.join(ticker_items)}
  </div>
</div>
<style>
@keyframes marquee {{
  0% {{ transform: translateX(0%); }}
  100% {{ transform: translateX(-100%); }}
}}
/* Adapt marquee speed for small screens */
@media (max-width:600px) {{
  div[style*="animation: marquee"] {{ animation-duration: 35s !important; font-size:0.95rem; }}
}}
</style>
"""
st.markdown(marquee_html, unsafe_allow_html=True)

# ---------------------------
# Main content
# ---------------------------
st.markdown("---")

if not role:
    st.info("Please log in via the left panel to access full features. Default Super Admin credentials can be set in Streamlit Secrets.")
    st.markdown("**Default superadmin (example)** - put in secrets.toml on cloud:\n\n```\n[users]\nsuperadmin = {password = 'rp@2025', role = 'superadmin'}\nadmin = {password = 'admin123', role = 'admin'}\n```")
    st.stop()

# Logged-in content
st.sidebar.markdown(f"**User:** {user}  •  **Role:** {role}")

# Controls Row
col1, col2, col3 = st.columns([3, 2, 1])
with col1:
    st.subheader("🔎 Search & Chart")
    ticker = st.text_input("Symbol (e.g., INFY.NS)", value="INFY.NS")
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=2)
    interval = st.selectbox("Interval", ["1d", "1wk"], index=0)

with col2:
    st.subheader("⚙️ Options")
    show_preds = st.checkbox("Show Predictions", value=True)
    show_breakout = st.checkbox("Show Breakout Detection", value=True)
    download_report = st.button("Download Excel Report")

with col3:
    # refresh indicator
    st.subheader("🔁 Live")
    st.write(f"Refresh every {int(refresh_ms/1000)}s")
    if st.button("Refresh Now"):
        st.experimental_memo.clear()
        st.rerun()

# Fetch and display
if ticker:
    with st.spinner("Fetching data..."):
        df = fetch_history(ticker, period=period, interval=interval)
        if df is None or df.empty:
            st.error("⚠️ No data found. Check symbol (use .NS suffix for NSE).")
        else:
            # Ensure necessary columns exist
            if "Close" not in df.columns:
                st.error("⚠️ Fetched data missing Close column.")
            else:
                df["MA20"] = df["Close"].rolling(20).mean()
                df["MA50"] = df["Close"].rolling(50).mean()

                # Plot with Plotly (responsive)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(width=2)))
                fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20", line=dict(dash="dash")))
                fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50", line=dict(dash="dash")))
                fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=420)
                st.plotly_chart(fig, use_container_width=True)

                # Recent table
                st.subheader("🔍 Recent Data")
                st.dataframe(df.tail(10))

                # Predictions
                if show_preds:
                    st.subheader("🔮 Predictions (simple linear forecast)")
                    try:
                        dfp = df.dropna(subset=["Close"])
                        X = np.arange(len(dfp)).reshape(-1, 1)
                        y = dfp["Close"].values
                        model = LinearRegression().fit(X, y)
                        future_x = np.arange(len(dfp), len(dfp) + 5).reshape(-1, 1)
                        preds = model.predict(future_x)
                        st.write("Next 5 periods forecast:", [safe_fmt(p) for p in preds])
                    except Exception as e:
                        st.warning("⚠️ Prediction error: simple model failed. " + str(e))

                # Breakout detection
                if show_breakout:
                    st.subheader("🚀 Breakout Detection (MA20 vs Close)")
                    try:
                        recent = df.tail(40).dropna()
                        if recent.empty or "MA20" not in recent.columns:
                            st.info("Not enough data for breakout analysis.")
                        else:
                            last = recent.iloc[-1]
                            signal = "No breakout"
                            if last["Close"] > last["MA20"]:
                                signal = "Breakout possible"
                                st.success(f"💥 {signal} — Close {safe_fmt(last['Close'])} > MA20 {safe_fmt(last['MA20'])}")
                            else:
                                st.info(f"{signal} — Close {safe_fmt(last['Close'])} <= MA20 {safe_fmt(last['MA20'])}")
                    except Exception as e:
                        st.warning("⚠️ Error analyzing breakout: " + str(e))

                # Download
                if download_report:
                    towrite = BytesIO()
                    df.to_excel(towrite, sheet_name="stock", index=True)
                    towrite.seek(0)
                    st.download_button("Download Excel", data=towrite.getvalue(), file_name=f"{ticker}_report.xlsx")

# Footer & legal
st.markdown("---")
st.markdown("**RP StockInsight** — Test mode only. Not financial advice. For live deployment ensure you comply with SEBI/regulatory rules before enabling subscriptions or paid services.")
