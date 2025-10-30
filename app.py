# app.py - RPStockInsight (ready-to-replace)
# - Admin login (local users.json, SHA256 hashed)
# - Pinned live ticker (auto-refresh every 60s)
# - Mobile-friendly / auto theme
# - Simple short/long prediction, breakout detection, sector placeholder
# - Safe handling of missing yfinance columns

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import json
import os
import hashlib
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import requests

# -----------------------
# Helper & config
# -----------------------

st.set_page_config(page_title="RPStockInsight", layout="wide", initial_sidebar_state="expanded")

USERS_FILE = "users.json"
AUTO_REFRESH_SECONDS = 60  # page auto-refresh interval

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def load_users():
    if not os.path.exists(USERS_FILE):
        return {"admins": []}
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"admins": []}

def save_users(data):
    with open(USERS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def register_admin(username, password):
    users = load_users()
    hashed = hash_password(password)
    # avoid duplicates
    for a in users.get("admins", []):
        if a["username"] == username:
            return False, "Username already exists."
    users.setdefault("admins", []).append({"username": username, "password": hashed})
    save_users(users)
    return True, "Admin created."

def verify_admin(username, password):
    users = load_users()
    hashed = hash_password(password)
    for a in users.get("admins", []):
        if a["username"] == username and a["password"] == hashed:
            return True
    return False

def safe_fetch(symbol, period="3mo", interval="1d"):
    """Fetch stock data and normalize columns - return DataFrame or None"""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty:
            return None
        # Ensure consistent column names: prefer 'Close', fallback to 'Adj Close'
        if "Close" not in df.columns and "adj close" in [c.lower() for c in df.columns]:
            # rename any lowercase adj close
            df.columns = [c.capitalize() for c in df.columns]
        # Keep Date as index - convert to column
        df = df.reset_index()
        # unify common variants
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        if "Close" not in df.columns:
            # try other candidates
            possible = [c for c in df.columns if c.lower().strip() in ("close", "adj close", "adjclose")]
            if possible:
                df["Close"] = df[possible[0]]
            else:
                return None
        # Volume and others
        if "Volume" not in df.columns:
            volcols = [c for c in df.columns if c.lower().startswith("volume")]
            if volcols:
                df["Volume"] = df[volcols[0]]
            else:
                df["Volume"] = 0
        return df
    except Exception as e:
        # don't crash app; return None
        return None

def add_moving_averages(df):
    if df is None or df.empty:
        return df
    # ensure Close exists and numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=1).mean()
    return df

def simple_prediction_linear(df, days_out=30):
    """Very simple demonstration: linear regression on last N days closes"""
    if df is None or df.empty or len(df) < 5:
        return None
    try:
        df = df.dropna(subset=["Close"])
        # use last 30 points at most
        n = min(len(df), 60)
        X = np.arange(n).reshape(-1,1)
        y = df["Close"].tail(n).values.reshape(-1,1)
        model = LinearRegression()
        model.fit(X, y)
        future_x = np.arange(n, n+days_out).reshape(-1,1)
        pred = model.predict(future_x)
        return float(pred[-1][0])
    except Exception:
        return None

def breakout_detection(df):
    """Detect basic breakout: latest close > max(highs last N days) & volume spike"""
    if df is None or df.empty or len(df) < 5:
        return False, "Insufficient data for breakout detection"
    try:
        # Use 'High' column if available
        if "High" not in df.columns:
            return False, "No High column available"
        # compute last 20-day high excluding latest day
        recent = df.tail(21).copy()
        if len(recent) < 2:
            return False, "Insufficient recent records"
        prev = recent.iloc[:-1]
        latest = recent.iloc[-1]
        prev_high = prev["High"].max()
        # volume spike check
        avg_vol = prev["Volume"].mean() if "Volume" in prev.columns else 0
        vol_spike = False
        if avg_vol > 0 and latest["Volume"] > avg_vol * 1.5:
            vol_spike = True
        is_breakout = (latest["Close"] > prev_high) and vol_spike
        if is_breakout:
            return True, f"Breakout! Close {latest['Close']:.2f} > previous high {prev_high:.2f} with volume spike."
        else:
            return False, "No breakout detected."
    except Exception as e:
        return False, "Error detecting breakout: " + str(e)

# -----------------------
# Top pinned ticker (CSS + meta refresh)
# -----------------------

def pinned_ticker_html(index_label="NIFTY", index_symbol="^NSEI"):
    # simple attempt to fetch index price via yfinance
    safe = safe_fetch(index_symbol, period="5d", interval="1d")
    if safe is not None and not safe.empty:
        latest = safe.iloc[-1]
        price = latest["Close"]
        # compute percent change if prev exists
        prev = safe.iloc[-2]["Close"] if len(safe) >= 2 else price
        pct = ((price - prev) / prev * 100) if prev != 0 else 0.0
        arrow = "▲" if pct >= 0 else "▼"
        pct_str = f"{pct:+.2f}%"
        html = f"""
        <div id="rp-ticker" style="position:fixed;top:0;left:0;right:0;z-index:9999;
             background:linear-gradient(90deg,#ffffffcc,#f0f8ffcc);backdrop-filter: blur(4px);
             border-bottom:1px solid rgba(0,0,0,0.08);padding:6px 12px;font-weight:600;">
            <span style="font-size:14px">{index_label}:</span>
            <span style="font-size:16px;margin-left:8px">{price:.2f}</span>
            <span style="color:{'green' if pct>=0 else 'red'};margin-left:8px">{arrow} {pct_str}</span>
        </div>
        """
        return html
    else:
        # fallback bar
        return f"""
        <div id="rp-ticker" style="position:fixed;top:0;left:0;right:0;z-index:9999;
             background:#f5f5f5;padding:6px 12px;border-bottom:1px solid rgba(0,0,0,0.06);">
           <span style="font-weight:600">Index:</span> <span style="margin-left:8px">N/A</span>
        </div>
        """

# -----------------------
# Sidebar (settings + admin)
# -----------------------

with st.sidebar:
    st.title("Dashboard Settings")
    symbol = st.text_input("Enter Stock Symbol (NSE/BSE) e.g. TCS.NS or INFY.NS", value="TCS.NS")
    period = st.selectbox("Select Period", ["1mo","3mo","6mo","1y","2y"], index=1)
    interval = st.selectbox("Interval", ["1d","1wk","1mo"], index=0)
    st.markdown("---")
    st.caption("Admin area (local test/auth)")
    st.session_state.setdefault("admin_logged_in", False)
    users = load_users()
    if not users.get("admins"):
        st.warning("No admin found. Please register an admin account below.")
    # admin register or login
    admin_action = st.radio("Admin action", ["Login","Register"] if not st.session_state["admin_logged_in"] else ["Logout"])
    if admin_action == "Register":
        with st.form("register_form"):
            ru = st.text_input("Username")
            rp = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Create Admin")
            if submitted:
                ok, msg = register_admin(ru, rp)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
    elif admin_action == "Login":
        with st.form("login_form"):
            lu = st.text_input("Username")
            lp = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if verify_admin(lu, lp):
                    st.success("Logged in as admin.")
                    st.session_state["admin_logged_in"] = True
                    st.session_state["admin_user"] = lu
                else:
                    st.error("Invalid credentials.")
    else:  # Logout
        if st.button("Logout Admin"):
            st.session_state["admin_logged_in"] = False
            st.experimental_rerun()

    st.markdown("---")
    st.write("Auto-refresh: page reloads every 60s for live ticker.")
    st.caption("For production, replace local auth with secure provider.")

# -----------------------
# Page-level header & tickers
# -----------------------

# Add meta refresh to reload the entire page every AUTO_REFRESH_SECONDS
# (Simple & robust; reloads all content — good for live ticker & demo)
meta_refresh = f'<meta http-equiv="refresh" content="{AUTO_REFRESH_SECONDS}">'
st.markdown(meta_refresh, unsafe_allow_html=True)

# pinned ticker (top)
index_html = pinned_ticker_html(index_label="NIFTY", index_symbol="^NSEI")
st.markdown(index_html, unsafe_allow_html=True)

# add padding to top of page content so ticker doesn't overlap
st.markdown("<div style='height:44px'></div>", unsafe_allow_html=True)

# small header
st.title("RPStockInsight")
st.write("Mobile-friendly · Auto light/dark theme · Pinned live ticker (auto-refresh every 60s)")

# -----------------------
# Main layout
# -----------------------

# Left: inputs & quick actions; Right: main report
container = st.container()
col1, col2 = container.columns([1, 2])

with col1:
    st.header("Controls")
    st.write(f"Selected symbol: **{symbol}**")
    if st.button("Fetch Data / Generate Report"):
        st.session_state["do_fetch"] = True
    # Admin quick actions
    if st.session_state.get("admin_logged_in"):
        st.markdown("**Admin Tools**")
        if st.button("Open Admin Panel"):
            st.session_state["open_admin_panel"] = True
        if st.button("Clear users.json (delete admins)"):
            try:
                save_users({"admins":[]})
                st.success("Cleared admin list. Re-register if needed.")
            except Exception as e:
                st.error("Could not clear: " + str(e))

with col2:
    st.header(f"Stock Summary for {symbol}")
    # Fetch data (either when clicking button or on load)
    do_fetch = st.session_state.get("do_fetch", True)

    df = safe_fetch(symbol, period=period, interval=interval) if do_fetch else None
    if df is None or df.empty:
        st.warning("⚠️ No data found. Please check the stock symbol or try another (NSE/BSE format: TCS.NS).")
    else:
        df = add_moving_averages(df)
        latest = df.iloc[-1]
        avg_close = df["Close"].mean()
        highest = df["Close"].max()
        lowest = df["Close"].min()
        st.metric("Latest Close", f"{latest['Close']:.2f}")
        # summary row
        c1, c2, c3, c4 = st.columns(4)
        c1.subheader("Average Close")
        c1.write(f"{avg_close:.2f}")
        c2.subheader("Highest Close")
        c2.write(f"{highest:.2f}")
        c3.subheader("Lowest Close")
        c3.write(f"{lowest:.2f}")
        c4.subheader("Period")
        c4.write(period)

    # Recent data table
    if df is not None and not df.empty:
        st.subheader("Recent Data")
        st.dataframe(df[['Date','Open','Close','High','Low','Volume','MA20','MA50']].tail(20), use_container_width=True)

# -----------------------
# Lower content: charts, predictions, breakout, sectors
# -----------------------
st.markdown("---")
cA, cB = st.columns([2,1])

with cA:
    st.subheader("Price Trend")
    if df is None or df.empty:
        st.info("No chart to show (no data).")
    else:
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(df["Date"], df["Close"], label="Close")
        if "MA20" in df.columns:
            ax.plot(df["Date"], df["MA20"], label="MA20", linestyle="--")
        if "MA50" in df.columns:
            ax.plot(df["Date"], df["MA50"], label="MA50", linestyle=":")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.xticks(rotation=25)
        st.pyplot(fig)

with cB:
    st.subheader("Predictions")
    if df is None or df.empty:
        st.info("No predictions (no data).")
    else:
        pred_30 = simple_prediction_linear(df, days_out=30)
        pred_90 = simple_prediction_linear(df, days_out=90)
        if pred_30:
            st.markdown(f"**Predicted price in 30 days ≈ ₹{pred_30:,.2f}**")
        else:
            st.info("Short-term prediction not available.")
        if pred_90:
            st.markdown(f"**Predicted price in 90 days ≈ ₹{pred_90:,.2f}**")
        else:
            st.info("Long-term prediction not available.")

st.markdown("---")
st.subheader("Breakout Detection")

if df is None or df.empty:
    st.info("No breakout detection (no data).")
else:
    is_bo, bo_msg = breakout_detection(df)
    if is_bo:
        st.success(bo_msg)
    else:
        st.warning(bo_msg)

st.markdown("---")
st.subheader("Sector Insights (Beta)")
st.info("Sector-wise analysis coming soon — placeholder for sector-level aggregation & comparison.")

# -----------------------
# Admin panel (below main)
# -----------------------
if st.session_state.get("admin_logged_in"):
    st.markdown("---")
    st.subheader("Admin Panel")
    st.write(f"Logged in as **{st.session_state.get('admin_user')}**")
    # admin-only quick features
    if st.button("Export latest data to CSV"):
        if df is None or df.empty:
            st.error("No data to export.")
        else:
            fname = f"{symbol.replace('.','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(fname, index=False)
            st.success(f"Exported {fname}")
            st.markdown(f"[Download {fname}](./{fname})")
    # Demo user management
    users = load_users()
    st.write("Registered admins:")
    st.write([a["username"] for a in users.get("admins", [])])

# -----------------------
# Footer & mobile tweaks
# -----------------------
st.markdown("""<hr style="opacity:0.1">""", unsafe_allow_html=True)
st.caption(f"© 2025 RPStockInsight · Built with ❤️ using Streamlit — Updated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# mobile-friendly CSS (small)
st.markdown(
    """
    <style>
      @media (max-width: 600px) {
         .css-1v0mbdj { padding: 8px; } /* content padding tuning for small screens - class may vary */
      }
      /* Make the sidebar a bit more compact */
      .stSidebar { padding-top: 60px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# end of app.py
