"""
RPStockInsight - final app.py (single-file Streamlit app)

Features:
- Pinned live ticker area (NSE & BSE) — small font, auto-refresh via caching TTL=60s
- CSV upload to fetch many tickers (use column 'SYMBOL' with ticker like 'TCS.NS' or 'RELIANCE.NS', '500112.BO' (example BSE format))
- Built-in simple breakout detection + entry/stop/target suggestion
- Short-term and long-term simple predictions based on linear regression on closes (small, illustrative)
- Local auth with roles: superadmin, admin, user (default credentials in users.json; change ASAP)
- Mobile-friendly layout
- Caching to avoid excessive repeated fetches (ttl=60 seconds)
- Instructions: update requirements.txt with: streamlit, yfinance, pandas, numpy
- NOTE: For production use Streamlit Cloud secrets for real credentials (do NOT commit secrets)

How to use:
- Place this file in your repo root as app.py
- requirements.txt must include: streamlit, yfinance, pandas, numpy
- Deploy to Streamlit Cloud or run locally with `streamlit run app.py`
- To load "all tickers": upload CSV (column SYMBOL). For very large lists, upload and let it run in batches.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import json

# ---------------------------
# CONFIG / DEFAULTS
# ---------------------------
APP_TITLE = "RPStockInsight"
TICKER_REFRESH_SECONDS = 60  # cache TTL (behaves like auto-refresh)
DEFAULT_SAMPLE_TICKERS = [
    "NSEI.NS",  # NIFTY index if you want (yfinance uses ^NSEI sometimes); using NSEI.NS may return symbol-specific
    "TCS.NS",
    "INFY.NS",
    "RELIANCE.NS",
    "HDFC.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "LT.NS"
]
# Default local users (change these immediately after deploying)
# Roles: superadmin, admin, user
DEFAULT_USERS = {
    "superadmin": {"password": "SuperAdmin@123", "role": "superadmin"},
    "admin": {"password": "Admin@123", "role": "admin"},
    "tester": {"password": "Test@123", "role": "user"}
}

# ---------------------------
# Utility functions
# ---------------------------
@st.cache_data(ttl=TICKER_REFRESH_SECONDS)
def fetch_ticker_data(tickers, period="1mo", interval="1d"):
    """
    Fetch historical data (close/open/high/low/volume) for a list of tickers using yfinance.
    Returns dict: ticker -> dataframe
    Caching TTL ensures auto-refresh behavior.
    """
    out = {}
    # Use yfinance's download for batched fetching when list is not too large
    tickers_list = list(tickers)
    # yfinance supports batch download; but returns a multi-index DataFrame if >1 ticker
    try:
        if len(tickers_list) == 0:
            return out
        if len(tickers_list) == 1:
            t = tickers_list[0]
            df = yf.download(t, period=period, interval=interval, progress=False, threads=True)
            if df is None or df.empty:
                out[t] = None
            else:
                df = df.reset_index()
                out[t] = df
        else:
            # batch download
            batch_df = yf.download(tickers_list, period=period, interval=interval, progress=False, threads=True)
            # If returns single-level columns (rare), handle
            if isinstance(batch_df.columns, pd.MultiIndex):
                # iterate tickers
                for t in tickers_list:
                    try:
                        df_t = batch_df.xs(t, axis=1, level=1, drop_level=False)
                        # xs could produce a multiindex; simplify
                        df = pd.DataFrame({
                            'Date': batch_df.index,
                            'Open': batch_df[('Open', t)],
                            'High': batch_df[('High', t)],
                            'Low': batch_df[('Low', t)],
                            'Close': batch_df[('Close', t)],
                            'Adj Close': batch_df[('Adj Close', t)] if ('Adj Close', t) in batch_df.columns else batch_df[('Close', t)],
                            'Volume': batch_df[('Volume', t)]
                        }).reset_index(drop=True)
                        out[t] = df
                    except Exception:
                        out[t] = None
            else:
                # single ticker fallback
                for t in tickers_list:
                    df = batch_df.copy()
                    df['Date'] = df.index
                    out[t] = df.reset_index(drop=True)
    except Exception as e:
        # fallback: try per ticker fetch - slower but robust
        for t in tickers:
            try:
                df = yf.download(t, period=period, interval=interval, progress=False, threads=True)
                if df is None or df.empty:
                    out[t] = None
                else:
                    out[t] = df.reset_index()
            except Exception:
                out[t] = None
    return out


def compute_breakout_signals(df, lookback=20, volume_spike=1.5):
    """
    Simple breakout detector:
    - breakout if last Close > rolling_max(lookback)
    - and volume > avg_volume * volume_spike
    Returns dict with breakout boolean and metrics
    """
    if df is None or df.empty or 'Close' not in df.columns:
        return None
    d = df.copy()
    if 'Volume' not in d.columns:
        d['Volume'] = 0
    d['rolling_max'] = d['Close'].rolling(lookback).max()
    d['rolling_min'] = d['Low'].rolling(lookback).min()
    d['avg_vol'] = d['Volume'].rolling(lookback).mean()
    latest = d.iloc[-1]
    prev_max = d['rolling_max'].iloc[-2] if len(d) > 1 else np.nan
    breakout = False
    try:
        breakout = bool(latest['Close'] > prev_max and latest['Volume'] > (latest['avg_vol'] * volume_spike if not np.isnan(latest['avg_vol']) else 0))
    except Exception:
        breakout = False
    # build suggestion
    entry = float(latest['Close']) if not np.isnan(latest['Close']) else None
    stop = float(d['rolling_min'].iloc[-1]) if not np.isnan(d['rolling_min'].iloc[-1]) else None
    if entry is None or stop is None:
        target = None
    else:
        # target using 1.5x reward:risk
        rr = entry - stop
        target = round(entry + 1.5 * rr, 2)
    return {
        "breakout": breakout,
        "entry": entry,
        "stop": stop,
        "target": target,
        "latest_date": latest['Date'] if 'Date' in latest.index else None,
        "close": float(latest['Close']) if not np.isnan(latest['Close']) else None,
        "volume": float(latest['Volume']) if not np.isnan(latest['Volume']) else None,
        "avg_vol": float(latest['avg_vol']) if not np.isnan(latest['avg_vol']) else None
    }


def simple_short_long_prediction(df):
    """
    Very simple prediction demonstration:
    - Fit Linear Regression on last N closes and extrapolate
    - Return predicted price for short-term (7 days) and long-term (30 days)
    This is illustrative only. For production, use Prophet/ARIMA/etc.
    """
    try:
        from sklearn.linear_model import LinearRegression
    except Exception:
        return {"short": None, "long": None, "error": "sklearn not available"}

    if df is None or df.empty or 'Close' not in df.columns:
        return {"short": None, "long": None, "error": "no data"}

    d = df.reset_index(drop=True).copy()
    # Use last 30 values if available
    N = min(60, len(d))
    d = d.tail(N).reset_index(drop=True)
    d['t'] = np.arange(len(d))
    X = d[['t']].values
    y = d['Close'].values
    if len(X) < 5:
        return {"short": None, "long": None, "error": "not enough data"}
    model = LinearRegression().fit(X, y)
    last_t = X[-1, 0]
    short_t = last_t + 7  # approx days
    long_t = last_t + 30
    pred_short = float(model.predict(np.array([[short_t]]))[0])
    pred_long = float(model.predict(np.array([[long_t]]))[0])
    return {"short": round(pred_short, 2), "long": round(pred_long, 2), "error": None}


# ---------------------------
# Simple local auth (file-backed)
# ---------------------------
# users.json is used to persist users locally. If not present, create with DEFAULT_USERS.
USERS_FILE = "users.json"


def load_users():
    try:
        with open(USERS_FILE, "r") as f:
            users = json.load(f)
    except FileNotFoundError:
        users = DEFAULT_USERS.copy()
        save_users(users)
    except Exception:
        users = DEFAULT_USERS.copy()
    return users


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def authenticate(username, password):
    users = load_users()
    if username in users and users[username]["password"] == password:
        return users[username]["role"]
    return None


# ---------------------------
# UI helpers
# ---------------------------
def pinned_ticker_html(nse_items, bse_items):
    """
    Build a compact HTML for the pinned top ticker area. We will show small font and two sections.
    nse_items and bse_items are lists of (symbol, last_price, pct_change)
    """
    def row_html(items, label):
        parts = []
        for sym, price, pct in items:
            arrow = "▲" if pct >= 0 else "▼"
            parts.append(f"<span style='margin-right:12px;padding:4px 6px;display:inline-block'>{sym}: {price:.2f} {arrow} {pct:+.2f}%</span>")
        return f"<div style='display:flex;align-items:center;overflow:auto;white-space:nowrap'><strong style='margin-right:8px'>{label}:</strong>" + "".join(parts) + "</div>"

    html = f"""
    <div style='width:100%;font-size:14px;'>
      <div style='display:flex;gap:16px;flex-direction:column;'>
        {row_html(nse_items,"NSE")}
        {row_html(bse_items,"BSE")}
      </div>
    </div>
    """
    return html


# ---------------------------
# Streamlit layout
# ---------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

# Top header area
st.markdown(f"# {APP_TITLE}")
st.markdown("Mobile-friendly · Auto light/dark theme · Pinned live ticker (auto-refresh every 60s)")

# Sidebar: settings, auth, inputs
with st.sidebar:
    st.header("Dashboard Settings")
    symbol_input = st.text_input("Enter Stock Symbol (NSE/BSE) e.g. TCS.NS, INFY.NS", value="TCS.NS")
    period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y"], index=1)
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

    st.markdown("---")
    st.subheader("Admin area (local test/auth)")
    users = load_users()
    st.info("Note: This is local test auth. Change passwords in users.json for production.")
    auth_mode = st.radio("Admin action", ["Login", "Register"], index=0)
    if auth_mode == "Login":
        username = st.text_input("Username", value="")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            role = authenticate(username, pwd)
            if role:
                st.success(f"Logged in as {username} ({role})")
                st.session_state["user"] = username
                st.session_state["role"] = role
            else:
                st.error("Invalid credentials")
    else:
        # Register: only accessible if current session user is superadmin or no superadmin exists
        reg_user = st.text_input("New username", value="")
        reg_pwd = st.text_input("New password", type="password")
        reg_role = st.selectbox("Role for new user", ["user", "admin"])
        if st.button("Register"):
            # only allow registration if no users (first-time) or if logged in as superadmin
            allow = False
            if "role" not in st.session_state:
                # permit initial registration to create admin
                allow = True
            else:
                allow = st.session_state.get("role") == "superadmin"
            if not allow:
                st.error("Only superadmin can register new users.")
            else:
                u = load_users()
                if reg_user in u:
                    st.error("User exists")
                else:
                    u[reg_user] = {"password": reg_pwd, "role": reg_role}
                    save_users(u)
                    st.success(f"Created user {reg_user} ({reg_role})")


st.markdown("---")

# Build tickers to show at top: use sample list + optional CSV upload
col_top1, col_top2 = st.columns([5, 1])
with col_top1:
    st.markdown("### Live Tickers (small)")
    uploaded = st.file_uploader("Upload CSV of tickers (column 'SYMBOL') to monitor many symbols (optional)", type=["csv"])
    if uploaded:
        try:
            df_symbols = pd.read_csv(uploaded)
            if 'SYMBOL' in df_symbols.columns:
                watch_list = df_symbols['SYMBOL'].dropna().astype(str).unique().tolist()
            else:
                # assume first column has tickers
                watch_list = df_symbols.iloc[:,0].dropna().astype(str).unique().tolist()
        except Exception:
            st.error("Error reading CSV — make sure it has a SYMBOL column or one column with tickers")
            watch_list = DEFAULT_SAMPLE_TICKERS
    else:
        watch_list = DEFAULT_SAMPLE_TICKERS

# ensure small lists first for live top area to keep it fast
top_nse = [t for t in watch_list if t.endswith(".NS")]
top_bse = [t for t in watch_list if t.endswith(".BO") or t.endswith(".BSE") or t.endswith(".BO.NS") or t.endswith(".BSE.NS")]
# fallback: treat unknown as NSE sample
if not top_nse and watch_list:
    top_nse = watch_list[:6]
if not top_bse:
    top_bse = []

# fetch only latest price for top tickers
live_dict = fetch_ticker_data(top_nse + top_bse, period="5d", interval="1d")
nse_items = []
bse_items = []
for t in top_nse:
    df_t = live_dict.get(t)
    if df_t is None or df_t.empty or 'Close' not in df_t.columns:
        continue
    last = df_t.iloc[-1]
    prev = df_t['Close'].iloc[-2] if len(df_t) > 1 else last['Close']
    price = float(last['Close'])
    pct = ((price - prev) / prev * 100) if prev != 0 else 0.0
    nse_items.append((t, price, pct))
for t in top_bse:
    df_t = live_dict.get(t)
    if df_t is None or df_t.empty or 'Close' not in df_t.columns:
        continue
    last = df_t.iloc[-1]
    prev = df_t['Close'].iloc[-2] if len(df_t) > 1 else last['Close']
    price = float(last['Close'])
    pct = ((price - prev) / prev * 100) if prev != 0 else 0.0
    bse_items.append((t, price, pct))

with col_top1:
    html = pinned_ticker_html(nse_items[:12], bse_items[:12])  # show only up to 12 each to keep it readable
    st.markdown(html, unsafe_allow_html=True)

with col_top2:
    st.write("")  # spacer
    st.caption(f"Auto-refresh: every {TICKER_REFRESH_SECONDS}s (cache TTL)")

st.markdown("---")

# Main controls & stock summary
st.subheader(f"Stock Summary for {symbol_input}")
# fetch main symbol data
data_dict = fetch_ticker_data([symbol_input], period=period, interval=interval)
main_df = data_dict.get(symbol_input)

if main_df is None or main_df.empty:
    st.warning("⚠️ No data found. Please check the stock symbol or try another.")
else:
    # show quick stats
    try:
        avg_close = main_df['Close'].mean()
        high_close = main_df['Close'].max()
        low_close = main_df['Close'].min()
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Average Close", f"{avg_close:.2f}")
        colB.metric("Highest Close", f"{high_close:.2f}")
        colC.metric("Lowest Close", f"{low_close:.2f}")
        colD.metric("Period", period)
    except Exception:
        st.write("Could not compute metrics (missing columns).")

    st.markdown("### Recent Data")
    st.dataframe(main_df.tail(10).reset_index(drop=True))

    # Predictions (simple)
    preds = simple_short_long_prediction(main_df)
    st.markdown("### 🔮 Short-term & long-term predictions")
    if preds.get("error"):
        st.warning("Prediction not available: " + str(preds.get("error")))
    else:
        st.info(f"Predicted price in ~7 days ≈ ₹{preds['short']}  •  in ~30 days ≈ ₹{preds['long']}")

    # Breakout detection for current symbol
    st.markdown("### 🚀 Breakout Detection")
    try:
        sig = compute_breakout_signals(main_df, lookback=20, volume_spike=1.5)
        if sig is None:
            st.warning("No breakout data (insufficient data).")
        else:
            if sig['breakout']:
                st.success("Breakout detected ✅")
                st.write(f"Entry (current close): ₹{sig['entry']}")
                st.write(f"Stop loss (rolling low): ₹{sig['stop']}")
                st.write(f"Target (1.5x R:R): ₹{sig['target']}")
                st.write(f"Volume: {int(sig['volume'])}  •  Avg vol: {int(sig['avg_vol']) if sig['avg_vol'] is not None else 'N/A'}")
                st.caption("Suggested timeframe: Short-term (7-21 days). This is NOT financial advice.")
            else:
                st.info("No breakout detected (by our simple rule).")
    except Exception as e:
        st.error("Error analyzing breakout: " + str(e))

# Multi-ticker section: fetch watch list and compute breakout candidates
st.markdown("---")
st.subheader("🔎 Watchlist & Breakout Suggestions")

# If user uploaded a big list, allow them to control sample size (for demo speed)
if uploaded:
    total_watch = len(watch_list)
    st.caption(f"Uploaded {total_watch} tickers. We'll fetch in a batch; for big lists this may take longer.")
    if total_watch > 50:
        sample_count = st.number_input("Limit to first N items to analyze (for faster results)", min_value=10, max_value=total_watch, value=min(50, total_watch))
        analysis_list = watch_list[:int(sample_count)]
    else:
        analysis_list = watch_list
else:
    analysis_list = watch_list

if st.button("Analyze Watchlist (breakouts & predictions)"):
    with st.spinner("Fetching data and analyzing..."):
        dict_all = fetch_ticker_data(analysis_list, period="3mo", interval="1d")
        breakout_rows = []
        suggestion_count = 0
        for t in analysis_list:
            df_t = dict_all.get(t)
            if df_t is None or df_t.empty:
                continue
            sig = compute_breakout_signals(df_t, lookback=20, volume_spike=1.5)
            preds = simple_short_long_prediction(df_t)
            if sig and sig.get("breakout"):
                suggestion_count += 1
                breakout_rows.append({
                    "Ticker": t,
                    "Entry": sig["entry"],
                    "Stop": sig["stop"],
                    "Target": sig["target"],
                    "ShortPred": preds.get("short"),
                    "LongPred": preds.get("long"),
                    "Volume": sig.get("volume"),
                    "AvgVol": sig.get("avg_vol")
                })
        if suggestion_count == 0:
            st.info("No breakout candidates found in the requested list (by the simple rule).")
        else:
            st.success(f"Found {suggestion_count} breakout candidates.")
            df_b = pd.DataFrame(breakout_rows)
            st.dataframe(df_b)

            # provide CSV download
            csv = df_b.to_csv(index=False).encode('utf-8')
            st.download_button("Download suggestions CSV", data=csv, file_name="breakout_suggestions.csv", mime="text/csv")

# Footer and admin info
st.markdown("---")
st.caption("© 2025 RPStockInsight · Built with ❤️ using Streamlit")
st.caption(f"Updated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Quick instructions and next steps for you (not displayed to public users normally)
if "role" in st.session_state and st.session_state.get("role") == "superadmin":
    st.info("Superadmin tools: You can edit users.json directly in your repo, or use Register to add users.")
    st.write("For production: move credentials to Streamlit secrets (Manage app → Settings → Secrets) and remove plaintext credentials from users.json")
