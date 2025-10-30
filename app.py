# app.py
# RPStockInsight - Mobile friendly + pinned ticker + auto-theme toggle (auto-refresh every 60s)
# Replace your existing app.py with this file and push to GitHub.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os, json, hashlib, secrets
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import streamlit.components.v1 as components

# --- Paths & config ---
BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
USERS_PATH = os.path.join(BASE_DIR, "users.json")

# create default files if missing
if not os.path.exists(CONFIG_PATH):
    cfg = {
        "project_name": "RPStockInsight",
        "admins": ["superadmin"],
        "testing_mode": True,
        "default_period": "6mo",
        "default_interval": "1d"
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

if not os.path.exists(USERS_PATH):
    # initial superadmin (username: superadmin, password: admin123) - hashed
    salt = "initialsalt"
    pw_hash = hashlib.sha256((salt + "admin123").encode("utf-8")).hexdigest()
    users_init = {
        "superadmin": {
            "salt": salt,
            "password": pw_hash,
            "role": "superadmin",
            "created": datetime.utcnow().isoformat(),
            "portfolio": []
        }
    }
    with open(USERS_PATH, "w") as f:
        json.dump(users_init, f, indent=2)

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

# helper user functions
def load_users():
    with open(USERS_PATH, "r") as f:
        return json.load(f)

def save_users(u):
    with open(USERS_PATH, "w") as f:
        json.dump(u, f, indent=2)

def hash_password(password, salt):
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

def create_user(username, password, role="user"):
    users = load_users()
    if not username or not password:
        return False, "Provide username and password"
    username = username.strip()
    if username in users:
        return False, "User already exists"
    salt = secrets.token_hex(8)
    users[username] = {
        "salt": salt,
        "password": hash_password(password, salt),
        "role": role,
        "created": datetime.utcnow().isoformat(),
        "portfolio": []
    }
    save_users(users)
    return True, "User created"

def verify_user(username, password):
    users = load_users()
    u = users.get(username)
    if not u:
        return False
    return u["password"] == hash_password(password, u["salt"])

# ---------------------------
# STREAMLIT UI + STYLES
# ---------------------------
st.set_page_config(page_title="RPStockInsight", layout="wide", initial_sidebar_state="expanded")

# pinned ticker CSS, mobile friendly
TICKER_CSS = """
<style>
/* pinned ticker */
#ticker-bar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 9999;
  padding: 8px 12px;
  display:flex;
  align-items:center;
  gap:12px;
  overflow-x:auto;
  white-space:nowrap;
  box-shadow: 0 2px 6px rgba(0,0,0,0.08);
  border-bottom: 1px solid rgba(0,0,0,0.06);
}
/* light theme */
@media (prefers-color-scheme: light) {
  #ticker-bar { background: #ffffff; color: #111827; }
  .ticker-item { background: rgba(17,24,39,0.04); padding:6px 10px; border-radius:999px; display:inline-block; margin-right:8px; font-weight:600; font-size:14px; }
  .ticker-up { color: #047857; } /* green */
  .ticker-down { color: #b91c1c; } /* red */
}
/* dark theme */
@media (prefers-color-scheme: dark) {
  #ticker-bar { background: #0b1220; color: #e6eef8; border-bottom: 1px solid rgba(255,255,255,0.04); }
  .ticker-item { background: rgba(255,255,255,0.03); padding:6px 10px; border-radius:999px; display:inline-block; margin-right:8px; font-weight:600; font-size:14px; }
  .ticker-up { color: #34d399; } /* green */
  .ticker-down { color: #f87171; } /* red */
}

/* leave space below ticker for content */
body > .main > div.block-container { padding-top: 72px !important; }

/* mobile adjustments */
@media (max-width: 600px) {
  .ticker-item { font-size: 13px; padding:6px 8px; }
  .small-btn { font-size:13px; padding:6px 8px; }
}
</style>
"""

# inject CSS
st.markdown(TICKER_CSS, unsafe_allow_html=True)

# small JS for auto refresh every 60 seconds (auto only)
# this reloads the page every 60 seconds so ticker and other cached data refresh automatically
AUTO_REFRESH_JS = """
<script>
setTimeout(function(){ window.location.reload(); }, 60000);
</script>
"""
components.html(AUTO_REFRESH_JS, height=0)

# ---------------------------
# Data utilities (cached)
# ---------------------------
@st.cache_data(ttl=60)
def fetch_price_change(symbol):
    """
    Return (last_price, change_pct, change_abs) or None if failed.
    symbol examples: 'INFY.NS', '^NSEI'
    """
    try:
        # fetch last two closes to compute percent change
        hist = yf.download(symbol, period="2d", interval="1d", progress=False, threads=False)
        if hist is None or hist.empty:
            return None
        # depending on yfinance version column names
        if "Close" in hist.columns:
            last = hist["Close"].iloc[-1]
            prev = hist["Close"].iloc[-2] if len(hist) >= 2 else last
        elif "Adj Close" in hist.columns:
            last = hist["Adj Close"].iloc[-1]
            prev = hist["Adj Close"].iloc[-2] if len(hist) >= 2 else last
        else:
            return None
        change_abs = last - prev
        change_pct = (change_abs / prev) * 100 if prev != 0 else 0.0
        return (float(last), float(change_pct), float(change_abs))
    except Exception:
        return None

@st.cache_data(ttl=60)
def fetch_top_active(candidates):
    """
    Given a list of tickers, return top 3 by absolute percent move (desc).
    """
    results = []
    for s in candidates:
        r = fetch_price_change(s)
        if r:
            results.append((s, r[0], r[1]))
    if not results:
        return []
    # sort by absolute change% desc
    results.sort(key=lambda x: abs(x[2]), reverse=True)
    return results[:3]

# ---------------------------
# TICKER BAR - pinned at top
# ---------------------------
def render_ticker_bar():
    # indices & fallback small list of active stocks
    indices = {
        "NIFTY": "^NSEI",
        "SENSEX": "^BSESN",
        "BANKNIFTY": "^NSEBANK"
    }
    active_candidates = ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS"]

    items_html = []
    # show indices
    for name, sym in indices.items():
        r = fetch_price_change(sym)
        if r:
            val, pct, _ = r
            arrow = "▲" if pct >= 0 else "▼"
            cls = "ticker-up" if pct >= 0 else "ticker-down"
            items_html.append(f'<span class="ticker-item">{name}: {val:,.2f} <span class="{cls}">{arrow} {pct:+.2f}%</span></span>')
        else:
            items_html.append(f'<span class="ticker-item">{name}: —</span>')

    # top active stocks
    top_active = fetch_top_active(active_candidates)
    for s, val, pct in top_active:
        arrow = "▲" if pct >= 0 else "▼"
        cls = "ticker-up" if pct >= 0 else "ticker-down"
        # show symbol short
        items_html.append(f'<span class="ticker-item">{s.replace(".NS","")}: {val:,.2f} <span class="{cls}">{arrow} {pct:+.2f}%</span></span>')

    # wrap in a container with JS freeflow (no manual refresh button since auto-only)
    html = f'<div id="ticker-bar"> {" ".join(items_html)} </div>'
    st.markdown(html, unsafe_allow_html=True)

# render ticker
render_ticker_bar()

# ---------------------------
# Page content (below ticker)
# ---------------------------
st.title("RPStockInsight")
st.write("Mobile-friendly · Auto light/dark theme · Pinned live ticker (auto-refresh every 60s)")

# layout: two columns on wide screens, stacked on mobile
is_wide = st.session_state.get("is_wide", True) if "is_wide" in st.session_state else True
col1, col2 = st.columns([2,1]) if st.beta_container is None else st.columns([2,1])

# Quick Home Dashboard in left column
with col1:
    st.header("Home Dashboard")
    st.markdown("**Top Movers & Quick Analyze**")

    # quick analysis input on top (ticker input pinned in content area)
    ticker_input = st.text_input("Analyze ticker (use .NS for NSE) — quick search:", value="INFY.NS", max_chars=20)
    st.markdown(" ")

    # show a compact card with latest price and small chart
    if st.button("Fetch & Analyze", key="fetch_main"):
        try:
            df = yf.download(ticker_input, period="6mo", interval="1d", progress=False, threads=False)
            if df is None or df.empty:
                st.error("No data found for that ticker. Try a different symbol or remove .NS to test fallback.")
            else:
                if "Close" not in df.columns and "Adj Close" in df.columns:
                    df["Close"] = df["Adj Close"]
                df.reset_index(inplace=True)
                last = df["Close"].iloc[-1]
                prev = df["Close"].iloc[-2] if len(df) >= 2 else last
                pct = ((last - prev) / prev) * 100 if prev != 0 else 0.0
                st.subheader(f"{ticker_input.upper()} — {last:,.2f} ({pct:+.2f}%)")
                st.line_chart(df.set_index("Date")["Close"])
                # simple MAs and breakout
                df["MA20"] = df["Close"].rolling(20).mean()
                df["MA50"] = df["Close"].rolling(50).mean()
                st.write("MA20 / MA50 (most recent):", f"{df['MA20'].iloc[-1]:.2f}" if not np.isnan(df['MA20'].iloc[-1]) else "n/a", "/", f"{df['MA50'].iloc[-1]:.2f}" if not np.isnan(df['MA50'].iloc[-1]) else "n/a")
                breakout = False
                try:
                    breakout = df["Close"].iloc[-1] > df["MA20"].iloc[-1] > df["MA50"].iloc[-1]
                except Exception:
                    breakout = False
                st.info("Breakout: " + ("✅ YES" if breakout else "— No"))
                # 30-day simple linear prediction
                df["X"] = np.arange(len(df)).reshape(-1,1)
                model = LinearRegression()
                model.fit(df[["X"]], df["Close"])
                future_x = np.arange(len(df), len(df)+30).reshape(-1,1)
                preds = model.predict(future_x)
                future_dates = [df["Date"].iloc[-1] + timedelta(days=i) for i in range(1,31)]
                pred_df = pd.DataFrame({"Date": future_dates, "Predicted": preds})
                st.line_chart(pred_df.set_index("Date")["Predicted"])
        except Exception as e:
            st.error("Fetch error: " + str(e))

    st.markdown("---")
    st.subheader("Quick top movers snapshot")
    # quick snapshot (showing the list again with small sparkline)
    snapshot_symbols = ["NIFTY", "SENSEX", "BANKNIFTY"]
    # reusing fetches: we already have top fetch functions cached
    snap_cols = st.columns(3)
    idx = 0
    for name, sym in [("NIFTY","^NSEI"),("SENSEX","^BSESN"),("BANKNIFTY","^NSEBANK")]:
        with snap_cols[idx]:
            r = fetch_price_change(sym)
            if r:
                val, pct, _ = r
                st.metric(label=name, value=f"{val:,.2f}", delta=f"{pct:+.2f}%")
            else:
                st.metric(label=name, value="—", delta="—")
        idx += 1

# Right column: Signup/Login/Admin
with col2:
    st.header("Account")
    menu = st.selectbox("Choose", ["Login","Sign Up","Admin"])
    if menu == "Sign Up":
        st.subheader("Create account")
        nu = st.text_input("Username", key="su_user")
        npw = st.text_input("Password (min 4 chars)", type="password", key="su_pass")
        if st.button("Create", key="create_user"):
            ok, msg = create_user(nu, npw)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    elif menu == "Login":
        st.subheader("Login")
        user = st.text_input("Username", key="li_user")
        pw = st.text_input("Password", type="password", key="li_pass")
        if st.button("Login", key="loginbtn"):
            if verify_user(user, pw):
                st.success(f"Welcome {user}!")
                users = load_users()
                profile = users.get(user, {})
                st.write("Role:", profile.get("role","user"))
                st.write("Member since:", profile.get("created","-"))
                st.markdown("---")
                st.subheader("Portfolio")
                holdings = profile.get("portfolio", [])
                st.write(holdings if holdings else "No holdings yet")
                add = st.text_input("Add ticker to portfolio (e.g. INFY.NS)", key="add_t")
                if st.button("Add to portfolio", key="addbtn"):
                    if add:
                        holdings.append(add.upper())
                        profile["portfolio"] = holdings
                        users[user] = profile
                        save_users(users)
                        st.success(f"Added {add.upper()}")

            else:
                st.error("Invalid credentials")

    else:  # Admin
        st.subheader("Admin (local access)")
        au = st.text_input("Admin username", key="ad_user")
        ap = st.text_input("Admin password", type="password", key="ad_pass")
        if st.button("Admin Login", key="admin_login"):
            if verify_user(au, ap):
                u = load_users().get(au)
                if u and u.get("role") in ("admin","superadmin"):
                    st.success("Admin access granted")
                    st.write("Registered users:")
                    st.write(load_users())
                    st.markdown("---")
                    st.write("Config")
                    st.write(CONFIG)
                    if st.button("Toggle testing mode"):
                        CONFIG["testing_mode"] = not CONFIG.get("testing_mode", True)
                        with open(CONFIG_PATH, "w") as f:
                            json.dump(CONFIG, f, indent=2)
                        st.success("Toggled testing mode")
                else:
                    st.error("User is not admin")
            else:
                st.error("Invalid admin credentials")

# footer
st.markdown("---")
st.caption("RPStockInsight — demo mode. For production use, migrate users.json to a proper DB and secure authentication.")

# (end)
