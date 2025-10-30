# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from datetime import datetime

# ----------------------
# Config / Files
# ----------------------
USERS_FILE = "users.json"
CONFIG_FILE = "config.json"

# Ensure files exist with safe defaults
if not os.path.exists(USERS_FILE):
    # default superadmin only; other user accounts can be created from UI
    with open(USERS_FILE, "w") as f:
        json.dump({
            "superadmin": {"password": "rp@2025", "role": "super_admin"}
        }, f, indent=2)

if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "w") as f:
        json.dump({
            "auto_refresh_seconds": 60,
            "nse_symbols": ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS"],
            "bse_symbols": ["SBIN.BO", "TATASTEEL.BO", "ITC.BO", "BAJFINANCE.BO", "HINDUNILVR.BO"]
        }, f, indent=2)

# Load data
with open(USERS_FILE, "r") as f:
    users_db = json.load(f)

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

AUTO_REFRESH = int(config.get("auto_refresh_seconds", 60))
NSE_SYMBOLS = config.get("nse_symbols", [])
BSE_SYMBOLS = config.get("bse_symbols", [])

# ----------------------
# Page config
# ----------------------
st.set_page_config(page_title="RPStockInsight", layout="wide")

# Global session state keys
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "role" not in st.session_state:
    st.session_state.role = None

# ----------------------
# Helper functions
# ----------------------
def save_users():
    with open(USERS_FILE, "w") as f:
        json.dump(users_db, f, indent=2)

def save_config():
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

def add_user(username: str, password: str, role: str = "user"):
    users_db[username] = {"password": password, "role": role}
    save_users()

def delete_user(username: str):
    if username in users_db:
        del users_db[username]
        save_users()
        return True
    return False

def update_user(username: str, password: str = None, role: str = None):
    if username in users_db:
        if password:
            users_db[username]["password"] = password
        if role:
            users_db[username]["role"] = role
        save_users()
        return True
    return False

def get_role(username):
    return users_db.get(username, {}).get("role")

def fetch_ticker_price(sym):
    try:
        t = yf.Ticker(sym)
        h = t.history(period="1d", interval="1m")
        if h.empty:
            return None
        price = float(h["Close"].iloc[-1])
        prev = float(h["Close"].iloc[-2]) if len(h) > 1 else price
        pct = ((price - prev) / prev * 100) if prev != 0 else 0.0
        return price, pct
    except Exception:
        return None

def build_ticker_html(label, symbols):
    parts = []
    for s in symbols:
        res = fetch_ticker_price(s)
        if res is None:
            continue
        price, pct = res
        arrow = "▲" if pct >= 0 else "▼"
        color = "limegreen" if pct >= 0 else "tomato"
        short = s.replace(".NS", "").replace(".BO", "")
        parts.append(f"{short}: {price:,.2f} <span style='color:{color}'>{arrow} {pct:.2f}%</span>")
    joined = " &nbsp;&nbsp;|&nbsp;&nbsp; ".join(parts) if parts else "No data"
    return f"<b>{label}</b> — {joined}"

# ----------------------
# Top banner + ticker (visible to all pages)
# ----------------------
st.markdown("""<style>
.ticker { white-space: nowrap; overflow: hidden; box-sizing: border-box;
  animation: marquee 28s linear infinite; font-size:15px; font-weight:500;
  padding:8px; color: white; background-color:#111; border-radius:6px; margin-bottom:6px;}
.ticker:hover { animation-play-state: paused; }
@keyframes marquee { 0% { text-indent: 100%; } 100% { text-indent: -100%; } }
.header { display:flex; align-items:center; justify-content:space-between; }
.header-title { font-size:22px; font-weight:700; }
.meta { font-size:12px; color:gray; }
</style>""", unsafe_allow_html=True)

st.markdown(f"""
<div class="header">
  <div class="header-title">📊 RPStockInsight</div>
  <div class="meta">Auto-refresh: {AUTO_REFRESH}s · Mobile-friendly · Role-based access</div>
</div>
""", unsafe_allow_html=True)

# Two tickers stacked
nse_html = build_ticker_html("NSE", NSE_SYMBOLS)
bse_html = build_ticker_html("BSE", BSE_SYMBOLS)

st.markdown(f"<div class='ticker'>{nse_html}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='ticker'>{bse_html}</div>", unsafe_allow_html=True)

# auto refresh meta tag (for page reload)
st.markdown(f"<meta http-equiv='refresh' content='{AUTO_REFRESH}'>", unsafe_allow_html=True)

# ----------------------
# Sidebar: Login / Logout / Role actions
# ----------------------
st.sidebar.title("Account")
if st.session_state.logged_in:
    st.sidebar.write(f"Logged in as: **{st.session_state.username}** ({st.session_state.role})")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.role = None
        st.experimental_rerun()
else:
    st.sidebar.subheader("Login")
    input_user = st.sidebar.text_input("Username")
    input_pass = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if input_user in users_db and users_db[input_user]["password"] == input_pass:
            st.session_state.logged_in = True
            st.session_state.username = input_user
            st.session_state.role = users_db[input_user]["role"]
            st.sidebar.success("Logged in")
            st.experimental_rerun()
        else:
            st.sidebar.error("Invalid credentials")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Register (Users only)")
    # simple registration for testing; role defaults to 'user'
    reg_user = st.sidebar.text_input("New username", key="reg_user")
    reg_pass = st.sidebar.text_input("New password", type="password", key="reg_pass")
    if st.sidebar.button("Register new user"):
        if reg_user in users_db:
            st.sidebar.error("User exists")
        elif not reg_user or not reg_pass:
            st.sidebar.error("Enter username and password")
        else:
            add_user(reg_user, reg_pass, role="user")
            st.sidebar.success("User created (role=user)")

# ----------------------
# Main content
# ----------------------
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Market Watch / Query")
    # input for the stock symbol to query
    query_symbol = st.text_input("Enter stock symbol (NSE/BSE) e.g. INFY.NS", value="INFY.NS")
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    if st.button("Fetch Data"):
        try:
            df = yf.download(query_symbol, period=period, interval=interval)
            if df is None or df.empty:
                st.warning("⚠️ No data found. Please verify symbol or try later.")
            else:
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']
                )])
                fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df.tail(10))
        except Exception as e:
            st.error(f"Error fetching data: {e}")

with col2:
    st.subheader("Quick Predictions & Signals")
    # By default run prediction for query_symbol if available
    try:
        df_pred = yf.download(query_symbol, period="6mo", interval="1d")
        if df_pred is None or df_pred.empty:
            st.info("No recent daily data to run predictions. Change symbol/period.")
        else:
            df_pred = df_pred.dropna()
            # Simple linear model on day index vs close
            df_pred = df_pred.reset_index()
            df_pred["Days"] = np.arange(len(df_pred))
            X = df_pred[["Days"]]
            y = df_pred["Close"]
            model = LinearRegression()
            model.fit(X, y)
            pred_30 = float(model.predict([[len(df_pred) + 30]])[0])
            pred_90 = float(model.predict([[len(df_pred) + 90]])[0])
            st.metric("Predicted price (30 days)", f"₹{pred_30:,.2f}")
            st.metric("Predicted price (90 days)", f"₹{pred_90:,.2f}")

            # Breakout detection - robust floats
            recent = df_pred["Close"].tail(10)
            avg_recent = float(recent.mean())
            last_price = float(recent.iloc[-1])
            if last_price > 1.05 * avg_recent:
                st.success("🚀 Potential Breakout: price above recent average")
            elif last_price < 0.95 * avg_recent:
                st.error("📉 Possible Breakdown: price below recent average")
            else:
                st.info("📊 No breakout: trend stable")
    except Exception as e:
        st.warning(f"Prediction/Signal error: {e}")

st.markdown("---")

# ----------------------
# Role-based panels
# ----------------------
role = st.session_state.role if st.session_state.logged_in else None

# Super Admin panel (full control)
if role == "super_admin":
    st.header("🟣 Super Admin Panel")
    st.markdown("Manage admins/users, config and reset files.")

    with st.expander("Users (add / edit / delete)"):
        st.write("Existing users and roles:")
        for uname, uinfo in users_db.items():
            st.write(f"- **{uname}** — role: `{uinfo.get('role')}`")
        st.write("")
        new_un = st.text_input("New username (create)", key="su_new_un")
        new_pw = st.text_input("New password", type="password", key="su_new_pw")
        new_role = st.selectbox("Role", ["user", "admin", "super_admin"], index=0, key="su_new_role")
        if st.button("Create user", key="su_create"):
            if not new_un or not new_pw:
                st.error("Provide username and password")
            else:
                add_user(new_un, new_pw, new_role)
                st.success(f"Created `{new_un}` with role `{new_role}`")

        edit_user = st.selectbox("Select user to edit/delete", list(users_db.keys()), key="su_edit_sel")
        edit_pw = st.text_input("Set new password (leave blank to keep)", type="password", key="su_edit_pw")
        edit_role = st.selectbox("Set role", ["user", "admin", "super_admin"], index=0, key="su_edit_role")
        if st.button("Update selected user", key="su_update"):
            if update_user(edit_user, password=(edit_pw if edit_pw else None), role=edit_role):
                st.success("User updated")
            else:
                st.error("Update failed")
        if st.button("Delete selected user", key="su_del"):
            if edit_user == st.session_state.username:
                st.error("Cannot delete yourself while logged in")
            else:
                if delete_user(edit_user):
                    st.success("User deleted")
                else:
                    st.error("Delete failed")

    with st.expander("Config (auto-refresh & tickers)"):
        auto_r = st.number_input("Auto-refresh seconds", min_value=10, max_value=600, value=AUTO_REFRESH)
        nse_txt = st.text_area("NSE symbols (comma separated)", value=",".join(NSE_SYMBOLS))
        bse_txt = st.text_area("BSE symbols (comma separated)", value=",".join(BSE_SYMBOLS))
        if st.button("Save config"):
            config["auto_refresh_seconds"] = int(auto_r)
            config["nse_symbols"] = [s.strip() for s in nse_txt.split(",") if s.strip()]
            config["bse_symbols"] = [s.strip() for s in bse_txt.split(",") if s.strip()]
            save_config()
            st.success("Config saved. Restart app to apply changes.")

    with st.expander("Maintenance"):
        if st.button("Reset users file (keep superadmin)"):
            with open(USERS_FILE, "w") as f:
                json.dump({"superadmin": {"password": "rp@2025", "role": "super_admin"}}, f, indent=2)
            st.success("Users reset to superadmin only. Please re-login.")
        if st.button("Download users.json"):
            with open(USERS_FILE, "r") as f:
                st.download_button("Download users.json", f.read(), file_name="users.json")

# Admin panel (limited)
elif role == "admin":
    st.header("🟢 Admin Panel")
    st.markdown("Manage users (not admins) and view basic reports.")

    with st.expander("Users (create / edit / delete - restricted)"):
        # Admin cannot create other admins or super_admin
        un = st.text_input("New username", key="ad_new_un")
        pw = st.text_input("Password", type="password", key="ad_new_pw")
        if st.button("Create user (role=user)", key="ad_create"):
            if un and pw:
                add_user(un, pw, role="user")
                st.success("User created (role=user)")
            else:
                st.error("Provide username and password")

        sel = st.selectbox("Select user", [u for u,info in users_db.items()], key="ad_sel")
        newpw = st.text_input("Set password (leave blank keep)", type="password", key="ad_up_pw")
        if st.button("Update password", key="ad_up"):
            if newpw:
                update_user(sel, password=newpw)
                st.success("Password updated")
            else:
                st.error("Enter a password")
        if st.button("Delete user", key="ad_del"):
            if sel == st.session_state.username:
                st.error("Cannot delete yourself")
            else:
                if delete_user(sel):
                    st.success("User deleted")
                else:
                    st.error("Delete failed")

# Regular user view
else:
    st.header("🔵 User Dashboard")
    st.markdown("You can query stock data, see quick predictions and breakout signals. Register/login to get personalized features.")

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.markdown(f"<div style='text-align:center;font-size:13px;'>© 2025 RPStockInsight · Built for testing · Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)
