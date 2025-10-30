
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os, json, hashlib, secrets
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
USERS_PATH = os.path.join(BASE_DIR, "users.json")

if not os.path.exists(CONFIG_PATH):
    default_config = {
        "project_name": "RPStockInsight",
        "admins": ["admin@example.com"],
        "testing_mode": True,
        "default_period": "6mo",
        "default_interval": "1d"
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(default_config, f, indent=2)

if not os.path.exists(USERS_PATH):
    with open(USERS_PATH, "w") as f:
        json.dump({}, f, indent=2)

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

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

st.set_page_config(page_title="RPStockInsight", layout="wide")
st.title("📊 RPStockInsight (Streamlit Cloud Ready)")

menu = ["Home","Sign Up","Login","Admin"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.header("Welcome to RPStockInsight")
    st.write("Features included: real-time data (Yahoo), simple predictions, breakout detection, sector insights, user portfolios.")
    st.write("Testing mode:", CONFIG.get("testing_mode", True))
    st.markdown("---")
    st.subheader("Quick demo (no login required)")
    ticker = st.text_input("Ticker (e.g. INFY.NS or RELIANCE.NS)", "INFY.NS")
    period = st.selectbox("Period", ["1mo","3mo","6mo","1y","2y"], index=2)
    interval = st.selectbox("Interval", ["1d","1wk"], index=0)
    if st.button("Fetch demo data"):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if df.empty:
                st.error("No data from Yahoo for that ticker. Try different symbol.")
            else:
                if "Close" not in df.columns and "Adj Close" in df.columns:
                    df["Close"] = df["Adj Close"]
                df.reset_index(inplace=True)
                st.dataframe(df.tail(10))
                st.line_chart(df.set_index("Date")["Close"])
        except Exception as e:
            st.error("Error fetching data: " + str(e))

elif choice == "Sign Up":
    st.header("Create an account (test mode)")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")
    if st.button("Create Account"):
        ok, msg = create_user(new_user, new_pass)
        if ok:
            st.success(msg + " — you can now login from the Login page.")
        else:
            st.error(msg)

elif choice == "Login":
    st.header("Login to your account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if verify_user(username, password):
            st.success(f"Welcome back, {username}!")
            users = load_users()
            user = users.get(username)
            st.subheader("Portfolio")
            cols = st.columns([3,1])
            with cols[0]:
                st.write("Holdings:", user.get("portfolio", []))
                add = st.text_input("Add ticker (use .NS for NSE)", key="add_"+username)
                if st.button("Add to portfolio", key="addbtn_"+username):
                    if add:
                        user["portfolio"].append(add.upper())
                        users[username] = user
                        save_users(users)
                        st.success(f"Added {add.upper()}")
            with cols[1]:
                if st.button("Show portfolio prices"):
                    if user.get("portfolio"):
                        for t in user["portfolio"]:
                            df = yf.download(t, period=CONFIG.get("default_period","6mo"), interval=CONFIG.get("default_interval","1d"), progress=False)
                            if df is None or df.empty:
                                st.warning(f"No data for {t}")
                                continue
                            if "Close" not in df.columns and "Adj Close" in df.columns:
                                df["Close"] = df["Adj Close"]
                            df.reset_index(inplace=True)
                            st.write(f"### {t}")
                            st.line_chart(df.set_index("Date")["Close"])
            st.markdown("---")
            st.subheader("Analyze a ticker")
            t = st.text_input("Ticker to analyze", "INFY.NS")
            if st.button("Analyze"):
                df = yf.download(t, period="1y", interval="1d", progress=False)
                if df is None or df.empty:
                    st.error("No data")
                else:
                    if "Close" not in df.columns and "Adj Close" in df.columns:
                        df["Close"] = df["Adj Close"]
                    df.reset_index(inplace=True)
                    df["MA20"] = df["Close"].rolling(20).mean()
                    df["MA50"] = df["Close"].rolling(50).mean()
                    st.line_chart(df.set_index("Date")[["Close","MA20","MA50"]])
                    latest = df.iloc[-1]
                    breakout = False
                    try:
                        breakout = latest["Close"] > latest["MA20"] and latest["MA20"] > latest["MA50"]
                    except Exception:
                        breakout = False
                    st.write("Breakout:", breakout)
                    df["X"] = np.arange(len(df))
                    model = LinearRegression()
                    model.fit(df[["X"]], df["Close"])
                    future_x = np.arange(len(df), len(df)+30).reshape(-1,1)
                    preds = model.predict(future_x)
                    future_dates = [df["Date"].iloc[-1] + timedelta(days=i) for i in range(1,31)]
                    pred_df = pd.DataFrame({"Date":future_dates, "Predicted":preds})
                    st.line_chart(pred_df.set_index("Date")["Predicted"])
        else:
            st.error("Invalid credentials")

elif choice == "Admin":
    st.header("Admin Panel")
    admin_user = st.text_input("Admin username")
    admin_pass = st.text_input("Admin password", type="password")
    if st.button("Admin Login"):
        if verify_user(admin_user, admin_pass):
            usr = load_users().get(admin_user)
            if usr and usr.get("role") in ("admin","superadmin"):
                st.success("Admin access granted")
                users = load_users()
                st.subheader("Registered users")
                st.write(users)
                st.markdown("### Config")
                st.write(CONFIG)
                if st.button("Toggle testing mode"):
                    CONFIG["testing_mode"] = not CONFIG.get("testing_mode", True)
                    with open(CONFIG_PATH,"w") as f:
                        json.dump(CONFIG, f, indent=2)
                    st.experimental_rerun()
            else:
                st.error("Not an admin")
        else:
            st.error("Invalid admin credentials")
