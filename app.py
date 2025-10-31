import streamlit as st
import pandas as pd
import yfinance as yf
import json
from datetime import datetime

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RPStockInsight", layout="wide")
st.title("📊 RPStockInsight")
st.caption("Live NSE/BSE Insights • Breakout Suggestions • Auto Light/Dark Mode")

CONFIG_FILE = "config.json"

# Load config safely
try:
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
        if "tickers" not in config:
            config["tickers"] = []
except (FileNotFoundError, json.JSONDecodeError):
    config = {"tickers": []}
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

# Load users safely
try:
    with open("users.json", "r") as f:
        users = json.load(f)
except FileNotFoundError:
    users = {"superadmin": {"password": "SuperAdmin@123", "role": "superadmin"}}
    with open("users.json", "w") as f:
        json.dump(users, f, indent=4)

default_tickers = [t["symbol"] for t in config["tickers"]] if config["tickers"] else ["TCS.NS", "INFY.NS", "RELIANCE.NS"]

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Dashboard Settings")
    selected_symbol = st.text_input("Enter Stock Symbol (e.g., TCS.NS, INFY.NS)", value="TCS.NS")
    period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=1)
    interval = st.selectbox("Select Interval", ["1d", "1h", "15m", "5m"], index=0)
    st.markdown("---")

    st.subheader("🔐 Admin Area")
    username = st.text_input("Username", "superadmin")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")

# ---------------- AUTH ----------------
logged_in = False
if login_btn:
    if username in users and users[username]["password"] == password:
        logged_in = True
        st.sidebar.success(f"✅ Logged in as {username}")
    else:
        st.sidebar.error("❌ Invalid credentials")

# ---------------- LIVE TICKER SECTION ----------------
st.subheader("📈 Live Market Tickers")
ticker_container = st.empty()


def get_live_prices(symbols):
    prices = {}
    for sym in symbols:
        try:
            data = yf.Ticker(sym).history(period="1d")
            if not data.empty:
                close_val = float(data["Close"].iloc[-1])
                change = float(data["Close"].pct_change().iloc[-1] * 100)
                prices[sym] = (close_val, change)
        except Exception:
            prices[sym] = (None, None)
    return prices


def display_scrolling_ticker(prices):
    if not prices:
        st.warning("No ticker data available.")
        return
    ticker_text = " | ".join(
        [f"{sym}: ₹{v[0]:.2f} ({v[1]:+.2f}%)" for sym, v in prices.items() if v[0] is not None]
    )
    ticker_html = f"""
    <marquee scrollamount="5" behavior="scroll" direction="left" style="font-size:18px; color:limegreen;">
        {ticker_text}
    </marquee>
    """
    ticker_container.markdown(ticker_html, unsafe_allow_html=True)


prices = get_live_prices(default_tickers)
display_scrolling_ticker(prices)

# ---------------- BREAKOUT SUGGESTIONS ----------------
st.subheader("🚀 Breakout Suggestions")
st.caption("Top potential breakouts by sector (short-term & long-term targets)")

breakout_data = []
for t in config["tickers"]:
    try:
        df = yf.download(t["symbol"], period="3mo", interval="1d", progress=False)
        if not df.empty:
            last = float(df["Close"].iloc[-1])
            ma20 = float(df["Close"].rolling(20).mean().iloc[-1])
            if last > ma20 * 1.03:  # breakout signal
                breakout_data.append({
                    "Stock": t["name"],
                    "Sector": t["sector"],
                    "Price (₹)": round(last, 2),
                    "Short Term Target (7–15d)": round(last * 1.05, 2),
                    "Long Term Target (30–60d)": round(last * 1.12, 2),
                    "Signal Date": datetime.today().strftime("%Y-%m-%d")
                })
    except Exception:
        pass

if breakout_data:
    st.dataframe(pd.DataFrame(breakout_data), width='stretch')
else:
    st.info("No breakout signals found currently.")

# ---------------- STOCK SUMMARY ----------------
st.subheader(f"📊 Stock Summary for {selected_symbol}")
try:
    df = yf.download(selected_symbol, period=period, interval=interval)
    if not df.empty:
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        last_price = float(df["Close"].iloc[-1])
        st.line_chart(df["Close"], width='stretch')
        st.metric(label="Latest Price", value=f"₹{last_price:.2f}")
    else:
        st.warning("No data available for the selected symbol.")
except Exception as e:
    st.error(f"Error fetching data: {e}")

# ---------------- ADMIN PANEL ----------------
if logged_in and users[username]["role"] == "superadmin":
    st.markdown("---")
    st.header("🛠 Manage Tickers (Super Admin Only)")

    st.subheader("Current Tickers")
    tick_df = pd.DataFrame(config["tickers"])
    if not tick_df.empty:
        st.dataframe(tick_df, width='stretch')
    else:
        st.info("No tickers configured yet.")

    st.subheader("➕ Add New Ticker")
    new_name = st.text_input("Company Name")
    new_symbol = st.text_input("Symbol (e.g., TCS.NS)")
    new_sector = st.text_input("Sector")
    if st.button("Add Ticker"):
        if new_symbol and new_name:
            config["tickers"].append({"name": new_name, "symbol": new_symbol, "sector": new_sector})
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
            st.success(f"✅ Added {new_name} ({new_symbol}) to config.")
        else:
            st.error("Please enter both name and symbol.")

    st.subheader("❌ Remove Ticker")
    all_symbols = [t["symbol"] for t in config["tickers"]]
    if all_symbols:
        remove_choice = st.selectbox("Select Ticker to Remove", all_symbols)
        if st.button("Remove Ticker"):
            config["tickers"] = [t for t in config["tickers"] if t["symbol"] != remove_choice]
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
            st.warning(f"Removed {remove_choice} from config.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("© 2025 RPStockInsight | Data: Yahoo Finance | For educational use only.")
