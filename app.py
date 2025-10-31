import streamlit as st
import pandas as pd
import yfinance as yf
import json
from datetime import datetime

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RPStockInsight", layout="wide")
st.title("📊 RPStockInsight")
st.caption("Live NSE/BSE insights • Breakout suggestions • Auto light/dark mode • Editable tickers")

CONFIG_FILE = "config.json"

def load_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=4)

config = load_config()
default_tickers = [t["symbol"] for t in config.get("tickers", [])]

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Dashboard Settings")
    selected_symbol = st.text_input("Enter Stock Symbol (e.g., TCS.NS, INFY.NS)", value="TCS.NS")
    period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)
    interval = st.selectbox("Select Interval", ["1d", "1h", "15m"], index=0)

    st.markdown("---")
    st.subheader("🧭 Admin Area")
    st.info("Local edit panel for tickers (saved in config.json)")
    admin_mode = st.checkbox("Enable Admin Edit Mode", False)

# ---------------- LIVE TICKER SECTION ----------------
st.subheader("📈 Live Tickers")
ticker_container = st.empty()

def get_live_prices(symbols):
    prices = {}
    for sym in symbols:
        try:
            data = yf.Ticker(sym).history(period="1d")
            if not data.empty:
                price = data["Close"].iloc[-1]
                change = data["Close"].pct_change().iloc[-1] * 100
                prices[sym] = (price, change)
        except Exception:
            prices[sym] = (None, None)
    return prices

def display_scrolling_ticker(prices):
    if not prices:
        st.warning("No ticker data available.")
        return
    ticker_text = " | ".join(
        [f"{sym}: ₹{v[0]:.2f} ({v[1]:+.2f}%)" for sym, v in prices.items() if v[0]]
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
st.subheader("🚀 Breakout Suggestions (Top Stocks by Sector)")
st.caption("Short-term and Long-term price targets")

breakout_data = []
for t in config["tickers"]:
    try:
        df = yf.download(t["symbol"], period="3mo", interval="1d", progress=False)
        if not df.empty:
            last = df["Close"].iloc[-1]
            ma20 = df["Close"].rolling(20).mean().iloc[-1]
            if last > ma20 * 1.03:  # breakout condition
                breakout_data.append({
                    "Stock": t["name"],
                    "Sector": t["sector"],
                    "Price (₹)": round(last, 2),
                    "Short Term Target (7-15d)": round(last * 1.05, 2),
                    "Long Term Target (30-60d)": round(last * 1.12, 2)
                })
    except Exception as e:
        print(f"Error fetching {t['symbol']}: {e}")

if breakout_data:
    st.dataframe(pd.DataFrame(breakout_data), use_container_width=True)
else:
    st.warning("No breakout signals found currently.")

# ---------------- STOCK SUMMARY ----------------
st.subheader(f"📊 Stock Summary for {selected_symbol}")
try:
    df = yf.download(selected_symbol, period=period, interval=interval)
    if not df.empty:
        st.line_chart(df["Close"], use_container_width=True)
        st.metric(label="Latest Price", value=f"₹{df['Close'].iloc[-1]:.2f}")
    else:
        st.warning("No data available for the selected symbol.")
except Exception as e:
    st.error(f"Error fetching data: {e}")

# ---------------- ADMIN: TICKER MANAGEMENT ----------------
if admin_mode:
    st.markdown("---")
    st.subheader("🧩 Manage Tickers")

    with st.expander("➕ Add New Ticker"):
        name = st.text_input("Company Name")
        symbol = st.text_input("Symbol (e.g., TCS.NS)")
        sector = st.text_input("Sector")
        if st.button("Add Ticker"):
            if name and symbol:
                config["tickers"].append({
                    "name": name.strip(),
                    "symbol": symbol.strip(),
                    "sector": sector.strip() or "General"
                })
                save_config(config)
                st.success(f"Added {symbol}")
                st.experimental_rerun()

    with st.expander("🗑️ Remove Ticker"):
        symbols = [t["symbol"] for t in config["tickers"]]
        to_remove = st.selectbox("Select ticker to remove", symbols)
        if st.button("Remove Selected"):
            config["tickers"] = [t for t in config["tickers"] if t["symbol"] != to_remove]
            save_config(config)
            st.warning(f"Removed {to_remove}")
            st.experimental_rerun()

    st.caption("Changes are auto-saved to config.json")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("© 2025 RPStockInsight | Data from Yahoo Finance | For educational use only.")
