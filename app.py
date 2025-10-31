"""
RPStockInsight - final app.py (v2.6)
Features:
 - Pinned NSE/BSE tickers (auto-refresh via cache TTL=60s)
 - Built-in tickers list (creates tickers.csv on first run)
 - Sidebar to add/remove/save tickers (persists to tickers.csv)
 - Breakout detection by sector (top 5-10 suggestions)
 - Simple short/long predictions (LinearRegression)
 - Local auth (users.json) with roles: superadmin/admin/user
 - Robust error handling for yfinance issues
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime
from io import StringIO

# ---------- CONFIG --------------
APP_TITLE = "RPStockInsight"
CACHE_TTL = 60  # seconds for fetch caching (auto-refresh)
TICKERS_FILE = "tickers.csv"
USERS_FILE = "users.json"
# Default users (first-run). Change passwords immediately after deploy.
DEFAULT_USERS = {
    "superadmin": {"password": "SuperAdmin@123", "role": "superadmin"},
    "admin": {"password": "Admin@123", "role": "admin"},
    "tester": {"password": "Test@123", "role": "user"}
}

# Embedded starter ticker list (symbol, sector). This will be written to tickers.csv on first run.
DEFAULT_TICKERS = [
    ("RELIANCE.NS","Oil & Gas - Refining & Marketing"),
    ("HDFCBANK.NS","Private Banks"),
    ("BHARTIARTL.NS","Telecom Services"),
    ("TCS.NS","IT Services & Consulting"),
    ("ICICIBANK.NS","Private Banks"),
    ("SBIN.NS","Public Banks"),
    ("BAJFINANCE.NS","Consumer Finance"),
    ("INFY.NS","IT Services & Consulting"),
    ("HINDUNILVR.NS","FMCG - Household Products"),
    ("LICI.NS","Insurance"),
    ("LT.NS","Construction & Engineering"),
    ("ITC.NS","FMCG - Tobacco"),
    ("MARUTI.NS","Four Wheelers"),
    ("KOTAKBANK.NS","Private Banks"),
    ("M&M.NS","Four Wheelers"),
    ("HCLTECH.NS","IT Services & Consulting"),
    ("SUNPHARMA.NS","Pharmaceuticals"),
    ("AXISBANK.NS","Private Banks"),
    ("ULTRACEMCO.NS","Cement"),
    ("BAJAJFINSV.NS","Insurance"),
    ("TITAN.NS","Precious Metals, Jewellery & Watches"),
    ("ONGC.NS","Oil & Gas - Exploration & Production"),
    ("ADANIPORTS.NS","Ports"),
    ("HAL.NS","Aerospace & Defense Equipments"),
    ("BEL.NS","Electronic Equipments"),
    ("JSWSTEEL.NS","Iron & Steel"),
    ("POWERGRID.NS","Power Transmission & Distribution"),
    ("DMART.NS","Retail - Department Stores"),
    ("WIPRO.NS","IT Services & Consulting"),
    ("BAJAJ-AUTO.NS","Two Wheelers"),
    ("NESTLEIND.NS","FMCG - Foods"),
    ("ASIANPAINT.NS","Paints"),
    ("COALINDIA.NS","Mining - Coal"),
    ("IOC.NS","Oil & Gas - Refining & Marketing"),
    ("TATASTEEL.NS","Iron & Steel"),
    ("INDIGO.NS","Airlines"),
    ("HINDZINC.NS","Mining - Diversified"),
    ("GRASIM.NS","Cement"),
    ("VEDL.NS","Metals - Diversified"),
    ("SBILIFE.NS","Insurance"),
    ("JIOFINANCE.NS","Consumer Finance"),
    ("HYUNDAI.NS","Four Wheelers"),
    ("HINDALCO.NS","Metals - Aluminium"),
    ("DLF.NS","Real Estate"),
    ("EICHERMOT.NS","Trucks & Buses"),
    ("DIVISLAB.NS","Labs & Life Sciences Services"),
    ("LTIMINDTREE.NS","IT Services & Consulting"),
    ("TVSMOTOR.NS","Two Wheelers"),
    ("VBL.NS","Soft Drinks"),
    ("IRFC.NS","Specialized Finance"),
    ("HDFCLIFE.NS","Insurance"),
    ("BPCL.NS","Oil & Gas - Refining & Marketing"),
    ("PIDILITIND.NS","Diversified Chemicals"),
    ("CHOLAFIN.NS","Consumer Finance"),
    ("BRITANNIA.NS","FMCG - Foods"),
    ("BANKBARODA.NS","Public Banks"),
    ("TECHM.NS","IT Services & Consulting"),
    ("AMBUJACEM.NS","Cement"),
    ("SHRIRAMFIN.NS","Consumer Finance"),
    ("BAJAJHLDNG.NS","Asset Management"),
    ("PNB.NS","Public Banks"),
    ("PFC.NS","Specialized Finance"),
    ("TATAPOWER.NS","Power Transmission & Distribution"),
    ("MUTHOOTFIN.NS","Consumer Finance"),
    ("SOLARINDS.NS","Commodity Chemicals"),
    ("CIPLA.NS","Pharmaceuticals"),
    ("TORNTPHARM.NS","Pharmaceuticals"),
    ("CUMMINSIND.NS","Industrial Machinery"),
    ("CANBK.NS","Public Banks"),
    ("GAIL.NS","Gas Distribution"),
    ("POLYCAB.NS","Electrical Components & Equipments"),
    ("TATACONSUM.NS","Tea & Coffee"),
    ("CGPOWER.NS","Heavy Electrical Equipments"),
    ("INDIANB.NS","Public Banks"),
    ("HDFCAMC.NS","Asset Management"),
    ("MAXHEALTH.NS","Hospitals & Diagnostic Centres"),
    ("SIEMENS.NS","Conglomerates"),
    ("HEROMOTOCO.NS","Two Wheelers"),
    ("BOSCHLTD.NS","Auto Parts"),
    ("JINDALSTEL.NS","Iron & Steel"),
    ("UNIONBANK.NS","Public Banks"),
    ("INDHOTEL.NS","Hotels, Resorts & Cruise Lines"),
    ("SHREECEM.NS","Cement"),
    ("UBL.NS","Alcoholic Beverages"),
    ("Mankind.NS","Pharmaceuticals")  # fallback entries; you can edit later
]

# ---------- Helpers --------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.write("RPStockInsight — live tickers, breakout suggestions, and simple predictions")

# Create default tickers file on first run
def ensure_tickers_file():
    try:
        df = pd.read_csv(TICKERS_FILE)
    except Exception:
        df = pd.DataFrame(DEFAULT_TICKERS, columns=["symbol","sector"])
        df.to_csv(TICKERS_FILE, index=False)
    return df

# Create default users file on first run
def ensure_users_file():
    try:
        with open(USERS_FILE,'r') as f:
            u = json.load(f)
    except Exception:
        with open(USERS_FILE,'w') as f:
            json.dump(DEFAULT_USERS, f, indent=2)
        u = DEFAULT_USERS.copy()
    return u

# Load tickers into dataframe
tickers_df = ensure_tickers_file()
users = ensure_users_file()

# --------- caching yfinance fetch ------------
@st.cache_data(ttl=CACHE_TTL)
def fetch_prices(symbols, period="1mo", interval="1d"):
    """Fetch price data for a list of symbols. Returns dict: symbol -> dataframe or None"""
    out = {}
    if not symbols:
        return out
    # yfinance.download can accept list but handle exceptions gracefully
    try:
        raw = yf.download(symbols, period=period, interval=interval, progress=False, threads=True)
        # If only single ticker, raw is normal DataFrame
        if isinstance(symbols, str) or (isinstance(symbols, list) and len(symbols)==1):
            # ensure mapping
            if isinstance(raw, pd.DataFrame):
                out[symbols if isinstance(symbols,str) else symbols[0]] = raw.reset_index()
            else:
                out[symbols if isinstance(symbols,str) else symbols[0]] = None
            return out
        # multiple tickers -> multiindex columns likely
        if raw is None or raw.empty:
            # per-symbol fallback
            for s in symbols:
                try:
                    df = yf.download(s, period=period, interval=interval, progress=False)
                    out[s] = df.reset_index() if (df is not None and not df.empty) else None
                except Exception:
                    out[s] = None
            return out
        # if multiindex columns
        if isinstance(raw.columns, pd.MultiIndex):
            for s in symbols:
                try:
                    cols = raw.xs(s, axis=1, level=1, drop_level=False)
                    # Build standardized dataframe
                    df = pd.DataFrame({
                        "Date": raw.index,
                        "Open": raw[("Open", s)],
                        "High": raw[("High", s)],
                        "Low": raw[("Low", s)],
                        "Close": raw[("Close", s)],
                        "Adj Close": raw[("Adj Close", s)] if ("Adj Close", s) in raw.columns else raw[("Close", s)],
                        "Volume": raw[("Volume", s)]
                    }).reset_index(drop=True)
                    out[s] = df
                except Exception:
                    out[s] = None
        else:
            # Not MultiIndex - try to map by symbol (may be single)
            for s in symbols:
                try:
                    df = raw.copy()
                    df['Date'] = df.index
                    out[s] = df.reset_index(drop=True)
                except Exception:
                    out[s] = None
    except Exception as e:
        # fallback to per-symbol download
        for s in symbols:
            try:
                df = yf.download(s, period=period, interval=interval, progress=False)
                out[s] = df.reset_index() if (df is not None and not df.empty) else None
            except Exception:
                out[s] = None
    return out

# ---------- Analysis helpers ----------
def compute_breakout(df, lookback=60, volume_mult=1.5):
    """Return dict with breakout boolean and metrics"""
    try:
        if df is None or df.empty or 'Close' not in df.columns:
            return None
        d = df.copy().reset_index(drop=True)
        if 'Volume' not in d.columns:
            d['Volume'] = 0
        d['rolling_high'] = d['Close'].rolling(lookback).max()
        d['rolling_low'] = d['Low'].rolling(lookback).min()
        d['avg_vol'] = d['Volume'].rolling(lookback).mean()
        if len(d) < 2:
            return None
        last = d.iloc[-1]
        prev_high = d['rolling_high'].iloc[-2] if len(d) > 1 else np.nan
        try:
            vol_ok = (last['Volume'] >= (last['avg_vol'] * volume_mult)) if not np.isnan(last['avg_vol']) else False
        except Exception:
            vol_ok = False
        breakout = False
        try:
            breakout = (last['Close'] > prev_high) and vol_ok
        except Exception:
            breakout = False
        entry = float(last['Close'])
        stop = float(d['rolling_low'].iloc[-1]) if not np.isnan(d['rolling_low'].iloc[-1]) else None
        rr = (entry - stop) if stop is not None else None
        target = round(entry + 1.5 * rr, 2) if rr is not None else None
        return {"breakout": breakout, "entry": entry, "stop": stop, "target": target, "close": entry, "volume": float(last['Volume']), "avg_vol": float(last['avg_vol']) if not np.isnan(last['avg_vol']) else None}
    except Exception:
        return None

def simple_predict(df):
    """Linear regression extrapolation - short (7) and long (30) days"""
    try:
        from sklearn.linear_model import LinearRegression
    except Exception:
        return {"short": None, "long": None, "error": "sklearn missing"}
    if df is None or df.empty or 'Close' not in df.columns:
        return {"short": None, "long": None, "error": "no data"}
    d = df.dropna(subset=['Close']).reset_index(drop=True)
    N = min(60, len(d))
    if N < 5:
        return {"short": None, "long": None, "error": "not enough data"}
    d = d.tail(N).reset_index(drop=True)
    d['t'] = np.arange(len(d))
    X = d[['t']].values
    y = d['Close'].values
    m = LinearRegression().fit(X, y)
    last_t = X[-1, 0]
    pred_short = float(m.predict([[last_t + 7]])[0])
    pred_long = float(m.predict([[last_t + 30]])[0])
    return {"short": round(pred_short,2), "long": round(pred_long,2), "error": None}

# ---------- UI: Sidebar ----------
with st.sidebar:
    st.header("Settings & Ticker Management")
    # Auth area
    st.subheader("Login / Register (local)")
    auth_mode = st.selectbox("Action", ["Login", "Register"], index=0)
    if auth_mode == "Login":
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            try:
                with open(USERS_FILE,'r') as f:
                    u = json.load(f)
            except Exception:
                u = DEFAULT_USERS.copy()
            if username in u and u[username]["password"] == password:
                st.success(f"Welcome {username} ({u[username]['role']})")
                st.session_state["user"] = username
                st.session_state["role"] = u[username]["role"]
            else:
                st.error("Invalid credentials")
    else:
        st.info("Register (Only superadmin may register new users)")
        r_user = st.text_input("New username", key="reg_user")
        r_pass = st.text_input("New password", key="reg_pass", type="password")
        r_role = st.selectbox("Role", ["user","admin"], key="reg_role")
        if st.button("Register User"):
            # Allow register if no users or current is superadmin
            try:
                with open(USERS_FILE,'r') as f:
                    u = json.load(f)
            except Exception:
                u = DEFAULT_USERS.copy()
            allow = ("role" in st.session_state and st.session_state.get("role")=="superadmin") or (len(u)==0)
            if not allow:
                st.error("Only superadmin may create users.")
            else:
                u[r_user] = {"password": r_pass, "role": r_role}
                with open(USERS_FILE,'w') as f:
                    json.dump(u,f,indent=2)
                st.success("User created")

    st.markdown("---")
    st.subheader("Manage Tickers (add/remove)")
    # load tickers from file
    try:
        tickers_df = pd.read_csv(TICKERS_FILE)
    except Exception:
        tickers_df = pd.DataFrame(DEFAULT_TICKERS, columns=["symbol","sector"])
        tickers_df.to_csv(TICKERS_FILE,index=False)
    st.write(f"Tickers loaded: {len(tickers_df)}")
    new_symbol = st.text_input("Add ticker symbol (e.g. TCS.NS)", key="add_sym")
    new_sector = st.text_input("Sector for new symbol", key="add_sector")
    if st.button("Add ticker"):
        if new_symbol:
            tickers_df = pd.concat([tickers_df, pd.DataFrame([[new_symbol.strip(), new_sector.strip() if new_sector else "Other"]], columns=["symbol","sector"])], ignore_index=True)
            tickers_df.to_csv(TICKERS_FILE,index=False)
            st.success(f"Added {new_symbol.strip()}")
    # remove
    rem_symbol = st.text_input("Remove ticker symbol (exact)", key="rem_sym")
    if st.button("Remove ticker"):
        before = len(tickers_df)
        tickers_df = tickers_df[ tickers_df['symbol'].str.upper() != rem_symbol.strip().upper() ]
        tickers_df.to_csv(TICKERS_FILE,index=False)
        st.success(f"Removed {rem_symbol.strip()}. {before - len(tickers_df)} removed.")
    if st.button("Reset to defaults"):
        pd.DataFrame(DEFAULT_TICKERS, columns=["symbol","sector"]).to_csv(TICKERS_FILE,index=False)
        st.success("Reset tickers to defaults")
    st.markdown("---")
    st.subheader("Quick controls")
    period = st.selectbox("Price period", ["1mo","3mo","6mo","1y"], index=1)
    interval = st.selectbox("Interval", ["1d","1wk"], index=0)
    st.write("Auto-refresh interval (cache TTL): 60s (fixed)")

# ---------- Top pinned tickers ----------
# load tickers & group by sector
tickers_df = pd.read_csv(TICKERS_FILE)
# show pinned top NSE + BSE (we'll show small font, auto-scroll)
def build_pinned_html(rows, label):
    parts=[]
    for r in rows:
        sym=r['symbol']
        price=r.get('price',0.0)
        pct=r.get('pct',0.0)
        arrow = "▲" if pct>=0 else "▼"
        parts.append(f"<span style='margin-right:14px;padding:3px 6px;display:inline-block'>{sym} {price:.2f} {arrow} {pct:+.2f}%</span>")
    return f"<div style='white-space:nowrap;overflow:auto;font-size:13px'><strong style='margin-right:10px'>{label}:</strong>" + " ".join(parts) + "</div>"

# fetch live price for top few (to avoid too slow UI, we show up to 30 in pinned area)
pinned_symbols = tickers_df['symbol'].tolist()[:40]
price_map = {}
if pinned_symbols:
    fetched = fetch_prices(pinned_symbols, period="5d", interval="1d")
    for s in pinned_symbols:
        df = fetched.get(s)
        if df is None or df.empty or 'Close' not in df.columns:
            price_map[s] = {"price":0.0,"pct":0.0}
        else:
            last = df.iloc[-1]
            prev = df['Close'].iloc[-2] if len(df)>1 else last['Close']
            price = float(last['Close'])
            pct = ((price - prev)/prev*100) if prev!=0 else 0.0
            price_map[s] = {"price":price,"pct":pct}
# build NSE and BSE lines (we'll show same symbols as NSE but user can map to .BO later if desired)
nse_rows = []
bse_rows = []
for s in pinned_symbols:
    p = price_map.get(s,{"price":0.0,"pct":0.0})
    nse_rows.append({"symbol":s, "price":p['price'], "pct":p['pct']})
    # naive BSE mapping: replace .NS with .BO
    b_sym = s.replace(".NS",".BO")
    bse_rows.append({"symbol":b_sym, "price":p['price'], "pct":p['pct']})

col1, col2 = st.columns([8,1])
with col1:
    st.markdown(build_pinned_html(nse_rows, "NSE"), unsafe_allow_html=True)
    st.markdown(build_pinned_html(bse_rows, "BSE"), unsafe_allow_html=True)
with col2:
    st.caption(f"Refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("---")

# ---------- Main: Watchlist analysis & Breakouts by sector ----------
st.header("Breakout Suggestions (by sector)")
# group tickers by sector
sectors = tickers_df['sector'].fillna("Other").unique().tolist()
col_left, col_right = st.columns([3,2])
with col_left:
    st.write("Select sector to analyze:")
    sel_sector = st.selectbox("Sector", ["All"] + sorted(sectors))
    max_per_sector = st.slider("Top N per sector (suggestions)", min_value=3, max_value=15, value=7)

# Build analysis list
if sel_sector == "All":
    analysis_symbols = tickers_df['symbol'].tolist()
else:
    analysis_symbols = tickers_df[tickers_df['sector']==sel_sector]['symbol'].tolist()

if st.button("Analyze watchlist for breakouts"):
    with st.spinner("Fetching data and analyzing..."):
        fetched = fetch_prices(analysis_symbols, period="3mo", interval="1d")
        # collect breakout candidates per sector
        rows_all = []
        for idx, row in tickers_df.iterrows():
            s = row['symbol']
            if s not in fetched:
                continue
            df = fetched.get(s)
            if df is None or df.empty:
                continue
            br = compute_breakout(df, lookback=60, volume_mult=1.5)
            preds = simple_predict(df)
            if br and br.get("breakout"):
                rows_all.append({
                    "symbol": s,
                    "sector": row['sector'],
                    "entry": br.get("entry"),
                    "stop": br.get("stop"),
                    "target": br.get("target"),
                    "volume": br.get("volume"),
                    "avg_vol": br.get("avg_vol"),
                    "pred_short": preds.get("short"),
                    "pred_long": preds.get("long")
                })
        if not rows_all:
            st.info("No breakout candidates detected by simple rules.")
        else:
            df_out = pd.DataFrame(rows_all)
            # group by sector and show top N by volume
            grouped = df_out.groupby("sector")
            for sector_name, grp in grouped:
                st.subheader(f"{sector_name} — Top {max_per_sector}")
                # rank by volume desc
                topn = grp.sort_values("volume", ascending=False).head(max_per_sector)
                st.dataframe(topn.reset_index(drop=True))
            # Download
            csv = df_out.to_csv(index=False).encode('utf-8')
            st.download_button("Download all breakout suggestions", data=csv, file_name="breakout_suggestions.csv", mime="text/csv")

# Quick search single symbol
st.markdown("---")
st.subheader("Quick Symbol Lookup")
sym_q = st.text_input("Symbol (e.g. TCS.NS)", "")
if st.button("Fetch symbol") and sym_q:
    with st.spinner("Fetching..."):
        fetched = fetch_prices([sym_q], period="6mo", interval="1d")
        df = fetched.get(sym_q)
        if df is None or df.empty:
            st.warning("No data found for symbol.")
        else:
            st.dataframe(df.tail(10))
            br = compute_breakout(df, lookback=60)
            preds = simple_predict(df)
            st.write("Prediction:", preds)
            st.write("Breakout:", br)

# Footer & admin notes
st.markdown("---")
st.caption("© RPStockInsight — Test mode. Not financial advice.")
if "role" in st.session_state and st.session_state.get("role")=="superadmin":
    st.info("You are superadmin. You can edit tickers and users.json directly in the workspace.")

