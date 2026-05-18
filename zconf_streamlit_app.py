#!/usr/bin/env python3
"""
Shantanu's VALUE MOMENTUM SWING TRADING SCANNER — Streamlit Web App
Based on ZConf Screener v1.6 (NSE Swing Trading)

To run locally:
    pip install streamlit yfinance pandas numpy reportlab plotly
    streamlit run zconf_streamlit_app.py

To deploy free on Streamlit Community Cloud:
    1. Push this file + requirements.txt to a GitHub repo
    2. Go to https://share.streamlit.io → Connect repo → Deploy
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import warnings
import io

warnings.filterwarnings("ignore")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

try:
    from reportlab.lib.pagesizes import landscape, A4
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                    Paragraph, Spacer, PageBreak)
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import cm
    _REPORTLAB = True
except ImportError:
    _REPORTLAB = False

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VMS Scanner — NSE Swing Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.stApp {
    background: #0b0f1a;
    color: #e8eaf6;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 1px solid #1e293b;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #131c2e, #1a2540);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
}
.metric-label {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-family: 'DM Mono', monospace;
}
.metric-value {
    font-size: 28px;
    font-weight: 800;
    font-family: 'Syne', sans-serif;
    margin-top: 4px;
}
.metric-green  { color: #34d399; }
.metric-red    { color: #f87171; }
.metric-blue   { color: #60a5fa; }
.metric-yellow { color: #fbbf24; }

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 800;
    letter-spacing: 0.5px;
    padding: 8px 0 6px 0;
    border-bottom: 2px solid #1e3a5f;
    margin-bottom: 12px;
}
.buy-header  { color: #34d399; }
.sell-header { color: #f87171; }
.conf-header { color: #60a5fa; }
.div-header  { color: #fbbf24; }
.bt-header   { color: #a78bfa; }

/* Table styling */
.stDataFrame { border-radius: 10px; overflow: hidden; }
[data-testid="stDataFrame"] table {
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
}

/* Status pills */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
}
.pill-strong-buy  { background: #065f46; color: #6ee7b7; }
.pill-buy         { background: #064e3b; color: #34d399; }
.pill-sell        { background: #7f1d1d; color: #fca5a5; }
.pill-strong-sell { background: #991b1b; color: #f87171; }
.pill-neutral     { background: #1e293b; color: #94a3b8; }

/* Scan button */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border: none !important;
    padding: 12px 32px !important;
    border-radius: 10px !important;
    width: 100%;
    letter-spacing: 0.5px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(37, 99, 235, 0.4);
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #1d4ed8, #34d399) !important;
    border-radius: 4px;
}

/* Info box */
.info-box {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-left: 4px solid #2563eb;
    border-radius: 8px;
    padding: 14px 18px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #94a3b8;
    line-height: 1.8;
}

/* Tab style override */
[data-baseweb="tab-list"] {
    background: #111827 !important;
    border-radius: 10px !important;
    gap: 4px !important;
}
[data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  DEFAULT CONFIG
# ══════════════════════════════════════════════════════════════
DEFAULT_CFG = {
    "use_cape": True, "cape_zlen": 252, "cape_bearish": True, "cape_max_q": 8,
    "vwap_len": 100, "vwap_zlen": 100, "vwap_contrarian": True,
    "rsi_len": 14, "rsi_zlen": 100, "rsi_contrarian": True,
    "rsi_dz_len": 5, "rsi_dz_weight": 0.4,
    "macd_fast": 12, "macd_slow": 26, "macd_sig": 9,
    "macd_zlen": 100, "macd_dz_len": 5, "macd_dz_weight": 0.5,
    "wt_cape": 25.0, "wt_vwap": 25.0, "wt_rsi": 25.0, "wt_macd": 25.0,
    "th_sbuy": 2.0, "th_buy": 1.0, "th_sell": -1.0, "th_ssell": -2.0,
    "clamp_val": 3.0, "min_composite": 1.0, "workers": 8,
    "div_enable": True, "piv_left": 5, "piv_right": 5, "div_lookback": 60,
    "conf_strong": 1.75, "conf_moderate": 1.10,
    "dz_accel_enable": True, "dz_accel_bars": 2, "dz_accel_require_both": False,
    "rsi_hard_max": 50.0,
    "candle_body_enable": True, "candle_body_hard": False,
    "candle_green_tol": 0.998, "hammer_mult": 2.0,
    "vwap_z_hard_min": 1.5,
    "add_conf_agree_min": 2,
    "bt_min_composite": 1.25,
    "hi52_enable": True, "hi52_bars": 252, "hi52_pct": 0.85,
    "backtest_profit_pct": 8.0,
}

# ══════════════════════════════════════════════════════════════
#  STOCK UNIVERSE
# ══════════════════════════════════════════════════════════════
_MPTDS_26 = [
    "NESTLEIND","IEX","IRCTC","ABBOTINDIA","TRITURBINE","BEL","INFY","ITC",
    "GRSE","CUMMINSIND","ASTRAZEN","NBCC","APARINDS","CRISIL","AJANTPHARM",
    "PERSISTENT","HEROMOTOCO","PIDILITIND","EICHERMOT","POLYCAB",
    "VOLTAMP","LTTS","SCHAEFFLER","TORNTPHARM","CHAMBLFERT","UNITDSPR",
    "GODFRYPHLP","BLUESTARCO","GABRIEL","JBCHEPHARM","ASIANPAINT","HAVELLS",
    "BERGEPAINT","ZYDUSLIFE","AVANTIFEED","ICICIGI","COROMANDEL","MGL",
    "MPHASIS","APLAPOLLO","ZENSARTECH","BSOFT","TIMKEN","GRINDWELL","ALKEM",
    "COFORGE","TITAN","SUNDRMFAST","GODREJAGRO","BPCL","FSL","TVSMOTOR",
    "ASHOKLEY","THANGAMAYL","AEGISLOG","CCL","HATSUN","APLLTD","POWERGRID",
    "BAJFINANCE","RECLTD","PFC",
]
_MPTDS_25 = [
    "ABB","ABBOTINDIA","ACE","ADANIENT","ADANIPORTS","ABCAPITAL","AJANTPHARM",
    "AKZOINDIA","ALKEM","ANANDRATHI","ANANTRAJ","ANGELONE","APARINDS","APLAPOLLO",
    "APOLLOHOSP","ASHOKLEY","ASTRAL","ASTRAZEN","AUBANK","BAJFINANCE","BAJAJFINSV",
    "BASF","BEL","BSOFT","BLUESTARCO","BRIGADE","BRITANNIA","BSE","CDSL","CAMS",
    "CANFINHOME","CAPLIPOINT","CARBORUNIV","CASTROLIND","CESC","CHOLAHLDNG",
    "CHOLAFIN","CIPLA","CUB","COALINDIA","COCHINSHIP","COFORGE","COLPAL",
    "CONCORDBIO","COROMANDEL","CRISIL","CUMMINSIND","DABUR","DIXON","LALPATHLAB",
    "DRREDDY","EICHERMOT","ELECON","ELGIEQUIP","ENDURANCE","ERIS","ESCORTS",
    "FSL","GABRIEL","GRSE","GODFRYPHLP","GODREJIND","GRASIM","GRAVITA","HUDCO",
    "HAVELLS","HBLENGINE","HCLTECH","HDFCAMC","HDFCBANK","HEROMOTOCO","HSCL",
    "HAL","HINDALCO","HINDCOPPER","HINDZINC","POWERINDIA","HOMEFIRST","IRCTC",
    "ICICIBANK","ICICIGI","IEX","INDHOTEL","NAUKRI","INFY","INGERRAND","IPCALAB",
    "IRCON","ITC","JBCHEPHARM","JBMA","JINDALSAW","JKCEMENT",
    "KARURVYSYA","KEI","KOTAKBANK","KPITTECH","LTFOODS","LTTS","LT","LICHSGFIN",
    "M&M","MARUTI","MAZDOCK","MINDSPACE","MOTILALOFS","MPHASIS","MUTHOOTFIN",
    "NH","NCC","NEULANDLAB","NEWGEN","NAM-INDIA","OFSS","PIIND","PAGEIND",
    "PCBL","PIDILITIND","POLYMED","POLYCAB","POWERGRID","RATNAMANI","REDINGTON",
    "RELIANCE","SBIN","SBILIFE","SCHAEFFLER","SHRIRAMFIN","SHRIPISTON","SOLARINDS",
    "SUNDRMFAST","SUPREMEIND","TARIL","TATACONSUM","TATAPOWER",
    "TCS","TECHNOE","THERMAX","TIMETECHNO","TITAN","TORNTPHARM","TRITURBINE",
    "TIINDIA","TVSHLTD","TVSMOTOR","UBL","UNOMINDA","USHAMART","UTIAMC",
    "VGUARD","VBL","VESUVIUS","WELCORP","ZENTEC","ZENSARTECH","ZYDUSLIFE",
]
_MDSPORT_26 = [
    "SANOFICONR","ICICIAMC","TCS","INGERRAND","ANANDRATHI","CAMS",
    "IEX","IRCTC","COALINDIA","BSE","ABBOTINDIA","CDSL","TRITURBINE",
    "NAM-INDIA","OFSS","INFY","ZENTEC","ITC","CUMMINSIND","TATAELXSI",
    "INDIAMART","NATCOPHARM","EMAMILTD","AJANTPHARM","HCLTECH","PERSISTENT",
    "HEROMOTOCO","ABB","PIDILITIND","NMDC","LALPATHLAB","CONCORDBIO",
    "LTTS","ECLERX","SCHAEFFLER","HBLENGINE","FINEORG","MANYAVAR",
    "CAPLIPOINT","JBCHEPHARM","SHRIPISTON","AVANTIFEED","PIIND","CIPLA",
    "ELGIEQUIP","RATNAMANI","VESUVIUS","ZENSARTECH","TIMKEN","GRINDWELL",
    "DIVISLAB","SUNTV","ALKEM","SUNPHARMA","POLYMED","AIAENG","USHAMART",
    "SONACOMS","OBEROIRLTY","INDHOTEL","AFFLE","GALLANTT",
]
_MPTDS_24 = [
    "360ONE","ABB","ABBOTINDIA","ACE","AEGISLOG","AFFLE","AIAENG",
    "AJANTPHARM","AKZOINDIA","ALKEM","APARINDS","APLAPOLLO",
    "APOLLOHOSP","APOLLOTYRE","ASIANPAINT","ASTRAL","ASTRAZEN",
    "BAJAJFINSV","BAJAJ-AUTO","BAJFINANCE","BASF","BEL","BSOFT","CAMS",
    "CANFINHOME","CAPLIPOINT","CARBORUNIV","CDSL","CESC","CHOLAFIN","CIPLA",
    "COALINDIA","COCHINSHIP","COFORGE","COLPAL","COROMANDEL","CRISIL",
    "CUMMINSIND","DABUR","DATAPATTNS","DIXON","DRREDDY","EICHERMOT",
    "ELECON","ELGIEQUIP","ENDURANCE","ESCORTS","FSL","GABRIEL","GRSE",
    "GODFRYPHLP","GODREJIND","GRASIM","GRINDWELL","HAVELLS","HCLTECH",
    "HDFCAMC","HDFCBANK","HEROMOTOCO","HINDUNILVR","HSCL","HUDCO",
    "ICICIBANK","INDHOTEL","INFY","IRCTC","ITC","JBCHEPHARM","JBMA",
    "JINDALSAW","JKCEMENT","KARURVYSYA","KEI","KOTAKBANK","KPITTECH",
    "LALPATHLAB","LTTS","LT","MANAPPURAM","MARICO","MARUTI","MAZDOCK",
    "MPHASIS","NATCOPHARM","NCC","NESTLEIND","NEWGEN","OFSS","PIIND",
    "PAGEIND","PIDILITIND","POLYMED","POLYCAB","POWERGRID","RATNAMANI",
    "RELIANCE","SBIN","SBILIFE","SCHAEFFLER","SHRIRAMFIN","SOLARINDS",
    "SUNDRMFAST","SUPREMEIND","TATACONSUM","TATAELXSI","TCS","TECHNOE",
    "THERMAX","TIINDIA","TITAN","TORNTPHARM","TRITURBINE","TVSMOTOR",
    "UBL","UNOMINDA","VBL","ZENSARTECH","ZYDUSLIFE",
]

_CORRECTIONS = {
    "M&M": "M%26M", "M_M": "M&M", "BAJAJ_AUTO": "BAJAJ-AUTO",
    "ARE_M": "ARE&M", "NAM-INDIA": "NAM-INDIA", "NAM_INDIA": "NAM-INDIA",
}

ALL_SYMBOLS = sorted(set(_MPTDS_26 + _MPTDS_25 + _MDSPORT_26 + _MPTDS_24))
INITIAL_CAPITAL = 1_000_000
POSITION_SIZE   =    50_000
MAX_POSITIONS   = INITIAL_CAPITAL // POSITION_SIZE

_INDIA_CPI = {
    2013: 10.9, 2014: 6.4, 2015: 4.9, 2016: 4.5, 2017: 3.3,
    2018: 3.9,  2019: 3.7, 2020: 6.6, 2021: 5.1, 2022: 6.7,
    2023: 5.4,  2024: 4.8, 2025: 4.9, 2026: 4.5,
}


def to_yf(sym: str) -> str:
    return _CORRECTIONS.get(sym, sym) + ".NS"


# ══════════════════════════════════════════════════════════════
#  INDICATOR MATH
# ══════════════════════════════════════════════════════════════

def _zscore(s, n):
    m  = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std(ddof=1)
    return (s - m) / sd.replace(0.0, np.nan)

def _clamp(s, v):
    return s.clip(-v, v)

def _wilder_rma(s, n):
    alpha  = 1.0 / n
    result = np.full(len(s), np.nan)
    valid  = s.dropna()
    if len(valid) < n:
        return pd.Series(result, index=s.index)
    pos      = s.index.get_loc(valid.index[0])
    seed_end = pos + n
    if seed_end > len(s):
        return pd.Series(result, index=s.index)
    result[seed_end - 1] = s.iloc[pos:seed_end].mean()
    for i in range(seed_end, len(s)):
        val = s.iloc[i]
        result[i] = (alpha * val + (1 - alpha) * result[i - 1]
                     if not np.isnan(val) else result[i - 1])
    return pd.Series(result, index=s.index)

def _rsi(close, n):
    d    = close.diff()
    gain = _wilder_rma(d.clip(lower=0), n)
    loss = _wilder_rma((-d).clip(lower=0), n)
    rs   = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

def _cpi_factor(from_year, to_year):
    if to_year <= from_year:
        return 1.0
    factor = 1.0
    for yr in range(from_year, to_year):
        factor *= 1.0 + _INDIA_CPI.get(yr, 5.5) / 100.0
    return max(factor, 1.0)

_USD_INR_CACHE = [None]

def _get_usd_inr():
    if _USD_INR_CACHE[0] is None:
        try:
            rate = yf.Ticker("USDINR=X").info.get("regularMarketPrice", None)
            _USD_INR_CACHE[0] = float(rate) if rate and 60 < float(rate) < 120 else 84.0
        except Exception:
            _USD_INR_CACHE[0] = 84.0
    return _USD_INR_CACHE[0]

def _eps_inr_factor(tk):
    try:
        info      = tk.info
        fin_ccy   = info.get("financialCurrency", "INR")
        price_ccy = info.get("currency", "INR")
        if fin_ccy == "USD" and price_ccy == "INR":
            return _get_usd_inr()
    except Exception:
        pass
    return 1.0

def _get_eps_series(tk):
    fx = _eps_inr_factor(tk)
    eps_map = {}
    for attr in ["quarterly_income_stmt", "quarterly_financials"]:
        try:
            q = getattr(tk, attr)
            if q is None or q.empty:
                continue
            for row_name in ["Diluted EPS", "Basic EPS"]:
                if row_name not in q.index:
                    continue
                row = q.loc[row_name]
                for col, val in row.items():
                    try:
                        v = float(val)
                        if not np.isnan(v):
                            eps_map[pd.Timestamp(col).normalize()] = v * fx
                    except (TypeError, ValueError):
                        pass
                if eps_map:
                    break
            if eps_map:
                break
        except Exception:
            pass
    if not eps_map:
        return pd.Series(dtype=float)
    return pd.Series(eps_map).sort_index()


def _compute_cape_z_series(tk, price_df, c=None):
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    eps_s = _get_eps_series(tk)
    if eps_s.empty:
        return pd.Series(dtype=float)
    price_idx = price_df.index
    if hasattr(price_idx, "tz") and price_idx.tz is not None:
        price_idx = price_idx.tz_convert(None)
    close = pd.Series(price_df["Close"].astype(float).values, index=price_idx)
    eps_s = eps_s.sort_index()
    n_total = len(eps_s)
    ttm_map = {}
    for i in range(n_total):
        report_date = eps_s.index[i]
        start       = max(0, i + 1 - c["cape_max_q"])
        window      = eps_s.iloc[start : i + 1]
        n           = len(window)
        weights     = np.array([max(0.1, 1.0 - (n - 1 - k) * 0.025) for k in range(n)])
        cpi_adj     = np.array([_cpi_factor(window.index[k].year, report_date.year) for k in range(n)])
        total_w     = weights.sum()
        if total_w <= 0:
            continue
        ttm = (window.values * cpi_adj * weights).sum() / total_w * 4.0
        if ttm > 0:
            ttm_map[report_date] = ttm
    if not ttm_map:
        return pd.Series(dtype=float)
    ttm_s        = pd.Series(ttm_map).sort_index()
    combined_idx = price_idx.union(ttm_s.index).sort_values()
    ttm_daily    = ttm_s.reindex(combined_idx).ffill().reindex(price_idx)
    cape_ratio   = close / ttm_daily.replace(0, np.nan)
    use_len      = min(c["cape_zlen"], len(cape_ratio.dropna()))
    if use_len < 30:
        return pd.Series(dtype=float)
    z = _zscore(cape_ratio, use_len).clip(-c["clamp_val"], c["clamp_val"])
    return (-z if c["cape_bearish"] else z)


def compute_cape_z(tk, price_df, c=None):
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    if not c["use_cape"]:
        return None, None, None
    try:
        z_series = _compute_cape_z_series(tk, price_df, c)
        if z_series.empty:
            return None, None, None
        eps_s = _get_eps_series(tk)
        price_idx = price_df.index
        if hasattr(price_idx, "tz") and price_idx.tz is not None:
            price_idx = price_idx.tz_convert(None)
        close = pd.Series(price_df["Close"].astype(float).values, index=price_idx)
        ttm_s = None
        for attr in ["quarterly_income_stmt", "quarterly_financials"]:
            try:
                q = getattr(tk, attr)
                if q is None or q.empty:
                    continue
                for row_name in ["Diluted EPS", "Basic EPS"]:
                    if row_name not in q.index:
                        continue
                    row = q.loc[row_name]
                    ep = {}
                    for col, val in row.items():
                        try:
                            v = float(val)
                            if not np.isnan(v):
                                ep[pd.Timestamp(col).normalize()] = v * _eps_inr_factor(tk)
                        except (TypeError, ValueError):
                            pass
                    if ep:
                        ttm_s = pd.Series(ep).sort_index()
                        break
                if ttm_s is not None:
                    break
            except Exception:
                pass
        last_z     = float(z_series.iloc[-1]) if not np.isnan(z_series.iloc[-1]) else None
        last_close = float(close.iloc[-1])
        last_ttm   = None
        last_ratio = None
        if ttm_s is not None and not ttm_s.empty:
            last_ttm   = float(ttm_s.iloc[-1])
            last_ratio = round(last_close / last_ttm, 2) if last_ttm and last_ttm > 0 else None
        return last_z, last_ratio, last_ttm
    except Exception:
        return None, None, None


def _pivot_low(series, left, right):
    n = len(series); result = pd.Series(np.nan, index=series.index); vals = series.values
    for i in range(left, n - right):
        v = vals[i]
        if np.isnan(v): continue
        window = vals[i - left : i + right + 1]
        if np.isnan(window).any(): continue
        if v == window.min() and (window == v).sum() == 1:
            result.iloc[i] = v
    return result

def _pivot_high(series, left, right):
    n = len(series); result = pd.Series(np.nan, index=series.index); vals = series.values
    for i in range(left, n - right):
        v = vals[i]
        if np.isnan(v): continue
        window = vals[i - left : i + right + 1]
        if np.isnan(window).any(): continue
        if v == window.max() and (window == v).sum() == 1:
            result.iloc[i] = v
    return result

def _nearest_pivot_value(piv_series, target_idx, tol):
    n = len(piv_series); lo = max(0, target_idx - tol); hi = min(n - 1, target_idx + tol)
    best_val, best_dist = None, None
    for j in range(lo, hi + 1):
        v = piv_series.iloc[j]
        if not np.isnan(v):
            d = abs(j - target_idx)
            if best_dist is None or d < best_dist:
                best_dist, best_val = d, float(v)
    return best_val

def detect_divergence(price, osc, name, c=None):
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    out = {"reg_bull": False, "hid_bull": False, "reg_bear": False, "hid_bear": False, "tag": ""}
    if not c["div_enable"]: return out
    if len(price) < c["piv_left"] + c["piv_right"] + 10: return out
    L, R, LB = c["piv_left"], c["piv_right"], c["div_lookback"]
    osc = osc.reindex(price.index)
    p_lows  = _pivot_low(price, L, R); p_highs = _pivot_high(price, L, R)
    o_lows  = _pivot_low(osc, L, R);  o_highs = _pivot_high(osc, L, R)
    last_idx = len(price) - 1
    pl_positions = [i for i, v in enumerate(p_lows.values) if not np.isnan(v)]
    if len(pl_positions) >= 2:
        i_curr, i_prev = pl_positions[-1], pl_positions[-2]
        if (last_idx - i_curr) <= LB and (i_curr - i_prev) <= LB:
            p_curr, p_prev = price.iloc[i_curr], price.iloc[i_prev]
            o_curr = _nearest_pivot_value(o_lows, i_curr, R)
            o_prev = _nearest_pivot_value(o_lows, i_prev, R)
            if o_curr is not None and o_prev is not None:
                if   p_curr < p_prev and o_curr > o_prev: out["reg_bull"] = True
                elif p_curr > p_prev and o_curr < o_prev: out["hid_bull"] = True
    ph_positions = [i for i, v in enumerate(p_highs.values) if not np.isnan(v)]
    if len(ph_positions) >= 2:
        i_curr, i_prev = ph_positions[-1], ph_positions[-2]
        if (last_idx - i_curr) <= LB and (i_curr - i_prev) <= LB:
            p_curr, p_prev = price.iloc[i_curr], price.iloc[i_prev]
            o_curr = _nearest_pivot_value(o_highs, i_curr, R)
            o_prev = _nearest_pivot_value(o_highs, i_prev, R)
            if o_curr is not None and o_prev is not None:
                if   p_curr > p_prev and o_curr < o_prev: out["reg_bear"] = True
                elif p_curr < p_prev and o_curr > o_prev: out["hid_bear"] = True
    parts = []
    if out["reg_bull"]: parts.append(f"BullReg({name})")
    if out["hid_bull"]: parts.append(f"BullHid({name})")
    if out["reg_bear"]: parts.append(f"BearReg({name})")
    if out["hid_bear"]: parts.append(f"BearHid({name})")
    out["tag"] = " | ".join(parts)
    return out

def _hi52_ok_series(high, close, c=None):
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    roll_max = high.rolling(c["hi52_bars"], min_periods=max(20, c["hi52_bars"] // 4)).max()
    ratio    = close / roll_max.replace(0, np.nan)
    return ratio <= c["hi52_pct"]

def _hi52_ok_last(high, close, c=None):
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    if not c["hi52_enable"]: return True
    s    = _hi52_ok_series(high, close, c)
    last = s.iloc[-1]
    return bool(last) if not pd.isna(last) else False

def compute_signals(df, c=None):
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    min_bars = max(c["vwap_zlen"], c["rsi_zlen"], c["macd_zlen"]) + c["macd_slow"] + 30
    if len(df) < min_bars: return None
    close  = df["Close"].astype(float)
    high   = df["High"].astype(float)
    low    = df["Low"].astype(float)
    volume = df["Volume"].astype(float).replace(0, np.nan)
    tp       = (high + low + close) / 3.0
    vwap_num = (tp * volume).rolling(c["vwap_len"], min_periods=c["vwap_len"] // 2).sum()
    vwap_den = volume.rolling(c["vwap_len"], min_periods=c["vwap_len"] // 2).sum()
    vwap_val = vwap_num / vwap_den.replace(0, np.nan)
    vwap_zraw = _zscore(close - vwap_val, c["vwap_zlen"])
    vwap_z   = _clamp(-vwap_zraw if c["vwap_contrarian"] else vwap_zraw, c["clamp_val"])
    rsi_val  = _rsi(close, c["rsi_len"])
    rsi_lz   = _zscore(rsi_val, c["rsi_zlen"])
    rsi_dz   = rsi_lz.diff(c["rsi_dz_len"]).rolling(2).mean()
    rsi_comb = rsi_lz * (1.0 - c["rsi_dz_weight"]) + rsi_dz * c["rsi_dz_weight"]
    rsi_z    = _clamp(-rsi_comb if c["rsi_contrarian"] else rsi_comb, c["clamp_val"])
    rsi_dz_accel = rsi_dz.diff(c["dz_accel_bars"])
    ema_f    = close.ewm(span=c["macd_fast"], adjust=False, min_periods=c["macd_fast"]).mean()
    ema_s    = close.ewm(span=c["macd_slow"], adjust=False, min_periods=c["macd_slow"]).mean()
    macd_hist = (ema_f - ema_s) - (ema_f - ema_s).ewm(span=c["macd_sig"], adjust=False, min_periods=c["macd_sig"]).mean()
    macd_pct  = macd_hist / close.replace(0, np.nan) * 100.0
    macd_lz   = _zscore(macd_pct, c["macd_zlen"])
    macd_dz   = macd_lz.diff(c["macd_dz_len"]).rolling(2).mean()
    macd_comb = macd_lz * (1.0 - c["macd_dz_weight"]) + macd_dz * c["macd_dz_weight"]
    macd_z    = _clamp(macd_comb, c["clamp_val"])
    macd_dz_accel = macd_dz.diff(c["dz_accel_bars"])
    div_rsi  = detect_divergence(close, rsi_lz,  "RSI_Z", c)
    div_macd = detect_divergence(close, macd_lz, "MACD_Z", c)
    hi52_pass = _hi52_ok_last(high, close, c)
    def _f(s):
        v = s.iloc[-1]
        return round(float(v), 3) if not (np.isnan(v) or np.isinf(v)) else None
    return {
        "close": round(float(close.iloc[-1]), 2),
        "rsi_val": round(float(rsi_val.iloc[-1]), 1) if not np.isnan(rsi_val.iloc[-1]) else None,
        "vwap_z": _f(vwap_z), "rsi_z": _f(rsi_z), "macd_z": _f(macd_z),
        "rsi_dz": _f(rsi_dz), "macd_dz": _f(macd_dz),
        "rsi_dz_accel": _f(rsi_dz_accel), "macd_dz_accel": _f(macd_dz_accel),
        "hi52_pass": hi52_pass, "div_rsi": div_rsi, "div_macd": div_macd,
    }

def _composite(sig, cape_z, c=None):
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    vz, rz, mz = sig.get("vwap_z"), sig.get("rsi_z"), sig.get("macd_z")
    if any(z is None for z in [vz, rz, mz]): return None, False
    cape_active = cape_z is not None and c["use_cape"]
    if cape_active:
        tot = c["wt_cape"] + c["wt_vwap"] + c["wt_rsi"] + c["wt_macd"]
        raw = (cape_z * c["wt_cape"] + vz * c["wt_vwap"] + rz * c["wt_rsi"] + mz * c["wt_macd"]) / tot
    else:
        tot = c["wt_vwap"] + c["wt_rsi"] + c["wt_macd"]
        raw = (vz * c["wt_vwap"] + rz * c["wt_rsi"] + mz * c["wt_macd"]) / tot
    clamped = float(_clamp(pd.Series([raw]), c["clamp_val"]).iloc[0])
    return round(clamped, 3), cape_active

def verdict(z, c=None):
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    if z is None: return "N/A"
    if z >= c["th_sbuy"]:  return "STRONG BUY"
    if z >= c["th_buy"]:   return "BUY"
    if z <= c["th_ssell"]: return "STRONG SELL"
    if z <= c["th_sell"]:  return "SELL"
    return "NEUTRAL"

def confidence(comp, sig, cape_z, cape_used, c=None):
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    if comp is None: return ""
    v = verdict(comp, c)
    if v in ("N/A", "NEUTRAL"): return ""
    direction  = 1 if comp > 0 else -1
    components = [sig.get("vwap_z"), sig.get("rsi_z"), sig.get("macd_z")]
    if cape_used and cape_z is not None: components.append(cape_z)
    agree   = sum(1 for z in components if z is not None and ((z > 0 and direction > 0) or (z < 0 and direction < 0)))
    abs_c   = abs(comp)
    if abs_c >= c["conf_strong"]   and agree >= 3: return "STRONG"
    if abs_c >= c["conf_moderate"] and agree >= 2: return "MODERATE"
    return "WEAK"

def _add_conf(vwap_z, cape_z, cape_used, rsi_val, agree, c=None):
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    if vwap_z is None or rsi_val is None: return False
    if float(vwap_z) < c["vwap_z_hard_min"]:  return False
    if float(rsi_val) >= c["rsi_hard_max"]:    return False
    if agree <= c["add_conf_agree_min"]:        return False
    if cape_used and cape_z is not None:
        if float(cape_z) <= 1.73:              return False
    return True

def _dz_accel_ok(sig, c=None):
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    if not c["dz_accel_enable"]: return True
    rsi_acc  = sig.get("rsi_dz_accel")
    macd_acc = sig.get("macd_dz_accel")
    if rsi_acc is None and macd_acc is None: return True
    rsi_ok  = (rsi_acc  is None) or (float(rsi_acc)  > 0)
    macd_ok = (macd_acc is None) or (float(macd_acc) > 0)
    if c.get("dz_accel_require_both", True): return rsi_ok and macd_ok
    return rsi_ok or macd_ok

def _candle_ok(open_price, high_price, low_price, close_price, c=None):
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    if not c["candle_body_enable"]: return True
    body       = abs(close_price - open_price)
    lower_wick = min(open_price, close_price) - low_price
    green      = close_price >= open_price * c["candle_green_tol"]
    hammer     = lower_wick >= c["hammer_mult"] * body if body > 0 else lower_wick > 0
    qualifies  = green or hammer
    if c.get("candle_body_hard", True): return qualifies
    return True

def _clean_df(raw):
    if isinstance(raw.columns, pd.MultiIndex):
        fields = raw.columns.get_level_values(0).tolist()
        data   = {f: raw.iloc[:, i].values for i, f in enumerate(fields)}
    else:
        data = {c: raw[c].values for c in raw.columns}
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in data for c in needed): return None
    idx = raw.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_convert(None)
    df = pd.DataFrame({c: data[c] for c in needed}, index=idx)
    df = df[df["Close"].notna()].copy()
    return df if not df.empty else None

def scan_stock(sym_raw, c=None):
    if c is None:
        c = DEFAULT_CFG
    ticker = to_yf(sym_raw)
    rows   = []
    try:
        tk  = yf.Ticker(ticker)
        raw = tk.history(period="3y", interval="1d", auto_adjust=True)
        if raw.empty: return sym_raw, [], "No data"
        df = _clean_df(raw)
        if df is None: return sym_raw, [], "Column parse failed"
        cape_z, cape_ratio, ttm_eps = compute_cape_z(tk, df, c)
        d_sig = compute_signals(df, c)
        weekly_raw = df.resample("W").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"})
        weekly = weekly_raw[weekly_raw["Close"].notna()].copy()
        w_sig  = compute_signals(weekly, c)
        for tf, sig, src_df in [("Daily", d_sig, df), ("Weekly", w_sig, weekly)]:
            if sig is None: continue
            comp, cape_used = _composite(sig, cape_z, c)
            if comp is None: continue
            vrd  = verdict(comp, c)
            conf = confidence(comp, sig, cape_z, cape_used, c)
            _ac_zs = [sig["vwap_z"], sig["rsi_z"], sig["macd_z"]]
            if cape_used and cape_z is not None: _ac_zs.append(cape_z)
            _ac_agree = sum(1 for z in _ac_zs if z is not None and float(z) > 0)
            ac         = _add_conf(sig["vwap_z"], cape_z, cape_used, sig["rsi_val"], _ac_agree, c)
            dz_acc_ok  = _dz_accel_ok(sig, c)
            hi52_ok    = sig.get("hi52_pass", True)
            last       = src_df.iloc[-1]
            candle_ok  = _candle_ok(float(last["Open"]), float(last["High"]), float(last["Low"]), float(last["Close"]), c)
            div_tags   = []
            if sig.get("div_rsi",  {}).get("tag"): div_tags.append(sig["div_rsi"]["tag"])
            if sig.get("div_macd", {}).get("tag"): div_tags.append(sig["div_macd"]["tag"])
            div_str = " | ".join(div_tags) if div_tags else ""
            rows.append({
                "Symbol": sym_raw, "TF": tf,
                "Signal": vrd, "Strength": conf,
                "Add_Conf": "YES" if ac else "NO",
                "ΔZ_Accel": "YES" if dz_acc_ok else "NO",
                "Candle_OK": "YES" if candle_ok else "NO",
                "Hi52_OK": "YES" if hi52_ok else "NO",
                "All_Gates": "YES" if (ac and dz_acc_ok and candle_ok and hi52_ok) else "NO",
                "Composite": comp, "CAPE_Z": cape_z if cape_used else None,
                "CAPE_PE": cape_ratio, "TTM_EPS": ttm_eps,
                "VWAP_Z": sig["vwap_z"], "RSI_Z": sig["rsi_z"], "MACD_Z": sig["macd_z"],
                "RSI_ΔZ": sig["rsi_dz"], "MACD_ΔZ": sig["macd_dz"],
                "RSI": sig["rsi_val"], "Close": sig["close"],
                "Divergence": div_str, "CAPE_Active": cape_used,
            })
        return sym_raw, rows, None
    except Exception as e:
        return sym_raw, [], f"{type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════
#  BACKTEST HELPERS
# ══════════════════════════════════════════════════════════════

def _weekly_signal_frame(df, c):
    min_bars = max(c["vwap_zlen"], c["rsi_zlen"], c["macd_zlen"]) + c["macd_slow"] + 30
    if len(df) < min_bars: return pd.DataFrame()
    close  = df["Close"].astype(float); high = df["High"].astype(float)
    low    = df["Low"].astype(float);   open_ = df["Open"].astype(float)
    volume = df["Volume"].astype(float).replace(0, np.nan)
    tp = (high + low + close) / 3.0
    vwap_num = (tp * volume).rolling(c["vwap_len"], min_periods=c["vwap_len"] // 2).sum()
    vwap_den = volume.rolling(c["vwap_len"], min_periods=c["vwap_len"] // 2).sum()
    vwap_val = vwap_num / vwap_den.replace(0, np.nan)
    vwap_zraw = _zscore(close - vwap_val, c["vwap_zlen"])
    vwap_z   = _clamp(-vwap_zraw if c["vwap_contrarian"] else vwap_zraw, c["clamp_val"])
    rsi_val  = _rsi(close, c["rsi_len"])
    rsi_lz   = _zscore(rsi_val, c["rsi_zlen"])
    rsi_dz   = rsi_lz.diff(c["rsi_dz_len"]).rolling(2).mean()
    rsi_comb = rsi_lz * (1 - c["rsi_dz_weight"]) + rsi_dz * c["rsi_dz_weight"]
    rsi_z    = _clamp(-rsi_comb if c["rsi_contrarian"] else rsi_comb, c["clamp_val"])
    rsi_dz_accel = rsi_dz.diff(c["dz_accel_bars"])
    ema_f    = close.ewm(span=c["macd_fast"], adjust=False, min_periods=c["macd_fast"]).mean()
    ema_s    = close.ewm(span=c["macd_slow"], adjust=False, min_periods=c["macd_slow"]).mean()
    macd_hist = (ema_f - ema_s) - (ema_f - ema_s).ewm(span=c["macd_sig"], adjust=False, min_periods=c["macd_sig"]).mean()
    macd_pct  = macd_hist / close.replace(0, np.nan) * 100.0
    macd_lz   = _zscore(macd_pct, c["macd_zlen"])
    macd_dz   = macd_lz.diff(c["macd_dz_len"]).rolling(2).mean()
    macd_comb = macd_lz * (1 - c["macd_dz_weight"]) + macd_dz * c["macd_dz_weight"]
    macd_z    = _clamp(macd_comb, c["clamp_val"])
    macd_dz_accel = macd_dz.diff(c["dz_accel_bars"])
    hi52_ok_s = _hi52_ok_series(high, close, c)
    return pd.DataFrame({
        "vwap_z": vwap_z, "rsi_z": rsi_z, "macd_z": macd_z,
        "rsi_dz": rsi_dz, "macd_dz": macd_dz,
        "rsi_dz_accel": rsi_dz_accel, "macd_dz_accel": macd_dz_accel,
        "hi52_ok": hi52_ok_s, "open": open_, "high": high,
        "low": low, "close": close, "rsi_val": rsi_val,
    }, index=df.index)


def backtest_one(sym_raw, lookback_weeks, profit_pct, c):
    ticker = to_yf(sym_raw)
    trades = []
    try:
        tk  = yf.Ticker(ticker)
        raw = tk.history(period="5y", interval="1d", auto_adjust=True)
        if raw.empty: return sym_raw, [], "No data"
        df = _clean_df(raw)
        if df is None: return sym_raw, [], "Column parse failed"
        cape_weekly = None
        if c["use_cape"]:
            cape_d = _compute_cape_z_series(tk, df, c)
            if not cape_d.empty:
                cape_w = cape_d.resample("W").last()
                if hasattr(cape_w.index, "tz") and cape_w.index.tz is not None:
                    cape_w.index = cape_w.index.tz_convert(None)
                cape_weekly = cape_w
        weekly_raw = df.resample("W").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"})
        weekly = weekly_raw[weekly_raw["Close"].notna()].copy()
        sf = _weekly_signal_frame(weekly, c)
        if sf.empty: return sym_raw, [], "Insufficient bars"
        if cape_weekly is not None:
            cape_weekly = cape_weekly.reindex(weekly.index).ffill()
        total_w    = len(weekly)
        scan_start = max(0, total_w - lookback_weeks - 1)
        scan_end   = total_w - 1
        open_positions = {}
        for i in range(scan_start, scan_end):
            vz_w = sf["vwap_z"].iloc[i]; rz_w = sf["rsi_z"].iloc[i]; mz_w = sf["macd_z"].iloc[i]
            if any(pd.isna(x) for x in (vz_w, rz_w, mz_w)): continue
            cz, cape_active = None, False
            if cape_weekly is not None:
                cv = cape_weekly.iloc[i]
                if not pd.isna(cv): cz, cape_active = float(cv), True
            if cape_active:
                tot    = c["wt_cape"] + c["wt_vwap"] + c["wt_rsi"] + c["wt_macd"]
                comp_w = (cz * c["wt_cape"] + float(vz_w) * c["wt_vwap"] + float(rz_w) * c["wt_rsi"] + float(mz_w) * c["wt_macd"]) / tot
            else:
                tot    = c["wt_vwap"] + c["wt_rsi"] + c["wt_macd"]
                comp_w = (float(vz_w) * c["wt_vwap"] + float(rz_w) * c["wt_rsi"] + float(mz_w) * c["wt_macd"]) / tot
            comp_w = float(np.clip(comp_w, -c["clamp_val"], c["clamp_val"]))
            if comp_w < c["bt_min_composite"]: continue
            vrd_w  = verdict(comp_w, c)
            if vrd_w not in ("BUY", "STRONG BUY"): continue
            sig_w  = {"vwap_z": float(vz_w), "rsi_z": float(rz_w), "macd_z": float(mz_w)}
            conf_w = confidence(comp_w, sig_w, cz, cape_active, c)
            if conf_w not in ("MODERATE", "STRONG"): continue
            if sym_raw in open_positions: continue
            rsi_entry_w = sf["rsi_val"].iloc[i]
            rsi_entry_w = float(rsi_entry_w) if not pd.isna(rsi_entry_w) else None
            e_zs_w    = [float(vz_w), float(rz_w), float(mz_w)]
            if cape_active and cz is not None: e_zs_w.append(cz)
            e_agree_w = sum(1 for z in e_zs_w if z > 0)
            ac_w = _add_conf(float(vz_w), cz, cape_active, rsi_entry_w, e_agree_w, c)
            if not ac_w: continue
            rsi_acc_w  = sf["rsi_dz_accel"].iloc[i]
            macd_acc_w = sf["macd_dz_accel"].iloc[i]
            if c["dz_accel_enable"]:
                rsi_ok_bt  = pd.isna(rsi_acc_w)  or float(rsi_acc_w)  > 0
                macd_ok_bt = pd.isna(macd_acc_w) or float(macd_acc_w) > 0
                if c.get("dz_accel_require_both", False):
                    if not (rsi_ok_bt and macd_ok_bt): continue
                else:
                    if not (rsi_ok_bt or macd_ok_bt): continue
            if c["hi52_enable"]:
                hi52_flag = sf["hi52_ok"].iloc[i]
                if not pd.isna(hi52_flag) and not bool(hi52_flag): continue
            bar_open  = float(sf["open"].iloc[i]); bar_high = float(sf["high"].iloc[i])
            bar_low   = float(sf["low"].iloc[i]);  bar_close = float(sf["close"].iloc[i])
            candle_ok_flag = _candle_ok(bar_open, bar_high, bar_low, bar_close, c)
            entry_date   = weekly.index[i]
            entry_price  = bar_close
            target_price = round(entry_price * (1 + profit_pct / 100), 2)
            exit_date, exit_price, hold_weeks = None, None, None
            status, exit_reason = "OPEN", "OPEN"
            for j in range(i + 1, total_w):
                if float(sf["high"].iloc[j]) >= target_price:
                    exit_date   = weekly.index[j]
                    exit_price  = target_price
                    hold_weeks  = j - i
                    status      = "HIT"
                    exit_reason = "TARGET"
                    break
            if exit_date is None:
                last_close = float(sf["close"].iloc[-1])
                exit_price = last_close
                hold_weeks = total_w - 1 - i
                open_ret   = (last_close - entry_price) / entry_price * 100
                status     = f"OPEN {open_ret:+.1f}%"
                exit_reason = "OPEN"
            ret_pct = round((exit_price - entry_price) / entry_price * 100, 2)
            trades.append({
                "Symbol": sym_raw, "Entry_Date": entry_date.strftime("%Y-%m-%d"),
                "Entry_Price": round(entry_price, 2), "Target": target_price,
                "Exit_Date": (exit_date.strftime("%Y-%m-%d") if exit_date else weekly.index[-1].strftime("%Y-%m-%d")),
                "Exit_Price": round(exit_price, 2),
                "Return_%": ret_pct, "Hold_Wks": hold_weeks, "Status": status,
                "W_Signal": vrd_w, "W_Strength": conf_w,
                "W_Comp": round(comp_w, 3), "Candle_OK": "YES" if candle_ok_flag else "NO",
            })
        return sym_raw, trades, None
    except Exception as e:
        return sym_raw, [], f"{type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════
#  PDF GENERATION
# ══════════════════════════════════════════════════════════════

def generate_scan_pdf(df_buy, df_sell, df_buy_conf, df_sell_conf, df_div, ts_str):
    if not _REPORTLAB:
        return None
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=1*cm,
                               rightMargin=1*cm, topMargin=1.2*cm, bottomMargin=1.2*cm)
    styles = getSampleStyleSheet()
    h1, h2, normal = styles["Heading1"], styles["Heading2"], styles["Normal"]
    PAGE_W = landscape(A4)[0] - 2*cm

    def _df_to_rl_table(df, col_widths=None):
        rows = [list(df.columns)] + [[str(v) if v is not None else "" for v in row] for row in df.itertuples(index=False)]
        n_c  = len(df.columns)
        cw   = col_widths or [PAGE_W / n_c] * n_c
        tbl  = Table(rows, colWidths=cw, repeatRows=1)
        style = TableStyle([
            ("BACKGROUND",    (0,0),(-1,0),  rl_colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",     (0,0),(-1,0),  rl_colors.white),
            ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 7),
            ("FONTNAME",      (0,1),(-1,-1), "Helvetica"),
            ("GRID",          (0,0),(-1,-1), 0.3, rl_colors.HexColor("#cccccc")),
            ("LEFTPADDING",   (0,0),(-1,-1), 3),
            ("RIGHTPADDING",  (0,0),(-1,-1), 3),
            ("TOPPADDING",    (0,0),(-1,-1), 2),
            ("BOTTOMPADDING", (0,0),(-1,-1), 2),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [rl_colors.HexColor("#f0f4ff"), rl_colors.white]),
        ])
        sig_col = list(df.columns).index("Signal") if "Signal" in df.columns else None
        for r_idx, row in enumerate(rows[1:], start=1):
            if sig_col is not None:
                sig = str(row[sig_col])
                if "BUY"  in sig: style.add("BACKGROUND",(0,r_idx),(-1,r_idx),rl_colors.HexColor("#d4edda"))
                elif "SELL" in sig: style.add("BACKGROUND",(0,r_idx),(-1,r_idx),rl_colors.HexColor("#f8d7da"))
        tbl.setStyle(style)
        return tbl

    def section(title, df, cols):
        elems = [Paragraph(title, h2), Spacer(1, 0.2*cm)]
        if df is None or df.empty:
            elems += [Paragraph("None.", normal), Spacer(1, 0.4*cm)]
            return elems
        avail = [c for c in cols if c in df.columns]
        sub   = df[avail].copy()
        cw    = [PAGE_W / len(avail)] * len(avail)
        elems += [_df_to_rl_table(sub, cw), Spacer(1, 0.5*cm)]
        return elems

    signal_cols = ["Symbol","TF","Signal","Strength","All_Gates","Add_Conf","ΔZ_Accel",
                   "Candle_OK","Hi52_OK","Composite","CAPE_Z","VWAP_Z","RSI_Z","MACD_Z","RSI","Close","Divergence"]
    conf_cols   = ["Symbol","D_Signal","D_Str","D_Comp","W_Signal","W_Str","W_Comp","CAPE_Z","Close","RSI(D)","Divergence"]
    div_cols    = ["Symbol","TF","Signal","Strength","Composite","Close","Divergence"]

    story = [
        Paragraph("SHANTANU'S VALUE MOMENTUM SWING TRADING SCANNER v1.6", h1),
        Paragraph(f"NSE  |  {ts_str}  |  IMP 1-7 active  |  8% BT target", normal),
        Spacer(1, 0.6*cm),
    ]
    for tf in ["Daily", "Weekly"]:
        sb = df_buy [df_buy ["TF"] == tf] if df_buy  is not None and not df_buy.empty  else pd.DataFrame()
        ss = df_sell[df_sell["TF"] == tf] if df_sell is not None and not df_sell.empty else pd.DataFrame()
        story += section(f"BUY  — {tf} ({len(sb)} stocks)",  sb, signal_cols)
        story += section(f"SELL — {tf} ({len(ss)} stocks)", ss, signal_cols)
    story.append(PageBreak())
    if df_buy_conf is not None and not df_buy_conf.empty:
        story += section(f"DUAL-TF BUY CONFLUENCE ({len(df_buy_conf)} stocks)", df_buy_conf, conf_cols)
    if df_sell_conf is not None and not df_sell_conf.empty:
        story += section(f"DUAL-TF SELL CONFLUENCE ({len(df_sell_conf)} stocks)", df_sell_conf, conf_cols)
    story += section(f"DIVERGENCE SIGNALS ({len(df_div) if df_div is not None else 0} entries)", df_div, div_cols)
    doc.build(story)
    buf.seek(0)
    return buf


def generate_backtest_pdf(df_trades, summary, ts_str):
    if not _REPORTLAB:
        return None
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=1*cm,
                               rightMargin=1*cm, topMargin=1.2*cm, bottomMargin=1.2*cm)
    styles = getSampleStyleSheet()
    PAGE_W = landscape(A4)[0] - 2*cm
    story  = [
        Paragraph("BACKTEST — WEEKLY BUY (IMP 1-7)  |  NSE Swing Trading", styles["Heading1"]),
        Paragraph(f"{ts_str}  |  Target +{summary['profit_pct']:.0f}%  |  No SL", styles["Normal"]),
        Spacer(1, 0.4*cm),
    ]
    smry_data = [
        ["Metric", "Value"],
        ["Total trades", str(summary["n_total"])],
        [f"HIT (≥+{summary['profit_pct']:.0f}%)", f"{summary['n_hit']}  Win rate: {summary['win_rate']:.1f}%"],
        ["OPEN", str(summary["n_open"])],
        ["Avg weeks to target", f"{summary['avg_hold_hit']:.1f} wks"],
        ["Portfolio value", f"₹{summary['portfolio_value']:,.0f}  ({summary['total_return_pct']:+.1f}%)"],
    ]
    smry_w = [PAGE_W * 0.35, PAGE_W * 0.55]
    t = Table(smry_data, colWidths=smry_w)
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),rl_colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",(0,0),(-1,0),rl_colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),8),
        ("FONTNAME",(0,1),(-1,-1),"Helvetica"),
        ("GRID",(0,0),(-1,-1),0.5,rl_colors.grey),
        ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]))
    story += [t, Spacer(1, 0.5*cm), Paragraph("TRADE LOG", styles["Heading2"]), Spacer(1, 0.2*cm)]
    cols  = list(df_trades.columns)
    n_c   = len(cols)
    col_w = [PAGE_W / n_c] * n_c
    rows  = [cols] + [[str(v) if v is not None else "" for v in row] for row in df_trades.itertuples(index=False)]
    tbl   = Table(rows, colWidths=col_w, repeatRows=1)
    ts_st = TableStyle([
        ("BACKGROUND",(0,0),(-1,0),rl_colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",(0,0),(-1,0),rl_colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),6.5),
        ("FONTNAME",(0,1),(-1,-1),"Helvetica"),
        ("GRID",(0,0),(-1,-1),0.3,rl_colors.HexColor("#cccccc")),
        ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),
        ("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[rl_colors.HexColor("#f0f4ff"),rl_colors.white]),
    ])
    ret_col = cols.index("Return_%") if "Return_%" in cols else None
    for r_idx, row in enumerate(rows[1:], start=1):
        ret = float(row[ret_col]) if ret_col is not None and row[ret_col] not in ("","None") else 0.0
        if   ret < 0:  ts_st.add("BACKGROUND",(0,r_idx),(-1,r_idx),rl_colors.HexColor("#f8d7da"))
        elif ret >= 8: ts_st.add("BACKGROUND",(0,r_idx),(-1,r_idx),rl_colors.HexColor("#d4edda"))
    tbl.setStyle(ts_st)
    story.append(tbl)
    doc.build(story)
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════

def signal_color(s):
    colors = {
        "STRONG BUY":  "background-color:#065f46;color:#6ee7b7;font-weight:600",
        "BUY":         "background-color:#064e3b;color:#34d399;font-weight:600",
        "SELL":        "background-color:#7f1d1d;color:#fca5a5;font-weight:600",
        "STRONG SELL": "background-color:#991b1b;color:#f87171;font-weight:600",
        "NEUTRAL":     "background-color:#1e293b;color:#94a3b8",
    }
    return colors.get(s, "")

def style_signal_col(df):
    def _color_signal(val):
        return signal_color(val)
    if "Signal" in df.columns:
        return df.style.applymap(_color_signal, subset=["Signal"])
    return df.style

def metric_card(label, value, color_class="metric-blue"):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{value}</div>
    </div>
    """


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<h2 style="font-family:Syne,sans-serif;color:#60a5fa;margin-bottom:4px">⚙️ Config</h2>', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b;font-size:12px;margin-bottom:16px">Adjust scanner parameters</div>', unsafe_allow_html=True)

    with st.expander("🔬 Filters & Thresholds", expanded=True):
        workers        = st.slider("Parallel workers",    4, 16, DEFAULT_CFG["workers"])
        min_composite  = st.slider("Min |Composite|",     0.5, 2.5, DEFAULT_CFG["min_composite"], 0.05)
        rsi_hard_max   = st.slider("RSI hard gate (<)",   30, 60,  int(DEFAULT_CFG["rsi_hard_max"]))
        vwap_z_hard_min = st.slider("VWAP_Z min",         0.5, 3.0, DEFAULT_CFG["vwap_z_hard_min"], 0.1)
        hi52_pct       = st.slider("52W high % max",      0.5, 1.0, DEFAULT_CFG["hi52_pct"], 0.01)
        bt_min_comp    = st.slider("BT composite floor",  0.5, 2.5, DEFAULT_CFG["bt_min_composite"], 0.05)

    with st.expander("📊 Indicator Weights"):
        wt_cape = st.slider("CAPE weight",  0, 50, int(DEFAULT_CFG["wt_cape"]))
        wt_vwap = st.slider("VWAP weight",  0, 50, int(DEFAULT_CFG["wt_vwap"]))
        wt_rsi  = st.slider("RSI weight",   0, 50, int(DEFAULT_CFG["wt_rsi"]))
        wt_macd = st.slider("MACD weight",  0, 50, int(DEFAULT_CFG["wt_macd"]))

    with st.expander("🔀 Feature Toggles"):
        use_cape       = st.checkbox("Use CAPE",              DEFAULT_CFG["use_cape"])
        div_enable     = st.checkbox("Divergence detection",  DEFAULT_CFG["div_enable"])
        dz_accel       = st.checkbox("ΔZ Acceleration (IMP1)", DEFAULT_CFG["dz_accel_enable"])
        require_both   = st.checkbox("Require BOTH ΔZ accel", DEFAULT_CFG["dz_accel_require_both"])
        hi52_enable    = st.checkbox("52W High gate (IMP7)",  DEFAULT_CFG["hi52_enable"])

    cfg = {**DEFAULT_CFG,
           "workers": workers, "min_composite": min_composite,
           "rsi_hard_max": float(rsi_hard_max), "vwap_z_hard_min": vwap_z_hard_min,
           "hi52_pct": hi52_pct, "bt_min_composite": bt_min_comp,
           "wt_cape": float(wt_cape), "wt_vwap": float(wt_vwap),
           "wt_rsi": float(wt_rsi),  "wt_macd": float(wt_macd),
           "use_cape": use_cape, "div_enable": div_enable,
           "dz_accel_enable": dz_accel, "dz_accel_require_both": require_both,
           "hi52_enable": hi52_enable}
    st.session_state["cfg"] = cfg

    st.markdown("---")
    st.markdown('<div style="font-size:11px;color:#475569;font-family:DM Mono,monospace">📦 Universe: ' + str(len(ALL_SYMBOLS)) + ' NSE stocks<br>📅 Data: 3Y daily via yfinance</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  MAIN HEADER
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div style="padding:28px 0 8px 0">
    <div style="font-family:Syne,sans-serif;font-size:11px;color:#3b82f6;letter-spacing:3px;text-transform:uppercase;margin-bottom:6px">NSE SWING TRADING</div>
    <h1 style="font-family:Syne,sans-serif;font-size:36px;font-weight:800;color:#f1f5f9;margin:0;line-height:1.1">
        Value Momentum Scanner
        <span style="font-size:16px;color:#64748b;font-weight:400;margin-left:12px">v1.6</span>
    </h1>
    <div style="color:#64748b;font-size:13px;margin-top:8px;font-family:DM Mono,monospace">
        CAPE · VWAP_Z · RSI_Z · MACD_Z  ·  IMP 1–7 Active  ·  Dual TF Confluence
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════

tab_scan, tab_bt, tab_focus, tab_about = st.tabs([
    "🔍 Live Scan", "📈 Backtest", "🔭 Focus Stock", "ℹ️ About"
])


# ══════════════════════════════════════════════════════════════
#  TAB 1: LIVE SCAN
# ══════════════════════════════════════════════════════════════

with tab_scan:
    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_scan = st.button("🚀  Run Live Scan", key="run_scan")

    if run_scan or "scan_results" in st.session_state:

        if run_scan:
            # Clear old results
            for k in ["scan_results", "scan_errors", "scan_ts"]:
                st.session_state.pop(k, None)

            symbols = ALL_SYMBOLS
            total   = len(symbols)
            prog    = st.progress(0, text=f"Scanning {total} NSE stocks…")
            status_txt = st.empty()
            all_rows, errors, done = [], [], 0

            with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
                futures = {ex.submit(scan_stock, s, cfg): s for s in symbols}
                for fut in as_completed(futures):
                    sym, rows, err = fut.result()
                    done += 1
                    if err: errors.append((sym, err))
                    all_rows.extend(rows)
                    pct = done / total
                    buys  = sum(1 for r in all_rows if "BUY"  in r["Signal"])
                    sells = sum(1 for r in all_rows if "SELL" in r["Signal"])
                    prog.progress(pct, text=f"[{done}/{total}] {sym} — buy={buys} sell={sells}")

            prog.empty(); status_txt.empty()

            st.session_state["scan_results"] = all_rows
            st.session_state["scan_errors"]  = errors
            st.session_state["scan_ts"]       = datetime.now().strftime("%d %b %Y %H:%M")

        # ── Load from session ─────────────────────────────────
        all_rows = st.session_state.get("scan_results", [])
        errors   = st.session_state.get("scan_errors", [])
        ts_str   = st.session_state.get("scan_ts", "")

        if not all_rows:
            st.warning("No signals found. Check your internet connection or relax filters.")
        else:
            df_all = pd.DataFrame(all_rows)
            df_signal = df_all[df_all["Composite"].abs() >= cfg["min_composite"]].copy()

            df_buy  = df_signal[df_signal["Signal"].isin(["BUY","STRONG BUY"])].copy()
            df_sell = df_signal[df_signal["Signal"].isin(["SELL","STRONG SELL"])].copy()
            df_div  = df_all[df_all["Divergence"] != ""].copy()

            # Dual-TF confluence
            d_buys  = set(df_buy [df_buy ["TF"] == "Daily"] ["Symbol"])
            w_buys  = set(df_buy [df_buy ["TF"] == "Weekly"]["Symbol"])
            d_sells = set(df_sell[df_sell["TF"] == "Daily"] ["Symbol"])
            w_sells = set(df_sell[df_sell["TF"] == "Weekly"]["Symbol"])
            buy_conf_syms  = sorted(d_buys  & w_buys)
            sell_conf_syms = sorted(d_sells & w_sells)

            def _build_conf_df(syms, df_src):
                rows_c = []
                for sym in syms:
                    d = df_src[(df_src["Symbol"] == sym) & (df_src["TF"] == "Daily")]
                    w = df_src[(df_src["Symbol"] == sym) & (df_src["TF"] == "Weekly")]
                    if d.empty or w.empty: continue
                    d, w = d.iloc[0], w.iloc[0]
                    rows_c.append({
                        "Symbol": sym, "D_Signal": d["Signal"], "D_Str": d["Strength"],
                        "D_Comp": d["Composite"], "W_Signal": w["Signal"],
                        "W_Str": w["Strength"], "W_Comp": w["Composite"],
                        "All_Gates": d["All_Gates"], "CAPE_Z": d["CAPE_Z"],
                        "Close": d["Close"], "RSI(D)": d["RSI"],
                        "Divergence": d["Divergence"] or w["Divergence"] or "",
                    })
                if not rows_c: return pd.DataFrame()
                return pd.DataFrame(rows_c).sort_values("D_Comp", ascending=False)

            df_buy_conf  = _build_conf_df(buy_conf_syms,  df_buy)
            df_sell_conf = _build_conf_df(sell_conf_syms, df_sell)

            # ── METRIC CARDS ───────────────────────────────────
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.markdown(metric_card("BUY Signals",     len(df_buy),          "metric-green"),  unsafe_allow_html=True)
            c2.markdown(metric_card("SELL Signals",    len(df_sell),         "metric-red"),    unsafe_allow_html=True)
            c3.markdown(metric_card("BUY Confluence",  len(buy_conf_syms),   "metric-blue"),   unsafe_allow_html=True)
            c4.markdown(metric_card("SELL Confluence", len(sell_conf_syms),  "metric-yellow"), unsafe_allow_html=True)
            c5.markdown(metric_card("Divergences",     len(df_div),          "metric-blue"),   unsafe_allow_html=True)

            st.markdown(f'<div style="color:#475569;font-size:11px;font-family:DM Mono,monospace;margin-top:6px;margin-bottom:16px">Scan completed: {ts_str} — Universe: {len(ALL_SYMBOLS)} stocks</div>', unsafe_allow_html=True)

            # ── CHARTS ─────────────────────────────────────────
            if _PLOTLY and not df_buy.empty:
                with st.expander("📊 Signal Distribution Charts", expanded=True):
                    ch1, ch2 = st.columns(2)

                    # Composite score distribution
                    with ch1:
                        fig1 = go.Figure()
                        fig1.add_trace(go.Histogram(
                            x=df_buy[df_buy["TF"]=="Daily"]["Composite"],
                            name="Daily BUY", marker_color="#34d399", opacity=0.7, nbinsx=20))
                        fig1.add_trace(go.Histogram(
                            x=df_buy[df_buy["TF"]=="Weekly"]["Composite"],
                            name="Weekly BUY", marker_color="#60a5fa", opacity=0.7, nbinsx=20))
                        fig1.update_layout(
                            title="Composite Score Distribution (BUY)",
                            paper_bgcolor="#111827", plot_bgcolor="#111827",
                            font=dict(color="#94a3b8", family="DM Sans"),
                            barmode="overlay", height=280,
                            margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig1, use_container_width=True)

                    # Top buy stocks by composite
                    with ch2:
                        top_buys = (df_buy[df_buy["TF"]=="Daily"]
                                    .nlargest(15, "Composite")[["Symbol","Composite","Strength"]]
                                    .copy())
                        if not top_buys.empty:
                            color_map = {"STRONG": "#34d399", "MODERATE": "#60a5fa", "WEAK": "#fbbf24", "": "#94a3b8"}
                            colors_list = [color_map.get(s, "#94a3b8") for s in top_buys["Strength"]]
                            fig2 = go.Figure(go.Bar(
                                x=top_buys["Symbol"], y=top_buys["Composite"],
                                marker_color=colors_list, text=top_buys["Strength"],
                                textposition="outside"))
                            fig2.update_layout(
                                title="Top 15 Daily BUY by Composite",
                                paper_bgcolor="#111827", plot_bgcolor="#111827",
                                font=dict(color="#94a3b8", family="DM Sans"),
                                height=280, margin=dict(l=20, r=20, t=40, b=20),
                                yaxis=dict(gridcolor="#1e293b"))
                            st.plotly_chart(fig2, use_container_width=True)

            # ── BUY TABLES ─────────────────────────────────────
            display_cols = ["Symbol","TF","Signal","Strength","All_Gates","Add_Conf",
                            "ΔZ_Accel","Candle_OK","Hi52_OK","Composite",
                            "VWAP_Z","RSI_Z","MACD_Z","RSI","Close","Divergence"]

            st.markdown('<div class="section-header buy-header">🟢 BUY Signals</div>', unsafe_allow_html=True)
            b1, b2 = st.tabs(["Daily", "Weekly"])
            for tf, tab_ref in [("Daily", b1), ("Weekly", b2)]:
                sub = df_buy[df_buy["TF"] == tf]
                with tab_ref:
                    if sub.empty:
                        st.info(f"No {tf} BUY signals.")
                    else:
                        avail = [c for c in display_cols if c in sub.columns]
                        styled = sub[avail].reset_index(drop=True)
                        st.dataframe(styled, use_container_width=True, height=min(400, 40 + 35 * len(styled)))

            # ── SELL TABLES ────────────────────────────────────
            st.markdown('<div class="section-header sell-header">🔴 SELL Signals</div>', unsafe_allow_html=True)
            s1, s2 = st.tabs(["Daily", "Weekly"])
            for tf, tab_ref in [("Daily", s1), ("Weekly", s2)]:
                sub = df_sell[df_sell["TF"] == tf]
                with tab_ref:
                    if sub.empty:
                        st.info(f"No {tf} SELL signals.")
                    else:
                        avail = [c for c in display_cols if c in sub.columns]
                        st.dataframe(sub[avail].reset_index(drop=True), use_container_width=True, height=min(400, 40 + 35 * len(sub)))

            # ── DUAL-TF CONFLUENCE ─────────────────────────────
            st.markdown('<div class="section-header conf-header">⚡ Dual-TF Confluence</div>', unsafe_allow_html=True)
            cx1, cx2 = st.columns(2)
            with cx1:
                st.markdown("**🟢 BUY Confluence** (Daily + Weekly both BUY)")
                if df_buy_conf.empty:
                    st.info("No BUY confluence stocks.")
                else:
                    st.dataframe(df_buy_conf.reset_index(drop=True), use_container_width=True)

            with cx2:
                st.markdown("**🔴 SELL Confluence** (Daily + Weekly both SELL)")
                if df_sell_conf.empty:
                    st.info("No SELL confluence stocks.")
                else:
                    st.dataframe(df_sell_conf.reset_index(drop=True), use_container_width=True)

            # ── DIVERGENCE ─────────────────────────────────────
            st.markdown('<div class="section-header div-header">📡 Divergence Signals</div>', unsafe_allow_html=True)
            if df_div.empty:
                st.info("No divergence signals detected.")
            else:
                div_cols = ["Symbol","TF","Signal","Strength","Composite","Close","Divergence"]
                avail    = [c for c in div_cols if c in df_div.columns]
                st.dataframe(df_div[avail].sort_values(["TF","Symbol"]).reset_index(drop=True), use_container_width=True)

            # ── DOWNLOADS ──────────────────────────────────────
            st.markdown("---")
            st.markdown("### 📥 Export Results")
            dl1, dl2, dl3 = st.columns(3)

            with dl1:
                csv_buf = io.StringIO()
                df_all.to_csv(csv_buf, index=False)
                st.download_button("⬇️ Download CSV (All)", csv_buf.getvalue().encode(),
                                   file_name=f"VMS_Scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                   mime="text/csv")
            with dl2:
                if not df_buy.empty:
                    buy_csv = io.StringIO()
                    df_buy.to_csv(buy_csv, index=False)
                    st.download_button("⬇️ Download BUY CSV", buy_csv.getvalue().encode(),
                                       file_name=f"VMS_BUY_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                       mime="text/csv")
            with dl3:
                if _REPORTLAB:
                    pdf_buf = generate_scan_pdf(df_buy, df_sell, df_buy_conf, df_sell_conf, df_div, ts_str)
                    if pdf_buf:
                        st.download_button("⬇️ Download PDF Report", pdf_buf,
                                           file_name=f"VMS_Scan_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                           mime="application/pdf")
                else:
                    st.caption("PDF unavailable — install reportlab")

            # Errors
            if errors:
                with st.expander(f"⚠️ {len(errors)} stocks failed"):
                    err_df = pd.DataFrame(errors, columns=["Symbol", "Error"])
                    st.dataframe(err_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  TAB 2: BACKTEST
# ══════════════════════════════════════════════════════════════

with tab_bt:
    st.markdown('<div class="section-header bt-header">📈 Historical Backtest (Weekly-only BUY)</div>', unsafe_allow_html=True)

    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        lookback_wks = st.slider("Lookback (weeks)", 52, 520, 260, 26, help="~5 years = 260 weeks")
    with bc2:
        profit_pct = st.slider("Profit target (%)", 4.0, 20.0, cfg["backtest_profit_pct"], 0.5)
    with bc3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_bt = st.button("🚀  Run Backtest", key="run_bt")

    st.markdown(f"""
    <div class="info-box">
    Entry conditions (all must pass):<br>
    BT-1  Weekly composite ≥ {cfg['bt_min_composite']}  (IMP-6)<br>
    BT-2  Weekly verdict = BUY or STRONG BUY<br>
    BT-3  Confidence = MODERATE or STRONG<br>
    BT-4  Add_Conf gate: VWAP_Z ≥ {cfg['vwap_z_hard_min']}, RSI &lt; {cfg['rsi_hard_max']}, ≥3 agree (IMP-5)<br>
    BT-5  ΔZ Acceleration: RSI OR MACD ΔZ accelerating (IMP-1)<br>
    BT-6  52W high proximity: close ≤ {cfg['hi52_pct']*100:.0f}% of 52W high (IMP-7)<br>
    Exit: weekly high ≥ entry × (1 + {profit_pct:.0f}%)
    </div>
    """, unsafe_allow_html=True)

    if run_bt or "bt_results" in st.session_state:
        if run_bt:
            for k in ["bt_results","bt_errors","bt_ts","bt_summary"]:
                st.session_state.pop(k, None)

            symbols  = ALL_SYMBOLS
            total    = len(symbols)
            prog_bt  = st.progress(0, text=f"Backtesting {total} stocks…")
            all_trades, errors_bt, done = [], [], 0

            with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
                futures = {ex.submit(backtest_one, s, lookback_wks, profit_pct, cfg): s for s in symbols}
                for fut in as_completed(futures):
                    sym, trades, err = fut.result()
                    done += 1
                    if err: errors_bt.append((sym, err))
                    all_trades.extend(trades)
                    prog_bt.progress(done / total, text=f"[{done}/{total}] {sym} — {len(all_trades)} signals")

            prog_bt.empty()
            st.session_state["bt_results"] = all_trades
            st.session_state["bt_errors"]  = errors_bt
            st.session_state["bt_ts"]      = datetime.now().strftime("%d %b %Y %H:%M")

        all_trades = st.session_state.get("bt_results", [])
        errors_bt  = st.session_state.get("bt_errors", [])
        ts_str_bt  = st.session_state.get("bt_ts", "")

        if not all_trades:
            st.warning("No qualifying backtest signals found. Try relaxing filters or extending lookback.")
        else:
            df_bt = pd.DataFrame(all_trades).sort_values(["Entry_Date","Symbol"])
            df_hit  = df_bt[df_bt["Status"] == "HIT"]
            df_open = df_bt[df_bt["Status"].str.startswith("OPEN")]
            n_total = len(df_bt)
            n_hit   = len(df_hit)
            n_open  = len(df_open)
            win_rate     = n_hit / n_total * 100 if n_total > 0 else 0.0
            avg_hold_hit = df_hit ["Hold_Wks"].mean() if n_hit  > 0 else 0.0
            avg_hold_open = df_open["Hold_Wks"].mean() if n_open > 0 else 0.0
            df_closed     = df_bt[~df_bt["Status"].str.startswith("OPEN")]
            realized_pnl  = (df_closed["Return_%"] / 100 * POSITION_SIZE).sum()
            unrealized_pnl = (df_open["Return_%"]  / 100 * POSITION_SIZE).sum()
            deployed_cap  = n_open * POSITION_SIZE
            available_cash = INITIAL_CAPITAL - deployed_cap + realized_pnl
            portfolio_value = INITIAL_CAPITAL + realized_pnl + unrealized_pnl
            total_return_pct = (portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

            summary = dict(n_total=n_total, n_hit=n_hit, n_open=n_open,
                           win_rate=win_rate, avg_hold_hit=avg_hold_hit,
                           avg_hold_open=avg_hold_open, profit_pct=profit_pct,
                           realized_pnl=realized_pnl, unrealized_pnl=unrealized_pnl,
                           deployed_cap=deployed_cap, available_cash=available_cash,
                           portfolio_value=portfolio_value, total_return_pct=total_return_pct)
            st.session_state["bt_summary"] = summary

            # Metrics
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.markdown(metric_card("Total Signals",  n_total,                             "metric-blue"),   unsafe_allow_html=True)
            m2.markdown(metric_card("HIT (Target)",   f"{n_hit} ({win_rate:.1f}%)",         "metric-green"),  unsafe_allow_html=True)
            m3.markdown(metric_card("Still Open",     n_open,                              "metric-yellow"), unsafe_allow_html=True)
            m4.markdown(metric_card("Avg Hold (HIT)", f"{avg_hold_hit:.1f} wks",            "metric-blue"),   unsafe_allow_html=True)
            pnl_color = "metric-green" if total_return_pct >= 0 else "metric-red"
            m5.markdown(metric_card("Portfolio Return", f"{total_return_pct:+.1f}%",        pnl_color),       unsafe_allow_html=True)

            # Portfolio summary
            st.markdown(f"""
            <div class="info-box" style="margin-top:12px">
            💰  Initial: ₹{INITIAL_CAPITAL:,.0f}  &nbsp;|&nbsp;
            Deployed: ₹{deployed_cap:,.0f}  &nbsp;|&nbsp;
            Realized P&L: ₹{realized_pnl:+,.0f}  &nbsp;|&nbsp;
            Unrealized P&L: ₹{unrealized_pnl:+,.0f}  &nbsp;|&nbsp;
            Portfolio: ₹{portfolio_value:,.0f}
            </div>
            """, unsafe_allow_html=True)

            # Charts
            if _PLOTLY and len(df_bt) > 0:
                ch1, ch2 = st.columns(2)
                with ch1:
                    # Return distribution
                    fig_r = go.Figure()
                    fig_r.add_trace(go.Histogram(
                        x=df_bt["Return_%"], nbinsx=30,
                        marker_color=["#34d399" if r >= profit_pct else "#f87171" for r in df_bt["Return_%"]],
                        marker_colorscale=None))
                    fig_r.add_vline(x=profit_pct, line_dash="dash", line_color="#fbbf24",
                                    annotation_text=f"Target {profit_pct:.0f}%")
                    fig_r.update_layout(
                        title="Return Distribution", paper_bgcolor="#111827", plot_bgcolor="#111827",
                        font=dict(color="#94a3b8", family="DM Sans"), height=280,
                        margin=dict(l=20, r=20, t=40, b=20), yaxis=dict(gridcolor="#1e293b"))
                    st.plotly_chart(fig_r, use_container_width=True)

                with ch2:
                    # Hold weeks distribution
                    fig_h = go.Figure(go.Histogram(
                        x=df_hit["Hold_Wks"], nbinsx=20, marker_color="#60a5fa"))
                    fig_h.update_layout(
                        title="Hold Duration (Winners)", paper_bgcolor="#111827", plot_bgcolor="#111827",
                        font=dict(color="#94a3b8", family="DM Sans"), height=280,
                        margin=dict(l=20, r=20, t=40, b=20), yaxis=dict(gridcolor="#1e293b"))
                    st.plotly_chart(fig_h, use_container_width=True)

            # Trade log
            st.markdown("#### 📋 Trade Log")
            st.dataframe(df_bt.reset_index(drop=True), use_container_width=True, height=400)

            # Downloads
            st.markdown("---")
            dl1, dl2 = st.columns(2)
            with dl1:
                bt_csv = io.StringIO()
                df_bt.to_csv(bt_csv, index=False)
                st.download_button("⬇️ Download Backtest CSV", bt_csv.getvalue().encode(),
                                   file_name=f"VMS_BT_{lookback_wks}wk_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                   mime="text/csv")
            with dl2:
                if _REPORTLAB:
                    bt_pdf = generate_backtest_pdf(df_bt, summary, ts_str_bt)
                    if bt_pdf:
                        st.download_button("⬇️ Download Backtest PDF", bt_pdf,
                                           file_name=f"VMS_BT_{lookback_wks}wk_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                           mime="application/pdf")

            if errors_bt:
                with st.expander(f"⚠️ {len(errors_bt)} stocks failed"):
                    st.dataframe(pd.DataFrame(errors_bt, columns=["Symbol","Error"]), use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  TAB 3: FOCUS STOCK
# ══════════════════════════════════════════════════════════════

with tab_focus:
    st.markdown('<div class="section-header conf-header">🔭 Focus Stock Deep-Dive</div>', unsafe_allow_html=True)

    focus_sym = st.selectbox("Select stock", [""] + ALL_SYMBOLS, key="focus_sym")
    run_focus = st.button("🔍 Analyse", key="run_focus")

    if run_focus and focus_sym:
        with st.spinner(f"Fetching data for {focus_sym}…"):
            sym_raw, rows, err = scan_stock(focus_sym, cfg)

        if err:
            st.error(f"Error: {err}")
        elif not rows:
            st.warning("Insufficient data.")
        else:
            df_focus = pd.DataFrame(rows)
            st.markdown(f"#### {focus_sym} — All Signals (Daily + Weekly)")

            for tf in ["Daily", "Weekly"]:
                sub = df_focus[df_focus["TF"] == tf]
                if sub.empty: continue
                row = sub.iloc[0]

                sig_color = "#34d399" if "BUY" in row["Signal"] else ("#f87171" if "SELL" in row["Signal"] else "#94a3b8")
                st.markdown(f"""
                <div style="background:#131c2e;border:1px solid #1e3a5f;border-left:4px solid {sig_color};
                            border-radius:10px;padding:16px 20px;margin-bottom:12px">
                    <span style="font-family:Syne,sans-serif;font-size:16px;font-weight:700;color:{sig_color}">{tf}</span>
                    &nbsp;&nbsp;
                    <span style="background:#1e293b;color:#e2e8f0;padding:3px 12px;border-radius:20px;font-size:13px;font-family:DM Mono,monospace">
                        {row['Signal']} · {row['Strength'] or 'N/A'}</span>
                    &nbsp;&nbsp;
                    <span style="color:#64748b;font-size:12px;font-family:DM Mono,monospace">
                        Close: ₹{row['Close']}  ·  RSI: {row['RSI']}  ·  Composite: {row['Composite']}</span>
                </div>
                """, unsafe_allow_html=True)

            # Full table
            focus_cols = ["TF","Signal","Strength","All_Gates","Add_Conf","ΔZ_Accel",
                          "Candle_OK","Hi52_OK","Composite","CAPE_Z","VWAP_Z","RSI_Z",
                          "MACD_Z","RSI_ΔZ","MACD_ΔZ","RSI","Close","Divergence"]
            avail = [c for c in focus_cols if c in df_focus.columns]
            st.dataframe(df_focus[avail].reset_index(drop=True), use_container_width=True)

            # Price chart
            if _PLOTLY:
                with st.spinner("Loading price chart…"):
                    try:
                        tk  = yf.Ticker(to_yf(focus_sym))
                        raw = tk.history(period="1y", interval="1d", auto_adjust=True)
                        if not raw.empty:
                            raw = raw.reset_index()
                            fig_c = go.Figure()
                            fig_c.add_trace(go.Candlestick(
                                x=raw["Date"], open=raw["Open"], high=raw["High"],
                                low=raw["Low"], close=raw["Close"], name="Price",
                                increasing_line_color="#34d399", decreasing_line_color="#f87171"))
                            fig_c.update_layout(
                                title=f"{focus_sym} — 1 Year Daily Chart",
                                paper_bgcolor="#111827", plot_bgcolor="#111827",
                                font=dict(color="#94a3b8", family="DM Sans"),
                                xaxis_rangeslider_visible=False,
                                height=400, margin=dict(l=20, r=20, t=40, b=20),
                                yaxis=dict(gridcolor="#1e293b"),
                                xaxis=dict(gridcolor="#1e293b"))
                            st.plotly_chart(fig_c, use_container_width=True)
                    except Exception as e:
                        st.caption(f"Chart unavailable: {e}")


# ══════════════════════════════════════════════════════════════
#  TAB 4: ABOUT
# ══════════════════════════════════════════════════════════════

with tab_about:
    st.markdown("""
## Shantanu's Value Momentum Swing Trading Scanner v1.6

A quantitative NSE swing trading screener combining **value** and **momentum** signals
to identify high-probability BUY/SELL setups across ~276 NSE stocks.

### Signals Used (25/25/25/25 composite)

| Signal | Description |
|--------|-------------|
| **CAPE Z-Score** | Cyclically-adjusted PE, India CPI-adjusted, Pine-style decay weights |
| **VWAP Z-Score** | Price deviation from volume-weighted average (contrarian) |
| **RSI Z + ΔZ** | RSI z-score blended with momentum acceleration |
| **MACD% Z + ΔZ** | MACD histogram z-score blended with momentum acceleration |

### IMP Filters (v1.6)

| # | Filter | Gate |
|---|--------|------|
| IMP-1 | ΔZ Acceleration | RSI OR MACD ΔZ accelerating |
| IMP-2 | RSI hard gate | RSI < 50 |
| IMP-3 | Candle body | Green or hammer (soft — informational) |
| IMP-4 | VWAP_Z floor | VWAP_Z ≥ 1.5 |
| IMP-5 | Add_Conf | VWAP+RSI+≥3 agree (hard gate) |
| IMP-6 | Composite floor | ≥ 1.25 (backtest only) |
| IMP-7 | 52W proximity | Price ≤ 85% of 52W high |

### Signal Verdict Thresholds

| Composite | Verdict |
|-----------|---------|
| ≥ 2.0 | STRONG BUY |
| ≥ 1.0 | BUY |
| ≤ -1.0 | SELL |
| ≤ -2.0 | STRONG SELL |

### Hosting on Streamlit Community Cloud (Free)

1. Push this file + `requirements.txt` to a **public GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub → Select repo → Select `zconf_streamlit_app.py`
4. Click **Deploy** — live in ~2 minutes!

### requirements.txt
```
streamlit
yfinance
pandas
numpy
plotly
reportlab
```

### Data Source
All market data via **yfinance** (Yahoo Finance). No API keys needed.
    """)
