#!/usr/bin/env python3
"""
Shantanu's VALUE MOMENTUM SWING TRADING SCANNER — v1.9 (Audited)
NSE Swing Trading Screener with CAPE / RSI / MACD signals

Improvements in v1.9 (audit fixes):
✓ H1: backtest allows re-entry after exit (was: one trade per symbol ever)
✓ H2: CAPE z included in backtest composite (same weights as live scan)
✓ H3: stop-loss + max-hold time exits in backtest; Open metric fixed
✓ H4: EPS dates shifted by reporting lag (60d) - no look-ahead in CAPE z
✓ H5: incomplete weekly bar dropped - weekly signals no longer repaint
✓ Vol/beta layer: annualised vol, vol z-score, beta vs NIFTY,
  risk-adjusted ranking, capitulation flag, high-beta caution, vol-scaled exits
✓ Direction-aware Add_Conf gate for SELL side; STRONG reachable w/o CAPE
✓ 5y data fetch (weekly z depth); thread-safe FX cache

Improvements in v1.8:
✓ Fixed syntax errors (_weekly_signal_frame)
✓ Added yfinance caching (1-hour TTL)
✓ Retry logic for flaky API calls
✓ Reuse scan results (avoid recomputation)
✓ Input validation (weight sums, bounds)
✓ Type hints throughout
✓ Better error messages
✓ Progress indicators
✓ Modular organization

To run:
    pip install streamlit yfinance pandas numpy plotly reportlab tenacity
    streamlit run vms_scanner_v1_8_improved.py
"""

# ══════════════════════════════════════════════════════════════
# IMPORTS & CONFIG
# ══════════════════════════════════════════════════════════════

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Tuple, Optional, Dict, List, Any
import warnings
import io
import threading

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
    _TENACITY = True
except ImportError:
    _TENACITY = False

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
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="VMS Scanner v1.9 — NSE Swing Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════
# DEFAULT CONFIGURATION
# ══════════════════════════════════════════════════════════════

DEFAULT_CFG: Dict[str, Any] = {
    # CAPE
    "use_cape": True, "cape_zlen": 252, "cape_bearish": True, "cape_max_q": 8,
    # RSI
    "rsi_len": 14, "rsi_zlen": 100, "rsi_contrarian": True,
    "rsi_dz_len": 5, "rsi_dz_weight": 0.4,
    # MACD (contrarian in v1.8)
    "macd_fast": 12, "macd_slow": 26, "macd_sig": 9,
    "macd_zlen": 100, "macd_dz_len": 5, "macd_dz_weight": 0.5,
    "macd_contrarian": True,
    # Weights (VWAP removed; CAPE/RSI/MACD only)
    "wt_cape": 33.0, "wt_rsi": 33.0, "wt_macd": 34.0,
    # Thresholds
    "th_sbuy": 2.0, "th_buy": 1.0, "th_sell": -1.0, "th_ssell": -2.0,
    "clamp_val": 3.0, "min_composite": 1.0, "workers": 8,
    # Divergence
    "div_enable": True, "piv_left": 5, "piv_right": 5, "div_lookback": 60,
    # Confidence
    "conf_strong": 1.75, "conf_moderate": 1.10,
    # ΔZ acceleration
    "dz_accel_enable": True, "dz_accel_bars": 2, "dz_accel_require_both": False,
    # Add_Conf gates
    "rsi_hard_max": 50.0,
    "add_conf_agree_min": 1,
    # Candle body
    "candle_body_enable": True, "candle_body_hard": False,
    "candle_green_tol": 0.998, "hammer_mult": 2.0,
    # 52-week high
    "hi52_enable": True, "hi52_bars": 252, "hi52_pct": 0.85,
    # Backtest
    "bt_min_composite": 1.25,
    "backtest_profit_pct": 8.0,
    "bt_stop_pct": 8.0, "bt_max_hold_wks": 26,
    "bt_use_cape": True, "bt_vol_scale": True,
    "bt_tgt_sigma": 1.0, "bt_stop_sigma": 0.7,
    # EPS reporting lag in days (H4 look-ahead fix)
    "eps_lag_days": 60,
    # Volatility / beta layer (v1.9)
    "vol_len": 20, "vol_zlen": 252, "beta_len": 252,
    "beta_high": 1.2, "capit_volz": 1.5,
    # CAPE Add_Conf gate threshold (was magic constant 1.73)
    "cape_gate": 1.73,
}

# ── Stock universe — merged from all watchlists (290 symbols) ──────────────────
# Sources: My_MPTDS_26 · 24MPTDS · MPTDS_MDSPORT_24_dr_ram
#          My_MDSPORT_26 · TV_CYCLICALS · TV_CYCLICALS-2 · TV_DEFENSIVES

# My MPTDS 26 watchlist
_MPTDS_26 = [
    "NESTLEIND", "IEX", "IRCTC", "ABBOTINDIA", "TRITURBINE", "BEL", "INFY", "ITC",
    "GRSE", "CUMMINSIND", "ASTRAZEN", "NBCC", "APARINDS", "CRISIL", "AJANTPHARM",
    "PERSISTENT", "HEROMOTOCO", "HEXT", "PIDILITIND", "EICHERMOT", "POLYCAB",
    "VOLTAMP", "LTTS", "SCHAEFFLER", "LTIM", "TORNTPHARM", "CHAMBLFERT", "UNITDSPR",
    "GODFRYPHLP", "BLUESTARCO", "GABRIEL", "JBCHEPHARM", "ASIANPAINT", "HAVELLS",
    "BERGEPAINT", "ZYDUSLIFE", "AVANTIFEED", "ICICIGI", "COROMANDEL", "MGL",
    "MPHASIS", "APLAPOLLO", "ZENSARTECH", "BSOFT", "TIMKEN", "GRINDWELL", "ALKEM",
    "COFORGE", "TITAN", "SUNDRMFAST", "GODREJAGRO", "BPCL", "FSL", "TVSMOTOR",
    "ASHOKLEY", "THANGAMAYL", "AEGISLOG", "CCL", "HATSUN", "APLLTD", "POWERGRID",
    "BAJFINANCE", "RECLTD", "PFC",
]

# My MDSPORT 26 watchlist
_MDSPORT_26 = [
    "SANOFICONR", "ICICIAMC", "ENRIN", "TCS", "INGERRAND", "ANANDRATHI", "CAMS",
    "IGIL", "IEX", "IRCTC", "CMPDI", "COALINDIA", "BSE", "ABBOTINDIA", "PRUDENT",
    "NATIONALUM", "CDSL", "TRITURBINE", "NAM-INDIA", "OFSS", "TRAVELFOOD", "INFY",
    "ZENTEC", "ITC", "CUMMINSIND", "TATAELXSI", "INDIAMART", "BLS", "NATCOPHARM",
    "KFINTECH", "EMAMILTD", "AJANTPHARM", "HCLTECH", "AIIL", "PERSISTENT",
    "TDPOWERSYS", "HEROMOTOCO", "ABB", "PIDILITIND", "NMDC", "LALPATHLAB", "ANTHEM",
    "CONCORDBIO", "LTTS", "ECLERX", "SCHAEFFLER", "LTIM", "HBLENGINE", "FINEORG",
    "MANYAVAR", "CAPLIPOINT", "TATATECH", "JBCHEPHARM", "SHRIPISTON", "SUMICHEM",
    "ALIVUS", "INDGN", "AVANTIFEED", "EIHOTEL", "GPIL", "PIIND", "CIPLA", "HSCL",
    "ELGIEQUIP", "PFIZER", "RATNAMANI", "VESUVIUS", "ZENSARTECH", "DATAPATTNS",
    "TIMKEN", "VIJAYA", "GRINDWELL", "VINATIORGA", "DIVISLAB", "SUNTV", "ALKEM",
    "SUNPHARMA", "ZFCVINDIA", "POLYMED", "ACUTAAS", "KPRMILL", "MEDANTA", "GALLANTT",
    "HAPPYFORGE", "AIAENG", "USHAMART", "SONACOMS", "TEGA", "OBEROIRLTY", "FINCABLES",
    "INDHOTEL", "NAVA", "CUPID", "AFFLE", "GESHIP", "BLACKBUCK",
]

# 24 MPTDS / Dr Ram combined list
_MPTDS_24 = [
    "360ONE", "ABB", "ABBOTINDIA", "ACE", "AEGISLOG", "AFFLE", "AIAENG", "AJANTPHARM",
    "AKZOINDIA", "ALKEM", "ALKYLAMINE", "APARINDS", "APLAPOLLO", "APOLLOHOSP",
    "APOLLOTYRE", "ARE&M", "ASAHIINDIA", "ASIANPAINT", "ASTRAL", "ASTRAZEN",
    "AUROPHARMA", "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BASF", "BATAINDIA",
    "BAYERCROP", "BEL", "BERGEPAINT", "BHARATFORG", "BHARTIARTL", "BLS", "BLUESTARCO",
    "BRIGADE", "BRITANNIA", "BSOFT", "CAMS", "CANFINHOME", "CAPLIPOINT", "CARBORUNIV",
    "CDSL", "CERA", "CESC", "CGCL", "CHAMBLFERT", "CHOLAFIN", "CIPLA", "CLEAN",
    "COALINDIA", "COCHINSHIP", "COFORGE", "COLPAL", "CONCOR", "CRISIL", "CUMMINSIND",
    "CYIENT", "DABUR", "DATAPATTNS", "DCMSHRIRAM", "DEEPAKNTR", "DIXON", "DMART",
    "DRREDDY", "EICHERMOT", "EIDPARRY", "ELECON", "ELECTCAST", "ELGIEQUIP",
    "ENDURANCE", "ESCORTS", "EXIDEIND", "FEDERALBNK", "FINCABLES", "FSL", "GODFRYPHLP",
    "GODREJIND", "GPPL", "GRANULES", "GRASIM", "GRINDWELL", "GSPL", "GUJGASLTD",
    "HAPPSTMNDS", "HAVELLS", "HBLPOWER", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDUNILVR", "HSCL", "HUDCO", "ICICIBANK", "INDHOTEL", "INFY",
    "IRCTC", "ITC", "JBCHEPHARM", "JBMA", "JINDALSAW", "JKCEMENT", "JKTYRE",
    "KAJARIACER", "KARURVYSYA", "KEI", "KIRLOSBROS", "KIRLOSENG", "KOTAKBANK",
    "KPITTECH", "KSB", "LALPATHLAB", "LAXMIMACH", "LINDEINDIA", "LT", "LTIM", "LTTS",
    "M&M", "MANAPPURAM", "MANKIND", "MARICO", "MARUTI", "MAZDOCK", "MOTILALOFS",
    "MPHASIS", "NAM-INDIA", "NATCOPHARM", "NCC", "NESTLEIND", "NEWGEN", "ONGC",
    "PERSISTENT", "PFC", "PIDILITIND", "PIIND", "POLYCAB", "POLYMED", "POWERGRID",
    "PRAJIND", "RADICO", "RATNAMANI", "RAYMOND", "REDINGTON", "RELIANCE", "RKFORGE",
    "RVNL", "SANOFI", "SBICARD", "SBIN", "SCHAEFFLER", "SHREECEM", "SHRIRAMFIN",
    "SIEMENS", "SKFINDIA", "SOLARINDS", "SONACOMS", "SRF", "SUNDARMFIN", "SUNDRMFAST",
    "SUNPHARMA", "SUPREMEIND", "TATACONSUM", "TATAELXSI", "TATAMOTORS", "TATASTEEL",
    "TCS", "TECHM", "TECHNOE", "TIINDIA", "TITAN", "TORNTPHARM", "TORNTPOWER",
    "TRENT", "TRIDENT", "TRITURBINE", "TTKPRESTIG", "TVSHLTD", "TVSMOTOR", "UBL",
    "ULTRACEMCO", "UNOMINDA", "VBL", "ZENSARTECH", "ZFCVINDIA",
]

# TV Cyclicals watchlist (Financials / Discretionary / Industrials / IT / Energy)
_TV_CYCLICALS = [
    # Financials
    "ABCAPITAL", "ANANDRATHI", "ANGELONE", "AUBANK", "BAJAJFINSV", "BAJFINANCE",
    "BSE", "CAMS", "CANFINHOME", "CDSL", "CHOLAFIN", "CHOLAHLDNG", "CRISIL", "CUB",
    "HDFCAMC", "HDFCBANK", "HOMEFIRST", "ICICIBANK", "ICICIGI", "KARURVYSYA",
    "KOTAKBANK", "LICHSGFIN", "MINDSPACE", "MOTILALOFS", "MUTHOOTFIN", "NAM-INDIA",
    "SBILIFE", "SBIN", "SHRIRAMFIN", "UTIAMC",
    # Consumer Discretionary
    "BLUESTARCO", "DIXON", "EICHERMOT", "ESCORTS", "HAVELLS", "HEROMOTOCO",
    "INDHOTEL", "IRCTC", "M&M", "MARUTI", "PAGEIND", "TATAMOTORS", "TITAN",
    "TVSHLTD", "TVSMOTOR", "UBL", "UNOMINDA", "VBL", "VGUARD",
    # Industrials
    "ABB", "ACE", "ADANIPORTS", "APARINDS", "APLAPOLLO", "ASHOKLEY", "BEL",
    "CARBORUNIV", "CEMPRO", "COCHINSHIP", "CUMMINSIND", "ELECON", "ELGIEQUIP",
    "ENDURANCE", "GABRIEL", "GRSE", "HAL", "HBLENGINE", "HUDCO", "INGERRAND",
    "IRCON", "JBMA", "JINDALSAW", "JKLAKSHMI", "KEI", "LT", "LTTS", "MAZDOCK",
    "NCC", "POLYCAB", "RATNAMANI", "SCHAEFFLER", "SHRIPISTON", "SOLARINDS",
    "SUNDRMFAST", "TARIL", "TECHNOE", "THERMAX", "TIINDIA", "TRITURBINE",
    "USHAMART", "VESUVIUS", "WELCORP",
    # Materials
    "ADANIENT", "ASTRAL", "BASF", "COROMANDEL", "GRASIM", "GRAVITA", "HINDALCO",
    "HINDCOPPER", "HINDZINC", "PCBL", "PIDILITIND", "PIIND", "SUPREMEIND",
    # Information Technology
    "BSOFT", "COFORGE", "FSL", "HCLTECH", "INFY", "KPITTECH", "MPHASIS", "NAUKRI",
    "NEWGEN", "OFSS", "REDINGTON", "TCS", "TIMETECHNO", "ZENSARTECH",
    # Energy & Utilities
    "CESC", "COALINDIA", "IEX", "POWERGRID", "TATAPOWER",
    # Real Estate & Conglomerate
    "ANANTRAJ", "BRIGADE", "RELIANCE",
]

# TV Defensives watchlist (Pharma / Healthcare / FMCG / Specialty)
_TV_DEFENSIVES = [
    # Pharmaceuticals
    "ABBOTINDIA", "AJANTPHARM", "ALKEM", "ASTRAZEN", "CAPLIPOINT", "CIPLA",
    "CONCORDBIO", "DRREDDY", "ERIS", "IPCALAB", "JBCHEPHARM", "NEULANDLAB",
    "TORNTPHARM", "ZYDUSLIFE",
    # Healthcare Services
    "APOLLOHOSP", "LALPATHLAB", "NH", "POLYMED",
    # FMCG & Consumer Staples
    "BRITANNIA", "CASTROLIND", "COLPAL", "DABUR", "GODFRYPHLP", "GODREJIND",
    "ITC", "LTFOODS", "TATACONSUM",
    # Specialty & Paints
    "AKZOINDIA",
    # Defense & Specialty
    "HSCL", "ZENTEC",
]

# ── Symbol corrections: variant → canonical yfinance-safe form ──────────────
# Symbols with & must be percent-encoded for yfinance URL construction
_CORRECTIONS: Dict[str, str] = {
    # & must be encoded as %26 for yfinance HTTP calls
    "M&M":      "M%26M",
    "M_M":      "M%26M",       # underscore alias
    "ARE&M":    "ARE%26M",
    "ARE_M":    "ARE%26M",     # underscore alias
    # Hyphen forms — yfinance accepts hyphens natively, no encoding needed
    "BAJAJ-AUTO": "BAJAJ-AUTO",
    "BAJAJ_AUTO": "BAJAJ-AUTO",
    "NAM-INDIA":  "NAM-INDIA",
    "NAM_INDIA":  "NAM-INDIA",
}

# ── Master universe: union of all watchlists, sorted ────────────────────────
ALL_SYMBOLS = sorted(set(
    _MPTDS_26 + _MDSPORT_26 + _MPTDS_24 + _TV_CYCLICALS + _TV_DEFENSIVES
))
INITIAL_CAPITAL = 1_000_000
POSITION_SIZE = 50_000
MAX_POSITIONS = INITIAL_CAPITAL // POSITION_SIZE

_INDIA_CPI: Dict[int, float] = {
    2013: 10.9, 2014: 6.4, 2015: 4.9, 2016: 4.5, 2017: 3.3,
    2018: 3.9, 2019: 3.7, 2020: 6.6, 2021: 5.1, 2022: 6.7,
    2023: 5.4, 2024: 4.8, 2025: 4.9, 2026: 4.5,
}


def to_yf(sym: str) -> str:
    """Convert NSE symbol to yfinance format."""
    return _CORRECTIONS.get(sym, sym) + ".NS"


# ══════════════════════════════════════════════════════════════
# CACHING & DATA FETCHING (Improvements 2-3)
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(ticker: str, period: str = "3y") -> Optional[pd.DataFrame]:
    """
    Fetch stock data with caching. Returns cleaned DataFrame or None.
    Wrapped with retry logic for reliability.
    """
    return _fetch_with_retry(ticker, period)


def _fetch_with_retry(ticker: str, period: str = "3y", max_retries: int = 3) -> Optional[pd.DataFrame]:
    """Fetch data with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            tk = yf.Ticker(ticker)
            raw = tk.history(period=period, interval="1d", auto_adjust=True)
            
            if raw.empty:
                return None
            
            # Clean and validate
            df = _clean_df(raw)
            return df
            
        except Exception as e:
            if attempt == max_retries - 1:
                return None
            # Exponential backoff: 1s, 2s, 4s
            wait_time = 2 ** attempt
            import time
            time.sleep(wait_time)
    
    return None


def _clean_df(raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Clean and validate OHLCV data."""
    if isinstance(raw.columns, pd.MultiIndex):
        fields = raw.columns.get_level_values(0).tolist()
        data = {f: raw.iloc[:, i].values for i, f in enumerate(fields)}
    else:
        data = {c: raw[c].values for c in raw.columns}
    
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in data for c in needed):
        return None
    
    idx = raw.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_convert(None)
    
    df = pd.DataFrame({c: data[c] for c in needed}, index=idx)
    df = df[df["Close"].notna()].copy()
    
    # Validate: High >= Low, High >= Open/Close, Low <= Open/Close
    df = df[(df["High"] >= df["Low"]) & 
            (df["High"] >= df["Open"]) & 
            (df["High"] >= df["Close"]) &
            (df["Low"] <= df["Open"]) &
            (df["Low"] <= df["Close"])].copy()
    
    return df if not df.empty else None


def _resample_weekly(df: pd.DataFrame, drop_incomplete: bool = True) -> pd.DataFrame:
    """Weekly OHLCV bars. v1.9 (H5): drops the current incomplete week so
    weekly z-scores and confluence flags do not repaint intra-week."""
    weekly_raw = df.resample("W").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    })
    weekly = weekly_raw[weekly_raw["Close"].notna()].copy()
    if drop_incomplete and len(weekly) >= 2:
        # Label is the week-ending Sunday; bar is complete once data
        # reaches Friday (label - 2 days).
        if df.index[-1] < weekly.index[-1] - pd.Timedelta(days=2):
            weekly = weekly.iloc[:-1]
    return weekly


@st.cache_data(ttl=3600)
def fetch_market_close(period: str = "5y") -> Optional[pd.Series]:
    """NIFTY 50 close series (beta / regime overlay). v1.9."""
    df = _fetch_with_retry("^NSEI", period)
    return df["Close"].astype(float) if df is not None else None


def _vol_ann_series(close: pd.Series, vol_len: int = 20) -> pd.Series:
    """Rolling annualised realised volatility. v1.9."""
    ret = close.astype(float).pct_change()
    return ret.rolling(vol_len).std(ddof=1) * np.sqrt(252.0)


def compute_vol_metrics(df: pd.DataFrame, market_close: Optional[pd.Series] = None,
                        c: Optional[Dict] = None) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (annualised vol, time-series vol z-score, beta vs NIFTY). v1.9.

    Vol z is a regime detector (capitulation / blow-off), NOT a directional
    signal - used as interaction flag and rank adjuster, never as a 4th
    weighted composite term."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    close = df["Close"].astype(float)
    vol = _vol_ann_series(close, c["vol_len"])
    use_len = min(c["vol_zlen"], len(vol.dropna()))
    vol_z = _zscore(vol, use_len).clip(-3.0, 3.0) if use_len >= 60 else pd.Series(np.nan, index=vol.index)
    beta = None
    if market_close is not None:
        ret = close.pct_change()
        mret = market_close.pct_change().reindex(ret.index)
        var = mret.rolling(c["beta_len"]).var()
        b = ret.rolling(c["beta_len"]).cov(mret) / var.replace(0, np.nan)
        if len(b) and not pd.isna(b.iloc[-1]):
            beta = round(float(b.iloc[-1]), 2)

    def _last(s: pd.Series) -> Optional[float]:
        if not len(s):
            return None
        v = s.iloc[-1]
        return None if (pd.isna(v) or np.isinf(v)) else float(v)

    return _last(vol), _last(vol_z), beta


@st.cache_data(ttl=3600)
def market_weekly_composite() -> Optional[float]:
    """Contrarian weekly composite (RSI_Z + MACD_Z) for NIFTY. v1.9.
    Negative => index stretched rich => caution on high-beta buys."""
    df = _fetch_with_retry("^NSEI", "5y")
    if df is None:
        return None
    weekly = _resample_weekly(df)
    sig = compute_signals(weekly, DEFAULT_CFG)
    if sig is None or sig.get("rsi_z") is None or sig.get("macd_z") is None:
        return None
    return round((sig["rsi_z"] + sig["macd_z"]) / 2.0, 3)


# ══════════════════════════════════════════════════════════════
# INDICATOR CALCULATIONS
# ══════════════════════════════════════════════════════════════

def _zscore(s: pd.Series, n: int) -> pd.Series:
    """Calculate z-score."""
    m = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std(ddof=1)
    return (s - m) / sd.replace(0.0, np.nan)


def _clamp(s: pd.Series, v: float) -> pd.Series:
    """Clamp series to [-v, v]."""
    return s.clip(-v, v)


def _wilder_rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's RMA (exponential moving average)."""
    alpha = 1.0 / n
    result = np.full(len(s), np.nan)
    valid = s.dropna()
    
    if len(valid) < n:
        return pd.Series(result, index=s.index)
    
    pos = s.index.get_loc(valid.index[0])
    seed_end = pos + n
    
    if seed_end > len(s):
        return pd.Series(result, index=s.index)
    
    result[seed_end - 1] = s.iloc[pos:seed_end].mean()
    
    for i in range(seed_end, len(s)):
        val = s.iloc[i]
        result[i] = (alpha * val + (1 - alpha) * result[i - 1]
                     if not np.isnan(val) else result[i - 1])
    
    return pd.Series(result, index=s.index)


def _rsi(close: pd.Series, n: int) -> pd.Series:
    """Calculate RSI."""
    d = close.diff()
    gain = _wilder_rma(d.clip(lower=0), n)
    loss = _wilder_rma((-d).clip(lower=0), n)
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _cpi_factor(from_year: int, to_year: int) -> float:
    """Calculate CPI adjustment factor for inflation."""
    if to_year <= from_year:
        return 1.0
    factor = 1.0
    for yr in range(from_year, to_year):
        factor *= 1.0 + _INDIA_CPI.get(yr, 5.5) / 100.0
    return max(factor, 1.0)


_USD_INR_CACHE = [None]
_FX_LOCK = threading.Lock()


def _get_usd_inr() -> float:
    """Get USD/INR rate with caching (thread-safe, v1.9)."""
    with _FX_LOCK:
        if _USD_INR_CACHE[0] is None:
            try:
                rate = yf.Ticker("USDINR=X").info.get("regularMarketPrice", None)
                _USD_INR_CACHE[0] = float(rate) if rate and 60 < float(rate) < 120 else 84.0
            except Exception:
                _USD_INR_CACHE[0] = 84.0
        return _USD_INR_CACHE[0]


def _eps_inr_factor(tk: yf.Ticker) -> float:
    """Get USD/INR conversion factor for EPS if needed."""
    try:
        info = tk.info
        fin_ccy = info.get("financialCurrency", "INR")
        price_ccy = info.get("currency", "INR")
        if fin_ccy == "USD" and price_ccy == "INR":
            return _get_usd_inr()
    except Exception:
        pass
    return 1.0


def _get_eps_series(tk: yf.Ticker, lag_days: int = 60) -> pd.Series:
    """Extract EPS series from quarterly data.

    v1.9 (H4): dates shifted forward by `lag_days` to approximate the
    announcement date, removing look-ahead from the CAPE z-window."""
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
                            eps_map[pd.Timestamp(col).normalize() + pd.Timedelta(days=lag_days)] = v * fx
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


def _cape_z_series(tk: yf.Ticker, price_df: pd.DataFrame, c: Dict) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
    """Full daily CAPE z series (contrarian-signed). v1.9 (H2/H4).
    Returns (z_series, ttm_series, close_series) or (None, None, None)."""
    eps_s = _get_eps_series(tk, int(c.get("eps_lag_days", 60)))
    if eps_s.empty:
        return None, None, None

    price_idx = price_df.index
    if hasattr(price_idx, "tz") and price_idx.tz is not None:
        price_idx = price_idx.tz_convert(None)

    close = pd.Series(price_df["Close"].astype(float).values, index=price_idx)
    eps_s = eps_s.sort_index()

    # TTM (trailing, recency-weighted, CPI-adjusted)
    n_total = len(eps_s)
    ttm_map = {}
    for i in range(n_total):
        report_date = eps_s.index[i]
        start = max(0, i + 1 - c["cape_max_q"])
        window = eps_s.iloc[start:i + 1]
        n = len(window)
        weights = np.array([max(0.1, 1.0 - (n - 1 - k) * 0.025) for k in range(n)])
        cpi_adj = np.array([_cpi_factor(window.index[k].year, report_date.year) for k in range(n)])
        total_w = weights.sum()
        if total_w <= 0:
            continue
        ttm = (window.values * cpi_adj * weights).sum() / total_w * 4.0
        if ttm > 0:
            ttm_map[report_date] = ttm

    if not ttm_map:
        return None, None, None

    ttm_s = pd.Series(ttm_map).sort_index()
    combined_idx = price_idx.union(ttm_s.index).sort_values()
    ttm_daily = ttm_s.reindex(combined_idx).ffill().reindex(price_idx)

    cape_ratio = close / ttm_daily.replace(0, np.nan)
    use_len = min(c["cape_zlen"], len(cape_ratio.dropna()))
    if use_len < 30:
        return None, None, None

    z = _zscore(cape_ratio, use_len).clip(-c["clamp_val"], c["clamp_val"])
    z_final = -z if c["cape_bearish"] else z
    return z_final, ttm_s, close


def compute_cape_z(tk: yf.Ticker, price_df: pd.DataFrame, c: Optional[Dict] = None) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute CAPE z-score, ratio, and TTM EPS (last values)."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)

    if not c["use_cape"]:
        return None, None, None

    try:
        z_final, ttm_s, close = _cape_z_series(tk, price_df, c)
        if z_final is None:
            return None, None, None

        last_z = float(z_final.iloc[-1]) if not np.isnan(z_final.iloc[-1]) else None
        last_close = float(close.iloc[-1])
        last_ttm = float(ttm_s.iloc[-1]) if not ttm_s.empty else None
        last_ratio = round(last_close / last_ttm, 2) if last_ttm and last_ttm > 0 else None

        return last_z, last_ratio, last_ttm

    except Exception:
        return None, None, None


def _pivot_low(series: pd.Series, left: int, right: int) -> pd.Series:
    """Find pivot lows."""
    n = len(series)
    result = pd.Series(np.nan, index=series.index)
    vals = series.values
    
    for i in range(left, n - right):
        v = vals[i]
        if np.isnan(v):
            continue
        window = vals[i - left:i + right + 1]
        if np.isnan(window).any():
            continue
        if v == window.min() and (window == v).sum() == 1:
            result.iloc[i] = v
    
    return result


def _pivot_high(series: pd.Series, left: int, right: int) -> pd.Series:
    """Find pivot highs."""
    n = len(series)
    result = pd.Series(np.nan, index=series.index)
    vals = series.values
    
    for i in range(left, n - right):
        v = vals[i]
        if np.isnan(v):
            continue
        window = vals[i - left:i + right + 1]
        if np.isnan(window).any():
            continue
        if v == window.max() and (window == v).sum() == 1:
            result.iloc[i] = v
    
    return result


def detect_divergence(price: pd.Series, osc: pd.Series, name: str, c: Optional[Dict] = None) -> Dict[str, Any]:
    """Detect price-oscillator divergences."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    
    out = {"reg_bull": False, "hid_bull": False, "reg_bear": False, "hid_bear": False, "tag": ""}
    
    if not c["div_enable"]:
        return out
    
    if len(price) < c["piv_left"] + c["piv_right"] + 10:
        return out
    
    L, R, LB = c["piv_left"], c["piv_right"], c["div_lookback"]
    osc = osc.reindex(price.index)
    
    p_lows = _pivot_low(price, L, R)
    p_highs = _pivot_high(price, L, R)
    o_lows = _pivot_low(osc, L, R)
    o_highs = _pivot_high(osc, L, R)
    
    last_idx = len(price) - 1
    
    # Bullish divergence
    pl_positions = [i for i, v in enumerate(p_lows.values) if not np.isnan(v)]
    if len(pl_positions) >= 2:
        i_curr, i_prev = pl_positions[-1], pl_positions[-2]
        if (last_idx - i_curr) <= LB and (i_curr - i_prev) <= LB:
            p_curr = price.iloc[i_curr]
            p_prev = price.iloc[i_prev]
            o_curr = _nearest_pivot_value(o_lows, i_curr, R)
            o_prev = _nearest_pivot_value(o_lows, i_prev, R)
            
            if o_curr is not None and o_prev is not None:
                if p_curr < p_prev and o_curr > o_prev:
                    out["reg_bull"] = True
                elif p_curr > p_prev and o_curr < o_prev:
                    out["hid_bull"] = True
    
    # Bearish divergence
    ph_positions = [i for i, v in enumerate(p_highs.values) if not np.isnan(v)]
    if len(ph_positions) >= 2:
        i_curr, i_prev = ph_positions[-1], ph_positions[-2]
        if (last_idx - i_curr) <= LB and (i_curr - i_prev) <= LB:
            p_curr = price.iloc[i_curr]
            p_prev = price.iloc[i_prev]
            o_curr = _nearest_pivot_value(o_highs, i_curr, R)
            o_prev = _nearest_pivot_value(o_highs, i_prev, R)
            
            if o_curr is not None and o_prev is not None:
                if p_curr > p_prev and o_curr < o_prev:
                    out["reg_bear"] = True
                elif p_curr < p_prev and o_curr > o_prev:
                    out["hid_bear"] = True
    
    parts = []
    if out["reg_bull"]:
        parts.append(f"BullReg({name})")
    if out["hid_bull"]:
        parts.append(f"BullHid({name})")
    if out["reg_bear"]:
        parts.append(f"BearReg({name})")
    if out["hid_bear"]:
        parts.append(f"BearHid({name})")
    
    out["tag"] = " | ".join(parts)
    return out


def _nearest_pivot_value(piv_series: pd.Series, target_idx: int, tol: int) -> Optional[float]:
    """Find nearest pivot value within tolerance."""
    n = len(piv_series)
    lo = max(0, target_idx - tol)
    hi = min(n - 1, target_idx + tol)
    best_val, best_dist = None, None
    
    for j in range(lo, hi + 1):
        v = piv_series.iloc[j]
        if not np.isnan(v):
            d = abs(j - target_idx)
            if best_dist is None or d < best_dist:
                best_dist, best_val = d, float(v)
    
    return best_val


def _hi52_ok_series(high: pd.Series, close: pd.Series, c: Optional[Dict] = None) -> pd.Series:
    """Check if close is within 52W high threshold."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    
    roll_max = high.rolling(c["hi52_bars"], min_periods=max(20, c["hi52_bars"] // 4)).max()
    ratio = close / roll_max.replace(0, np.nan)
    return ratio <= c["hi52_pct"]


def _hi52_ok_last(high: pd.Series, close: pd.Series, c: Optional[Dict] = None) -> bool:
    """Check if last bar passes 52W high test."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    
    if not c["hi52_enable"]:
        return True
    
    s = _hi52_ok_series(high, close, c)
    last = s.iloc[-1]
    return bool(last) if not pd.isna(last) else False


def compute_signals(df: pd.DataFrame, c: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """Compute RSI, MACD, and momentum signals."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    
    min_bars = max(c["rsi_zlen"], c["macd_zlen"]) + c["macd_slow"] + 30
    if len(df) < min_bars:
        return None
    
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float).replace(0, np.nan)
    
    # ── RSI Z (contrarian) ───────────────────────────────────
    rsi_val = _rsi(close, c["rsi_len"])
    rsi_lz = _zscore(rsi_val, c["rsi_zlen"])
    rsi_dz = rsi_lz.diff(c["rsi_dz_len"]).rolling(2).mean()
    rsi_comb = rsi_lz * (1.0 - c["rsi_dz_weight"]) + rsi_dz * c["rsi_dz_weight"]
    rsi_z = _clamp(-rsi_comb if c["rsi_contrarian"] else rsi_comb, c["clamp_val"])
    rsi_dz_accel = rsi_dz.diff(c["dz_accel_bars"])
    
    # ── MACD% Z (contrarian) ─────────────────────────────────
    ema_f = close.ewm(span=c["macd_fast"], adjust=False, min_periods=c["macd_fast"]).mean()
    ema_s = close.ewm(span=c["macd_slow"], adjust=False, min_periods=c["macd_slow"]).mean()
    macd_hist = (ema_f - ema_s) - (ema_f - ema_s).ewm(span=c["macd_sig"], adjust=False, min_periods=c["macd_sig"]).mean()
    macd_pct = macd_hist / close.replace(0, np.nan) * 100.0
    macd_lz = _zscore(macd_pct, c["macd_zlen"])
    macd_dz = macd_lz.diff(c["macd_dz_len"]).rolling(2).mean()
    macd_comb = macd_lz * (1.0 - c["macd_dz_weight"]) + macd_dz * c["macd_dz_weight"]
    macd_z = _clamp(-macd_comb if c["macd_contrarian"] else macd_comb, c["clamp_val"])
    macd_dz_accel = macd_dz.diff(c["dz_accel_bars"])
    
    div_rsi = detect_divergence(close, rsi_lz, "RSI_Z", c)
    div_macd = detect_divergence(close, macd_lz, "MACD_Z", c)
    hi52_pass = _hi52_ok_last(high, close, c)
    
    def _f(s: pd.Series) -> Optional[float]:
        v = s.iloc[-1]
        return round(float(v), 3) if not (np.isnan(v) or np.isinf(v)) else None
    
    return {
        "close": round(float(close.iloc[-1]), 2),
        "rsi_val": round(float(rsi_val.iloc[-1]), 1) if not np.isnan(rsi_val.iloc[-1]) else None,
        "rsi_z": _f(rsi_z),
        "macd_z": _f(macd_z),
        "rsi_dz": _f(rsi_dz),
        "macd_dz": _f(macd_dz),
        "rsi_dz_accel": _f(rsi_dz_accel),
        "macd_dz_accel": _f(macd_dz_accel),
        "hi52_pass": hi52_pass,
        "div_rsi": div_rsi,
        "div_macd": div_macd,
    }


def _composite(sig: Dict[str, Any], cape_z: Optional[float], c: Optional[Dict] = None) -> Tuple[Optional[float], bool]:
    """Compute composite score from components."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    
    rz = sig.get("rsi_z")
    mz = sig.get("macd_z")
    
    if any(z is None for z in [rz, mz]):
        return None, False
    
    cape_active = cape_z is not None and c["use_cape"]
    
    if cape_active:
        tot = c["wt_cape"] + c["wt_rsi"] + c["wt_macd"]
        raw = (cape_z * c["wt_cape"] + rz * c["wt_rsi"] + mz * c["wt_macd"]) / tot
    else:
        tot = c["wt_rsi"] + c["wt_macd"]
        raw = (rz * c["wt_rsi"] + mz * c["wt_macd"]) / tot
    
    clamped = float(_clamp(pd.Series([raw]), c["clamp_val"]).iloc[0])
    return round(clamped, 3), cape_active


def verdict(z: Optional[float], c: Optional[Dict] = None) -> str:
    """Determine verdict from composite score."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    
    if z is None:
        return "N/A"
    if z >= c["th_sbuy"]:
        return "STRONG BUY"
    if z >= c["th_buy"]:
        return "BUY"
    if z <= c["th_ssell"]:
        return "STRONG SELL"
    if z <= c["th_sell"]:
        return "SELL"
    return "NEUTRAL"


def confidence(comp: Optional[float], sig: Dict[str, Any], cape_z: Optional[float], cape_used: bool, c: Optional[Dict] = None) -> str:
    """Determine confidence level."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    
    if comp is None:
        return ""
    
    v = verdict(comp, c)
    if v in ("N/A", "NEUTRAL"):
        return ""
    
    direction = 1 if comp > 0 else -1
    components = [sig.get("rsi_z"), sig.get("macd_z")]
    if cape_used and cape_z is not None:
        components.append(cape_z)
    
    agree = sum(1 for z in components if z is not None and ((z > 0 and direction > 0) or (z < 0 and direction < 0)))
    abs_c = abs(comp)
    
    if abs_c >= c["conf_strong"] and agree >= min(3, len(components)):
        return "STRONG"
    if abs_c >= c["conf_moderate"] and agree >= 2:
        return "MODERATE"
    return "WEAK"


def _add_conf(cape_z: Optional[float], cape_used: bool, rsi_val: Optional[float], agree: int, c: Optional[Dict] = None, direction: int = 1) -> bool:
    """Check additional confirmation gate. v1.9: direction-aware, so the
    gate is meaningful on the SELL side too (mirrored thresholds)."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)

    if rsi_val is None:
        return False
    if direction >= 0:
        if float(rsi_val) >= c["rsi_hard_max"]:
            return False
    else:
        if float(rsi_val) <= 100.0 - c["rsi_hard_max"]:
            return False
    if agree <= c["add_conf_agree_min"]:
        return False
    if cape_used and cape_z is not None:
        if direction >= 0 and float(cape_z) <= c["cape_gate"]:
            return False
        if direction < 0 and float(cape_z) >= -c["cape_gate"]:
            return False

    return True


def _dz_accel_ok(sig: Dict[str, Any], c: Optional[Dict] = None) -> bool:
    """Check ΔZ acceleration filter."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    
    if not c["dz_accel_enable"]:
        return True
    
    rsi_acc = sig.get("rsi_dz_accel")
    macd_acc = sig.get("macd_dz_accel")
    
    if rsi_acc is None and macd_acc is None:
        return True
    
    rsi_ok = (rsi_acc is None) or (float(rsi_acc) > 0)
    macd_ok = (macd_acc is None) or (float(macd_acc) > 0)
    
    if c.get("dz_accel_require_both", False):
        return rsi_ok and macd_ok
    return rsi_ok or macd_ok


def _candle_ok(open_price: float, high_price: float, low_price: float, close_price: float, c: Optional[Dict] = None) -> bool:
    """Check candle body qualifications."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    
    if not c["candle_body_enable"]:
        return True
    
    body = abs(close_price - open_price)
    lower_wick = min(open_price, close_price) - low_price
    green = close_price >= open_price * c["candle_green_tol"]
    hammer = lower_wick >= c["hammer_mult"] * body if body > 0 else lower_wick > 0
    qualifies = green or hammer
    
    if c.get("candle_body_hard", False):
        return qualifies
    return True


# ══════════════════════════════════════════════════════════════
# SCANNING & BACKTESTING
# ══════════════════════════════════════════════════════════════

def scan_stock(sym_raw: str, c: Optional[Dict] = None) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    """Scan single stock for signals (daily + weekly). v1.9."""
    if c is None:
        c = DEFAULT_CFG

    ticker = to_yf(sym_raw)
    rows = []

    try:
        df = fetch_stock_data(ticker, "5y")   # v1.9: 5y so weekly z has depth (M7)
        if df is None:
            return sym_raw, [], "⚠️ No data available"

        if len(df) < 100:
            return sym_raw, [], "⚠️ Insufficient bars (< 100)"

        tk = yf.Ticker(ticker)
        cape_z, cape_ratio, ttm_eps = compute_cape_z(tk, df, c)
        d_sig = compute_signals(df, c)

        weekly = _resample_weekly(df)          # v1.9 (H5): incomplete week dropped
        w_sig = compute_signals(weekly, c)

        # v1.9: volatility / beta / market regime layer
        mkt_close = fetch_market_close("5y")
        vol_ann, vol_z, beta = compute_vol_metrics(df, mkt_close, c)
        mkt_comp = market_weekly_composite()

        for tf, sig, src_df in [("Daily", d_sig, df), ("Weekly", w_sig, weekly)]:
            if sig is None:
                continue

            comp, cape_used = _composite(sig, cape_z, c)
            if comp is None:
                continue

            vrd = verdict(comp, c)
            conf = confidence(comp, sig, cape_z, cape_used, c)

            _dir = 1 if comp >= 0 else -1
            _ac_zs = [sig["rsi_z"], sig["macd_z"]]
            if cape_used and cape_z is not None:
                _ac_zs.append(cape_z)
            _ac_agree = sum(1 for z in _ac_zs if z is not None and
                            ((float(z) > 0) if _dir > 0 else (float(z) < 0)))

            ac = _add_conf(cape_z, cape_used, sig["rsi_val"], _ac_agree, c, _dir)
            dz_acc_ok = _dz_accel_ok(sig, c)
            hi52_ok = sig.get("hi52_pass", True)

            last = src_df.iloc[-1]
            candle_ok = _candle_ok(float(last["Open"]), float(last["High"]), float(last["Low"]), float(last["Close"]), c)

            div_tags = []
            if sig.get("div_rsi", {}).get("tag"):
                div_tags.append(sig["div_rsi"]["tag"])
            if sig.get("div_macd", {}).get("tag"):
                div_tags.append(sig["div_macd"]["tag"])
            div_str = " | ".join(div_tags) if div_tags else ""

            # v1.9: risk-adjusted rank, capitulation flag, high-beta caution
            risk_adj = round(comp / vol_ann, 3) if (vol_ann and vol_ann > 0) else None
            capitulation = (vol_z is not None and vol_z >= c["capit_volz"]
                            and comp >= 1.0 and dz_acc_ok)
            hibeta_risk = (beta is not None and beta > c["beta_high"]
                           and mkt_comp is not None and mkt_comp < 0 and comp > 0)

            rows.append({
                "Symbol": sym_raw,
                "TF": tf,
                "Signal": vrd,
                "Strength": conf,
                "Add_Conf": "YES" if ac else "NO",
                "ΔZ_Accel": "YES" if dz_acc_ok else "NO",
                "Candle_OK": "YES" if candle_ok else "NO",
                "Hi52_OK": "YES" if hi52_ok else "NO",
                "All_Gates": "YES" if (ac and dz_acc_ok and candle_ok and hi52_ok) else "NO",
                "Composite": comp,
                "RiskAdj_Comp": risk_adj,
                "Vol_Ann%": round(vol_ann * 100, 1) if vol_ann else None,
                "Vol_Z": round(vol_z, 2) if vol_z is not None else None,
                "Beta": beta,
                "Capitulation": "YES" if capitulation else "",
                "HiBeta_Risk": "⚠ YES" if hibeta_risk else "",
                "CAPE_Z": cape_z if cape_used else None,
                "CAPE_PE": cape_ratio,
                "TTM_EPS": ttm_eps,
                "RSI_Z": sig["rsi_z"],
                "MACD_Z": sig["macd_z"],
                "RSI_ΔZ": sig["rsi_dz"],
                "MACD_ΔZ": sig["macd_dz"],
                "RSI": sig["rsi_val"],
                "Close": sig["close"],
                "Divergence": div_str,
                "CAPE_Active": cape_used,
            })

        return sym_raw, rows, None

    except Exception as e:
        return sym_raw, [], f"⚠️ {type(e).__name__}: {str(e)[:50]}"


def _weekly_signal_frame(df: pd.DataFrame, c: Dict[str, Any]) -> pd.DataFrame:
    """Compute signal frame for weekly backtesting. FIXED in v1.8."""
    min_bars = max(c["rsi_zlen"], c["macd_zlen"]) + c["macd_slow"] + 30
    if len(df) < min_bars:
        return pd.DataFrame()
    
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)  # FIXED: was open*
    volume = df["Volume"].astype(float).replace(0, np.nan)
    
    # RSI Z (contrarian)
    rsi_val = _rsi(close, c["rsi_len"])
    rsi_lz = _zscore(rsi_val, c["rsi_zlen"])
    rsi_dz = rsi_lz.diff(c["rsi_dz_len"]).rolling(2).mean()
    rsi_comb = rsi_lz * (1 - c["rsi_dz_weight"]) + rsi_dz * c["rsi_dz_weight"]
    rsi_z = _clamp(-rsi_comb if c["rsi_contrarian"] else rsi_comb, c["clamp_val"])
    rsi_dz_accel = rsi_dz.diff(c["dz_accel_bars"])
    
    # MACD% Z (contrarian)
    ema_f = close.ewm(span=c["macd_fast"], adjust=False, min_periods=c["macd_fast"]).mean()
    ema_s = close.ewm(span=c["macd_slow"], adjust=False, min_periods=c["macd_slow"]).mean()
    macd_hist = (ema_f - ema_s) - (ema_f - ema_s).ewm(span=c["macd_sig"], adjust=False, min_periods=c["macd_sig"]).mean()
    macd_pct = macd_hist / close.replace(0, np.nan) * 100.0
    macd_lz = _zscore(macd_pct, c["macd_zlen"])
    macd_dz = macd_lz.diff(c["macd_dz_len"]).rolling(2).mean()
    macd_comb = macd_lz * (1 - c["macd_dz_weight"]) + macd_dz * c["macd_dz_weight"]
    macd_z = _clamp(-macd_comb if c["macd_contrarian"] else macd_comb, c["clamp_val"])
    macd_dz_accel = macd_dz.diff(c["dz_accel_bars"])
    
    hi52_ok_s = _hi52_ok_series(high, close, c)
    
    return pd.DataFrame({
        "rsi_z": rsi_z,
        "macd_z": macd_z,
        "rsi_dz": rsi_dz,
        "macd_dz": macd_dz,
        "rsi_dz_accel": rsi_dz_accel,
        "macd_dz_accel": macd_dz_accel,
        "hi52_ok": hi52_ok_s,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "rsi_val": rsi_val,
    }, index=df.index)


def backtest_one(sym_raw: str, lookback_weeks: int, profit_pct: float, c: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    """Run backtest for single stock.

    v1.9 fixes:
    H1 - re-entry allowed after a position exits (was: 1 trade/symbol ever)
    H2 - CAPE z included in the composite with live-scan weights
    H3 - stop-loss + max-hold time exits; vol-scaled target/stop option
    H5 - incomplete final weekly bar dropped
    Weekly-bar granularity caveat: if stop AND target are both touched in
    the same week, the stop is assumed to hit first (conservative)."""
    ticker = to_yf(sym_raw)
    trades = []

    try:
        df = fetch_stock_data(ticker, "5y")
        if df is None:
            return sym_raw, [], "⚠️ No data available"

        if len(df) < 200:
            return sym_raw, [], "⚠️ Insufficient bars"

        weekly = _resample_weekly(df)
        sf = _weekly_signal_frame(weekly, c)
        if sf.empty:
            return sym_raw, [], "⚠️ Insufficient bars for signals"

        # ── CAPE z aligned to weekly bars (H2) ─────────────────────
        cape_w = None
        if c["use_cape"] and c.get("bt_use_cape", True):
            try:
                tk = yf.Ticker(ticker)
                z_daily, _, _ = _cape_z_series(tk, df, c)
                if z_daily is not None:
                    cape_w = (z_daily.reindex(z_daily.index.union(weekly.index))
                              .ffill().reindex(weekly.index))
            except Exception:
                cape_w = None

        # ── Annualised vol aligned to weekly bars (vol-scaled exits) ──
        vol_w = None
        if c.get("bt_vol_scale", True):
            vd = _vol_ann_series(df["Close"], c["vol_len"])
            vol_w = (vd.reindex(vd.index.union(weekly.index))
                     .ffill().reindex(weekly.index))

        total_w = len(weekly)
        scan_start = max(0, total_w - lookback_weeks - 1)
        scan_end = total_w - 1
        max_hold = int(c.get("bt_max_hold_wks", 26))
        open_until = -1   # bar index at which the current position exited

        for i in range(scan_start, scan_end):
            if i <= open_until:
                continue   # H1 fix: block only OVERLAPPING positions

            rz_w = sf["rsi_z"].iloc[i]
            mz_w = sf["macd_z"].iloc[i]
            if any(pd.isna(x) for x in (rz_w, mz_w)):
                continue

            cz_w = None
            if cape_w is not None and not pd.isna(cape_w.iloc[i]):
                cz_w = float(cape_w.iloc[i])

            # ── Composite: SAME weights as the live scan (H2) ──
            if cz_w is not None:
                tot = c["wt_cape"] + c["wt_rsi"] + c["wt_macd"]
                comp_w = (cz_w * c["wt_cape"] + float(rz_w) * c["wt_rsi"]
                          + float(mz_w) * c["wt_macd"]) / tot
            else:
                tot = c["wt_rsi"] + c["wt_macd"]
                comp_w = (float(rz_w) * c["wt_rsi"] + float(mz_w) * c["wt_macd"]) / tot
            comp_w = float(np.clip(comp_w, -c["clamp_val"], c["clamp_val"]))

            if comp_w < c["bt_min_composite"]:
                continue

            vrd_w = verdict(comp_w, c)
            if vrd_w not in ("BUY", "STRONG BUY"):
                continue

            sig_w = {"rsi_z": float(rz_w), "macd_z": float(mz_w)}
            conf_w = confidence(comp_w, sig_w, cz_w, cz_w is not None, c)
            if conf_w not in ("MODERATE", "STRONG"):
                continue

            bar_open = float(sf["open"].iloc[i])
            bar_high = float(sf["high"].iloc[i])
            bar_low = float(sf["low"].iloc[i])
            bar_close = float(sf["close"].iloc[i])
            candle_ok_flag = _candle_ok(bar_open, bar_high, bar_low, bar_close, c)

            # ── Vol-scaled or flat target/stop (H3) ────────────
            v_i = None
            if vol_w is not None and not pd.isna(vol_w.iloc[i]):
                v_i = float(vol_w.iloc[i])
            if v_i and v_i > 0 and c.get("bt_vol_scale", True):
                sigma_h = v_i / np.sqrt(52.0) * np.sqrt(max_hold)   # vol over max-hold horizon
                tgt_pct = float(np.clip(c["bt_tgt_sigma"] * sigma_h * 100.0, 4.0, 25.0))
                stp_pct = float(np.clip(c["bt_stop_sigma"] * sigma_h * 100.0, 4.0, 20.0))
            else:
                tgt_pct = float(profit_pct)
                stp_pct = float(c.get("bt_stop_pct", 8.0))

            entry_date = weekly.index[i]
            entry_price = bar_close
            target_price = round(entry_price * (1 + tgt_pct / 100), 2)
            stop_price = round(entry_price * (1 - stp_pct / 100), 2)

            exit_idx, exit_price, status = None, None, "OPEN"
            for j in range(i + 1, total_w):
                lo_j = float(sf["low"].iloc[j])
                hi_j = float(sf["high"].iloc[j])
                cl_j = float(sf["close"].iloc[j])
                if lo_j <= stop_price:          # conservative: stop first
                    exit_idx, exit_price, status = j, stop_price, "STOP"
                    break
                if hi_j >= target_price:
                    exit_idx, exit_price, status = j, target_price, "HIT"
                    break
                if j - i >= max_hold:           # time exit
                    exit_idx, exit_price, status = j, cl_j, "TIME"
                    break

            if exit_idx is None:
                last_close = float(sf["close"].iloc[-1])
                exit_price = last_close
                hold_weeks = total_w - 1 - i
                exit_date = weekly.index[-1]
                open_ret = (last_close - entry_price) / entry_price * 100
                status = f"OPEN {open_ret:+.1f}%"
                open_until = total_w            # still open - no re-entry
            else:
                exit_date = weekly.index[exit_idx]
                hold_weeks = exit_idx - i
                open_until = exit_idx           # re-entry allowed after exit (H1)

            ret_pct = round((exit_price - entry_price) / entry_price * 100, 2)

            trades.append({
                "Symbol": sym_raw,
                "Entry_Date": entry_date.strftime("%Y-%m-%d"),
                "Entry_Price": round(entry_price, 2),
                "Target": target_price,
                "Stop": stop_price,
                "Tgt_%": round(tgt_pct, 1),
                "Stop_%": round(stp_pct, 1),
                "Exit_Date": exit_date.strftime("%Y-%m-%d"),
                "Exit_Price": round(exit_price, 2),
                "Return_%": ret_pct,
                "Hold_Wks": hold_weeks,
                "Status": status,
                "W_Signal": vrd_w,
                "W_Strength": conf_w,
                "W_Comp": round(comp_w, 3),
                "CAPE_Z_W": round(cz_w, 3) if cz_w is not None else None,
                "Candle_OK": "YES" if candle_ok_flag else "NO",
            })

        return sym_raw, trades, None

    except Exception as e:
        return sym_raw, [], f"⚠️ {type(e).__name__}: {str(e)[:50]}"


# ══════════════════════════════════════════════════════════════
# PDF GENERATION
# ══════════════════════════════════════════════════════════════

def generate_scan_pdf(
    df_buy: pd.DataFrame,
    df_sell: pd.DataFrame,
    df_buy_conf: pd.DataFrame,
    df_sell_conf: pd.DataFrame,
    df_div: pd.DataFrame,
    ts_str: str,
) -> Optional[io.BytesIO]:
    """Generate scan results PDF including confluence section."""
    if not _REPORTLAB:
        return None

    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=1*cm,
                               rightMargin=1*cm, topMargin=1.2*cm, bottomMargin=1.2*cm)
    styles = getSampleStyleSheet()
    h1, h2, normal = styles["Heading1"], styles["Heading2"], styles["Normal"]
    PAGE_W = landscape(A4)[0] - 2*cm

    # ── helpers ────────────────────────────────────────────────────────────
    def _tbl(df: pd.DataFrame, col_widths: Optional[List[float]] = None) -> Table:
        rows_data = [list(df.columns)] + [
            [str(v) if v is not None else "" for v in row]
            for row in df.itertuples(index=False)
        ]
        n_c = len(df.columns)
        cw  = col_widths or [PAGE_W / n_c] * n_c
        tbl = Table(rows_data, colWidths=cw, repeatRows=1)

        ts = TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  rl_colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  rl_colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 7),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("GRID",          (0, 0), (-1, -1), 0.3, rl_colors.HexColor("#cccccc")),
            ("LEFTPADDING",   (0, 0), (-1, -1), 3),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 3),
            ("TOPPADDING",    (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [rl_colors.HexColor("#f0f4ff"), rl_colors.white]),
        ])

        sig_col = list(df.columns).index("Signal") if "Signal" in df.columns else None
        for r_idx, row in enumerate(rows_data[1:], start=1):
            if sig_col is not None:
                sig = str(row[sig_col])
                if "BUY"  in sig: ts.add("BACKGROUND", (0, r_idx), (-1, r_idx), rl_colors.HexColor("#d4edda"))
                elif "SELL" in sig: ts.add("BACKGROUND", (0, r_idx), (-1, r_idx), rl_colors.HexColor("#f8d7da"))

        # Highlight ⭐ STRONG BOTH rows in confluence tables
        cs_col = list(df.columns).index("Confluence_Str") if "Confluence_Str" in df.columns else None
        if cs_col is not None:
            for r_idx, row in enumerate(rows_data[1:], start=1):
                if "STRONG BOTH" in str(row[cs_col]):
                    ts.add("BACKGROUND", (0, r_idx), (-1, r_idx), rl_colors.HexColor("#bbf7d0"))

        tbl.setStyle(ts)
        return tbl

    def section(title: str, df: Optional[pd.DataFrame], cols: List[str], color: str = "#000000") -> List:
        elems = [Paragraph(f'<font color="{color}">{title}</font>', h2), Spacer(1, 0.2*cm)]
        if df is None or df.empty:
            elems += [Paragraph("None.", normal), Spacer(1, 0.4*cm)]
            return elems
        avail = [c for c in cols if c in df.columns]
        cw    = [PAGE_W / len(avail)] * len(avail)
        elems += [_tbl(df[avail], cw), Spacer(1, 0.5*cm)]
        return elems

    # ── Column definitions ─────────────────────────────────────────────────
    sig_cols  = ["Symbol", "TF", "Signal", "Strength", "All_Gates", "Add_Conf",
                 "Composite", "RiskAdj_Comp", "Vol_Z", "Beta",
                 "CAPE_Z", "RSI_Z", "MACD_Z", "RSI", "Close", "Divergence"]
    conf_cols = ["Symbol", "Confluence_Str", "Combined_Comp",
                 "D_Signal", "D_Strength", "D_Composite",
                 "W_Signal", "W_Strength", "W_Composite",
                 "All_Gates", "CAPE_Z", "RSI_Z", "MACD_Z", "Close", "Divergence"]
    div_cols  = ["Symbol", "TF", "Signal", "Strength", "Composite", "Close", "Divergence"]

    # ── Build story ────────────────────────────────────────────────────────
    story = [
        Paragraph("SHANTANU'S VALUE MOMENTUM SWING TRADING SCANNER v1.9", h1),
        Paragraph(f"NSE  |  {ts_str}  |  292 stocks  |  Daily + Weekly", normal),
        Spacer(1, 0.6*cm),
    ]

    # 1 ── Dual-TF Confluence (Page 1 — highest priority)
    story.append(Paragraph("⚡ DUAL-TF CONFLUENCE — Highest-Conviction Setups", h2))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Stocks where BOTH Daily and Weekly produce a BUY or SELL signal simultaneously. "
        "⭐ STRONG BOTH = highest-priority entries (green rows).",
        normal,
    ))
    story.append(Spacer(1, 0.3*cm))

    story += section(f"BUY Confluence  ({len(df_buy_conf)} stocks)", df_buy_conf, conf_cols, "#065f46")
    story += section(f"SELL Confluence ({len(df_sell_conf)} stocks)", df_sell_conf, conf_cols, "#991b1b")
    story.append(PageBreak())

    # 2 ── All BUY / SELL by timeframe
    for tf in ["Daily", "Weekly"]:
        sb = df_buy[df_buy["TF"] == tf]  if not df_buy.empty  else pd.DataFrame()
        ss = df_sell[df_sell["TF"] == tf] if not df_sell.empty else pd.DataFrame()
        story += section(f"BUY  — {tf}  ({len(sb)} stocks)",  sb,  sig_cols, "#065f46")
        story += section(f"SELL — {tf}  ({len(ss)} stocks)",  ss,  sig_cols, "#991b1b")

    story.append(PageBreak())

    # 3 ── Divergence
    story += section(f"📡 Divergence Signals ({len(df_div)} entries)", df_div, div_cols, "#5b21b6")

    doc.build(story)
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════════
# UI HELPERS & STYLING
# ══════════════════════════════════════════════════════════════

def _load_css() -> None:
    """Load custom CSS styling."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #1a1f2e;
    }
    
    .stApp { background: #f4f6fb; }
    
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.30) !important;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    }
    
    .metric-value {
        font-size: 30px;
        font-weight: 800;
        margin-top: 6px;
    }
    
    .metric-green { color: #059669; }
    .metric-red { color: #dc2626; }
    .metric-blue { color: #2563eb; }
    
    .section-header {
        font-size: 17px;
        font-weight: 800;
        padding: 20px 0 10px 0;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        border-radius: 10px;
        padding: 16px 20px;
        font-size: 12px;
        color: #1e40af;
        margin: 12px 0;
    }
    </style>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, color_class: str = "metric-blue") -> str:
    """Generate HTML metric card."""
    return f"""
    <div class="metric-card">
        <div style="font-size:10.5px;color:#94a3b8;text-transform:uppercase;letter-spacing:1.6px;font-weight:500">{label}</div>
        <div class="metric-value {color_class}">{value}</div>
    </div>
    """


def _build_confluence_df(
    syms: List[str],
    df_buy_or_sell: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a dual-timeframe confluence DataFrame.

    For each symbol that appears in BOTH Daily and Weekly signal sets,
    produce one merged row with columns from both timeframes plus a
    combined composite score for ranking.
    """
    rows: List[Dict[str, Any]] = []

    for sym in syms:
        d = df_buy_or_sell[(df_buy_or_sell["Symbol"] == sym) & (df_buy_or_sell["TF"] == "Daily")]
        w = df_buy_or_sell[(df_buy_or_sell["Symbol"] == sym) & (df_buy_or_sell["TF"] == "Weekly")]

        if d.empty or w.empty:
            continue

        d, w = d.iloc[0], w.iloc[0]

        # Combined composite: average of daily + weekly composites
        d_comp = float(d["Composite"]) if d["Composite"] is not None else 0.0
        w_comp = float(w["Composite"]) if w["Composite"] is not None else 0.0
        combined = round((d_comp + w_comp) / 2.0, 3)

        # Badge: STRONG only if BOTH are STRONG; MODERATE if at least one is
        d_str = d.get("Strength", "")
        w_str = w.get("Strength", "")
        if d_str == "STRONG" and w_str == "STRONG":
            combined_strength = "⭐ STRONG BOTH"
        elif "STRONG" in (d_str, w_str):
            combined_strength = "STRONG / MODERATE"
        elif d_str == "MODERATE" and w_str == "MODERATE":
            combined_strength = "MODERATE BOTH"
        else:
            combined_strength = f"{d_str} / {w_str}".strip(" /")

        rows.append({
            "Symbol":          sym,
            "Combined_Comp":   combined,
            "D_Signal":        d["Signal"],
            "D_Strength":      d_str,
            "D_Composite":     round(d_comp, 3),
            "W_Signal":        w["Signal"],
            "W_Strength":      w_str,
            "W_Composite":     round(w_comp, 3),
            "Confluence_Str":  combined_strength,
            "All_Gates":       d.get("All_Gates", ""),
            "RiskAdj_Comp":    d.get("RiskAdj_Comp"),
            "Vol_Z":           d.get("Vol_Z"),
            "Beta":            d.get("Beta"),
            "Capitulation":    d.get("Capitulation", ""),
            "CAPE_Z":          d.get("CAPE_Z"),
            "RSI_Z":           d.get("RSI_Z"),
            "MACD_Z":          d.get("MACD_Z"),
            "RSI":             d.get("RSI"),
            "Close":           d.get("Close"),
            "Divergence":      (d.get("Divergence", "") or "") + (" | " + w.get("Divergence", "") if w.get("Divergence") else ""),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Sort: higher combined composite first for BUY; lower first for SELL
    ascending = df["Combined_Comp"].iloc[0] < 0 if len(df) > 0 else False
    return df.sort_values("Combined_Comp", ascending=ascending).reset_index(drop=True)


def _confluence_row_html(row: pd.Series, direction: str) -> str:
    """
    Render a single confluence stock as a rich HTML card row.
    direction: 'buy' or 'sell'
    NOTE: No HTML comments inside the return string — Streamlit markdown
    breaks on HTML comment syntax, rendering raw source instead of styled HTML.
    """
    is_strong_both = "STRONG BOTH" in str(row.get("Confluence_Str", ""))
    border_col = "#059669" if direction == "buy" else "#dc2626"
    bg_col     = "#f0fdf4" if direction == "buy" else "#fff5f5"
    badge_bg   = "#065f46" if direction == "buy" else "#991b1b"

    star   = "⭐ " if is_strong_both else ""
    d_sig  = row.get("D_Signal",     "—")
    w_sig  = row.get("W_Signal",     "—")
    d_comp = row.get("D_Composite",  "—")
    w_comp = row.get("W_Composite",  "—")
    comb   = row.get("Combined_Comp","—")
    close  = row.get("Close",        "—")
    rsi    = row.get("RSI",          "—")
    gates  = row.get("All_Gates",    "")
    div    = row.get("Divergence",   "") or ""
    cape_z = row.get("CAPE_Z")
    cape_s = f"{cape_z:.2f}" if cape_z is not None else "—"
    conf_s = row.get("Confluence_Str", "")
    symbol = row.get("Symbol", "")

    gates_badge = (
        '<span style="background:#065f46;color:#6ee7b7;border-radius:4px;padding:2px 6px;font-size:10px;font-weight:700">ALL GATES ✓</span>'
        if gates == "YES" else
        '<span style="background:#78350f;color:#fde68a;border-radius:4px;padding:2px 6px;font-size:10px;font-weight:700">PARTIAL GATES</span>'
    )
    div_badge = (
        f'<span style="background:#312e81;color:#c7d2fe;border-radius:4px;padding:2px 6px;font-size:10px">📡 {div[:35]}</span>'
        if div else ""
    )
    strong_glow = f"box-shadow:0 0 0 2px {border_col}55;" if is_strong_both else ""

    return (
        f'<div style="background:{bg_col};border:1px solid #e2e8f0;border-left:5px solid {border_col};'
        f'border-radius:12px;padding:14px 18px;margin-bottom:8px;{strong_glow}">'
        f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:10px">'

        f'<span style="background:{badge_bg};color:#fff;border-radius:8px;'
        f'padding:5px 14px;font-size:14px;font-weight:800;min-width:110px;text-align:center">'
        f'{star}{symbol}</span>'

        f'<div style="display:flex;flex-direction:column;gap:3px">'
        f'<span style="font-size:10px;color:#94a3b8;font-weight:600">DAILY</span>'
        f'<span style="background:{border_col}22;color:{border_col};border:1px solid {border_col}55;'
        f'border-radius:20px;padding:2px 10px;font-size:11px;font-weight:700">'
        f'{d_sig} &nbsp;({d_comp})</span></div>'

        f'<div style="display:flex;flex-direction:column;gap:3px">'
        f'<span style="font-size:10px;color:#94a3b8;font-weight:600">WEEKLY</span>'
        f'<span style="background:{border_col}22;color:{border_col};border:1px solid {border_col}55;'
        f'border-radius:20px;padding:2px 10px;font-size:11px;font-weight:700">'
        f'{w_sig} &nbsp;({w_comp})</span></div>'

        f'<div style="display:flex;flex-direction:column;gap:3px">'
        f'<span style="font-size:10px;color:#94a3b8;font-weight:600">CONFLUENCE</span>'
        f'<span style="font-size:12px;font-weight:700;color:#1e40af">{conf_s}</span></div>'

        f'<div style="margin-left:auto;display:flex;flex-direction:column;align-items:flex-end;gap:3px">'
        f'<span style="font-size:11px;color:#374151">&#8377;{close} &nbsp;&#183;&nbsp; RSI {rsi} &nbsp;&#183;&nbsp; CAPE_Z {cape_s}</span>'
        f'<span style="font-size:12px;font-weight:700;color:#1e40af">&#9889; Combined Z: {comb}</span>'
        f'</div>'

        f'<div style="width:100%;display:flex;gap:6px;margin-top:4px;flex-wrap:wrap">'
        f'{gates_badge}{div_badge}</div>'

        f'</div></div>'
    )


def render_confluence_section(
    df_buy_conf: pd.DataFrame,
    df_sell_conf: pd.DataFrame,
) -> None:
    """
    Render the full Dual-TF Confluence section in the scan report.
    Includes header, summary, card view and dataframe table for both
    BUY confluence and SELL confluence.
    """
    n_buy  = len(df_buy_conf)
    n_sell = len(df_sell_conf)

    # ── Section header ────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1e3a8a,#312e81);
                border-radius:14px;padding:22px 28px;margin:28px 0 20px 0;
                display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px">
      <div>
        <div style="font-size:10px;color:rgba(255,255,255,0.6);letter-spacing:3px;
                    text-transform:uppercase;margin-bottom:6px">High-Conviction Setups</div>
        <div style="font-size:22px;font-weight:800;color:#fff">⚡ Dual-TF Confluence</div>
        <div style="font-size:12px;color:rgba(255,255,255,0.7);margin-top:4px">
          Daily &amp; Weekly signals agree — highest-probability setups
        </div>
      </div>
      <div style="display:flex;gap:12px">
        <div style="background:rgba(255,255,255,0.12);border-radius:10px;padding:12px 20px;text-align:center">
          <div style="font-size:28px;font-weight:800;color:#6ee7b7">{n_buy}</div>
          <div style="font-size:10px;color:rgba(255,255,255,0.7);letter-spacing:1px">BUY</div>
        </div>
        <div style="background:rgba(255,255,255,0.12);border-radius:10px;padding:12px 20px;text-align:center">
          <div style="font-size:28px;font-weight:800;color:#fca5a5">{n_sell}</div>
          <div style="font-size:10px;color:rgba(255,255,255,0.7);letter-spacing:1px">SELL</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── What is confluence? ────────────────────────────────────────
    st.markdown("""
    <div style="background:#eff6ff;border-left:4px solid #2563eb;border-radius:8px;
                padding:12px 18px;margin-bottom:20px;font-size:12px;color:#1e40af;line-height:1.9">
      <b>📐 Confluence logic:</b>
      A stock qualifies when it independently generates a BUY (or SELL) signal on
      <b>both</b> the Daily AND Weekly timeframe.
      ⭐ <b>Strong Both</b> = STRONG signal on both TFs — highest-conviction entry.
    </div>
    """, unsafe_allow_html=True)

    # ── BUY CONFLUENCE ────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:16px;font-weight:800;color:#065f46;
                padding:14px 0 10px 0;border-bottom:2px solid #d1fae5;margin-bottom:14px">
      🟢 BUY Confluence Stocks
    </div>
    """, unsafe_allow_html=True)

    if df_buy_conf.empty:
        st.info("No BUY confluence stocks at current filter settings.")
    else:
        # Card view
        with st.expander("📋 Card View (BUY)", expanded=True):
            for _, row in df_buy_conf.iterrows():
                st.markdown(_confluence_row_html(row, "buy"), unsafe_allow_html=True)

        # Table view
        with st.expander("🗂 Table View (BUY)"):
            table_cols = ["Symbol", "Confluence_Str", "Combined_Comp",
                          "D_Signal", "D_Strength", "D_Composite",
                          "W_Signal", "W_Strength", "W_Composite",
                          "All_Gates", "RiskAdj_Comp", "Vol_Z", "Beta", "Capitulation",
                          "CAPE_Z", "RSI_Z", "MACD_Z",
                          "RSI", "Close", "Divergence"]
            avail = [c for c in table_cols if c in df_buy_conf.columns]
            st.dataframe(
                df_buy_conf[avail].reset_index(drop=True),
                use_container_width=True,
                height=min(500, 45 + 38 * len(df_buy_conf)),
            )

    st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:20px 0'>", unsafe_allow_html=True)

    # ── SELL CONFLUENCE ───────────────────────────────────────────
    st.markdown("""
    <div style="font-size:16px;font-weight:800;color:#991b1b;
                padding:14px 0 10px 0;border-bottom:2px solid #fee2e2;margin-bottom:14px">
      🔴 SELL Confluence Stocks
    </div>
    """, unsafe_allow_html=True)

    if df_sell_conf.empty:
        st.info("No SELL confluence stocks at current filter settings.")
    else:
        with st.expander("📋 Card View (SELL)", expanded=True):
            for _, row in df_sell_conf.iterrows():
                st.markdown(_confluence_row_html(row, "sell"), unsafe_allow_html=True)

        with st.expander("🗂 Table View (SELL)"):
            table_cols = ["Symbol", "Confluence_Str", "Combined_Comp",
                          "D_Signal", "D_Strength", "D_Composite",
                          "W_Signal", "W_Strength", "W_Composite",
                          "All_Gates", "RiskAdj_Comp", "Vol_Z", "Beta", "Capitulation",
                          "CAPE_Z", "RSI_Z", "MACD_Z",
                          "RSI", "Close", "Divergence"]
            avail = [c for c in table_cols if c in df_sell_conf.columns]
            st.dataframe(
                df_sell_conf[avail].reset_index(drop=True),
                use_container_width=True,
                height=min(500, 45 + 38 * len(df_sell_conf)),
            )


# ══════════════════════════════════════════════════════════════
# SIDEBAR CONFIGURATION (Improvement 5)
# ══════════════════════════════════════════════════════════════

def render_sidebar() -> Dict[str, Any]:
    """Render sidebar and return config. Added validation in v1.8."""
    with st.sidebar:
        st.markdown('<div style="font-size:18px;font-weight:800;color:#1e3a8a;margin-bottom:16px">⚙️ Scanner Config</div>', unsafe_allow_html=True)
        
        with st.expander("🔬 Filters & Thresholds", expanded=True):
            workers = st.slider("Parallel workers", 4, 16, DEFAULT_CFG["workers"])
            min_composite = st.slider("Min |Composite|", 0.5, 2.5, DEFAULT_CFG["min_composite"], 0.05)
            rsi_hard_max = st.slider("RSI hard gate (<)", 30, 60, int(DEFAULT_CFG["rsi_hard_max"]))
            hi52_pct = st.slider("52W high % max", 0.5, 1.0, DEFAULT_CFG["hi52_pct"], 0.01)
            bt_min_comp = st.slider("BT composite floor", 0.5, 2.5, DEFAULT_CFG["bt_min_composite"], 0.05)
        
        with st.expander("📊 Indicator Weights"):
            wt_cape = st.slider("CAPE weight", 0, 50, int(DEFAULT_CFG["wt_cape"]))
            wt_rsi = st.slider("RSI weight", 0, 50, int(DEFAULT_CFG["wt_rsi"]))
            wt_macd = st.slider("MACD weight", 0, 50, int(DEFAULT_CFG["wt_macd"]))
            
            # IMPROVEMENT 5: Validate weight sum
            total_wt = wt_cape + wt_rsi + wt_macd
            if total_wt == 0:
                st.error("❌ Sum of weights must be > 0")
            elif total_wt != 100:
                st.warning(f"⚠️ Weights sum to {total_wt} (not 100). Will normalize proportionally.")
        
        with st.expander("🔀 Feature Toggles"):
            use_cape = st.checkbox("Use CAPE", DEFAULT_CFG["use_cape"])
            div_enable = st.checkbox("Divergence detection", DEFAULT_CFG["div_enable"])
            dz_accel = st.checkbox("ΔZ Acceleration", DEFAULT_CFG["dz_accel_enable"])
            hi52_enable = st.checkbox("52W High gate", DEFAULT_CFG["hi52_enable"])

        with st.expander("🧪 Backtest Rules (v1.9)"):
            bt_use_cape = st.checkbox("Include CAPE in backtest", DEFAULT_CFG["bt_use_cape"])
            bt_vol_scale = st.checkbox("Vol-scaled target/stop", DEFAULT_CFG["bt_vol_scale"])
            bt_stop_pct = st.slider("Stop loss % (flat mode)", 4.0, 20.0, DEFAULT_CFG["bt_stop_pct"], 0.5)
            bt_max_hold = st.slider("Max hold (weeks)", 8, 52, DEFAULT_CFG["bt_max_hold_wks"], 2)
        
        st.markdown("---")
        st.markdown(f'📦 **Universe:** {len(ALL_SYMBOLS)} NSE stocks')
        
    cfg = {
        **DEFAULT_CFG,
        "workers": workers,
        "min_composite": min_composite,
        "rsi_hard_max": float(rsi_hard_max),
        "hi52_pct": hi52_pct,
        "bt_min_composite": bt_min_comp,
        "wt_cape": float(wt_cape),
        "wt_rsi": float(wt_rsi),
        "wt_macd": float(wt_macd),
        "use_cape": use_cape,
        "div_enable": div_enable,
        "dz_accel_enable": dz_accel,
        "hi52_enable": hi52_enable,
        "bt_use_cape": bt_use_cape,
        "bt_vol_scale": bt_vol_scale,
        "bt_stop_pct": float(bt_stop_pct),
        "bt_max_hold_wks": int(bt_max_hold),
    }
    
    st.session_state["cfg"] = cfg
    return cfg


# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════

def main():
    """Main application."""
    _load_css()
    
    # Header
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e3a8a,#0284c7);padding:36px;border-radius:18px;margin-bottom:28px;color:white">
        <div style="font-size:11px;color:rgba(255,255,255,0.65);letter-spacing:3px;text-transform:uppercase;margin-bottom:10px">📈 NSE Swing Trading</div>
        <div style="font-size:34px;font-weight:800;line-height:1.15;margin-bottom:10px">Value Momentum Scanner</div>
        <div style="font-size:14px;color:rgba(255,255,255,0.72)">CAPE · RSI Z · MACD Z · Vol/Beta Layer · Dual Timeframe Confluence</div>
        <div style="margin-top:14px;display:flex;gap:8px">
            <span style="background:rgba(255,255,255,0.18);color:white;border-radius:20px;padding:4px 14px;font-size:11px;font-weight:700">v1.9</span>
            <span style="background:rgba(255,255,255,0.18);color:white;border-radius:20px;padding:4px 14px;font-size:11px;font-weight:700">290 NSE Stocks</span>
            <span style="background:rgba(255,255,255,0.18);color:white;border-radius:20px;padding:4px 14px;font-size:11px;font-weight:700">7 Watchlists Merged</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar config
    cfg = render_sidebar()
    
    # Tabs
    tab_scan, tab_bt, tab_focus, tab_about = st.tabs(["🔍 Live Scan", "📈 Backtest", "🔭 Focus Stock", "ℹ️ About"])
    
    # ── TAB 1: LIVE SCAN ──────────────────────────────────────
    with tab_scan:
        st.markdown("### 🔍 Real-time NSE Stock Scanner")
        
        col_run, col_info = st.columns([1, 3])
        with col_run:
            run_scan = st.button("🚀 Run Live Scan", key="run_scan")
        with col_info:
            st.info("Scans all stocks for BUY/SELL signals across daily + weekly timeframes.")
        
        if run_scan or "scan_results" in st.session_state:
            if run_scan:
                for k in ["scan_results", "scan_errors", "scan_ts"]:
                    st.session_state.pop(k, None)
                
                symbols = ALL_SYMBOLS
                total = len(symbols)
                prog = st.progress(0, text=f"Scanning {total} NSE stocks…")
                status_txt = st.empty()
                all_rows, errors, done = [], [], 0
                
                with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
                    futures = {ex.submit(scan_stock, s, cfg): s for s in symbols}
                    for fut in as_completed(futures):
                        sym, rows, err = fut.result()
                        done += 1
                        if err:
                            errors.append((sym, err))
                        all_rows.extend(rows)
                        pct = done / total
                        buys = sum(1 for r in all_rows if "BUY" in r["Signal"])
                        sells = sum(1 for r in all_rows if "SELL" in r["Signal"])
                        prog.progress(pct, text=f"[{done}/{total}] {sym} — buy={buys} sell={sells}")
                
                prog.empty()
                status_txt.empty()
                
                st.session_state["scan_results"] = all_rows
                st.session_state["scan_errors"] = errors
                st.session_state["scan_ts"] = datetime.now().strftime("%d %b %Y %H:%M")
            
            all_rows = st.session_state.get("scan_results", [])
            errors = st.session_state.get("scan_errors", [])
            ts_str = st.session_state.get("scan_ts", "")
            
            if not all_rows:
                st.warning("No signals found. Try relaxing filters or check connectivity.")
            else:
                df_all = pd.DataFrame(all_rows)
                df_signal = df_all[df_all["Composite"].abs() >= cfg["min_composite"]].copy()
                df_buy  = df_signal[df_signal["Signal"].isin(["BUY", "STRONG BUY"])].copy()
                df_sell = df_signal[df_signal["Signal"].isin(["SELL", "STRONG SELL"])].copy()
                df_div  = df_all[df_all["Divergence"] != ""].copy()

                # ── Build confluence sets ──────────────────────────────
                d_buy_syms  = set(df_buy[df_buy["TF"] == "Daily"]["Symbol"])
                w_buy_syms  = set(df_buy[df_buy["TF"] == "Weekly"]["Symbol"])
                d_sell_syms = set(df_sell[df_sell["TF"] == "Daily"]["Symbol"])
                w_sell_syms = set(df_sell[df_sell["TF"] == "Weekly"]["Symbol"])

                buy_conf_syms  = sorted(d_buy_syms  & w_buy_syms)
                sell_conf_syms = sorted(d_sell_syms & w_sell_syms)

                df_buy_conf  = _build_confluence_df(buy_conf_syms,  df_buy)
                df_sell_conf = _build_confluence_df(sell_conf_syms, df_sell)

                # ── Metric cards ───────────────────────────────────────
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.markdown(metric_card("BUY Signals",     str(len(df_buy)),                        "metric-green"),  unsafe_allow_html=True)
                c2.markdown(metric_card("SELL Signals",    str(len(df_sell)),                       "metric-red"),    unsafe_allow_html=True)
                c3.markdown(metric_card("⚡ BUY Conf",    str(len(df_buy_conf)),                   "metric-blue"),   unsafe_allow_html=True)
                c4.markdown(metric_card("⚡ SELL Conf",   str(len(df_sell_conf)),                  "metric-yellow"), unsafe_allow_html=True)
                c5.markdown(metric_card("Divergences",     str(len(df_div)),                        "metric-blue"),   unsafe_allow_html=True)
                c6.markdown(metric_card("Total Signals",   str(len(df_signal)),                     "metric-blue"),   unsafe_allow_html=True)

                _mkt = market_weekly_composite()
                _regime = "⚠ risk-off for high-β buys" if (_mkt is not None and _mkt < 0) else "neutral / supportive"
                st.markdown(f'<div class="info-box">✓ Scan completed: <b>{ts_str}</b> &nbsp;·&nbsp; Universe: <b>{len(ALL_SYMBOLS)} stocks</b> &nbsp;·&nbsp; NIFTY weekly composite: <b>{_mkt}</b> ({_regime})</div>', unsafe_allow_html=True)

                # ── Display columns ────────────────────────────────────
                display_cols = ["Symbol", "TF", "Signal", "Strength", "All_Gates",
                                "Add_Conf", "ΔZ_Accel", "Candle_OK", "Hi52_OK",
                                "Composite", "RiskAdj_Comp", "Vol_Ann%", "Vol_Z",
                                "Beta", "Capitulation", "HiBeta_Risk",
                                "CAPE_Z", "RSI_Z", "MACD_Z",
                                "RSI", "Close", "Divergence"]

                # ══════════════════════════════════════════════════════
                # ⚡ DUAL-TF CONFLUENCE SECTION (top of report)
                # ══════════════════════════════════════════════════════
                render_confluence_section(df_buy_conf, df_sell_conf)

                st.markdown("<hr style='border:none;border-top:2px solid #e2e8f0;margin:28px 0'>", unsafe_allow_html=True)

                # ── BUY Signals table ──────────────────────────────────
                st.markdown('<div class="section-header" style="color:#065f46;font-size:17px;font-weight:800;padding:14px 0 10px 0;border-bottom:2px solid #d1fae5;margin-bottom:12px">🟢 All BUY Signals — Daily &amp; Weekly</div>', unsafe_allow_html=True)
                b1, b2 = st.tabs(["Daily BUY", "Weekly BUY"])
                for tf, tab_ref in [("Daily", b1), ("Weekly", b2)]:
                    sub = df_buy[df_buy["TF"] == tf]
                    if "RiskAdj_Comp" in sub.columns:
                        sub = sub.sort_values("RiskAdj_Comp", ascending=False, na_position="last")
                    with tab_ref:
                        if sub.empty:
                            st.info(f"No {tf} BUY signals.")
                        else:
                            avail = [c for c in display_cols if c in sub.columns]
                            # Highlight confluence stocks in bold
                            conf_syms = buy_conf_syms
                            sub_display = sub[avail].reset_index(drop=True).copy()
                            # Add confluence marker
                            sub_display.insert(0, "⚡Conf", sub["Symbol"].map(lambda s: "⚡ YES" if s in conf_syms else "").values)
                            st.dataframe(sub_display, use_container_width=True, height=min(500, 45 + 38 * len(sub_display)))

                st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:20px 0'>", unsafe_allow_html=True)

                # ── SELL Signals table ─────────────────────────────────
                st.markdown('<div class="section-header" style="color:#991b1b;font-size:17px;font-weight:800;padding:14px 0 10px 0;border-bottom:2px solid #fee2e2;margin-bottom:12px">🔴 All SELL Signals — Daily &amp; Weekly</div>', unsafe_allow_html=True)
                s1, s2 = st.tabs(["Daily SELL", "Weekly SELL"])
                for tf, tab_ref in [("Daily", s1), ("Weekly", s2)]:
                    sub = df_sell[df_sell["TF"] == tf]
                    sub = sub.sort_values("Composite", ascending=True)
                    with tab_ref:
                        if sub.empty:
                            st.info(f"No {tf} SELL signals.")
                        else:
                            avail = [c for c in display_cols if c in sub.columns]
                            sub_display = sub[avail].reset_index(drop=True).copy()
                            conf_syms = sell_conf_syms
                            sub_display.insert(0, "⚡Conf", sub["Symbol"].map(lambda s: "⚡ YES" if s in conf_syms else "").values)
                            st.dataframe(sub_display, use_container_width=True, height=min(500, 45 + 38 * len(sub_display)))

                st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:20px 0'>", unsafe_allow_html=True)

                # ── Divergence signals ─────────────────────────────────
                if not df_div.empty:
                    st.markdown('<div style="font-size:16px;font-weight:800;color:#5b21b6;padding:14px 0 10px 0;border-bottom:2px solid #ede9fe;margin-bottom:12px">📡 Divergence Signals</div>', unsafe_allow_html=True)
                    div_cols = ["Symbol", "TF", "Signal", "Strength", "Composite", "Close", "Divergence"]
                    avail = [c for c in div_cols if c in df_div.columns]
                    st.dataframe(df_div[avail].sort_values(["TF", "Symbol"]).reset_index(drop=True), use_container_width=True)

                # ── Downloads ──────────────────────────────────────────
                st.markdown("---")
                st.markdown("### 📥 Export Results")
                dl1, dl2, dl3, dl4 = st.columns(4)

                with dl1:
                    st.download_button(
                        "⬇️ All Signals CSV",
                        df_all.to_csv(index=False),
                        file_name=f"VMS_AllSignals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                    )
                with dl2:
                    if not df_buy_conf.empty:
                        st.download_button(
                            "⬇️ BUY Confluence CSV",
                            df_buy_conf.to_csv(index=False),
                            file_name=f"VMS_BUYConf_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                        )
                with dl3:
                    if not df_sell_conf.empty:
                        st.download_button(
                            "⬇️ SELL Confluence CSV",
                            df_sell_conf.to_csv(index=False),
                            file_name=f"VMS_SELLConf_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                        )
                with dl4:
                    if _REPORTLAB:
                        pdf_buf = generate_scan_pdf(df_buy, df_sell, df_buy_conf, df_sell_conf, df_div, ts_str)
                        if pdf_buf:
                            st.download_button(
                                "⬇️ Full PDF Report",
                                pdf_buf,
                                file_name=f"VMS_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                mime="application/pdf",
                            )
                    else:
                        st.caption("PDF unavailable — pip install reportlab")

                if errors:
                    with st.expander(f"⚠️ {len(errors)} stocks skipped (no data / errors)"):
                        st.dataframe(pd.DataFrame(errors, columns=["Symbol", "Error"]), use_container_width=True)
    
    # ── TAB 2: BACKTEST ───────────────────────────────────────
    with tab_bt:
        st.markdown("### 📈 Historical Backtest (Weekly BUY Signals)")
        
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            lookback_wks = st.slider("Lookback (weeks)", 52, 520, 260, 26)
        with bc2:
            profit_pct = st.slider("Profit target (%)", 4.0, 20.0, cfg["backtest_profit_pct"], 0.5)
        with bc3:
            st.markdown("<br>", unsafe_allow_html=True)
            run_bt = st.button("🚀 Run Backtest", key="run_bt")
        
        if run_bt or "bt_results" in st.session_state:
            if run_bt:
                for k in ["bt_results", "bt_errors", "bt_ts"]:
                    st.session_state.pop(k, None)
                
                symbols = ALL_SYMBOLS
                total = len(symbols)
                prog_bt = st.progress(0, text=f"Backtesting {total} stocks…")
                all_trades, errors_bt, done = [], [], 0
                
                with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
                    futures = {ex.submit(backtest_one, s, lookback_wks, profit_pct, cfg): s for s in symbols}
                    for fut in as_completed(futures):
                        sym, trades, err = fut.result()
                        done += 1
                        if err:
                            errors_bt.append((sym, err))
                        all_trades.extend(trades)
                        prog_bt.progress(done / total, text=f"[{done}/{total}] {sym} — {len(all_trades)} trades")
                
                prog_bt.empty()
                st.session_state["bt_results"] = all_trades
                st.session_state["bt_errors"] = errors_bt
                st.session_state["bt_ts"] = datetime.now().strftime("%d %b %Y %H:%M")
            
            all_trades = st.session_state.get("bt_results", [])
            errors_bt = st.session_state.get("bt_errors", [])
            ts_str_bt = st.session_state.get("bt_ts", "")
            
            if not all_trades:
                st.warning("No qualifying backtest signals found.")
            else:
                df_bt = pd.DataFrame(all_trades)
                df_hit  = df_bt[df_bt["Status"] == "HIT"]
                df_stop = df_bt[df_bt["Status"] == "STOP"]
                df_time = df_bt[df_bt["Status"] == "TIME"]
                df_open = df_bt[df_bt["Status"].str.startswith("OPEN")]
                closed  = df_bt[~df_bt["Status"].str.startswith("OPEN")]
                n_total = len(df_bt)
                win_rate = (closed["Return_%"] > 0).mean() * 100 if len(closed) else 0.0
                avg_ret  = closed["Return_%"].mean() if len(closed) else 0.0

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.markdown(metric_card("Total", str(n_total), "metric-blue"), unsafe_allow_html=True)
                m2.markdown(metric_card("Target Hit", str(len(df_hit)), "metric-green"), unsafe_allow_html=True)
                m3.markdown(metric_card("Stopped", str(len(df_stop)), "metric-red"), unsafe_allow_html=True)
                m4.markdown(metric_card("Time / Open", f"{len(df_time)} / {len(df_open)}", "metric-blue"), unsafe_allow_html=True)
                m5.markdown(metric_card("Win % (closed)", f"{win_rate:.1f}%", "metric-green"), unsafe_allow_html=True)
                m6.markdown(metric_card("Avg Ret (closed)", f"{avg_ret:+.2f}%", "metric-blue"), unsafe_allow_html=True)

                st.markdown("#### 📋 Trade Log")
                st.dataframe(df_bt.reset_index(drop=True), use_container_width=True)
    
    # ── TAB 3: FOCUS STOCK ────────────────────────────────────
    with tab_focus:
        st.markdown("### 🔭 Deep-Dive Analysis")
        
        # IMPROVEMENT 4: Reuse scan results if available
        focus_sym = st.selectbox("Select stock", [""] + ALL_SYMBOLS, key="focus_sym")
        run_focus = st.button("🔍 Analyse", key="run_focus")
        
        if run_focus and focus_sym:
            # Check if we already have this data
            existing_rows = None
            if "scan_results" in st.session_state:
                existing_rows = [r for r in st.session_state["scan_results"] if r["Symbol"] == focus_sym]
            
            if existing_rows:
                st.success(f"✓ Using cached scan data for {focus_sym}")
                df_focus = pd.DataFrame(existing_rows)
            else:
                with st.spinner(f"🔄 Fetching & analyzing {focus_sym}…"):
                    sym_raw, rows, err = scan_stock(focus_sym, cfg)
                
                if err:
                    st.error(f"⚠️ Error: {err}")
                else:
                    df_focus = pd.DataFrame(rows)
            
            if len(df_focus) > 0:
                st.markdown(f"#### {focus_sym} — Signal Summary")
                focus_cols = ["TF", "Signal", "Strength", "Composite", "RiskAdj_Comp", "Vol_Ann%", "Vol_Z", "Beta", "Capitulation", "RSI_Z", "MACD_Z", "RSI", "Close"]
                avail = [c for c in focus_cols if c in df_focus.columns]
                st.dataframe(df_focus[avail].reset_index(drop=True), use_container_width=True)
            else:
                st.warning("No data found.")
    
    # ── TAB 4: ABOUT ──────────────────────────────────────────
    with tab_about:
        st.markdown("""
        ## VMS Scanner v1.9 — Audit Fixes

        ### What's New in v1.9
        ✅ **Backtest re-entry** — positions can re-enter after exit (was: 1 trade per symbol, ever)
        ✅ **CAPE in backtest** — backtest composite now uses the SAME weights as the live scan
        ✅ **Stops & time exits** — stop-loss + max-hold exits; optional vol-scaled target/stop
        ✅ **No EPS look-ahead** — EPS dated +60d (reporting lag) before building the CAPE z
        ✅ **No weekly repaint** — incomplete current week dropped from weekly signals
        ✅ **Vol/Beta layer** — annualised vol, vol z-score (capitulation flag), beta vs NIFTY,
        risk-adjusted composite ranking, high-beta market-regime caution
        ✅ **Direction-aware gates** — Add_Conf now mirrored for SELL side
        ✅ **5y data fetch** — weekly z-scores estimated on ~260 weekly bars instead of ~156

        ## VMS Scanner v1.8 — Improvements

        ### What's New in v1.8
        ✅ **Fixed syntax errors** in `_weekly_signal_frame()` (was: `open*`)
        ✅ **Caching** — yfinance data cached for 1 hour (avoid re-downloads)
        ✅ **Retry logic** — Exponential backoff for flaky API calls
        ✅ **Reuse scan results** — Focus tab reuses existing data
        ✅ **Input validation** — Weight sums now validated
        ✅ **Type hints** — Full type hints throughout
        ✅ **Better errors** — More descriptive error messages
        ✅ **Progress bars** — Live feedback during scans
        ✅ **Data validation** — OHLCV bounds checked
        
        ### How It Works
        
        **Signals:**
        - CAPE Z-Score: Cyclically-adjusted PE (India CPI-adjusted)
        - RSI Z + ΔZ: RSI momentum with acceleration
        - MACD% Z + ΔZ: MACD histogram momentum (contrarian mode)
        
        **Composite Score:**
        Weighted average of CAPE (33%), RSI (33%), MACD (34%)
        
        **Confidence Levels:**
        - **STRONG:** |composite| ≥ 1.75 AND all available components aligned
        - **MODERATE:** |composite| ≥ 1.10 AND ≥2 components aligned
        - **WEAK:** Lower thresholds
        
        **Dual Timeframe:**
        Analyzes both daily and weekly charts for multi-frame confirmation.
        
        ### Known Limitations
        - No transaction costs (slippage, brokerage)
        - Weekly-bar granularity: if stop AND target hit in the same week, stop is assumed first (conservative)
        - Vol z / beta need ≥1y of data; new listings show blanks
        - Fixed position sizing
        - Assumes perfect fills
        
        **Use for:** Signal ranking, relative comparisons, trend identification
        **Not for:** Absolute profit forecasting
        
        ---
        *Made by Shantanu | NSE Swing Trading | Data via yfinance | v1.9 audited*
        """)


if __name__ == "__main__":
    main()
