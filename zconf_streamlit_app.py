#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
 VALUE MOMENTUM SWING TRADING SCANNER — v1.9
 NSE Swing Trading Screener with CAPE / RSI / MACD signals
───────────────────────────────────────────────────────────────
 Copyright © 2026 Dr Shantanu Samanta. All rights reserved.

 Author  : Dr Shantanu Samanta
 Contact : dr.shantanu.samanta@gmail.com
 Version : 1.9
 Licence : Proprietary — not for redistribution without the
           express written permission of the copyright holder.
═══════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════
v1.9 — ENTRY TIMING & TIME-TO-TARGET TUNING
No new data inputs. Everything below is derived from the OHLCV
and EPS series v1.8 already downloaded.
═══════════════════════════════════════════════════════════════

CORRECTNESS FIXES (always on — these were bugs):
  1. Backtest/live parity — the backtest now applies the SAME gate
     stack as the live scan (Add_Conf, ΔZ_Accel, Candle, Hi52,
     regime, ATR-reachability) and includes CAPE in the composite.
     In v1.8 the backtest ignored every gate and dropped CAPE, so
     it validated a strategy you were not actually trading.
  2. Stop-loss, max-hold cutoff and MAE tracking in the backtest.
     v1.8 had no stop, so a trade could sit at -30% and still be
     scored a clean HIT, making hold-time statistics meaningless.
  3. Weekly timeframe gets its own z-score lengths. v1.8 applied
     rsi_zlen/macd_zlen = 100 to weekly bars, needing 156 weeks of
     history against a 3y fetch — weekly z-scores were computed
     over essentially all available data and barely adapted.
     Live scan now fetches 5y.

NEW FILTERS (all sidebar-toggled, ALL DEFAULT OFF so that an
untouched v1.9 reproduces v1.8 signal-for-signal):
  A. regime_enable   — 200-DMA / 40-WMA trend gate. Oversold in an
                       uptrend resolves in weeks; oversold in a
                       downtrend takes quarters.
  B. hi52_band_enable— replaces the one-sided 52W ceiling with a
                       band, so you stop selecting only names that
                       are already ≥15% broken.
  C. cross_enable    — entry requires the composite to be CROSSING
                       UP through the threshold, not merely sitting
                       above it. Adds Bars_Since_Cross + recency gate.
  H. atr_target_enable— rejects candidates where the profit target is
                       an unreasonable multiple of ATR, i.e. where
                       the move cannot plausibly happen quickly.

To run:
    pip install streamlit yfinance pandas numpy plotly reportlab tenacity
    streamlit run vms_scanner_v1_9.py
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

# ══════════════════════════════════════════════════════════════
# COPYRIGHT
# ══════════════════════════════════════════════════════════════
__author__ = "Dr Shantanu Samanta"
__email__ = "dr.shantanu.samanta@gmail.com"
__copyright__ = "Copyright © 2026 Dr Shantanu Samanta. All rights reserved."
__license__ = "Proprietary"
__version__ = "1.9"

COPYRIGHT_LINE = "© 2026 Dr Shantanu Samanta · All rights reserved"

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
    page_title="VMS Scanner v1.8 — NSE Swing Trading",
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

    # ══════════════════════════════════════════════════════════
    # v1.9 ADDITIONS
    # ══════════════════════════════════════════════════════════

    # ── FIX 3: separate weekly z-score lengths ────────────────
    # v1.8 reused the daily lengths on weekly bars. 100 weekly bars
    # is ~2 years, and min_bars worked out to 156 weeks against a
    # 3y fetch, so weekly z-scores spanned nearly all history.
    "weekly_zlen_enable": True,      # correctness fix — on by default
    "w_rsi_zlen": 52,
    "w_macd_zlen": 52,
    "w_cape_zlen": 104,
    "live_fetch_period": "5y",       # was "3y"

    # ── A: trend regime gate ──────────────────────────────────
    "regime_enable": False,
    "regime_ma_len_daily": 200,
    "regime_ma_len_weekly": 40,
    "regime_require_above": True,    # price must be above the MA
    "regime_require_slope": True,    # MA must be rising
    "regime_slope_bars": 20,         # slope measured over N bars
    "regime_hard": True,             # True = hard gate, False = report only

    # ── B: 52-week high BAND (replaces one-sided ceiling) ─────
    # v1.8: close/52W-high <= 0.85, i.e. mandatory >=15% drawdown.
    # That structurally selects damaged names, the slowest movers.
    "hi52_band_enable": False,
    "hi52_pct_min": 0.70,            # floor — exclude the wreckage
    "hi52_pct_max": 0.93,            # ceiling — exclude the extended

    # ── C: cross-based entry ──────────────────────────────────
    # v1.8 read only .iloc[-1], so a stock parked at composite 2.5
    # for 30 bars looked identical to one that crossed yesterday.
    "cross_enable": False,
    "cross_max_bars": 3,             # signal must be <= N bars old
    "cross_hard": True,              # True = gate, False = report only
    "cross_scan_back": 40,           # how far back to search for the cross

    # ── H: ATR-normalised target reachability ─────────────────
    # A flat 8% is trivial at 3% ATR and a multi-month trek at 0.8%.
    "atr_target_enable": False,
    "atr_len": 14,
    "atr_max_mult": 3.0,             # reject if target > N x ATR%
    "atr_target_mode": "gate",       # "gate" | "adaptive"
    "atr_target_mult": 2.5,          # adaptive mode: target = N x ATR%
    "atr_target_floor_pct": 4.0,     # adaptive mode: clamp target range
    "atr_target_cap_pct": 20.0,

    # ── FIX 1 & 2: backtest realism ───────────────────────────
    "bt_apply_gates": True,          # correctness fix — on by default
    "bt_use_cape": True,             # include CAPE, matching live
    "bt_stop_enable": True,
    "bt_stop_pct": 8.0,              # stop distance below entry
    "bt_stop_mode": "pct",           # "pct" | "atr"
    "bt_stop_atr_mult": 2.0,
    "bt_max_hold_wks": 26,           # force-close beyond this
    "bt_hit_window_wks": 8,          # "fast hit" reporting window
    "bt_entry_next_open": True,      # enter next bar's open, not signal close

    # ══════════════════════════════════════════════════════════
    # v2.0 ADDITIONS — items D–K from the review
    # All entry filters default OFF; the two items that are
    # bug-fixes (K's magic number, scan target levels) are on.
    # ══════════════════════════════════════════════════════════

    # ── D: use the divergence that was already computed ───────
    # v1.9 and earlier computed div_rsi/div_macd, built a display
    # string from them, and then never used them for anything.
    "div_use_enable": False,
    "div_mode": "bonus",             # "bonus" (add to composite) | "gate"
    "div_bonus": 0.35,               # per divergent oscillator
    "div_regular_only": True,        # ignore hidden divergence
    "div_gate_require_both": False,  # gate mode: need RSI *and* MACD divergence

    # ── E: volume — loaded since v1.8, never used ─────────────
    "vol_enable": False,
    "vol_len": 20,
    "vol_mult": 1.3,                 # bar volume vs its own baseline
    "vol_baseline": "median",        # "median" | "mean"
    "vol_obv_enable": False,         # additionally require OBV rising
    "vol_obv_len": 20,

    # ── F: ΔZ acceleration, tightened ─────────────────────────
    # Defaults reproduce the old bare "> 0, either indicator" test.
    "dz_accel_min": 0.0,             # magnitude floor on the acceleration
    "dz_accel_consec": 1,            # consecutive rising ΔZ bars required

    # ── G: RSI floor / reclaim (the falling-knife filter) ─────
    # rsi_hard_max caps the top at 50; nothing capped the bottom,
    # so an RSI of 12 sailed through.
    "rsi_floor_enable": False,
    "rsi_hard_min": 25.0,
    "rsi_reclaim_enable": False,     # stronger: must cross back UP
    "rsi_reclaim_level": 30.0,
    "rsi_reclaim_lookback": 10,

    # ── I: distance to structural support ─────────────────────
    "support_enable": False,
    "support_mode": "either",        # "swing" | "donchian" | "either"
    "support_lookback": 60,
    "support_max_dist_pct": 6.0,     # how far ABOVE support is acceptable
    "support_min_dist_pct": -2.0,    # tolerance for trading just below

    # ── J: cross-sectional ranking (scan only) ────────────────
    # min_composite is an absolute z-score, so a broad selloff yields
    # 200 candidates and a melt-up yields none. Ranking fixes the
    # candidate count instead of the threshold.
    "rank_enable": False,
    "rank_mode": "percentile",       # "percentile" | "topn"
    "rank_pct": 10.0,                # keep the best N% per timeframe
    "rank_top_n": 25,
    "rank_within_tf": True,

    # ── K: CAPE treatment ─────────────────────────────────────
    "add_conf_cape_min": 1.73,       # was HARDCODED inside _add_conf()
    "cape_mode": "weight",           # "weight" | "gate" | "both"
    "cape_gate_min_z": -0.5,         # gate mode: reject the expensive tail
    "cape_daily_scale": 1.0,         # scale CAPE's weight on Daily only

    # ── Live-scan target / stop levels ────────────────────────
    # v1.9 emitted Target_% but no price, and read the hardcoded
    # backtest_profit_pct, which no sidebar control ever set.
    "scan_profit_pct": 8.0,
    "scan_stop_pct": 8.0,
    "scan_stop_mode": "pct",         # "pct" | "atr"
    "scan_stop_atr_mult": 2.0,
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


def _get_usd_inr() -> float:
    """Get USD/INR rate with caching."""
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


def _get_eps_series(tk: yf.Ticker) -> pd.Series:
    """Extract EPS series from quarterly data."""
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


def compute_cape_z(tk: yf.Ticker, price_df: pd.DataFrame,
                   c: Optional[Dict] = None) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute CAPE z-score, ratio, and TTM EPS (last values only)."""
    z_s, ratio, ttm = compute_cape_z_series(tk, price_df, c)
    if z_s is None:
        return None, ratio, ttm
    last = z_s.iloc[-1] if len(z_s) else np.nan
    last_z = float(last) if not (pd.isna(last) or np.isinf(last)) else None
    return last_z, ratio, ttm


def compute_cape_z_series(tk: yf.Ticker, price_df: pd.DataFrame,
                          c: Optional[Dict] = None) -> Tuple[Optional[pd.Series], Optional[float], Optional[float]]:
    """
    Compute the FULL CAPE z-score series aligned to price_df.index.

    v1.8 only ever produced the last value. The series is needed so
    that (a) cross detection can evaluate the true composite bar by
    bar, and (b) the backtest can include CAPE — v1.8 dropped it
    entirely with the comment "for simplicity", while CAPE carries
    33% of the live composite. That mismatch meant the backtest
    validated a different strategy from the one being traded.
    """
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)

    if not c["use_cape"]:
        return None, None, None

    try:
        eps_s = _get_eps_series(tk)
        if eps_s.empty:
            return None, None, None
        
        price_idx = price_df.index
        if hasattr(price_idx, "tz") and price_idx.tz is not None:
            price_idx = price_idx.tz_convert(None)
        
        close = pd.Series(price_df["Close"].astype(float).values, index=price_idx)
        eps_s = eps_s.sort_index()
        
        # Calculate TTM (Trailing Twelve Months)
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

        # Realign to the caller's original index (tz may have been stripped)
        z_final = pd.Series(z_final.values, index=price_df.index)

        last_close = float(close.iloc[-1])
        last_ttm = float(ttm_s.iloc[-1]) if not ttm_s.empty else None
        last_ratio = round(last_close / last_ttm, 2) if last_ttm and last_ttm > 0 else None

        return z_final, last_ratio, last_ttm

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


def _hi52_ratio_series(high: pd.Series, close: pd.Series, c: Optional[Dict] = None) -> pd.Series:
    """close / rolling 52-week high, as a ratio in (0, 1]."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)

    roll_max = high.rolling(c["hi52_bars"], min_periods=max(20, c["hi52_bars"] // 4)).max()
    return close / roll_max.replace(0, np.nan)


def _hi52_ok_series(high: pd.Series, close: pd.Series, c: Optional[Dict] = None) -> pd.Series:
    """
    Check if close sits in the acceptable zone relative to the 52W high.

    v1.8 behaviour (hi52_band_enable=False): one-sided ceiling,
        ratio <= hi52_pct (0.85), i.e. a mandatory >=15% drawdown.
        This structurally selects damaged names — precisely the ones
        that take longest to travel a fixed profit target.

    v1.9 BAND (hi52_band_enable=True): hi52_pct_min <= ratio <= hi52_pct_max.
        The floor drops the wreckage; the ceiling still keeps you out
        of names that have already run.
    """
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)

    ratio = _hi52_ratio_series(high, close, c)

    if c.get("hi52_band_enable", False):
        return (ratio >= c["hi52_pct_min"]) & (ratio <= c["hi52_pct_max"])
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


# ══════════════════════════════════════════════════════════════
# v1.9 HELPERS — ATR, TREND REGIME, TIMEFRAME CONFIG, CROSS DETECT
# ══════════════════════════════════════════════════════════════

def _tf_cfg(c: Dict[str, Any], tf: str) -> Dict[str, Any]:
    """
    FIX 3 — return a config specialised for the timeframe.

    v1.8 applied rsi_zlen/macd_zlen = 100 to weekly bars as well as
    daily. On weekly data that is ~2 years of lookback, and
    min_bars = 100 + 26 + 30 = 156 weeks ≈ 3 years — exactly the
    fetch window. The weekly z-scores were therefore computed over
    almost the whole available history and barely adapted to regime.
    """
    if tf != "Weekly" or not c.get("weekly_zlen_enable", True):
        return c

    return {
        **c,
        "rsi_zlen": c["w_rsi_zlen"],
        "macd_zlen": c["w_macd_zlen"],
        "cape_zlen": c["w_cape_zlen"],
        "hi52_bars": max(20, int(c["hi52_bars"] / 5)),      # 252d ≈ 52w
        "div_lookback": max(10, int(c["div_lookback"] / 5)),
        "regime_ma_len": c["regime_ma_len_weekly"],
        "atr_len": c["atr_len"],
    }


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    """Wilder's Average True Range. Pure OHLC — no new data required."""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return _wilder_rma(tr, n)


def _atr_pct_series(high: pd.Series, low: pd.Series, close: pd.Series,
                    c: Optional[Dict] = None) -> pd.Series:
    """ATR expressed as a percentage of price — the natural unit of 'a move'."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    atr = _atr(high, low, close, c["atr_len"])
    return atr / close.replace(0, np.nan) * 100.0


def _atr_reach_ok_series(atr_pct: pd.Series, target_pct: float,
                         c: Optional[Dict] = None) -> pd.Series:
    """
    FILTER H — is the profit target plausibly reachable *quickly*?

    A flat 8% target is a routine two-week move for a stock with 3%
    daily ATR, and a multi-month trek for one at 0.8%. Rather than
    asking "will it get there", this asks "is the target a sane
    multiple of how far this stock actually travels".

    Passes when target_pct <= atr_max_mult * ATR%.
    """
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)

    if not c.get("atr_target_enable", False):
        return pd.Series(True, index=atr_pct.index)

    required_mult = target_pct / atr_pct.replace(0, np.nan)
    return required_mult <= c["atr_max_mult"]


def _adaptive_target_pct(atr_pct_last: Optional[float],
                         base_pct: float,
                         c: Optional[Dict] = None) -> float:
    """
    FILTER H, adaptive mode — size the target to the stock's own
    volatility instead of using one fixed percentage for all 290 names.
    Falls back to the base target when ATR is unavailable.
    """
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)

    if not c.get("atr_target_enable", False):
        return base_pct
    if c.get("atr_target_mode", "gate") != "adaptive":
        return base_pct
    if atr_pct_last is None or not np.isfinite(atr_pct_last) or atr_pct_last <= 0:
        return base_pct

    tgt = atr_pct_last * c["atr_target_mult"]
    return float(np.clip(tgt, c["atr_target_floor_pct"], c["atr_target_cap_pct"]))


def _regime_ok_series(close: pd.Series, c: Optional[Dict] = None,
                      tf: str = "Daily") -> pd.Series:
    """
    FILTER A — trend regime gate.

    The scanner is 100% contrarian: RSI, MACD and CAPE are all
    inverted, so it buys depth. Nothing in v1.8 asked whether the
    decline had actually stopped. Buying oversold inside an uptrend
    typically resolves in weeks; buying oversold inside a downtrend
    can take quarters. This is the single largest lever on
    time-to-target available from the existing price series.

    Requires (configurably) price above its long MA and that MA rising.
    """
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)

    idx = close.index
    if not c.get("regime_enable", False):
        return pd.Series(True, index=idx)

    ma_len = int(c.get("regime_ma_len",
                       c["regime_ma_len_weekly"] if tf == "Weekly"
                       else c["regime_ma_len_daily"]))

    ma = close.rolling(ma_len, min_periods=max(20, ma_len // 4)).mean()

    ok = pd.Series(True, index=idx)
    if c.get("regime_require_above", True):
        ok &= (close >= ma)
    if c.get("regime_require_slope", True):
        ok &= (ma.diff(int(c["regime_slope_bars"])) > 0)

    return ok.fillna(False)


def _composite_series(rsi_z: pd.Series, macd_z: pd.Series,
                      cape_z_s: Optional[pd.Series],
                      c: Optional[Dict] = None) -> pd.Series:
    """
    Bar-by-bar composite, using the same weighting as _composite().
    Needed because v1.8 only ever evaluated .iloc[-1], which made a
    stock parked above the threshold for 30 bars indistinguishable
    from one that crossed it yesterday.
    """
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)

    cape_active = cape_z_s is not None and c["use_cape"]

    if cape_active:
        cz = cape_z_s.reindex(rsi_z.index).ffill()
        tot = c["wt_cape"] + c["wt_rsi"] + c["wt_macd"]
        if tot <= 0:
            return pd.Series(np.nan, index=rsi_z.index)
        raw = (cz * c["wt_cape"] + rsi_z * c["wt_rsi"] + macd_z * c["wt_macd"]) / tot
    else:
        tot = c["wt_rsi"] + c["wt_macd"]
        if tot <= 0:
            return pd.Series(np.nan, index=rsi_z.index)
        raw = (rsi_z * c["wt_rsi"] + macd_z * c["wt_macd"]) / tot

    return _clamp(raw, c["clamp_val"])


def _bars_since_cross(comp_s: pd.Series, threshold: float, direction: int,
                      c: Optional[Dict] = None,
                      end_pos: Optional[int] = None) -> Optional[int]:
    """
    FILTER C — how many bars ago did the composite CROSS the threshold?

    direction = +1 : cross up   (comp[i] >= th and comp[i-1] < th)
    direction = -1 : cross down (comp[i] <= th and comp[i-1] > th)

    Returns None if no cross found inside the scan-back window, which
    means the signal is a stale plateau rather than a fresh turn.
    """
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)

    n = len(comp_s)
    if n < 2:
        return None

    if end_pos is None:
        end_pos = n - 1
    if end_pos < 1:
        return None

    vals = comp_s.values
    start = max(1, end_pos - int(c.get("cross_scan_back", 40)) + 1)

    for i in range(end_pos, start - 1, -1):
        cur, prev = vals[i], vals[i - 1]
        if np.isnan(cur) or np.isnan(prev):
            continue
        if direction > 0 and cur >= threshold and prev < threshold:
            return end_pos - i
        if direction < 0 and cur <= threshold and prev > threshold:
            return end_pos - i

    return None


def _cross_ok(bars_since: Optional[int], c: Optional[Dict] = None) -> bool:
    """Signal must be a recent cross, not a long-standing plateau."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)

    if not c.get("cross_enable", False):
        return True
    if bars_since is None:
        return False
    return bars_since <= int(c["cross_max_bars"])


def compute_signals(df: pd.DataFrame, c: Optional[Dict] = None,
                    tf: str = "Daily") -> Optional[Dict[str, Any]]:
    """Compute RSI, MACD, and momentum signals."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)

    # FIX 3 — weekly gets its own z-lengths
    c = _tf_cfg(c, tf)

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

    # ── v1.9: 52W ratio, ATR%, regime ────────────────────────
    hi52_ratio_s = _hi52_ratio_series(high, close, c)
    atr_pct_s = _atr_pct_series(high, low, close, c)
    regime_s = _regime_ok_series(close, c, tf)

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
        # ── v1.9 ────────────────────────────────────────────
        "hi52_ratio": _f(hi52_ratio_s),
        "atr_pct": _f(atr_pct_s),
        "regime_pass": bool(regime_s.iloc[-1]) if len(regime_s) else True,
        # series kept for cross detection in scan_stock — never written
        # into result rows, so the exported tables are unchanged.
        "_rsi_z_s": rsi_z,
        "_macd_z_s": macd_z,
        "_index": df.index,
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
    
    if abs_c >= c["conf_strong"] and agree >= 3:
        return "STRONG"
    if abs_c >= c["conf_moderate"] and agree >= 2:
        return "MODERATE"
    return "WEAK"


def _add_conf(cape_z: Optional[float], cape_used: bool, rsi_val: Optional[float], agree: int, c: Optional[Dict] = None) -> bool:
    """Check additional confirmation gate."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    
    if rsi_val is None:
        return False
    if float(rsi_val) >= c["rsi_hard_max"]:
        return False
    if agree <= c["add_conf_agree_min"]:
        return False
    if cape_used and cape_z is not None:
        if float(cape_z) <= 1.73:
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
    """Scan single stock for signals (daily + weekly)."""
    if c is None:
        c = DEFAULT_CFG
    
    ticker = to_yf(sym_raw)
    rows = []

    try:
        # FIX 3 — 5y so weekly z-scores have room to be adaptive
        df = fetch_stock_data(ticker, c.get("live_fetch_period", "5y"))
        if df is None:
            return sym_raw, [], "⚠️ No data available"

        if len(df) < 100:
            return sym_raw, [], "⚠️ Insufficient bars (< 100)"

        tk = yf.Ticker(ticker)
        cape_z_s, cape_ratio, ttm_eps = compute_cape_z_series(tk, df, c)
        cape_z = None
        if cape_z_s is not None and len(cape_z_s):
            _lz = cape_z_s.iloc[-1]
            cape_z = float(_lz) if not (pd.isna(_lz) or np.isinf(_lz)) else None

        d_sig = compute_signals(df, c, "Daily")

        weekly_raw = df.resample("W").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
        })
        weekly = weekly_raw[weekly_raw["Close"].notna()].copy()
        w_sig = compute_signals(weekly, c, "Weekly")

        for tf, sig, src_df in [("Daily", d_sig, df), ("Weekly", w_sig, weekly)]:
            if sig is None:
                continue

            comp, cape_used = _composite(sig, cape_z, c)
            if comp is None:
                continue

            vrd = verdict(comp, c)
            conf = confidence(comp, sig, cape_z, cape_used, c)

            _ac_zs = [sig["rsi_z"], sig["macd_z"]]
            if cape_used and cape_z is not None:
                _ac_zs.append(cape_z)
            _ac_agree = sum(1 for z in _ac_zs if z is not None and float(z) > 0)

            ac = _add_conf(cape_z, cape_used, sig["rsi_val"], _ac_agree, c)
            dz_acc_ok = _dz_accel_ok(sig, c)
            hi52_ok = sig.get("hi52_pass", True)

            last = src_df.iloc[-1]
            candle_ok = _candle_ok(float(last["Open"]), float(last["High"]),
                                   float(last["Low"]), float(last["Close"]), c)

            # ══════════════════════════════════════════════════
            # v1.9 GATES
            # ══════════════════════════════════════════════════

            # ── A: trend regime ───────────────────────────────
            regime_ok = bool(sig.get("regime_pass", True))
            regime_gate = regime_ok if (c.get("regime_enable", False)
                                        and c.get("regime_hard", True)) else True

            # ── H: is the target reachable in a sane number of ATRs?
            atr_pct = sig.get("atr_pct")
            target_pct = _adaptive_target_pct(atr_pct, c["backtest_profit_pct"], c)
            atr_ok = True
            atr_mult_needed = None
            if atr_pct and atr_pct > 0:
                atr_mult_needed = round(target_pct / atr_pct, 2)
                if c.get("atr_target_enable", False):
                    atr_ok = atr_mult_needed <= c["atr_max_mult"]
            elif c.get("atr_target_enable", False):
                atr_ok = False

            # ── C: is this a fresh CROSS or a stale plateau? ──
            bars_since = None
            if c.get("cross_enable", False) and vrd not in ("N/A", "NEUTRAL"):
                _tfc = _tf_cfg(c, tf)
                comp_s = _composite_series(sig["_rsi_z_s"], sig["_macd_z_s"],
                                           cape_z_s, _tfc)
                if "BUY" in vrd:
                    thr = _tfc["th_sbuy"] if vrd == "STRONG BUY" else _tfc["th_buy"]
                    bars_since = _bars_since_cross(comp_s, thr, +1, _tfc)
                else:
                    thr = _tfc["th_ssell"] if vrd == "STRONG SELL" else _tfc["th_sell"]
                    bars_since = _bars_since_cross(comp_s, thr, -1, _tfc)

            cross_ok = _cross_ok(bars_since, c)
            cross_gate = cross_ok if (c.get("cross_enable", False)
                                      and c.get("cross_hard", True)) else True

            div_tags = []
            if sig.get("div_rsi", {}).get("tag"):
                div_tags.append(sig["div_rsi"]["tag"])
            if sig.get("div_macd", {}).get("tag"):
                div_tags.append(sig["div_macd"]["tag"])
            div_str = " | ".join(div_tags) if div_tags else ""

            all_gates = (ac and dz_acc_ok and candle_ok and hi52_ok
                         and regime_gate and atr_ok and cross_gate)

            rows.append({
                "Symbol": sym_raw,
                "TF": tf,
                "Signal": vrd,
                "Strength": conf,
                "Add_Conf": "YES" if ac else "NO",
                "ΔZ_Accel": "YES" if dz_acc_ok else "NO",
                "Candle_OK": "YES" if candle_ok else "NO",
                "Hi52_OK": "YES" if hi52_ok else "NO",
                # ── v1.9 gate columns ────────────────────────
                "Regime_OK": "YES" if regime_ok else "NO",
                "ATR_OK": "YES" if atr_ok else "NO",
                "Cross_OK": "YES" if cross_ok else "NO",
                "Bars_Since_Cross": bars_since,
                "ATR_%": atr_pct,
                "ATRs_To_Target": atr_mult_needed,
                "Target_%": round(target_pct, 2),
                "Hi52_Ratio": sig.get("hi52_ratio"),
                # ─────────────────────────────────────────────
                "All_Gates": "YES" if all_gates else "NO",
                "Composite": comp,
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
    """
    Compute signal frame for weekly backtesting.

    v1.9: uses the weekly-specialised config (FIX 3) and additionally
    emits ATR%, regime and the 52W ratio so the backtest can apply the
    same gate stack as the live scan (FIX 1).
    """
    c = _tf_cfg(c, "Weekly")

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

    # ── v1.9 additions, needed for backtest/live parity ───────
    hi52_ratio_s = _hi52_ratio_series(high, close, c)
    atr_pct_s = _atr_pct_series(high, low, close, c)
    atr_abs_s = _atr(high, low, close, c["atr_len"])
    regime_s = _regime_ok_series(close, c, "Weekly")

    return pd.DataFrame({
        "rsi_z": rsi_z,
        "macd_z": macd_z,
        "rsi_dz": rsi_dz,
        "macd_dz": macd_dz,
        "rsi_dz_accel": rsi_dz_accel,
        "macd_dz_accel": macd_dz_accel,
        "hi52_ok": hi52_ok_s,
        "hi52_ratio": hi52_ratio_s,
        "atr_pct": atr_pct_s,
        "atr_abs": atr_abs_s,
        "regime_ok": regime_s,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "rsi_val": rsi_val,
    }, index=df.index)


def backtest_one(sym_raw: str, lookback_weeks: int, profit_pct: float,
                 c: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    """
    Run backtest for a single stock.

    ══════════════════════════════════════════════════════════════
    v1.9 — FIX 1 (parity) and FIX 2 (stop / MAE / max-hold)
    ══════════════════════════════════════════════════════════════

    What was wrong in v1.8:

      • Gate stack ignored. Add_Conf, ΔZ_Accel, Hi52 and Candle were
        never applied as filters — candle_ok_flag was computed and
        merely recorded in the trade log. The live scan filters on
        all of them. The backtest therefore measured a materially
        different (and much looser) strategy than the one producing
        your signals.

      • CAPE dropped, with the comment "for simplicity". CAPE carries
        33% of the live composite, so backtest composites were not
        comparable to live composites at all.

      • No stop-loss and no cap on holding period. A trade could fall
        30%, sit underwater for two years, and still be scored a clean
        "HIT" the moment it eventually touched target. Both the win
        rate and the average hold time were consequently meaningless
        as a guide to time-to-target.

      • No MAE (maximum adverse excursion), so there was no way to see
        how much pain a "winning" trade inflicted on the way.

    All four are now addressed. bt_apply_gates and bt_use_cape default
    to True because they are correctness fixes, not tuning choices.
    """
    ticker = to_yf(sym_raw)
    trades = []

    try:
        df = fetch_stock_data(ticker, "5y")
        if df is None:
            return sym_raw, [], "⚠️ No data available"

        if len(df) < 200:
            return sym_raw, [], "⚠️ Insufficient bars"

        weekly_raw = df.resample("W").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
        })
        weekly = weekly_raw[weekly_raw["Close"].notna()].copy()

        sf = _weekly_signal_frame(weekly, c)
        if sf.empty:
            return sym_raw, [], "⚠️ Insufficient bars for signals"

        # ── FIX 1: CAPE series, matching the live composite ───
        cape_w = None
        if c["use_cape"] and c.get("bt_use_cape", True):
            try:
                tk = yf.Ticker(ticker)
                cape_daily, _, _ = compute_cape_z_series(tk, df, c)
                if cape_daily is not None:
                    cape_w = cape_daily.reindex(
                        cape_daily.index.union(weekly.index)
                    ).ffill().reindex(weekly.index)
            except Exception:
                cape_w = None

        cw = _tf_cfg(c, "Weekly")
        cape_active = cape_w is not None and c["use_cape"] and c.get("bt_use_cape", True)

        total_w = len(weekly)
        scan_start = max(0, total_w - lookback_weeks - 1)
        scan_end = total_w - 1

        for i in range(scan_start, scan_end):
            rz_w = sf["rsi_z"].iloc[i]
            mz_w = sf["macd_z"].iloc[i]

            if any(pd.isna(x) for x in (rz_w, mz_w)):
                continue

            # ── Composite, now including CAPE (FIX 1) ─────────
            cape_i = None
            if cape_active:
                _cv = cape_w.iloc[i]
                if not pd.isna(_cv):
                    cape_i = float(_cv)

            if cape_i is not None:
                tot = c["wt_cape"] + c["wt_rsi"] + c["wt_macd"]
                comp_w = (cape_i * c["wt_cape"] + float(rz_w) * c["wt_rsi"]
                          + float(mz_w) * c["wt_macd"]) / tot
                cape_used_i = True
            else:
                tot = c["wt_rsi"] + c["wt_macd"]
                comp_w = (float(rz_w) * c["wt_rsi"] + float(mz_w) * c["wt_macd"]) / tot
                cape_used_i = False

            if tot <= 0:
                continue
            comp_w = float(np.clip(comp_w, -c["clamp_val"], c["clamp_val"]))

            if comp_w < c["bt_min_composite"]:
                continue

            vrd_w = verdict(comp_w, c)
            if vrd_w not in ("BUY", "STRONG BUY"):
                continue

            sig_w = {"rsi_z": float(rz_w), "macd_z": float(mz_w)}
            conf_w = confidence(comp_w, sig_w, cape_i, cape_used_i, c)

            if conf_w not in ("MODERATE", "STRONG"):
                continue

            bar_open = float(sf["open"].iloc[i])
            bar_high = float(sf["high"].iloc[i])
            bar_low = float(sf["low"].iloc[i])
            bar_close = float(sf["close"].iloc[i])

            candle_ok_flag = _candle_ok(bar_open, bar_high, bar_low, bar_close, c)

            # ══════════════════════════════════════════════════
            # FIX 1 — apply the SAME gates the live scan applies
            # ══════════════════════════════════════════════════
            rsi_i = sf["rsi_val"].iloc[i]
            rsi_i = None if pd.isna(rsi_i) else float(rsi_i)

            _ac_zs = [float(rz_w), float(mz_w)]
            if cape_used_i and cape_i is not None:
                _ac_zs.append(cape_i)
            _ac_agree = sum(1 for z in _ac_zs if z > 0)

            ac_i = _add_conf(cape_i, cape_used_i, rsi_i, _ac_agree, c)

            _accel_sig = {
                "rsi_dz_accel": (None if pd.isna(sf["rsi_dz_accel"].iloc[i])
                                 else float(sf["rsi_dz_accel"].iloc[i])),
                "macd_dz_accel": (None if pd.isna(sf["macd_dz_accel"].iloc[i])
                                  else float(sf["macd_dz_accel"].iloc[i])),
            }
            dz_ok_i = _dz_accel_ok(_accel_sig, c)

            _h52 = sf["hi52_ok"].iloc[i]
            hi52_ok_i = True if pd.isna(_h52) else bool(_h52)

            _reg = sf["regime_ok"].iloc[i]
            regime_ok_i = True if pd.isna(_reg) else bool(_reg)
            regime_gate_i = regime_ok_i if (c.get("regime_enable", False)
                                            and c.get("regime_hard", True)) else True

            # ── H: per-trade, volatility-aware target ─────────
            atr_pct_i = sf["atr_pct"].iloc[i]
            atr_pct_i = None if pd.isna(atr_pct_i) else float(atr_pct_i)
            tgt_pct_i = _adaptive_target_pct(atr_pct_i, profit_pct, c)

            atr_ok_i = True
            atrs_needed = None
            if atr_pct_i and atr_pct_i > 0:
                atrs_needed = round(tgt_pct_i / atr_pct_i, 2)
                if c.get("atr_target_enable", False):
                    atr_ok_i = atrs_needed <= c["atr_max_mult"]
            elif c.get("atr_target_enable", False):
                atr_ok_i = False

            # ── C: fresh cross, not a stale plateau ───────────
            bars_since_i = None
            if c.get("cross_enable", False):
                comp_hist = _composite_series(sf["rsi_z"], sf["macd_z"],
                                              cape_w if cape_active else None, cw)
                thr = cw["th_sbuy"] if vrd_w == "STRONG BUY" else cw["th_buy"]
                bars_since_i = _bars_since_cross(comp_hist, thr, +1, cw, end_pos=i)
            cross_ok_i = _cross_ok(bars_since_i, c)
            cross_gate_i = cross_ok_i if (c.get("cross_enable", False)
                                          and c.get("cross_hard", True)) else True

            if c.get("bt_apply_gates", True):
                if not (ac_i and dz_ok_i and hi52_ok_i and regime_gate_i
                        and atr_ok_i and cross_gate_i):
                    continue
                if c.get("candle_body_hard", False) and not candle_ok_flag:
                    continue

            # One trade per symbol (first qualifying signal in window)
            if trades:
                break

            # ══════════════════════════════════════════════════
            # ENTRY
            # ══════════════════════════════════════════════════
            # v1.8 entered at the signal bar's own close, using a
            # composite computed from that same close. Entering at the
            # next bar's open removes that same-bar assumption.
            if c.get("bt_entry_next_open", True) and (i + 1) <= scan_end:
                entry_pos = i + 1
                entry_price = float(sf["open"].iloc[entry_pos])
            else:
                entry_pos = i
                entry_price = bar_close

            if not np.isfinite(entry_price) or entry_price <= 0:
                continue

            entry_date = weekly.index[entry_pos]
            target_price = round(entry_price * (1 + tgt_pct_i / 100.0), 2)

            # ── FIX 2: stop level ─────────────────────────────
            stop_price = None
            if c.get("bt_stop_enable", True):
                if c.get("bt_stop_mode", "pct") == "atr" and atr_pct_i:
                    stop_dist_pct = atr_pct_i * c["bt_stop_atr_mult"]
                else:
                    stop_dist_pct = c["bt_stop_pct"]
                stop_price = round(entry_price * (1 - stop_dist_pct / 100.0), 2)

            max_hold = int(c.get("bt_max_hold_wks", 26))

            exit_date, exit_price, hold_weeks = None, None, None
            status = None
            mae_pct = 0.0          # worst drawdown while in the trade
            mfe_pct = 0.0          # best excursion while in the trade

            for j in range(entry_pos + 1, total_w):
                hi_j = float(sf["high"].iloc[j])
                lo_j = float(sf["low"].iloc[j])

                mae_pct = min(mae_pct, (lo_j - entry_price) / entry_price * 100.0)
                mfe_pct = max(mfe_pct, (hi_j - entry_price) / entry_price * 100.0)

                # Stop is checked FIRST. Within a weekly bar we cannot
                # know the intra-bar sequence, so assuming the adverse
                # touch happens first is the conservative choice.
                if stop_price is not None and lo_j <= stop_price:
                    exit_date, exit_price = weekly.index[j], stop_price
                    hold_weeks, status = j - entry_pos, "STOP"
                    break

                if hi_j >= target_price:
                    exit_date, exit_price = weekly.index[j], target_price
                    hold_weeks, status = j - entry_pos, "HIT"
                    break

                if (j - entry_pos) >= max_hold:
                    exit_date, exit_price = weekly.index[j], float(sf["close"].iloc[j])
                    hold_weeks, status = j - entry_pos, "TIMEOUT"
                    break

            if status is None:
                last_close = float(sf["close"].iloc[-1])
                exit_price = last_close
                hold_weeks = total_w - 1 - entry_pos
                open_ret = (last_close - entry_price) / entry_price * 100
                status = f"OPEN {open_ret:+.1f}%"

            ret_pct = round((exit_price - entry_price) / entry_price * 100, 2)
            fast_hit = (status == "HIT"
                        and hold_weeks is not None
                        and hold_weeks <= int(c.get("bt_hit_window_wks", 8)))

            trades.append({
                "Symbol": sym_raw,
                "Entry_Date": entry_date.strftime("%Y-%m-%d"),
                "Entry_Price": round(entry_price, 2),
                "Target": target_price,
                "Target_%": round(tgt_pct_i, 2),
                "Stop": stop_price,
                "Exit_Date": (exit_date.strftime("%Y-%m-%d") if exit_date
                              else weekly.index[-1].strftime("%Y-%m-%d")),
                "Exit_Price": round(exit_price, 2),
                "Return_%": ret_pct,
                "Hold_Wks": hold_weeks,
                "Fast_Hit": "YES" if fast_hit else "NO",
                "MAE_%": round(mae_pct, 2),
                "MFE_%": round(mfe_pct, 2),
                "Status": status,
                "W_Signal": vrd_w,
                "W_Strength": conf_w,
                "W_Comp": round(comp_w, 3),
                "CAPE_Z": round(cape_i, 3) if cape_i is not None else None,
                "ATR_%": round(atr_pct_i, 2) if atr_pct_i else None,
                "ATRs_To_Target": atrs_needed,
                "Bars_Since_Cross": bars_since_i,
                "Regime_OK": "YES" if regime_ok_i else "NO",
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

    # ── Page furniture: copyright footer on EVERY page ─────────────────────
    def _page_furniture(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 6.5)
        canvas.setFillColor(rl_colors.HexColor("#64748b"))
        canvas.drawString(
            1 * cm, 0.6 * cm,
            f"{__copyright__}  |  {__author__}  |  {__email__}  |  Proprietary — not for redistribution"
        )
        canvas.drawRightString(
            landscape(A4)[0] - 1 * cm, 0.6 * cm,
            f"VMS Scanner v{__version__}  |  Page {canvas.getPageNumber()}"
        )
        canvas.restoreState()

    # ── Column definitions ─────────────────────────────────────────────────
    sig_cols  = ["Symbol", "TF", "Signal", "Strength", "All_Gates", "Add_Conf",
                 "Regime_OK", "Cross_OK", "ATRs_To_Target",
                 "Composite", "CAPE_Z", "RSI_Z", "MACD_Z", "RSI", "Close", "Divergence"]
    conf_cols = ["Symbol", "Confluence_Str", "Combined_Comp",
                 "D_Signal", "D_Strength", "D_Composite",
                 "W_Signal", "W_Strength", "W_Composite",
                 "All_Gates", "CAPE_Z", "RSI_Z", "MACD_Z", "Close", "Divergence"]
    div_cols  = ["Symbol", "TF", "Signal", "Strength", "Composite", "Close", "Divergence"]

    # ── Build story ────────────────────────────────────────────────────────
    story = [
        Paragraph(f"VALUE MOMENTUM SWING TRADING SCANNER v{__version__}", h1),
        Paragraph(
            f'<font size="9" color="#1e3a8a"><b>{__copyright__}</b></font>',
            normal,
        ),
        Paragraph(
            f'<font size="8" color="#475569">Author: {__author__} &nbsp;|&nbsp; '
            f'{__email__} &nbsp;|&nbsp; Proprietary — not for redistribution '
            f'without written permission.</font>',
            normal,
        ),
        Spacer(1, 0.25*cm),
        Paragraph(f"NSE  |  {ts_str}  |  {len(ALL_SYMBOLS)} stocks  |  Daily + Weekly", normal),
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

    # ── Closing copyright block ────────────────────────────────────────────
    story += [
        Spacer(1, 0.8*cm),
        Paragraph(f'<font size="9" color="#1e3a8a"><b>{__copyright__}</b></font>', normal),
        Paragraph(
            f'<font size="7.5" color="#475569">'
            f'Value Momentum Swing Trading Scanner v{__version__} — authored by {__author__} '
            f'({__email__}). The signal logic, scoring methodology, filter design and stock '
            f'universe contained in this report are proprietary intellectual property and may '
            f'not be copied, redistributed or used commercially without express written '
            f'permission. Data via yfinance. Research and educational use only; nothing in '
            f'this report is investment advice.</font>',
            normal,
        ),
    ]

    doc.build(story, onFirstPage=_page_furniture, onLaterPages=_page_furniture)
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
            # ── v1.9 entry-timing context, daily timeframe ────
            "Regime_OK":       d.get("Regime_OK", ""),
            "Cross_OK":        d.get("Cross_OK", ""),
            "D_BarsSinceX":    d.get("Bars_Since_Cross"),
            "W_BarsSinceX":    w.get("Bars_Since_Cross"),
            "ATR_%":           d.get("ATR_%"),
            "ATRs_To_Target":  d.get("ATRs_To_Target"),
            "Hi52_Ratio":      d.get("Hi52_Ratio"),
            # ─────────────────────────────────────────────────
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
                          "All_Gates", "CAPE_Z", "RSI_Z", "MACD_Z",
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
                          "All_Gates", "CAPE_Z", "RSI_Z", "MACD_Z",
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

        # ══════════════════════════════════════════════════════
        # v1.9 — ENTRY TIMING FILTERS
        # Every one of these defaults to OFF. Leave them all off
        # and v1.9 reproduces v1.8 signal-for-signal, so you can
        # A/B each against a genuine baseline in the backtest.
        # ══════════════════════════════════════════════════════
        with st.expander("⏱️ v1.9 — Entry Timing Filters"):
            st.caption(
                "All default OFF. Turn one on at a time and re-run the "
                "backtest — compare median Hold_Wks and Fast-Hit rate."
            )

            st.markdown("**A · Trend regime**")
            regime_enable = st.checkbox(
                "Trend regime gate", DEFAULT_CFG["regime_enable"],
                help="The scanner is fully contrarian, so it buys depth. "
                     "Oversold inside an uptrend resolves in weeks; oversold "
                     "inside a downtrend can take quarters. Biggest single "
                     "lever on time-to-target."
            )
            regime_ma_d = st.slider("Daily MA length", 50, 300,
                                    DEFAULT_CFG["regime_ma_len_daily"], 10)
            regime_ma_w = st.slider("Weekly MA length", 10, 60,
                                    DEFAULT_CFG["regime_ma_len_weekly"], 2)
            regime_above = st.checkbox("Require price above MA",
                                       DEFAULT_CFG["regime_require_above"])
            regime_slope = st.checkbox("Require MA rising",
                                       DEFAULT_CFG["regime_require_slope"])
            regime_hard = st.checkbox("Hard gate (else report-only)",
                                      DEFAULT_CFG["regime_hard"])

            st.markdown("---")
            st.markdown("**B · 52-week band**")
            hi52_band_enable = st.checkbox(
                "Use 52W BAND instead of ceiling", DEFAULT_CFG["hi52_band_enable"],
                help="v1.8's one-sided 0.85 ceiling mandates a >=15% drawdown, "
                     "which structurally selects damaged names — the slowest "
                     "to reach a fixed target. A band drops the wreckage too."
            )
            hi52_min = st.slider("52W ratio floor", 0.40, 0.95,
                                 DEFAULT_CFG["hi52_pct_min"], 0.01)
            hi52_max = st.slider("52W ratio ceiling", 0.60, 1.00,
                                 DEFAULT_CFG["hi52_pct_max"], 0.01)
            if hi52_min >= hi52_max:
                st.error("❌ 52W floor must be below the ceiling")

            st.markdown("---")
            st.markdown("**C · Fresh cross**")
            cross_enable = st.checkbox(
                "Require composite CROSS (not plateau)", DEFAULT_CFG["cross_enable"],
                help="v1.8 read only the last bar, so a stock parked above the "
                     "threshold for 30 bars looked identical to one that crossed "
                     "yesterday. This requires a recent threshold crossing."
            )
            cross_max_bars = st.slider("Max bars since cross", 1, 10,
                                       DEFAULT_CFG["cross_max_bars"])
            cross_hard = st.checkbox("Hard gate (else report-only)",
                                     DEFAULT_CFG["cross_hard"])

            st.markdown("---")
            st.markdown("**H · ATR-normalised target**")
            atr_target_enable = st.checkbox(
                "ATR reachability filter", DEFAULT_CFG["atr_target_enable"],
                help="A flat 8% is a two-week move at 3% ATR and a multi-month "
                     "trek at 0.8%. Rejects candidates where the target is an "
                     "implausible multiple of how far the stock actually travels."
            )
            atr_len = st.slider("ATR length", 7, 30, DEFAULT_CFG["atr_len"])
            atr_max_mult = st.slider("Max ATRs to target", 1.0, 8.0,
                                     DEFAULT_CFG["atr_max_mult"], 0.25)
            atr_mode = st.radio(
                "Target mode", ["gate", "adaptive"],
                index=0 if DEFAULT_CFG["atr_target_mode"] == "gate" else 1,
                horizontal=True,
                help="gate = keep the fixed % target, just filter out "
                     "unreachable ones. adaptive = size each target to the "
                     "stock's own ATR."
            )
            atr_target_mult = st.slider("Adaptive target (× ATR)", 1.0, 5.0,
                                        DEFAULT_CFG["atr_target_mult"], 0.25)

        with st.expander("🧪 v1.9 — Backtest Realism"):
            st.caption(
                "These are correctness fixes, not tuning knobs — v1.8's "
                "backtest applied none of the live gates, dropped CAPE, and "
                "had no stop, so its hold-time statistics could not be trusted. "
                "Leave them on."
            )
            bt_apply_gates = st.checkbox(
                "Apply live gates in backtest", DEFAULT_CFG["bt_apply_gates"],
                help="v1.8 ignored Add_Conf / ΔZ_Accel / Hi52 / Candle in the "
                     "backtest while the live scan filtered on all of them."
            )
            bt_use_cape = st.checkbox(
                "Include CAPE in backtest composite", DEFAULT_CFG["bt_use_cape"],
                help="v1.8 dropped CAPE 'for simplicity' while it carried 33% "
                     "of the live composite."
            )
            bt_stop_enable = st.checkbox("Stop-loss", DEFAULT_CFG["bt_stop_enable"])
            bt_stop_mode = st.radio(
                "Stop mode", ["pct", "atr"],
                index=0 if DEFAULT_CFG["bt_stop_mode"] == "pct" else 1,
                horizontal=True
            )
            bt_stop_pct = st.slider("Stop distance (%)", 2.0, 25.0,
                                    DEFAULT_CFG["bt_stop_pct"], 0.5)
            bt_stop_atr_mult = st.slider("Stop distance (× ATR)", 0.5, 6.0,
                                         DEFAULT_CFG["bt_stop_atr_mult"], 0.25)
            bt_max_hold = st.slider("Max hold (weeks)", 4, 104,
                                    DEFAULT_CFG["bt_max_hold_wks"], 2)
            bt_hit_window = st.slider("'Fast hit' window (weeks)", 2, 26,
                                      DEFAULT_CFG["bt_hit_window_wks"])
            bt_entry_next_open = st.checkbox(
                "Enter at next bar's open", DEFAULT_CFG["bt_entry_next_open"],
                help="v1.8 entered at the signal bar's own close using a "
                     "composite derived from that same close."
            )

        with st.expander("📐 v1.9 — Weekly Z-Score Lengths"):
            st.caption(
                "v1.8 applied the daily z-lengths (100) to weekly bars too, "
                "requiring 156 weeks of history against a 3y fetch — weekly "
                "z-scores spanned nearly all available data and barely adapted."
            )
            weekly_zlen_enable = st.checkbox("Separate weekly z-lengths",
                                             DEFAULT_CFG["weekly_zlen_enable"])
            w_rsi_zlen = st.slider("Weekly RSI z-length", 26, 156,
                                   DEFAULT_CFG["w_rsi_zlen"], 2)
            w_macd_zlen = st.slider("Weekly MACD z-length", 26, 156,
                                    DEFAULT_CFG["w_macd_zlen"], 2)
            w_cape_zlen = st.slider("Weekly CAPE z-length", 52, 260,
                                    DEFAULT_CFG["w_cape_zlen"], 4)
            fetch_period = st.selectbox(
                "Live scan history", ["3y", "5y", "10y"],
                index=["3y", "5y", "10y"].index(DEFAULT_CFG["live_fetch_period"])
            )

        st.markdown("---")
        st.markdown(f'📦 **Universe:** {len(ALL_SYMBOLS)} NSE stocks')

        _active = sum([regime_enable, hi52_band_enable, cross_enable, atr_target_enable])
        if _active == 0:
            st.caption("🔵 v1.9 filters all OFF — baseline (v1.8-equivalent) signals.")
        else:
            st.caption(f"🟢 {_active} of 4 v1.9 entry-timing filters active.")

        st.markdown("---")
        st.markdown(
            f'<div style="font-size:10.5px;color:#64748b;line-height:1.6">'
            f'<b style="color:#1e3a8a">{__copyright__}</b><br>'
            f'{__author__}<br>{__email__}<br>'
            f'<span style="color:#94a3b8">v{__version__} · Proprietary</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

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
        # ── v1.9: A ───────────────────────────────────────────
        "regime_enable": regime_enable,
        "regime_ma_len_daily": int(regime_ma_d),
        "regime_ma_len_weekly": int(regime_ma_w),
        "regime_require_above": regime_above,
        "regime_require_slope": regime_slope,
        "regime_hard": regime_hard,
        # ── v1.9: B ───────────────────────────────────────────
        "hi52_band_enable": hi52_band_enable,
        "hi52_pct_min": float(hi52_min),
        "hi52_pct_max": float(hi52_max),
        # ── v1.9: C ───────────────────────────────────────────
        "cross_enable": cross_enable,
        "cross_max_bars": int(cross_max_bars),
        "cross_hard": cross_hard,
        # ── v1.9: H ───────────────────────────────────────────
        "atr_target_enable": atr_target_enable,
        "atr_len": int(atr_len),
        "atr_max_mult": float(atr_max_mult),
        "atr_target_mode": atr_mode,
        "atr_target_mult": float(atr_target_mult),
        # ── v1.9: backtest realism ────────────────────────────
        "bt_apply_gates": bt_apply_gates,
        "bt_use_cape": bt_use_cape,
        "bt_stop_enable": bt_stop_enable,
        "bt_stop_mode": bt_stop_mode,
        "bt_stop_pct": float(bt_stop_pct),
        "bt_stop_atr_mult": float(bt_stop_atr_mult),
        "bt_max_hold_wks": int(bt_max_hold),
        "bt_hit_window_wks": int(bt_hit_window),
        "bt_entry_next_open": bt_entry_next_open,
        # ── v1.9: weekly z-lengths ────────────────────────────
        "weekly_zlen_enable": weekly_zlen_enable,
        "w_rsi_zlen": int(w_rsi_zlen),
        "w_macd_zlen": int(w_macd_zlen),
        "w_cape_zlen": int(w_cape_zlen),
        "live_fetch_period": fetch_period,
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
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1e3a8a,#0284c7);padding:36px;border-radius:18px;margin-bottom:28px;color:white">
        <div style="font-size:11px;color:rgba(255,255,255,0.65);letter-spacing:3px;text-transform:uppercase;margin-bottom:10px">📈 NSE Swing Trading</div>
        <div style="font-size:34px;font-weight:800;line-height:1.15;margin-bottom:6px">Value Momentum Scanner</div>
        <div style="font-size:13px;font-weight:700;color:rgba(255,255,255,0.92);margin-bottom:10px">
            © 2026 Dr Shantanu Samanta &nbsp;·&nbsp; All rights reserved
        </div>
        <div style="font-size:14px;color:rgba(255,255,255,0.72)">CAPE · RSI Z · MACD Z · Dual Timeframe Confluence · Entry Timing</div>
        <div style="margin-top:14px;display:flex;gap:8px;flex-wrap:wrap">
            <span style="background:rgba(255,255,255,0.18);color:white;border-radius:20px;padding:4px 14px;font-size:11px;font-weight:700">v{__version__}</span>
            <span style="background:rgba(255,255,255,0.18);color:white;border-radius:20px;padding:4px 14px;font-size:11px;font-weight:700">290 NSE Stocks</span>
            <span style="background:rgba(255,255,255,0.18);color:white;border-radius:20px;padding:4px 14px;font-size:11px;font-weight:700">7 Watchlists Merged</span>
            <span style="background:rgba(255,255,255,0.18);color:white;border-radius:20px;padding:4px 14px;font-size:11px;font-weight:700">Dr Shantanu Samanta</span>
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

                st.markdown(f'<div class="info-box">✓ Scan completed: <b>{ts_str}</b> &nbsp;·&nbsp; Universe: <b>{len(ALL_SYMBOLS)} stocks</b></div>', unsafe_allow_html=True)

                # ── Display columns ────────────────────────────────────
                display_cols = ["Symbol", "TF", "Signal", "Strength", "All_Gates",
                                "Add_Conf", "ΔZ_Accel", "Candle_OK", "Hi52_OK",
                                # ── v1.9 entry-timing columns ──
                                "Regime_OK", "Cross_OK", "Bars_Since_Cross",
                                "ATR_OK", "ATR_%", "ATRs_To_Target", "Target_%",
                                "Hi52_Ratio",
                                # ───────────────────────────────
                                "Composite", "CAPE_Z", "RSI_Z", "MACD_Z",
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
                df_hit = df_bt[df_bt["Status"] == "HIT"]
                df_stop = df_bt[df_bt["Status"] == "STOP"]
                df_timeout = df_bt[df_bt["Status"] == "TIMEOUT"]
                df_open = df_bt[df_bt["Status"].str.startswith("OPEN")]

                n_total = len(df_bt)
                n_hit = len(df_hit)
                win_rate = n_hit / n_total * 100 if n_total > 0 else 0.0
                n_fast = int((df_bt["Fast_Hit"] == "YES").sum()) if "Fast_Hit" in df_bt else 0
                fast_rate = n_fast / n_total * 100 if n_total > 0 else 0.0

                # ══════════════════════════════════════════════
                # Row 1 — outcome mix
                # ══════════════════════════════════════════════
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.markdown(metric_card("Total", str(n_total), "metric-blue"), unsafe_allow_html=True)
                m2.markdown(metric_card("HIT", f"{n_hit} ({win_rate:.1f}%)", "metric-green"), unsafe_allow_html=True)
                m3.markdown(metric_card("STOPPED", str(len(df_stop)), "metric-red"), unsafe_allow_html=True)
                m4.markdown(metric_card("TIMEOUT", str(len(df_timeout)), "metric-yellow"), unsafe_allow_html=True)
                m5.markdown(metric_card("Still Open", str(len(df_open)), "metric-blue"), unsafe_allow_html=True)

                # ══════════════════════════════════════════════
                # Row 2 — THE objective function you are tuning.
                # Median (not mean) hold, because hold times are
                # heavily right-skewed by a few long grinds.
                # ══════════════════════════════════════════════
                hit_hold = df_hit["Hold_Wks"].dropna() if n_hit else pd.Series(dtype=float)
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.markdown(metric_card(
                    "Median Hold (HIT)",
                    f"{hit_hold.median():.0f}w" if len(hit_hold) else "N/A",
                    "metric-green"), unsafe_allow_html=True)
                k2.markdown(metric_card(
                    "Mean Hold (HIT)",
                    f"{hit_hold.mean():.1f}w" if len(hit_hold) else "N/A",
                    "metric-blue"), unsafe_allow_html=True)
                k3.markdown(metric_card(
                    f"Fast Hit (≤{cfg['bt_hit_window_wks']}w)",
                    f"{n_fast} ({fast_rate:.1f}%)", "metric-green"), unsafe_allow_html=True)
                k4.markdown(metric_card(
                    "Median MAE",
                    f"{df_bt['MAE_%'].median():.1f}%" if "MAE_%" in df_bt else "N/A",
                    "metric-red"), unsafe_allow_html=True)
                k5.markdown(metric_card(
                    "Avg Return",
                    f"{df_bt['Return_%'].mean():+.2f}%", "metric-blue"), unsafe_allow_html=True)

                st.markdown(
                    '<div class="info-box">'
                    '<b>Read Median Hold and Fast-Hit together.</b> A filter that '
                    'shortens median hold while shrinking Fast-Hit count has simply '
                    'thrown away trades rather than found faster ones. Median MAE '
                    'tells you how much pain the survivors absorbed — v1.8 could not '
                    'report this at all, because it had no stop and no excursion tracking.'
                    '</div>', unsafe_allow_html=True)

                # ── Hold-time distribution ────────────────────
                if _PLOTLY and len(hit_hold) > 1:
                    fig_hold = px.histogram(
                        hit_hold, nbins=min(30, int(hit_hold.max()) + 1),
                        labels={"value": "Weeks to target"},
                        title="Time-to-target distribution (HIT trades only)",
                    )
                    fig_hold.update_layout(showlegend=False, height=320,
                                           margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig_hold, use_container_width=True)

                st.markdown("#### 📋 Trade Log")
                st.dataframe(df_bt.reset_index(drop=True), use_container_width=True)

                st.download_button(
                    "⬇️ Backtest Trades CSV",
                    df_bt.to_csv(index=False),
                    file_name=f"VMS_Backtest_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )
    
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
                focus_cols = ["TF", "Signal", "Strength", "Composite", "RSI_Z", "MACD_Z", "RSI", "Close"]
                avail = [c for c in focus_cols if c in df_focus.columns]
                st.dataframe(df_focus[avail].reset_index(drop=True), use_container_width=True)
            else:
                st.warning("No data found.")
    
    # ── TAB 4: ABOUT ──────────────────────────────────────────
    with tab_about:
        st.markdown("""
        ## VMS Scanner v1.9 — Entry Timing & Time-to-Target

        No new data sources. Everything below is derived from the OHLCV
        and EPS series the scanner already downloads.

        ---

        ### 🔧 Correctness fixes (on by default — these were bugs)

        **1 · Backtest/live parity.**
        v1.8's backtest filtered on composite, verdict and confidence only.
        `Add_Conf`, `ΔZ_Accel`, `Hi52_OK` and `Candle_OK` were never applied —
        `candle_ok_flag` was computed and merely written into the trade log.
        The live scan filters on all of them. It also dropped CAPE entirely
        ("for simplicity") while CAPE carries 33% of the live composite.
        **The backtest was therefore validating a different, much looser
        strategy than the one generating your signals.** Both are now fixed.

        **2 · Stop-loss, max-hold and MAE.**
        v1.8 had no stop. A trade could fall 30%, sit underwater for two
        years, and still be scored a clean `HIT` the moment it eventually
        touched target. Win rate and average hold were unusable as a guide
        to time-to-target. v1.9 adds a stop (% or ATR-based), a max-hold
        cutoff producing `TIMEOUT`, and per-trade MAE/MFE.

        **3 · Weekly z-score lengths.**
        v1.8 applied `rsi_zlen`/`macd_zlen` = 100 to weekly bars. That is
        ~2 years of lookback, and `min_bars` worked out to 156 weeks against
        a 3-year fetch — weekly z-scores spanned nearly all available history
        and barely adapted to regime. Weekly now has its own lengths (52 by
        default) and the live scan fetches 5 years.

        ---

        ### ⏱️ Entry-timing filters (all default **OFF**)

        Leave every one of them off and v1.9 reproduces v1.8 signal-for-signal.
        Turn on **one at a time** and re-run the backtest, so each change is
        measured against a real baseline.

        **The underlying problem:** RSI, MACD and CAPE are *all* inverted
        (`rsi_contrarian`, `macd_contrarian`, `cape_bearish`). The system buys
        *depth*, and nothing in it asks whether the decline has actually
        stopped. That is the structural reason time-to-target is long.

        **A · Trend regime gate** — price above its 200-DMA (40-WMA on weekly)
        with the MA rising. Oversold inside an uptrend resolves in weeks;
        oversold inside a downtrend can take quarters. Expect the largest
        single improvement here.

        **B · 52-week band** — v1.8's one-sided ceiling (`ratio ≤ 0.85`)
        *mandates* a ≥15% drawdown, structurally selecting damaged names,
        which are the slowest to travel a fixed target. The band adds a floor
        so you drop the wreckage as well as the extended.

        **C · Fresh cross** — v1.8 read only `.iloc[-1]`, so a stock parked at
        composite 2.5 for thirty bars was indistinguishable from one that
        crossed 2.5 yesterday. Now requires a recent threshold *crossing*,
        and reports `Bars_Since_Cross`.

        **H · ATR-normalised target** — a flat 8% is a routine two-week move
        at 3% ATR and a multi-month trek at 0.8%. *Gate* mode keeps your fixed
        target and filters out candidates needing an implausible number of
        ATRs; *adaptive* mode sizes each target to the stock's own volatility.

        ---

        ### 📏 How to read the backtest

        `Median Hold (HIT)` is the number you are trying to reduce — median,
        not mean, because hold times are heavily right-skewed by a few long
        grinds. But read it **together with** `Fast Hit` count: a filter that
        shortens median hold while shrinking the Fast-Hit count has simply
        discarded trades rather than found faster ones. `Median MAE` shows how
        much drawdown the survivors absorbed.

        ---

        ### Still open (suggested but not implemented)

        These were in the original review and remain available if you want them:

        - **D** — divergence is computed but discarded; `div_rsi`/`div_macd`
          still only produce a display tag and never enter `All_Gates`
        - **E** — `volume` is loaded then thrown away in both signal functions
        - **F** — the ΔZ gate still uses OR with a bare `> 0` threshold
        - **G** — `rsi_hard_max` caps the top but there is no RSI floor, so
          RSI 12 passes cleanly
        - **I** — distance to prior swing low / Donchian support
        - **J** — cross-sectional ranking instead of a fixed `min_composite`
        - **K** — CAPE's 33% weight on the daily timeframe, and the hardcoded
          magic `1.73` cutoff inside `_add_conf()` that is neither in
          `DEFAULT_CFG` nor exposed in the sidebar

        ---

        ### Inherited from v1.8
        ✅ Fixed syntax errors in `_weekly_signal_frame()` (was: `open*`)
        ✅ yfinance caching (1-hour TTL) · retry with exponential backoff
        ✅ Focus tab reuses scan results · weight-sum validation · type hints

        ### How It Works
        
        **Signals:**
        - CAPE Z-Score: Cyclically-adjusted PE (India CPI-adjusted)
        - RSI Z + ΔZ: RSI momentum with acceleration
        - MACD% Z + ΔZ: MACD histogram momentum (contrarian mode)
        
        **Composite Score:**
        Weighted average of CAPE (33%), RSI (33%), MACD (34%)
        
        **Confidence Levels:**
        - **STRONG:** ≥2.0 composite AND ≥3 signals aligned
        - **MODERATE:** ≥1.0 composite AND ≥2 signals aligned
        - **WEAK:** Lower thresholds
        
        **Dual Timeframe:**
        Analyzes both daily and weekly charts for multi-frame confirmation.
        
        ### Known Limitations
        - No transaction costs (slippage, brokerage, STT)
        - Fixed position sizing
        - Assumes fills at the modelled price
        - Weekly bars hide intra-bar sequence, so when both stop and target
          fall inside the same week the backtest assumes the **stop** hit
          first — conservative, but it will understate some winners
        - Survivorship bias: the universe is today's watchlist, applied
          backwards through history
        - CAPE uses yfinance EPS history, which is sparse and occasionally
          revised for Indian names

        **Use for:** Signal ranking, relative comparisons, filter A/B testing
        **Not for:** Absolute profit forecasting

        ---

        ### © Copyright

        **Copyright © 2026 Dr Shantanu Samanta. All rights reserved.**

        Author: Dr Shantanu Samanta · dr.shantanu.samanta@gmail.com
        Version 1.9 · Licence: Proprietary

        This application, its signal logic, scoring methodology, filter design
        and stock universe are the intellectual property of Dr Shantanu Samanta
        and may not be copied, redistributed, published or used commercially
        without express written permission.

        *Data via yfinance. For research and educational use. Nothing here is
        investment advice.*
        """)

    # ══════════════════════════════════════════════════════════
    # GLOBAL FOOTER — copyright
    # ══════════════════════════════════════════════════════════
    st.markdown(
        f"""
        <hr style="border:none;border-top:1px solid #e2e8f0;margin:36px 0 14px 0">
        <div style="text-align:center;padding:18px 12px 28px 12px;color:#64748b;
                    font-size:12px;line-height:1.8">
            <div style="font-size:13px;font-weight:800;color:#1e3a8a">
                {__copyright__}
            </div>
            <div>
                Value Momentum Swing Trading Scanner v{__version__} &nbsp;·&nbsp;
                Author: <b>{__author__}</b> &nbsp;·&nbsp; {__email__}
            </div>
            <div style="margin-top:6px;color:#94a3b8">
                Proprietary — not for redistribution without written permission.
                Data via yfinance. Research and educational use only; not investment advice.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
