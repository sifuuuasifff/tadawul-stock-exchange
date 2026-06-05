"""
Stock Memory Engine v2 — Central Configuration
All provider keys and settings live here. Change provider = change this file only.
"""

import os
from pathlib import Path

# ─── Project Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW       = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MEMORY_DIR     = BASE_DIR / "memory"
REPORTS_DIR    = BASE_DIR / "reports"
CONFIG_DIR     = BASE_DIR / "config"

# ─── Primary Stock ────────────────────────────────────────────────────────────
PRIMARY_STOCK = {
    "symbol":      "1120",          # SAHMK symbol
    "yahoo":       "1120.SR",       # Yahoo Finance ticker
    "name":        "Al Rajhi Bank",
    "sector":      "Banking",
    "market":      "TASI",
    "currency":    "SAR",
}

# ─── API Keys ─────────────────────────────────────────────────────────────────
# Supports three sources (priority order):
#   1. Streamlit Cloud secrets (st.secrets) — used when deployed
#   2. Environment variable — set manually or via .env
#   3. Hardcoded fallback — for local development

def _get_secret(key: str, fallback: str) -> str:
    """Load secret from Streamlit Cloud, then env, then fallback."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key) or fallback

SAHMK_API_KEY     = _get_secret("SAHMK_API_KEY",     "")
ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY", "")
SAHMK_BASE_URL = "https://app.sahmk.sa/api/v1"

# Plan capabilities — update this when you know your plan
# Options: "free" | "starter" | "pro" | "business"
SAHMK_PLAN = "pro"

SAHMK_PLAN_LIMITS = {
    "free":     {"quarterly": False, "history_years": 0,  "technicals": False},
    "starter":  {"quarterly": False, "history_years": 3,  "technicals": False},
    "pro":      {"quarterly": True,  "history_years": 99, "technicals": True},
    "business": {"quarterly": True,  "history_years": 99, "technicals": True},
}

# ─── Free / Fallback Data Sources ────────────────────────────────────────────
YAHOO_FINANCE_ENABLED = True
FRED_API_KEY          = os.environ.get("FRED_API_KEY", "")   # free at fred.stlouisfed.org

# ─── History Windows ─────────────────────────────────────────────────────────
PRICE_HISTORY_START      = "2010-01-01"
FINANCIAL_HISTORY_START  = "2010-01-01"
MACRO_HISTORY_START      = "2005-01-01"

# ─── Peer Banks (for sector memory) ──────────────────────────────────────────
PEER_BANKS = {
    "1180": "Saudi National Bank (SNB)",
    "1050": "Banque Saudi Fransi",
    "1060": "Saudi British Bank (SABB)",
    "1080": "Arab National Bank",
    "1010": "Riyad Bank",
}

# ─── Forward Return Windows (days) ───────────────────────────────────────────
FORWARD_WINDOWS = [30, 60, 90]

# ─── Walk-Forward Settings ───────────────────────────────────────────────────
WALKFORWARD_START     = "2016-01-01"
WALKFORWARD_STEP_DAYS = 90           # predict every quarter

# ─── Confidence Thresholds ───────────────────────────────────────────────────
MIN_SAMPLE_SIZE_LOW    = 10
MIN_SAMPLE_SIZE_MEDIUM = 30
MIN_SAMPLE_SIZE_HIGH   = 60
