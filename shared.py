"""
Shared code for both portal.py and pages/1_Stock_Analysis.py
All stock names verified against SAHMK API on 2026-06-06.
"""
import json, os
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

from config.settings import MEMORY_DIR, REPORTS_DIR, DATA_RAW, ANTHROPIC_API_KEY

os.environ.setdefault("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)

# ── Styling ────────────────────────────────────────────────────────────────────
CSS = """
<style>
    .main-header  { font-size:1.8rem; font-weight:700; color:#1a1a2e; margin-bottom:0.2rem; }
    .sub-header   { font-size:0.95rem; color:#666; margin-bottom:1.5rem; }
    .forecast-box { background:#f0f7ff; border-radius:10px; padding:1.2rem;
                    margin:0.5rem 0; border:1px solid #cce0ff; }
    .warning-box  { background:#fff8e1; border-radius:8px; padding:0.8rem;
                    border-left:3px solid #ffc107; }
    .chat-user    { background:#e8f4fd; border-radius:12px; padding:0.8rem 1rem; margin:0.4rem 0; }
    .chat-ai      { background:#f8f9fa; border-radius:12px; padding:0.8rem 1rem;
                    margin:0.4rem 0; border-left:3px solid #0066cc; }
    .signal-good  { color:#28a745; font-weight:600; }
    .signal-bad   { color:#dc3545; font-weight:600; }
</style>
"""

# ── Stock registry — verified against SAHMK company profiles 2026-06-06 ───────
STOCKS = {
    # ── Banking ─────────────────────────────────────────────────────────────
    "1120": {"name": "Al Rajhi Bank",               "sector": "Banking",           "emoji": "🏦"},
    "1180": {"name": "Saudi National Bank (SNB)",   "sector": "Banking",           "emoji": "🏦"},
    "1010": {"name": "Riyad Bank",                  "sector": "Banking",           "emoji": "🏦"},
    "1060": {"name": "Saudi Awwal Bank (SABB)",     "sector": "Banking",           "emoji": "🏦"},
    "1080": {"name": "Arab National Bank",          "sector": "Banking",           "emoji": "🏦"},
    "1050": {"name": "BSF (Saudi Fransi)",          "sector": "Banking",           "emoji": "🏦"},
    "1150": {"name": "Alinma Bank",                 "sector": "Banking",           "emoji": "🏦"},
    "1140": {"name": "Bank Albilad",                "sector": "Banking",           "emoji": "🏦"},
    # ── Energy ──────────────────────────────────────────────────────────────
    "2222": {"name": "Saudi Aramco",                "sector": "Energy",            "emoji": "🛢️"},
    "2380": {"name": "Rabigh Refining & Petrochem", "sector": "Energy",            "emoji": "🛢️"},
    # ── Petrochemicals ──────────────────────────────────────────────────────
    "2010": {"name": "SABIC",                       "sector": "Petrochemicals",    "emoji": "⚗️"},
    "2020": {"name": "SAFCO (Saudi Fertilizers)",   "sector": "Petrochemicals",    "emoji": "⚗️"},
    "2290": {"name": "Yanbu National Petrochem",    "sector": "Petrochemicals",    "emoji": "⚗️"},
    "2350": {"name": "Saudi Kayan Petrochem",       "sector": "Petrochemicals",    "emoji": "⚗️"},
    "2310": {"name": "Sahara International Petrochem", "sector": "Petrochemicals", "emoji": "⚗️"},
    "2210": {"name": "Nama Chemicals",              "sector": "Petrochemicals",    "emoji": "⚗️"},
    "2060": {"name": "National Industrialization",  "sector": "Petrochemicals",    "emoji": "⚗️"},
    "2030": {"name": "SARCO",                       "sector": "Petrochemicals",    "emoji": "⚗️"},
    "2250": {"name": "Saudi Industrial Investment", "sector": "Petrochemicals",    "emoji": "⚗️"},
    # ── Mining & Chemicals ───────────────────────────────────────────────────
    "1211": {"name": "Maaden (Saudi Mining)",       "sector": "Mining",            "emoji": "⛏️"},
    "2230": {"name": "Saudi Chemical",              "sector": "Chemicals",         "emoji": "🧪"},
    # ── Telecom ─────────────────────────────────────────────────────────────
    "7010": {"name": "STC",                         "sector": "Telecom",           "emoji": "📡"},
    "7020": {"name": "Mobily (Etihad Etisalat)",    "sector": "Telecom",           "emoji": "📡"},
    "7030": {"name": "Zain Saudi",                  "sector": "Telecom",           "emoji": "📡"},
    "7040": {"name": "Go (formerly Etihad Atheeb)", "sector": "Telecom",           "emoji": "📡"},
    "7200": {"name": "Al Moammar Info Systems",     "sector": "Technology",        "emoji": "💻"},
    # ── Retail & Consumer ────────────────────────────────────────────────────
    "4190": {"name": "Jarir Marketing",             "sector": "Retail",            "emoji": "🛒"},
    "4003": {"name": "United Electronics (Extra)",  "sector": "Retail",            "emoji": "🛒"},
    "4012": {"name": "Thob Al Aseel",               "sector": "Retail",            "emoji": "🛒"},
    "4161": {"name": "BinDawood",                   "sector": "Retail",            "emoji": "🛒"},
    "4001": {"name": "Al Othaim Markets",           "sector": "Retail",            "emoji": "🛒"},
    "4006": {"name": "Saudi Marketing Co.",         "sector": "Retail",            "emoji": "🛒"},
    # ── Real Estate ──────────────────────────────────────────────────────────
    "4300": {"name": "Dar Al Arkan",                "sector": "Real Estate",       "emoji": "🏢"},
    "4220": {"name": "Emaar The Economic City",     "sector": "Real Estate",       "emoji": "🏢"},
    "4322": {"name": "Retal Urban Dev",             "sector": "Real Estate",       "emoji": "🏢"},
    "4323": {"name": "SUMOU Real Estate",           "sector": "Real Estate",       "emoji": "🏢"},
    "4020": {"name": "Saudi Real Estate Co.",       "sector": "Real Estate",       "emoji": "🏢"},
    "4250": {"name": "Jabal Omar Development",      "sector": "Real Estate",       "emoji": "🏢"},
    "4100": {"name": "Makkah Construction",         "sector": "Real Estate",       "emoji": "🏢"},
    "4090": {"name": "Taiba Investments",           "sector": "Real Estate",       "emoji": "🏢"},
    "4345": {"name": "Alinma Retail REIT",          "sector": "Real Estate",       "emoji": "🏢"},
    # ── Healthcare ───────────────────────────────────────────────────────────
    "4013": {"name": "Dr Sulaiman Al Habib",        "sector": "Healthcare",        "emoji": "🏥"},
    "4002": {"name": "Mouwasat Medical",            "sector": "Healthcare",        "emoji": "🏥"},
    "4004": {"name": "Dallah Healthcare",           "sector": "Healthcare",        "emoji": "🏥"},
    "4007": {"name": "Al Hammadi Hospitals",        "sector": "Healthcare",        "emoji": "🏥"},
    "4005": {"name": "National Medical Care",       "sector": "Healthcare",        "emoji": "🏥"},
    "4163": {"name": "Al Dawaa Medical (Nahdi)",    "sector": "Healthcare",        "emoji": "🏥"},
    "2070": {"name": "Saudi Pharmaceutical Ind.",   "sector": "Healthcare",        "emoji": "🏥"},
    "4015": {"name": "Jamjoom Pharma",              "sector": "Healthcare",        "emoji": "💊"},
    # ── Food & Beverages ─────────────────────────────────────────────────────
    "2280": {"name": "Almarai",                     "sector": "Food & Beverages",  "emoji": "🥛"},
    "2050": {"name": "Savola Group",                "sector": "Food & Beverages",  "emoji": "🥛"},
    "2270": {"name": "SADAFCO (Saudia Dairy)",      "sector": "Food & Beverages",  "emoji": "🥛"},
    "2100": {"name": "Wafrah for Industry",         "sector": "Food & Beverages",  "emoji": "🥛"},
    "6010": {"name": "NADEC",                       "sector": "Food & Beverages",  "emoji": "🥛"},
    "6020": {"name": "Al Gassim Investment",        "sector": "Food & Beverages",  "emoji": "🥛"},
    # ── Cement ───────────────────────────────────────────────────────────────
    "3010": {"name": "Arabian Cement",              "sector": "Cement",            "emoji": "🏗️"},
    "3020": {"name": "Yamama Cement",               "sector": "Cement",            "emoji": "🏗️"},
    "3030": {"name": "Saudi Cement Co.",            "sector": "Cement",            "emoji": "🏗️"},
    "3040": {"name": "Qassim Cement",               "sector": "Cement",            "emoji": "🏗️"},
    "3050": {"name": "Southern Province Cement",    "sector": "Cement",            "emoji": "🏗️"},
    "3060": {"name": "Yanbu Cement",                "sector": "Cement",            "emoji": "🏗️"},
    "3080": {"name": "Eastern Province Cement",     "sector": "Cement",            "emoji": "🏗️"},
    "3090": {"name": "TCC (Tabuk Cement)",          "sector": "Cement",            "emoji": "🏗️"},
    # ── Transport & Logistics ────────────────────────────────────────────────
    "4030": {"name": "BAHRI (National Shipping)",   "sector": "Transport",         "emoji": "🚢"},
    "6004": {"name": "Saudi Airlines Catering",     "sector": "Transport",         "emoji": "✈️"},
    # ── Utilities ────────────────────────────────────────────────────────────
    "5110": {"name": "Saudi Electricity Co.",       "sector": "Utilities",         "emoji": "⚡"},
    "2080": {"name": "National Gas & Industrialization (GASCO)", "sector": "Utilities", "emoji": "⚡"},
    # ── Industrials & Diversified ────────────────────────────────────────────
    "2200": {"name": "Arabian Pipes Co.",           "sector": "Industrials",       "emoji": "🏭"},
    "4140": {"name": "Saudi Industrial Export",     "sector": "Industrials",       "emoji": "🏭"},
    "4110": {"name": "Batic Investments",           "sector": "Industrials",       "emoji": "🏭"},
    "2180": {"name": "Filing & Packing Materials",  "sector": "Industrials",       "emoji": "🏭"},
    "4082": {"name": "MRNA",                        "sector": "Financial Services","emoji": "💼"},
    "4280": {"name": "Kingdom Holding",             "sector": "Diversified",       "emoji": "📦"},
    # ── Consumer Services ────────────────────────────────────────────────────
    "1830": {"name": "Leejam Sports",               "sector": "Consumer Services", "emoji": "🏋️"},
}


@st.cache_data(ttl=300)
def load_all_states() -> dict:
    p = MEMORY_DIR / "all_current_states.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("stocks", {})


@st.cache_data(ttl=300)
def load_json_file(path_str: str) -> dict:
    p = Path(path_str)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build_stock_df() -> pd.DataFrame:
    states = load_all_states()
    rows = []
    for sym, s in states.items():
        if not s:
            continue
        info = STOCKS.get(sym, {})
        if not info:
            continue   # skip symbols not in our verified registry
        fc = s.get("forecast", {})
        rows.append({
            "symbol":      sym,
            "name":        info.get("name", sym),
            "sector":      info.get("sector", "Other"),
            "emoji":       info.get("emoji", "📊"),
            "price":       s.get("price", 0),
            "rsi":         s.get("rsi", 50),
            "composite":   s.get("composite", 0),
            "env_score":   s.get("env_score", 0),
            "tech_score":  s.get("tech_score", 0),
            "fund_score":  s.get("fund_score", 0),
            "forecast_90d":fc.get("base_90d_pct", 0),
            "confidence":  fc.get("confidence", 0),
            "conf_label":  fc.get("confidence_label", "?"),
            "rate_regime": s.get("rate_regime", "stable"),
            "repo_rate":   s.get("repo_rate", 4.25),
            "vix":         s.get("vix", 20),
            "oil":         s.get("oil_price", 80),
            "target_90d":  fc.get("target_90d", 0),
        })
    return pd.DataFrame(rows)


def load_stock_engine(sym: str) -> dict:
    prefix = f"{sym}_" if sym != "1120" else ""
    data = {}
    for key, fname in [
        ("hypotheses", f"{prefix}HYPOTHESIS_REPORT.json"),
        ("backtest",   f"{prefix}PREDICTION_BACKTEST_REPORT.json"),
        ("signals",    f"{prefix}SIGNAL_DISCOVERY_REPORT.json"),
    ]:
        p = REPORTS_DIR / fname
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data[key] = json.load(f)
    for key, fname in [
        ("memory",              f"memory_{sym}.json"),
        ("personality",         f"personality_{sym}.json"),
        ("personality_summary", "personality_summary.json"),
    ]:
        p = MEMORY_DIR / fname
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data[key] = json.load(f)
    mp = MEMORY_DIR / f"mistake_vault_{sym}.json"
    if mp.exists():
        with open(mp, encoding="utf-8") as f:
            raw = json.load(f)
            data["mistakes"] = raw.get("mistakes", raw) if isinstance(raw, dict) else raw
    return data
