"""
PHASE 1 — SIGNAL DISCOVERY ENGINE
===================================
Finds which variables genuinely predict future Al Rajhi price movements.
Tests every signal group: Technical, Macro, Fundamental, Earnings/Events.

For every signal measures:
  - Occurrences (sample size)
  - 30 / 60 / 90-day forward returns
  - Directional accuracy vs random
  - TASI outperformance
  - Volatility of outcomes
  - Reliability score

No look-ahead bias. Only uses data available at each signal date.

Run:  python phases/phase1_signal_discovery.py
"""

import sys
import json
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import DATA_RAW, DATA_PROCESSED, REPORTS_DIR, FORWARD_WINDOWS

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_master_dataset() -> pd.DataFrame:
    """
    Builds a single daily master DataFrame with all signals merged.
    Everything aligned to trading days. No look-ahead.
    """
    print("  Loading price data (Yahoo — 4,031 days)...")
    price = pd.read_csv(DATA_RAW / "alrajhi_price_daily.csv", index_col=0, parse_dates=True)
    price.index = pd.to_datetime(price.index).tz_localize(None)
    price = price[["close", "volume"]].dropna()
    price.columns = ["close", "volume"]

    print("  Loading TASI...")
    tasi = pd.read_csv(DATA_RAW / "tasi_daily.csv", index_col=0, parse_dates=True)
    tasi.index = pd.to_datetime(tasi.index).tz_localize(None)
    tasi.columns = ["tasi"]

    print("  Loading macro data...")
    oil = pd.read_csv(DATA_RAW / "brent_oil.csv", index_col=0, parse_dates=True)
    oil.index = pd.to_datetime(oil.index).tz_localize(None)
    oil.columns = ["oil"]

    vix = pd.read_csv(DATA_RAW / "vix.csv", index_col=0, parse_dates=True)
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix.columns = ["vix"]

    rate = pd.read_csv(DATA_RAW / "saudi_repo_rate.csv", index_col=0, parse_dates=True)
    rate.index = pd.to_datetime(rate.index).tz_localize(None)
    rate.columns = ["repo_rate"]

    print("  Loading peer bank prices...")
    peers = {}
    for sym in ["1180", "1050", "1060", "1080", "1010"]:
        p = DATA_RAW / f"peer_{sym}_price.csv"
        if p.exists():
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[["close"]].rename(columns={"close": f"peer_{sym}"})
            peers[sym] = df

    print("  Merging into master dataset...")
    master = price.copy()
    for df in [tasi, oil, vix, rate] + list(peers.values()):
        master = master.join(df, how="left")
    master = master.ffill()

    # ── Technical indicators ─────────────────────────────────────────────────
    c = master["close"]
    master["ret_1d"]  = c.pct_change(1)
    master["ret_5d"]  = c.pct_change(5)
    master["ret_20d"] = c.pct_change(20)

    # RSI (14)
    delta  = c.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    master["rsi"] = 100 - (100 / (1 + rs))

    # Moving averages
    master["ma20"]  = c.rolling(20).mean()
    master["ma50"]  = c.rolling(50).mean()
    master["ma200"] = c.rolling(200).mean()

    # Price vs MAs
    master["above_ma20"]  = (c > master["ma20"]).astype(int)
    master["above_ma50"]  = (c > master["ma50"]).astype(int)
    master["above_ma200"] = (c > master["ma200"]).astype(int)

    # MA crossovers (golden/death cross)
    master["ma50_above_ma200"] = (master["ma50"] > master["ma200"]).astype(int)

    # Volatility (20-day annualised)
    master["vol_20d"] = master["ret_1d"].rolling(20).std() * np.sqrt(252)

    # Distance from 52-week high/low
    master["high_52w"] = c.rolling(252).max()
    master["low_52w"]  = c.rolling(252).min()
    master["dist_from_high"] = (c - master["high_52w"]) / master["high_52w"]
    master["dist_from_low"]  = (c - master["low_52w"])  / master["low_52w"]

    # Bollinger Bands
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    master["bb_upper"] = bb_mid + 2 * bb_std
    master["bb_lower"] = bb_mid - 2 * bb_std
    master["bb_pct"]   = (c - master["bb_lower"]) / (master["bb_upper"] - master["bb_lower"])

    # Volume ratio
    master["vol_ratio"] = master["volume"] / master["volume"].rolling(20).mean()

    # TASI relative strength
    if "tasi" in master.columns:
        master["rs_vs_tasi_20d"] = master["ret_20d"] - master["tasi"].pct_change(20)

    # Repo rate regime
    r = master["repo_rate"]
    master["rate_change_3m"] = r - r.shift(63)
    master["rate_regime"] = "stable"
    master.loc[master["rate_change_3m"] > 0.25, "rate_regime"] = "rising"
    master.loc[master["rate_change_3m"] < -0.25, "rate_regime"] = "falling"

    # VIX regime
    master["vix_regime"] = "normal"
    master.loc[master["vix"] > 30, "vix_regime"] = "high_fear"
    master.loc[master["vix"] > 20, "vix_regime"] = pd.np.where(
        master["vix"] <= 30, "elevated", master.loc[master["vix"] > 30, "vix_regime"]
    ) if hasattr(pd, 'np') else master["vix_regime"]
    # Simpler approach
    master["vix_high"]     = (master["vix"] > 30).astype(int)
    master["vix_elevated"] = (master["vix"] > 20).astype(int)

    # Oil regime
    master["oil_change_3m"] = master["oil"].pct_change(63)
    master["oil_bull"]      = (master["oil_change_3m"] > 0.10).astype(int)
    master["oil_bear"]      = (master["oil_change_3m"] < -0.10).astype(int)

    # ── Forward returns (no look-ahead — these are the OUTCOMES we predict) ──
    for w in FORWARD_WINDOWS:
        master[f"fwd_{w}d"] = c.shift(-w) / c - 1
        if "tasi" in master.columns:
            master[f"fwd_{w}d_vs_tasi"] = (
                c.shift(-w) / c - master["tasi"].shift(-w) / master["tasi"]
            )

    print(f"  Master dataset: {len(master)} rows, {len(master.columns)} columns")
    master.to_csv(DATA_PROCESSED / "master_dataset.csv")
    return master


def load_fundamentals() -> pd.DataFrame:
    """Load quarterly/annual fundamental data."""
    income = pd.read_csv(DATA_RAW / "alrajhi_income_quarterly.csv", parse_dates=["report_date"])
    income = income.sort_values("report_date").reset_index(drop=True)

    # For pre-2022 data, only annual net income is available
    # Label them correctly
    income["data_type"] = "annual"
    recent_mask = income["fiscal_quarter"].notna() & (income["fiscal_year"] >= 2022)
    income.loc[recent_mask, "data_type"] = "quarterly"

    # Net income in SAR billions for readability
    income["net_income_bn"] = income["net_income"] / 1e9

    # YoY growth (annual net income)
    annual = income[income["data_type"] == "annual"].copy()
    annual["ni_yoy_growth"] = annual["net_income"].pct_change()

    return income, annual


def load_dividends() -> pd.DataFrame:
    divs = pd.read_csv(DATA_RAW / "alrajhi_dividends_sahmk.csv", parse_dates=["announcement_date"])
    divs = divs.sort_values("announcement_date").reset_index(drop=True)
    divs["dividend_bn"] = divs["value"]
    return divs


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL TESTER  (core function used by all signal groups)
# ══════════════════════════════════════════════════════════════════════════════

def test_signal(master: pd.DataFrame, signal_mask: pd.Series,
                signal_name: str, signal_group: str,
                min_gap_days: int = 20) -> dict:
    """
    Given a boolean mask of dates when a signal fired, compute forward return stats.
    Enforces minimum gap between signals to avoid overlap contamination.
    """
    # Only use dates where we have forward returns (not at end of series)
    valid = master.dropna(subset=["fwd_90d"])
    fired = valid[signal_mask.reindex(valid.index, fill_value=False)]

    # Enforce minimum gap to avoid signal clustering
    if min_gap_days > 0 and len(fired) > 1:
        gaps = fired.index.to_series().diff().dt.days
        fired = fired[gaps.isna() | (gaps >= min_gap_days)]

    n = len(fired)

    result = {
        "signal":       signal_name,
        "group":        signal_group,
        "occurrences":  n,
        "sample_size_label": "LOW" if n < 15 else ("MEDIUM" if n < 40 else "HIGH"),
    }

    if n < 5:
        result["reliability"] = "INSUFFICIENT SAMPLE"
        return result

    for w in FORWARD_WINDOWS:
        col = f"fwd_{w}d"
        if col not in fired.columns:
            continue
        fwd = fired[col].dropna()
        if len(fwd) < 5:
            continue
        result[f"avg_return_{w}d"]     = round(fwd.mean() * 100, 2)
        result[f"median_return_{w}d"]  = round(fwd.median() * 100, 2)
        result[f"pct_positive_{w}d"]   = round((fwd > 0).mean() * 100, 1)
        result[f"volatility_{w}d"]     = round(fwd.std() * 100, 2)
        result[f"max_drawdown_{w}d"]   = round(fwd.min() * 100, 2)

        # Outperformance vs TASI
        vs_col = f"fwd_{w}d_vs_tasi"
        if vs_col in fired.columns:
            vs = fired[vs_col].dropna()
            result[f"pct_outperform_tasi_{w}d"] = round((vs > 0).mean() * 100, 1)
            result[f"avg_alpha_{w}d"]            = round(vs.mean() * 100, 2)

        # T-test: is avg return significantly different from zero?
        t_stat, p_val = stats.ttest_1samp(fwd.dropna(), 0)
        result[f"p_value_{w}d"] = round(p_val, 3)
        result[f"significant_{w}d"] = p_val < 0.05

    # Composite reliability score (0-100)
    score = 0
    if n >= 15:   score += 20
    if n >= 40:   score += 10
    if n >= 80:   score += 10
    p90 = result.get("p_value_90d", 1.0)
    if p90 < 0.05: score += 30
    elif p90 < 0.10: score += 15
    avg90 = abs(result.get("avg_return_90d", 0))
    if avg90 > 5:  score += 20
    elif avg90 > 2: score += 10
    pct90 = result.get("pct_positive_90d", 50)
    if pct90 > 65 or pct90 < 35: score += 10
    result["reliability_score"] = score
    result["reliability_label"] = (
        "HIGH" if score >= 60 else ("MEDIUM" if score >= 35 else "LOW")
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL GROUP 1 — TECHNICAL TIMING
# ══════════════════════════════════════════════════════════════════════════════

def discover_technical_signals(master: pd.DataFrame) -> list:
    print("\n  [Technical] RSI zones...")
    results = []

    # RSI zones
    for lo, hi, label in [
        (0, 20, "RSI Extreme Oversold (<20)"),
        (20, 30, "RSI Oversold (20-30)"),
        (30, 40, "RSI Low (30-40)"),
        (40, 50, "RSI Neutral-Low (40-50)"),
        (50, 60, "RSI Neutral-High (50-60)"),
        (60, 70, "RSI High (60-70)"),
        (70, 80, "RSI Overbought (70-80)"),
        (80, 100, "RSI Extreme Overbought (>80)"),
    ]:
        mask = (master["rsi"] >= lo) & (master["rsi"] < hi)
        results.append(test_signal(master, mask, label, "Technical — RSI", min_gap_days=5))

    # MA signals
    print("  [Technical] Moving average signals...")
    results.append(test_signal(master,
        (master["above_ma20"] == 1) & (master["above_ma50"] == 1) & (master["above_ma200"] == 1),
        "Price above MA20 + MA50 + MA200 (all bullish)", "Technical — MA", min_gap_days=10))

    results.append(test_signal(master,
        (master["above_ma20"] == 0) & (master["above_ma50"] == 0) & (master["above_ma200"] == 0),
        "Price below MA20 + MA50 + MA200 (all bearish)", "Technical — MA", min_gap_days=10))

    results.append(test_signal(master,
        master["ma50_above_ma200"] == 1,
        "Golden Cross regime (MA50 > MA200)", "Technical — MA", min_gap_days=20))

    results.append(test_signal(master,
        master["ma50_above_ma200"] == 0,
        "Death Cross regime (MA50 < MA200)", "Technical — MA", min_gap_days=20))

    # Bollinger Band signals
    print("  [Technical] Bollinger Bands...")
    results.append(test_signal(master,
        master["bb_pct"] < 0.05,
        "BB: Near lower band (bottom 5%)", "Technical — BB", min_gap_days=10))

    results.append(test_signal(master,
        master["bb_pct"] > 0.95,
        "BB: Near upper band (top 5%)", "Technical — BB", min_gap_days=10))

    # Volume surge
    print("  [Technical] Volume surges...")
    results.append(test_signal(master,
        master["vol_ratio"] > 2.5,
        "Volume surge (>2.5x 20-day avg)", "Technical — Volume", min_gap_days=5))

    results.append(test_signal(master,
        (master["vol_ratio"] > 2.5) & (master["ret_1d"] > 0.02),
        "High volume + price up >2%", "Technical — Volume", min_gap_days=5))

    results.append(test_signal(master,
        (master["vol_ratio"] > 2.5) & (master["ret_1d"] < -0.02),
        "High volume + price down >2%", "Technical — Volume", min_gap_days=5))

    # Distance from highs/lows
    print("  [Technical] Distance from highs/lows...")
    results.append(test_signal(master,
        master["dist_from_high"] < -0.20,
        "Price >20% below 52-week high", "Technical — Momentum", min_gap_days=10))

    results.append(test_signal(master,
        master["dist_from_high"] < -0.30,
        "Price >30% below 52-week high", "Technical — Momentum", min_gap_days=10))

    results.append(test_signal(master,
        master["dist_from_low"] < 0.10,
        "Price within 10% of 52-week low", "Technical — Momentum", min_gap_days=10))

    # Relative strength vs TASI
    if "rs_vs_tasi_20d" in master.columns:
        print("  [Technical] Relative strength...")
        results.append(test_signal(master,
            master["rs_vs_tasi_20d"] > 0.05,
            "Outperforming TASI by >5% over 20 days", "Technical — Relative Strength", min_gap_days=10))

        results.append(test_signal(master,
            master["rs_vs_tasi_20d"] < -0.05,
            "Underperforming TASI by >5% over 20 days", "Technical — Relative Strength", min_gap_days=10))

    # Recent momentum
    results.append(test_signal(master,
        master["ret_20d"] > 0.10,
        "Strong 20-day momentum (>10%)", "Technical — Momentum", min_gap_days=10))

    results.append(test_signal(master,
        master["ret_20d"] < -0.10,
        "Weak 20-day momentum (<-10%)", "Technical — Momentum", min_gap_days=10))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL GROUP 2 — MACRO / ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

def discover_macro_signals(master: pd.DataFrame) -> list:
    results = []
    print("\n  [Macro] Rate regimes...")

    results.append(test_signal(master,
        master["rate_regime"] == "rising",
        "Saudi Rate Regime: Rising", "Macro — Rates", min_gap_days=20))

    results.append(test_signal(master,
        master["rate_regime"] == "stable",
        "Saudi Rate Regime: Stable", "Macro — Rates", min_gap_days=20))

    results.append(test_signal(master,
        master["rate_regime"] == "falling",
        "Saudi Rate Regime: Falling", "Macro — Rates", min_gap_days=20))

    # Absolute rate level
    results.append(test_signal(master,
        master["repo_rate"] >= 4.0,
        "Saudi Repo Rate >= 4% (high rate environment)", "Macro — Rates", min_gap_days=20))

    results.append(test_signal(master,
        master["repo_rate"] <= 2.0,
        "Saudi Repo Rate <= 2% (low rate environment)", "Macro — Rates", min_gap_days=20))

    print("  [Macro] VIX / global fear...")
    results.append(test_signal(master,
        master["vix_high"] == 1,
        "VIX > 30 (high global fear)", "Macro — Global Risk", min_gap_days=10))

    results.append(test_signal(master,
        master["vix"] < 15,
        "VIX < 15 (calm markets)", "Macro — Global Risk", min_gap_days=10))

    results.append(test_signal(master,
        master["vix_elevated"] == 1,
        "VIX > 20 (elevated risk)", "Macro — Global Risk", min_gap_days=10))

    # VIX spike then calm (mean-reversion opportunity)
    results.append(test_signal(master,
        (master["vix"].shift(10) > 30) & (master["vix"] < 25),
        "VIX spike followed by calm (fear recovery)", "Macro — Global Risk", min_gap_days=20))

    print("  [Macro] Oil price regimes...")
    results.append(test_signal(master,
        master["oil_bull"] == 1,
        "Oil in uptrend (>10% 3-month gain)", "Macro — Oil", min_gap_days=20))

    results.append(test_signal(master,
        master["oil_bear"] == 1,
        "Oil in downtrend (>10% 3-month fall)", "Macro — Oil", min_gap_days=20))

    results.append(test_signal(master,
        master["oil"] > 90,
        "Brent Oil > $90/barrel", "Macro — Oil", min_gap_days=20))

    results.append(test_signal(master,
        master["oil"] < 50,
        "Brent Oil < $50/barrel", "Macro — Oil", min_gap_days=20))

    print("  [Macro] TASI trend...")
    if "tasi" in master.columns:
        master["tasi_20d"] = master["tasi"].pct_change(20)
        master["tasi_60d"] = master["tasi"].pct_change(60)

        results.append(test_signal(master,
            master["tasi_20d"] > 0.05,
            "TASI in uptrend (>5% over 20 days)", "Macro — TASI", min_gap_days=10))

        results.append(test_signal(master,
            master["tasi_20d"] < -0.05,
            "TASI in downtrend (>5% drop over 20 days)", "Macro — TASI", min_gap_days=10))

        results.append(test_signal(master,
            master["tasi_60d"] > 0.10,
            "TASI strong bull (>10% over 60 days)", "Macro — TASI", min_gap_days=20))

    # Combined environment
    results.append(test_signal(master,
        (master["rate_regime"] == "rising") & (master["vix"] < 20),
        "Rising rates + calm VIX (ideal bank environment)", "Macro — Combined", min_gap_days=20))

    results.append(test_signal(master,
        (master["rate_regime"] == "falling") & (master["vix"] > 25),
        "Falling rates + high VIX (stressed bank environment)", "Macro — Combined", min_gap_days=20))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL GROUP 3 — FUNDAMENTAL / EARNINGS
# ══════════════════════════════════════════════════════════════════════════════

def discover_fundamental_signals(master: pd.DataFrame, income: pd.DataFrame,
                                  annual: pd.DataFrame, divs: pd.DataFrame) -> list:
    results = []

    # We map annual fundamental signals to the trading day after announcement
    # (first trading day after fiscal year end) — no look-ahead
    print("\n  [Fundamental] Net income growth signals...")

    for i, row in annual.iterrows():
        if i == 0:
            continue
        prev = annual.iloc[i - 1]
        annual.loc[annual.index[i], "ni_yoy"] = (
            (row["net_income"] - prev["net_income"]) / abs(prev["net_income"])
        ) if prev["net_income"] != 0 else np.nan

    # Map to first trading day after report date
    def signal_dates_from_fundamental(df_fund, condition_col, threshold,
                                       direction="above") -> pd.Series:
        """
        Returns a boolean Series on master index.
        Signal fires on the first trading day AFTER the fundamental report date.
        """
        triggered_dates = []
        for _, row in df_fund.iterrows():
            val = row.get(condition_col)
            if pd.isna(val):
                continue
            if direction == "above" and val > threshold:
                triggered_dates.append(row["report_date"])
            elif direction == "below" and val < threshold:
                triggered_dates.append(row["report_date"])

        mask = pd.Series(False, index=master.index)
        for d in triggered_dates:
            # Find next available trading day
            future = master.index[master.index > d]
            if len(future) > 0:
                mask.iloc[mask.index.get_loc(future[0])] = True
        return mask

    results.append(test_signal(master,
        signal_dates_from_fundamental(annual, "ni_yoy", 0.10, "above"),
        "Annual net income growth > 10% YoY", "Fundamental", min_gap_days=200))

    results.append(test_signal(master,
        signal_dates_from_fundamental(annual, "ni_yoy", 0.20, "above"),
        "Annual net income growth > 20% YoY", "Fundamental", min_gap_days=200))

    results.append(test_signal(master,
        signal_dates_from_fundamental(annual, "ni_yoy", 0, "below"),
        "Annual net income declined YoY", "Fundamental", min_gap_days=200))

    # Dividend signals
    print("  [Fundamental] Dividend signals...")

    def dividend_signal_mask(min_value=None, max_value=None) -> pd.Series:
        mask = pd.Series(False, index=master.index)
        for _, row in divs.iterrows():
            d = row.get("announcement_date")
            if pd.isna(d):
                continue
            val = row.get("value", np.nan)
            if pd.isna(val):
                continue
            passes = True
            if min_value and val < min_value:
                passes = False
            if max_value and val > max_value:
                passes = False
            if passes:
                future = master.index[master.index >= d]
                if len(future) > 0:
                    mask.iloc[mask.index.get_loc(future[0])] = True
        return mask

    results.append(test_signal(master,
        dividend_signal_mask(),
        "Any dividend announcement", "Fundamental — Dividends", min_gap_days=0))

    results.append(test_signal(master,
        dividend_signal_mask(min_value=1.0),
        "Large dividend announcement (>= 1.0 SAR)", "Fundamental — Dividends", min_gap_days=0))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL GROUP 4 — MEAN REVERSION
# ══════════════════════════════════════════════════════════════════════════════

def discover_meanreversion_signals(master: pd.DataFrame) -> list:
    results = []
    print("\n  [Mean Reversion] Sharp drawdown then recovery signals...")

    # Sharp drops followed by signal
    for drop, label in [(0.10, "10%"), (0.15, "15%"), (0.20, "20%"), (0.25, "25%")]:
        mask = master["ret_20d"] < -drop
        results.append(test_signal(master, mask,
            f"Price dropped >{label} in 20 days (mean-reversion entry)", "Mean Reversion", min_gap_days=20))

    # Combined: oversold + macro calm
    results.append(test_signal(master,
        (master["rsi"] < 30) & (master["vix"] < 25),
        "RSI < 30 + VIX < 25 (oversold, calm market)", "Mean Reversion", min_gap_days=10))

    results.append(test_signal(master,
        (master["rsi"] < 30) & (master["rate_regime"] != "falling"),
        "RSI < 30 + non-falling rate regime", "Mean Reversion", min_gap_days=10))

    results.append(test_signal(master,
        (master["dist_from_high"] < -0.25) & (master["rsi"] < 35),
        "Price >25% below high + RSI < 35", "Mean Reversion", min_gap_days=20))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# REGIME ANALYSIS — Performance by environment
# ══════════════════════════════════════════════════════════════════════════════

def analyse_regimes(master: pd.DataFrame) -> dict:
    print("\n  [Regimes] Analysing performance by market environment...")
    regimes = {}

    valid = master.dropna(subset=["fwd_90d", "fwd_30d"])

    for regime_col, regime_vals in [
        ("rate_regime", ["rising", "stable", "falling"]),
        ("vix_high", [0, 1]),
        ("oil_bull", [0, 1]),
        ("ma50_above_ma200", [0, 1]),
    ]:
        if regime_col not in valid.columns:
            continue
        for val in regime_vals:
            sub = valid[valid[regime_col] == val]
            key = f"{regime_col}={val}"
            regimes[key] = {
                "regime_col":     regime_col,
                "regime_value":   str(val),
                "n":              len(sub),
                "avg_30d":        round(sub["fwd_30d"].mean() * 100, 2) if len(sub) > 5 else None,
                "avg_90d":        round(sub["fwd_90d"].mean() * 100, 2) if len(sub) > 5 else None,
                "pct_pos_90d":    round((sub["fwd_90d"] > 0).mean() * 100, 1) if len(sub) > 5 else None,
                "volatility_90d": round(sub["fwd_90d"].std() * 100, 2) if len(sub) > 5 else None,
            }
            if "fwd_90d_vs_tasi" in sub.columns:
                regimes[key]["avg_alpha_90d"] = round(sub["fwd_90d_vs_tasi"].mean() * 100, 2)

    return regimes


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_baselines(master: pd.DataFrame) -> dict:
    valid = master.dropna(subset=["fwd_90d"])
    baselines = {}
    for w in FORWARD_WINDOWS:
        col = f"fwd_{w}d"
        if col not in valid.columns:
            continue
        baselines[f"always_long_{w}d"] = {
            "avg_return":    round(valid[col].mean() * 100, 2),
            "pct_positive":  round((valid[col] > 0).mean() * 100, 1),
            "n":             len(valid),
        }
    return baselines


# ══════════════════════════════════════════════════════════════════════════════
# PRINT SIGNAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_signal_report(all_signals: list, regimes: dict, baselines: dict):
    print("\n" + "═" * 80)
    print("  SIGNAL DISCOVERY REPORT — Al Rajhi Bank (1120.SR)")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═" * 80)

    # Baseline first
    print("\n  BASELINE (buy-and-hold, all days):")
    for k, v in baselines.items():
        print(f"    {k}: avg={v['avg_return']}%  pct_positive={v['pct_positive']}%  n={v['n']}")

    # Signals sorted by reliability then 90d return
    valid_signals = [s for s in all_signals if "avg_return_90d" in s]
    valid_signals.sort(key=lambda x: (-x.get("reliability_score", 0), -abs(x.get("avg_return_90d", 0))))

    # Group by signal group
    groups = sorted(set(s["group"] for s in valid_signals))
    for grp in groups:
        grp_sigs = [s for s in valid_signals if s["group"] == grp]
        if not grp_sigs:
            continue
        print(f"\n{'─' * 70}")
        print(f"  {grp.upper()}")
        print(f"{'─' * 70}")
        print(f"  {'Signal':<48} {'N':>5} {'30d%':>6} {'90d%':>6} {'Win%':>5} {'Rel':>5} {'P90':>6}")
        print(f"  {'-'*48} {'-'*5} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*6}")
        for s in grp_sigs:
            sig_name = s["signal"][:47]
            n        = s.get("occurrences", 0)
            ret30    = s.get("avg_return_30d", "N/A")
            ret90    = s.get("avg_return_90d", "N/A")
            win90    = s.get("pct_positive_90d", "N/A")
            rel      = s.get("reliability_label", "")[:3]
            p90      = s.get("p_value_90d", "N/A")
            sig_flag = " **" if s.get("significant_90d") else ""
            print(f"  {sig_name:<48} {n:>5} {str(ret30):>6} {str(ret90):>6} {str(win90):>5} {rel:>5} {str(p90):>6}{sig_flag}")

    # Regime analysis
    print(f"\n{'─' * 70}")
    print("  REGIME ANALYSIS (unconditional performance by environment)")
    print(f"{'─' * 70}")
    print(f"  {'Regime':<40} {'N':>6} {'30d%':>6} {'90d%':>6} {'Win%':>6} {'Alpha':>6}")
    for k, v in sorted(regimes.items()):
        print(f"  {k:<40} {v['n']:>6} {str(v.get('avg_30d','N/A')):>6} "
              f"{str(v.get('avg_90d','N/A')):>6} {str(v.get('pct_pos_90d','N/A')):>6} "
              f"{str(v.get('avg_alpha_90d','N/A')):>6}")

    # Top 10 signals
    print(f"\n{'═' * 80}")
    print("  TOP 10 SIGNALS BY RELIABILITY")
    print(f"{'═' * 80}")
    top10 = sorted(valid_signals, key=lambda x: -x.get("reliability_score", 0))[:10]
    for i, s in enumerate(top10, 1):
        print(f"\n  {i}. {s['signal']}")
        print(f"     Group: {s['group']} | N={s['occurrences']} | Reliability: {s.get('reliability_label','?')} (score={s.get('reliability_score','?')})")
        print(f"     30d avg: {s.get('avg_return_30d','N/A')}% | 90d avg: {s.get('avg_return_90d','N/A')}% | 90d win rate: {s.get('pct_positive_90d','N/A')}%")
        print(f"     90d p-value: {s.get('p_value_90d','N/A')} | Significant: {s.get('significant_90d',False)}")

    # Weakest signals
    print(f"\n{'─' * 70}")
    print("  WEAK / INSUFFICIENT SIGNALS (flagged — do not use)")
    print(f"{'─' * 70}")
    weak = [s for s in all_signals if s.get("reliability_label") == "LOW"
            or s.get("reliability") == "INSUFFICIENT SAMPLE"
            or s.get("occurrences", 0) < 10]
    for s in weak:
        print(f"  ✗ {s['signal']} (n={s.get('occurrences',0)}, reliability={s.get('reliability_label','?')})")

    print(f"\n  TOTAL SIGNALS TESTED: {len(all_signals)}")
    print(f"  High reliability: {len([s for s in all_signals if s.get('reliability_label') == 'HIGH'])}")
    print(f"  Medium reliability: {len([s for s in all_signals if s.get('reliability_label') == 'MEDIUM'])}")
    print(f"  Low / insufficient: {len([s for s in all_signals if s.get('reliability_label') in ('LOW', None) or s.get('occurrences',0) < 10])}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\nSTOCK MEMORY ENGINE v2 — PHASE 1: SIGNAL DISCOVERY")
    print(f"Stock: Al Rajhi Bank (1120.SR)")
    print(f"Date : {date.today()}")
    print("─" * 60)

    print("\nStep 1: Building master dataset...")
    master = load_master_dataset()

    print("\nStep 2: Loading fundamental data...")
    income, annual = load_fundamentals()
    divs = load_dividends()

    print("\nStep 3: Discovering technical signals...")
    tech_signals = discover_technical_signals(master)

    print("\nStep 4: Discovering macro/environment signals...")
    macro_signals = discover_macro_signals(master)

    print("\nStep 5: Discovering fundamental signals...")
    fund_signals  = discover_fundamental_signals(master, income, annual, divs)

    print("\nStep 6: Discovering mean-reversion signals...")
    mr_signals    = discover_meanreversion_signals(master)

    print("\nStep 7: Regime analysis...")
    regimes       = analyse_regimes(master)

    print("\nStep 8: Computing baselines...")
    baselines     = compute_baselines(master)

    all_signals = tech_signals + macro_signals + fund_signals + mr_signals

    # Save
    output = {
        "generated":  datetime.now().isoformat(),
        "stock":      "1120.SR — Al Rajhi Bank",
        "signals":    all_signals,
        "regimes":    regimes,
        "baselines":  baselines,
    }
    with open(REPORTS_DIR / "SIGNAL_DISCOVERY_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print_signal_report(all_signals, regimes, baselines)
    print(f"  Full report saved → reports/SIGNAL_DISCOVERY_REPORT.json")


if __name__ == "__main__":
    main()
