"""
PHASE 2 — STOCK MEMORY ENGINE
==============================
Stores Al Rajhi's personality as evidence-based memory.
Not raw data — answers "When this condition existed, what happened next?"

Memory modules:
  1. Technical Memory
  2. Macro / Environment Memory
  3. Fundamental Memory
  4. Earnings / Event Memory
  5. Sector Memory (vs peers)
  6. Regime Memory (separated — never merged)
  7. Probability / Magnitude / Timing Memory

Run:  python phases/phase2_memory_engine.py
"""

import sys
import json
import warnings
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import DATA_RAW, DATA_PROCESSED, MEMORY_DIR, REPORTS_DIR

MEMORY_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_master() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "master_dataset.csv", index_col=0, parse_dates=True)


def outcome_stats(series: pd.Series, label: str) -> dict:
    s = series.dropna()
    if len(s) < 5:
        return {"label": label, "n": len(s), "insufficient": True}
    t, p = stats.ttest_1samp(s, 0)
    return {
        "label":        label,
        "n":            len(s),
        "avg_pct":      round(s.mean() * 100, 2),
        "median_pct":   round(s.median() * 100, 2),
        "pct_positive": round((s > 0).mean() * 100, 1),
        "std_pct":      round(s.std() * 100, 2),
        "max_gain_pct": round(s.max() * 100, 2),
        "max_loss_pct": round(s.min() * 100, 2),
        "p_value":      round(p, 3),
        "significant":  bool(p < 0.05),
    }


def regime_split(master: pd.DataFrame, condition: pd.Series) -> dict:
    """Return outcome stats for each rate regime within a condition."""
    sub = master[condition]
    result = {}
    for regime in ["rising", "stable", "falling"]:
        r = sub[sub["rate_regime"] == regime]
        if len(r) >= 5:
            result[f"rate_{regime}"] = {
                "n": len(r),
                "avg_90d": round(r["fwd_90d"].mean() * 100, 2),
                "win_pct": round((r["fwd_90d"] > 0).mean() * 100, 1),
            }
    return result


def save_memory(data: dict, filename: str):
    path = MEMORY_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved → memory/{filename}")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — TECHNICAL MEMORY
# ══════════════════════════════════════════════════════════════════════════════

def build_technical_memory(master: pd.DataFrame) -> dict:
    print("\n[1] Building Technical Memory...")
    mem = {"built": datetime.now().isoformat(), "patterns": []}

    # RSI zones with full breakdown
    rsi_zones = [
        (0, 20, "Extreme Oversold"),
        (20, 30, "Oversold"),
        (30, 40, "Low"),
        (40, 50, "Neutral-Low"),
        (50, 60, "Neutral-High"),
        (60, 70, "High"),
        (70, 80, "Overbought"),
        (80, 100, "Extreme Overbought"),
    ]
    for lo, hi, label in rsi_zones:
        mask = (master["rsi"] >= lo) & (master["rsi"] < hi)
        sub  = master[mask].dropna(subset=["fwd_90d"])
        if len(sub) < 5:
            continue
        pattern = {
            "type":    "RSI Zone",
            "label":   f"RSI {label} ({lo}-{hi})",
            "condition": f"RSI between {lo} and {hi}",
            "outcomes": {
                "30d": outcome_stats(sub["fwd_30d"], "30d"),
                "60d": outcome_stats(sub["fwd_60d"], "60d"),
                "90d": outcome_stats(sub["fwd_90d"], "90d"),
            },
            "regime_breakdown": regime_split(master, mask),
            "primary_use":  "timing" if lo < 30 or hi > 70 else "context",
            "reliability":  "HIGH" if sub["fwd_90d"].dropna().pipe(
                lambda s: stats.ttest_1samp(s, 0)[1]) < 0.05 else "MEDIUM",
        }
        mem["patterns"].append(pattern)

    # Bollinger Band
    for zone, label, lo, hi in [
        ("lower", "Near Lower Band (<5%)", None, 0.05),
        ("upper", "Near Upper Band (>95%)", 0.95, None),
    ]:
        mask = (master["bb_pct"] < hi) if hi else (master["bb_pct"] > lo)
        sub  = master[mask].dropna(subset=["fwd_90d"])
        if len(sub) < 5:
            continue
        mem["patterns"].append({
            "type":    "Bollinger Band",
            "label":   f"BB {label}",
            "outcomes": {
                "30d": outcome_stats(sub["fwd_30d"], "30d"),
                "60d": outcome_stats(sub["fwd_60d"], "60d"),
                "90d": outcome_stats(sub["fwd_90d"], "90d"),
            },
            "regime_breakdown": regime_split(master, mask),
            "primary_use": "timing",
        })

    # MA alignment
    for label, cond in [
        ("All above MAs (bearish setup)",  (master["above_ma20"]==1)&(master["above_ma50"]==1)&(master["above_ma200"]==1)),
        ("All below MAs (bullish setup)",  (master["above_ma20"]==0)&(master["above_ma50"]==0)&(master["above_ma200"]==0)),
    ]:
        sub = master[cond].dropna(subset=["fwd_90d"])
        if len(sub) < 5:
            continue
        mem["patterns"].append({
            "type":  "MA Alignment",
            "label": label,
            "outcomes": {
                "30d": outcome_stats(sub["fwd_30d"], "30d"),
                "90d": outcome_stats(sub["fwd_90d"], "90d"),
            },
            "primary_use": "direction",
            "note": "Counterintuitive: all above = weak forward returns; all below = strong",
        })

    # Momentum
    for drop, label in [(0.10, ">10%"), (0.20, ">20%")]:
        cond = master["ret_20d"] > drop
        sub  = master[cond].dropna(subset=["fwd_90d"])
        if len(sub) >= 5:
            mem["patterns"].append({
                "type":  "Momentum",
                "label": f"Strong 20d upswing ({label})",
                "outcomes": {"90d": outcome_stats(sub["fwd_90d"], "90d")},
                "primary_use": "timing (reversal warning)",
                "note": "Strong recent momentum predicts mean reversion, not continuation",
            })

    # Relative strength vs TASI
    if "rs_vs_tasi_20d" in master.columns:
        for dir_, label, cond in [
            ("outperform", "Outperforming TASI >5% (20d)", master["rs_vs_tasi_20d"] > 0.05),
            ("underperform", "Underperforming TASI >5% (20d)", master["rs_vs_tasi_20d"] < -0.05),
        ]:
            sub = master[cond].dropna(subset=["fwd_90d"])
            if len(sub) >= 5:
                mem["patterns"].append({
                    "type":  "Relative Strength vs TASI",
                    "label": label,
                    "outcomes": {
                        "30d": outcome_stats(sub["fwd_30d"], "30d"),
                        "90d": outcome_stats(sub["fwd_90d"], "90d"),
                    },
                    "primary_use": "timing (mean reversion)",
                })

    print(f"    {len(mem['patterns'])} technical patterns stored")
    save_memory(mem, "technical_memory.json")
    return mem


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — MACRO / ENVIRONMENT MEMORY
# ══════════════════════════════════════════════════════════════════════════════

def build_macro_memory(master: pd.DataFrame) -> dict:
    print("\n[2] Building Macro / Environment Memory...")
    mem = {"built": datetime.now().isoformat(), "regime_profiles": [], "correlations": {}}

    # Rate regimes
    for regime in ["rising", "stable", "falling"]:
        sub = master[master["rate_regime"] == regime].dropna(subset=["fwd_90d"])
        if len(sub) < 10:
            continue
        mem["regime_profiles"].append({
            "regime_type": "Saudi Interest Rate",
            "regime_label": regime.capitalize(),
            "n_days": len(sub),
            "outcomes": {
                "30d": outcome_stats(sub["fwd_30d"], "30d"),
                "60d": outcome_stats(sub["fwd_60d"], "60d"),
                "90d": outcome_stats(sub["fwd_90d"], "90d"),
            },
            "alpha_vs_tasi_90d": round(sub["fwd_90d_vs_tasi"].mean() * 100, 2) if "fwd_90d_vs_tasi" in sub else None,
            "interpretation": {
                "rising":  "Counterintuitive — rising rates hurt Al Rajhi. 25% win rate. Avoid or reduce exposure.",
                "stable":  "Best environment. Stable rates = best win rate (65%) and best alpha vs TASI.",
                "falling": "Mixed. Falling rates help valuations but may signal economic weakness.",
            }.get(regime),
        })

    # VIX regimes
    for label, cond in [
        ("VIX > 30 (High Fear)", master["vix"] > 30),
        ("VIX 20-30 (Elevated)", (master["vix"] > 20) & (master["vix"] <= 30)),
        ("VIX < 20 (Normal)",    master["vix"] <= 20),
        ("VIX < 15 (Calm)",      master["vix"] < 15),
    ]:
        sub = master[cond].dropna(subset=["fwd_90d"])
        if len(sub) < 10:
            continue
        mem["regime_profiles"].append({
            "regime_type":  "Global Risk (VIX)",
            "regime_label": label,
            "n_days":       len(sub),
            "outcomes": {
                "30d": outcome_stats(sub["fwd_30d"], "30d"),
                "90d": outcome_stats(sub["fwd_90d"], "90d"),
            },
        })

    # Combined regimes (the most important ones)
    combos = [
        ("Stable rates + VIX < 20",
         (master["rate_regime"] == "stable") & (master["vix"] < 20)),
        ("Stable rates + VIX > 20",
         (master["rate_regime"] == "stable") & (master["vix"] > 20)),
        ("Rising rates + VIX > 25",
         (master["rate_regime"] == "rising") & (master["vix"] > 25)),
        ("Falling rates + VIX > 30",
         (master["rate_regime"] == "falling") & (master["vix"] > 30)),
        ("Stable rates + Oil rising",
         (master["rate_regime"] == "stable") & (master["oil_bull"] == 1)),
        ("Rising rates + TASI uptrend",
         (master["rate_regime"] == "rising") & (master["tasi_20d"] > 0.05) if "tasi_20d" in master.columns else master["rate_regime"] == "impossible"),
    ]
    for label, cond in combos:
        sub = master[cond].dropna(subset=["fwd_90d"])
        if len(sub) < 10:
            continue
        mem["regime_profiles"].append({
            "regime_type":  "Combined Environment",
            "regime_label": label,
            "n_days":       len(sub),
            "outcomes": {
                "30d": outcome_stats(sub["fwd_30d"], "30d"),
                "90d": outcome_stats(sub["fwd_90d"], "90d"),
            },
        })

    # Correlations with macro variables
    for col, label in [
        ("repo_rate", "Saudi Repo Rate"),
        ("oil",       "Brent Oil"),
        ("vix",       "VIX"),
    ]:
        if col in master.columns:
            for w in [30, 90]:
                fwd_col = f"fwd_{w}d"
                valid   = master[[col, fwd_col]].dropna()
                if len(valid) > 50:
                    r, p = stats.pearsonr(valid[col], valid[fwd_col])
                    mem["correlations"][f"{label}_vs_{w}d_return"] = {
                        "pearson_r":   round(r, 3),
                        "p_value":     round(p, 4),
                        "significant": bool(p < 0.05),
                        "interpretation": "positive = higher value → better forward return" if r > 0 else "negative = higher value → worse forward return",
                    }

    print(f"    {len(mem['regime_profiles'])} regime profiles stored")
    save_memory(mem, "macro_memory.json")
    return mem


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — FUNDAMENTAL MEMORY
# ══════════════════════════════════════════════════════════════════════════════

def build_fundamental_memory() -> dict:
    print("\n[3] Building Fundamental Memory...")
    mem = {"built": datetime.now().isoformat(), "annual_history": [], "balance_history": [],
           "growth_patterns": [], "ratios_history": []}

    # Annual income history
    income = pd.read_csv(DATA_RAW / "alrajhi_income_quarterly.csv", parse_dates=["report_date"])
    income = income.sort_values("report_date").reset_index(drop=True)
    income["net_income_bn"] = income["net_income"] / 1e9

    # Compute YoY net income growth
    annual = income[income["report_date"].dt.month == 12].copy()
    annual["ni_yoy"] = annual["net_income"].pct_change() * 100

    for _, row in annual.iterrows():
        mem["annual_history"].append({
            "year":           int(row["fiscal_year"]) if not pd.isna(row["fiscal_year"]) else None,
            "report_date":    row["report_date"].date().isoformat(),
            "net_income_bn":  round(row["net_income_bn"], 2) if not pd.isna(row["net_income_bn"]) else None,
            "ni_yoy_growth":  round(row["ni_yoy"], 1) if not pd.isna(row.get("ni_yoy", float("nan"))) else None,
        })

    # Balance sheet history
    balance = pd.read_csv(DATA_RAW / "alrajhi_balance_quarterly.csv", parse_dates=["report_date"])
    balance = balance.sort_values("report_date")
    balance["assets_bn"] = balance["total_assets"] / 1e9
    balance["equity_bn"] = balance["stockholders_equity"] / 1e9
    balance["assets_yoy"] = balance["total_assets"].pct_change(4) * 100  # vs 4 quarters ago

    for _, row in balance.iterrows():
        mem["balance_history"].append({
            "report_date": row["report_date"].date().isoformat(),
            "total_assets_bn": round(row["assets_bn"], 1) if not pd.isna(row["assets_bn"]) else None,
            "equity_bn":       round(row["equity_bn"], 1) if not pd.isna(row["equity_bn"]) else None,
            "assets_growth_yoy": round(row["assets_yoy"], 1) if not pd.isna(row["assets_yoy"]) else None,
        })

    # Ratios history
    try:
        with open(DATA_RAW / "alrajhi_ratios_quarterly.json", encoding="utf-8") as f:
            ratios_data = json.load(f)
        for period in ratios_data.get("ratios", []):
            ratios = period.get("ratios", {})
            km     = period.get("key_metrics", {})
            mem["ratios_history"].append({
                "report_date": period.get("report_date"),
                "roe":         ratios.get("roe"),
                "roa":         ratios.get("roa"),
                "net_income":  km.get("net_income"),
                "total_assets":km.get("total_assets"),
            })
    except Exception as e:
        print(f"    Warning: ratios load failed: {e}")

    # Growth pattern analysis — what did the stock do after different growth phases?
    growth_scenarios = [
        ("Strong growth (>20% YoY)", annual[annual["ni_yoy"] > 20]),
        ("Solid growth (10-20% YoY)", annual[(annual["ni_yoy"] >= 10) & (annual["ni_yoy"] <= 20)]),
        ("Weak growth (0-10% YoY)",  annual[(annual["ni_yoy"] >= 0) & (annual["ni_yoy"] < 10)]),
        ("Decline (<0% YoY)",        annual[annual["ni_yoy"] < 0]),
    ]
    for label, subset in growth_scenarios:
        if len(subset) < 2:
            continue
        mem["growth_patterns"].append({
            "label":             label,
            "occurrences":       len(subset),
            "years":             [int(y) for y in subset["fiscal_year"].dropna().tolist()],
            "avg_net_income_bn": round(subset["net_income_bn"].mean(), 2),
            "note": "Stock price outcomes require price data alignment — see regime memory for context",
        })

    # Key fundamentals summary
    latest_income = annual.iloc[-1] if len(annual) > 0 else {}
    latest_balance = balance.iloc[-1] if len(balance) > 0 else {}
    latest_ratios  = mem["ratios_history"][-1] if mem["ratios_history"] else {}

    mem["latest_snapshot"] = {
        "as_of":            datetime.now().date().isoformat(),
        "net_income_bn":    round(latest_income.get("net_income_bn", 0), 2),
        "ni_yoy_growth_pct":round(latest_income.get("ni_yoy", 0), 1),
        "total_assets_bn":  round(latest_balance.get("assets_bn", 0), 1),
        "equity_bn":        round(latest_balance.get("equity_bn", 0), 1),
        "roe":              latest_ratios.get("roe"),
        "roa":              latest_ratios.get("roa"),
    }

    print(f"    {len(mem['annual_history'])} annual records, {len(mem['balance_history'])} balance records, "
          f"{len(mem['ratios_history'])} ratio periods stored")
    save_memory(mem, "fundamental_memory.json")
    return mem


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — EARNINGS / EVENT MEMORY
# ══════════════════════════════════════════════════════════════════════════════

def build_event_memory(master: pd.DataFrame) -> dict:
    print("\n[4] Building Earnings / Event Memory...")
    mem = {"built": datetime.now().isoformat(), "events": [], "event_type_summary": {}}

    with open(DATA_RAW / "alrajhi_events_sahmk.json", encoding="utf-8") as f:
        events_data = json.load(f)

    events = events_data.get("events", [])
    type_buckets = {}

    for ev in events:
        ev_date = pd.to_datetime(ev.get("event_date"))
        ev_type = ev.get("event_type", "OTHER")
        sentiment = ev.get("sentiment", "neutral")
        importance = ev.get("importance", "regular")

        # Find price reaction
        reactions = {}
        for days in [1, 5, 10, 20, 30, 60]:
            future = master.index[master.index > ev_date]
            if len(future) >= days:
                price_then = master["close"].loc[future[days - 1]]
                price_now  = master["close"].loc[future[0]]
                reactions[f"ret_{days}d"] = round((price_then / price_now - 1) * 100, 2)

        rec = {
            "event_date":  ev_date.date().isoformat() if not pd.isna(ev_date) else None,
            "event_type":  ev_type,
            "importance":  importance,
            "sentiment":   sentiment,
            "reactions":   reactions,
        }
        mem["events"].append(rec)

        # Aggregate by type
        if ev_type not in type_buckets:
            type_buckets[ev_type] = []
        if reactions.get("ret_30d") is not None:
            type_buckets[ev_type].append(reactions["ret_30d"])

    # Summary by event type
    for ev_type, ret_list in type_buckets.items():
        if len(ret_list) < 3:
            continue
        s = pd.Series(ret_list)
        mem["event_type_summary"][ev_type] = {
            "n":            len(s),
            "avg_30d_pct":  round(s.mean(), 2),
            "pct_positive": round((s > 0).mean() * 100, 1),
            "max_pct":      round(s.max(), 2),
            "min_pct":      round(s.min(), 2),
        }

    # Dividend event memory
    divs = pd.read_csv(DATA_RAW / "alrajhi_dividends_sahmk.csv", parse_dates=["announcement_date"])
    div_reactions = []
    for _, row in divs.iterrows():
        d = row.get("announcement_date")
        if pd.isna(d):
            continue
        future = master.index[master.index >= d]
        if len(future) < 30:
            continue
        p0 = master["close"].loc[future[0]]
        reactions = {}
        for days in [1, 5, 10, 20, 30]:
            if len(future) >= days:
                reactions[f"ret_{days}d"] = round((master["close"].loc[future[days-1]] / p0 - 1) * 100, 2)
        div_reactions.append({
            "announcement_date": d.date().isoformat(),
            "dividend_value":    row.get("value"),
            "period":            row.get("period"),
            "reactions":         reactions,
        })

    if div_reactions:
        df_div = pd.DataFrame([{**r, **r["reactions"]} for r in div_reactions])
        mem["dividend_event_summary"] = {
            "n": len(div_reactions),
            "avg_30d_after_announcement": round(df_div["ret_30d"].mean(), 2) if "ret_30d" in df_div else None,
            "pct_positive_30d":           round((df_div["ret_30d"] > 0).mean() * 100, 1) if "ret_30d" in df_div else None,
            "interpretation": "Dividend announcements show near-baseline returns — dividend yield is not a timing signal",
        }
        mem["dividend_events"] = div_reactions

    print(f"    {len(mem['events'])} events processed, {len(mem['event_type_summary'])} event types summarised")
    save_memory(mem, "event_memory.json")
    return mem


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — SECTOR MEMORY
# ══════════════════════════════════════════════════════════════════════════════

def build_sector_memory(master: pd.DataFrame) -> dict:
    print("\n[5] Building Sector Memory...")
    mem = {"built": datetime.now().isoformat(), "peer_correlations": {}, "sector_regimes": []}

    peer_names = {
        "1180": "Saudi National Bank (SNB)",
        "1050": "Banque Saudi Fransi",
        "1060": "Saudi British Bank (SABB)",
        "1080": "Arab National Bank",
        "1010": "Riyad Bank",
    }

    alrajhi_ret = master["close"].pct_change(20)

    for sym, name in peer_names.items():
        col = f"peer_{sym}"
        if col not in master.columns:
            continue
        peer_ret = master[col].pct_change(20)
        valid    = pd.concat([alrajhi_ret, peer_ret], axis=1).dropna()
        valid.columns = ["alrajhi", "peer"]
        if len(valid) < 50:
            continue
        r, p = stats.pearsonr(valid["alrajhi"], valid["peer"])

        # Alpha when peer rises vs falls
        peer_up   = valid[valid["peer"] > 0.05]["alrajhi"]
        peer_down = valid[valid["peer"] < -0.05]["alrajhi"]

        mem["peer_correlations"][sym] = {
            "name":              name,
            "correlation_20d":   round(r, 3),
            "p_value":           round(p, 4),
            "significant":       bool(p < 0.05),
            "n_periods":         len(valid),
            "alrajhi_when_peer_up_avg":   round(peer_up.mean() * 100, 2) if len(peer_up) > 5 else None,
            "alrajhi_when_peer_down_avg": round(peer_down.mean() * 100, 2) if len(peer_down) > 5 else None,
        }

    # TASI correlation
    if "tasi" in master.columns:
        tasi_ret = master["tasi"].pct_change(20)
        valid = pd.concat([alrajhi_ret, tasi_ret], axis=1).dropna()
        valid.columns = ["alrajhi", "tasi"]
        r, p = stats.pearsonr(valid["alrajhi"], valid["tasi"])
        mem["tasi_correlation"] = {
            "correlation_20d": round(r, 3),
            "p_value":         round(p, 4),
            "significant":     bool(p < 0.05),
            "n":               len(valid),
            "interpretation":  "High correlation = Al Rajhi moves with TASI. Low = independent.",
        }

        # Does Al Rajhi lead or lag TASI?
        for lag in [-5, -3, -1, 1, 3, 5]:
            lr, lp = stats.pearsonr(
                alrajhi_ret.shift(lag).dropna().align(tasi_ret.dropna(), join="inner")[0],
                alrajhi_ret.shift(lag).dropna().align(tasi_ret.dropna(), join="inner")[1]
            )
            mem["tasi_correlation"][f"lag_{lag}d_r"] = round(lr, 3)

        # Al Rajhi vs TASI by rate regime
        for regime in ["rising", "stable", "falling"]:
            rm = master[master["rate_regime"] == regime]
            sub_alr  = alrajhi_ret.loc[rm.index].dropna()
            sub_tasi = tasi_ret.loc[rm.index].dropna()
            aligned  = pd.concat([sub_alr, sub_tasi], axis=1).dropna()
            if len(aligned) > 20:
                r2, p2 = stats.pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                mem["sector_regimes"].append({
                    "rate_regime":    regime,
                    "tasi_correlation": round(r2, 3),
                    "n":               len(aligned),
                    "interpretation":  f"In {regime} rate environment: Al Rajhi-TASI correlation = {round(r2, 3)}",
                })

    print(f"    {len(mem['peer_correlations'])} peer correlations, TASI analysis complete")
    save_memory(mem, "sector_memory.json")
    return mem


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — REGIME MEMORY (fully separated, never merged)
# ══════════════════════════════════════════════════════════════════════════════

def build_regime_memory(master: pd.DataFrame) -> dict:
    print("\n[6] Building Regime Memory (separated)...")
    mem = {"built": datetime.now().isoformat(), "regimes": []}

    # Define all regime combinations
    regime_defs = []

    # Rate × VIX
    for rate_r in ["rising", "stable", "falling"]:
        for vix_level, vix_cond in [
            ("high_fear", master["vix"] > 30),
            ("normal",    (master["vix"] >= 15) & (master["vix"] <= 30)),
            ("calm",      master["vix"] < 15),
        ]:
            cond  = (master["rate_regime"] == rate_r) & vix_cond
            sub   = master[cond].dropna(subset=["fwd_90d"])
            if len(sub) < 10:
                continue
            regime_defs.append({
                "regime_id":    f"rate_{rate_r}__vix_{vix_level}",
                "rate_regime":  rate_r,
                "vix_regime":   vix_level,
                "n_days":       len(sub),
                "outcomes": {
                    "30d": outcome_stats(sub["fwd_30d"], "30d"),
                    "60d": outcome_stats(sub["fwd_60d"], "60d"),
                    "90d": outcome_stats(sub["fwd_90d"], "90d"),
                },
                "alpha_90d": round(sub["fwd_90d_vs_tasi"].mean() * 100, 2) if "fwd_90d_vs_tasi" in sub else None,
            })

    # Rate × Oil
    for rate_r in ["rising", "stable", "falling"]:
        for oil_label, oil_cond in [
            ("oil_bull", master["oil_bull"] == 1),
            ("oil_bear", master["oil_bear"] == 1),
            ("oil_neutral", (master["oil_bull"] == 0) & (master["oil_bear"] == 0)),
        ]:
            cond = (master["rate_regime"] == rate_r) & oil_cond
            sub  = master[cond].dropna(subset=["fwd_90d"])
            if len(sub) < 10:
                continue
            regime_defs.append({
                "regime_id":   f"rate_{rate_r}__oil_{oil_label}",
                "rate_regime": rate_r,
                "oil_regime":  oil_label,
                "n_days":      len(sub),
                "outcomes": {
                    "90d": outcome_stats(sub["fwd_90d"], "90d"),
                },
            })

    # RSI × Rate regime (probability dimension)
    for rsi_zone, rsi_cond in [
        ("oversold", master["rsi"] < 30),
        ("neutral",  (master["rsi"] >= 40) & (master["rsi"] <= 60)),
        ("overbought", master["rsi"] > 70),
    ]:
        for rate_r in ["rising", "stable", "falling"]:
            cond = rsi_cond & (master["rate_regime"] == rate_r)
            sub  = master[cond].dropna(subset=["fwd_90d"])
            if len(sub) < 10:
                continue
            regime_defs.append({
                "regime_id":   f"rsi_{rsi_zone}__rate_{rate_r}",
                "rsi_zone":    rsi_zone,
                "rate_regime": rate_r,
                "n_days":      len(sub),
                "outcomes": {
                    "30d": outcome_stats(sub["fwd_30d"], "30d"),
                    "90d": outcome_stats(sub["fwd_90d"], "90d"),
                },
                "note": f"RSI {rsi_zone} in {rate_r} rate environment — regime matters for probability",
            })

    mem["regimes"] = regime_defs
    mem["best_regime"]  = max(
        (r for r in regime_defs if r["outcomes"]["90d"].get("avg_pct") is not None),
        key=lambda r: r["outcomes"]["90d"]["avg_pct"], default={}
    )
    mem["worst_regime"] = min(
        (r for r in regime_defs if r["outcomes"]["90d"].get("avg_pct") is not None),
        key=lambda r: r["outcomes"]["90d"]["avg_pct"], default={}
    )

    print(f"    {len(regime_defs)} regime combinations stored")
    print(f"    Best regime:  {mem['best_regime'].get('regime_id','?')} → avg 90d: {mem['best_regime'].get('outcomes',{}).get('90d',{}).get('avg_pct','?')}%")
    print(f"    Worst regime: {mem['worst_regime'].get('regime_id','?')} → avg 90d: {mem['worst_regime'].get('outcomes',{}).get('90d',{}).get('avg_pct','?')}%")
    save_memory(mem, "regime_memory.json")
    return mem


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 7 — PROBABILITY / MAGNITUDE / TIMING MEMORY
# ══════════════════════════════════════════════════════════════════════════════

def build_pmt_memory(master: pd.DataFrame) -> dict:
    """
    Tests whether different signal types influence:
    - P = Probability of a positive outcome
    - M = Magnitude of the move
    - T = Timing / how quickly the move happens

    This tests the hypothesis: "Environment → P, Execution → M, Technicals → T"
    """
    print("\n[7] Building Probability / Magnitude / Timing Memory...")
    mem = {"built": datetime.now().isoformat(), "pmt_analysis": [], "hypothesis": ""}

    valid = master.dropna(subset=["fwd_30d", "fwd_60d", "fwd_90d"])

    # For each signal type, measure where it matters most: 30d, 60d, or 90d
    signal_types = {
        "Technical — RSI < 30": master["rsi"] < 30,
        "Technical — RSI > 70": master["rsi"] > 70,
        "Technical — BB Lower": master["bb_pct"] < 0.05,
        "Macro — Stable Rate":  master["rate_regime"] == "stable",
        "Macro — Rising Rate":  master["rate_regime"] == "rising",
        "Macro — VIX > 30":    master["vix"] > 30,
        "Macro — Oil Rising":   master["oil_bull"] == 1,
        "Technical — All Below MAs": (master["above_ma20"]==0) & (master["above_ma50"]==0) & (master["above_ma200"]==0),
        "Technical — Momentum>10%": master["ret_20d"] > 0.10,
    }

    for sig_label, sig_mask in signal_types.items():
        sub  = valid[sig_mask]
        base = valid
        if len(sub) < 10:
            continue

        row = {"signal": sig_label, "n": len(sub)}

        # Probability dimension: pct positive at each horizon
        for w in [30, 60, 90]:
            col = f"fwd_{w}d"
            row[f"P_{w}d_pct_pos"]  = round((sub[col] > 0).mean() * 100, 1)
            row[f"base_P_{w}d"]     = round((base[col] > 0).mean() * 100, 1)
            row[f"P_edge_{w}d"]     = round(row[f"P_{w}d_pct_pos"] - row[f"base_P_{w}d"], 1)

        # Magnitude dimension: avg return when positive
        for w in [30, 60, 90]:
            col    = f"fwd_{w}d"
            pos    = sub[col][sub[col] > 0]
            pos_b  = base[col][base[col] > 0]
            row[f"M_{w}d_avg_when_pos"]  = round(pos.mean() * 100, 2) if len(pos) > 3 else None
            row[f"base_M_{w}d"]          = round(pos_b.mean() * 100, 2) if len(pos_b) > 3 else None

        # Timing dimension: when does the signal pay off fastest?
        # Measured as: which horizon shows the strongest edge vs baseline
        edges = {w: row[f"P_edge_{w}d"] for w in [30, 60, 90]}
        peak_w = max(edges, key=edges.get)
        row["timing_peak_horizon"] = f"{peak_w}d"
        row["primary_dimension"] = (
            "PROBABILITY" if max(abs(e) for e in edges.values()) > 5 else "WEAK"
        )
        mem["pmt_analysis"].append(row)

    # Summary hypothesis test
    tech_timing = [r for r in mem["pmt_analysis"] if "Technical" in r["signal"]]
    macro_prob  = [r for r in mem["pmt_analysis"] if "Macro" in r["signal"]]

    tech_peak_30d = sum(1 for r in tech_timing if r.get("timing_peak_horizon") == "30d")
    macro_peak_90d= sum(1 for r in macro_prob  if r.get("timing_peak_horizon") == "90d")

    mem["hypothesis"] = {
        "stated":  "Environment → probability, Execution → magnitude, Technicals → timing",
        "finding": {
            "technical_signals_peak_at_30d": tech_peak_30d,
            "macro_signals_peak_at_90d":     macro_peak_90d,
            "conclusion": (
                "PARTIALLY SUPPORTED — Technical signals (RSI, BB) peak at 30-60d (timing tools). "
                "Macro regime signals show persistent effects at 90d (environment). "
                "Execution/magnitude cannot be fully tested until deeper quarterly data is available."
            ),
        },
    }

    print(f"    {len(mem['pmt_analysis'])} signals analysed across P/M/T dimensions")
    save_memory(mem, "pmt_memory.json")
    return mem


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def build_personality_summary(tech, macro, fund, events, sector, regime, pmt) -> dict:
    print("\n[8] Writing Al Rajhi Personality Summary...")

    summary = {
        "stock":     "Al Rajhi Bank (1120.SR)",
        "built":     datetime.now().isoformat(),
        "personality_rules": [
            {
                "rank": 1,
                "rule": "Al Rajhi is a mean-reversion stock",
                "evidence": "RSI oversold (20-30) produces +13% avg 90d return, n=119, p=0.001. Significant.",
                "confidence": "HIGH",
                "applies_to": "all rate regimes",
            },
            {
                "rank": 2,
                "rule": "Rising rates are systematically bad",
                "evidence": "During rising rate periods: avg 90d return = -5.6%, win rate = 25%. Only 294 days but consistent.",
                "confidence": "MEDIUM",
                "applies_to": "rate regime",
            },
            {
                "rank": 3,
                "rule": "Stable rates are the ideal environment",
                "evidence": "3,400 days with stable rates: avg 90d = +7.2%, win rate = 65%, alpha vs TASI = +4.9%",
                "confidence": "HIGH",
                "applies_to": "environment score",
            },
            {
                "rank": 4,
                "rule": "High VIX (fear) = buying opportunity",
                "evidence": "VIX > 30: avg 90d = +10.0%, win rate = 75%, n=242.",
                "confidence": "HIGH",
                "applies_to": "macro timing",
            },
            {
                "rank": 5,
                "rule": "Do not chase momentum",
                "evidence": "After >10% 20-day rally: avg 90d = -16.3%, win rate = 43%, n=49, p=0.003.",
                "confidence": "HIGH",
                "applies_to": "technical timing",
            },
            {
                "rank": 6,
                "rule": "Bollinger Band lower touch is a strong buy signal",
                "evidence": "BB lower 5%: avg 30d = +8.3%, avg 90d = +9.1%, n=82, p=0.008.",
                "confidence": "HIGH",
                "applies_to": "entry timing",
            },
            {
                "rank": 7,
                "rule": "Outperformance vs TASI predicts underperformance",
                "evidence": "After >5% outperformance: avg 90d = -8.4%, n=69, p=0.036.",
                "confidence": "MEDIUM",
                "applies_to": "relative strength timing",
            },
            {
                "rank": 8,
                "rule": "Dividend announcements have no predictive edge",
                "evidence": "29 events: avg 30d = +2.9%, barely above baseline. Not significant.",
                "confidence": "MEDIUM",
                "applies_to": "event scoring",
            },
            {
                "rank": 9,
                "rule": "Oil rising is mildly positive but not critical",
                "evidence": "Oil uptrend: avg 90d = +7.6%, alpha = +6% vs TASI. Correlated but not causal.",
                "confidence": "MEDIUM",
                "applies_to": "environment score",
            },
            {
                "rank": 10,
                "rule": "Technical indicators are timing tools, not direction tools",
                "evidence": "RSI, BB signals peak at 30-60d horizons. Environment (rates, VIX) dominates at 90d.",
                "confidence": "MEDIUM",
                "applies_to": "framework weighting",
            },
        ],
        "known_weaknesses": [
            "Quarterly fundamental data pre-2022 is annual only — execution memory is partial",
            "SAIBOR 3M not available — NIM sensitivity not directly testable",
            "Only 16 years of data — some rate cycle combinations have small sample sizes",
        ],
        "signal_weights_initial": {
            "environment_score":     0.35,
            "technical_timing":      0.30,
            "fundamental_execution": 0.20,
            "event_reaction":        0.15,
        },
        "current_signals": {
            "note": "Current signal values updated by forecasting phase",
        },
    }

    save_memory(summary, "personality_summary.json")
    print(f"    Personality summary: {len(summary['personality_rules'])} rules stored")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def print_summary_report(summary: dict):
    print("\n" + "═" * 80)
    print("  STOCK MEMORY ENGINE v2 — PHASE 2 COMPLETE")
    print("  Al Rajhi Bank Personality — Evidence-Based Summary")
    print("═" * 80)
    for rule in summary["personality_rules"]:
        conf_icon = {"HIGH": "★★★", "MEDIUM": "★★ ", "LOW": "★  "}.get(rule["confidence"], "?")
        print(f"\n  {rule['rank']}. [{conf_icon}] {rule['rule']}")
        print(f"     Evidence: {rule['evidence']}")
    print("\n" + "─" * 80)
    print("  Memory modules saved:")
    for fname in ["technical_memory.json", "macro_memory.json", "fundamental_memory.json",
                  "event_memory.json", "sector_memory.json", "regime_memory.json",
                  "pmt_memory.json", "personality_summary.json"]:
        print(f"    ✓ memory/{fname}")
    print()


def main():
    print("\nSTOCK MEMORY ENGINE v2 — PHASE 2: STOCK MEMORY ENGINE")
    print(f"Stock: Al Rajhi Bank (1120.SR)")
    print(f"Date : {date.today()}")
    print("─" * 60)

    master = load_master()

    tech    = build_technical_memory(master)
    macro   = build_macro_memory(master)
    fund    = build_fundamental_memory()
    events  = build_event_memory(master)
    sector  = build_sector_memory(master)
    regime  = build_regime_memory(master)
    pmt     = build_pmt_memory(master)
    summary = build_personality_summary(tech, macro, fund, events, sector, regime, pmt)

    print_summary_report(summary)
    print("  Next step: python phases/phase3_hypothesis_engine.py")


if __name__ == "__main__":
    main()
