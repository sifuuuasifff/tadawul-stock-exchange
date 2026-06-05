"""
STOCK EXPANSION — Saudi Chemical (2230), Maaden (1211), Jamjoom Pharma (4139)
+ Al Rajhi events pagination fix
Pulls all data via SAHMK, runs signal discovery and memory for each stock.

Run:  python phases/expand_stocks.py
"""

import sys
import json
import time
import warnings
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import DATA_RAW, DATA_PROCESSED, MEMORY_DIR, REPORTS_DIR, SAHMK_API_KEY
from config.providers import SAHMKProvider, YahooProvider, MacroProvider

DATA_RAW.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

NEW_STOCKS = {
    "2230": {"name": "Saudi Chemical",   "yahoo": "2230.SR", "sector": "Chemicals / Healthcare Dist."},
    "1211": {"name": "Maaden",           "yahoo": "1211.SR", "sector": "Mining"},
    "4139": {"name": "Jamjoom Pharma",   "yahoo": "4139.SR", "sector": "Pharmaceuticals"},
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Fix Al Rajhi events — paginate to get full history
# ══════════════════════════════════════════════════════════════════════════════

def fix_alrajhi_events(api: SAHMKProvider):
    print("\n[Fix] Al Rajhi events — pulling full history with pagination...")
    all_events = []

    # Try pulling with higher limit and different type filters
    for event_type in ["FINANCIAL_REPORT", "DIVIDEND_ANNOUNCEMENT", "REGULATORY_ACTION",
                        "MANAGEMENT_CHANGE", "EARNINGS_SURPRISE", "ANALYST_RATING_CHANGE"]:
        try:
            data = api.get_events("1120", event_type=event_type, limit=500)
            events = data.get("events", data.get("results", data.get("data", [])))
            all_events.extend(events)
            print(f"  {event_type}: {len(events)} events")
            time.sleep(0.3)
        except Exception as e:
            print(f"  {event_type}: error — {e}")

    # Also pull unfiltered (catches OTHER types)
    try:
        data = api.get_events("1120", limit=500)
        events = data.get("events", [])
        # Deduplicate by event_date + event_type
        all_events.extend(events)
    except Exception as e:
        print(f"  Unfiltered: {e}")

    # Deduplicate
    seen = set()
    unique = []
    for e in all_events:
        key = (e.get("event_date"), e.get("event_type"))
        if key not in seen:
            seen.add(key)
            unique.append(e)

    unique.sort(key=lambda x: x.get("event_date", ""), reverse=True)

    out = {"events": unique, "count": len(unique), "available_types": list(set(e["event_type"] for e in unique))}
    with open(DATA_RAW / "alrajhi_events_full.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"  Total unique events after pagination: {len(unique)}")

    # Parse into clean DataFrame
    df = pd.DataFrame(unique)
    if not df.empty:
        df["event_date"] = pd.to_datetime(df["event_date"])
        df = df.sort_values("event_date")
        df.to_csv(DATA_RAW / "alrajhi_events_full.csv", index=False)

    return unique


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Pull data for each new stock
# ══════════════════════════════════════════════════════════════════════════════

def pull_stock_data(sym: str, info: dict, api: SAHMKProvider, yahoo: YahooProvider) -> dict:
    name   = info["name"]
    folder = DATA_RAW / f"stock_{sym}"
    folder.mkdir(exist_ok=True)
    log    = {}

    print(f"\n  [{sym}] {name}")

    # Price from SAHMK
    try:
        df = api.get_historical_price(sym, from_date="2010-01-01", interval="1d")
        df.to_csv(folder / "price_sahmk.csv")
        log["price_sahmk"] = {"rows": len(df), "start": str(df.index[0].date()), "end": str(df.index[-1].date())}
        print(f"    Price (SAHMK): {len(df)} rows")
    except Exception as e:
        print(f"    Price (SAHMK) failed: {e}")
        log["price_sahmk"] = {"error": str(e)}

    time.sleep(0.4)

    # Price from Yahoo as backup / longer history
    try:
        df_y = yahoo.get_historical_price(info["yahoo"], start="2010-01-01")
        df_y.to_csv(folder / "price_yahoo.csv")
        log["price_yahoo"] = {"rows": len(df_y)}
        print(f"    Price (Yahoo): {len(df_y)} rows")
    except Exception as e:
        print(f"    Price (Yahoo) failed: {e}")
        log["price_yahoo"] = {"error": str(e)}

    # Quarterly financials
    try:
        data = api.get_financials(sym, period="quarterly", history="max", stmt_type="all")
        with open(folder / "financials_quarterly.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        income = data.get("income_statements", [])
        balance = data.get("balance_sheets", [])
        log["financials"] = {"income_quarters": len(income), "balance_quarters": len(balance)}
        print(f"    Financials: {len(income)} income, {len(balance)} balance periods")
    except Exception as e:
        print(f"    Financials failed: {e}")
        log["financials"] = {"error": str(e)}

    time.sleep(0.4)

    # Ratios
    try:
        data = api.get_ratios(sym, period="quarterly", history="max")
        with open(folder / "ratios.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        periods = data.get("ratios", [])
        log["ratios"] = {"periods": len(periods)}
        print(f"    Ratios: {len(periods)} periods")
    except Exception as e:
        print(f"    Ratios failed: {e}")

    time.sleep(0.4)

    # Dividends
    try:
        data = api.get_dividends(sym, limit=200)
        with open(folder / "dividends.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        hist = data.get("history", [])
        log["dividends"] = {"events": len(hist)}
        print(f"    Dividends: {len(hist)} events")
    except Exception as e:
        print(f"    Dividends failed: {e}")

    time.sleep(0.4)

    # Events
    try:
        all_events = []
        for ev_type in ["FINANCIAL_REPORT", "DIVIDEND_ANNOUNCEMENT", "REGULATORY_ACTION"]:
            d = api.get_events(sym, event_type=ev_type, limit=200)
            all_events.extend(d.get("events", []))
            time.sleep(0.2)
        d2 = api.get_events(sym, limit=200)
        all_events.extend(d2.get("events", []))

        seen = set()
        unique = []
        for e in all_events:
            key = (e.get("event_date"), e.get("event_type"))
            if key not in seen:
                seen.add(key)
                unique.append(e)

        out = {"events": unique, "count": len(unique)}
        with open(folder / "events.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        log["events"] = {"count": len(unique)}
        print(f"    Events: {len(unique)} unique events")
    except Exception as e:
        print(f"    Events failed: {e}")

    time.sleep(0.4)

    # Company profile
    try:
        data = api.get_company(sym)
        with open(folder / "profile.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"    Profile: retrieved")
    except Exception as e:
        print(f"    Profile failed: {e}")

    return log


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Build master dataset for each new stock
# ══════════════════════════════════════════════════════════════════════════════

def build_stock_master(sym: str, info: dict) -> pd.DataFrame:
    folder = DATA_RAW / f"stock_{sym}"
    macro  = MacroProvider()

    # Load price — prefer Yahoo (more rows)
    yahoo_path = folder / "price_yahoo.csv"
    sahmk_path = folder / "price_sahmk.csv"

    if yahoo_path.exists():
        price = pd.read_csv(yahoo_path, index_col=0, parse_dates=True)
        price.index = pd.to_datetime(price.index).tz_localize(None)
        price = price[["close", "volume"]].dropna()
    elif sahmk_path.exists():
        price = pd.read_csv(sahmk_path, index_col=0, parse_dates=True)
        price.index = pd.to_datetime(price.index).tz_localize(None)
        price = price[["close", "volume"]].dropna()
    else:
        print(f"  No price data for {sym}")
        return pd.DataFrame()

    # Load macro (reuse the same files)
    tasi  = pd.read_csv(DATA_RAW / "tasi_daily.csv", index_col=0, parse_dates=True)
    tasi.index = pd.to_datetime(tasi.index).tz_localize(None)
    tasi.columns = ["tasi"]

    oil   = pd.read_csv(DATA_RAW / "brent_oil.csv", index_col=0, parse_dates=True)
    oil.index = pd.to_datetime(oil.index).tz_localize(None)
    oil.columns = ["oil"]

    vix   = pd.read_csv(DATA_RAW / "vix.csv", index_col=0, parse_dates=True)
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix.columns = ["vix"]

    rate  = pd.read_csv(DATA_RAW / "saudi_repo_rate.csv", index_col=0, parse_dates=True)
    rate.index = pd.to_datetime(rate.index).tz_localize(None)
    rate.columns = ["repo_rate"]

    # Al Rajhi price for relative comparison
    alr   = pd.read_csv(DATA_RAW / "alrajhi_price_daily.csv", index_col=0, parse_dates=True)
    alr.index = pd.to_datetime(alr.index).tz_localize(None)
    alr   = alr[["close"]].rename(columns={"close": "alrajhi"})

    master = price.copy()
    for df in [tasi, oil, vix, rate, alr]:
        master = master.join(df, how="left")
    master = master.ffill()

    c = master["close"]

    # Technical indicators
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    master["rsi"] = 100 - (100 / (1 + rs))

    master["ma20"]  = c.rolling(20).mean()
    master["ma50"]  = c.rolling(50).mean()
    master["ma200"] = c.rolling(200).mean()
    master["above_ma20"]  = (c > master["ma20"]).astype(int)
    master["above_ma50"]  = (c > master["ma50"]).astype(int)
    master["above_ma200"] = (c > master["ma200"]).astype(int)

    master["ret_1d"]  = c.pct_change(1)
    master["ret_20d"] = c.pct_change(20)
    master["vol_ratio"]= master["volume"] / master["volume"].rolling(20).mean()

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    master["bb_pct"] = (c - (bb_mid - 2*bb_std)) / (4 * bb_std)

    master["high_52w"]       = c.rolling(252).max()
    master["dist_from_high"] = (c - master["high_52w"]) / master["high_52w"]

    # Rate regime
    r = master["repo_rate"]
    master["rate_change_3m"] = r - r.shift(63)
    master["rate_regime"] = "stable"
    master.loc[master["rate_change_3m"] > 0.25, "rate_regime"] = "rising"
    master.loc[master["rate_change_3m"] < -0.25, "rate_regime"] = "falling"

    master["vix_high"]  = (master["vix"] > 30).astype(int)
    master["oil_bull"]  = (master["oil"].pct_change(63) > 0.10).astype(int)
    master["oil_bear"]  = (master["oil"].pct_change(63) < -0.10).astype(int)

    if "tasi" in master.columns:
        master["tasi_20d"] = master["tasi"].pct_change(20)
        master["rs_vs_tasi_20d"] = master["ret_20d"] - master["tasi"].pct_change(20)

    # Forward returns
    for w in [30, 60, 90]:
        master[f"fwd_{w}d"] = c.shift(-w) / c - 1
        if "tasi" in master.columns:
            master[f"fwd_{w}d_vs_tasi"] = c.shift(-w) / c - master["tasi"].shift(-w) / master["tasi"]

    master.to_csv(DATA_PROCESSED / f"master_{sym}.csv")
    return master


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Signal discovery for each stock (condensed version)
# ══════════════════════════════════════════════════════════════════════════════

def quick_signal_discovery(sym: str, name: str, master: pd.DataFrame) -> dict:
    """Runs the most important signal tests for a new stock."""
    valid = master.dropna(subset=["fwd_90d"])
    results = {}

    def test(label, mask, metric="fwd_90d"):
        sub = valid[mask.reindex(valid.index, fill_value=False)]
        if len(sub) < 10:
            return {"n": len(sub), "insufficient": True}
        fwd = sub[metric].dropna()
        t, p = stats.ttest_1samp(fwd, 0)
        return {
            "n":           len(sub),
            "avg_pct":     round(fwd.mean() * 100, 2),
            "win_pct":     round((fwd > 0).mean() * 100, 1),
            "p_value":     round(p, 3),
            "significant": bool(p < 0.05),
        }

    # Baseline
    results["baseline_90d"] = {
        "avg_pct": round(valid["fwd_90d"].mean() * 100, 2),
        "win_pct": round((valid["fwd_90d"] > 0).mean() * 100, 1),
        "n":       len(valid),
    }

    # Key signals
    results["rsi_oversold"]       = test("RSI<30",         master["rsi"] < 30)
    results["rsi_overbought"]     = test("RSI>70",         master["rsi"] > 70)
    results["bb_lower"]           = test("BB lower",       master["bb_pct"] < 0.05)
    results["momentum_strong_up"] = test("Ret20>10%",      master["ret_20d"] > 0.10)
    results["momentum_strong_dn"] = test("Ret20<-10%",     master["ret_20d"] < -0.10)
    results["stable_rates"]       = test("Stable rates",   master["rate_regime"] == "stable")
    results["rising_rates"]       = test("Rising rates",   master["rate_regime"] == "rising")
    results["vix_fear"]           = test("VIX>30",         master["vix"] > 30)
    results["oil_bull"]           = test("Oil rising",     master["oil_bull"] == 1)
    results["all_below_mas"]      = test("All below MAs",  (master["above_ma20"]==0)&(master["above_ma50"]==0)&(master["above_ma200"]==0))
    results["tasi_uptrend"]       = test("TASI up 5%",     master.get("tasi_20d", pd.Series(dtype=float)) > 0.05 if "tasi_20d" in master.columns else pd.Series(False, index=master.index))

    # Regime analysis
    regime_results = {}
    for regime in ["rising", "stable", "falling"]:
        sub = valid[valid["rate_regime"] == regime]
        if len(sub) >= 10:
            regime_results[regime] = {
                "n":       len(sub),
                "avg_90d": round(sub["fwd_90d"].mean() * 100, 2),
                "win_pct": round((sub["fwd_90d"] > 0).mean() * 100, 1),
            }
    results["regime_breakdown"] = regime_results

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Stock personality summary
# ══════════════════════════════════════════════════════════════════════════════

def build_stock_personality(sym: str, info: dict, signals: dict, master: pd.DataFrame) -> dict:
    name   = info["name"]
    sector = info["sector"]
    valid  = master.dropna(subset=["fwd_90d"])

    # Load financials if available
    fin_path = DATA_RAW / f"stock_{sym}" / "financials_quarterly.json"
    fin_summary = {}
    if fin_path.exists():
        with open(fin_path, encoding="utf-8") as f:
            fin = json.load(f)
        income = fin.get("income_statements", [])
        if income:
            ni_series = [r.get("net_income") for r in income if r.get("net_income")]
            if len(ni_series) >= 2:
                fin_summary = {
                    "n_income_periods": len(income),
                    "latest_net_income_bn": round(income[0].get("net_income", 0) / 1e9, 2) if income else None,
                    "earliest_date": income[-1].get("report_date") if income else None,
                    "latest_date":   income[0].get("report_date")  if income else None,
                }

    # Load events
    ev_path = DATA_RAW / f"stock_{sym}" / "events.json"
    ev_summary = {}
    if ev_path.exists():
        with open(ev_path, encoding="utf-8") as f:
            ev_data = json.load(f)
        events = ev_data.get("events", [])
        from collections import Counter
        type_counts = Counter(e["event_type"] for e in events)
        ev_summary = {"total": len(events), "by_type": dict(type_counts)}

    # Personality rules (evidence-based)
    rules = []

    # RSI
    rsi_os = signals.get("rsi_oversold", {})
    if not rsi_os.get("insufficient") and rsi_os.get("avg_pct", 0) > signals.get("baseline_90d", {}).get("avg_pct", 0):
        rules.append({
            "rule": f"Mean-reversion tendency: RSI oversold produces above-average returns",
            "evidence": f"RSI<30: avg={rsi_os.get('avg_pct')}% vs baseline={signals['baseline_90d']['avg_pct']}%, n={rsi_os.get('n')}",
            "confidence": "HIGH" if rsi_os.get("significant") else "MEDIUM",
        })

    # Rate regime
    rd = signals.get("regime_breakdown", {})
    if "rising" in rd and "stable" in rd:
        diff = rd["stable"]["avg_90d"] - rd["rising"]["avg_90d"]
        rules.append({
            "rule": f"Rate regime matters: stable={rd['stable']['avg_90d']}% avg vs rising={rd['rising']['avg_90d']}% avg (Δ={diff:+.1f}%)",
            "evidence": f"n_stable={rd['stable']['n']}, n_rising={rd['rising']['n']}",
            "confidence": "HIGH" if abs(diff) > 5 else "MEDIUM",
        })

    # Momentum
    mom = signals.get("momentum_strong_up", {})
    if not mom.get("insufficient") and mom.get("avg_pct", 0) < signals.get("baseline_90d", {}).get("avg_pct", 0):
        rules.append({
            "rule": "Strong momentum (>10% 20-day rally) does not continue — mean reversion observed",
            "evidence": f"After >10% rally: avg={mom.get('avg_pct')}%, win={mom.get('win_pct')}%, n={mom.get('n')}",
            "confidence": "MEDIUM" if mom.get("significant") else "LOW",
        })

    personality = {
        "stock":          name,
        "symbol":         sym,
        "sector":         sector,
        "built":          datetime.now().isoformat(),
        "price_history":  f"{master.index[0].date()} → {master.index[-1].date()}" if not master.empty else "N/A",
        "n_trading_days": len(master),
        "latest_price":   round(master["close"].iloc[-1], 2) if not master.empty else None,
        "baseline_90d":   signals.get("baseline_90d", {}),
        "key_signals":    signals,
        "personality_rules": rules,
        "fundamentals_summary": fin_summary,
        "events_summary":       ev_summary,
    }

    path = MEMORY_DIR / f"personality_{sym}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(personality, f, indent=2, default=str)
    print(f"  Personality saved → memory/personality_{sym}.json")
    return personality


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\nSTOCK MEMORY ENGINE v2 — STOCK EXPANSION")
    print(f"Adding: Saudi Chemical (2230), Maaden (1211), Jamjoom Pharma (4139)")
    print(f"Date: {date.today()}")
    print("─" * 60)

    if not SAHMK_API_KEY:
        print("ERROR: SAHMK_API_KEY not set")
        return

    api   = SAHMKProvider()
    yahoo = YahooProvider()

    # Fix Al Rajhi events
    print("\n=== STEP 1: Fix Al Rajhi Events (full history) ===")
    alr_events = fix_alrajhi_events(api)

    # Pull data for each new stock
    print("\n=== STEP 2: Pulling Data for New Stocks ===")
    all_logs = {}
    for sym, info in NEW_STOCKS.items():
        log = pull_stock_data(sym, info, api, yahoo)
        all_logs[sym] = log

    # Build master datasets and signal discovery
    print("\n=== STEP 3: Building Master Datasets + Signal Discovery ===")
    personalities = {}
    summary_table = []

    for sym, info in NEW_STOCKS.items():
        print(f"\n--- {info['name']} ({sym}) ---")
        master = build_stock_master(sym, info)
        if master.empty:
            print(f"  Skipping {sym} — no price data")
            continue
        print(f"  Master: {len(master)} trading days, {len(master.columns)} columns")

        signals     = quick_signal_discovery(sym, info["name"], master)
        personality = build_stock_personality(sym, info, signals, master)
        personalities[sym] = personality

        baseline = signals.get("baseline_90d", {})
        stable   = signals.get("regime_breakdown", {}).get("stable", {})
        rising   = signals.get("regime_breakdown", {}).get("rising", {})

        summary_table.append({
            "symbol":             sym,
            "name":               info["name"],
            "sector":             info["sector"],
            "trading_days":       len(master),
            "date_range":         f"{master.index[0].date()} → {master.index[-1].date()}",
            "latest_price":       round(master["close"].iloc[-1], 2),
            "baseline_90d_avg":   baseline.get("avg_pct"),
            "baseline_win_pct":   baseline.get("win_pct"),
            "stable_regime_90d":  stable.get("avg_90d"),
            "rising_regime_90d":  rising.get("avg_90d"),
            "rsi_oversold_90d":   signals.get("rsi_oversold", {}).get("avg_pct"),
            "momentum_signal":    signals.get("momentum_strong_up", {}).get("avg_pct"),
        })

    # Print summary
    print("\n" + "═" * 80)
    print("  EXPANSION SUMMARY")
    print("═" * 80)
    print(f"\n  Al Rajhi events (full): {len(alr_events)} unique events retrieved")
    print()
    print(f"  {'Stock':<22} {'Days':>5} {'Latest':>8} {'Base90d':>8} {'Win%':>6} {'Stable90d':>10} {'Rising90d':>10}")
    print(f"  {'-'*22} {'-'*5} {'-'*8} {'-'*8} {'-'*6} {'-'*10} {'-'*10}")
    for row in summary_table:
        print(f"  {row['name']:<22} {row['trading_days']:>5} {str(row['latest_price']):>8} "
              f"{str(row['baseline_90d_avg']):>8} {str(row['baseline_win_pct']):>6} "
              f"{str(row['stable_regime_90d']):>10} {str(row['rising_regime_90d']):>10}")

    # Compare personalities
    print(f"\n  COMPARISON vs Al Rajhi Bank:")
    print(f"  Al Rajhi: mean-reversion, stable rates = ideal, rising rates = bad (-12.8% edge)")
    for row in summary_table:
        mom = personalities.get(row["symbol"], {}).get("key_signals", {}).get("momentum_strong_up", {})
        rsi = personalities.get(row["symbol"], {}).get("key_signals", {}).get("rsi_oversold", {})
        rd  = personalities.get(row["symbol"], {}).get("key_signals", {}).get("regime_breakdown", {})
        rate_diff = (rd.get("stable", {}).get("avg_90d", 0) or 0) - (rd.get("rising", {}).get("avg_90d", 0) or 0)
        print(f"\n  {row['name']} ({row['symbol']}):")
        print(f"    RSI oversold 90d avg: {rsi.get('avg_pct','N/A')}% (n={rsi.get('n','?')})")
        print(f"    Stable vs rising rate edge: {rate_diff:+.1f}%")
        rules = personalities.get(row["symbol"], {}).get("personality_rules", [])
        for r in rules[:2]:
            print(f"    Rule: {r['rule'][:75]}")

    # Save full expansion log
    with open(REPORTS_DIR / "EXPANSION_REPORT.json", "w", encoding="utf-8") as f:
        json.dump({
            "generated":    datetime.now().isoformat(),
            "new_stocks":   summary_table,
            "data_logs":    all_logs,
        }, f, indent=2, default=str)

    print(f"\n  Report → reports/EXPANSION_REPORT.json")
    print(f"  Memory → memory/personality_{{sym}}.json for each stock")
    print()


if __name__ == "__main__":
    main()
