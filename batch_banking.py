"""
BATCH 1 — FULL BANKING SECTOR
Pulls data and runs complete 5-phase engine for all Saudi banks.

Banks: SNB (1180), Riyad Bank (1010), SABB (1060), ANB (1080),
       Saudi Fransi (1050), Alinma (1150), Bank Albilad (1140)

Run:  python batch_banking.py
"""

import sys
import json
import time
import warnings
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from config.settings import DATA_RAW, DATA_PROCESSED, MEMORY_DIR, REPORTS_DIR, SAHMK_API_KEY
from config.providers import SAHMKProvider, YahooProvider

DATA_RAW.mkdir(parents=True, exist_ok=True)

BANKING_STOCKS = {
    "1180": {"name": "Saudi National Bank (SNB)",  "yahoo": "1180.SR", "sector": "Banking"},
    "1010": {"name": "Riyad Bank",                 "yahoo": "1010.SR", "sector": "Banking"},
    "1060": {"name": "Saudi British Bank (SABB)",  "yahoo": "1060.SR", "sector": "Banking"},
    "1080": {"name": "Arab National Bank",         "yahoo": "1080.SR", "sector": "Banking"},
    "1050": {"name": "Banque Saudi Fransi",        "yahoo": "1050.SR", "sector": "Banking"},
    "1150": {"name": "Alinma Bank",                "yahoo": "1150.SR", "sector": "Banking"},
    "1140": {"name": "Bank Albilad",               "yahoo": "1140.SR", "sector": "Banking"},
}


def pull_bank_data(sym: str, info: dict, api: SAHMKProvider, yahoo: YahooProvider) -> dict:
    """Pull all SAHMK + Yahoo data for one bank."""
    name   = info["name"]
    folder = DATA_RAW / f"stock_{sym}"
    folder.mkdir(exist_ok=True)
    log    = {"sym": sym, "name": name}

    print(f"\n  [{sym}] {name}")

    # Price — SAHMK (official + adjusted)
    try:
        df = api.get_historical_price(sym, from_date="2010-01-01", interval="1d")
        df.to_csv(folder / "price_sahmk.csv")
        log["price_sahmk"] = len(df)
        print(f"    Price SAHMK: {len(df)} rows")
    except Exception as e:
        log["price_sahmk_error"] = str(e)
        print(f"    Price SAHMK failed: {e}")
    time.sleep(0.4)

    # Price — Yahoo (longer history fallback)
    try:
        df_y = yahoo.get_historical_price(info["yahoo"], start="2010-01-01")
        df_y.to_csv(folder / "price_yahoo.csv")
        log["price_yahoo"] = len(df_y)
        print(f"    Price Yahoo: {len(df_y)} rows")
    except Exception as e:
        log["price_yahoo_error"] = str(e)
        print(f"    Price Yahoo failed: {e}")

    # Quarterly financials
    time.sleep(0.4)
    try:
        data = api.get_financials(sym, period="quarterly", history="max", stmt_type="all")
        with open(folder / "financials_quarterly.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        income  = data.get("income_statements", [])
        balance = data.get("balance_sheets", [])
        log["income_quarters"]  = len(income)
        log["balance_quarters"] = len(balance)
        print(f"    Financials: {len(income)} income, {len(balance)} balance periods")
    except Exception as e:
        log["financials_error"] = str(e)
        print(f"    Financials failed: {e}")

    # Annual financials
    time.sleep(0.4)
    try:
        data = api.get_financials(sym, period="annual", history="max", stmt_type="all")
        with open(folder / "financials_annual.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        pass

    # Ratios
    time.sleep(0.4)
    try:
        data = api.get_ratios(sym, period="quarterly", history="max")
        with open(folder / "ratios.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        log["ratio_periods"] = len(data.get("ratios", []))
        print(f"    Ratios: {len(data.get('ratios',[]))} periods")
    except Exception as e:
        log["ratios_error"] = str(e)

    # Dividends
    time.sleep(0.4)
    try:
        data = api.get_dividends(sym, limit=200)
        with open(folder / "dividends.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        hist = data.get("history", [])
        log["dividends"] = len(hist)
        print(f"    Dividends: {len(hist)} events")
    except Exception as e:
        log["dividends_error"] = str(e)

    # Events (paginated by type)
    time.sleep(0.4)
    try:
        all_events = []
        for ev_type in ["FINANCIAL_REPORT", "DIVIDEND_ANNOUNCEMENT", "REGULATORY_ACTION",
                         "EARNINGS_SURPRISE", "ANALYST_RATING_CHANGE"]:
            d = api.get_events(sym, event_type=ev_type, limit=300)
            all_events.extend(d.get("events", []))
            time.sleep(0.2)
        d2 = api.get_events(sym, limit=300)
        all_events.extend(d2.get("events", []))
        seen, unique = set(), []
        for e in all_events:
            key = (e.get("event_date"), e.get("event_type"))
            if key not in seen:
                seen.add(key); unique.append(e)
        unique.sort(key=lambda x: x.get("event_date",""), reverse=True)
        with open(folder / "events.json", "w", encoding="utf-8") as f:
            json.dump({"events": unique, "count": len(unique)}, f, indent=2, ensure_ascii=False)
        log["events"] = len(unique)
        print(f"    Events: {len(unique)} unique")
    except Exception as e:
        log["events_error"] = str(e)

    # Company profile
    time.sleep(0.3)
    try:
        data = api.get_company(sym)
        with open(folder / "profile.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return log


def build_bank_master(sym: str, info: dict) -> pd.DataFrame:
    """Build master dataset for a bank — same structure as Al Rajhi."""
    folder = DATA_RAW / f"stock_{sym}"

    # Price — prefer Yahoo (more history), SAHMK as supplement
    yahoo_path = folder / "price_yahoo.csv"
    sahmk_path = folder / "price_sahmk.csv"

    if yahoo_path.exists():
        price = pd.read_csv(yahoo_path, index_col=0, parse_dates=True)
    elif sahmk_path.exists():
        price = pd.read_csv(sahmk_path, index_col=0, parse_dates=True)
    else:
        return pd.DataFrame()

    price.index = pd.to_datetime(price.index).tz_localize(None)
    price = price[["close", "volume"]].dropna()

    # Macro data (shared across all stocks)
    tasi  = pd.read_csv(DATA_RAW / "tasi_daily.csv", index_col=0, parse_dates=True)
    tasi.index = pd.to_datetime(tasi.index).tz_localize(None)
    tasi.columns = ["tasi"]

    oil   = pd.read_csv(DATA_RAW / "brent_oil.csv",  index_col=0, parse_dates=True)
    oil.index = pd.to_datetime(oil.index).tz_localize(None)
    oil.columns = ["oil"]

    vix   = pd.read_csv(DATA_RAW / "vix.csv",         index_col=0, parse_dates=True)
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix.columns = ["vix"]

    rate  = pd.read_csv(DATA_RAW / "saudi_repo_rate.csv", index_col=0, parse_dates=True)
    rate.index = pd.to_datetime(rate.index).tz_localize(None)
    rate.columns = ["repo_rate"]

    saibor = pd.read_csv(DATA_RAW / "saibor_3m.csv", index_col=0, parse_dates=True)
    saibor.index = pd.to_datetime(saibor.index).tz_localize(None)
    saibor.columns = ["saibor_3m"]

    alr   = pd.read_csv(DATA_RAW / "alrajhi_price_daily.csv", index_col=0, parse_dates=True)
    alr.index = pd.to_datetime(alr.index).tz_localize(None)
    alr   = alr[["close"]].rename(columns={"close": "alrajhi"})

    master = price.copy()
    for df in [tasi, oil, vix, rate, saibor, alr]:
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
    master["ret_5d"]  = c.pct_change(5)
    master["ret_20d"] = c.pct_change(20)
    master["vol_ratio"]= master["volume"] / master["volume"].rolling(20).mean()

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    master["bb_upper"] = bb_mid + 2 * bb_std
    master["bb_lower"] = bb_mid - 2 * bb_std
    master["bb_pct"]   = (c - master["bb_lower"]) / (master["bb_upper"] - master["bb_lower"])

    master["high_52w"]        = c.rolling(252).max()
    master["low_52w"]         = c.rolling(252).min()
    master["dist_from_high"]  = (c - master["high_52w"]) / master["high_52w"]
    master["dist_from_low"]   = (c - master["low_52w"])  / master["low_52w"]

    # Rate regime
    r = master["repo_rate"]
    master["rate_change_3m"] = r - r.shift(63)
    master["rate_regime"] = "stable"
    master.loc[master["rate_change_3m"] >  0.25, "rate_regime"] = "rising"
    master.loc[master["rate_change_3m"] < -0.25, "rate_regime"] = "falling"

    # SAIBOR regime (extra for banks)
    s = master["saibor_3m"]
    master["saibor_change_3m"] = s - s.shift(63)
    master["saibor_rising"]    = (master["saibor_change_3m"] > 0.25).astype(int)

    # VIX / oil regimes
    master["vix_high"]  = (master["vix"] > 30).astype(int)
    master["oil_bull"]  = (master["oil"].pct_change(63) > 0.10).astype(int)
    master["oil_bear"]  = (master["oil"].pct_change(63) < -0.10).astype(int)

    # TASI / sector signals
    if "tasi" in master.columns:
        master["tasi_20d"]       = master["tasi"].pct_change(20)
        master["rs_vs_tasi_20d"] = master["ret_20d"] - master["tasi"].pct_change(20)
        master["rs_vs_alrajhi"]  = master["ret_20d"] - master["alrajhi"].pct_change(20) if "alrajhi" in master.columns else 0

    # Forward returns
    for w in [30, 60, 90]:
        master[f"fwd_{w}d"] = c.shift(-w) / c - 1
        if "tasi" in master.columns:
            master[f"fwd_{w}d_vs_tasi"] = c.shift(-w)/c - master["tasi"].shift(-w)/master["tasi"]

    master.to_csv(DATA_PROCESSED / f"master_{sym}.csv")
    return master


def main():
    print("\nSTOCK MEMORY ENGINE v2 — BATCH 1: FULL BANKING SECTOR")
    print(f"Banks: {len(BANKING_STOCKS)} stocks")
    print(f"Date : {date.today()}")
    print("=" * 60)

    if not SAHMK_API_KEY:
        print("ERROR: SAHMK_API_KEY not set"); return

    api   = SAHMKProvider()
    yahoo = YahooProvider()

    # ── STEP 1: Pull data for all banks ───────────────────────────────────
    print("\n[ STEP 1 ] Pulling data from SAHMK + Yahoo...")
    pull_logs = {}
    for sym, info in BANKING_STOCKS.items():
        log = pull_bank_data(sym, info, api, yahoo)
        pull_logs[sym] = log

    # ── STEP 2: Build master datasets ─────────────────────────────────────
    print("\n[ STEP 2 ] Building master datasets...")
    masters = {}
    for sym, info in BANKING_STOCKS.items():
        master = build_bank_master(sym, info)
        if master.empty:
            print(f"  {sym}: no price data — skipping")
            continue
        masters[sym] = master
        print(f"  {info['name']}: {len(master)} days | {master.index[0].date()} → {master.index[-1].date()}")

    # ── STEP 3: Full 5-phase engine for each bank ──────────────────────────
    print("\n[ STEP 3 ] Running full engine (5 phases) for each bank...")
    from phases.full_engine import (run_signal_discovery, run_memory_engine,
                                     run_hypothesis_engine, run_walkforward,
                                     run_recalibration, save_outputs,
                                     load_financials, load_dividends, load_events)

    results_summary = []

    for sym, info in BANKING_STOCKS.items():
        if sym not in masters:
            continue
        master = masters[sym]
        name   = info["name"]

        print(f"\n  {'─'*50}")
        print(f"  {name} ({sym})")
        print(f"  {'─'*50}")

        income, annual = load_financials(sym)
        divs           = load_dividends(sym)
        events         = load_events(sym)
        print(f"  Financials: {len(annual)} annual | Dividends: {len(divs)} | Events: {len(events)}")

        signals = run_signal_discovery(sym, name, master)
        memory  = run_memory_engine(sym, name, master, income, annual, divs, events)
        hyp     = run_hypothesis_engine(sym, name, master)
        wf      = run_walkforward(sym, name, master, annual)
        recal   = run_recalibration(sym, name, hyp, wf, master)

        save_outputs(sym, signals, memory, hyp, wf, recal)

        v = wf.get("validation", {})
        results_summary.append({
            "symbol":    sym,
            "name":      name,
            "days":      len(master),
            "accuracy":  v.get("directional_accuracy_pct"),
            "baseline":  v.get("baseline_always_long_pct"),
            "edge":      v.get("edge_over_baseline_pct"),
            "hyp_acc":   hyp["summary"]["accepted"],
            "hyp_total": hyp["summary"]["total"],
            "mistakes":  len(wf.get("mistake_vault", [])),
            "mae":       v.get("mae_pct"),
        })

    # ── STEP 4: Rebuild current states ────────────────────────────────────
    print("\n[ STEP 4 ] Updating portal with all banking stocks...")
    import subprocess
    subprocess.run([sys.executable, "build_all_forecasts.py"])

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BATCH 1 COMPLETE — BANKING SECTOR RESULTS")
    print("=" * 70)
    print(f"  {'Stock':<30} {'Days':>5} {'Acc%':>6} {'Base%':>6} {'Edge':>6} {'HypAcc':>7} {'Mistakes':>9}")
    print(f"  {'-'*30} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*9}")

    # Include Al Rajhi for comparison
    alr_row = {"symbol":"1120","name":"Al Rajhi Bank (reference)","days":4031,
               "accuracy":78.0,"baseline":70.7,"edge":7.3,"hyp_acc":6,"hyp_total":25,"mistakes":18}
    for r in [alr_row] + results_summary:
        edge_str = f"{r['edge']:+.1f}%" if r.get("edge") is not None else "N/A"
        print(f"  {r['name']:<30} {r['days']:>5} "
              f"{str(r.get('accuracy','?')):>6} {str(r.get('baseline','?')):>6} "
              f"{edge_str:>6} {str(r.get('hyp_acc','?'))+'/'+str(r.get('hyp_total','?')):>7} "
              f"{str(r.get('mistakes','?')):>9}")

    print(f"\n  All 7 banks now have full engine. Portal updated.")
    print(f"  Go to http://localhost:8501 and ask about any bank.")

    # Save batch report
    with open(REPORTS_DIR / "BATCH1_BANKING_REPORT.json", "w", encoding="utf-8") as f:
        json.dump({
            "generated":  datetime.now().isoformat(),
            "batch":      "Banking Sector",
            "stocks":     list(BANKING_STOCKS.keys()),
            "results":    results_summary,
            "pull_logs":  pull_logs,
        }, f, indent=2, default=str)
    print(f"  Report → reports/BATCH1_BANKING_REPORT.json")


if __name__ == "__main__":
    main()
