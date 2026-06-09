"""
DAILY PRICE REFRESH
===================
Pulls latest prices from Yahoo Finance for all 74 stocks.
Recalculates RSI, Bollinger Bands, momentum, composite scores.
Updates all_current_states.json and pushes to portal.

Can be run:
  - Manually:           python daily_refresh.py
  - From portal button: triggered via subprocess
  - Automatically:      GitHub Actions every weekday 9am Saudi time

Yahoo uses same Tadawul codes as SAHMK (just adds .SR suffix).
Code 1120 on Yahoo = same Al Rajhi Bank as code 1120 on SAHMK.
No data consistency risk.
"""

import sys, json, subprocess, warnings
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import DATA_PROCESSED, MEMORY_DIR
from shared import STOCKS

MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# Yahoo suffix for Saudi Exchange
YAHOO_SUFFIX = ".SR"

# Short period ensures daily (not weekly) data from Yahoo
LOOKBACK_DAYS = "1mo"


def refresh_stock_prices(sym: str, info: dict) -> bool:
    """
    Pull latest prices for one stock from Yahoo.
    Updates the master_{sym}.csv file with new rows.
    Returns True if successful.
    """
    yahoo_ticker = f"{sym}{YAHOO_SUFFIX}"
    master_path  = DATA_PROCESSED / ("master_dataset.csv" if sym == "1120" else f"master_{sym}.csv")

    if not master_path.exists():
        return False

    try:
        # Pull fresh data from Yahoo
        ticker = yf.Ticker(yahoo_ticker)
        new_data = ticker.history(period=LOOKBACK_DAYS, auto_adjust=True)
        if new_data.empty:
            return False

        new_data.index = pd.to_datetime(new_data.index).tz_convert(None).normalize()
        # Normalise column names — yfinance returns capitalized
        new_data.columns = [c.lower() for c in new_data.columns]
        if "close" not in new_data.columns:
            return False
        new_data = new_data[["close","volume"]].dropna()

        # Load existing master
        master = pd.read_csv(master_path, index_col=0, parse_dates=True)
        master.index = pd.to_datetime(master.index).tz_localize(None)

        # Update existing rows and append any new dates
        for col in ["close", "volume"]:
            if col in new_data.columns and col in master.columns:
                existing = new_data.index[new_data.index.isin(master.index)]
                master.loc[existing, col] = new_data.loc[existing, col]

        new_rows = new_data[~new_data.index.isin(master.index)]
        if not new_rows.empty:
            master = pd.concat([master, new_rows.reindex(columns=master.columns)])

        master = master.sort_index()

        # Recalculate technical indicators on fresh data
        c = master["close"]

        # RSI (14)
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        master["rsi"] = 100 - (100 / (1 + rs))

        # Moving averages
        master["ma20"]  = c.rolling(20).mean()
        master["ma50"]  = c.rolling(50).mean()
        master["ma200"] = c.rolling(200).mean()
        master["above_ma20"]  = (c > master["ma20"]).astype(int)
        master["above_ma50"]  = (c > master["ma50"]).astype(int)
        master["above_ma200"] = (c > master["ma200"]).astype(int)

        # Momentum
        master["ret_1d"]  = c.pct_change(1)
        master["ret_5d"]  = c.pct_change(5)
        master["ret_20d"] = c.pct_change(20)
        master["vol_ratio"]= master["volume"] / master["volume"].rolling(20).mean()

        # Bollinger Bands
        bb_mid = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        master["bb_upper"] = bb_mid + 2 * bb_std
        master["bb_lower"] = bb_mid - 2 * bb_std
        master["bb_pct"]   = (c - master["bb_lower"]) / (master["bb_upper"] - master["bb_lower"])

        # 52-week range
        master["high_52w"]       = c.rolling(252).max()
        master["low_52w"]        = c.rolling(252).min()
        master["dist_from_high"] = (c - master["high_52w"]) / master["high_52w"]

        # Save updated master
        master.to_csv(master_path)
        return True

    except Exception as e:
        print(f"  Warning: {sym} price update failed — {e}")
        return False


def run_daily_refresh(push_to_github: bool = True, verbose: bool = True) -> dict:
    """
    Main refresh function. Returns summary dict.
    push_to_github=False when called from portal button (portal pushes separately).
    """
    start_time = datetime.now()
    results    = {"timestamp": start_time.isoformat(), "updated": [], "failed": [], "skipped": []}

    if verbose:
        print(f"\nDAILY PRICE REFRESH — {date.today()}")
        print(f"Stocks: {len(STOCKS)}")
        print("-" * 50)

    for sym, info in STOCKS.items():
        success = refresh_stock_prices(sym, info)
        if success:
            results["updated"].append(sym)
            if verbose:
                print(f"  ✓ {info['name']}")
        else:
            results["skipped"].append(sym)
            if verbose:
                print(f"  ~ {info['name']} (skipped)")

    if verbose:
        print(f"\nUpdated: {len(results['updated'])} | Skipped: {len(results['skipped'])}")
        print("Recalculating composite scores...")

    # Rebuild composite scores with fresh prices
    subprocess.run([sys.executable, "build_all_forecasts_v3.py"],
                   capture_output=not verbose)

    results["duration_seconds"] = (datetime.now() - start_time).total_seconds()
    results["latest_prices"] = {}

    # Capture latest price for report
    states_path = MEMORY_DIR / "all_current_states.json"
    if states_path.exists():
        with open(states_path) as f:
            states = json.load(f).get("stocks", {})
        for sym in list(STOCKS.keys())[:10]:
            s = states.get(sym, {})
            results["latest_prices"][sym] = {
                "name":      STOCKS[sym]["name"],
                "price":     s.get("price"),
                "rsi":       s.get("rsi"),
                "composite": s.get("composite"),
            }

    # Save refresh log
    log_path = MEMORY_DIR / "last_refresh.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if push_to_github:
        if verbose:
            print("Pushing to portal...")
        subprocess.run([sys.executable, "update_and_push.py",
                        f"Daily refresh {date.today()} — prices updated"])

    if verbose:
        print(f"\nDone in {results['duration_seconds']:.0f}s")
        print(f"Portal: https://tadawul-stock-exchange.streamlit.app/")

    return results


if __name__ == "__main__":
    run_daily_refresh(push_to_github=True, verbose=True)
