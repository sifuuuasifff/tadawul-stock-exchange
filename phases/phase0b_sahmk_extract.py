"""
PHASE 0b — SAHMK BULK EXTRACTOR
================================
Run this ONCE after adding your SAHMK API key to config/settings.py.
Pulls everything we need in a single session.
Saves all raw data to data/raw/ as CSV files.

Run:  python phases/phase0b_sahmk_extract.py
"""

import sys
import json
import time
import traceback
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import (
    PRIMARY_STOCK, PEER_BANKS, DATA_RAW, REPORTS_DIR, SAHMK_API_KEY,
)
from config.providers import SAHMKProvider

DATA_RAW.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_report_line(name, status, rows=None, coverage=None, error=None):
    return {
        "dataset": name,
        "status":  status,
        "rows":    rows,
        "coverage":coverage,
        "error":   error,
        "timestamp": datetime.now().isoformat(),
    }


def safe_extract(fn, name, log):
    """Wrap any extraction step with error capture."""
    try:
        result = fn()
        print(f"  ✓ {name}")
        return result
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        log.append(extract_report_line(name, "FAILED", error=str(e)))
        return None


def main():
    if not SAHMK_API_KEY:
        print("\nERROR: SAHMK_API_KEY is not set.")
        print("Open config/settings.py and paste your key into SAHMK_API_KEY = '...'")
        print("Then run this script again.")
        return

    print("\nSTOCK MEMORY ENGINE v2 — PHASE 0b: SAHMK BULK EXTRACT")
    print(f"Stock  : {PRIMARY_STOCK['name']} ({PRIMARY_STOCK['symbol']})")
    print(f"Date   : {date.today()}")
    print(f"Key    : {SAHMK_API_KEY[:12]}...{SAHMK_API_KEY[-4:]}")
    print("─" * 60)

    api = SAHMKProvider()
    log = []
    sym = PRIMARY_STOCK["symbol"]

    # ── 1. Al Rajhi daily price (SAHMK — official, adjusted) ──────────────
    print("\n[1] Al Rajhi daily price (2010 → today)...")
    def get_price():
        df = api.get_historical_price(sym, from_date="2010-01-01", interval="1d")
        df.to_csv(DATA_RAW / "alrajhi_price_sahmk.csv")
        log.append(extract_report_line("Al Rajhi Daily Price (SAHMK)", "OK",
                                       rows=len(df), coverage=f"{df.index[0].date()} → {df.index[-1].date()}"))
        return df
    price_df = safe_extract(get_price, "Al Rajhi Daily Price", log)
    time.sleep(0.5)

    # ── 2. TASI index daily ────────────────────────────────────────────────
    print("\n[2] TASI index daily (2010 → today)...")
    def get_tasi():
        df = api.get_tasi_historical(from_date="2010-01-01")
        df.to_csv(DATA_RAW / "tasi_sahmk.csv")
        log.append(extract_report_line("TASI Daily (SAHMK)", "OK",
                                       rows=len(df), coverage=f"{df.index[0].date()} → {df.index[-1].date()}"))
        return df
    safe_extract(get_tasi, "TASI Daily", log)
    time.sleep(0.5)

    # ── 3. Quarterly financial statements ─────────────────────────────────
    print("\n[3] Al Rajhi quarterly financials (max history)...")
    def get_quarterly():
        data = api.get_financials(sym, period="quarterly", history="max", stmt_type="all")
        with open(DATA_RAW / "alrajhi_financials_quarterly_raw.json", "w") as f:
            json.dump(data, f, indent=2, default=str)

        # Parse income statements into a clean DataFrame
        income = data.get("income_statements", [])
        df_income = pd.DataFrame(income)
        if not df_income.empty and "report_date" in df_income.columns:
            df_income["report_date"] = pd.to_datetime(df_income["report_date"])
            df_income = df_income.sort_values("report_date")
            df_income.to_csv(DATA_RAW / "alrajhi_income_quarterly.csv", index=False)

        # Parse balance sheets
        balance = data.get("balance_sheets", [])
        df_balance = pd.DataFrame(balance)
        if not df_balance.empty and "report_date" in df_balance.columns:
            df_balance["report_date"] = pd.to_datetime(df_balance["report_date"])
            df_balance = df_balance.sort_values("report_date")
            df_balance.to_csv(DATA_RAW / "alrajhi_balance_quarterly.csv", index=False)

        # Parse cash flow
        cashflow = data.get("cash_flows", [])
        df_cf = pd.DataFrame(cashflow)
        if not df_cf.empty and "report_date" in df_cf.columns:
            df_cf["report_date"] = pd.to_datetime(df_cf["report_date"])
            df_cf.to_csv(DATA_RAW / "alrajhi_cashflow_quarterly.csv", index=False)

        n = len(df_income)
        log.append(extract_report_line("Al Rajhi Quarterly Financials", "OK", rows=n,
                                       coverage=f"{df_income['report_date'].min().date() if n > 0 else 'N/A'} → {df_income['report_date'].max().date() if n > 0 else 'N/A'}"))
        return data
    safe_extract(get_quarterly, "Quarterly Financials", log)
    time.sleep(0.5)

    # ── 4. Annual financial statements ────────────────────────────────────
    print("\n[4] Al Rajhi annual financials (max history)...")
    def get_annual():
        data = api.get_financials(sym, period="annual", history="max", stmt_type="all")
        with open(DATA_RAW / "alrajhi_financials_annual_raw.json", "w") as f:
            json.dump(data, f, indent=2, default=str)
        income = data.get("income_statements", [])
        n = len(income)
        log.append(extract_report_line("Al Rajhi Annual Financials", "OK", rows=n))
        return data
    safe_extract(get_annual, "Annual Financials", log)
    time.sleep(0.5)

    # ── 5. Ratios / analytics ──────────────────────────────────────────────
    print("\n[5] Al Rajhi ratios & analytics (quarterly, max)...")
    def get_ratios():
        data = api.get_ratios(sym, period="quarterly", history="max")
        with open(DATA_RAW / "alrajhi_ratios_quarterly.json", "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.append(extract_report_line("Al Rajhi Ratios (Quarterly)", "OK"))
        return data
    safe_extract(get_ratios, "Ratios", log)
    time.sleep(0.5)

    # ── 6. Dividends ───────────────────────────────────────────────────────
    print("\n[6] Al Rajhi dividend history...")
    def get_divs():
        data = api.get_dividends(sym, limit=200)
        with open(DATA_RAW / "alrajhi_dividends_sahmk.json", "w") as f:
            json.dump(data, f, indent=2, default=str)
        hist = data.get("history", [])
        df = pd.DataFrame(hist)
        if not df.empty:
            df.to_csv(DATA_RAW / "alrajhi_dividends_sahmk.csv", index=False)
        log.append(extract_report_line("Al Rajhi Dividends (SAHMK)", "OK", rows=len(hist)))
        return data
    safe_extract(get_divs, "Dividends", log)
    time.sleep(0.5)

    # ── 7. Events (earnings announcements, corporate actions) ──────────────
    print("\n[7] Al Rajhi events (earnings, dividends, corporate actions)...")
    def get_events():
        data = api.get_events(sym, limit=500)
        with open(DATA_RAW / "alrajhi_events_sahmk.json", "w") as f:
            json.dump(data, f, indent=2, default=str)
        events = data.get("results", data.get("data", []))
        df = pd.DataFrame(events)
        if not df.empty:
            df.to_csv(DATA_RAW / "alrajhi_events_sahmk.csv", index=False)
        log.append(extract_report_line("Al Rajhi Events (SAHMK)", "OK", rows=len(events)))
        return data
    safe_extract(get_events, "Events", log)
    time.sleep(0.5)

    # ── 8. Company profile ─────────────────────────────────────────────────
    print("\n[8] Al Rajhi company profile...")
    def get_profile():
        data = api.get_company(sym)
        with open(DATA_RAW / "alrajhi_company_profile.json", "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.append(extract_report_line("Al Rajhi Company Profile", "OK"))
        return data
    safe_extract(get_profile, "Company Profile", log)
    time.sleep(0.5)

    # ── 9. Peer bank data ──────────────────────────────────────────────────
    print("\n[9] Peer bank daily prices...")
    for peer_sym, peer_name in PEER_BANKS.items():
        def get_peer(s=peer_sym, n=peer_name):
            df = api.get_historical_price(s, from_date="2010-01-01", interval="1d")
            df.to_csv(DATA_RAW / f"peer_{s}_price_sahmk.csv")
            log.append(extract_report_line(f"{n} Daily Price", "OK", rows=len(df)))
            return df
        safe_extract(get_peer, f"Peer: {peer_name}", log)
        time.sleep(0.5)

    # ── 10. Peer bank financials ───────────────────────────────────────────
    print("\n[10] Peer bank quarterly financials...")
    for peer_sym, peer_name in PEER_BANKS.items():
        def get_peer_fin(s=peer_sym, n=peer_name):
            data = api.get_financials(s, period="quarterly", history="max", stmt_type="income")
            with open(DATA_RAW / f"peer_{s}_financials_quarterly.json", "w") as f:
                json.dump(data, f, indent=2, default=str)
            log.append(extract_report_line(f"{n} Quarterly Financials", "OK"))
            return data
        safe_extract(get_peer_fin, f"Peer Financials: {peer_name}", log)
        time.sleep(0.5)

    # ── 11. Market sectors ─────────────────────────────────────────────────
    print("\n[11] TASI sector data...")
    def get_sectors():
        data = api.get_sectors("TASI")
        with open(DATA_RAW / "tasi_sectors.json", "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.append(extract_report_line("TASI Sectors", "OK"))
        return data
    safe_extract(get_sectors, "TASI Sectors", log)

    # ── Summary ────────────────────────────────────────────────────────────
    ok     = [x for x in log if x["status"] == "OK"]
    failed = [x for x in log if x["status"] == "FAILED"]

    print("\n" + "═" * 60)
    print("  SAHMK EXTRACTION COMPLETE")
    print("═" * 60)
    print(f"  Succeeded : {len(ok)}")
    print(f"  Failed    : {len(failed)}")
    if failed:
        print("\n  FAILED DATASETS:")
        for f in failed:
            print(f"    ✗ {f['dataset']}: {f['error']}")
    print("\n  All data saved to: data/raw/")
    print("  Next step: python phases/phase1_signal_discovery.py")

    with open(REPORTS_DIR / "SAHMK_EXTRACTION_LOG.json", "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"\n  Extraction log → reports/SAHMK_EXTRACTION_LOG.json")


if __name__ == "__main__":
    main()
