"""
BATCH 2 — LARGE CAPS
Saudi Aramco (2222), SABIC (2010), STC (7010), Almarai (2280)

Run:  python batch_largecaps.py
"""

import sys, json, time, warnings, subprocess
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from config.settings import DATA_RAW, DATA_PROCESSED, MEMORY_DIR, REPORTS_DIR, SAHMK_API_KEY
from config.providers import SAHMKProvider, YahooProvider
from phases.full_engine import (
    run_signal_discovery, run_memory_engine, run_hypothesis_engine,
    run_walkforward, run_recalibration, save_outputs,
    load_financials, load_dividends, load_events,
)
from batch_banking import pull_bank_data, build_bank_master   # reuse same data + master builders

BATCH2_STOCKS = {
    "2222": {"name": "Saudi Aramco",  "yahoo": "2222.SR", "sector": "Energy"},
    "2010": {"name": "SABIC",         "yahoo": "2010.SR", "sector": "Petrochemicals"},
    "7010": {"name": "STC",           "yahoo": "7010.SR", "sector": "Telecom"},
    "2280": {"name": "Almarai",       "yahoo": "2280.SR", "sector": "Food & Beverages"},
}


def main():
    print("\nSTOCK MEMORY ENGINE v2 — BATCH 2: LARGE CAPS")
    print(f"Stocks: {', '.join(v['name'] for v in BATCH2_STOCKS.values())}")
    print(f"Date  : {date.today()}")
    print("=" * 60)

    if not SAHMK_API_KEY:
        print("ERROR: SAHMK_API_KEY not set"); return

    api   = SAHMKProvider()
    yahoo = YahooProvider()

    # ── STEP 1: Pull data ─────────────────────────────────────────────────
    print("\n[ STEP 1 ] Pulling data from SAHMK + Yahoo...")
    for sym, info in BATCH2_STOCKS.items():
        pull_bank_data(sym, info, api, yahoo)   # same function works for any stock

    # ── STEP 2: Build master datasets ─────────────────────────────────────
    print("\n[ STEP 2 ] Building master datasets...")
    masters = {}
    for sym, info in BATCH2_STOCKS.items():
        master = build_bank_master(sym, info)
        if master.empty:
            print(f"  {sym}: no price data — skipping")
            continue
        masters[sym] = master
        print(f"  {info['name']}: {len(master)} days | "
              f"{master.index[0].date()} → {master.index[-1].date()}")

    # ── STEP 3: Full 5-phase engine ────────────────────────────────────────
    print("\n[ STEP 3 ] Running full engine (5 phases)...")
    results_summary = []

    for sym, info in BATCH2_STOCKS.items():
        if sym not in masters:
            continue
        master = masters[sym]
        name   = info["name"]

        print(f"\n  {'─'*50}")
        print(f"  {name} ({sym}) — {info['sector']}")
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
            "sector":    info["sector"],
            "days":      len(master),
            "accuracy":  v.get("directional_accuracy_pct"),
            "baseline":  v.get("baseline_always_long_pct"),
            "edge":      v.get("edge_over_baseline_pct"),
            "hyp_acc":   hyp["summary"]["accepted"],
            "hyp_total": hyp["summary"]["total"],
            "mistakes":  len(wf.get("mistake_vault", [])),
        })

    # ── STEP 4: Update portal ──────────────────────────────────────────────
    print("\n[ STEP 4 ] Updating portal...")

    # Add to build_all_forecasts.py stock registry
    forecasts_path = Path("build_all_forecasts.py")
    content = forecasts_path.read_text(encoding="utf-8")
    new_entries = ""
    for sym, info in BATCH2_STOCKS.items():
        line = f'    "{sym}": {{"name": "{info["name"]}", "yahoo": "{info["yahoo"]}", "sector": "{info["sector"]}"}},\n'
        if f'"{sym}"' not in content:
            new_entries += line
    if new_entries:
        content = content.replace(
            '    "4015": {"name": "Jamjoom Pharma",',
            '    "4015": {"name": "Jamjoom Pharma",',
        )
        # Insert before closing brace of STOCKS dict
        content = content.replace(
            '    "4015": {"name": "Jamjoom Pharma",            "yahoo": "4015.SR", "sector": "Pharmaceuticals"},\n}',
            f'    "4015": {{"name": "Jamjoom Pharma",            "yahoo": "4015.SR", "sector": "Pharmaceuticals"}},\n{new_entries}}}'
        )
        forecasts_path.write_text(content, encoding="utf-8")

    # Add to portal.py STOCKS dict
    portal_path = Path("portal.py")
    portal_content = portal_path.read_text(encoding="utf-8")
    portal_new = ""
    sector_emojis = {"Energy": "🛢️", "Petrochemicals": "⚗️", "Telecom": "📡", "Food & Beverages": "🥛"}
    for sym, info in BATCH2_STOCKS.items():
        emoji = sector_emojis.get(info["sector"], "📊")
        line = f'    "{sym}": {{"name": "{info["name"]}", "sector": "{info["sector"]}", "emoji": "{emoji}"}},\n'
        if f'"{sym}"' not in portal_content:
            portal_new += line
    if portal_new:
        portal_content = portal_content.replace(
            '    "4015": {"name": "Jamjoom Pharma",            "sector": "Pharmaceuticals","emoji": "💊"},\n}',
            f'    "4015": {{"name": "Jamjoom Pharma",            "sector": "Pharmaceuticals","emoji": "💊"}},\n    # Large Caps\n{portal_new}}}'
        )
        portal_path.write_text(portal_content, encoding="utf-8")

    subprocess.run([sys.executable, "build_all_forecasts.py"])

    # ── FINAL SUMMARY ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BATCH 2 COMPLETE — LARGE CAPS RESULTS")
    print("=" * 70)
    print(f"  {'Stock':<25} {'Sector':<18} {'Days':>5} {'Acc%':>6} {'Base%':>6} {'Edge':>6} {'HypAcc':>7}")
    print(f"  {'-'*25} {'-'*18} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    for r in results_summary:
        edge_str = f"{r['edge']:+.1f}%" if r.get("edge") is not None else "N/A"
        print(f"  {r['name']:<25} {r['sector']:<18} {r['days']:>5} "
              f"{str(r.get('accuracy','?')):>6} {str(r.get('baseline','?')):>6} "
              f"{edge_str:>6} {str(r.get('hyp_acc','?'))+'/'+str(r.get('hyp_total','?')):>7}")

    with open(REPORTS_DIR / "BATCH2_LARGECAPS_REPORT.json", "w", encoding="utf-8") as f:
        json.dump({"generated": datetime.now().isoformat(),
                   "batch": "Large Caps", "results": results_summary}, f, indent=2, default=str)

    print(f"\n  Portal now has {11 + len(results_summary)} stocks total.")
    print(f"  Restart portal and open http://localhost:8501")
    print(f"  Report → reports/BATCH2_LARGECAPS_REPORT.json")


if __name__ == "__main__":
    main()
