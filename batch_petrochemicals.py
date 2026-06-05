"""
BATCH 3 — PETROCHEMICALS SECTOR
Key Tadawul petrochemical stocks — full 5-phase engine for each.

Run:  python batch_petrochemicals.py
"""

import sys, json, time, warnings, subprocess
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import DATA_RAW, DATA_PROCESSED, MEMORY_DIR, REPORTS_DIR, SAHMK_API_KEY
from config.providers import SAHMKProvider, YahooProvider
from phases.full_engine import (
    run_signal_discovery, run_memory_engine, run_hypothesis_engine,
    run_walkforward, run_recalibration, save_outputs,
    load_financials, load_dividends, load_events,
)
from batch_banking import pull_bank_data, build_bank_master

BATCH3_STOCKS = {
    "2020": {"name": "SABIC Agri-Nutrients",    "yahoo": "2020.SR", "sector": "Petrochemicals"},
    "2290": {"name": "Yanbu National Petro",    "yahoo": "2290.SR", "sector": "Petrochemicals"},
    "2350": {"name": "Saudi Kayan Petrochem",   "yahoo": "2350.SR", "sector": "Petrochemicals"},
    "2310": {"name": "Sipchem",                  "yahoo": "2310.SR", "sector": "Petrochemicals"},
    "2380": {"name": "Petro Rabigh",             "yahoo": "2380.SR", "sector": "Petrochemicals"},
    "2210": {"name": "Nama Chemicals",           "yahoo": "2210.SR", "sector": "Petrochemicals"},
    "2060": {"name": "National Industrialization","yahoo": "2060.SR", "sector": "Petrochemicals"},
    "2030": {"name": "Advanced Petrochem",       "yahoo": "2030.SR", "sector": "Petrochemicals"},
    "2260": {"name": "Sahara International Petrochem","yahoo": "2260.SR","sector":"Petrochemicals"},
    "2250": {"name": "Saudi Industrial Investment","yahoo":"2250.SR", "sector": "Petrochemicals"},
}

PORTAL_STOCKS_ADDITION = {k: {"name": v["name"], "sector": v["sector"], "emoji": "⚗️"}
                           for k, v in BATCH3_STOCKS.items()}


def update_stock_registries(new_stocks: dict):
    """Add new stocks to portal.py and build_all_forecasts.py."""
    for filepath, marker, entry_template in [
        ("portal.py",
         '    "2280": {"name": "Almarai", "sector": "Food & Beverages", "emoji": "🥛"},',
         '    "{sym}": {{"name": "{name}", "sector": "{sector}", "emoji": "⚗️"}},'),
        ("build_all_forecasts.py",
         '    "2280": {"name": "Almarai", "yahoo": "2280.SR", "sector": "Food & Beverages"},',
         '    "{sym}": {{"name": "{name}", "yahoo": "{yahoo}", "sector": "{sector}"}},'),
    ]:
        path    = Path(filepath)
        content = path.read_text(encoding="utf-8")
        new_lines = ""
        for sym, info in new_stocks.items():
            if f'"{sym}"' not in content:
                new_lines += entry_template.format(
                    sym=sym, name=info["name"],
                    sector=info.get("sector","Petrochemicals"),
                    yahoo=info.get("yahoo", f"{sym}.SR")
                ) + "\n"
        if new_lines:
            content = content.replace(marker, marker + "\n    # Petrochemicals\n" + new_lines)
            path.write_text(content, encoding="utf-8")


def main():
    print("\nSTOCK MEMORY ENGINE v2 — BATCH 3: PETROCHEMICALS")
    print(f"Stocks: {len(BATCH3_STOCKS)}")
    print(f"Date  : {date.today()}")
    print("=" * 60)

    if not SAHMK_API_KEY:
        print("ERROR: SAHMK_API_KEY not set"); return

    api   = SAHMKProvider()
    yahoo = YahooProvider()

    print("\n[ STEP 1 ] Pulling data...")
    for sym, info in BATCH3_STOCKS.items():
        pull_bank_data(sym, info, api, yahoo)

    print("\n[ STEP 2 ] Building master datasets...")
    masters = {}
    for sym, info in BATCH3_STOCKS.items():
        master = build_bank_master(sym, info)
        if master.empty:
            print(f"  {sym}: no data — skipping")
            continue
        masters[sym] = master
        print(f"  {info['name']}: {len(master)} days | {master.index[0].date()} → {master.index[-1].date()}")

    print("\n[ STEP 3 ] Running full engine...")
    results = []
    for sym, info in BATCH3_STOCKS.items():
        if sym not in masters:
            continue
        master = masters[sym]
        name   = info["name"]
        print(f"\n  {'─'*48}\n  {name} ({sym})\n  {'─'*48}")

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
        results.append({
            "symbol":   sym, "name": name,
            "days":     len(master),
            "accuracy": v.get("directional_accuracy_pct"),
            "baseline": v.get("baseline_always_long_pct"),
            "edge":     v.get("edge_over_baseline_pct"),
            "hyp_acc":  hyp["summary"]["accepted"],
            "mistakes": len(wf.get("mistake_vault", [])),
        })

    print("\n[ STEP 4 ] Updating registries and portal...")
    update_stock_registries(BATCH3_STOCKS)
    subprocess.run([sys.executable, "build_all_forecasts.py"])

    print("\n" + "=" * 70)
    print("  BATCH 3 COMPLETE — PETROCHEMICALS")
    print("=" * 70)
    print(f"  {'Stock':<28} {'Days':>5} {'Acc%':>6} {'Base%':>6} {'Edge':>7} {'HypAcc':>7}")
    print(f"  {'-'*28} {'-'*5} {'-'*6} {'-'*6} {'-'*7} {'-'*7}")
    for r in results:
        edge_str = f"{r['edge']:+.1f}%" if r.get("edge") is not None else "N/A"
        print(f"  {r['name']:<28} {r['days']:>5} "
              f"{str(r.get('accuracy','?')):>6} {str(r.get('baseline','?')):>6} "
              f"{edge_str:>7} {str(r.get('hyp_acc','?'))+'/20':>7}")

    total = 15 + len(results)
    print(f"\n  Portal now has {total} stocks. Restart to see Petrochemicals sector.")

    with open(REPORTS_DIR / "BATCH3_PETROCHEMICALS_REPORT.json", "w", encoding="utf-8") as f:
        json.dump({"generated": datetime.now().isoformat(),
                   "batch": "Petrochemicals", "results": results}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
