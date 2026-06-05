"""
BATCH 7 — HEALTHCARE SECTOR
Dr Sulaiman Al Habib (4013), Mouwasat (4002), Dallah Healthcare (4004),
Saudi German Health (4345), Al Hammadi (4007), National Medical Care (4005),
Specialized Medical (4006), Saudi Pharma (2070)

Run:  python batch_healthcare.py
"""

import sys, json, subprocess, warnings
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import REPORTS_DIR, SAHMK_API_KEY
from config.providers import SAHMKProvider, YahooProvider
from phases.full_engine import (
    run_signal_discovery, run_memory_engine, run_hypothesis_engine,
    run_walkforward, run_recalibration, save_outputs,
    load_financials, load_dividends, load_events,
)
from batch_banking import pull_bank_data, build_bank_master

BATCH7_STOCKS = {
    "4013": {"name": "Dr Sulaiman Al Habib",    "yahoo": "4013.SR", "sector": "Healthcare"},
    "4002": {"name": "Mouwasat Medical",         "yahoo": "4002.SR", "sector": "Healthcare"},
    "4004": {"name": "Dallah Healthcare",        "yahoo": "4004.SR", "sector": "Healthcare"},
    "4345": {"name": "Saudi German Health",      "yahoo": "4345.SR", "sector": "Healthcare"},
    "4007": {"name": "Al Hammadi",               "yahoo": "4007.SR", "sector": "Healthcare"},
    "4005": {"name": "National Medical Care",    "yahoo": "4005.SR", "sector": "Healthcare"},
    "4006": {"name": "Specialized Medical",      "yahoo": "4006.SR", "sector": "Healthcare"},
    "2070": {"name": "Saudi Pharmaceutical",     "yahoo": "2070.SR", "sector": "Healthcare"},
}


def update_registries():
    for filepath, marker, template in [
        ("portal.py",
         '    "7010": {"name": "STC", "sector": "Telecom", "emoji": "📡"},',
         '    "{sym}": {{"name": "{name}", "sector": "Healthcare", "emoji": "🏥"}},'),
        ("build_all_forecasts.py",
         '    "7010": {"name": "STC", "yahoo": "7010.SR", "sector": "Telecom"},',
         '    "{sym}": {{"name": "{name}", "yahoo": "{yahoo}", "sector": "Healthcare"}},'),
    ]:
        path    = Path(filepath)
        content = path.read_text(encoding="utf-8")
        new_lines = ""
        for sym, info in BATCH7_STOCKS.items():
            if f'"{sym}"' not in content:
                new_lines += template.format(sym=sym, name=info["name"],
                                             yahoo=info.get("yahoo", "")) + "\n"
        if new_lines:
            content = content.replace(marker, marker + "\n    # Healthcare\n" + new_lines)
            path.write_text(content, encoding="utf-8")


def main():
    print("\nSTOCK MEMORY ENGINE v2 — BATCH 7: HEALTHCARE")
    print(f"Stocks: {len(BATCH7_STOCKS)}")
    print(f"Date  : {date.today()}")
    print("=" * 60)

    if not SAHMK_API_KEY:
        print("ERROR: SAHMK_API_KEY not set"); return

    api   = SAHMKProvider()
    yahoo = YahooProvider()

    print("\n[ STEP 1 ] Pulling data...")
    for sym, info in BATCH7_STOCKS.items():
        pull_bank_data(sym, info, api, yahoo)

    print("\n[ STEP 2 ] Building master datasets...")
    masters = {}
    for sym, info in BATCH7_STOCKS.items():
        master = build_bank_master(sym, info)
        if master.empty:
            print(f"  {sym}: no data — skipping")
            continue
        masters[sym] = master
        print(f"  {info['name']}: {len(master)} days | "
              f"{master.index[0].date()} → {master.index[-1].date()}")

    print("\n[ STEP 3 ] Running full engine...")
    results = []
    for sym, info in BATCH7_STOCKS.items():
        if sym not in masters:
            continue
        master = masters[sym]
        name   = info["name"]
        print(f"\n  {'─'*48}\n  {name} ({sym})\n  {'─'*48}")

        income, annual = load_financials(sym)
        divs           = load_dividends(sym)
        events         = load_events(sym)
        print(f"  Financials: {len(annual)} annual | Divs: {len(divs)} | Events: {len(events)}")

        signals = run_signal_discovery(sym, name, master)
        memory  = run_memory_engine(sym, name, master, income, annual, divs, events)
        hyp     = run_hypothesis_engine(sym, name, master)
        wf      = run_walkforward(sym, name, master, annual)
        recal   = run_recalibration(sym, name, hyp, wf, master)
        save_outputs(sym, signals, memory, hyp, wf, recal)

        v = wf.get("validation", {})
        results.append({
            "symbol": sym, "name": name, "days": len(master),
            "accuracy": v.get("directional_accuracy_pct"),
            "baseline": v.get("baseline_always_long_pct"),
            "edge":     v.get("edge_over_baseline_pct"),
            "hyp_acc":  hyp["summary"]["accepted"],
            "mistakes": len(wf.get("mistake_vault", [])),
        })

    print("\n[ STEP 4 ] Updating registries and pushing live...")
    update_registries()
    subprocess.run([sys.executable, "update_and_push.py",
                    f"Batch 7 - Healthcare - {len(results)} stocks added"])

    print("\n" + "=" * 68)
    print("  BATCH 7 COMPLETE — HEALTHCARE")
    print("=" * 68)
    print(f"  {'Stock':<28} {'Days':>5} {'Acc%':>6} {'Base%':>6} {'Edge':>7} {'HypAcc':>7}")
    print(f"  {'-'*28} {'-'*5} {'-'*6} {'-'*6} {'-'*7} {'-'*7}")
    for r in results:
        edge_str = f"{r['edge']:+.1f}%" if r.get("edge") is not None else "N/A"
        print(f"  {r['name']:<28} {r['days']:>5} "
              f"{str(r.get('accuracy','?')):>6} {str(r.get('baseline','?')):>6} "
              f"{edge_str:>7} {str(r.get('hyp_acc','?'))+'/20':>7}")

    total_now = 44 + len(results)
    print(f"\n  Portal now has {total_now} stocks — live in ~2 min.")
    print(f"  URL: https://tadawul-stock-exchange.streamlit.app/")

    with open(REPORTS_DIR / "BATCH7_HEALTHCARE_REPORT.json", "w", encoding="utf-8") as f:
        json.dump({"generated": datetime.now().isoformat(),
                   "batch": "Healthcare", "results": results},
                  f, indent=2, default=str)


if __name__ == "__main__":
    main()
