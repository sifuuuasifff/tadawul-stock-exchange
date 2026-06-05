"""
BATCH 4 — TELECOM SECTOR
Mobily (7020), Zain Saudi (7030), Etihad Atheeb (7040), Solutions by STC (7200)

Run:  python batch_telecom.py
"""

import sys, json, subprocess, warnings
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

BATCH4_STOCKS = {
    "7020": {"name": "Mobily",              "yahoo": "7020.SR", "sector": "Telecom"},
    "7030": {"name": "Zain Saudi",          "yahoo": "7030.SR", "sector": "Telecom"},
    "7040": {"name": "Etihad Atheeb",       "yahoo": "7040.SR", "sector": "Telecom"},
    "7200": {"name": "Solutions by STC",    "yahoo": "7200.SR", "sector": "Telecom"},
}

PORTAL_EMOJI = "📡"


def update_registries():
    for filepath, marker, template in [
        ("portal.py",
         '    "7010": {"name": "STC", "sector": "Telecom", "emoji": "📡"},',
         '    "{sym}": {{"name": "{name}", "sector": "Telecom", "emoji": "📡"}},'),
        ("build_all_forecasts.py",
         '    "7010": {"name": "STC", "yahoo": "7010.SR", "sector": "Telecom"},',
         '    "{sym}": {{"name": "{name}", "yahoo": "{yahoo}", "sector": "Telecom"}},'),
    ]:
        path    = Path(filepath)
        content = path.read_text(encoding="utf-8")
        new_lines = ""
        for sym, info in BATCH4_STOCKS.items():
            if f'"{sym}"' not in content:
                new_lines += template.format(sym=sym, name=info["name"], yahoo=info.get("yahoo","")) + "\n"
        if new_lines:
            content = content.replace(marker, marker + "\n" + new_lines)
            path.write_text(content, encoding="utf-8")


def main():
    print("\nSTOCK MEMORY ENGINE v2 — BATCH 4: TELECOM")
    print(f"Stocks: {', '.join(v['name'] for v in BATCH4_STOCKS.values())}")
    print(f"Date  : {date.today()}")
    print("=" * 60)

    if not SAHMK_API_KEY:
        print("ERROR: SAHMK_API_KEY not set"); return

    api   = SAHMKProvider()
    yahoo = YahooProvider()

    print("\n[ STEP 1 ] Pulling data...")
    for sym, info in BATCH4_STOCKS.items():
        pull_bank_data(sym, info, api, yahoo)

    print("\n[ STEP 2 ] Building master datasets...")
    masters = {}
    for sym, info in BATCH4_STOCKS.items():
        master = build_bank_master(sym, info)
        if master.empty:
            print(f"  {sym}: no data — skipping")
            continue
        masters[sym] = master
        print(f"  {info['name']}: {len(master)} days | {master.index[0].date()} → {master.index[-1].date()}")

    print("\n[ STEP 3 ] Running full engine...")
    results = []
    for sym, info in BATCH4_STOCKS.items():
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
            "symbol": sym, "name": name,
            "days":     len(master),
            "accuracy": v.get("directional_accuracy_pct"),
            "baseline": v.get("baseline_always_long_pct"),
            "edge":     v.get("edge_over_baseline_pct"),
            "hyp_acc":  hyp["summary"]["accepted"],
            "mistakes": len(wf.get("mistake_vault", [])),
        })

    print("\n[ STEP 4 ] Updating portal and pushing live...")
    update_registries()
    subprocess.run([sys.executable, "update_and_push.py", "Batch 4 - Telecom sector added"])

    print("\n" + "=" * 65)
    print("  BATCH 4 COMPLETE — TELECOM")
    print("=" * 65)
    print(f"  {'Stock':<25} {'Days':>5} {'Acc%':>6} {'Base%':>6} {'Edge':>7} {'HypAcc':>7}")
    print(f"  {'-'*25} {'-'*5} {'-'*6} {'-'*6} {'-'*7} {'-'*7}")
    for r in results:
        edge_str = f"{r['edge']:+.1f}%" if r.get("edge") is not None else "N/A"
        print(f"  {r['name']:<25} {r['days']:>5} "
              f"{str(r.get('accuracy','?')):>6} {str(r.get('baseline','?')):>6} "
              f"{edge_str:>7} {str(r.get('hyp_acc','?'))+'/20':>7}")

    print(f"\n  Portal now has {24 + len(results)} stocks — live in ~2 min.")
    print(f"  URL: https://tadawul-stock-exchange.streamlit.app/")

    with open(REPORTS_DIR / "BATCH4_TELECOM_REPORT.json", "w", encoding="utf-8") as f:
        json.dump({"generated": datetime.now().isoformat(),
                   "batch": "Telecom", "results": results}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
