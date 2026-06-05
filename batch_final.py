"""
FINAL BATCHES — 8, 9, 10
Batch 8: Food & Beverages
Batch 9: Cement & Building Materials
Batch 10: Transport, Utilities & Industrial

Runs all three in sequence, pushes once at the end.

Run:  python batch_final.py
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

# ── Stock lists ────────────────────────────────────────────────────────────────

BATCH8_FOOD = {
    "2270": {"name": "Savola Group",            "yahoo": "2270.SR", "sector": "Food & Beverages"},
    "2050": {"name": "Saudi Arabia Fertilizers","yahoo": "2050.SR", "sector": "Food & Beverages"},
    "6020": {"name": "Halwani Brothers",         "yahoo": "6020.SR", "sector": "Food & Beverages"},
    "2100": {"name": "Wafrah for Industry",      "yahoo": "2100.SR", "sector": "Food & Beverages"},
    "6010": {"name": "NADEC",                    "yahoo": "6010.SR", "sector": "Food & Beverages"},
    "2200": {"name": "Arabian Food Industries",  "yahoo": "2200.SR", "sector": "Food & Beverages"},
}

BATCH9_CEMENT = {
    "3010": {"name": "Yamama Cement",            "yahoo": "3010.SR", "sector": "Cement"},
    "3020": {"name": "Saudi Cement",             "yahoo": "3020.SR", "sector": "Cement"},
    "3030": {"name": "Qassim Cement",            "yahoo": "3030.SR", "sector": "Cement"},
    "3040": {"name": "Southern Province Cement", "yahoo": "3040.SR", "sector": "Cement"},
    "3050": {"name": "Yanbu Cement",             "yahoo": "3050.SR", "sector": "Cement"},
    "3060": {"name": "City Cement",              "yahoo": "3060.SR", "sector": "Cement"},
    "3080": {"name": "Tabuk Cement",             "yahoo": "3080.SR", "sector": "Cement"},
    "3090": {"name": "Arabian Cement",           "yahoo": "3090.SR", "sector": "Cement"},
}

BATCH10_OTHER = {
    "4110": {"name": "Arriyadh Dev Auth",        "yahoo": "4110.SR", "sector": "Diversified"},
    "2180": {"name": "Fitaihi Holding",          "yahoo": "2180.SR", "sector": "Diversified"},
    "4140": {"name": "Saudi Ground Services",    "yahoo": "4140.SR", "sector": "Transport"},
    "4030": {"name": "Saudi Airlines Catering",  "yahoo": "4030.SR", "sector": "Transport"},
    "2080": {"name": "Saudi Electricity",        "yahoo": "2080.SR", "sector": "Utilities"},
    "5110": {"name": "Saudi Telecom Infra",      "yahoo": "5110.SR", "sector": "Utilities"},
    "1830": {"name": "Leejam Sports",            "yahoo": "1830.SR", "sector": "Consumer Services"},
    "4280": {"name": "Tabreed",                  "yahoo": "4280.SR", "sector": "Utilities"},
}

ALL_BATCHES = {
    "Batch 8 — Food & Beverages":        BATCH8_FOOD,
    "Batch 9 — Cement":                  BATCH9_CEMENT,
    "Batch 10 — Transport & Utilities":  BATCH10_OTHER,
}

SECTOR_EMOJI = {
    "Food & Beverages": "🥛",
    "Cement":           "🏗️",
    "Transport":        "✈️",
    "Utilities":        "⚡",
    "Diversified":      "📦",
    "Consumer Services":"🏋️",
}


def update_registries(new_stocks: dict):
    for filepath, marker, tmpl_portal, tmpl_forecast in [
        ("portal.py",
         '    "7010": {"name": "STC", "sector": "Telecom", "emoji": "📡"},',
         '    "{sym}": {{"name": "{name}", "sector": "{sector}", "emoji": "{emoji}"}},',
         None),
        ("build_all_forecasts.py",
         '    "7010": {"name": "STC", "yahoo": "7010.SR", "sector": "Telecom"},',
         None,
         '    "{sym}": {{"name": "{name}", "yahoo": "{yahoo}", "sector": "{sector}"}},'),
    ]:
        path    = Path(filepath)
        content = path.read_text(encoding="utf-8")
        new_lines = ""
        tmpl = tmpl_portal if tmpl_portal else tmpl_forecast
        for sym, info in new_stocks.items():
            if f'"{sym}"' not in content:
                emoji = SECTOR_EMOJI.get(info["sector"], "📊")
                new_lines += tmpl.format(
                    sym=sym, name=info["name"],
                    sector=info["sector"],
                    yahoo=info.get("yahoo", f"{sym}.SR"),
                    emoji=emoji,
                ) + "\n"
        if new_lines:
            content = content.replace(marker, marker + "\n" + new_lines)
            path.write_text(content, encoding="utf-8")


def run_batch(batch_name: str, stocks: dict,
              api: SAHMKProvider, yahoo: YahooProvider) -> list:
    print(f"\n{'='*65}")
    print(f"  {batch_name.upper()}")
    print(f"{'='*65}")

    # Pull data
    print(f"\n  Pulling data for {len(stocks)} stocks...")
    for sym, info in stocks.items():
        pull_bank_data(sym, info, api, yahoo)

    # Build masters
    print(f"\n  Building master datasets...")
    masters = {}
    for sym, info in stocks.items():
        master = build_bank_master(sym, info)
        if master.empty:
            print(f"    {sym}: no data — skipping")
            continue
        masters[sym] = master
        print(f"    {info['name']}: {len(master)} days | "
              f"{master.index[0].date()} → {master.index[-1].date()}")

    # Full engine
    print(f"\n  Running full engine...")
    results = []
    for sym, info in stocks.items():
        if sym not in masters:
            continue
        master = masters[sym]
        name   = info["name"]
        print(f"\n    ── {name} ({sym}) ──")

        income, annual = load_financials(sym)
        divs           = load_dividends(sym)
        events         = load_events(sym)
        print(f"    Financials: {len(annual)} annual | Divs: {len(divs)} | Events: {len(events)}")

        signals = run_signal_discovery(sym, name, master)
        memory  = run_memory_engine(sym, name, master, income, annual, divs, events)
        hyp     = run_hypothesis_engine(sym, name, master)
        wf      = run_walkforward(sym, name, master, annual)
        recal   = run_recalibration(sym, name, hyp, wf, master)
        save_outputs(sym, signals, memory, hyp, wf, recal)

        v = wf.get("validation", {})
        r = {
            "symbol":   sym, "name": name, "sector": info["sector"],
            "days":     len(master),
            "accuracy": v.get("directional_accuracy_pct"),
            "baseline": v.get("baseline_always_long_pct"),
            "edge":     v.get("edge_over_baseline_pct"),
            "hyp_acc":  hyp["summary"]["accepted"],
            "mistakes": len(wf.get("mistake_vault", [])),
        }
        results.append(r)
        edge_str = f"{r['edge']:+.1f}%" if r["edge"] is not None else "N/A"
        print(f"    DONE: acc={r['accuracy']}% baseline={r['baseline']}% edge={edge_str} hyp={r['hyp_acc']}/20")

    return results


def main():
    print("\nSTOCK MEMORY ENGINE v2 — FINAL BATCHES (8, 9, 10)")
    print(f"Total stocks: {sum(len(v) for v in ALL_BATCHES.values())}")
    print(f"Date: {date.today()}")

    if not SAHMK_API_KEY:
        print("ERROR: SAHMK_API_KEY not set"); return

    api   = SAHMKProvider()
    yahoo = YahooProvider()

    all_results = {}
    all_stocks_combined = {}

    for batch_name, stocks in ALL_BATCHES.items():
        results = run_batch(batch_name, stocks, api, yahoo)
        all_results[batch_name] = results
        all_stocks_combined.update(stocks)

    # Update registries once for all new stocks
    print("\n\nUpdating portal registries...")
    update_registries(all_stocks_combined)

    # Save batch reports
    for batch_name, results in all_results.items():
        safe_name = batch_name.split("—")[1].strip().replace(" ", "_").replace("&", "and")
        with open(REPORTS_DIR / f"FINAL_{safe_name}_REPORT.json", "w", encoding="utf-8") as f:
            json.dump({"generated": datetime.now().isoformat(),
                       "batch": batch_name, "results": results},
                      f, indent=2, default=str)

    # Single push at the end
    total_added = sum(len(r) for r in all_results.values())
    print(f"\nPushing all {total_added} new stocks to live portal...")
    subprocess.run([sys.executable, "update_and_push.py",
                    f"Final batches 8-10: Food, Cement, Transport — {total_added} stocks added"])

    # ── FULL SUMMARY ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL BATCHES COMPLETE — FULL SUMMARY")
    print("=" * 70)

    for batch_name, results in all_results.items():
        print(f"\n  {batch_name}")
        print(f"  {'─'*60}")
        print(f"  {'Stock':<28} {'Days':>5} {'Acc%':>6} {'Base%':>6} {'Edge':>7} {'Hyp':>5}")
        print(f"  {'-'*28} {'-'*5} {'-'*6} {'-'*6} {'-'*7} {'-'*5}")
        for r in results:
            edge_str = f"{r['edge']:+.1f}%" if r.get("edge") is not None else "N/A"
            print(f"  {r['name']:<28} {r['days']:>5} "
                  f"{str(r.get('accuracy','?')):>6} {str(r.get('baseline','?')):>6} "
                  f"{edge_str:>7} {str(r.get('hyp_acc','?'))+'/20':>5}")

    total_engine = 52 + total_added
    print(f"\n{'='*70}")
    print(f"  ENGINE COMPLETE")
    print(f"  Total stocks in engine: {total_engine}")
    print(f"  Portal: https://tadawul-stock-exchange.streamlit.app/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
