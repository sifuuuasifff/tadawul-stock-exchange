"""Check what quarterly data exists and what the current code is ignoring."""
import sys, json
sys.path.insert(0, ".")
from rebuild_engine import load_enriched_financials
from shared import STOCKS
import pandas as pd

print("QUARTERLY DATA AVAILABILITY CHECK\n")
print(f"{'Stock':<30} {'Latest Q':>12} {'Q1 2026?':>10} {'Annual only?':>13} {'Quarters used':>14}")
print("-" * 82)

for sym in ["4015","1120","2050","3020","7010","4190"]:
    info = STOCKS.get(sym, {})
    income_df, _, _, _, annual_df = load_enriched_financials(sym)

    if income_df.empty:
        print(f"  {info.get('name','?'):<28} {'NO DATA':>12}")
        continue

    latest_q = income_df["report_date"].max().date().isoformat() if not income_df.empty else "N/A"
    has_q1_2026 = any(income_df["report_date"].dt.year == 2026) if not income_df.empty else False
    n_annual = len(annual_df)
    n_quarters = len(income_df)

    # What's the latest record being USED in the model?
    latest_used = annual_df["report_date"].max().date().isoformat() if not annual_df.empty else "N/A"

    flag = " ← MISSING RECENT!" if has_q1_2026 and latest_used < "2026-01-01" else ""
    print(f"  {info.get('name','?'):<28} {latest_q:>12} {'YES' if has_q1_2026 else 'no':>10} "
          f"{n_annual:>13} {n_quarters:>14}{flag}")
    print(f"    Latest Q available: {latest_q}  |  Latest Q actually used: {latest_used}")
