import sys; sys.path.insert(0, ".")
import json
from config.settings import DATA_RAW, MEMORY_DIR
from rebuild_engine import load_enriched_financials
import pandas as pd

sym = "4015"
income_df, balance_df, cashflow_df, ratios_df, annual_df = load_enriched_financials(sym)

print("=== CASH FLOW ===")
print(f"cashflow_df shape: {cashflow_df.shape}")
if not cashflow_df.empty:
    print(cashflow_df[["report_date","operating_cash_flow","free_cash_flow"]].tail(5).to_string())

print("\n=== ANNUAL DF columns ===")
print(list(annual_df.columns))

print("\n=== RATIOS DF ===")
print(f"ratios_df shape: {ratios_df.shape}")
if not ratios_df.empty:
    print(ratios_df.columns.tolist())
    print(ratios_df.tail(3).to_string())

# Check ratios.json directly
with open(DATA_RAW / "stock_4015" / "ratios.json") as f:
    rat = json.load(f)
periods = rat.get("ratios", [])
print(f"\nRatios periods: {len(periods)}")
if periods:
    print("First period keys:", list(periods[0].keys()))
    print("Ratios in first period:", periods[0].get("ratios", {}))
