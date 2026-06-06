"""Verify the TTM fix is correct."""
import sys; sys.path.insert(0, ".")
from rebuild_engine import load_enriched_financials
import pandas as pd

print("VERIFYING TTM FIX — AL RAJHI (1120)")
print("="*55)

_, _, _, _, annual_df = load_enriched_financials("1120")

print(f"\nAnnual records used by model (last 5):")
cols = ["report_date","net_income_bn","ni_yoy","_is_ttm"]
avail = [c for c in cols if c in annual_df.columns]
print(annual_df[avail].tail(5).to_string(index=False))

latest = annual_df.iloc[-1]
print(f"\nLatest entry:")
print(f"  Date:          {latest['report_date'].date()}")
print(f"  Net income:    {latest.get('net_income_bn', 0):.2f} bn SAR")
print(f"  Is TTM:        {latest.get('_is_ttm', False)}")
print(f"  NI YoY:        {latest.get('ni_yoy', 0):.1f}%")
if latest.get('_q_yoy'):
    print(f"  Q1 standalone YoY: {latest.get('_q_yoy', 0):.1f}%")

print(f"\nExpected correct TTM: ~26.3 bn (not 44 bn)")
print(f"{'✅ FIXED' if latest.get('net_income_bn', 0) < 35 else '❌ STILL WRONG'}")
