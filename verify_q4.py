"""Verify that Q4 = full year and measure the TTM error."""
import sys; sys.path.insert(0, ".")
import json
import pandas as pd
from config.settings import DATA_RAW

with open(DATA_RAW / "alrajhi_financials_quarterly_raw.json") as f:
    data = json.load(f)

income = pd.DataFrame(data["income_statements"])
income["report_date"] = pd.to_datetime(income["report_date"])
income = income.sort_values("report_date")
income["ni_bn"] = income["net_income"] / 1e9

print("AL RAJHI — QUARTERLY NET INCOME (SAR bn)")
print("="*65)
print(f"{'Date':<14} {'Q':>3} {'NI (bn)':>10} {'Note'}")
print("-"*65)
for _, r in income[income["report_date"].dt.year >= 2023].iterrows():
    q = int(r.get("fiscal_quarter", 0)) if r.get("fiscal_quarter") else "?"
    note = "← FULL YEAR (not Q4 standalone)" if r["report_date"].month == 12 else ""
    print(f"  {str(r['report_date'].date()):<12} Q{q:>1} {r['ni_bn']:>10.2f}  {note}")

print("\nPROBLEM — Our current wrong TTM calculation:")
last_4 = income.dropna(subset=["net_income"]).tail(4)["net_income"].sum() / 1e9
print(f"  Last 4 records summed = {last_4:.2f} bn  ← WRONG (Q4 double-counts Q1+Q2+Q3)")

print("\nCORRECT TTM calculation:")
annual_2025 = income[(income["report_date"].dt.year==2025) & (income["report_date"].dt.month==12)]["net_income"].iloc[0] / 1e9
q1_2025     = income[(income["report_date"].dt.year==2025) & (income["report_date"].dt.month==3)]["net_income"].iloc[0] / 1e9
q1_2026     = income[(income["report_date"].dt.year==2026) & (income["report_date"].dt.month==3)]["net_income"].iloc[0] / 1e9
ttm_correct = annual_2025 - q1_2025 + q1_2026
print(f"  Full Year 2025: {annual_2025:.2f} bn")
print(f"  Minus Q1 2025:  {q1_2025:.2f} bn")
print(f"  Plus  Q1 2026:  {q1_2026:.2f} bn")
print(f"  = TTM:          {ttm_correct:.2f} bn  ← CORRECT")
print(f"\nYoY growth (correct): {(ttm_correct/annual_2025 - 1)*100:+.1f}%")
print(f"YoY Q1 standalone:    {(q1_2026/q1_2025 - 1)*100:+.1f}%  (Q1 2026 vs Q1 2025)")
print(f"\nOverstatement from bug: {last_4/ttm_correct:.1f}x  ({last_4:.2f} vs {ttm_correct:.2f})")
