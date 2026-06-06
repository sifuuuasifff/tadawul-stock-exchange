"""
Check fiscal year-end month for every stock.
Some Saudi companies may not have December year-end.
This determines whether our 'month==12 = annual' filter is correct.
"""
import sys, json
sys.path.insert(0, ".")
import pandas as pd
from collections import Counter
from config.settings import DATA_RAW
from shared import STOCKS

print("FISCAL YEAR END ANALYSIS — ALL STOCKS")
print("=" * 75)
print(f"\n  {'Stock':<35} {'Sym':>6}  {'Fiscal YE Month':>16}  {'Largest NI month':>18}  {'Note'}")
print(f"  {'-'*35} {'-'*6}  {'-'*16}  {'-'*18}  {'-'*20}")

anomalies = []

for sym, info in STOCKS.items():
    if sym == "1120":
        fin_path = DATA_RAW / "alrajhi_financials_quarterly_raw.json"
    else:
        fin_path = DATA_RAW / f"stock_{sym}" / "financials_quarterly.json"

    if not fin_path.exists():
        continue

    with open(fin_path, encoding="utf-8") as f:
        data = json.load(f)

    income = pd.DataFrame(data.get("income_statements", []))
    if income.empty or "report_date" not in income.columns:
        continue

    income["report_date"] = pd.to_datetime(income["report_date"])
    income["month"] = income["report_date"].dt.month
    income["net_income_abs"] = income["net_income"].abs()

    # Find which month has the largest net income values
    # (annual/full-year records will have the largest values)
    month_avg_ni = income.groupby("month")["net_income_abs"].mean()
    largest_ni_month = month_avg_ni.idxmax() if not month_avg_ni.empty else None

    # Check fiscal year field if available
    fiscal_info = ""
    if "fiscal_quarter" in income.columns:
        q4_records = income[income["fiscal_quarter"] == 4]
        if not q4_records.empty:
            q4_months = q4_records["month"].value_counts()
            fiscal_ye_month = q4_months.index[0] if not q4_months.empty else None
            fiscal_info = f"Q4={fiscal_ye_month}"

    # Count records per month
    month_counts = income["month"].value_counts().sort_index()
    months_present = [f"{m}({c})" for m, c in month_counts.items()]

    # Flag if year-end is NOT December
    ye_month = largest_ni_month
    note = ""
    if ye_month and ye_month != 12:
        note = f"⚠️ YEAR-END = MONTH {ye_month}!"
        anomalies.append({"sym": sym, "name": info["name"], "ye_month": ye_month})
    elif ye_month == 12:
        note = "✓ Dec year-end"

    months_str = ", ".join(months_present[:6])
    print(f"  {info['name']:<35} {sym:>6}  {str(ye_month):>16}  {months_str:>18}  {note}")

print(f"\n{'='*75}")
print(f"ANOMALIES — Stocks with non-December fiscal year end:")
if anomalies:
    for a in anomalies:
        print(f"  ⚠️  {a['name']} ({a['sym']}) — Year end month: {a['ye_month']}")
else:
    print("  None found — all stocks appear to use December year-end")

print(f"\nCONCLUSION:")
print(f"  If all stocks end in December → our December filter is CORRECT")
print(f"  If any stock ends in another month → we need per-stock fiscal calendar")
