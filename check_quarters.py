"""Show exactly what quarterly data we have for every stock."""
import sys, json
sys.path.insert(0, ".")
import pandas as pd
from config.settings import DATA_RAW
from shared import STOCKS

def get_quarterly_data(sym):
    if sym == "1120":
        fin_path = DATA_RAW / "alrajhi_financials_quarterly_raw.json"
    else:
        fin_path = DATA_RAW / f"stock_{sym}" / "financials_quarterly.json"

    if not fin_path.exists():
        return None, None, None

    with open(fin_path, encoding="utf-8") as f:
        data = json.load(f)

    income  = pd.DataFrame(data.get("income_statements", []))
    balance = pd.DataFrame(data.get("balance_sheets", []))
    cashflow= pd.DataFrame(data.get("cash_flows", []))

    for df in [income, balance, cashflow]:
        if not df.empty and "report_date" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date"])

    return income, balance, cashflow

# ── Detailed view for Al Rajhi ─────────────────────────────────────────────
print("=" * 70)
print("AL RAJHI BANK (1120) — DETAILED QUARTERLY DATA")
print("=" * 70)

income, balance, cashflow = get_quarterly_data("1120")

if income is not None and not income.empty:
    income = income.sort_values("report_date")
    print(f"\nINCOME STATEMENT: {len(income)} records")
    print(f"  First: {income['report_date'].min().date()}  Last: {income['report_date'].max().date()}")
    print(f"\n  {'Date':<14} {'Quarter':>8} {'Net Income (SAR bn)':>20} {'Revenue (SAR bn)':>18} {'Type'}")
    print(f"  {'-'*14} {'-'*8} {'-'*20} {'-'*18} {'-'*10}")
    for _, row in income.tail(12).iterrows():
        ni  = row.get("net_income")
        rev = row.get("total_revenue")
        ni_s  = f"{ni/1e9:.2f}" if ni and not pd.isna(ni) else "N/A"
        rev_s = f"{rev/1e9:.2f}" if rev and not pd.isna(rev) else "N/A"
        qtr = f"Q{int(row.get('fiscal_quarter',0))}" if row.get('fiscal_quarter') else "Annual"
        rtype = "QUARTERLY" if row.get("statement_period") == "quarterly" else "ANNUAL"
        print(f"  {str(row['report_date'].date()):<14} {qtr:>8} {ni_s:>20} {rev_s:>18} {rtype}")

if balance is not None and not balance.empty:
    balance = balance.sort_values("report_date")
    print(f"\nBALANCE SHEET: {len(balance)} records")
    print(f"  First: {balance['report_date'].min().date()}  Last: {balance['report_date'].max().date()}")
    print(f"\n  {'Date':<14} {'Total Assets (SAR bn)':>22} {'Equity (SAR bn)':>17} {'Debt (SAR bn)':>15}")
    for _, row in balance.tail(6).iterrows():
        ta = row.get("total_assets")
        eq = row.get("stockholders_equity")
        dt = row.get("total_debt")
        print(f"  {str(row['report_date'].date()):<14} "
              f"{ta/1e9:.0f}" if ta and not pd.isna(ta) else f"  {str(row['report_date'].date()):<14} N/A", end="")
        print(f"  {'':>20}") if ta and not pd.isna(ta) else None

if cashflow is not None and not cashflow.empty:
    cashflow = cashflow.sort_values("report_date")
    print(f"\nCASH FLOW: {len(cashflow)} records")
    print(f"  First: {cashflow['report_date'].min().date()}  Last: {cashflow['report_date'].max().date()}")

# ── Summary for all stocks ─────────────────────────────────────────────────
print("\n\n" + "=" * 80)
print("ALL STOCKS — QUARTERLY DATA SUMMARY")
print("=" * 80)
print(f"\n  {'Stock':<35} {'Symbol':>7} {'Income Qtrs':>12} {'Date Range':>28} {'Latest':>12} {'True Qtrs':>10}")
print(f"  {'-'*35} {'-'*7} {'-'*12} {'-'*28} {'-'*12} {'-'*10}")

for sym, info in STOCKS.items():
    try:
        income, _, _ = get_quarterly_data(sym)
        if income is None or income.empty:
            print(f"  {info['name']:<35} {sym:>7} {'NO DATA':>12}")
            continue

        income = income.sort_values("report_date")
        n_total   = len(income)
        first_date= income["report_date"].min().date()
        last_date = income["report_date"].max().date()

        # Count TRUE quarterly records (not December-only)
        n_true_quarterly = len(income[income["report_date"].dt.month != 12])
        has_2026q1 = any(
            (income["report_date"].dt.year == 2026) & (income["report_date"].dt.month == 3)
        )

        latest_flag = "✓ Q1-2026" if has_2026q1 else "✗ No Q1-2026"
        print(f"  {info['name']:<35} {sym:>7} {n_total:>12} "
              f"{str(first_date)+' → '+str(last_date):>28} {latest_flag:>12} {n_true_quarterly:>10}")
    except Exception as e:
        print(f"  {info['name']:<35} {sym:>7} ERROR: {e}")
