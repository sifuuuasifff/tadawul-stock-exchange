"""Test what AI context looks like for Jamjoom and flag gaps."""
import sys, json
sys.path.insert(0, ".")
from config.settings import MEMORY_DIR, DATA_RAW
from shared import STOCKS

# Simulate what build_context produces for Jamjoom
sym = "4015"

with open(MEMORY_DIR / "all_current_states.json") as f:
    states = json.load(f)["stocks"]

state = states.get(sym, {})

print("AUDIT: What AI can and cannot answer for Jamjoom (4015)")
print("=" * 65)

checks = {
    "Current price":             state.get("price") is not None,
    "RSI":                       state.get("rsi") is not None,
    "Composite score":           state.get("composite") is not None,
    "30d/60d/90d forecasts":     state.get("forecast_30d") is not None,
    "Rate regime":               state.get("rate_regime") is not None,
    "Env/Tech/Fund/Val scores":  state.get("env_score") is not None,
    "Active signals (env/tech)": bool(state.get("env_signals")),
    "Quarterly income records":  len(state.get("quarterly_records", [])) > 0,
    "TTM net income":            bool(state.get("ttm_summary")),
    "Q1 2026 standalone YoY":    state.get("ttm_summary", {}).get("q1_yoy_pct") is not None,
    "P/E ratio":                 state.get("pe_ratio") is not None,
    "P/B ratio":                 state.get("price_to_book") is not None,
    "Analyst target mean":       state.get("analyst_target_mean") is not None,
    "Analyst upside %":          state.get("analyst_upside_pct") is not None,
    "Num analysts":              state.get("num_analysts") is not None,
    "SAHMK fair value":          state.get("fair_price") is not None,
    "52w high distance":         state.get("dist_from_52w_high_pct") is not None,
    "Dividend events (count)":   False,  # check below
    "Dividend amounts/dates":    False,  # check below
    "Balance sheet (assets)":    False,  # check below
    "Debt level / D/E ratio":    False,  # check below
    "Cash flow quality ratio":   state.get("ttm_summary", {}).get("cash_flow_quality") is not None,
    "ROE / ROA ratios":          bool(state.get("ttm_summary", {}).get("roe_pct")),
}

# Check dividend data
div_path = DATA_RAW / "stock_4015" / "dividends.json"
if div_path.exists():
    with open(div_path) as f:
        div_data = json.load(f)
    hist = div_data.get("history", [])
    checks["Dividend events (count)"] = len(hist) > 0
    checks["Dividend amounts/dates"]  = len(hist) > 0

# Check balance sheet
qr = state.get("quarterly_records", [])
# Currently only income items in quarterly_records
# Check if balance sheet is embedded
checks["Balance sheet (assets)"] = any(r.get("total_assets_mn") for r in qr)
checks["Debt level / D/E ratio"] = any(r.get("debt_mn") for r in qr)

# Check ratios file for ROE/ROA
ratios_path = DATA_RAW / "stock_4015" / "ratios.json"
if ratios_path.exists():
    with open(ratios_path) as f:
        rat = json.load(f)
    periods = rat.get("ratios", [])
    if periods:
        p = periods[0].get("ratios", {})
        checks["ROE / ROA ratios"] = bool(p.get("roe"))

# Cash flow quality
ttm = state.get("ttm_summary", {})
checks["Cash flow quality ratio"] = False  # not in current TTM summary

for item, ok in checks.items():
    icon = "✅" if ok else "❌"
    print(f"  {icon} {item}")

missing = [k for k, v in checks.items() if not v]
print(f"\nMISSING ({len(missing)}):")
for m in missing:
    print(f"  ❌ {m}")
