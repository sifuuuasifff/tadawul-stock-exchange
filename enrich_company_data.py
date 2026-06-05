"""
Pull company-level KPIs from SAHMK /company/ endpoint for all stocks.
Extracts: P/E, P/B, EPS, analyst targets, fair value, revenue growth, beta.
Saves to data/raw/company_kpis.json
Then rebuilds current state with these new signals.
"""
import sys, json, time
sys.path.insert(0, ".")
from datetime import date, datetime
from pathlib import Path
import pandas as pd
from config.settings import DATA_RAW, MEMORY_DIR
from config.providers import SAHMKProvider
from shared import STOCKS

api = SAHMKProvider()
DATA_RAW.mkdir(parents=True, exist_ok=True)

print("Pulling company KPIs from SAHMK /company/ endpoint...")
print(f"Stocks: {len(STOCKS)}\n")

all_kpis = {}

for sym, info in STOCKS.items():
    try:
        data = api.get_company(sym)
        fund = data.get("fundamentals", {}) or {}
        tech = data.get("technicals", {}) or {}
        val  = data.get("valuation", {}) or {}
        anal = data.get("analysts", {}) or {}

        kpis = {
            "symbol":            sym,
            "name":              info["name"],
            "sector":            info["sector"],
            "current_price":     data.get("current_price"),
            # Valuation
            "pe_ratio":          fund.get("pe_ratio"),
            "forward_pe":        fund.get("forward_pe"),
            "eps_ttm":           fund.get("eps_ttm"),
            "basic_eps":         fund.get("basic_eps"),
            "book_value":        fund.get("book_value"),
            "price_to_book":     fund.get("price_to_book"),
            "market_cap":        fund.get("market_cap"),
            "beta":              fund.get("beta"),
            "shares_outstanding":fund.get("shares_outstanding"),
            # 52w range
            "week_52_high":      fund.get("fifty_two_week_high"),
            "week_52_low":       fund.get("fifty_two_week_low"),
            # Technicals (pre-calculated by SAHMK)
            "rsi_14":            tech.get("rsi_14"),
            "macd_line":         tech.get("macd_line"),
            "macd_signal":       tech.get("macd_signal"),
            "macd_histogram":    tech.get("macd_histogram"),
            "day_avg_50":        tech.get("fifty_day_average"),
            # Fair value
            "fair_price":        val.get("fair_price"),
            "fair_price_confidence": val.get("fair_price_confidence"),
            # Analyst consensus
            "analyst_target_mean":   anal.get("target_mean"),
            "analyst_target_median": anal.get("target_median"),
            "analyst_target_high":   anal.get("target_high"),
            "analyst_target_low":    anal.get("target_low"),
            "analyst_consensus":     anal.get("consensus"),
            "num_analysts":          anal.get("num_analysts"),
            "fetched_at":            datetime.now().isoformat(),
        }

        # Derived signals
        price = kpis["current_price"] or 0
        if kpis["analyst_target_mean"] and price > 0:
            kpis["analyst_upside_pct"] = round((kpis["analyst_target_mean"] / price - 1) * 100, 1)
        if kpis["fair_price"] and price > 0:
            kpis["sahmk_fair_upside_pct"] = round((kpis["fair_price"] / price - 1) * 100, 1)

        all_kpis[sym] = kpis

        pe_str   = f"P/E={kpis['pe_ratio']:.1f}" if kpis['pe_ratio'] else "P/E=N/A"
        pb_str   = f"P/B={kpis['price_to_book']:.2f}" if kpis['price_to_book'] else "P/B=N/A"
        an_str   = f"Analyst={kpis.get('analyst_upside_pct',0):+.1f}%" if kpis.get('analyst_upside_pct') else "Analyst=N/A"
        fv_str   = f"FairV={kpis.get('sahmk_fair_upside_pct',0):+.1f}%" if kpis.get('sahmk_fair_upside_pct') else ""
        n_an     = f"({kpis['num_analysts']} analysts)" if kpis['num_analysts'] else ""
        print(f"  {info['name']:<35} {pe_str:<12} {pb_str:<12} {an_str:<14} {n_an} {fv_str}")
        time.sleep(0.25)

    except Exception as e:
        print(f"  {info['name']:<35} ERROR: {e}")
        all_kpis[sym] = {"symbol": sym, "error": str(e)}
        time.sleep(0.25)

# Save
with open(DATA_RAW / "company_kpis.json", "w", encoding="utf-8") as f:
    json.dump({"generated": datetime.now().isoformat(), "stocks": all_kpis}, f,
              indent=2, ensure_ascii=False)

print(f"\nSaved to data/raw/company_kpis.json")

# Summary stats
valid = [v for v in all_kpis.values() if not v.get("error")]
has_analysts = [v for v in valid if v.get("analyst_target_mean")]
has_fv       = [v for v in valid if v.get("fair_price")]
has_pe       = [v for v in valid if v.get("pe_ratio")]

print(f"\nSUMMARY:")
print(f"  Total stocks:          {len(all_kpis)}")
print(f"  Have P/E data:         {len(has_pe)}")
print(f"  Have analyst targets:  {len(has_analysts)}")
print(f"  Have SAHMK fair value: {len(has_fv)}")

# Show top analyst upsides
print(f"\nTOP ANALYST UPSIDE (stocks analysts love most):")
upsides = [(v["name"], v.get("analyst_upside_pct",0), v.get("num_analysts",0))
           for v in valid if v.get("analyst_upside_pct")]
for name, up, n in sorted(upsides, key=lambda x: -x[1])[:10]:
    print(f"  {name:<35} {up:+.1f}%  ({n} analysts)")

print(f"\nSAHMK FAIR VALUE DISCOUNTS (stocks SAHMK model says are cheap/expensive):")
fvs = [(v["name"], v.get("sahmk_fair_upside_pct",0), v.get("fair_price_confidence",0))
       for v in valid if v.get("sahmk_fair_upside_pct")]
for name, up, conf in sorted(fvs, key=lambda x: x[1])[:10]:
    tag = "CHEAP" if up > 0 else "EXPENSIVE"
    print(f"  {name:<35} {up:+.1f}%  conf={conf:.2f}  [{tag}]")
