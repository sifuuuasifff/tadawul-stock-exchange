"""
Apply empirically optimized weights to build_all_forecasts_v3.py
Now produces THREE composite scores per stock: 30d, 60d, 90d
Each uses the sector-specific optimal weights for that horizon.
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from datetime import date, datetime
from pathlib import Path
from config.settings import DATA_RAW, DATA_PROCESSED, MEMORY_DIR
from shared import STOCKS
from rebuild_engine import load_enriched_financials, score_fundamental_enriched
from final_complete_rebuild import score_environment_sector, SECTOR_MACRO_WEIGHTS

# Load optimal weights
with open(MEMORY_DIR / "optimal_weights.json") as f:
    OPT = json.load(f)["sectors"]

# Load company KPIs
kpi_path = DATA_RAW / "company_kpis.json"
COMPANY_KPIS = {}
if kpi_path.exists():
    with open(kpi_path, encoding="utf-8") as f:
        COMPANY_KPIS = json.load(f).get("stocks", {})

# Fallback weights when sector has no data
DEFAULT_WEIGHTS = {
    30: {"env": 0.30, "tech": 0.40, "fund": 0.20, "val": 0.10},
    60: {"env": 0.38, "tech": 0.30, "fund": 0.22, "val": 0.10},
    90: {"env": 0.40, "tech": 0.20, "fund": 0.25, "val": 0.15},
}


def get_weights(sector: str, horizon: int) -> dict:
    s = OPT.get(sector, {})
    h = s.get(f"horizon_{horizon}d", {})
    w = h.get("weights")
    if w:
        return w
    return DEFAULT_WEIGHTS[horizon]


def score_valuation_current(sym: str, price: float) -> tuple:
    kpis  = COMPANY_KPIS.get(sym, {})
    score = 50
    sigs  = {}
    analyst_upside = kpis.get("analyst_upside_pct")
    n_analysts     = kpis.get("num_analysts", 0) or 0
    if analyst_upside is not None and n_analysts >= 3:
        if analyst_upside > 30:   score += 15; sigs["analyst"] = f"+{analyst_upside:.1f}% ({n_analysts} analysts) (+15)"
        elif analyst_upside > 15: score += 10; sigs["analyst"] = f"+{analyst_upside:.1f}% (+10)"
        elif analyst_upside > 5:  score += 5;  sigs["analyst"] = f"+{analyst_upside:.1f}% (+5)"
        elif analyst_upside < -10:score -= 10; sigs["analyst"] = f"{analyst_upside:.1f}% (-10)"
        elif analyst_upside < 0:  score -= 5;  sigs["analyst"] = f"{analyst_upside:.1f}% (-5)"
    pe = kpis.get("pe_ratio")
    if pe and 1 < pe < 500:
        if pe < 6:    score += 10; sigs["pe"] = f"P/E={pe:.1f} cheap (+10)"
        elif pe < 10: score += 5;  sigs["pe"] = f"P/E={pe:.1f} fair (+5)"
        elif pe < 18: score += 0;  sigs["pe"] = f"P/E={pe:.1f} neutral"
        elif pe < 25: score -= 5;  sigs["pe"] = f"P/E={pe:.1f} elevated (-5)"
        elif pe >= 25:score -= 10; sigs["pe"] = f"P/E={pe:.1f} expensive (-10)"
    pb = kpis.get("price_to_book")
    if pb and 0 < pb < 100:
        if pb < 0.8:  score += 8;  sigs["pb"] = f"P/B={pb:.2f} below book (+8)"
        elif pb < 1.2:score += 3;  sigs["pb"] = f"P/B={pb:.2f} near book (+3)"
        elif pb > 5:  score -= 5;  sigs["pb"] = f"P/B={pb:.2f} premium (-5)"
    fv = kpis.get("sahmk_fair_upside_pct")
    fc = kpis.get("fair_price_confidence", 0) or 0
    if fv is not None and fc >= 0.8:
        if fv > 15:   score += 8;  sigs["fair"] = f"SAHMK FV {fv:+.1f}% (+8)"
        elif fv > 5:  score += 4;  sigs["fair"] = f"SAHMK FV {fv:+.1f}% (+4)"
        elif fv < -20:score -= 8;  sigs["fair"] = f"SAHMK FV {fv:+.1f}% (-8)"
        elif fv < -10:score -= 4;  sigs["fair"] = f"SAHMK FV {fv:+.1f}% (-4)"
    return max(0, min(100, score)), sigs


print("Building current state with OPTIMIZED sector+horizon weights...")
print(f"Date: {date.today()}\n")

all_states = {}

for sym, info in STOCKS.items():
    path = DATA_PROCESSED / ("master_dataset.csv" if sym == "1120" else f"master_{sym}.csv")
    if not path.exists():
        continue

    master = pd.read_csv(path, index_col=0, parse_dates=True)
    master.index = pd.to_datetime(master.index).tz_localize(None)
    row     = master.iloc[-1]
    price   = float(row["close"])
    sector  = info["sector"]

    # Environment score (sector-aware)
    env, env_sigs = score_environment_sector(row, sector)

    # Technical score
    tech = 50; tech_sigs = {}
    rsi  = float(row.get("rsi", 50)) if not pd.isna(row.get("rsi", float("nan"))) else 50.0
    if rsi < 20:    tech += 20; tech_sigs["rsi"] = f"{rsi:.1f} extreme oversold (+20)"
    elif rsi < 30:  tech += 15; tech_sigs["rsi"] = f"{rsi:.1f} oversold (+15)"
    elif rsi < 40:  tech += 8;  tech_sigs["rsi"] = f"{rsi:.1f} low (+8)"
    elif rsi > 80:  tech -= 20; tech_sigs["rsi"] = f"{rsi:.1f} extreme overbought (-20)"
    elif rsi > 70:  tech -= 12; tech_sigs["rsi"] = f"{rsi:.1f} overbought (-12)"
    else:           tech_sigs["rsi"] = f"{rsi:.1f} neutral"
    bb_pct = float(row.get("bb_pct", 0.5)) if not pd.isna(row.get("bb_pct", float("nan"))) else 0.5
    if bb_pct < 0.05:    tech += 20; tech_sigs["bb"] = "lower band (+20)"
    elif bb_pct < 0.15:  tech += 10; tech_sigs["bb"] = "low BB (+10)"
    elif bb_pct > 0.95:  tech -= 15; tech_sigs["bb"] = "upper band (-15)"
    ret20 = float(row.get("ret_20d", 0)) if not pd.isna(row.get("ret_20d", float("nan"))) else 0.0
    if ret20 > 0.15:    tech -= 20; tech_sigs["mom"] = f"{ret20:.1%} strong rally (-20)"
    elif ret20 > 0.10:  tech -= 10; tech_sigs["mom"] = f"{ret20:.1%} rally (-10)"
    elif ret20 < -0.15: tech += 15; tech_sigs["mom"] = f"{ret20:.1%} drop (+15)"
    elif ret20 < -0.10: tech += 8;  tech_sigs["mom"] = f"{ret20:.1%} drop (+8)"
    tech = max(0, min(100, tech))

    # Fundamental score (with Q1 2026 TTM)
    _, _, _, _, annual_df = load_enriched_financials(sym)
    fund, fund_sigs = score_fundamental_enriched(
        pd.Timestamp(date.today()), annual_df, pd.DataFrame(), pd.DataFrame())

    # Valuation score
    val, val_sigs = score_valuation_current(sym, price)

    # Composite scores — THREE versions, one per horizon, each with optimal weights
    composites = {}
    forecasts  = {}
    rate_r = str(row.get("rate_regime", "stable"))
    repo   = float(row.get("repo_rate", 4.25)) if not pd.isna(row.get("repo_rate", float("nan"))) else 4.25
    vix    = float(row.get("vix", 20)) if not pd.isna(row.get("vix", float("nan"))) else 20.0
    oil    = float(row.get("oil", 80)) if not pd.isna(row.get("oil", float("nan"))) else 80.0

    mem_path = MEMORY_DIR / f"memory_{sym}.json"
    baseline_90d = 6.0
    if mem_path.exists():
        with open(mem_path) as f:
            mem = json.load(f)
        baseline_90d = mem.get("baseline_90d", {}).get("avg_pct", 6.0) or 6.0

    for hz in [30, 60, 90]:
        w = get_weights(sector, hz)
        comp = (env*w["env"] + tech*w["tech"] + fund*w["fund"] + val*w["val"])
        composites[hz] = round(comp, 1)

        # Forecast for this horizon
        if comp > 74:   base = max(baseline_90d * 1.3, 9.5) * (hz/90)
        elif comp > 67: base = max(baseline_90d * 1.1, 7.5) * (hz/90)
        elif comp > 60: base = baseline_90d * (hz/90)
        elif comp > 53: base = baseline_90d * 0.5 * (hz/90)
        elif comp > 46: base = baseline_90d * 0.1 * (hz/90)
        else:           base = -3.5 * (hz/90)
        if rate_r == "rising": base -= 5.0 * (hz/90)

        conf = int(40 + abs(comp-50)/50*35 + (1-abs(env-tech)/100)*25)
        conf = max(20, min(90, conf))
        forecasts[hz] = {
            "base_pct":      round(base, 1),
            "target":        round(price * (1 + base/100), 2),
            "bull_pct":      round(base + 5*(hz/90), 1),
            "bear_pct":      round(base - 5*(hz/90), 1),
            "confidence":    conf,
            "conf_label":    "HIGH" if conf >= 65 else ("MEDIUM" if conf >= 45 else "LOW"),
            "weights_used":  w,
        }

    # Primary composite = 90d (most strategic)
    primary_comp = composites[90]
    primary_fc   = forecasts[90]

    c = master["close"]
    h52 = float(c.rolling(252).max().iloc[-1]) if len(c) >= 252 else float(c.max())
    kpis = COMPANY_KPIS.get(sym, {})

    state = {
        "symbol": sym, "name": info["name"], "sector": sector,
        "as_of":  date.today().isoformat(),
        "price":  round(price, 2),
        "rsi":    round(rsi, 1), "bb_pct": round(bb_pct, 3),
        "ret_20d_pct": round(ret20*100, 1),
        "rate_regime": rate_r, "repo_rate": round(repo, 2),
        "vix": round(vix, 1), "oil_price": round(oil, 1),
        "env_score":  round(env, 1),
        "tech_score": round(tech, 1),
        "fund_score": round(fund, 1),
        "val_score":  round(val, 1),
        "composite":  primary_comp,   # 90d composite for main display
        "composite_30d": composites[30],
        "composite_60d": composites[60],
        "composite_90d": composites[90],
        "sector_weights_used": {
            "30d": get_weights(sector, 30),
            "60d": get_weights(sector, 60),
            "90d": get_weights(sector, 90),
        },
        "env_signals": env_sigs, "tech_signals": tech_sigs,
        "fund_signals": fund_sigs, "val_signals": val_sigs,
        "trading_days": len(master),
        "dist_from_52w_high_pct": round((price-h52)/h52*100, 1),
        "pe_ratio":            kpis.get("pe_ratio"),
        "price_to_book":       kpis.get("price_to_book"),
        "analyst_target_mean": kpis.get("analyst_target_mean"),
        "analyst_upside_pct":  kpis.get("analyst_upside_pct"),
        "num_analysts":        kpis.get("num_analysts"),
        "fair_price":          kpis.get("fair_price"),
        "sahmk_fair_upside_pct": kpis.get("sahmk_fair_upside_pct"),
        # Three horizon forecasts
        "forecast_30d": forecasts[30],
        "forecast_60d": forecasts[60],
        "forecast_90d": forecasts[90],
        # Legacy field for portal compatibility
        "forecast": {
            "base_30d_pct":  forecasts[30]["base_pct"],
            "base_60d_pct":  forecasts[60]["base_pct"],
            "base_90d_pct":  forecasts[90]["base_pct"],
            "target_30d":    forecasts[30]["target"],
            "target_60d":    forecasts[60]["target"],
            "target_90d":    forecasts[90]["target"],
            "bull_90d_pct":  forecasts[90]["bull_pct"],
            "bear_90d_pct":  forecasts[90]["bear_pct"],
            "bull_target_90d": round(price*(1+forecasts[90]["bull_pct"]/100), 2),
            "bear_target_90d": round(price*(1+forecasts[90]["bear_pct"]/100), 2),
            "confidence":    forecasts[90]["confidence"],
            "confidence_label": forecasts[90]["conf_label"],
        },
    }
    # ── Embed quarterly financial records so cloud AI can access them ────────
    try:
        income_df2, _, _, _, annual_df2 = load_enriched_financials(sym)
        quarterly_records = []
        if not income_df2.empty:
            for _, r in income_df2.sort_values("report_date").tail(12).iterrows():
                ni  = r.get("net_income")
                rev = r.get("total_revenue")
                quarterly_records.append({
                    "date":    r["report_date"].date().isoformat(),
                    "quarter": f"Q{int(r.get('fiscal_quarter',0))}" if r.get("fiscal_quarter") else "Annual",
                    "type":    "Full Year" if r["report_date"].month == 12 else "Quarterly",
                    "net_income_mn":  round(ni/1e6, 1)  if ni  and not pd.isna(ni)  else None,
                    "revenue_mn":     round(rev/1e6, 1) if rev and not pd.isna(rev) else None,
                })
        # TTM summary
        ttm_summary = {}
        if not annual_df2.empty:
            lat = annual_df2.iloc[-1]
            ttm_summary = {
                "ttm_ni_mn":       round(lat.get("net_income_bn", 0) * 1000, 1),
                "ttm_yoy_pct":     round(lat.get("ni_yoy", 0), 1),
                "q1_yoy_pct":      round(lat.get("_q_yoy", 0), 1) if lat.get("_q_yoy") else None,
                "latest_q_ni_mn":  round(lat.get("_latest_q_ni", 0)/1e6, 1) if lat.get("_latest_q_ni") else None,
                "is_ttm":          bool(lat.get("_is_ttm", False)),
                "as_of":           lat["report_date"].date().isoformat(),
            }
        state["quarterly_records"] = quarterly_records
        state["ttm_summary"]       = ttm_summary
    except Exception:
        state["quarterly_records"] = []
        state["ttm_summary"]       = {}

    all_states[sym] = state

    w90 = get_weights(sector, 90)
    print(f"  {info['name']:<35} 30d={composites[30]:.0f} 60d={composites[60]:.0f} 90d={composites[90]:.0f} "
          f"| 90d weights: E={w90['env']:.0%} T={w90['tech']:.0%} F={w90['fund']:.0%} V={w90['val']:.0%}")

with open(MEMORY_DIR / "all_current_states.json", "w") as f:
    json.dump({"generated": datetime.now().isoformat(), "stocks": all_states}, f, indent=2, default=str)

print(f"\nDone. {len(all_states)} stocks updated with optimized sector+horizon weights.")
print("Each stock now has composite_30d, composite_60d, composite_90d separately.")
