"""
Build current state v3 — full enriched scoring with:
  + Analyst consensus (expectation gap signal)
  + P/E valuation signal
  + P/B valuation signal
  + SAHMK fair value signal
  + Cash flow quality
  + Debt/leverage
  + All existing macro + technical signals
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

# Load company KPIs
kpi_path = DATA_RAW / "company_kpis.json"
COMPANY_KPIS = {}
if kpi_path.exists():
    with open(kpi_path, encoding="utf-8") as f:
        COMPANY_KPIS = json.load(f).get("stocks", {})

def score_valuation(sym: str, price: float) -> tuple:
    """
    Valuation score using P/E, P/B, analyst consensus, SAHMK fair value.
    Returns (score 0-100, signals dict)
    """
    kpis  = COMPANY_KPIS.get(sym, {})
    score = 50
    sigs  = {}

    # ── Analyst consensus (Expectation Gap signal) ────────────────────────────
    analyst_upside = kpis.get("analyst_upside_pct")
    n_analysts     = kpis.get("num_analysts", 0) or 0
    if analyst_upside is not None and n_analysts >= 3:
        if analyst_upside > 30:
            score += 15; sigs["analyst"] = f"+{analyst_upside:.1f}% analyst upside ({n_analysts} analysts) (+15)"
        elif analyst_upside > 15:
            score += 10; sigs["analyst"] = f"+{analyst_upside:.1f}% analyst upside ({n_analysts} analysts) (+10)"
        elif analyst_upside > 5:
            score += 5;  sigs["analyst"] = f"+{analyst_upside:.1f}% analyst upside ({n_analysts} analysts) (+5)"
        elif analyst_upside < -10:
            score -= 10; sigs["analyst"] = f"{analyst_upside:.1f}% analyst downside ({n_analysts} analysts) (-10)"
        elif analyst_upside < 0:
            score -= 5;  sigs["analyst"] = f"{analyst_upside:.1f}% analysts negative ({n_analysts} analysts) (-5)"
    elif analyst_upside is not None and n_analysts in [1, 2]:
        # Only 1-2 analysts — use but low weight
        if analyst_upside > 20:
            score += 5; sigs["analyst"] = f"+{analyst_upside:.1f}% (only {n_analysts} analyst — low confidence) (+5)"
        elif analyst_upside < -10:
            score -= 5; sigs["analyst"] = f"{analyst_upside:.1f}% (only {n_analysts} analyst — low confidence) (-5)"

    # ── P/E ratio signal ──────────────────────────────────────────────────────
    pe = kpis.get("pe_ratio")
    if pe and 1 < pe < 500:   # filter nonsense values
        if pe < 6:
            score += 10; sigs["pe"] = f"P/E={pe:.1f} cheap (+10)"
        elif pe < 10:
            score += 5;  sigs["pe"] = f"P/E={pe:.1f} fair (+5)"
        elif pe < 15:
            score += 0;  sigs["pe"] = f"P/E={pe:.1f} neutral"
        elif pe < 25:
            score -= 5;  sigs["pe"] = f"P/E={pe:.1f} elevated (-5)"
        elif pe >= 25:
            score -= 10; sigs["pe"] = f"P/E={pe:.1f} expensive (-10)"

    # ── P/B ratio signal ──────────────────────────────────────────────────────
    pb = kpis.get("price_to_book")
    if pb and 0 < pb < 100:   # filter nonsense
        if pb < 0.8:
            score += 8;  sigs["pb"] = f"P/B={pb:.2f} below book (+8)"
        elif pb < 1.2:
            score += 3;  sigs["pb"] = f"P/B={pb:.2f} near book (+3)"
        elif pb > 5:
            score -= 5;  sigs["pb"] = f"P/B={pb:.2f} premium (-5)"

    # ── SAHMK fair value signal ───────────────────────────────────────────────
    fv_upside = kpis.get("sahmk_fair_upside_pct")
    fv_conf   = kpis.get("fair_price_confidence", 0) or 0
    if fv_upside is not None and fv_conf >= 0.8:
        if fv_upside > 15:
            score += 8;  sigs["fair_value"] = f"SAHMK FV={fv_upside:+.1f}% undervalued (+8)"
        elif fv_upside > 5:
            score += 4;  sigs["fair_value"] = f"SAHMK FV={fv_upside:+.1f}% slight upside (+4)"
        elif fv_upside < -20:
            score -= 8;  sigs["fair_value"] = f"SAHMK FV={fv_upside:+.1f}% overvalued (-8)"
        elif fv_upside < -10:
            score -= 4;  sigs["fair_value"] = f"SAHMK FV={fv_upside:+.1f}% slightly rich (-4)"

    return max(0, min(100, score)), sigs


print("Building current state v3 — full enriched scoring for all stocks...")
print(f"Stocks: {len(STOCKS)}\n")

all_states = {}

for sym, info in STOCKS.items():
    path = DATA_PROCESSED / ("master_dataset.csv" if sym == "1120" else f"master_{sym}.csv")
    if not path.exists():
        continue

    master = pd.read_csv(path, index_col=0, parse_dates=True)
    master.index = pd.to_datetime(master.index).tz_localize(None)
    row     = master.iloc[-1]
    price   = float(row["close"])

    # Macro signals
    rsi     = float(row.get("rsi", 50)) if not pd.isna(row.get("rsi", float("nan"))) else 50.0
    ret_20d = float(row.get("ret_20d", 0)) if not pd.isna(row.get("ret_20d", float("nan"))) else 0.0
    bb_pct  = float(row.get("bb_pct", 0.5)) if not pd.isna(row.get("bb_pct", float("nan"))) else 0.5
    rate_r  = str(row.get("rate_regime", "stable"))
    repo    = float(row.get("repo_rate", 4.25)) if not pd.isna(row.get("repo_rate", float("nan"))) else 4.25
    vix     = float(row.get("vix", 20)) if not pd.isna(row.get("vix", float("nan"))) else 20.0
    oil     = float(row.get("oil", 80)) if not pd.isna(row.get("oil", float("nan"))) else 80.0
    oil_b   = int(row.get("oil_bull", 0))

    # ── Environment score (40%) ───────────────────────────────────────────────
    env = 50; env_sigs = {}
    if rate_r == "stable":    env += 20; env_sigs["rate"] = "stable (+20)"
    elif rate_r == "rising":  env -= 25; env_sigs["rate"] = "rising (-25)"
    elif rate_r == "falling": env -= 5;  env_sigs["rate"] = "falling (-5)"
    if vix > 30:  env += 15; env_sigs["vix"] = f"{vix:.1f} fear (+15)"
    elif vix > 20:env += 5;  env_sigs["vix"] = f"{vix:.1f} elevated (+5)"
    elif vix < 15:env -= 5;  env_sigs["vix"] = f"{vix:.1f} calm (-5)"
    if oil_b:     env += 10; env_sigs["oil"] = "oil uptrend (+10)"
    env = max(0, min(100, env))

    # ── Technical score (25%) ─────────────────────────────────────────────────
    tech = 50; tech_sigs = {}
    if rsi < 20:    tech += 20; tech_sigs["rsi"] = f"{rsi:.1f} extreme oversold (+20)"
    elif rsi < 30:  tech += 15; tech_sigs["rsi"] = f"{rsi:.1f} oversold (+15)"
    elif rsi < 40:  tech += 8;  tech_sigs["rsi"] = f"{rsi:.1f} low (+8)"
    elif rsi > 80:  tech -= 20; tech_sigs["rsi"] = f"{rsi:.1f} extreme overbought (-20)"
    elif rsi > 70:  tech -= 12; tech_sigs["rsi"] = f"{rsi:.1f} overbought (-12)"
    else:           tech_sigs["rsi"] = f"{rsi:.1f} neutral"
    if bb_pct < 0.05:    tech += 20; tech_sigs["bb"] = "lower band (+20)"
    elif bb_pct < 0.15:  tech += 10; tech_sigs["bb"] = "low BB (+10)"
    elif bb_pct > 0.95:  tech -= 15; tech_sigs["bb"] = "upper band (-15)"
    if ret_20d > 0.15:   tech -= 20; tech_sigs["mom"] = f"{ret_20d:.1%} strong rally (-20)"
    elif ret_20d > 0.10: tech -= 10; tech_sigs["mom"] = f"{ret_20d:.1%} rally (-10)"
    elif ret_20d <-0.15: tech += 15; tech_sigs["mom"] = f"{ret_20d:.1%} drop (+15)"
    elif ret_20d <-0.10: tech += 8;  tech_sigs["mom"] = f"{ret_20d:.1%} drop (+8)"
    tech = max(0, min(100, tech))

    # ── Fundamental score (20%) — enriched ───────────────────────────────────
    _, _, _, _, annual_df = load_enriched_financials(sym)
    fund, fund_sigs = score_fundamental_enriched(
        pd.Timestamp(date.today()), annual_df,
        pd.DataFrame(), pd.DataFrame()
    )

    # ── Valuation score (15%) — NEW ───────────────────────────────────────────
    val_score, val_sigs = score_valuation(sym, price)

    # ── Composite (new weights) ───────────────────────────────────────────────
    # Environment 40% | Technical 25% | Fundamental 20% | Valuation 15%
    composite = (env * 0.40 + tech * 0.25 + fund * 0.20 + val_score * 0.15)

    # Load baseline from memory
    mem_path = MEMORY_DIR / f"memory_{sym}.json"
    baseline_90d = 6.0
    if mem_path.exists():
        with open(mem_path, encoding="utf-8") as f:
            mem = json.load(f)
        baseline_90d = mem.get("baseline_90d", {}).get("avg_pct", 6.0) or 6.0

    # Regime-adjusted forecast
    if composite > 74:   base_90d = max(baseline_90d * 1.35, 10.0)
    elif composite > 67: base_90d = max(baseline_90d * 1.15, 7.5)
    elif composite > 60: base_90d = baseline_90d
    elif composite > 53: base_90d = baseline_90d * 0.55
    elif composite > 45: base_90d = baseline_90d * 0.15
    else:                base_90d = -3.5
    if rate_r == "rising": base_90d -= 5.0

    base_30d = base_90d * 0.40
    base_60d = base_90d * 0.70
    spread   = 5.0 + (1 - abs(composite-50)/50) * 5.0
    conf     = int(40 + abs(composite-50)/50 * 35 + (1 - abs(env-tech)/100) * 25)
    conf     = max(20, min(90, conf))

    c       = master["close"]
    h52     = float(c.rolling(252).max().iloc[-1]) if len(c) >= 252 else float(c.max())
    l52     = float(c.rolling(252).min().iloc[-1]) if len(c) >= 252 else float(c.min())
    kpis    = COMPANY_KPIS.get(sym, {})

    state = {
        "symbol": sym, "name": info["name"], "sector": info["sector"],
        "as_of":  date.today().isoformat(),
        "price":  round(price, 2),
        "rsi":    round(rsi, 1), "bb_pct": round(bb_pct, 3),
        "ret_20d_pct": round(ret_20d * 100, 1),
        "rate_regime": rate_r, "repo_rate": round(repo, 2),
        "vix": round(vix, 1), "oil_price": round(oil, 1),
        "env_score":   round(env, 1),
        "tech_score":  round(tech, 1),
        "fund_score":  round(fund, 1),
        "val_score":   round(val_score, 1),
        "composite":   round(composite, 1),
        "env_signals": env_sigs, "tech_signals": tech_sigs,
        "fund_signals": fund_sigs, "val_signals": val_sigs,
        "trading_days": len(master),
        "dist_from_52w_high_pct": round((price - h52) / h52 * 100, 1),
        # KPI snapshot
        "pe_ratio":           kpis.get("pe_ratio"),
        "price_to_book":      kpis.get("price_to_book"),
        "eps_ttm":            kpis.get("eps_ttm"),
        "analyst_target_mean":kpis.get("analyst_target_mean"),
        "analyst_upside_pct": kpis.get("analyst_upside_pct"),
        "num_analysts":       kpis.get("num_analysts"),
        "fair_price":         kpis.get("fair_price"),
        "sahmk_fair_upside_pct": kpis.get("sahmk_fair_upside_pct"),
        "forecast": {
            "base_30d_pct":  round(base_30d, 1),
            "base_60d_pct":  round(base_60d, 1),
            "base_90d_pct":  round(base_90d, 1),
            "bull_90d_pct":  round(base_90d + spread, 1),
            "bear_90d_pct":  round(base_90d - spread, 1),
            "target_30d":    round(price * (1 + base_30d/100), 2),
            "target_60d":    round(price * (1 + base_60d/100), 2),
            "target_90d":    round(price * (1 + base_90d/100), 2),
            "bull_target_90d": round(price * (1 + (base_90d+spread)/100), 2),
            "bear_target_90d": round(price * (1 + (base_90d-spread)/100), 2),
            "confidence":    conf,
            "confidence_label": "HIGH" if conf >= 65 else ("MEDIUM" if conf >= 45 else "LOW"),
        },
    }
    all_states[sym] = state

    val_tag = f"Analyst={kpis.get('analyst_upside_pct',0):+.1f}%" if kpis.get('analyst_upside_pct') else ""
    print(f"  {info['name']:<35} comp={composite:.0f}/100  "
          f"env={env:.0f} tech={tech:.0f} fund={fund:.0f} val={val_score:.0f}  "
          f"90d={base_90d:+.1f}%  {val_tag}")

with open(MEMORY_DIR / "all_current_states.json", "w") as f:
    json.dump({"generated": datetime.now().isoformat(), "stocks": all_states}, f,
              indent=2, default=str)

print(f"\nDone. {len(all_states)} stocks updated with full 4-layer scoring.")
print("Weights: Environment 40% | Technical 25% | Fundamental 20% | Valuation 15%")
