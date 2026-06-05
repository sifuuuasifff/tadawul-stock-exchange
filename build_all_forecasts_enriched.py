"""
Build current state for all stocks using enriched fundamental scoring.
Replaces build_all_forecasts.py for post-rebuild runs.
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

print("Building enriched current state for all stocks...")

all_states = {}

for sym, info in STOCKS.items():
    path = DATA_PROCESSED / ("master_dataset.csv" if sym == "1120" else f"master_{sym}.csv")
    if not path.exists():
        continue

    master = pd.read_csv(path, index_col=0, parse_dates=True)
    master.index = pd.to_datetime(master.index).tz_localize(None)

    row     = master.iloc[-1]
    price   = float(row["close"])
    rsi     = float(row.get("rsi", 50)) if not pd.isna(row.get("rsi", float("nan"))) else 50.0
    ret_20d = float(row.get("ret_20d", 0)) if not pd.isna(row.get("ret_20d", float("nan"))) else 0.0
    bb_pct  = float(row.get("bb_pct", 0.5)) if not pd.isna(row.get("bb_pct", float("nan"))) else 0.5
    rate_r  = str(row.get("rate_regime", "stable"))
    repo    = float(row.get("repo_rate", 4.25)) if not pd.isna(row.get("repo_rate", float("nan"))) else 4.25
    vix     = float(row.get("vix", 20)) if not pd.isna(row.get("vix", float("nan"))) else 20.0
    oil     = float(row.get("oil", 80)) if not pd.isna(row.get("oil", float("nan"))) else 80.0
    oil_b   = int(row.get("oil_bull", 0))

    # Environment score
    env = 50
    env_sigs = {}
    if rate_r == "stable":   env += 20; env_sigs["rate"] = "stable (+20)"
    elif rate_r == "rising": env -= 25; env_sigs["rate"] = "rising (-25)"
    elif rate_r == "falling":env -= 5;  env_sigs["rate"] = "falling (-5)"
    if vix > 30:  env += 15; env_sigs["vix"] = f"{vix:.1f} fear (+15)"
    elif vix > 20:env += 5;  env_sigs["vix"] = f"{vix:.1f} elevated (+5)"
    elif vix < 15:env -= 5;  env_sigs["vix"] = f"{vix:.1f} calm (-5)"
    if oil_b:     env += 10; env_sigs["oil"] = "uptrend (+10)"
    env = max(0, min(100, env))

    # Technical score
    tech = 50
    tech_sigs = {}
    if rsi < 20:    tech += 20; tech_sigs["rsi"] = f"{rsi:.1f} extreme oversold (+20)"
    elif rsi < 30:  tech += 15; tech_sigs["rsi"] = f"{rsi:.1f} oversold (+15)"
    elif rsi < 40:  tech += 8;  tech_sigs["rsi"] = f"{rsi:.1f} low (+8)"
    elif rsi > 80:  tech -= 20; tech_sigs["rsi"] = f"{rsi:.1f} extreme overbought (-20)"
    elif rsi > 70:  tech -= 12; tech_sigs["rsi"] = f"{rsi:.1f} overbought (-12)"
    else:           tech_sigs["rsi"] = f"{rsi:.1f} neutral"
    if bb_pct < 0.05:   tech += 20; tech_sigs["bb"] = f"lower band (+20)"
    elif bb_pct < 0.15: tech += 10; tech_sigs["bb"] = f"low BB (+10)"
    elif bb_pct > 0.95: tech -= 15; tech_sigs["bb"] = f"upper band (-15)"
    if ret_20d > 0.15:  tech -= 20; tech_sigs["mom"] = f"{ret_20d:.1%} strong rally (-20)"
    elif ret_20d > 0.10:tech -= 10; tech_sigs["mom"] = f"{ret_20d:.1%} rally (-10)"
    elif ret_20d <-0.15:tech += 15; tech_sigs["mom"] = f"{ret_20d:.1%} drop (+15)"
    elif ret_20d <-0.10:tech += 8;  tech_sigs["mom"] = f"{ret_20d:.1%} drop (+8)"
    tech = max(0, min(100, tech))

    # Enriched fundamental score
    _, _, _, _, annual_df = load_enriched_financials(sym)
    balance_df = pd.DataFrame()  # used in scoring
    cashflow_df = pd.DataFrame()
    fund, fund_sigs = score_fundamental_enriched(
        pd.Timestamp(date.today()), annual_df,
        balance_df, cashflow_df
    )

    # Composite (higher fundamental weight)
    composite = env * 0.40 + tech * 0.30 + fund * 0.30

    # Load baseline from memory
    mem_path = MEMORY_DIR / f"memory_{sym}.json"
    baseline_90d = 6.0
    if mem_path.exists():
        with open(mem_path, encoding="utf-8") as f:
            mem = json.load(f)
        baseline_90d = mem.get("baseline_90d", {}).get("avg_pct", 6.0) or 6.0

    # Regime-adjusted base return
    if composite > 72:   base_90d = max(baseline_90d * 1.3, 9.0)
    elif composite > 65: base_90d = max(baseline_90d * 1.1, 7.0)
    elif composite > 58: base_90d = baseline_90d
    elif composite > 50: base_90d = baseline_90d * 0.6
    elif composite > 42: base_90d = baseline_90d * 0.2
    else:                base_90d = -3.0
    if rate_r == "rising": base_90d -= 5.0

    base_30d = base_90d * 0.40
    base_60d = base_90d * 0.70
    spread   = 5.0 + (1 - abs(composite-50)/50) * 5.0
    conf     = int(40 + abs(composite-50)/50*35 + (1-abs(env-tech)/100)*25)
    conf     = max(20, min(90, conf))

    # 52w high/low
    c = master["close"]
    high_52w = float(c.rolling(252).max().iloc[-1]) if len(c) >= 252 else float(c.max())
    low_52w  = float(c.rolling(252).min().iloc[-1]) if len(c) >= 252 else float(c.min())

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
        "composite":   round(composite, 1),
        "env_signals": env_sigs, "tech_signals": tech_sigs, "fund_signals": fund_sigs,
        "trading_days": len(master),
        "dist_from_52w_high_pct": round((price - high_52w) / high_52w * 100, 1),
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
    print(f"  {info['name']:<32} composite={composite:.0f}/100  "
          f"90d={base_90d:+.1f}%  fund={fund:.0f}/100  "
          f"{'CFQ✓' if fund_sigs.get('cfq') else ''}")

with open(MEMORY_DIR / "all_current_states.json", "w") as f:
    json.dump({"generated": datetime.now().isoformat(), "stocks": all_states}, f,
              indent=2, default=str)

print(f"\nDone. {len(all_states)} stocks updated with enriched fundamental scoring.")
