"""
WEIGHT OPTIMIZER
================
Finds the optimal layer weights (Environment / Technical / Fundamental / Valuation)
for each sector AND for each time horizon (30d / 60d / 90d).

Method: Grid search over weight combinations, evaluate directional accuracy
on historical walk-forward predictions.

Results saved to memory/optimal_weights.json
"""

import sys, json, warnings
from datetime import date, datetime
from pathlib import Path
from itertools import product

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import DATA_PROCESSED, MEMORY_DIR, REPORTS_DIR
from shared import STOCKS
from rebuild_engine import (load_enriched_financials, score_fundamental_enriched)
from final_complete_rebuild import score_environment_sector
from rebuild_with_valuation import build_historical_valuation


# ── Weight grid to test ───────────────────────────────────────────────────────
# Each tuple: (env_weight, tech_weight, fund_weight, val_weight)
# Constraint: must sum to 1.0

WEIGHT_GRID = []
for e in [0.20, 0.30, 0.40, 0.50, 0.60]:
    for t in [0.15, 0.25, 0.35, 0.45]:
        for f in [0.10, 0.15, 0.20, 0.25, 0.30]:
            v = round(1.0 - e - t - f, 2)
            if 0.05 <= v <= 0.20:
                WEIGHT_GRID.append((e, t, f, v))

print(f"Testing {len(WEIGHT_GRID)} weight combinations per sector per horizon")


def score_composite(row, sector, annual_df, balance_df, cashflow_df, w_env, w_tech, w_fund, w_val):
    """Score one prediction date with given weights."""
    env, _   = score_environment_sector(row, sector)

    tech = 50
    rsi  = row.get("rsi", 50)
    if not pd.isna(rsi):
        if rsi < 20:   tech += 20
        elif rsi < 30: tech += 15
        elif rsi < 40: tech += 8
        elif rsi > 80: tech -= 20
        elif rsi > 70: tech -= 12
    bb = row.get("bb_pct", 0.5)
    if not pd.isna(bb):
        if bb < 0.05:   tech += 20
        elif bb < 0.15: tech += 10
        elif bb > 0.95: tech -= 15
    ret20 = row.get("ret_20d", 0)
    if not pd.isna(ret20):
        if ret20 > 0.15:   tech -= 20
        elif ret20 > 0.10: tech -= 10
        elif ret20 < -0.15:tech += 15
        elif ret20 < -0.10:tech += 8
    tech = max(0, min(100, tech))

    fund, _ = score_fundamental_enriched(row.name, annual_df, balance_df, cashflow_df)

    val = 50
    pe  = row.get("pe_hist")
    if pe and not pd.isna(pe) and 0 < pe < 200:
        if pe < 6:     val += 15
        elif pe < 9:   val += 10
        elif pe < 12:  val += 4
        elif pe < 18:  val -= 4
        elif pe >= 18: val -= 10
    pb = row.get("pb_hist")
    if pb and not pd.isna(pb) and 0 < pb < 50:
        if pb < 0.8:   val += 10
        elif pb < 1.2: val += 4
        elif pb > 5:   val -= 6
    val = max(0, min(100, val))

    return env*w_env + tech*w_tech + fund*w_fund + val*w_val


def evaluate_weights(predictions_df, w_env, w_tech, w_fund, w_val, horizon=90):
    """Evaluate a weight combination on historical predictions."""
    col = f"actual_{horizon}d_pct"
    df  = predictions_df.dropna(subset=[col])
    if len(df) < 5:
        return 0.0

    # Recompute direction with these weights
    # composite already stored — but we need to reweight
    # Use stored component scores
    comp = (df["env_score"]*w_env + df["tech_score"]*w_tech +
            df["fund_score"]*w_fund + df.get("val_score", 50)*w_val)
    pred_dir   = (comp > 50).astype(int)
    actual_dir = (df[col] > 0).astype(int)
    return (pred_dir == actual_dir).mean()


def optimize_sector(sym_list: list, sector: str) -> dict:
    """Find optimal weights for a sector across all horizons."""
    print(f"\n  [{sector}] {len(sym_list)} stocks...")

    all_preds = []
    for sym in sym_list:
        path = DATA_PROCESSED / ("master_dataset.csv" if sym == "1120" else f"master_{sym}.csv")
        if not path.exists():
            continue
        prefix = f"{sym}_" if sym != "1120" else ""
        bt_path = REPORTS_DIR / f"{prefix}PREDICTION_BACKTEST_REPORT.json"
        if not bt_path.exists():
            continue
        with open(bt_path) as f:
            bt = json.load(f)
        preds = bt.get("predictions", [])
        if preds:
            df = pd.DataFrame(preds)
            df["sym"] = sym
            all_preds.append(df)

    if not all_preds:
        return {"sector": sector, "note": "no data"}

    combined = pd.concat(all_preds, ignore_index=True)
    combined  = combined.dropna(subset=["env_score","tech_score","fund_score"])
    if "val_score" not in combined.columns:
        combined["val_score"] = 50

    results = {"sector": sector}

    for horizon in [30, 60, 90]:
        best_acc   = 0
        best_weights = (0.40, 0.25, 0.20, 0.15)
        best_edge    = 0

        baseline_col = f"actual_{horizon}d_pct"
        df_h = combined.dropna(subset=[baseline_col])
        if len(df_h) < 10:
            results[f"horizon_{horizon}d"] = {
                "weights": {"env": 0.40, "tech": 0.25, "fund": 0.20, "val": 0.15},
                "accuracy": None,
                "note": "insufficient data"
            }
            continue

        baseline_acc = (df_h[baseline_col] > 0).mean()

        for w_env, w_tech, w_fund, w_val in WEIGHT_GRID:
            acc = evaluate_weights(df_h, w_env, w_tech, w_fund, w_val, horizon)
            if acc > best_acc:
                best_acc     = acc
                best_weights = (w_env, w_tech, w_fund, w_val)
                best_edge    = acc - baseline_acc

        results[f"horizon_{horizon}d"] = {
            "weights": {
                "env":  best_weights[0],
                "tech": best_weights[1],
                "fund": best_weights[2],
                "val":  best_weights[3],
            },
            "accuracy":          round(best_acc * 100, 1),
            "baseline":          round(baseline_acc * 100, 1),
            "edge_over_baseline":round(best_edge * 100, 1),
            "n_predictions":     len(df_h),
        }
        print(f"    {horizon}d → env={best_weights[0]:.0%} tech={best_weights[1]:.0%} "
              f"fund={best_weights[2]:.0%} val={best_weights[3]:.0%} "
              f"| acc={best_acc*100:.1f}% (was {baseline_acc*100:.1f}%) "
              f"edge={best_edge*100:+.1f}%")

    return results


def main():
    print(f"\nWEIGHT OPTIMIZER — Sector + Horizon Specific")
    print(f"Date: {date.today()}")
    print(f"Testing {len(WEIGHT_GRID)} combinations × 3 horizons × {len(set(v['sector'] for v in STOCKS.values()))} sectors")

    # Group stocks by sector
    sector_stocks = {}
    for sym, info in STOCKS.items():
        s = info["sector"]
        if s not in sector_stocks:
            sector_stocks[s] = []
        sector_stocks[s].append(sym)

    all_results = {}

    for sector, syms in sorted(sector_stocks.items()):
        result = optimize_sector(syms, sector)
        all_results[sector] = result

    # Save
    output = {
        "generated": datetime.now().isoformat(),
        "note": "Empirically optimized weights per sector per prediction horizon",
        "sectors": all_results,
    }
    with open(MEMORY_DIR / "optimal_weights.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*75}")
    print(f"  OPTIMAL WEIGHTS BY SECTOR AND HORIZON")
    print(f"{'='*75}")
    print(f"  {'Sector':<22} {'Horizon':>8} {'Env':>6} {'Tech':>6} {'Fund':>6} {'Val':>5} {'Acc%':>6} {'Edge':>6}")
    print(f"  {'-'*22} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*6} {'-'*6}")

    for sector, r in sorted(all_results.items()):
        for hz in [30, 60, 90]:
            h = r.get(f"horizon_{hz}d", {})
            if not h or not h.get("weights"):
                continue
            w = h["weights"]
            print(f"  {sector:<22} {hz:>7}d {w['env']:>6.0%} {w['tech']:>6.0%} "
                  f"{w['fund']:>6.0%} {w['val']:>5.0%} "
                  f"{h.get('accuracy','?'):>6} {h.get('edge_over_baseline',0):>+6.1f}%")

    print(f"\n  Saved → memory/optimal_weights.json")
    print(f"  Run rebuild_with_optimal_weights.py to apply these weights")

    return all_results


if __name__ == "__main__":
    main()
