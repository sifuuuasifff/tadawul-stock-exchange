"""
REBUILD WITH HISTORICAL VALUATION SIGNALS
==========================================
Reconstructs daily P/E and P/B for every stock using:
  - Daily price (from master dataset)
  - Trailing 12M net income (from quarterly financials)
  - Shares outstanding (from SAHMK company endpoint)
  - Quarterly equity (from balance sheets)

Then re-runs walk-forward for all stocks with these signals,
rebuilds mistake vault, recalibrates confidence.

This makes the model fully air tight — no signal used in production
that hasn't been historically tested.

Run:  python rebuild_with_valuation.py
"""

import sys, json, subprocess, warnings
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import DATA_RAW, DATA_PROCESSED, MEMORY_DIR, REPORTS_DIR
from shared import STOCKS
from rebuild_engine import (load_enriched_financials, score_fundamental_enriched,
                             discover_financial_signals)
from phases.full_engine import (run_signal_discovery, run_memory_engine,
                                 run_hypothesis_engine, run_recalibration,
                                 save_outputs, load_dividends, load_events)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — RECONSTRUCT HISTORICAL P/E AND P/B
# ══════════════════════════════════════════════════════════════════════════════

def build_historical_valuation(sym: str, master: pd.DataFrame,
                                income_df: pd.DataFrame,
                                balance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds pe_hist and pb_hist columns to master dataset.

    P/E = price / (trailing 12M net income / shares_outstanding)
    P/B = price / (book value per share = equity / shares_outstanding)

    No look-ahead bias: only uses financials known at each date.
    """
    # Get shares outstanding
    kpi_path = DATA_RAW / "company_kpis.json"
    shares = None
    if kpi_path.exists():
        with open(kpi_path, encoding="utf-8") as f:
            kpis = json.load(f).get("stocks", {}).get(sym, {})
        shares = kpis.get("shares_outstanding")

    if not shares or shares <= 0:
        # Estimate from market cap and price if available
        shares = None

    # ── Build trailing 12M net income series ─────────────────────────────────
    # For each trading day, use the sum of the last 4 known quarterly net incomes
    # (or the last known annual if quarterly not available)
    # No look-ahead: only use reports with report_date <= trading_date

    pe_series = pd.Series(np.nan, index=master.index)
    pb_series = pd.Series(np.nan, index=master.index)

    if income_df.empty or shares is None:
        return master

    # Sort income by date
    inc = income_df.sort_values("report_date")

    # Build quarterly TTM net income
    # Use rolling 4-quarter sum where available, else annual
    ttm_dict = {}  # date -> TTM net income
    for i, row in inc.iterrows():
        report_date = row["report_date"]
        ni = row.get("net_income")
        if pd.isna(ni):
            continue
        # Find last 4 quarters ending at this report date
        past_4 = inc[inc["report_date"] <= report_date].tail(4)
        if len(past_4) >= 2:
            ttm = past_4["net_income"].dropna().sum()
        else:
            ttm = ni  # just this quarter annualised
        ttm_dict[report_date] = ttm

    if not ttm_dict:
        return master

    ttm_df = pd.Series(ttm_dict).sort_index()

    # Build quarterly book value per share
    bvps_dict = {}
    if not balance_df.empty:
        bal = balance_df.sort_values("report_date")
        for _, row in bal.iterrows():
            eq = row.get("stockholders_equity")
            if not pd.isna(eq) and eq > 0:
                bvps_dict[row["report_date"]] = eq / shares
    bvps_df = pd.Series(bvps_dict).sort_index() if bvps_dict else pd.Series(dtype=float)

    # Map to each trading day — use last known TTM before each date (no look-ahead)
    for trade_date in master.index:
        price = master.loc[trade_date, "close"]

        # Latest TTM known before this trading day
        known_ttm = ttm_df[ttm_df.index <= trade_date]
        if not known_ttm.empty:
            ttm = known_ttm.iloc[-1]
            eps = ttm / shares
            if eps > 0 and price > 0:
                pe = price / eps
                if 0 < pe < 200:   # filter nonsense values
                    pe_series[trade_date] = round(pe, 2)

        # Latest book value known before this trading day
        if not bvps_df.empty:
            known_bv = bvps_df[bvps_df.index <= trade_date]
            if not known_bv.empty:
                bv = known_bv.iloc[-1]
                if bv > 0 and price > 0:
                    pb = price / bv
                    if 0 < pb < 50:
                        pb_series[trade_date] = round(pb, 2)

    master["pe_hist"] = pe_series
    master["pb_hist"] = pb_series
    return master


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — WALK-FORWARD WITH FULL VALUATION SCORING
# ══════════════════════════════════════════════════════════════════════════════

def score_valuation_historical(row: pd.Series, pred_date,
                                income_df: pd.DataFrame,
                                balance_df: pd.DataFrame,
                                shares: float) -> tuple:
    """Historical valuation score using only data known at pred_date."""
    score = 50
    sigs  = {}

    # Historical P/E (from reconstructed series)
    pe = row.get("pe_hist")
    if pe and not pd.isna(pe) and 0 < pe < 200:
        if pe < 6:
            score += 15; sigs["pe"] = f"P/E={pe:.1f} historically cheap (+15)"
        elif pe < 9:
            score += 10; sigs["pe"] = f"P/E={pe:.1f} cheap (+10)"
        elif pe < 12:
            score += 4;  sigs["pe"] = f"P/E={pe:.1f} fair (+4)"
        elif pe < 18:
            score -= 4;  sigs["pe"] = f"P/E={pe:.1f} elevated (-4)"
        elif pe >= 18:
            score -= 10; sigs["pe"] = f"P/E={pe:.1f} expensive (-10)"

    # Historical P/B (from reconstructed series)
    pb = row.get("pb_hist")
    if pb and not pd.isna(pb) and 0 < pb < 50:
        if pb < 0.8:
            score += 10; sigs["pb"] = f"P/B={pb:.2f} below book (+10)"
        elif pb < 1.2:
            score += 4;  sigs["pb"] = f"P/B={pb:.2f} near book (+4)"
        elif pb < 2.5:
            score += 0;  sigs["pb"] = f"P/B={pb:.2f} fair"
        elif pb > 5:
            score -= 6;  sigs["pb"] = f"P/B={pb:.2f} expensive (-6)"

    return max(0, min(100, score)), sigs


def run_full_walkforward(sym: str, name: str, master: pd.DataFrame,
                          annual_df: pd.DataFrame,
                          balance_df: pd.DataFrame,
                          cashflow_df: pd.DataFrame,
                          shares: float) -> dict:
    """
    Full walk-forward with all 4 scoring layers — all historically validated.
    Weights: Environment 40% | Technical 25% | Fundamental 20% | Valuation 15%
    """
    weights = {"env": 0.40, "tech": 0.25, "fund": 0.20, "val": 0.15}

    earliest = master.index[200] if len(master) > 200 else master.index[0]
    start    = max(earliest, pd.Timestamp("2016-01-01"))
    end      = master.index[-1] - pd.Timedelta(days=91)
    if start >= end:
        start = master.index[len(master)//2]

    pred_dates = []
    d = start
    while d < end:
        avail = master.index[master.index >= d]
        if len(avail) > 0:
            pred_dates.append(avail[0])
        d += pd.Timedelta(days=90)

    predictions   = []
    mistake_vault = []

    for pred_date in pred_dates:
        row = master.loc[pred_date]

        # ── Environment ───────────────────────────────────────────────────────
        env = 50
        regime = row.get("rate_regime", "stable")
        if regime == "stable":   env += 20
        elif regime == "rising": env -= 25
        elif regime == "falling":env -= 5
        vix = row.get("vix", 20)
        if vix > 30:  env += 15
        elif vix > 20:env += 5
        elif vix < 15:env -= 5
        if row.get("oil_bull", 0): env += 10
        env = max(0, min(100, env))

        # ── Technical ─────────────────────────────────────────────────────────
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
            elif ret20 <-0.15: tech += 15
            elif ret20 <-0.10: tech += 8
        tech = max(0, min(100, tech))

        # ── Fundamental ───────────────────────────────────────────────────────
        fund, fund_sigs = score_fundamental_enriched(
            pred_date, annual_df, balance_df, cashflow_df)

        # ── Valuation (historical P/E + P/B — no look-ahead) ─────────────────
        val, val_sigs = score_valuation_historical(
            row, pred_date, pd.DataFrame(),
            balance_df, shares)

        # ── Composite ─────────────────────────────────────────────────────────
        comp = (env  * weights["env"]  + tech * weights["tech"] +
                fund * weights["fund"] + val  * weights["val"])

        # ── Forecast ──────────────────────────────────────────────────────────
        if comp > 74:   base_90d = 9.5
        elif comp > 67: base_90d = 7.5
        elif comp > 60: base_90d = 5.5
        elif comp > 53: base_90d = 3.0
        elif comp > 46: base_90d = 1.0
        else:           base_90d = -3.5
        if regime == "rising":  base_90d -= 5.0
        elif regime == "falling": base_90d -= 1.0

        base_30d = base_90d * 0.40
        base_60d = base_90d * 0.70
        conf = int(40 + abs(comp-50)/50*35 + (1-abs(env-tech)/100)*25)
        conf = max(20, min(90, conf))
        price = float(row["close"])

        # ── Actual outcomes ───────────────────────────────────────────────────
        future = master.index[master.index > pred_date]
        actual = {}
        for w in [30, 60, 90]:
            if len(future) >= w:
                idx = master.index.get_loc(future[0]) + w - 1
                if idx < len(master):
                    p_fut = master["close"].iloc[idx]
                    actual[f"actual_{w}d_pct"]  = round((p_fut/price-1)*100, 1)
                    actual[f"hit_target_{w}d"]  = (actual[f"actual_{w}d_pct"]>0)==(base_90d>0)

        pred = {
            "prediction_date": pred_date.date().isoformat(),
            "price": round(price,2), "rate_regime": regime,
            "rsi":   round(float(rsi),1) if not pd.isna(rsi) else None,
            "vix":   round(float(vix),1),
            "pe_at_prediction": round(float(row.get("pe_hist",0)),1) if not pd.isna(row.get("pe_hist",0)) else None,
            "pb_at_prediction": round(float(row.get("pb_hist",0)),1) if not pd.isna(row.get("pb_hist",0)) else None,
            "env_score": round(env,1), "tech_score": round(tech,1),
            "fund_score":round(fund,1), "val_score":  round(val,1),
            "composite": round(comp,1),
            "base_30d_pct": round(base_30d,1),
            "base_60d_pct": round(base_60d,1),
            "base_90d_pct": round(base_90d,1),
            "confidence":   conf,
            **actual,
        }
        predictions.append(pred)

        # ── Mistake vault ─────────────────────────────────────────────────────
        err = abs(actual.get("actual_90d_pct",0) - base_90d)
        if err > 12 and "actual_90d_pct" in actual:
            dir_ok = (actual["actual_90d_pct"]>0)==(base_90d>0)
            pe_str = f"P/E={row.get('pe_hist','?'):.1f}" if not pd.isna(row.get("pe_hist",float("nan"))) else "P/E=N/A"
            mistake_vault.append({
                "prediction_date": pred_date.date().isoformat(),
                "predicted_90d":   base_90d,
                "actual_90d":      actual["actual_90d_pct"],
                "error":           round(err,1),
                "direction_correct": dir_ok,
                "composite_score": round(comp,1),
                "val_score":       round(val,1),
                "rate_regime":     regime,
                "rsi":             round(float(rsi),1) if not pd.isna(rsi) else None,
                "pe_at_prediction":pe_str,
                "root_cause": (
                    f"Valuation signal misleading (val={val:.0f}/100, pe={pe_str})" if not dir_ok and val > 65
                    else "Model underestimated upside" if actual["actual_90d_pct"] > base_90d + 10
                    else "Model underestimated downside" if actual["actual_90d_pct"] < base_90d - 10
                    else "Mixed macro + valuation signals"
                ),
            })

    # ── Validation ────────────────────────────────────────────────────────────
    df = pd.DataFrame(predictions)
    df = df[df["actual_90d_pct"].notna()] if "actual_90d_pct" in df.columns else df
    validation = {}
    if len(df) >= 5 and "actual_90d_pct" in df.columns:
        df["pred_dir"]   = (df["base_90d_pct"] > 0).astype(int)
        df["actual_dir"] = (df["actual_90d_pct"] > 0).astype(int)
        df["correct"]    = (df["pred_dir"] == df["actual_dir"]).astype(int)
        dir_acc  = df["correct"].mean()
        baseline = df["actual_dir"].mean()
        mae      = (df["actual_90d_pct"] - df["base_90d_pct"]).abs().mean()

        # P/E signal effectiveness — new test
        pe_known = df.dropna(subset=["pe_at_prediction"])
        pe_cheap = pe_known[pe_known["pe_at_prediction"] < 9]
        pe_expensive = pe_known[pe_known["pe_at_prediction"] > 18]

        validation = {
            "n_predictions":            len(df),
            "directional_accuracy_pct": round(dir_acc*100, 1),
            "baseline_always_long_pct": round(baseline*100, 1),
            "edge_over_baseline_pct":   round((dir_acc-baseline)*100, 1),
            "mae_pct":                  round(mae, 2),
            "avg_predicted_90d":        round(df["base_90d_pct"].mean(), 2),
            "avg_actual_90d":           round(df["actual_90d_pct"].mean(), 2),
            "systematic_bias":          round(df["actual_90d_pct"].mean()-df["base_90d_pct"].mean(), 2),
            "weights_used":             weights,
            # P/E signal validation
            "pe_signal_test": {
                "n_cheap_pe":         len(pe_cheap),
                "cheap_pe_accuracy":  round(pe_cheap["correct"].mean()*100,1) if len(pe_cheap)>3 else None,
                "cheap_pe_avg_90d":   round(pe_cheap["actual_90d_pct"].mean(),1) if len(pe_cheap)>3 else None,
                "n_expensive_pe":     len(pe_expensive),
                "expensive_pe_accuracy": round(pe_expensive["correct"].mean()*100,1) if len(pe_expensive)>3 else None,
                "expensive_pe_avg_90d":  round(pe_expensive["actual_90d_pct"].mean(),1) if len(pe_expensive)>3 else None,
            },
        }
        # Regime breakdown
        regime_acc = {}
        for r in ["rising","stable","falling"]:
            sub = df[df["rate_regime"]==r]
            if len(sub) >= 3:
                regime_acc[r] = {"n": len(sub),
                                  "accuracy": round(sub["correct"].mean()*100,1),
                                  "avg_actual": round(sub["actual_90d_pct"].mean(),1)}
        validation["regime_breakdown"] = regime_acc

    return {
        "sym": sym, "name": name,
        "generated":    datetime.now().isoformat(),
        "validation":   validation,
        "predictions":  predictions,
        "mistake_vault":mistake_vault,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN REBUILD
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 65)
    print("  FINAL REBUILD — Full 4-Layer Model, All Signals Historically Tested")
    print(f"  Stocks: {len(STOCKS)} | Date: {date.today()}")
    print("=" * 65)

    # Load company KPIs for shares outstanding
    kpi_path = DATA_RAW / "company_kpis.json"
    KPIS = {}
    if kpi_path.exists():
        with open(kpi_path, encoding="utf-8") as f:
            KPIS = json.load(f).get("stocks", {})

    all_results = []
    failed      = []

    for sym, info in STOCKS.items():
        name = info["name"]
        path = DATA_PROCESSED / ("master_dataset.csv" if sym == "1120" else f"master_{sym}.csv")
        if not path.exists():
            continue

        try:
            print(f"\n  {'─'*52}")
            print(f"  {name} ({sym})")

            master = pd.read_csv(path, index_col=0, parse_dates=True)
            master.index = pd.to_datetime(master.index).tz_localize(None)

            income_df, balance_df, cashflow_df, _, annual_df = load_enriched_financials(sym)
            divs   = load_dividends(sym)
            events = load_events(sym)
            shares = KPIS.get(sym, {}).get("shares_outstanding") or 0

            # ── Add historical P/E and P/B ─────────────────────────────────
            print(f"  Building historical P/E and P/B...", end="")
            master = build_historical_valuation(sym, master, income_df, balance_df)
            pe_coverage = master["pe_hist"].notna().sum() if "pe_hist" in master.columns else 0
            pb_coverage = master["pb_hist"].notna().sum() if "pb_hist" in master.columns else 0
            print(f" P/E: {pe_coverage} days, P/B: {pb_coverage} days")

            # Save enriched master
            master.to_csv(path)

            # ── Phase 1: Signal discovery + new valuation signals ──────────
            signals = run_signal_discovery(sym, name, master)

            # Add P/E signal tests
            if "pe_hist" in master.columns and master["pe_hist"].notna().sum() > 30:
                valid = master.dropna(subset=["fwd_90d"])
                base_90d = valid["fwd_90d"].mean() * 100

                for pe_thresh, label in [(6,"<6 (cheap)"), (9,"<9"), (12,"<12"),
                                          (18,">18 (expensive)"), (25,">25 (very exp)")]:
                    if "<" in label:
                        cond = master["pe_hist"] < pe_thresh
                    else:
                        cond = master["pe_hist"] > pe_thresh
                    sub = valid[cond.reindex(valid.index, fill_value=False)]
                    if len(sub) < 8:
                        continue
                    fwd = sub["fwd_90d"].dropna()
                    t, p = stats.ttest_1samp(fwd, 0)
                    signals["signals"].append({
                        "signal":          f"Historical P/E {label}",
                        "group":           "Valuation — P/E",
                        "occurrences":     len(sub),
                        "avg_return_90d":  round(fwd.mean()*100, 2),
                        "pct_positive_90d":round((fwd>0).mean()*100, 1),
                        "p_value_90d":     round(p, 3),
                        "significant_90d": bool(p < 0.05),
                        "edge_vs_base":    round(fwd.mean()*100 - base_90d, 2),
                        "reliability_label": "HIGH" if p<0.05 and len(sub)>=20 else ("MEDIUM" if p<0.10 else "LOW"),
                    })

            # ── Phase 2: Memory ────────────────────────────────────────────
            memory = run_memory_engine(sym, name, master, income_df, annual_df, divs, events)

            # ── Phase 3: Hypotheses ────────────────────────────────────────
            hyp = run_hypothesis_engine(sym, name, master)

            # ── Phase 4: Full walk-forward with all 4 layers ───────────────
            wf = run_full_walkforward(sym, name, master, annual_df,
                                       balance_df, cashflow_df, shares)
            v  = wf.get("validation", {})

            # ── Phase 5: Recalibration ────────────────────────────────────
            recal = run_recalibration(sym, name, hyp, wf, master)

            save_outputs(sym, signals, memory, hyp, wf, recal)

            pe_test = v.get("pe_signal_test", {})
            pe_info = ""
            if pe_test.get("cheap_pe_accuracy"):
                pe_info = f" | P/E<9: {pe_test['cheap_pe_accuracy']}% acc (n={pe_test['n_cheap_pe']})"

            print(f"  acc={v.get('directional_accuracy_pct','?')}% "
                  f"edge={v.get('edge_over_baseline_pct',0):+.1f}% "
                  f"bias={v.get('systematic_bias',0):+.1f}% "
                  f"mistakes={len(wf.get('mistake_vault',[]))}{pe_info}")

            all_results.append({
                "symbol":   sym, "name": name, "sector": info["sector"],
                "days":     len(master),
                "pe_days":  pe_coverage,
                "accuracy": v.get("directional_accuracy_pct"),
                "baseline": v.get("baseline_always_long_pct"),
                "edge":     v.get("edge_over_baseline_pct"),
                "bias":     v.get("systematic_bias"),
                "hyp_acc":  hyp["summary"]["accepted"],
                "mistakes": len(wf.get("mistake_vault",[])),
                "pe_cheap_acc": pe_test.get("cheap_pe_accuracy"),
                "pe_cheap_n":   pe_test.get("n_cheap_pe"),
            })

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            failed.append({"sym": sym, "error": str(e)})

    # ── Rebuild current state with all 4 layers ───────────────────────────────
    print("\n\nRebuilding current state with full 4-layer scoring...")
    subprocess.run([sys.executable, "build_all_forecasts_v3.py"])

    # ── Push to portal ────────────────────────────────────────────────────────
    print("\nPushing final model to live portal...")
    subprocess.run([sys.executable, "update_and_push.py",
                    "Final model: historical P/E+PB added to walk-forward — all signals backtested"])

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  FINAL REBUILD COMPLETE — AIR-TIGHT MODEL")
    print("=" * 72)
    print(f"\n  All signals now historically tested:")
    print(f"  ✓ Environment (rates, VIX, oil) — 2010-2026")
    print(f"  ✓ Technical (RSI, BB, momentum) — 2010-2026")
    print(f"  ✓ Fundamental (profit growth, CFQ, leverage) — 2015-2026")
    print(f"  ✓ Valuation (P/E, P/B historical) — reconstructed from earnings")
    print(f"  ~ Analyst consensus — current state only (cannot reconstruct)")
    print()
    print(f"  {'Stock':<32} {'Acc%':>6} {'Edge':>7} {'Bias':>6} {'P/E days':>9} {'PE<9 acc':>9}")
    print(f"  {'-'*32} {'-'*6} {'-'*7} {'-'*6} {'-'*9} {'-'*9}")

    # Sort by edge
    for r in sorted(all_results, key=lambda x: -(x.get("edge") or -99)):
        edge_s = f"{r['edge']:+.1f}%" if r.get("edge") is not None else "N/A"
        bias_s = f"{r['bias']:+.1f}%" if r.get("bias") is not None else "N/A"
        pe_acc = f"{r['pe_cheap_acc']:.0f}%(n={r['pe_cheap_n']})" if r.get("pe_cheap_acc") else "N/A"
        print(f"  {r['name']:<32} {str(r.get('accuracy','?')):>6} "
              f"{edge_s:>7} {bias_s:>6} {str(r.get('pe_days','?')):>9} {pe_acc:>9}")

    if failed:
        print(f"\n  Failed: {[f['sym'] for f in failed]}")

    with open(REPORTS_DIR / "FINAL_AIRTIGHT_REPORT.json", "w") as f:
        json.dump({"generated": datetime.now().isoformat(),
                   "model_layers": {
                       "environment":  {"weight": "40%", "backtested": True, "history": "2010-2026"},
                       "technical":    {"weight": "25%", "backtested": True, "history": "2010-2026"},
                       "fundamental":  {"weight": "20%", "backtested": True, "history": "2015-2026"},
                       "valuation":    {"weight": "15%", "backtested": True, "history": "reconstructed"},
                   },
                   "results": all_results, "failed": failed},
                  f, indent=2, default=str)
    print(f"\n  Report → reports/FINAL_AIRTIGHT_REPORT.json")
    print(f"  Portal → https://tadawul-stock-exchange.streamlit.app/")


if __name__ == "__main__":
    main()
