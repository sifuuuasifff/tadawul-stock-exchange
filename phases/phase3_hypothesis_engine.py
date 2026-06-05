"""
PHASE 3 — HYPOTHESIS ENGINE
============================
Tests explicit hypotheses about Al Rajhi using historical evidence.
Each hypothesis is ACCEPTED, REJECTED, or INCONCLUSIVE with evidence.

Run:  python phases/phase3_hypothesis_engine.py
"""

import sys
import json
import warnings
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import DATA_RAW, DATA_PROCESSED, MEMORY_DIR, REPORTS_DIR

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_master():
    return pd.read_csv(DATA_PROCESSED / "master_dataset.csv", index_col=0, parse_dates=True)


def test_hypothesis(h_id: str, statement: str, master: pd.DataFrame,
                    group_a: pd.Series, group_b: pd.Series = None,
                    metric: str = "fwd_90d", min_n: int = 10) -> dict:
    """
    Tests one hypothesis.
    group_a = condition when hypothesis is TRUE
    group_b = baseline (if None, uses full dataset)
    Verdict: ACCEPTED / REJECTED / INCONCLUSIVE
    """
    valid  = master.dropna(subset=[metric])
    sub_a  = valid[group_a.reindex(valid.index, fill_value=False)]
    sub_b  = valid[group_b.reindex(valid.index, fill_value=False)] if group_b is not None else valid

    result = {
        "id":           h_id,
        "hypothesis":   statement,
        "metric":       metric,
        "n_a":          len(sub_a),
        "n_b":          len(sub_b),
    }

    if len(sub_a) < min_n:
        result.update({"verdict": "INCONCLUSIVE", "reason": f"Insufficient sample (n={len(sub_a)}, need {min_n})"})
        return result

    a_vals = sub_a[metric].dropna()
    b_vals = sub_b[metric].dropna()

    result["avg_a"]     = round(a_vals.mean() * 100, 2)
    result["avg_b"]     = round(b_vals.mean() * 100, 2)
    result["diff"]      = round(result["avg_a"] - result["avg_b"], 2)
    result["win_pct_a"] = round((a_vals > 0).mean() * 100, 1)
    result["win_pct_b"] = round((b_vals > 0).mean() * 100, 1)

    # Two-sample t-test
    t, p = stats.ttest_ind(a_vals, b_vals, equal_var=False)
    result["t_stat"]  = round(t, 3)
    result["p_value"] = round(p, 3)

    # Verdict
    meaningful = abs(result["diff"]) > 2.0
    if p < 0.05 and meaningful:
        direction = "hypothesis CONFIRMED" if result["diff"] > 0 else "hypothesis REVERSED (opposite true)"
        result["verdict"] = "ACCEPTED" if result["diff"] > 0 else "REJECTED"
        result["reason"]  = f"p={p:.3f} | Group A avg {result['avg_a']}% vs Group B {result['avg_b']}% | {direction}"
    elif p < 0.10 and meaningful:
        result["verdict"] = "INCONCLUSIVE"
        result["reason"]  = f"Trend in expected direction but marginal significance (p={p:.3f})"
    elif not meaningful:
        result["verdict"] = "REJECTED"
        result["reason"]  = f"Difference too small to matter ({result['diff']}%) even if p={p:.3f}"
    else:
        result["verdict"] = "INCONCLUSIVE"
        result["reason"]  = f"p={p:.3f} — not significant at 5% level"

    return result


def run_all_hypotheses(master: pd.DataFrame) -> list:
    results = []

    baseline = pd.Series(True, index=master.index)

    # ── HYPOTHESIS 1 ──────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H01", "Rising Saudi rates BENEFIT Al Rajhi (bank profits from higher rates)",
        master,
        group_a = master["rate_regime"] == "rising",
        group_b = master["rate_regime"] == "stable",
    ))

    # ── HYPOTHESIS 2 ──────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H02", "Stable Saudi rates are the best environment for Al Rajhi stock returns",
        master,
        group_a = master["rate_regime"] == "stable",
        group_b = (master["rate_regime"] == "rising") | (master["rate_regime"] == "falling"),
    ))

    # ── HYPOTHESIS 3 ──────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H03", "RSI below 30 (oversold) produces significantly better 90d returns than baseline",
        master,
        group_a = master["rsi"] < 30,
    ))

    # ── HYPOTHESIS 4 ──────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H04", "RSI above 70 (overbought) produces significantly worse 90d returns than baseline",
        master,
        group_a = master["rsi"] > 70,
    ))

    # ── HYPOTHESIS 5 ──────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H05", "High global fear (VIX > 30) creates buying opportunities for Al Rajhi",
        master,
        group_a = master["vix"] > 30,
    ))

    # ── HYPOTHESIS 6 ──────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H06", "Oil price rising improves Al Rajhi forward returns vs oil falling",
        master,
        group_a = master["oil_bull"] == 1,
        group_b = master["oil_bear"] == 1,
    ))

    # ── HYPOTHESIS 7 ──────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H07", "Strong 20-day price momentum (>10%) predicts NEGATIVE 90d returns (mean reversion)",
        master,
        group_a = master["ret_20d"] > 0.10,
        group_b = (master["ret_20d"] >= -0.05) & (master["ret_20d"] <= 0.05),
    ))

    # ── HYPOTHESIS 8 ──────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H08", "Price near Bollinger Band lower bound predicts positive 30d returns",
        master,
        group_a = master["bb_pct"] < 0.05,
        metric  = "fwd_30d",
    ))

    # ── HYPOTHESIS 9 ──────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H09", "Price near Bollinger Band upper bound predicts negative 30d returns",
        master,
        group_a = master["bb_pct"] > 0.95,
        metric  = "fwd_30d",
    ))

    # ── HYPOTHESIS 10 ─────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H10", "Outperforming TASI by >5% over 20 days predicts underperformance over next 90 days",
        master,
        group_a = master["rs_vs_tasi_20d"] > 0.05,
        group_b = (master["rs_vs_tasi_20d"] >= -0.02) & (master["rs_vs_tasi_20d"] <= 0.02),
    ))

    # ── HYPOTHESIS 11 ─────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H11", "Underperforming TASI by >5% over 20 days predicts outperformance over next 90 days",
        master,
        group_a = master["rs_vs_tasi_20d"] < -0.05,
        group_b = (master["rs_vs_tasi_20d"] >= -0.02) & (master["rs_vs_tasi_20d"] <= 0.02),
    ))

    # ── HYPOTHESIS 12 ─────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H12", "Al Rajhi below ALL moving averages (MA20, MA50, MA200) predicts positive 90d returns",
        master,
        group_a = (master["above_ma20"]==0) & (master["above_ma50"]==0) & (master["above_ma200"]==0),
    ))

    # ── HYPOTHESIS 13 ─────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H13", "Al Rajhi above ALL moving averages predicts negative 90d returns (overbought)",
        master,
        group_a = (master["above_ma20"]==1) & (master["above_ma50"]==1) & (master["above_ma200"]==1),
    ))

    # ── HYPOTHESIS 14 ─────────────────────────────────────────────────────────
    # Oversold in stable rates vs oversold in rising rates
    results.append(test_hypothesis(
        "H14", "RSI oversold in STABLE rates regime produces better outcomes than RSI oversold in RISING rates",
        master,
        group_a = (master["rsi"] < 30) & (master["rate_regime"] == "stable"),
        group_b = (master["rsi"] < 30) & (master["rate_regime"] == "rising"),
    ))

    # ── HYPOTHESIS 15 ─────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H15", "Volume surges (>2.5x average) are predictive of 30d forward returns",
        master,
        group_a = master["vol_ratio"] > 2.5,
        metric  = "fwd_30d",
    ))

    # ── HYPOTHESIS 16 ─────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H16", "RSI oversold + VIX < 25 is a better signal than RSI oversold alone",
        master,
        group_a = (master["rsi"] < 30) & (master["vix"] < 25),
        group_b = (master["rsi"] < 30) & (master["vix"] >= 25),
    ))

    # ── HYPOTHESIS 17 ─────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H17", "TASI uptrend (>5% over 20d) improves Al Rajhi 90d returns",
        master,
        group_a = master.get("tasi_20d", pd.Series(dtype=float)) > 0.05 if "tasi_20d" in master.columns else pd.Series(False, index=master.index),
        group_b = master.get("tasi_20d", pd.Series(dtype=float)) < -0.05 if "tasi_20d" in master.columns else pd.Series(False, index=master.index),
    ))

    # ── HYPOTHESIS 18 ─────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H18", "Falling rate regime produces better 90d returns than rising rate regime",
        master,
        group_a = master["rate_regime"] == "falling",
        group_b = master["rate_regime"] == "rising",
    ))

    # ── HYPOTHESIS 19 ─────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H19", "RSI extreme oversold (<20) outperforms RSI oversold (20-30)",
        master,
        group_a = master["rsi"] < 20,
        group_b = (master["rsi"] >= 20) & (master["rsi"] < 30),
    ))

    # ── HYPOTHESIS 20 ─────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H20", "High volume with negative price day predicts further weakness (momentum) vs recovery (reversion)",
        master,
        group_a = (master["vol_ratio"] > 2.5) & (master["ret_1d"] < -0.02),
        metric  = "fwd_30d",
    ))

    # ── HYPOTHESIS 21 ─────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H21", "VIX spike-then-calm (VIX was >30, now <25) is a stronger buy signal than calm-only",
        master,
        group_a = (master["vix"].shift(10) > 30) & (master["vix"] < 25),
        group_b = (master["vix"].shift(10) <= 20) & (master["vix"] < 25),
    ))

    # ── HYPOTHESIS 22 ─────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H22", "Al Rajhi is more sensitive to TASI movements during rising rate regimes",
        master,
        group_a = master["rate_regime"] == "rising",
        group_b = master["rate_regime"] == "stable",
        metric  = "fwd_30d",
    ))

    # ── HYPOTHESIS 23 ─────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H23", "Oil bear markets hurt Al Rajhi more than oil bull markets help it (asymmetry)",
        master,
        group_a = master["oil_bear"] == 1,
        group_b = master["oil_bull"] == 1,
    ))

    # ── HYPOTHESIS 24 ─────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H24", "Falling rates + high VIX is the single worst environment for Al Rajhi",
        master,
        group_a = (master["rate_regime"] == "falling") & (master["vix"] > 30),
        group_b = (master["rate_regime"] == "stable")  & (master["vix"] < 20),
    ))

    # ── HYPOTHESIS 25 ─────────────────────────────────────────────────────────
    results.append(test_hypothesis(
        "H25", "RSI neutral zone (40-60) combined with stable rates is a reliable low-risk accumulation zone",
        master,
        group_a = (master["rsi"] >= 40) & (master["rsi"] <= 60) & (master["rate_regime"] == "stable"),
    ))

    return results


def print_hypothesis_report(results: list):
    accepted     = [r for r in results if r["verdict"] == "ACCEPTED"]
    rejected     = [r for r in results if r["verdict"] == "REJECTED"]
    inconclusive = [r for r in results if r["verdict"] == "INCONCLUSIVE"]

    print("\n" + "═" * 80)
    print("  HYPOTHESIS REPORT — Al Rajhi Bank (1120.SR)")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═" * 80)
    print(f"\n  Total hypotheses: {len(results)}")
    print(f"  Accepted:         {len(accepted)}")
    print(f"  Rejected:         {len(rejected)}")
    print(f"  Inconclusive:     {len(inconclusive)}")

    for verdict, group in [("ACCEPTED", accepted), ("REJECTED", rejected), ("INCONCLUSIVE", inconclusive)]:
        if not group:
            continue
        icon = {"ACCEPTED": "✓", "REJECTED": "✗", "INCONCLUSIVE": "~"}[verdict]
        print(f"\n{'─' * 70}")
        print(f"  [{icon}] {verdict} ({len(group)})")
        print(f"{'─' * 70}")
        for r in group:
            diff_str = f"Δ={r.get('diff',0):+.1f}%" if "diff" in r else ""
            print(f"\n  {r['id']}: {r['hypothesis']}")
            print(f"    {r.get('reason','')}")
            print(f"    Group A avg: {r.get('avg_a','N/A')}% | Baseline avg: {r.get('avg_b','N/A')}% | {diff_str} | n={r['n_a']}")

    print("\n" + "═" * 80)
    print("  KEY DISCOVERIES")
    print("═" * 80)
    key = [r for r in accepted if abs(r.get("diff", 0)) > 5]
    for r in sorted(key, key=lambda x: -abs(x.get("diff", 0))):
        print(f"  {r['id']}: {r['hypothesis'][:70]}")
        print(f"    Edge: {r.get('diff',0):+.1f}% | p={r.get('p_value','?')}")
    print()


def main():
    print("\nSTOCK MEMORY ENGINE v2 — PHASE 3: HYPOTHESIS ENGINE")
    print(f"Stock: Al Rajhi Bank (1120.SR)")
    print(f"Date : {date.today()}")
    print("─" * 60)

    master   = load_master()
    results  = run_all_hypotheses(master)

    # Save
    out = {
        "generated": datetime.now().isoformat(),
        "stock":     "1120.SR — Al Rajhi Bank",
        "results":   results,
        "summary": {
            "total":       len(results),
            "accepted":    len([r for r in results if r["verdict"] == "ACCEPTED"]),
            "rejected":    len([r for r in results if r["verdict"] == "REJECTED"]),
            "inconclusive":len([r for r in results if r["verdict"] == "INCONCLUSIVE"]),
        },
    }
    with open(REPORTS_DIR / "HYPOTHESIS_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print_hypothesis_report(results)
    print(f"  Report saved → reports/HYPOTHESIS_REPORT.json")
    print("  Next step: python phases/phase4_walkforward.py")


if __name__ == "__main__":
    main()
