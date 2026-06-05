"""
PHASE 4 — WALK-FORWARD PREDICTION + MISTAKE VAULT
===================================================
Simulates predictions at quarterly intervals from 2016 to present.
At every prediction point, only data available at that date is used.
Compares predictions against actual outcomes.
Stores every large error in the Mistake Vault.

Run:  python phases/phase4_walkforward.py
"""

import sys
import json
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import (
    DATA_RAW, DATA_PROCESSED, MEMORY_DIR, REPORTS_DIR,
    WALKFORWARD_START, WALKFORWARD_STEP_DAYS,
)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def load_master():
    return pd.read_csv(DATA_PROCESSED / "master_dataset.csv", index_col=0, parse_dates=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE  (4-layer composite)
# ══════════════════════════════════════════════════════════════════════════════

def score_environment(row: pd.Series) -> tuple:
    """Returns (score 0-100, signals dict)."""
    signals = {}
    score   = 50  # neutral start

    # Rate regime
    regime = row.get("rate_regime", "stable")
    if regime == "stable":
        score += 20
        signals["rate_regime"] = "stable (+20)"
    elif regime == "rising":
        score -= 25
        signals["rate_regime"] = "rising (-25)"
    elif regime == "falling":
        score -= 5
        signals["rate_regime"] = "falling (-5)"

    # VIX
    vix = row.get("vix", 20)
    if vix > 30:
        score += 15
        signals["vix"] = f"{vix:.1f} >30 fear=opportunity (+15)"
    elif vix > 20:
        score += 5
        signals["vix"] = f"{vix:.1f} elevated (+5)"
    elif vix < 15:
        score -= 5
        signals["vix"] = f"{vix:.1f} calm (-5)"

    # Oil
    if row.get("oil_bull", 0) == 1:
        score += 10
        signals["oil"] = "uptrend (+10)"
    elif row.get("oil_bear", 0) == 1:
        score -= 5
        signals["oil"] = "downtrend (-5)"

    # TASI trend
    tasi_20d = row.get("tasi_20d", 0)
    if not pd.isna(tasi_20d):
        if tasi_20d > 0.05:
            score += 5
            signals["tasi"] = f"uptrend {tasi_20d:.1%} (+5)"
        elif tasi_20d < -0.05:
            score -= 5
            signals["tasi"] = f"downtrend {tasi_20d:.1%} (-5)"

    return max(0, min(100, score)), signals


def score_technical(row: pd.Series) -> tuple:
    """Returns (score 0-100, signals dict)."""
    signals = {}
    score   = 50

    # RSI
    rsi = row.get("rsi", 50)
    if not pd.isna(rsi):
        if rsi < 20:
            score += 20
            signals["rsi"] = f"{rsi:.1f} extreme oversold (+20)"
        elif rsi < 30:
            score += 15
            signals["rsi"] = f"{rsi:.1f} oversold (+15)"
        elif rsi < 40:
            score += 8
            signals["rsi"] = f"{rsi:.1f} low (+8)"
        elif rsi > 80:
            score -= 20
            signals["rsi"] = f"{rsi:.1f} extreme overbought (-20)"
        elif rsi > 70:
            score -= 12
            signals["rsi"] = f"{rsi:.1f} overbought (-12)"

    # Bollinger Band
    bb = row.get("bb_pct", 0.5)
    if not pd.isna(bb):
        if bb < 0.05:
            score += 20
            signals["bb"] = f"near lower band {bb:.2f} (+20)"
        elif bb < 0.15:
            score += 10
            signals["bb"] = f"low {bb:.2f} (+10)"
        elif bb > 0.95:
            score -= 15
            signals["bb"] = f"near upper band {bb:.2f} (-15)"

    # Momentum (mean-reversion)
    ret20 = row.get("ret_20d", 0)
    if not pd.isna(ret20):
        if ret20 > 0.15:
            score -= 20
            signals["momentum"] = f"{ret20:.1%} strong rally=caution (-20)"
        elif ret20 > 0.10:
            score -= 10
            signals["momentum"] = f"{ret20:.1%} rally (-10)"
        elif ret20 < -0.15:
            score += 15
            signals["momentum"] = f"{ret20:.1%} sharp drop=opportunity (+15)"
        elif ret20 < -0.10:
            score += 8
            signals["momentum"] = f"{ret20:.1%} drop (+8)"

    # MA alignment
    above_all = (row.get("above_ma20", 1) == 0 and row.get("above_ma50", 1) == 0
                 and row.get("above_ma200", 1) == 0)
    below_all = (row.get("above_ma20", 0) == 1 and row.get("above_ma50", 0) == 1
                 and row.get("above_ma200", 0) == 1)
    if above_all:
        score += 8
        signals["ma"] = "price below all MAs (+8)"
    elif below_all:
        score -= 8
        signals["ma"] = "price above all MAs (-8)"

    return max(0, min(100, score)), signals


def score_fundamental(row: pd.Series, annual: pd.DataFrame) -> tuple:
    """Returns (score 0-100, signals dict). Uses only data known at prediction date."""
    signals = {}
    score   = 50

    # Find most recent annual report before prediction date
    pred_date = row.name if hasattr(row, "name") else datetime.now()
    known = annual[annual["report_date"] < pred_date]
    if known.empty:
        return score, {"note": "no fundamental data before prediction date"}

    latest = known.iloc[-1]
    ni_yoy = latest.get("ni_yoy", None)

    if ni_yoy is not None and not pd.isna(ni_yoy):
        if ni_yoy > 20:
            score += 15
            signals["net_income_growth"] = f"+{ni_yoy:.1f}% YoY strong (+15)"
        elif ni_yoy > 10:
            score += 8
            signals["net_income_growth"] = f"+{ni_yoy:.1f}% YoY solid (+8)"
        elif ni_yoy > 0:
            score += 2
            signals["net_income_growth"] = f"+{ni_yoy:.1f}% YoY weak (+2)"
        elif ni_yoy < -10:
            score -= 15
            signals["net_income_growth"] = f"{ni_yoy:.1f}% YoY declining (-15)"

    return max(0, min(100, score)), signals


def composite_score(env_s, tech_s, fund_s, weights) -> float:
    return (
        env_s  * weights["environment"] +
        tech_s * weights["technical"] +
        fund_s * weights["fundamental"]
    )


def generate_forecast(price: float, composite: float, rate_regime: str,
                      env_s: float, tech_s: float) -> dict:
    """Generate base/bull/bear targets and confidence."""

    # Base return estimate from composite score
    # Calibrated from regime analysis:
    # Score > 70 → historically +7-10% over 90d
    # Score 50-70 → +3-6%
    # Score < 50 → 0 to -5%
    if composite > 70:
        base_90d = 0.075
    elif composite > 60:
        base_90d = 0.050
    elif composite > 50:
        base_90d = 0.030
    elif composite > 40:
        base_90d = 0.010
    else:
        base_90d = -0.030

    # Rate regime adjustment
    if rate_regime == "rising":
        base_90d -= 0.05
    elif rate_regime == "falling":
        base_90d -= 0.01

    base_30d = base_90d * 0.4
    base_60d = base_90d * 0.7

    # Confidence: based on how strong and aligned the signals are
    signal_strength = abs(composite - 50) / 50  # 0 = neutral, 1 = max conviction
    agreement = 1 - abs(env_s - tech_s) / 100    # 1 = signals agree, 0 = diverge
    confidence = int(40 + signal_strength * 35 + agreement * 25)
    confidence = max(20, min(90, confidence))

    spread = 0.05 + (1 - signal_strength) * 0.05

    return {
        "price_at_forecast":    round(price, 2),
        "composite_score":      round(composite, 1),
        "env_score":            round(env_s, 1),
        "tech_score":           round(tech_s, 1),
        "base_30d_pct":         round(base_30d * 100, 1),
        "base_60d_pct":         round(base_60d * 100, 1),
        "base_90d_pct":         round(base_90d * 100, 1),
        "target_30d":           round(price * (1 + base_30d), 2),
        "target_60d":           round(price * (1 + base_60d), 2),
        "target_90d":           round(price * (1 + base_90d), 2),
        "bull_90d_pct":         round((base_90d + spread) * 100, 1),
        "bear_90d_pct":         round((base_90d - spread) * 100, 1),
        "bull_target_90d":      round(price * (1 + base_90d + spread), 2),
        "bear_target_90d":      round(price * (1 + base_90d - spread), 2),
        "confidence":           confidence,
        "confidence_label":     "HIGH" if confidence >= 65 else ("MEDIUM" if confidence >= 45 else "LOW"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_walkforward(master: pd.DataFrame, annual: pd.DataFrame) -> tuple:
    """Returns (predictions list, mistake_vault list)."""

    weights = {"environment": 0.40, "technical": 0.35, "fundamental": 0.25}

    predictions  = []
    mistake_vault = []

    # Generate prediction dates (quarterly from 2016)
    start = pd.Timestamp(WALKFORWARD_START)
    end   = master.index[-1] - pd.Timedelta(days=91)
    pred_dates = []
    d = start
    while d < end:
        # Find next available trading day
        avail = master.index[master.index >= d]
        if len(avail) > 0:
            pred_dates.append(avail[0])
        d += pd.Timedelta(days=WALKFORWARD_STEP_DAYS)

    print(f"  Running {len(pred_dates)} quarterly predictions ({WALKFORWARD_START} → {end.date()})...")

    for pred_date in pred_dates:
        row = master.loc[pred_date]

        # Only use data known at prediction date — no look-ahead
        env_score,  env_sigs  = score_environment(row)
        tech_score, tech_sigs = score_technical(row)
        fund_score, fund_sigs = score_fundamental(row, annual)

        comp = composite_score(env_score, tech_score, fund_score, weights)
        fc   = generate_forecast(row["close"], comp, row.get("rate_regime","stable"),
                                 env_score, tech_score)

        # Get actual outcomes
        future = master.index[master.index > pred_date]
        actual = {}
        for w in [30, 60, 90]:
            if len(future) >= w:
                p_future = master["close"].iloc[master.index.get_loc(future[0]) + w - 1]
                actual[f"actual_{w}d_pct"] = round((p_future / row["close"] - 1) * 100, 1)
                actual[f"actual_{w}d_price"] = round(p_future, 2)
                actual[f"hit_target_{w}d"] = (
                    actual[f"actual_{w}d_pct"] > 0
                    if fc[f"base_{w}d_pct"] > 0
                    else actual[f"actual_{w}d_pct"] < 0
                )

        pred = {
            "prediction_date": pred_date.date().isoformat(),
            "price":           round(row["close"], 2),
            "rate_regime":     row.get("rate_regime", "?"),
            "rsi":             round(row.get("rsi", 50), 1),
            "vix":             round(row.get("vix", 20), 1),
            "env_score":       round(env_score, 1),
            "tech_score":      round(tech_score, 1),
            "fund_score":      round(fund_score, 1),
            "composite":       round(comp, 1),
            **fc,
            **actual,
            "env_signals":     env_sigs,
            "tech_signals":    tech_sigs,
            "fund_signals":    fund_sigs,
        }
        predictions.append(pred)

        # Flag large errors for mistake vault
        err_90d = abs(actual.get("actual_90d_pct", 0) - fc["base_90d_pct"])
        if err_90d > 12 and "actual_90d_pct" in actual:
            mistake_vault.append({
                "prediction_date":  pred_date.date().isoformat(),
                "predicted_90d":    fc["base_90d_pct"],
                "actual_90d":       actual["actual_90d_pct"],
                "error":            round(err_90d, 1),
                "composite_score":  round(comp, 1),
                "rate_regime":      row.get("rate_regime", "?"),
                "rsi":              round(row.get("rsi", 50), 1),
                "vix":              round(row.get("vix", 20), 1),
                "direction_correct":actual["actual_90d_pct"] > 0 and fc["base_90d_pct"] > 0
                    or actual["actual_90d_pct"] < 0 and fc["base_90d_pct"] < 0,
                "root_cause":       classify_mistake(pred, fc, actual, row),
            })

    return predictions, mistake_vault


def classify_mistake(pred: dict, fc: dict, actual: dict, row: pd.Series) -> str:
    """Guess the likely root cause of a large error."""
    act = actual.get("actual_90d_pct", 0)
    pred_val = fc["base_90d_pct"]
    err = act - pred_val

    if row.get("vix", 20) < 20 and act < -10:
        return "Unexpected macro shock (low VIX at prediction but major event followed)"
    if row.get("rate_regime") == "rising" and act > 10:
        return "Rate sensitivity overestimated — market priced in positives despite rising rates"
    if row.get("rsi", 50) < 30 and act < 0:
        return "Mean-reversion signal failed — downtrend continued further than expected"
    if err > 15:
        return "Model underestimated upside — large upside move not anticipated by signals"
    if err < -15:
        return "Model underestimated downside — unexpected negative event or sector weakness"
    if row.get("rate_regime") == "stable" and act < -5:
        return "Stable rate regime did not protect — external shock or company-specific event"
    return "Mixed signals — multiple factors cancelled each other out"


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION AND BASELINE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def validate_predictions(predictions: list) -> dict:
    df = pd.DataFrame(predictions)
    df = df[df["actual_90d_pct"].notna()]

    if len(df) < 5:
        return {"error": "Insufficient predictions with known outcomes"}

    # Directional accuracy
    df["predicted_direction"] = (df["base_90d_pct"] > 0).astype(int)
    df["actual_direction"]    = (df["actual_90d_pct"] > 0).astype(int)
    df["correct_direction"]   = (df["predicted_direction"] == df["actual_direction"]).astype(int)

    dir_accuracy = df["correct_direction"].mean()

    # Baselines
    always_long_accuracy = df["actual_direction"].mean()

    # MAE and RMSE
    mae  = (df["actual_90d_pct"] - df["base_90d_pct"]).abs().mean()
    rmse = np.sqrt(((df["actual_90d_pct"] - df["base_90d_pct"]) ** 2).mean())

    # By regime
    regime_results = {}
    for regime in ["rising", "stable", "falling"]:
        sub = df[df["rate_regime"] == regime]
        if len(sub) >= 3:
            regime_results[regime] = {
                "n":              len(sub),
                "dir_accuracy":   round(sub["correct_direction"].mean() * 100, 1),
                "avg_error":      round((sub["actual_90d_pct"] - sub["base_90d_pct"]).mean(), 1),
            }

    # Confidence calibration
    high_conf   = df[df["confidence"] >= 65]
    medium_conf = df[(df["confidence"] >= 45) & (df["confidence"] < 65)]
    low_conf    = df[df["confidence"] < 45]

    return {
        "n_predictions":            len(df),
        "directional_accuracy_pct": round(dir_accuracy * 100, 1),
        "baseline_always_long_pct": round(always_long_accuracy * 100, 1),
        "edge_over_baseline_pct":   round((dir_accuracy - always_long_accuracy) * 100, 1),
        "mae_pct":                  round(mae, 2),
        "rmse_pct":                 round(rmse, 2),
        "avg_predicted_90d":        round(df["base_90d_pct"].mean(), 2),
        "avg_actual_90d":           round(df["actual_90d_pct"].mean(), 2),
        "regime_breakdown":         regime_results,
        "confidence_calibration": {
            "high_conf_n":     len(high_conf),
            "high_conf_acc":   round(high_conf["correct_direction"].mean() * 100, 1) if len(high_conf) > 3 else None,
            "medium_conf_n":   len(medium_conf),
            "medium_conf_acc": round(medium_conf["correct_direction"].mean() * 100, 1) if len(medium_conf) > 3 else None,
            "low_conf_n":      len(low_conf),
            "low_conf_acc":    round(low_conf["correct_direction"].mean() * 100, 1) if len(low_conf) > 3 else None,
        },
        "p_value": round(stats.ttest_1samp(
            (df["actual_90d_pct"] - df["base_90d_pct"]).values, 0)[1], 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PRINT REPORTS
# ══════════════════════════════════════════════════════════════════════════════

def print_backtest_report(validation: dict, mistake_vault: list):
    print("\n" + "═" * 80)
    print("  PREDICTION BACKTEST REPORT — Al Rajhi Bank (1120.SR)")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═" * 80)

    v = validation
    print(f"\n  Predictions tested:        {v['n_predictions']}")
    print(f"  Directional accuracy:      {v['directional_accuracy_pct']}%")
    print(f"  Baseline (always bullish): {v['baseline_always_long_pct']}%")
    print(f"  Edge over baseline:        {v['edge_over_baseline_pct']:+.1f}%")
    print(f"  Mean absolute error:       {v['mae_pct']}% (90d)")
    print(f"  RMSE:                      {v['rmse_pct']}%")
    print(f"  Avg predicted 90d:         {v['avg_predicted_90d']}%")
    print(f"  Avg actual 90d:            {v['avg_actual_90d']}%")

    print(f"\n  Regime breakdown:")
    for regime, r in v.get("regime_breakdown", {}).items():
        print(f"    {regime:<10} n={r['n']:<4} accuracy={r['dir_accuracy']}%  avg_error={r['avg_error']:+.1f}%")

    cc = v.get("confidence_calibration", {})
    print(f"\n  Confidence calibration:")
    print(f"    High conf   n={cc.get('high_conf_n','?'):<4} accuracy={cc.get('high_conf_acc','?')}%")
    print(f"    Medium conf n={cc.get('medium_conf_n','?'):<4} accuracy={cc.get('medium_conf_acc','?')}%")
    print(f"    Low conf    n={cc.get('low_conf_n','?'):<4} accuracy={cc.get('low_conf_acc','?')}%")

    print(f"\n  {'─' * 60}")
    print(f"  MISTAKE VAULT ({len(mistake_vault)} large errors recorded)")
    print(f"  {'─' * 60}")
    for m in sorted(mistake_vault, key=lambda x: -x["error"])[:10]:
        dir_ok = "DIR✓" if m.get("direction_correct") else "DIR✗"
        print(f"\n  {m['prediction_date']} | {dir_ok} | predicted={m['predicted_90d']:+.1f}% | actual={m['actual_90d']:+.1f}% | error={m['error']:.1f}%")
        print(f"    RSI={m['rsi']} | VIX={m['vix']} | regime={m['rate_regime']}")
        print(f"    Root cause: {m['root_cause']}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# CURRENT FORECAST
# ══════════════════════════════════════════════════════════════════════════════

def generate_current_forecast(master: pd.DataFrame, annual: pd.DataFrame) -> dict:
    latest = master.iloc[-1]
    weights = {"environment": 0.40, "technical": 0.35, "fundamental": 0.25}

    env_s,  env_sigs  = score_environment(latest)
    tech_s, tech_sigs = score_technical(latest)
    fund_s, fund_sigs = score_fundamental(latest, annual)
    comp = composite_score(env_s, tech_s, fund_s, weights)
    fc   = generate_forecast(latest["close"], comp, latest.get("rate_regime","stable"), env_s, tech_s)

    # Find similar historical setups for context
    similar = []
    df = master.dropna(subset=["fwd_90d"])
    env_match  = (abs(df["repo_rate"] - latest["repo_rate"]) < 0.5)
    rsi_match  = (abs(df["rsi"]       - latest["rsi"])       < 8)
    regime_match = (df["rate_regime"] == latest.get("rate_regime","stable"))
    similar_mask = env_match & rsi_match & regime_match
    similar_df   = df[similar_mask].dropna(subset=["fwd_90d"])
    if len(similar_df) > 3:
        similar = {
            "n_similar_setups":  len(similar_df),
            "avg_90d_return":    round(similar_df["fwd_90d"].mean() * 100, 1),
            "pct_positive_90d":  round((similar_df["fwd_90d"] > 0).mean() * 100, 1),
            "date_range":        f"{similar_df.index[0].date()} → {similar_df.index[-1].date()}",
        }

    forecast = {
        "forecast_date":  date.today().isoformat(),
        "price":          round(latest["close"], 2),
        "rsi":            round(latest["rsi"], 1),
        "vix":            round(latest["vix"], 1),
        "rate_regime":    latest.get("rate_regime", "?"),
        "repo_rate":      round(latest["repo_rate"], 2),
        "oil":            round(latest["oil"], 2),
        "env_score":      round(env_s, 1),
        "tech_score":     round(tech_s, 1),
        "fund_score":     round(fund_s, 1),
        "composite":      round(comp, 1),
        **fc,
        "env_signals":    env_sigs,
        "tech_signals":   tech_sigs,
        "fund_signals":   fund_sigs,
        "similar_setups": similar,
        "main_drivers":   _top_drivers(env_sigs, tech_sigs, fund_sigs),
        "key_risks":      _key_risks(latest),
    }
    return forecast


def _top_drivers(env_sigs, tech_sigs, fund_sigs) -> list:
    drivers = []
    for sigs, label in [(env_sigs, "Environment"), (tech_sigs, "Technical"), (fund_sigs, "Fundamental")]:
        for k, v in sigs.items():
            drivers.append(f"[{label}] {k}: {v}")
    return drivers[:6]


def _key_risks(row: pd.Series) -> list:
    risks = []
    if row.get("rate_regime") == "rising":
        risks.append("Rate regime: rising rates — historically worst environment for Al Rajhi")
    if row.get("vix", 20) < 15:
        risks.append("VIX very low — markets may be complacent, downside risk underpriced")
    if row.get("rsi", 50) > 70:
        risks.append("RSI overbought — mean reversion risk elevated")
    if row.get("ret_20d", 0) > 0.10:
        risks.append("Strong recent rally — historically followed by underperformance")
    if not risks:
        risks.append("No major technical risk flags at current levels")
    return risks


def print_current_forecast(fc: dict):
    print("\n" + "═" * 80)
    print("  CURRENT FORECAST — Al Rajhi Bank (1120.SR)")
    print(f"  As of: {fc['forecast_date']}")
    print("═" * 80)
    print(f"\n  Price:        {fc['price']} SAR")
    print(f"  RSI:          {fc['rsi']}")
    print(f"  VIX:          {fc['vix']}")
    print(f"  Rate regime:  {fc['rate_regime'].upper()} @ {fc['repo_rate']}%")
    print(f"  Oil (Brent):  ${fc['oil']}")
    print(f"\n  SCORES:")
    print(f"    Environment:  {fc['env_score']}/100")
    print(f"    Technical:    {fc['tech_score']}/100")
    print(f"    Fundamental:  {fc['fund_score']}/100")
    print(f"    COMPOSITE:    {fc['composite']}/100")
    print(f"\n  FORECAST:")
    print(f"    30d target:  {fc['target_30d']} SAR  ({fc['base_30d_pct']:+.1f}%)")
    print(f"    60d target:  {fc['target_60d']} SAR  ({fc['base_60d_pct']:+.1f}%)")
    print(f"    90d target:  {fc['target_90d']} SAR  ({fc['base_90d_pct']:+.1f}%)")
    print(f"    Bull case:   {fc['bull_target_90d']} SAR  ({fc['bull_90d_pct']:+.1f}%)")
    print(f"    Bear case:   {fc['bear_target_90d']} SAR  ({fc['bear_90d_pct']:+.1f}%)")
    print(f"    Confidence:  {fc['confidence']}/100 ({fc['confidence_label']})")
    print(f"\n  MAIN DRIVERS:")
    for d in fc.get("main_drivers", []):
        print(f"    • {d}")
    print(f"\n  KEY RISKS:")
    for r in fc.get("key_risks", []):
        print(f"    ⚠ {r}")
    if fc.get("similar_setups"):
        s = fc["similar_setups"]
        print(f"\n  SIMILAR HISTORICAL SETUPS:")
        print(f"    n={s['n_similar_setups']} periods | avg 90d={s['avg_90d_return']}% | win rate={s['pct_positive_90d']}%")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\nSTOCK MEMORY ENGINE v2 — PHASE 4: WALK-FORWARD + MISTAKE VAULT")
    print(f"Stock: Al Rajhi Bank (1120.SR)")
    print(f"Date : {date.today()}")
    print("─" * 60)

    master = load_master()

    income = pd.read_csv(DATA_RAW / "alrajhi_income_quarterly.csv", parse_dates=["report_date"])
    income = income.sort_values("report_date")
    annual = income[income["report_date"].dt.month == 12].copy()
    annual["ni_yoy"] = annual["net_income"].pct_change() * 100

    print("\nStep 1: Running walk-forward predictions...")
    predictions, mistake_vault = run_walkforward(master, annual)

    print(f"\nStep 2: Validating {len(predictions)} predictions...")
    validation = validate_predictions(predictions)

    print("\nStep 3: Generating current forecast...")
    current_fc = generate_current_forecast(master, annual)

    # Save all outputs
    with open(REPORTS_DIR / "PREDICTION_BACKTEST_REPORT.json", "w", encoding="utf-8") as f:
        json.dump({"generated": datetime.now().isoformat(),
                   "validation": validation, "predictions": predictions}, f, indent=2, default=str)

    with open(MEMORY_DIR / "mistake_vault.json", "w", encoding="utf-8") as f:
        json.dump({"generated": datetime.now().isoformat(),
                   "mistakes": mistake_vault}, f, indent=2, default=str)

    with open(MEMORY_DIR / "current_forecast.json", "w", encoding="utf-8") as f:
        json.dump(current_fc, f, indent=2, default=str)

    print_backtest_report(validation, mistake_vault)
    print_current_forecast(current_fc)

    print(f"  Backtest report  → reports/PREDICTION_BACKTEST_REPORT.json")
    print(f"  Mistake Vault    → memory/mistake_vault.json")
    print(f"  Current forecast → memory/current_forecast.json")
    print(f"\n  Next step: python phases/phase5_recalibration.py")


if __name__ == "__main__":
    main()
