"""
FINAL COMPLETE REBUILD
======================
1. Pull EARNINGS_SURPRISE + ANALYST_RATING_CHANGE events (before SAHMK cancel)
2. Add sector-specific macro weights (oil, rates, VIX per sector)
3. Rebuild all 5 phases for all stocks
4. Generate before/after comparison table
5. Push to portal

Run: python final_complete_rebuild.py
"""

import sys, json, time, subprocess, warnings
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import DATA_RAW, DATA_PROCESSED, MEMORY_DIR, REPORTS_DIR
from config.providers import SAHMKProvider
from shared import STOCKS
from rebuild_engine import (load_enriched_financials, score_fundamental_enriched)
from phases.full_engine import (run_signal_discovery, run_memory_engine,
                                 run_hypothesis_engine, run_recalibration,
                                 save_outputs, load_dividends, load_events)
from rebuild_with_valuation import build_historical_valuation, run_full_walkforward

DATA_RAW.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTOR-SPECIFIC MACRO WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

SECTOR_MACRO_WEIGHTS = {
    # Format: rate_bonus_stable, rate_penalty_rising, vix_fear_bonus, oil_bull_bonus
    # Positive = helps, negative = hurts

    "Banking": {
        "rate_stable":   +22,   # Banks love stable rates — highest weight
        "rate_rising":   -28,   # Rising rates hurt most (confirmed by H01)
        "rate_falling":  -8,
        "vix_high":      +15,   # Fear = buying opportunity for banks
        "vix_elevated":  +5,
        "vix_calm":      -5,
        "oil_bull":      +8,    # Indirect — Saudi economy benefits
        "oil_bear":      -5,
        "note": "Rate regime dominant. Oil secondary via Saudi economy.",
    },
    "Energy": {
        "rate_stable":   +12,   # Less rate sensitive
        "rate_rising":   -15,
        "rate_falling":  -3,
        "vix_high":      +8,    # Some fear premium but oil matters more
        "vix_elevated":  +3,
        "vix_calm":      -3,
        "oil_bull":      +25,   # Oil IS the business — very high weight
        "oil_bear":      -20,
        "note": "Oil price dominant signal. Rate secondary.",
    },
    "Petrochemicals": {
        "rate_stable":   +15,
        "rate_rising":   -18,
        "rate_falling":  -5,
        "vix_high":      +10,
        "vix_elevated":  +3,
        "vix_calm":      -3,
        "oil_bull":      +5,    # Mixed: higher oil = higher product prices BUT higher feedstock cost
        "oil_bear":      -3,    # Lower oil can help margins for some
        "note": "Oil effect is mixed — product price vs feedstock cost. Net: small positive.",
    },
    "Mining": {
        "rate_stable":   +15,
        "rate_rising":   -18,
        "rate_falling":  -4,
        "vix_high":      +12,
        "vix_elevated":  +4,
        "vix_calm":      -3,
        "oil_bull":      +12,   # Higher oil = commodity cycle bullish (correlated)
        "oil_bear":      -10,
        "note": "Commodity cycle correlated. Oil proxy for global demand.",
    },
    "Cement": {
        "rate_stable":   +18,   # Construction activity linked to credit
        "rate_rising":   -15,   # Higher rates slow construction
        "rate_falling":  -3,
        "vix_high":      +8,
        "vix_elevated":  +3,
        "vix_calm":      -2,
        "oil_bull":      +5,    # Construction activity follows oil wealth
        "oil_bear":      -5,
        "note": "Government spending / construction cycle matters. Oil via Vision 2030 spend.",
    },
    "Real Estate": {
        "rate_stable":   +18,
        "rate_rising":   -22,   # Rising rates kill mortgage demand
        "rate_falling":  +5,    # Falling rates help real estate
        "vix_high":      +8,
        "vix_elevated":  +3,
        "vix_calm":      -3,
        "oil_bull":      +10,   # Oil wealth drives property demand in Saudi
        "oil_bear":      -8,
        "note": "Rate sensitive (mortgage demand). Oil via wealth effect.",
    },
    "Healthcare": {
        "rate_stable":   +12,   # Defensive — less rate sensitive
        "rate_rising":   -10,
        "rate_falling":  -2,
        "vix_high":      +8,    # Defensive — doesn't benefit as much from fear
        "vix_elevated":  +3,
        "vix_calm":      -2,
        "oil_bull":      +3,    # Very little direct oil sensitivity
        "oil_bear":      -2,
        "note": "Defensive sector. Limited macro sensitivity. Earnings-driven.",
    },
    "Pharmaceuticals": {
        "rate_stable":   +12,
        "rate_rising":   -10,
        "rate_falling":  -2,
        "vix_high":      +5,
        "vix_elevated":  +2,
        "vix_calm":      -2,
        "oil_bull":      +2,
        "oil_bear":      -2,
        "note": "Defensive. Demand-driven. Minimal macro sensitivity.",
    },
    "Retail": {
        "rate_stable":   +15,
        "rate_rising":   -15,   # Higher rates = less consumer spending
        "rate_falling":  -3,
        "vix_high":      +8,
        "vix_elevated":  +3,
        "vix_calm":      -3,
        "oil_bull":      +5,    # Higher oil = more consumer spending via higher incomes
        "oil_bear":      -5,
        "note": "Consumer spending cycle. Oil via income effect in Saudi.",
    },
    "Food & Beverages": {
        "rate_stable":   +12,
        "rate_rising":   -12,
        "rate_falling":  -2,
        "vix_high":      +5,    # Defensive consumer staples
        "vix_elevated":  +2,
        "vix_calm":      -2,
        "oil_bull":      +3,    # Oil can raise input costs — mixed
        "oil_bear":      -2,
        "note": "Defensive staples. Limited macro sensitivity.",
    },
    "Telecom": {
        "rate_stable":   +12,
        "rate_rising":   -10,
        "rate_falling":  -2,
        "vix_high":      +5,
        "vix_elevated":  +2,
        "vix_calm":      -2,
        "oil_bull":      +3,
        "oil_bear":      -2,
        "note": "Defensive utility-like. Subscriber growth more important than macro.",
    },
    "Technology": {
        "rate_stable":   +12,
        "rate_rising":   -15,   # Rate rises hurt growth stocks
        "rate_falling":  +5,    # Rate cuts help growth
        "vix_high":      +5,
        "vix_elevated":  +2,
        "vix_calm":      -2,
        "oil_bull":      +3,
        "oil_bear":      -2,
        "note": "Growth-oriented. Rate sensitive. Low oil correlation.",
    },
    "Chemicals": {
        "rate_stable":   +15,
        "rate_rising":   -18,
        "rate_falling":  -4,
        "vix_high":      +10,
        "vix_elevated":  +3,
        "vix_calm":      -3,
        "oil_bull":      +8,    # Chemical feedstock linked to oil
        "oil_bear":      -6,
        "note": "Petrochemical adjacent. Oil feedstock sensitive.",
    },
    "Transport": {
        "rate_stable":   +15,
        "rate_rising":   -15,
        "rate_falling":  -3,
        "vix_high":      +8,
        "vix_elevated":  +3,
        "vix_calm":      -3,
        "oil_bull":      +8,    # Trade volumes linked to oil economy
        "oil_bear":      -6,
        "note": "Trade/logistics — oil economy proxy.",
    },
    "Utilities": {
        "rate_stable":   +15,
        "rate_rising":   -12,
        "rate_falling":  +3,    # Like bonds — benefits from rate cuts
        "vix_high":      +8,
        "vix_elevated":  +3,
        "vix_calm":      -3,
        "oil_bull":      +5,
        "oil_bear":      -3,
        "note": "Utility-like. Rate sensitive. Defensive.",
    },
    "Industrials": {
        "rate_stable":   +15,
        "rate_rising":   -15,
        "rate_falling":  -3,
        "vix_high":      +10,
        "vix_elevated":  +3,
        "vix_calm":      -3,
        "oil_bull":      +8,    # Industrial activity follows oil economy
        "oil_bear":      -6,
        "note": "Linked to Saudi industrial capex and Vision 2030.",
    },
    "Diversified": {
        "rate_stable":   +15,
        "rate_rising":   -18,
        "rate_falling":  -4,
        "vix_high":      +10,
        "vix_elevated":  +4,
        "vix_calm":      -4,
        "oil_bull":      +8,
        "oil_bear":      -6,
        "note": "Diversified holdings — average macro sensitivity.",
    },
    "Financial Services": {
        "rate_stable":   +18,
        "rate_rising":   -20,
        "rate_falling":  -6,
        "vix_high":      +12,
        "vix_elevated":  +4,
        "vix_calm":      -4,
        "oil_bull":      +7,
        "oil_bear":      -5,
        "note": "Financial adjacent to banking. Rate sensitive.",
    },
    "Consumer Services": {
        "rate_stable":   +12,
        "rate_rising":   -12,
        "rate_falling":  -2,
        "vix_high":      +6,
        "vix_elevated":  +2,
        "vix_calm":      -2,
        "oil_bull":      +5,
        "oil_bear":      -4,
        "note": "Consumer discretionary services. Income-driven.",
    },
    "Consumer Cyclical": {
        "rate_stable":   +14,
        "rate_rising":   -14,
        "rate_falling":  -3,
        "vix_high":      +7,
        "vix_elevated":  +3,
        "vix_calm":      -3,
        "oil_bull":      +5,
        "oil_bear":      -4,
        "note": "Consumer cyclical. Spending cycle sensitive.",
    },
}

DEFAULT_WEIGHTS = {
    "rate_stable": +20, "rate_rising": -25, "rate_falling": -5,
    "vix_high": +15, "vix_elevated": +5, "vix_calm": -5,
    "oil_bull": +10, "oil_bear": -5,
}


def score_environment_sector(row: pd.Series, sector: str) -> tuple:
    """Sector-aware environment score."""
    w    = SECTOR_MACRO_WEIGHTS.get(sector, DEFAULT_WEIGHTS)
    score= 50
    sigs = {}

    regime = row.get("rate_regime", "stable")
    if regime == "stable":
        score += w["rate_stable"]; sigs["rate"] = f"stable ({w['rate_stable']:+d})"
    elif regime == "rising":
        score += w["rate_rising"]; sigs["rate"] = f"rising ({w['rate_rising']:+d})"
    elif regime == "falling":
        score += w["rate_falling"]; sigs["rate"] = f"falling ({w['rate_falling']:+d})"

    vix = row.get("vix", 20)
    if vix > 30:
        score += w["vix_high"];     sigs["vix"] = f"{vix:.1f} fear ({w['vix_high']:+d})"
    elif vix > 20:
        score += w["vix_elevated"]; sigs["vix"] = f"{vix:.1f} elevated ({w['vix_elevated']:+d})"
    elif vix < 15:
        score += w["vix_calm"];     sigs["vix"] = f"{vix:.1f} calm ({w['vix_calm']:+d})"

    if row.get("oil_bull", 0):
        score += w["oil_bull"];  sigs["oil"] = f"uptrend ({w['oil_bull']:+d})"
    elif row.get("oil_bear", 0):
        score += w.get("oil_bear", -5); sigs["oil"] = f"downtrend ({w.get('oil_bear',-5):+d})"

    return max(0, min(100, score)), sigs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: PULL MISSING EVENTS BEFORE SAHMK CANCELLATION
# ══════════════════════════════════════════════════════════════════════════════

def pull_missing_events():
    """Pull EARNINGS_SURPRISE and ANALYST_RATING_CHANGE events for all stocks."""
    print("\n[STEP 1] Pulling EARNINGS_SURPRISE + ANALYST_RATING_CHANGE events...")
    api = SAHMKProvider()
    all_surprise = {}
    all_ratings  = {}

    for sym, info in STOCKS.items():
        try:
            surprise = api.get_events(sym, event_type="EARNINGS_SURPRISE", limit=100)
            s_events = surprise.get("events", [])

            ratings  = api.get_events(sym, event_type="ANALYST_RATING_CHANGE", limit=100)
            r_events = ratings.get("events", [])

            if s_events or r_events:
                all_surprise[sym] = s_events
                all_ratings[sym]  = r_events
                print(f"  {info['name']:<35} Surprises={len(s_events)} RatingChanges={len(r_events)}")
            time.sleep(0.2)
        except Exception as e:
            print(f"  {info['name']}: {e}")
            time.sleep(0.2)

    with open(DATA_RAW / "earnings_surprises.json", "w", encoding="utf-8") as f:
        json.dump({"generated": datetime.now().isoformat(), "stocks": all_surprise},
                  f, indent=2, ensure_ascii=False)
    with open(DATA_RAW / "analyst_rating_changes.json", "w", encoding="utf-8") as f:
        json.dump({"generated": datetime.now().isoformat(), "stocks": all_ratings},
                  f, indent=2, ensure_ascii=False)

    total_s = sum(len(v) for v in all_surprise.values())
    total_r = sum(len(v) for v in all_ratings.values())
    print(f"\n  Total earnings surprises: {total_s}")
    print(f"  Total analyst rating changes: {total_r}")
    return all_surprise, all_ratings


# ══════════════════════════════════════════════════════════════════════════════
# EARNINGS SURPRISE SIGNAL TEST
# ══════════════════════════════════════════════════════════════════════════════

def test_earnings_surprise_signal(sym: str, master: pd.DataFrame,
                                   surprises: list) -> dict:
    """Test: what happens after POSITIVE vs NEGATIVE earnings surprise events?"""
    if not surprises:
        return {}

    valid = master.dropna(subset=["fwd_90d"])
    pos_mask = pd.Series(False, index=master.index)
    neg_mask = pd.Series(False, index=master.index)

    for ev in surprises:
        ev_date   = pd.to_datetime(ev.get("event_date"))
        sentiment = ev.get("sentiment", "neutral")
        future    = master.index[master.index > ev_date]
        if len(future) > 0:
            if sentiment in ("positive", "very_positive"):
                pos_mask[future[0]] = True
            elif sentiment in ("negative", "very_negative"):
                neg_mask[future[0]] = True

    results = {}
    for label, mask in [("positive_surprise", pos_mask), ("negative_surprise", neg_mask)]:
        sub = valid[mask.reindex(valid.index, fill_value=False)]
        if len(sub) >= 5:
            fwd = sub["fwd_90d"].dropna()
            t, p = stats.ttest_1samp(fwd, 0)
            results[label] = {
                "n":            len(sub),
                "avg_90d":      round(fwd.mean() * 100, 2),
                "win_pct":      round((fwd > 0).mean() * 100, 1),
                "p_value":      round(p, 3),
                "significant":  bool(p < 0.05),
            }
    return results


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD WITH SECTOR WEIGHTS + EARNINGS SURPRISE
# ══════════════════════════════════════════════════════════════════════════════

def run_sector_aware_walkforward(sym: str, name: str, sector: str,
                                  master: pd.DataFrame, annual_df: pd.DataFrame,
                                  balance_df: pd.DataFrame, cashflow_df: pd.DataFrame,
                                  shares: float, surprise_results: dict) -> dict:

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
        if len(avail) > 0: pred_dates.append(avail[0])
        d += pd.Timedelta(days=90)

    predictions   = []
    mistake_vault = []

    for pred_date in pred_dates:
        row = master.loc[pred_date]

        # Sector-aware environment score
        env, env_sigs = score_environment_sector(row, sector)

        # Technical score (unchanged across sectors)
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

        # Fundamental
        fund, _ = score_fundamental_enriched(pred_date, annual_df, balance_df, cashflow_df)

        # Valuation (historical P/E + P/B)
        val = 50
        pe  = row.get("pe_hist")
        if pe and not pd.isna(pe) and 0 < pe < 200:
            if pe < 6:      val += 15
            elif pe < 9:    val += 10
            elif pe < 12:   val += 4
            elif pe < 18:   val -= 4
            elif pe >= 18:  val -= 10
        pb = row.get("pb_hist")
        if pb and not pd.isna(pb) and 0 < pb < 50:
            if pb < 0.8:    val += 10
            elif pb < 1.2:  val += 4
            elif pb > 5:    val -= 6
        val = max(0, min(100, val))

        comp = (env*weights["env"] + tech*weights["tech"] +
                fund*weights["fund"] + val*weights["val"])

        regime = row.get("rate_regime", "stable")
        if comp > 74:   base_90d = 9.5
        elif comp > 67: base_90d = 7.5
        elif comp > 60: base_90d = 5.5
        elif comp > 53: base_90d = 3.0
        elif comp > 46: base_90d = 1.0
        else:           base_90d = -3.5
        if regime == "rising":   base_90d -= 5.0
        elif regime == "falling":base_90d -= 1.0

        conf  = int(40 + abs(comp-50)/50*35 + (1-abs(env-tech)/100)*25)
        conf  = max(20, min(90, conf))
        price = float(row["close"])

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
            "sector": sector,
            "rsi":    round(float(rsi),1) if not pd.isna(rsi) else None,
            "vix":    round(float(row.get("vix",20)),1),
            "pe_at_prediction": round(float(row.get("pe_hist",0)),1) if not pd.isna(row.get("pe_hist",0)) else None,
            "env_score": round(env,1), "tech_score": round(tech,1),
            "fund_score":round(fund,1), "val_score":  round(val,1),
            "composite": round(comp,1),
            "base_30d_pct":round(base_90d*0.4,1),
            "base_60d_pct":round(base_90d*0.7,1),
            "base_90d_pct":round(base_90d,1),
            "confidence":  conf,
            **actual,
        }
        predictions.append(pred)

        err = abs(actual.get("actual_90d_pct",0) - base_90d)
        if err > 12 and "actual_90d_pct" in actual:
            dir_ok = (actual["actual_90d_pct"]>0)==(base_90d>0)
            mistake_vault.append({
                "prediction_date": pred_date.date().isoformat(),
                "predicted_90d":   base_90d, "actual_90d": actual["actual_90d_pct"],
                "error":           round(err,1), "direction_correct": dir_ok,
                "composite_score": round(comp,1), "sector": sector,
                "rate_regime":     regime,
                "root_cause": (
                    f"Sector weights may need tuning for {sector}" if not dir_ok and abs(env-50) > 20
                    else "Model underestimated upside" if actual["actual_90d_pct"] > base_90d+10
                    else "Model underestimated downside" if actual["actual_90d_pct"] < base_90d-10
                    else "Mixed signals"
                ),
            })

    df = pd.DataFrame(predictions)
    df = df[df["actual_90d_pct"].notna()] if "actual_90d_pct" in df.columns else df
    validation = {}
    if len(df) >= 5 and "actual_90d_pct" in df.columns:
        df["pred_dir"]   = (df["base_90d_pct"]>0).astype(int)
        df["actual_dir"] = (df["actual_90d_pct"]>0).astype(int)
        df["correct"]    = (df["pred_dir"]==df["actual_dir"]).astype(int)
        dir_acc  = df["correct"].mean()
        baseline = df["actual_dir"].mean()
        mae      = (df["actual_90d_pct"]-df["base_90d_pct"]).abs().mean()
        validation = {
            "n_predictions":            len(df),
            "directional_accuracy_pct": round(dir_acc*100,1),
            "baseline_always_long_pct": round(baseline*100,1),
            "edge_over_baseline_pct":   round((dir_acc-baseline)*100,1),
            "mae_pct":                  round(mae,2),
            "avg_predicted_90d":        round(df["base_90d_pct"].mean(),2),
            "avg_actual_90d":           round(df["actual_90d_pct"].mean(),2),
            "systematic_bias":          round(df["actual_90d_pct"].mean()-df["base_90d_pct"].mean(),2),
            "sector_weights_applied":   SECTOR_MACRO_WEIGHTS.get(sector, {}).get("note","default"),
        }
        regime_acc = {}
        for r in ["rising","stable","falling"]:
            sub = df[df["rate_regime"]==r]
            if len(sub) >= 3:
                regime_acc[r] = {"n":len(sub), "accuracy":round(sub["correct"].mean()*100,1)}
        validation["regime_breakdown"] = regime_acc

    return {"sym":sym,"name":name,"generated":datetime.now().isoformat(),
            "validation":validation,"predictions":predictions,"mistake_vault":mistake_vault}


# ══════════════════════════════════════════════════════════════════════════════
# BEFORE/AFTER COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════

def load_previous_scores() -> dict:
    """Load scores from the last rebuild for comparison."""
    prev = {}
    path = REPORTS_DIR / "FINAL_AIRTIGHT_REPORT.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        for r in data.get("results", []):
            prev[r["symbol"]] = {
                "accuracy": r.get("accuracy"),
                "edge":     r.get("edge"),
            }
    return prev


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  FINAL COMPLETE REBUILD — Sector Weights + Events + Air Tight")
    print(f"  Stocks: {len(STOCKS)} | Date: {date.today()}")
    print("=" * 70)

    # Load previous scores for before/after comparison
    prev_scores = load_previous_scores()

    # Load company KPIs
    kpi_path = DATA_RAW / "company_kpis.json"
    KPIS = {}
    if kpi_path.exists():
        with open(kpi_path, encoding="utf-8") as f:
            KPIS = json.load(f).get("stocks", {})

    # Step 1: Pull missing events
    all_surprises, all_ratings = pull_missing_events()

    # Step 2: Rebuild all stocks
    print("\n[STEP 2] Rebuilding all stocks with sector-specific weights...")
    all_results = []
    failed      = []

    for sym, info in STOCKS.items():
        name   = info["name"]
        sector = info["sector"]
        path   = DATA_PROCESSED / ("master_dataset.csv" if sym == "1120" else f"master_{sym}.csv")
        if not path.exists():
            continue

        try:
            print(f"\n  {name} ({sym}) — {sector}")

            master = pd.read_csv(path, index_col=0, parse_dates=True)
            master.index = pd.to_datetime(master.index).tz_localize(None)

            income_df, balance_df, cashflow_df, _, annual_df = load_enriched_financials(sym)
            divs   = load_dividends(sym)
            events = load_events(sym)
            shares = KPIS.get(sym, {}).get("shares_outstanding") or 0

            # Add historical P/E and P/B
            master = build_historical_valuation(sym, master, income_df, balance_df)
            master.to_csv(path)

            # Signal discovery
            signals = run_signal_discovery(sym, name, master)

            # Test earnings surprise signal
            surprises = all_surprises.get(sym, [])
            surprise_results = test_earnings_surprise_signal(sym, master, surprises)
            if surprise_results:
                for label, res in surprise_results.items():
                    signals["signals"].append({
                        "signal":           f"Earnings Surprise — {label.replace('_',' ').title()}",
                        "group":            "Events — Earnings Surprise",
                        "occurrences":      res["n"],
                        "avg_return_90d":   res["avg_90d"],
                        "pct_positive_90d": res["win_pct"],
                        "p_value_90d":      res["p_value"],
                        "significant_90d":  res["significant"],
                        "reliability_label":"HIGH" if res["significant"] and res["n"]>=10 else "MEDIUM",
                    })

            memory = run_memory_engine(sym, name, master, income_df, annual_df, divs, events)
            hyp    = run_hypothesis_engine(sym, name, master)

            # Walk-forward with SECTOR weights
            wf    = run_sector_aware_walkforward(
                sym, name, sector, master, annual_df,
                balance_df, cashflow_df, shares, surprise_results)
            v     = wf.get("validation", {})
            recal = run_recalibration(sym, name, hyp, wf, master)
            save_outputs(sym, signals, memory, hyp, wf, recal)

            new_acc  = v.get("directional_accuracy_pct")
            new_edge = v.get("edge_over_baseline_pct")
            prev_acc = prev_scores.get(sym, {}).get("accuracy")
            prev_edge= prev_scores.get(sym, {}).get("edge")

            print(f"  acc={new_acc}% (prev={prev_acc}%)  "
                  f"edge={new_edge:+.1f}% (prev={prev_edge:+.1f}%)" if prev_edge else
                  f"  acc={new_acc}%  edge={new_edge:+.1f}%")

            all_results.append({
                "symbol":    sym, "name": name, "sector": sector,
                "days":      len(master),
                "accuracy":  new_acc,  "prev_accuracy": prev_acc,
                "edge":      new_edge, "prev_edge":     prev_edge,
                "bias":      v.get("systematic_bias"),
                "hyp_acc":   hyp["summary"]["accepted"],
                "mistakes":  len(wf.get("mistake_vault",[])),
                "has_surprises": len(surprises),
                "sector_weight_note": SECTOR_MACRO_WEIGHTS.get(sector,{}).get("note","default"),
            })

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            failed.append({"sym": sym, "error": str(e)})

    # Step 3: Rebuild current state
    print("\n[STEP 3] Rebuilding current state...")
    subprocess.run([sys.executable, "build_all_forecasts_v3.py"])

    # Step 4: Before/After comparison table
    print("\n" + "=" * 75)
    print("  BEFORE / AFTER COMPARISON TABLE")
    print("  (Previous = last rebuild | New = sector weights + events + P/E)")
    print("=" * 75)
    print(f"\n  {'Stock':<35} {'Sector':<18} {'Prev Acc':>9} {'New Acc':>8} {'Chg':>5} {'Prev Edge':>10} {'New Edge':>9} {'Chg':>5}")
    print(f"  {'-'*35} {'-'*18} {'-'*9} {'-'*8} {'-'*5} {'-'*10} {'-'*9} {'-'*5}")

    improved = 0
    declined = 0
    for r in sorted(all_results, key=lambda x: x["name"]):
        pa   = r.get("prev_accuracy")
        na   = r.get("accuracy")
        pe   = r.get("prev_edge")
        ne   = r.get("edge")
        pa_s = f"{pa:.1f}%" if pa else "N/A"
        na_s = f"{na:.1f}%" if na else "N/A"
        pe_s = f"{pe:+.1f}%" if pe is not None else "N/A"
        ne_s = f"{ne:+.1f}%" if ne is not None else "N/A"

        acc_chg = ""
        if pa and na:
            diff = na - pa
            acc_chg = f"{diff:+.1f}"
            if diff > 0: improved += 1
            elif diff < 0: declined += 1

        edge_chg = ""
        if pe is not None and ne is not None:
            edge_chg = f"{ne-pe:+.1f}"

        print(f"  {r['name']:<35} {r['sector']:<18} {pa_s:>9} {na_s:>8} {acc_chg:>5} "
              f"{pe_s:>10} {ne_s:>9} {edge_chg:>5}")

    print(f"\n  Stocks improved: {improved} | Declined: {declined} | "
          f"Unchanged: {len(all_results)-improved-declined}")

    # Step 5: Push
    print("\n[STEP 4] Pushing to live portal...")
    subprocess.run([sys.executable, "update_and_push.py",
                    "Final rebuild: sector-specific oil/rate weights + earnings surprise events + historical PE/PB"])

    # Save report
    with open(REPORTS_DIR / "BEFORE_AFTER_COMPARISON.json", "w") as f:
        json.dump({"generated": datetime.now().isoformat(), "results": all_results,
                   "sector_weights": SECTOR_MACRO_WEIGHTS}, f, indent=2, default=str)

    print(f"\n  Before/After table → reports/BEFORE_AFTER_COMPARISON.json")
    print(f"  Portal → https://tadawul-stock-exchange.streamlit.app/")
    print(f"\n  ⚠️  You can now cancel your SAHMK subscription safely.")
    print(f"  Re-subscribe for 1 month each quarter when results are published.")
    print(f"  Next bulk extract: July 2026 (Q2 results season)")

    if failed:
        print(f"\n  Failed: {[f['sym'] for f in failed]}")


if __name__ == "__main__":
    main()
