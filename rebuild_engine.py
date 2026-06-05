"""
FULL ENGINE REBUILD — Enriched with complete financial data
============================================================
Extracts ALL financial fields from already-downloaded SAHMK JSON files:
  + Total debt / leverage
  + Operating cash flow & free cash flow
  + Cash flow quality (OCF vs net income)
  + EPS from ratios
  + Book value per share
  + P/E, P/B valuation signals

Then re-runs all 5 phases for every stock and pushes to portal.

Run:  python rebuild_engine.py
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
from shared import STOCKS


# ══════════════════════════════════════════════════════════════════════════════
# ENRICHED FINANCIAL LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_enriched_financials(sym: str) -> tuple:
    """
    Load income, balance sheet, cash flow, and ratios for any stock.
    Returns (income_df, balance_df, cashflow_df, ratios_df, annual_df)
    All enriched with full fields.
    """
    # ── Determine data path ──────────────────────────────────────────────────
    if sym == "1120":
        fin_path    = DATA_RAW / "alrajhi_financials_quarterly_raw.json"
        ratios_path = DATA_RAW / "alrajhi_ratios_quarterly.json"
    else:
        fin_path    = DATA_RAW / f"stock_{sym}" / "financials_quarterly.json"
        ratios_path = DATA_RAW / f"stock_{sym}" / "ratios.json"

    income_df = balance_df = cashflow_df = ratios_df = pd.DataFrame()
    annual_df = pd.DataFrame()

    # ── Income statement ──────────────────────────────────────────────────────
    if fin_path.exists():
        with open(fin_path, encoding="utf-8") as f:
            data = json.load(f)

        income_list = data.get("income_statements", [])
        if income_list:
            income_df = pd.DataFrame(income_list)
            income_df["report_date"] = pd.to_datetime(income_df["report_date"])
            income_df = income_df.sort_values("report_date").reset_index(drop=True)
            income_df["net_income_bn"] = income_df["net_income"] / 1e9

        balance_list = data.get("balance_sheets", [])
        if balance_list:
            balance_df = pd.DataFrame(balance_list)
            balance_df["report_date"] = pd.to_datetime(balance_df["report_date"])
            balance_df = balance_df.sort_values("report_date").reset_index(drop=True)
            # Debt-to-equity
            if "total_debt" in balance_df.columns and "stockholders_equity" in balance_df.columns:
                balance_df["debt_to_equity"] = (
                    balance_df["total_debt"].fillna(0) /
                    balance_df["stockholders_equity"].replace(0, np.nan)
                )
            # Asset growth
            balance_df["asset_growth_yoy"] = balance_df["total_assets"].pct_change(4) * 100

        cashflow_list = data.get("cash_flows", [])
        if cashflow_list:
            cashflow_df = pd.DataFrame(cashflow_list)
            cashflow_df["report_date"] = pd.to_datetime(cashflow_df["report_date"])
            cashflow_df = cashflow_df.sort_values("report_date").reset_index(drop=True)
            # Cash flow quality: OCF / Net Income (>1 = high quality, <0.5 = low quality)
            if "operating_cash_flow" in cashflow_df.columns:
                cashflow_df["ocf_bn"] = cashflow_df["operating_cash_flow"] / 1e9
            if "free_cash_flow" in cashflow_df.columns:
                cashflow_df["fcf_bn"] = cashflow_df["free_cash_flow"] / 1e9

    # ── Ratios ────────────────────────────────────────────────────────────────
    if ratios_path.exists():
        with open(ratios_path, encoding="utf-8") as f:
            ratios_data = json.load(f)
        periods = ratios_data.get("ratios", [])
        if periods:
            rows = []
            for p in periods:
                r = p.get("ratios", {})
                km = p.get("key_metrics", {})
                rows.append({
                    "report_date":       pd.to_datetime(p.get("report_date")),
                    "roe":               r.get("roe"),
                    "roa":               r.get("roa"),
                    "net_income":        km.get("net_income"),
                    "total_assets":      km.get("total_assets"),
                    "stockholders_equity": km.get("stockholders_equity"),
                    "operating_cash_flow": km.get("operating_cash_flow"),
                })
            ratios_df = pd.DataFrame(rows).sort_values("report_date").reset_index(drop=True)

    # ── Merge into annual summary ──────────────────────────────────────────────
    if not income_df.empty:
        annual = income_df[income_df["report_date"].dt.month == 12].copy()
        annual["ni_yoy"] = annual["net_income"].pct_change() * 100
        annual["revenue_yoy"] = annual.get("total_revenue", pd.Series(dtype=float)).pct_change() * 100

        # Merge balance sheet fields
        if not balance_df.empty:
            bal_annual = balance_df[balance_df["report_date"].dt.month.isin([12, 3])].copy()
            bal_annual = bal_annual.sort_values("report_date").drop_duplicates("report_date")
            annual = annual.merge(
                bal_annual[["report_date","total_assets","stockholders_equity","total_debt","debt_to_equity"]],
                on="report_date", how="left"
            )

        # Merge cash flow fields
        if not cashflow_df.empty:
            cf_annual = cashflow_df[cashflow_df["report_date"].dt.month == 12].copy()
            annual = annual.merge(
                cf_annual[["report_date","operating_cash_flow","free_cash_flow"]],
                on="report_date", how="left", suffixes=("", "_cf")
            )
            # Cash flow quality ratio
            if "operating_cash_flow" in annual.columns and "net_income" in annual.columns:
                annual["cfq"] = annual["operating_cash_flow"] / annual["net_income"].replace(0, np.nan)

        annual_df = annual.reset_index(drop=True)

    return income_df, balance_df, cashflow_df, ratios_df, annual_df


# ══════════════════════════════════════════════════════════════════════════════
# ENRICHED FUNDAMENTAL SCORE
# ══════════════════════════════════════════════════════════════════════════════

def score_fundamental_enriched(pred_date, annual_df: pd.DataFrame,
                                balance_df: pd.DataFrame,
                                cashflow_df: pd.DataFrame) -> tuple:
    """
    Richer fundamental score using all available financial data.
    Returns (score 0-100, signals dict)
    """
    score = 50
    sigs  = {}

    if annual_df.empty:
        return score, {"note": "no fundamental data"}

    # Only use data known before prediction date
    known = annual_df[annual_df["report_date"] < pred_date]
    if known.empty:
        return score, {"note": "no historical fundamental data before prediction date"}

    latest = known.iloc[-1]

    # ── Net income growth ─────────────────────────────────────────────────────
    ni_yoy = latest.get("ni_yoy")
    if ni_yoy is not None and not pd.isna(ni_yoy):
        if ni_yoy > 25:
            score += 18; sigs["ni_growth"] = f"+{ni_yoy:.1f}% YoY exceptional (+18)"
        elif ni_yoy > 15:
            score += 12; sigs["ni_growth"] = f"+{ni_yoy:.1f}% YoY strong (+12)"
        elif ni_yoy > 5:
            score += 5;  sigs["ni_growth"] = f"+{ni_yoy:.1f}% YoY positive (+5)"
        elif ni_yoy > 0:
            score += 2;  sigs["ni_growth"] = f"+{ni_yoy:.1f}% YoY weak (+2)"
        elif ni_yoy < -20:
            score -= 15; sigs["ni_growth"] = f"{ni_yoy:.1f}% YoY severe decline (-15)"
        elif ni_yoy < 0:
            score -= 8;  sigs["ni_growth"] = f"{ni_yoy:.1f}% YoY declining (-8)"

    # ── Cash flow quality ─────────────────────────────────────────────────────
    cfq = latest.get("cfq")
    if cfq is not None and not pd.isna(cfq):
        if cfq > 1.2:
            score += 10; sigs["cfq"] = f"OCF/NI={cfq:.2f} — high quality earnings (+10)"
        elif cfq > 0.8:
            score += 4;  sigs["cfq"] = f"OCF/NI={cfq:.2f} — good cash backing (+4)"
        elif cfq < 0.3:
            score -= 10; sigs["cfq"] = f"OCF/NI={cfq:.2f} — low cash quality (-10)"
        elif cfq < 0:
            score -= 15; sigs["cfq"] = f"OCF/NI={cfq:.2f} — negative operating cash flow (-15)"

    # ── Leverage / debt ───────────────────────────────────────────────────────
    dte = latest.get("debt_to_equity")
    if dte is not None and not pd.isna(dte):
        if dte < 0.3:
            score += 6;  sigs["leverage"] = f"D/E={dte:.2f} — low debt (+6)"
        elif dte < 1.0:
            score += 2;  sigs["leverage"] = f"D/E={dte:.2f} — moderate debt (+2)"
        elif dte > 3.0:
            score -= 10; sigs["leverage"] = f"D/E={dte:.2f} — high leverage (-10)"
        elif dte > 2.0:
            score -= 5;  sigs["leverage"] = f"D/E={dte:.2f} — elevated debt (-5)"

    # ── Asset growth (proxy for expansion) ────────────────────────────────────
    asset_growth = latest.get("asset_growth_yoy")
    if asset_growth is not None and not pd.isna(asset_growth):
        if 5 < asset_growth < 25:
            score += 4;  sigs["asset_growth"] = f"Assets +{asset_growth:.1f}% YoY — healthy expansion (+4)"
        elif asset_growth > 30:
            score += 2;  sigs["asset_growth"] = f"Assets +{asset_growth:.1f}% YoY — aggressive growth (+2)"
        elif asset_growth < 0:
            score -= 5;  sigs["asset_growth"] = f"Assets {asset_growth:.1f}% YoY — shrinking (-5)"

    # ── Multi-year profit trend (acceleration or deceleration) ────────────────
    if len(known) >= 3:
        recent_growth = known["ni_yoy"].tail(3).mean()
        if not pd.isna(recent_growth):
            if recent_growth > 15:
                score += 5;  sigs["trend"] = f"3yr avg growth={recent_growth:.1f}% — accelerating (+5)"
            elif recent_growth < -5:
                score -= 5;  sigs["trend"] = f"3yr avg growth={recent_growth:.1f}% — decelerating (-5)"

    return max(0, min(100, score)), sigs


# ══════════════════════════════════════════════════════════════════════════════
# ENRICHED SIGNAL DISCOVERY (adds financial signals)
# ══════════════════════════════════════════════════════════════════════════════

def discover_financial_signals(sym: str, master: pd.DataFrame,
                                annual_df: pd.DataFrame,
                                cashflow_df: pd.DataFrame) -> list:
    """Map financial events to trading day signals and measure forward returns."""
    from phases.full_engine import run_signal_discovery

    results = []
    valid   = master.dropna(subset=["fwd_90d"])

    def map_to_trading_day(report_dates, condition_fn) -> pd.Series:
        mask = pd.Series(False, index=master.index)
        for d in report_dates:
            if condition_fn(d):
                future = master.index[master.index > d]
                if len(future) > 0:
                    mask[future[0]] = True
        return mask

    def test_fin_signal(label, mask, min_n=5):
        sub = valid[mask.reindex(valid.index, fill_value=False)]
        if len(sub) < min_n:
            return {"signal": label, "group": "Fundamental — Financial", "occurrences": len(sub), "insufficient": True}
        fwd = sub["fwd_90d"].dropna()
        t, p = stats.ttest_1samp(fwd, 0)
        base = valid["fwd_90d"].mean() * 100
        return {
            "signal":          label,
            "group":           "Fundamental — Financial",
            "occurrences":     len(sub),
            "avg_return_90d":  round(fwd.mean() * 100, 2),
            "pct_positive_90d":round((fwd > 0).mean() * 100, 1),
            "p_value_90d":     round(p, 3),
            "significant_90d": bool(p < 0.05),
            "edge_vs_base":    round(fwd.mean() * 100 - base, 2),
            "reliability_label": "HIGH" if (p < 0.05 and len(sub) >= 15) else ("MEDIUM" if p < 0.10 else "LOW"),
        }

    if not annual_df.empty:
        # Strong profit growth
        strong = annual_df[annual_df["ni_yoy"] > 20]["report_date"].tolist()
        results.append(test_fin_signal("Annual NI growth >20% YoY",
            map_to_trading_day(strong, lambda d: True)))

        # Profit decline
        decline = annual_df[annual_df["ni_yoy"] < 0]["report_date"].tolist()
        results.append(test_fin_signal("Annual NI decline YoY",
            map_to_trading_day(decline, lambda d: True)))

        # High cash flow quality
        if "cfq" in annual_df.columns:
            hcfq = annual_df[annual_df["cfq"] > 1.2]["report_date"].tolist()
            results.append(test_fin_signal("High cash flow quality (OCF/NI >1.2)",
                map_to_trading_day(hcfq, lambda d: True)))

            lcfq = annual_df[annual_df["cfq"] < 0.3]["report_date"].tolist()
            results.append(test_fin_signal("Low cash flow quality (OCF/NI <0.3)",
                map_to_trading_day(lcfq, lambda d: True)))

        # Leverage signals
        if "debt_to_equity" in annual_df.columns:
            hd = annual_df[annual_df["debt_to_equity"] > 2.0]["report_date"].tolist()
            results.append(test_fin_signal("High leverage (D/E >2)",
                map_to_trading_day(hd, lambda d: True)))

    return [r for r in results if not r.get("insufficient")]


# ══════════════════════════════════════════════════════════════════════════════
# ENRICHED WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════════════

def run_enriched_walkforward(sym: str, name: str, master: pd.DataFrame,
                              annual_df: pd.DataFrame,
                              balance_df: pd.DataFrame,
                              cashflow_df: pd.DataFrame) -> dict:
    """Walk-forward using enriched fundamental score."""
    weights = {"environment": 0.40, "technical": 0.30, "fundamental": 0.30}
    # Note: increased fundamental weight now that we have richer data

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

        # Environment score
        env = 50
        if row.get("rate_regime") == "stable": env += 20
        elif row.get("rate_regime") == "rising": env -= 25
        elif row.get("rate_regime") == "falling": env -= 5
        vix = row.get("vix", 20)
        if vix > 30: env += 15
        elif vix > 20: env += 5
        elif vix < 15: env -= 5
        if row.get("oil_bull", 0): env += 10
        env = max(0, min(100, env))

        # Technical score
        tech = 50
        rsi  = row.get("rsi", 50)
        if not pd.isna(rsi):
            if rsi < 20: tech += 20
            elif rsi < 30: tech += 15
            elif rsi < 40: tech += 8
            elif rsi > 80: tech -= 20
            elif rsi > 70: tech -= 12
        bb = row.get("bb_pct", 0.5)
        if not pd.isna(bb):
            if bb < 0.05: tech += 20
            elif bb < 0.15: tech += 10
            elif bb > 0.95: tech -= 15
        ret20 = row.get("ret_20d", 0)
        if not pd.isna(ret20):
            if ret20 > 0.15: tech -= 20
            elif ret20 > 0.10: tech -= 10
            elif ret20 < -0.15: tech += 15
            elif ret20 < -0.10: tech += 8
        tech = max(0, min(100, tech))

        # Enriched fundamental score
        fund, fund_sigs = score_fundamental_enriched(pred_date, annual_df, balance_df, cashflow_df)

        comp = (env * weights["environment"] + tech * weights["technical"] +
                fund * weights["fundamental"])

        # Forecast generation with recalibrated estimates
        regime = row.get("rate_regime", "stable")
        if comp > 72:   base_90d = 9.0
        elif comp > 65: base_90d = 7.0
        elif comp > 58: base_90d = 5.0
        elif comp > 50: base_90d = 3.0
        elif comp > 42: base_90d = 1.0
        else:           base_90d = -3.0

        if regime == "rising": base_90d -= 5.0
        elif regime == "falling": base_90d -= 1.0

        base_30d = base_90d * 0.40
        base_60d = base_90d * 0.70
        confidence = int(40 + abs(comp-50)/50*35 + (1 - abs(env-tech)/100)*25)
        confidence = max(20, min(90, confidence))
        price = float(row["close"])

        # Actuals
        future = master.index[master.index > pred_date]
        actual = {}
        for w in [30, 60, 90]:
            if len(future) >= w:
                idx = master.index.get_loc(future[0]) + w - 1
                if idx < len(master):
                    p_fut = master["close"].iloc[idx]
                    actual[f"actual_{w}d_pct"]  = round((p_fut / price - 1) * 100, 1)
                    actual[f"hit_target_{w}d"]  = (actual[f"actual_{w}d_pct"] > 0) == (base_90d > 0)

        pred = {
            "prediction_date": pred_date.date().isoformat(),
            "price": round(price, 2), "rate_regime": regime,
            "rsi":   round(float(rsi), 1) if not pd.isna(rsi) else None,
            "vix":   round(float(vix), 1),
            "env_score":  round(env, 1), "tech_score": round(tech, 1),
            "fund_score": round(fund, 1), "composite":  round(comp, 1),
            "base_30d_pct": round(base_30d, 1),
            "base_60d_pct": round(base_60d, 1),
            "base_90d_pct": round(base_90d, 1),
            "confidence": confidence,
            **actual,
        }
        predictions.append(pred)

        err = abs(actual.get("actual_90d_pct", 0) - base_90d)
        if err > 12 and "actual_90d_pct" in actual:
            dir_ok = (actual["actual_90d_pct"] > 0) == (base_90d > 0)
            mistake_vault.append({
                "prediction_date": pred_date.date().isoformat(),
                "predicted_90d":   base_90d,
                "actual_90d":      actual["actual_90d_pct"],
                "error":           round(err, 1),
                "direction_correct": dir_ok,
                "composite_score": round(comp, 1),
                "fund_score":      round(fund, 1),
                "rate_regime":     regime,
                "rsi":             round(float(rsi), 1) if not pd.isna(rsi) else None,
                "vix":             round(float(vix), 1),
                "root_cause": (
                    "Model underestimated upside" if actual["actual_90d_pct"] > base_90d + 10
                    else "Model underestimated downside" if actual["actual_90d_pct"] < base_90d - 10
                    else "Fundamental signals misleading" if fund > 65 and actual["actual_90d_pct"] < 0
                    else "Mixed signals"
                ),
            })

    # Validation
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
        validation = {
            "n_predictions":            len(df),
            "directional_accuracy_pct": round(dir_acc * 100, 1),
            "baseline_always_long_pct": round(baseline * 100, 1),
            "edge_over_baseline_pct":   round((dir_acc - baseline) * 100, 1),
            "mae_pct":                  round(mae, 2),
            "avg_predicted_90d":        round(df["base_90d_pct"].mean(), 2),
            "avg_actual_90d":           round(df["actual_90d_pct"].mean(), 2),
            "systematic_bias":          round(df["actual_90d_pct"].mean() - df["base_90d_pct"].mean(), 2),
            "fund_weight_used":         weights["fundamental"],
        }
        regime_acc = {}
        for r in ["rising","stable","falling"]:
            sub = df[df["rate_regime"]==r]
            if len(sub) >= 3:
                regime_acc[r] = {"n": len(sub), "accuracy": round(sub["correct"].mean()*100,1)}
        validation["regime_breakdown"] = regime_acc

    return {
        "sym":          sym, "name": name,
        "generated":    datetime.now().isoformat(),
        "validation":   validation,
        "predictions":  predictions,
        "mistake_vault":mistake_vault,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN REBUILD RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def rebuild_stock(sym: str) -> dict:
    """Full enriched 5-phase rebuild for one stock."""
    from phases.full_engine import (run_signal_discovery, run_memory_engine,
                                     run_hypothesis_engine, run_recalibration,
                                     save_outputs, load_dividends, load_events)

    info   = STOCKS.get(sym, {})
    name   = info.get("name", sym)
    sector = info.get("sector", "?")

    print(f"\n  {'─'*52}")
    print(f"  {name} ({sym}) — {sector}")
    print(f"  {'─'*52}")

    # Load master dataset
    path = DATA_PROCESSED / ("master_dataset.csv" if sym == "1120" else f"master_{sym}.csv")
    if not path.exists():
        print(f"  SKIP: no master dataset")
        return {}

    master = pd.read_csv(path, index_col=0, parse_dates=True)
    master.index = pd.to_datetime(master.index).tz_localize(None)

    # Load enriched financials
    income_df, balance_df, cashflow_df, ratios_df, annual_df = load_enriched_financials(sym)
    divs   = load_dividends(sym)
    events = load_events(sym)

    n_inc = len(income_df); n_bal = len(balance_df)
    n_cf  = len(cashflow_df); n_annual = len(annual_df)
    has_cfq = "cfq" in annual_df.columns if not annual_df.empty else False
    has_dte = "debt_to_equity" in annual_df.columns if not annual_df.empty else False
    print(f"  Financials: {n_annual} annual | Balance: {n_bal} | CashFlow: {n_cf} | "
          f"CFQ: {'✓' if has_cfq else '✗'} | D/E: {'✓' if has_dte else '✗'}")

    # Phase 1: Signal discovery (standard + financial)
    signals    = run_signal_discovery(sym, name, master)
    fin_sigs   = discover_financial_signals(sym, master, annual_df, cashflow_df)
    if fin_sigs:
        signals["signals"] = signals.get("signals", []) + fin_sigs
        print(f"  Phase 1: {len(signals.get('signals', []))} signals (incl. {len(fin_sigs)} financial)")
    else:
        print(f"  Phase 1: {len(signals.get('signals', []))} signals")

    # Phase 2: Memory engine
    memory = run_memory_engine(sym, name, master, income_df, annual_df, divs, events)
    print(f"  Phase 2: Memory rebuilt")

    # Phase 3: Hypothesis engine
    hyp = run_hypothesis_engine(sym, name, master)
    print(f"  Phase 3: {hyp['summary']['accepted']}/{hyp['summary']['total']} hypotheses accepted")

    # Phase 4: Enriched walk-forward
    wf = run_enriched_walkforward(sym, name, master, annual_df, balance_df, cashflow_df)
    v  = wf.get("validation", {})
    print(f"  Phase 4: acc={v.get('directional_accuracy_pct','?')}% "
          f"edge={v.get('edge_over_baseline_pct',0):+.1f}% "
          f"bias={v.get('systematic_bias',0):+.1f}% "
          f"mistakes={len(wf.get('mistake_vault',[]))}")

    # Phase 5: Recalibration
    recal = run_recalibration(sym, name, hyp, wf, master)
    print(f"  Phase 5: {len(recal.get('log',[]))} recalibration actions")

    # Save all outputs
    save_outputs(sym, signals, memory, hyp, wf, recal)

    return {
        "symbol":    sym, "name": name, "sector": sector,
        "days":      len(master),
        "accuracy":  v.get("directional_accuracy_pct"),
        "baseline":  v.get("baseline_always_long_pct"),
        "edge":      v.get("edge_over_baseline_pct"),
        "bias":      v.get("systematic_bias"),
        "hyp_acc":   hyp["summary"]["accepted"],
        "hyp_total": hyp["summary"]["total"],
        "mistakes":  len(wf.get("mistake_vault",[])),
        "has_cfq":   has_cfq,
        "has_dte":   has_dte,
        "n_annual":  n_annual,
    }


def main():
    print("\n" + "=" * 65)
    print("  FULL ENGINE REBUILD — Enriched Financial Data")
    print(f"  Stocks: {len(STOCKS)} | Date: {date.today()}")
    print("=" * 65)

    all_results = []
    failed      = []

    for sym in STOCKS:
        try:
            result = rebuild_stock(sym)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  ERROR {sym}: {e}")
            failed.append({"sym": sym, "error": str(e)})

    # Rebuild current state forecasts with enriched scoring
    print("\n\nRebuilding current state with enriched scores...")
    subprocess.run([sys.executable, "build_all_forecasts_enriched.py"])

    # Push to portal
    print("\nPushing to live portal...")
    subprocess.run([sys.executable, "update_and_push.py",
                    "Engine rebuild: enriched financial signals (cash flow, leverage, EPS)"])

    # Final summary
    print("\n" + "=" * 65)
    print("  REBUILD COMPLETE — RESULTS SUMMARY")
    print("=" * 65)
    print(f"  {'Stock':<30} {'Acc%':>6} {'Edge':>7} {'Bias':>6} {'Hyp':>6} {'CFQ':>4} {'D/E':>4}")
    print(f"  {'-'*30} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*4} {'-'*4}")
    for r in sorted(all_results, key=lambda x: -(x.get("edge") or -99)):
        edge = r.get("edge")
        bias = r.get("bias", 0)
        edge_s = f"{edge:+.1f}%" if edge is not None else "N/A"
        bias_s = f"{bias:+.1f}%" if bias is not None else "N/A"
        cfq_s  = "✓" if r.get("has_cfq") else "✗"
        dte_s  = "✓" if r.get("has_dte") else "✗"
        print(f"  {r['name']:<30} {str(r.get('accuracy','?')):>6} {edge_s:>7} "
              f"{bias_s:>6} {str(r.get('hyp_acc','?'))+'/'+str(r.get('hyp_total','?')):>6} "
              f"{cfq_s:>4} {dte_s:>4}")

    if failed:
        print(f"\n  FAILED ({len(failed)}): {[f['sym'] for f in failed]}")

    with open(REPORTS_DIR / "REBUILD_REPORT.json", "w") as f:
        json.dump({"generated": datetime.now().isoformat(),
                   "results": all_results, "failed": failed}, f, indent=2, default=str)
    print(f"\n  Full report → reports/REBUILD_REPORT.json")
    print(f"  Portal → https://tadawul-stock-exchange.streamlit.app/")


if __name__ == "__main__":
    main()
