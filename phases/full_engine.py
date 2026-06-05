"""
FULL ENGINE — Runs all 5 phases for any stock symbol.
Produces the same depth as Al Rajhi for every stock.

Run:  python phases/full_engine.py 2230
      python phases/full_engine.py 1211
      python phases/full_engine.py 4015
      python phases/full_engine.py all
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

from config.settings import DATA_RAW, DATA_PROCESSED, MEMORY_DIR, REPORTS_DIR, FORWARD_WINDOWS

MEMORY_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_STOCKS = {
    "1120": {"name": "Al Rajhi Bank",   "sector": "Banking",         "yahoo": "1120.SR"},
    "2230": {"name": "Saudi Chemical",  "sector": "Chemicals",       "yahoo": "2230.SR"},
    "1211": {"name": "Maaden",          "sector": "Mining",          "yahoo": "1211.SR"},
    "4015": {"name": "Jamjoom Pharma",  "sector": "Pharmaceuticals", "yahoo": "4015.SR"},
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_master(sym: str) -> pd.DataFrame:
    if sym == "1120":
        path = DATA_PROCESSED / "master_dataset.csv"
    else:
        path = DATA_PROCESSED / f"master_{sym}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Master dataset not found: {path}. Run expand_stocks.py first.")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def load_financials(sym: str) -> tuple:
    """Returns (income_df, annual_df). Works for all stocks."""
    if sym == "1120":
        path = DATA_RAW / "alrajhi_income_quarterly.csv"
    else:
        json_path = DATA_RAW / f"stock_{sym}" / "financials_quarterly.json"
        if json_path.exists():
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            income_list = data.get("income_statements", [])
            if income_list:
                df = pd.DataFrame(income_list)
                df["report_date"] = pd.to_datetime(df["report_date"])
                df = df.sort_values("report_date").reset_index(drop=True)
                df["net_income_bn"] = df["net_income"] / 1e9
                annual = df[df["report_date"].dt.month == 12].copy()
                annual["ni_yoy"] = annual["net_income"].pct_change() * 100
                return df, annual
        return pd.DataFrame(), pd.DataFrame()

    income = pd.read_csv(path, parse_dates=["report_date"])
    income = income.sort_values("report_date").reset_index(drop=True)
    income["net_income_bn"] = income["net_income"] / 1e9
    annual = income[income["report_date"].dt.month == 12].copy()
    annual["ni_yoy"] = annual["net_income"].pct_change() * 100
    return income, annual


def load_dividends(sym: str) -> pd.DataFrame:
    if sym == "1120":
        path = DATA_RAW / "alrajhi_dividends_sahmk.csv"
        if path.exists():
            return pd.read_csv(path, parse_dates=["announcement_date"])
    else:
        json_path = DATA_RAW / f"stock_{sym}" / "dividends.json"
        if json_path.exists():
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            hist = data.get("history", [])
            if hist:
                df = pd.DataFrame(hist)
                for col in ["announcement_date", "eligibility_date", "distribution_date"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                return df
    return pd.DataFrame()


def load_events(sym: str) -> list:
    if sym == "1120":
        path = DATA_RAW / "alrajhi_events_full.json"
    else:
        path = DATA_RAW / f"stock_{sym}" / "events.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("events", [])
    return []


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def outcome_stats(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) < 5:
        return {"n": len(s), "insufficient": True}
    t, p = stats.ttest_1samp(s, 0)
    return {
        "n":            len(s),
        "avg_pct":      round(s.mean() * 100, 2),
        "median_pct":   round(s.median() * 100, 2),
        "pct_positive": round((s > 0).mean() * 100, 1),
        "std_pct":      round(s.std() * 100, 2),
        "max_gain_pct": round(s.max() * 100, 2),
        "max_loss_pct": round(s.min() * 100, 2),
        "p_value":      round(p, 3),
        "significant":  bool(p < 0.05),
    }


def test_signal(master, mask, signal_name, group, min_gap=5):
    valid  = master.dropna(subset=["fwd_90d"])
    fired  = valid[mask.reindex(valid.index, fill_value=False)]
    if min_gap > 0 and len(fired) > 1:
        gaps  = fired.index.to_series().diff().dt.days
        fired = fired[gaps.isna() | (gaps >= min_gap)]
    n = len(fired)
    result = {"signal": signal_name, "group": group, "occurrences": n}
    if n < 5:
        result["reliability_label"] = "INSUFFICIENT"
        return result
    for w in [30, 60, 90]:
        col = f"fwd_{w}d"
        if col not in fired.columns:
            continue
        fwd = fired[col].dropna()
        if len(fwd) < 5:
            continue
        t, p = stats.ttest_1samp(fwd, 0)
        result[f"avg_return_{w}d"]   = round(fwd.mean() * 100, 2)
        result[f"pct_positive_{w}d"] = round((fwd > 0).mean() * 100, 1)
        result[f"p_value_{w}d"]      = round(p, 3)
        result[f"significant_{w}d"]  = bool(p < 0.05)
        if "fwd_90d_vs_tasi" in fired.columns:
            vs = fired["fwd_90d_vs_tasi"].dropna()
            result[f"avg_alpha_{w}d"] = round(vs.mean() * 100, 2)
    score = 0
    if n >= 15: score += 20
    if n >= 40: score += 15
    p90 = result.get("p_value_90d", 1.0)
    if p90 < 0.05: score += 30
    elif p90 < 0.10: score += 15
    avg90 = abs(result.get("avg_return_90d", 0))
    if avg90 > 5: score += 20
    elif avg90 > 2: score += 10
    result["reliability_score"] = score
    result["reliability_label"] = "HIGH" if score >= 60 else ("MEDIUM" if score >= 35 else "LOW")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — SIGNAL DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def run_signal_discovery(sym: str, name: str, master: pd.DataFrame) -> dict:
    print(f"\n  Phase 1: Signal Discovery...")
    signals = []

    # RSI zones
    for lo, hi, label in [(0,20,"Extreme Oversold"),(20,30,"Oversold"),(30,40,"Low"),
                           (40,50,"Neutral-Low"),(50,60,"Neutral-High"),(60,70,"High"),
                           (70,80,"Overbought"),(80,100,"Extreme Overbought")]:
        signals.append(test_signal(master, (master["rsi"]>=lo)&(master["rsi"]<hi),
                                   f"RSI {label} ({lo}-{hi})", "Technical — RSI", min_gap=5))

    # Bollinger Bands
    signals.append(test_signal(master, master["bb_pct"]<0.05,  "BB Near Lower Band", "Technical — BB", 10))
    signals.append(test_signal(master, master["bb_pct"]>0.95,  "BB Near Upper Band", "Technical — BB", 10))

    # Momentum
    signals.append(test_signal(master, master["ret_20d"]>0.10, "Strong Momentum >10%", "Technical — Momentum", 10))
    signals.append(test_signal(master, master["ret_20d"]<-0.10,"Weak Momentum <-10%", "Technical — Momentum", 10))

    # MA
    signals.append(test_signal(master,
        (master["above_ma20"]==0)&(master["above_ma50"]==0)&(master["above_ma200"]==0),
        "Below All MAs", "Technical — MA", 10))
    signals.append(test_signal(master,
        (master["above_ma20"]==1)&(master["above_ma50"]==1)&(master["above_ma200"]==1),
        "Above All MAs", "Technical — MA", 10))

    # Rate regimes
    for regime in ["rising", "stable", "falling"]:
        signals.append(test_signal(master, master["rate_regime"]==regime,
                                   f"Rate Regime: {regime}", "Macro — Rates", 20))

    # VIX
    signals.append(test_signal(master, master["vix"]>30,  "VIX > 30 (fear)",  "Macro — Risk", 10))
    signals.append(test_signal(master, master["vix"]<15,  "VIX < 15 (calm)",  "Macro — Risk", 10))

    # Oil
    signals.append(test_signal(master, master["oil_bull"]==1, "Oil Uptrend",   "Macro — Oil", 20))
    signals.append(test_signal(master, master["oil_bear"]==1, "Oil Downtrend", "Macro — Oil", 20))

    # TASI
    if "tasi_20d" in master.columns:
        signals.append(test_signal(master, master["tasi_20d"]>0.05,  "TASI Uptrend >5%",  "Macro — TASI", 10))
        signals.append(test_signal(master, master["tasi_20d"]<-0.05, "TASI Downtrend >5%","Macro — TASI", 10))

    # Relative strength
    if "rs_vs_tasi_20d" in master.columns:
        signals.append(test_signal(master, master["rs_vs_tasi_20d"]>0.05,  "Outperform TASI >5%",  "Relative Strength", 10))
        signals.append(test_signal(master, master["rs_vs_tasi_20d"]<-0.05, "Underperform TASI >5%","Relative Strength", 10))

    # Mean reversion combos
    signals.append(test_signal(master,
        (master["rsi"]<30)&(master["rate_regime"]=="stable"),
        "RSI Oversold + Stable Rates", "Mean Reversion", 10))
    signals.append(test_signal(master,
        (master["rsi"]<30)&(master["vix"]<25),
        "RSI Oversold + Low VIX", "Mean Reversion", 10))
    signals.append(test_signal(master,
        (master["bb_pct"]<0.05)&(master["rate_regime"]=="stable"),
        "BB Lower + Stable Rates", "Mean Reversion", 10))

    # Volume surges
    signals.append(test_signal(master, master["vol_ratio"]>2.5, "Volume Surge >2.5x", "Technical — Volume", 5))

    # Distance from high
    signals.append(test_signal(master, master["dist_from_high"]<-0.20, "Price >20% Below 52w High", "Technical — Momentum", 10))

    valid = [s for s in signals if "avg_return_90d" in s]
    baseline = master.dropna(subset=["fwd_90d"])

    result = {
        "sym":      sym,
        "name":     name,
        "generated":datetime.now().isoformat(),
        "signals":  signals,
        "baseline": {
            "n":        len(baseline),
            "avg_30d":  round(baseline["fwd_30d"].mean()*100, 2),
            "avg_90d":  round(baseline["fwd_90d"].mean()*100, 2),
            "win_pct":  round((baseline["fwd_90d"]>0).mean()*100, 1),
        },
    }
    top = sorted(valid, key=lambda x: -x.get("reliability_score",0))[:5]
    print(f"    {len(signals)} signals tested | Top signal: {top[0]['signal'] if top else 'N/A'}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — MEMORY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_memory_engine(sym: str, name: str, master: pd.DataFrame,
                      income: pd.DataFrame, annual: pd.DataFrame,
                      divs: pd.DataFrame, events: list) -> dict:
    print(f"  Phase 2: Memory Engine...")
    memory = {"sym": sym, "name": name, "generated": datetime.now().isoformat()}

    # Technical memory
    tech_patterns = []
    for lo, hi, label in [(0,30,"Oversold"),(30,70,"Neutral"),(70,100,"Overbought")]:
        mask = (master["rsi"]>=lo)&(master["rsi"]<hi)
        sub  = master[mask].dropna(subset=["fwd_90d"])
        if len(sub) >= 5:
            tech_patterns.append({
                "label":   f"RSI {label}",
                "outcomes":{"30d": outcome_stats(sub["fwd_30d"]),
                            "90d": outcome_stats(sub["fwd_90d"])},
                "regime_breakdown": {
                    r: {"n": len(sub[sub["rate_regime"]==r]),
                        "avg_90d": round(sub[sub["rate_regime"]==r]["fwd_90d"].mean()*100,2)
                        if len(sub[sub["rate_regime"]==r])>=5 else None}
                    for r in ["rising","stable","falling"]
                }
            })
    memory["technical"] = tech_patterns

    # Macro / regime memory
    regime_profiles = []
    for rate_r in ["rising","stable","falling"]:
        sub = master[master["rate_regime"]==rate_r].dropna(subset=["fwd_90d"])
        if len(sub) >= 10:
            regime_profiles.append({
                "type":    "Rate Regime",
                "label":   rate_r,
                "n_days":  len(sub),
                "outcomes":{"30d": outcome_stats(sub["fwd_30d"]),
                            "90d": outcome_stats(sub["fwd_90d"])},
            })
    for vix_label, vix_cond in [("High Fear (VIX>30)", master["vix"]>30),
                                  ("Normal (VIX 15-30)", (master["vix"]>=15)&(master["vix"]<=30)),
                                  ("Calm (VIX<15)", master["vix"]<15)]:
        sub = master[vix_cond].dropna(subset=["fwd_90d"])
        if len(sub) >= 10:
            regime_profiles.append({
                "type":    "VIX Regime",
                "label":   vix_label,
                "n_days":  len(sub),
                "outcomes":{"90d": outcome_stats(sub["fwd_90d"])},
            })
    memory["regime_profiles"] = regime_profiles

    # Fundamental memory
    fund_history = []
    if not annual.empty:
        for _, row in annual.iterrows():
            fund_history.append({
                "year":          int(row["fiscal_year"]) if "fiscal_year" in row and not pd.isna(row["fiscal_year"]) else None,
                "report_date":   row["report_date"].date().isoformat(),
                "net_income_bn": round(row["net_income_bn"], 2) if not pd.isna(row.get("net_income_bn", float("nan"))) else None,
                "ni_yoy_pct":    round(row["ni_yoy"], 1) if not pd.isna(row.get("ni_yoy", float("nan"))) else None,
            })
    memory["fundamental"] = fund_history

    # Event memory
    ev_summary = {}
    from collections import Counter
    if events:
        type_counts = Counter(e.get("event_type","OTHER") for e in events)
        ev_reactions = {}
        for ev in events:
            ev_date = pd.to_datetime(ev.get("event_date"))
            ev_type = ev.get("event_type","OTHER")
            future  = master.index[master.index > ev_date]
            if len(future) >= 30:
                p0   = master["close"].loc[future[0]]
                ret30= (master["close"].iloc[master.index.get_loc(future[0])+29] / p0 - 1) * 100
                if ev_type not in ev_reactions:
                    ev_reactions[ev_type] = []
                ev_reactions[ev_type].append(round(ret30, 2))
        for ev_type, rets in ev_reactions.items():
            if len(rets) >= 3:
                s = pd.Series(rets)
                ev_summary[ev_type] = {
                    "n":            len(s),
                    "avg_30d_pct":  round(s.mean(), 2),
                    "pct_positive": round((s>0).mean()*100, 1),
                }
    memory["events"] = {"summary": ev_summary, "total_events": len(events)}

    # Dividend memory
    div_summary = {}
    if not divs.empty and "announcement_date" in divs.columns:
        div_rets = []
        for _, row in divs.iterrows():
            d = row.get("announcement_date")
            if pd.isna(d):
                continue
            future = master.index[master.index >= d]
            if len(future) >= 30:
                p0 = master["close"].loc[future[0]]
                r  = (master["close"].iloc[master.index.get_loc(future[0])+29] / p0 - 1) * 100
                div_rets.append(r)
        if div_rets:
            s = pd.Series(div_rets)
            div_summary = {"n": len(s), "avg_30d": round(s.mean(),2), "win_pct": round((s>0).mean()*100,1)}
    memory["dividends"] = div_summary

    # Regime-separated memory (RSI × Rate)
    regime_combos = []
    for rsi_label, rsi_cond in [("oversold",master["rsi"]<30),("neutral",(master["rsi"]>=40)&(master["rsi"]<=60)),("overbought",master["rsi"]>70)]:
        for rate_r in ["rising","stable","falling"]:
            sub = master[rsi_cond & (master["rate_regime"]==rate_r)].dropna(subset=["fwd_90d"])
            if len(sub) >= 10:
                regime_combos.append({
                    "id":       f"rsi_{rsi_label}_rate_{rate_r}",
                    "n":        len(sub),
                    "avg_30d":  round(sub["fwd_30d"].mean()*100, 2),
                    "avg_90d":  round(sub["fwd_90d"].mean()*100, 2),
                    "win_90d":  round((sub["fwd_90d"]>0).mean()*100, 1),
                })
    memory["regime_combos"] = regime_combos

    # Personality rules
    baseline_90d = master.dropna(subset=["fwd_90d"])["fwd_90d"].mean() * 100
    rules = []

    # RSI rule
    rsi_os = master[master["rsi"]<30].dropna(subset=["fwd_90d"])
    if len(rsi_os) >= 10:
        t, p = stats.ttest_1samp(rsi_os["fwd_90d"].dropna(), 0)
        diff  = rsi_os["fwd_90d"].mean()*100 - baseline_90d
        rules.append({
            "rule":       f"RSI oversold produces {'above' if diff>0 else 'below'}-average returns",
            "evidence":   f"RSI<30: avg={rsi_os['fwd_90d'].mean()*100:.1f}% vs baseline={baseline_90d:.1f}%, n={len(rsi_os)}, p={p:.3f}",
            "confidence": "HIGH" if p<0.05 else "MEDIUM",
        })

    # Rate regime rule
    stable = master[master["rate_regime"]=="stable"].dropna(subset=["fwd_90d"])
    rising = master[master["rate_regime"]=="rising"].dropna(subset=["fwd_90d"])
    if len(stable)>=20 and len(rising)>=10:
        diff = stable["fwd_90d"].mean()*100 - rising["fwd_90d"].mean()*100
        t, p = stats.ttest_ind(stable["fwd_90d"].dropna(), rising["fwd_90d"].dropna(), equal_var=False)
        rules.append({
            "rule":       f"Stable rates {'better' if diff>0 else 'worse'} than rising rates by {abs(diff):.1f}%",
            "evidence":   f"Stable: {stable['fwd_90d'].mean()*100:.1f}% avg, n={len(stable)} | Rising: {rising['fwd_90d'].mean()*100:.1f}%, n={len(rising)} | p={p:.3f}",
            "confidence": "HIGH" if p<0.05 and abs(diff)>3 else "MEDIUM",
        })

    # Momentum rule
    mom_up = master[master["ret_20d"]>0.10].dropna(subset=["fwd_90d"])
    if len(mom_up) >= 10:
        diff = mom_up["fwd_90d"].mean()*100 - baseline_90d
        rules.append({
            "rule":       f"Strong 20-day momentum (>10% rally) {'reverses' if diff<0 else 'continues'}",
            "evidence":   f"After >10% rally: avg={mom_up['fwd_90d'].mean()*100:.1f}% vs baseline={baseline_90d:.1f}%, n={len(mom_up)}",
            "confidence": "MEDIUM",
        })

    memory["personality_rules"] = rules
    memory["baseline_90d"] = {
        "avg_pct": round(baseline_90d, 2),
        "win_pct": round((master.dropna(subset=["fwd_90d"])["fwd_90d"]>0).mean()*100, 1),
        "n":       len(master.dropna(subset=["fwd_90d"])),
    }

    print(f"    {len(regime_profiles)} regime profiles | {len(fund_history)} fundamental periods | {len(rules)} personality rules")
    return memory


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — HYPOTHESIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_hypothesis_engine(sym: str, name: str, master: pd.DataFrame) -> dict:
    print(f"  Phase 3: Hypothesis Engine...")
    results = []

    def hyp(h_id, statement, group_a, group_b=None, metric="fwd_90d", min_n=10):
        valid  = master.dropna(subset=[metric])
        sub_a  = valid[group_a.reindex(valid.index, fill_value=False)]
        sub_b  = valid[group_b.reindex(valid.index, fill_value=False)] if group_b is not None else valid
        r = {"id": h_id, "hypothesis": statement, "n_a": len(sub_a), "n_b": len(sub_b)}
        if len(sub_a) < min_n:
            r.update({"verdict": "INCONCLUSIVE", "reason": f"Insufficient sample (n={len(sub_a)})"})
            return r
        a, b = sub_a[metric].dropna(), sub_b[metric].dropna()
        r["avg_a"]     = round(a.mean()*100, 2)
        r["avg_b"]     = round(b.mean()*100, 2)
        r["diff"]      = round(r["avg_a"] - r["avg_b"], 2)
        r["win_pct_a"] = round((a>0).mean()*100, 1)
        t, p = stats.ttest_ind(a, b, equal_var=False)
        r["p_value"] = round(p, 3)
        meaningful = abs(r["diff"]) > 2.0
        if p < 0.05 and meaningful:
            r["verdict"] = "ACCEPTED" if r["diff"] > 0 else "REJECTED"
            r["reason"]  = f"p={p:.3f} | A={r['avg_a']}% vs B={r['avg_b']}% | {'confirmed' if r['diff']>0 else 'reversed'}"
        elif not meaningful:
            r["verdict"] = "REJECTED"
            r["reason"]  = f"Difference too small ({r['diff']}%)"
        else:
            r["verdict"] = "INCONCLUSIVE"
            r["reason"]  = f"p={p:.3f} — not significant"
        return r

    # Universal hypotheses — run for every stock
    results.append(hyp("H01", "Rising rates hurt this stock",
        master["rate_regime"]=="rising", master["rate_regime"]=="stable"))
    results.append(hyp("H02", "Stable rates are best environment",
        master["rate_regime"]=="stable",
        (master["rate_regime"]=="rising")|(master["rate_regime"]=="falling")))
    results.append(hyp("H03", "RSI below 30 produces better-than-baseline 90d returns",
        master["rsi"]<30))
    results.append(hyp("H04", "RSI above 70 produces worse-than-baseline 90d returns",
        master["rsi"]>70))
    results.append(hyp("H05", "High VIX (>30) creates buying opportunities",
        master["vix"]>30))
    results.append(hyp("H06", "Oil uptrend improves returns vs oil downtrend",
        master["oil_bull"]==1, master["oil_bear"]==1))
    results.append(hyp("H07", "Strong 20d momentum (>10%) predicts negative 90d (mean reversion)",
        master["ret_20d"]>0.10,
        (master["ret_20d"]>=-0.05)&(master["ret_20d"]<=0.05)))
    results.append(hyp("H08", "BB lower band touch predicts positive 30d returns",
        master["bb_pct"]<0.05, metric="fwd_30d"))
    results.append(hyp("H09", "BB upper band touch predicts negative 30d returns",
        master["bb_pct"]>0.95, metric="fwd_30d"))
    results.append(hyp("H10", "Below all MAs predicts positive 90d returns",
        (master["above_ma20"]==0)&(master["above_ma50"]==0)&(master["above_ma200"]==0)))
    results.append(hyp("H11", "Above all MAs predicts negative 90d returns",
        (master["above_ma20"]==1)&(master["above_ma50"]==1)&(master["above_ma200"]==1)))
    results.append(hyp("H12", "RSI oversold + stable rates outperforms RSI oversold + rising rates",
        (master["rsi"]<30)&(master["rate_regime"]=="stable"),
        (master["rsi"]<30)&(master["rate_regime"]=="rising")))
    results.append(hyp("H13", "Falling rates better than rising rates",
        master["rate_regime"]=="falling", master["rate_regime"]=="rising"))
    results.append(hyp("H14", "TASI uptrend improves stock returns",
        master["tasi_20d"]>0.05 if "tasi_20d" in master.columns else pd.Series(False, index=master.index),
        master["tasi_20d"]<-0.05 if "tasi_20d" in master.columns else pd.Series(False, index=master.index)))
    results.append(hyp("H15", "VIX fear spike followed by calm is a buy signal",
        (master["vix"].shift(10)>30)&(master["vix"]<25)))
    results.append(hyp("H16", "Outperforming TASI >5% over 20d predicts underperformance",
        master["rs_vs_tasi_20d"]>0.05 if "rs_vs_tasi_20d" in master.columns else pd.Series(False, index=master.index),
        (master["rs_vs_tasi_20d"]>=-0.02)&(master["rs_vs_tasi_20d"]<=0.02) if "rs_vs_tasi_20d" in master.columns else pd.Series(True, index=master.index)))
    results.append(hyp("H17", "Price >20% below 52w high is a buying opportunity",
        master["dist_from_high"]<-0.20))
    results.append(hyp("H18", "Volume surge (>2.5x) is predictive of 30d returns",
        master["vol_ratio"]>2.5, metric="fwd_30d"))
    results.append(hyp("H19", "RSI oversold + VIX<25 is better than RSI oversold alone",
        (master["rsi"]<30)&(master["vix"]<25),
        (master["rsi"]<30)&(master["vix"]>=25)))
    results.append(hyp("H20", "Stable rates + oil uptrend is best combined environment",
        (master["rate_regime"]=="stable")&(master["oil_bull"]==1)))

    # Sector-specific hypotheses
    sector = ALL_STOCKS.get(sym, {}).get("sector", "")
    if sector == "Mining":
        results.append(hyp("H21", "Oil uptrend especially benefits mining stock (commodity correlation)",
            master["oil_bull"]==1))
        results.append(hyp("H22", "Oil downtrend especially hurts mining stock",
            master["oil_bear"]==1, master["oil_bull"]==1))
    if sector == "Pharmaceuticals":
        results.append(hyp("H21", "Stable rates produce consistent pharma returns",
            master["rate_regime"]=="stable"))
        results.append(hyp("H22", "Low VIX environment is better for pharma",
            master["vix"]<15, master["vix"]>20))
    if sector == "Chemicals":
        results.append(hyp("H21", "Oil uptrend benefits chemical company differently than banks",
            master["oil_bull"]==1))
        results.append(hyp("H22", "Rising rates may actually help chemicals (unlike banks)",
            master["rate_regime"]=="rising", master["rate_regime"]=="stable"))

    accepted     = [r for r in results if r.get("verdict")=="ACCEPTED"]
    rejected     = [r for r in results if r.get("verdict")=="REJECTED"]
    inconclusive = [r for r in results if r.get("verdict")=="INCONCLUSIVE"]

    print(f"    {len(results)} hypotheses | {len(accepted)} accepted | {len(rejected)} rejected | {len(inconclusive)} inconclusive")
    return {
        "sym":      sym,
        "name":     name,
        "generated":datetime.now().isoformat(),
        "results":  results,
        "summary":  {"total": len(results), "accepted": len(accepted),
                     "rejected": len(rejected), "inconclusive": len(inconclusive)},
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — WALK-FORWARD + MISTAKE VAULT
# ══════════════════════════════════════════════════════════════════════════════

def run_walkforward(sym: str, name: str, master: pd.DataFrame, annual: pd.DataFrame) -> dict:
    print(f"  Phase 4: Walk-Forward Predictions...")

    weights = {"environment": 0.45, "technical": 0.30, "fundamental": 0.25}

    # Determine start date — need at least 200 days of history before predicting
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

    predictions  = []
    mistake_vault= []

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

        # Fundamental score
        fund = 50
        if not annual.empty:
            known = annual[annual["report_date"] < pred_date]
            if len(known) >= 2:
                ni_yoy = known.iloc[-1].get("ni_yoy", None)
                if ni_yoy is not None and not pd.isna(ni_yoy):
                    if ni_yoy > 20: fund += 15
                    elif ni_yoy > 10: fund += 8
                    elif ni_yoy > 0: fund += 2
                    elif ni_yoy < -10: fund -= 15
        fund = max(0, min(100, fund))

        comp = env * weights["environment"] + tech * weights["technical"] + fund * weights["fundamental"]

        rate_regime = row.get("rate_regime", "stable")
        if comp > 70:   base_90d = 7.5
        elif comp > 60: base_90d = 5.0
        elif comp > 50: base_90d = 3.0
        elif comp > 40: base_90d = 1.0
        else:           base_90d = -3.0
        if rate_regime == "rising": base_90d -= 5.0
        elif rate_regime == "falling": base_90d -= 1.0

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
                    actual[f"actual_{w}d_pct"]   = round((p_fut / price - 1) * 100, 1)
                    actual[f"hit_target_{w}d"]    = (actual[f"actual_{w}d_pct"] > 0) == (base_90d > 0)

        pred = {
            "prediction_date": pred_date.date().isoformat(),
            "price":           round(price, 2),
            "rate_regime":     rate_regime,
            "rsi":             round(float(rsi), 1) if not pd.isna(rsi) else None,
            "vix":             round(float(vix), 1),
            "env_score":       round(env, 1),
            "tech_score":      round(tech, 1),
            "fund_score":      round(fund, 1),
            "composite":       round(comp, 1),
            "base_30d_pct":    round(base_30d, 1),
            "base_60d_pct":    round(base_60d, 1),
            "base_90d_pct":    round(base_90d, 1),
            "confidence":      confidence,
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
                "rate_regime":     rate_regime,
                "rsi":             round(float(rsi), 1) if not pd.isna(rsi) else None,
                "vix":             round(float(vix), 1),
                "root_cause":      "Model underestimated upside" if actual["actual_90d_pct"] > base_90d + 10
                                   else "Model underestimated downside" if actual["actual_90d_pct"] < base_90d - 10
                                   else "Mixed signals",
            })

    # Validation
    df = pd.DataFrame(predictions)
    df = df[df["actual_90d_pct"].notna()] if "actual_90d_pct" in df.columns else df
    validation = {}
    if len(df) >= 5 and "actual_90d_pct" in df.columns:
        df["pred_dir"]    = (df["base_90d_pct"] > 0).astype(int)
        df["actual_dir"]  = (df["actual_90d_pct"] > 0).astype(int)
        df["correct"]     = (df["pred_dir"] == df["actual_dir"]).astype(int)
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
        }
        regime_acc = {}
        for r in ["rising","stable","falling"]:
            sub = df[df["rate_regime"]==r]
            if len(sub) >= 3:
                regime_acc[r] = {"n": len(sub), "accuracy": round(sub["correct"].mean()*100,1)}
        validation["regime_breakdown"] = regime_acc

    print(f"    {len(predictions)} predictions | accuracy={validation.get('directional_accuracy_pct','?')}% | "
          f"baseline={validation.get('baseline_always_long_pct','?')}% | mistakes={len(mistake_vault)}")

    return {
        "sym":         sym,
        "name":        name,
        "generated":   datetime.now().isoformat(),
        "validation":  validation,
        "predictions": predictions,
        "mistake_vault": mistake_vault,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — RECALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def run_recalibration(sym: str, name: str, hyp_results: dict, wf_results: dict, master: pd.DataFrame) -> dict:
    print(f"  Phase 5: Recalibration...")
    validation = wf_results.get("validation", {})
    mistakes   = wf_results.get("mistake_vault", [])
    hyps       = hyp_results.get("results", [])

    weights = {"environment": 0.45, "technical": 0.30, "fundamental": 0.25}
    log     = []

    # Check if rising rates hypothesis was rejected (opposite)
    h01 = next((h for h in hyps if h["id"]=="H01"), {})
    if h01.get("verdict") == "REJECTED" and h01.get("diff", 0) < 0:
        log.append({"factor": "Rising rate penalty", "action": "CONFIRMED — keep at -25pts",
                    "evidence": f"H01 confirmed rising rates hurt: diff={h01.get('diff')}%"})
    elif h01.get("verdict") == "ACCEPTED":
        log.append({"factor": "Rising rate penalty", "action": "REDUCE — rising rates help this stock",
                    "evidence": f"H01: rising rates help: diff={h01.get('diff')}%"})

    # Magnitude bias
    avg_pred   = validation.get("avg_predicted_90d", 0)
    avg_actual = validation.get("avg_actual_90d", 0)
    bias       = avg_actual - avg_pred if avg_pred and avg_actual else 0
    if abs(bias) > 3:
        log.append({"factor": "Base return magnitude", "action": f"Adjust upward by {bias:.1f}%",
                    "evidence": f"Systematic {'underestimate' if bias>0 else 'overestimate'}: predicted={avg_pred}%, actual={avg_actual}%"})

    # Large errors
    upside_misses = [m for m in mistakes if m.get("actual_90d", 0) > m.get("predicted_90d", 0) + 10]
    if len(upside_misses) > len(mistakes) * 0.6:
        log.append({"factor": "Upside capture", "action": "Model consistently underestimates upside — increase bull spread",
                    "evidence": f"{len(upside_misses)}/{len(mistakes)} large errors were upside misses"})

    print(f"    {len(log)} recalibration actions | bias={bias:+.1f}% | accuracy={validation.get('directional_accuracy_pct','?')}%")
    return {
        "sym":       sym,
        "name":      name,
        "generated": datetime.now().isoformat(),
        "weights":   weights,
        "log":       log,
        "bias":      round(bias, 2),
        "before_accuracy": validation.get("directional_accuracy_pct"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SAVE ALL OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(sym: str, signals: dict, memory: dict, hyp: dict, wf: dict, recal: dict):
    prefix = f"{sym}_" if sym != "1120" else ""

    with open(REPORTS_DIR / f"{prefix}SIGNAL_DISCOVERY_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(signals, f, indent=2, default=str)
    with open(MEMORY_DIR / f"memory_{sym}.json", "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, default=str, ensure_ascii=False)
    with open(REPORTS_DIR / f"{prefix}HYPOTHESIS_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(hyp, f, indent=2, default=str)
    with open(REPORTS_DIR / f"{prefix}PREDICTION_BACKTEST_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(wf, f, indent=2, default=str)
    with open(REPORTS_DIR / f"{prefix}RECALIBRATION_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(recal, f, indent=2, default=str)
    with open(MEMORY_DIR / f"mistake_vault_{sym}.json", "w", encoding="utf-8") as f:
        json.dump({"mistakes": wf.get("mistake_vault", [])}, f, indent=2, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_stock(sym: str):
    info = ALL_STOCKS.get(sym)
    if not info:
        print(f"Unknown symbol: {sym}")
        return

    name   = info["name"]
    sector = info["sector"]
    print(f"\n{'='*60}")
    print(f"  FULL ENGINE: {name} ({sym}) — {sector}")
    print(f"{'='*60}")

    master = load_master(sym)
    print(f"  Data: {len(master)} trading days | {master.index[0].date()} → {master.index[-1].date()}")

    income, annual = load_financials(sym)
    divs   = load_dividends(sym)
    events = load_events(sym)
    print(f"  Financials: {len(annual)} annual periods | Dividends: {len(divs)} events | Events: {len(events)}")

    signals = run_signal_discovery(sym, name, master)
    memory  = run_memory_engine(sym, name, master, income, annual, divs, events)
    hyp     = run_hypothesis_engine(sym, name, master)
    wf      = run_walkforward(sym, name, master, annual)
    recal   = run_recalibration(sym, name, hyp, wf, master)

    save_outputs(sym, signals, memory, hyp, wf, recal)

    v = wf.get("validation", {})
    print(f"\n  RESULT SUMMARY — {name}")
    print(f"  Directional accuracy: {v.get('directional_accuracy_pct','?')}%")
    print(f"  Baseline:             {v.get('baseline_always_long_pct','?')}%")
    print(f"  Edge:                 {v.get('edge_over_baseline_pct',0):+.1f}%")
    print(f"  Hypotheses accepted:  {hyp['summary']['accepted']}/{hyp['summary']['total']}")
    print(f"  Mistake vault:        {len(wf.get('mistake_vault',[]))} large errors")
    print(f"  All files saved to reports/ and memory/")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target == "all":
        symbols = [s for s in ALL_STOCKS if s != "1120"]  # 1120 already done
    else:
        symbols = [target]

    print(f"\nSTOCK MEMORY ENGINE v2 — FULL ENGINE RUNNER")
    print(f"Date: {date.today()}")
    print(f"Target stocks: {[ALL_STOCKS[s]['name'] for s in symbols if s in ALL_STOCKS]}")

    for sym in symbols:
        if sym not in ALL_STOCKS:
            print(f"Unknown: {sym}")
            continue
        run_stock(sym)

    # Rebuild current states
    print("\n\nRebuilding current states for portal...")
    import subprocess
    subprocess.run([sys.executable, "build_all_forecasts.py"])
    print("\nAll done. Portal AI context will now have full data for all stocks.")


if __name__ == "__main__":
    main()
