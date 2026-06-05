"""
Generate current state + forecast for ALL stocks.
Saves to memory/current_state_{sym}.json for each stock.
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from datetime import date, datetime
from pathlib import Path
from config.settings import DATA_RAW, DATA_PROCESSED, MEMORY_DIR

STOCKS = {
    "1120": {"name": "Al Rajhi Bank",            "yahoo": "1120.SR", "sector": "Banking"},
    "1180": {"name": "Saudi National Bank (SNB)", "yahoo": "1180.SR", "sector": "Banking"},
    "1010": {"name": "Riyad Bank",                "yahoo": "1010.SR", "sector": "Banking"},
    "1060": {"name": "Saudi British Bank (SABB)", "yahoo": "1060.SR", "sector": "Banking"},
    "1080": {"name": "Arab National Bank",        "yahoo": "1080.SR", "sector": "Banking"},
    "1050": {"name": "Banque Saudi Fransi",       "yahoo": "1050.SR", "sector": "Banking"},
    "1150": {"name": "Alinma Bank",               "yahoo": "1150.SR", "sector": "Banking"},
    "1140": {"name": "Bank Albilad",              "yahoo": "1140.SR", "sector": "Banking"},
    "2230": {"name": "Saudi Chemical",            "yahoo": "2230.SR", "sector": "Chemicals"},
    "1211": {"name": "Maaden",                    "yahoo": "1211.SR", "sector": "Mining"},
    "4015": {"name": "Jamjoom Pharma",            "yahoo": "4015.SR", "sector": "Pharmaceuticals"},
    "2222": {"name": "Saudi Aramco", "yahoo": "2222.SR", "sector": "Energy"},
    "2010": {"name": "SABIC", "yahoo": "2010.SR", "sector": "Petrochemicals"},
    "7010": {"name": "STC", "yahoo": "7010.SR", "sector": "Telecom"},
    # Retail & Consumer
    "4190": {"name": "Jarir Bookstore", "yahoo": "4190.SR", "sector": "Retail"},
    "4012": {"name": "Extra (United Electronics)", "yahoo": "4012.SR", "sector": "Retail"},
    "4163": {"name": "Nahdi Medical", "yahoo": "4163.SR", "sector": "Retail"},
    "4161": {"name": "BinDawood Holding", "yahoo": "4161.SR", "sector": "Retail"},
    "4001": {"name": "Al Othaim Markets", "yahoo": "4001.SR", "sector": "Retail"},
    "4082": {"name": "SACO", "yahoo": "4082.SR", "sector": "Retail"},
    "6004": {"name": "Fawaz Abdulaziz Alhokair", "yahoo": "6004.SR", "sector": "Retail"},
    "4003": {"name": "Astra Industrial", "yahoo": "4003.SR", "sector": "Retail"},

    "7020": {"name": "Mobily", "yahoo": "7020.SR", "sector": "Telecom"},
    "7030": {"name": "Zain Saudi", "yahoo": "7030.SR", "sector": "Telecom"},
    "7040": {"name": "Etihad Atheeb", "yahoo": "7040.SR", "sector": "Telecom"},
    "7200": {"name": "Solutions by STC", "yahoo": "7200.SR", "sector": "Telecom"},

    "2280": {"name": "Almarai", "yahoo": "2280.SR", "sector": "Food & Beverages"},
    # Petrochemicals
    "2020": {"name": "SABIC Agri-Nutrients", "yahoo": "2020.SR", "sector": "Petrochemicals"},
    "2290": {"name": "Yanbu National Petro", "yahoo": "2290.SR", "sector": "Petrochemicals"},
    "2350": {"name": "Saudi Kayan Petrochem", "yahoo": "2350.SR", "sector": "Petrochemicals"},
    "2310": {"name": "Sipchem", "yahoo": "2310.SR", "sector": "Petrochemicals"},
    "2380": {"name": "Petro Rabigh", "yahoo": "2380.SR", "sector": "Petrochemicals"},
    "2210": {"name": "Nama Chemicals", "yahoo": "2210.SR", "sector": "Petrochemicals"},
    "2060": {"name": "National Industrialization", "yahoo": "2060.SR", "sector": "Petrochemicals"},
    "2030": {"name": "Advanced Petrochem", "yahoo": "2030.SR", "sector": "Petrochemicals"},
    "2260": {"name": "Sahara International Petrochem", "yahoo": "2260.SR", "sector": "Petrochemicals"},
    "2250": {"name": "Saudi Industrial Investment", "yahoo": "2250.SR", "sector": "Petrochemicals"},

}

def load_master(sym):
    p = DATA_PROCESSED / f"master_{sym}.csv"
    if sym == "1120":
        p = DATA_PROCESSED / "master_dataset.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def current_state(sym, info, master):
    if master.empty:
        return {}
    row = master.iloc[-1]
    c   = master["close"]

    # RSI
    rsi = float(row.get("rsi", 50)) if not pd.isna(row.get("rsi", float("nan"))) else 50.0

    # Momentum
    ret_20d = float(row.get("ret_20d", 0)) if not pd.isna(row.get("ret_20d", float("nan"))) else 0.0
    ret_5d  = float(c.pct_change(5).iloc[-1]) if len(c) > 5 else 0.0

    # BB position
    bb_pct = float(row.get("bb_pct", 0.5)) if not pd.isna(row.get("bb_pct", float("nan"))) else 0.5

    # Rate regime
    rate_regime = str(row.get("rate_regime", "stable"))
    repo_rate   = float(row.get("repo_rate", 4.25)) if not pd.isna(row.get("repo_rate", float("nan"))) else 4.25

    # VIX
    vix = float(row.get("vix", 20)) if not pd.isna(row.get("vix", float("nan"))) else 20.0

    # Oil
    oil = float(row.get("oil", 80)) if not pd.isna(row.get("oil", float("nan"))) else 80.0
    oil_bull = int(row.get("oil_bull", 0))

    # MA positions
    above_ma20  = int(row.get("above_ma20",  1))
    above_ma50  = int(row.get("above_ma50",  1))
    above_ma200 = int(row.get("above_ma200", 1))

    # 52w high/low
    high_52w     = float(c.rolling(252).max().iloc[-1]) if len(c) >= 252 else float(c.max())
    low_52w      = float(c.rolling(252).min().iloc[-1]) if len(c) >= 252 else float(c.min())
    dist_high    = (row["close"] - high_52w) / high_52w * 100
    dist_low     = (row["close"] - low_52w)  / low_52w  * 100

    # Vol ratio
    vol_ratio = float(row.get("vol_ratio", 1.0)) if not pd.isna(row.get("vol_ratio", float("nan"))) else 1.0

    # ── Environment Score ──────────────────────────────────────────────────
    env_score = 50
    env_sigs  = {}
    if rate_regime == "stable":
        env_score += 20; env_sigs["rate"] = "stable (+20)"
    elif rate_regime == "rising":
        env_score -= 25; env_sigs["rate"] = "rising (-25)"
    elif rate_regime == "falling":
        env_score -= 5;  env_sigs["rate"] = "falling (-5)"

    if vix > 30:
        env_score += 15; env_sigs["vix"] = f"{vix:.1f} fear=opportunity (+15)"
    elif vix > 20:
        env_score += 5;  env_sigs["vix"] = f"{vix:.1f} elevated (+5)"
    elif vix < 15:
        env_score -= 5;  env_sigs["vix"] = f"{vix:.1f} calm (-5)"

    if oil_bull:
        env_score += 10; env_sigs["oil"] = "uptrend (+10)"

    env_score = max(0, min(100, env_score))

    # ── Technical Score ────────────────────────────────────────────────────
    tech_score = 50
    tech_sigs  = {}

    if rsi < 20:
        tech_score += 20; tech_sigs["rsi"] = f"{rsi:.1f} extreme oversold (+20)"
    elif rsi < 30:
        tech_score += 15; tech_sigs["rsi"] = f"{rsi:.1f} oversold (+15)"
    elif rsi < 40:
        tech_score += 8;  tech_sigs["rsi"] = f"{rsi:.1f} low (+8)"
    elif rsi > 80:
        tech_score -= 20; tech_sigs["rsi"] = f"{rsi:.1f} extreme overbought (-20)"
    elif rsi > 70:
        tech_score -= 12; tech_sigs["rsi"] = f"{rsi:.1f} overbought (-12)"
    else:
        tech_sigs["rsi"] = f"{rsi:.1f} neutral"

    if bb_pct < 0.05:
        tech_score += 20; tech_sigs["bb"] = f"near lower band ({bb_pct:.2f}) (+20)"
    elif bb_pct < 0.15:
        tech_score += 10; tech_sigs["bb"] = f"low ({bb_pct:.2f}) (+10)"
    elif bb_pct > 0.95:
        tech_score -= 15; tech_sigs["bb"] = f"near upper band ({bb_pct:.2f}) (-15)"
    else:
        tech_sigs["bb"] = f"mid band ({bb_pct:.2f})"

    if ret_20d > 0.15:
        tech_score -= 20; tech_sigs["momentum"] = f"{ret_20d:.1%} strong rally — caution (-20)"
    elif ret_20d > 0.10:
        tech_score -= 10; tech_sigs["momentum"] = f"{ret_20d:.1%} rally (-10)"
    elif ret_20d < -0.15:
        tech_score += 15; tech_sigs["momentum"] = f"{ret_20d:.1%} sharp drop=opportunity (+15)"
    elif ret_20d < -0.10:
        tech_score += 8;  tech_sigs["momentum"] = f"{ret_20d:.1%} drop (+8)"
    else:
        tech_sigs["momentum"] = f"{ret_20d:.1%} neutral"

    if above_ma20 == 0 and above_ma50 == 0 and above_ma200 == 0:
        tech_score += 8; tech_sigs["ma"] = "below all MAs (+8)"
    elif above_ma20 == 1 and above_ma50 == 1 and above_ma200 == 1:
        tech_score -= 8; tech_sigs["ma"] = "above all MAs (-8)"

    tech_score = max(0, min(100, tech_score))

    # ── Load fundamental score from personality ────────────────────────────
    fund_score = 50
    fund_sigs  = {}
    p_path = MEMORY_DIR / f"personality_{sym}.json"
    if p_path.exists():
        with open(p_path, encoding="utf-8") as f:
            pers = json.load(f)
        baseline = pers.get("baseline_90d", {})
        rules    = pers.get("personality_rules", [])
        fund_sigs["baseline_90d_avg"] = f"{baseline.get('avg_pct','?')}%"
        fund_sigs["baseline_win_pct"] = f"{baseline.get('win_pct','?')}%"
        # Load latest financials if available
        fin_path = DATA_RAW / (f"stock_{sym}" if sym != "1120" else "") / "alrajhi_income_quarterly.csv" if sym == "1120" else DATA_RAW / f"stock_{sym}" / "financials_quarterly.json"
        if sym == "1120":
            fin_csv = DATA_RAW / "alrajhi_income_quarterly.csv"
            if fin_csv.exists():
                df_fin = pd.read_csv(fin_csv, parse_dates=["report_date"])
                df_fin = df_fin.sort_values("report_date")
                annual = df_fin[df_fin["report_date"].dt.month == 12]
                if len(annual) >= 2:
                    latest_ni = annual["net_income"].iloc[-1]
                    prev_ni   = annual["net_income"].iloc[-2]
                    ni_yoy    = (latest_ni - prev_ni) / abs(prev_ni) * 100
                    if ni_yoy > 20:
                        fund_score += 15; fund_sigs["net_income"] = f"+{ni_yoy:.1f}% YoY strong (+15)"
                    elif ni_yoy > 10:
                        fund_score += 8;  fund_sigs["net_income"] = f"+{ni_yoy:.1f}% YoY solid (+8)"
                    elif ni_yoy > 0:
                        fund_score += 2;  fund_sigs["net_income"] = f"+{ni_yoy:.1f}% YoY (+2)"
                    elif ni_yoy < -10:
                        fund_score -= 15; fund_sigs["net_income"] = f"{ni_yoy:.1f}% YoY declining (-15)"
        else:
            fin_json = DATA_RAW / f"stock_{sym}" / "financials_quarterly.json"
            if fin_json.exists():
                with open(fin_json, encoding="utf-8") as f:
                    fin_data = json.load(f)
                income = fin_data.get("income_statements", [])
                if len(income) >= 2:
                    ni_latest = income[0].get("net_income")
                    ni_prev   = income[1].get("net_income")
                    if ni_latest and ni_prev and ni_prev != 0:
                        ni_growth = (ni_latest - ni_prev) / abs(ni_prev) * 100
                        if ni_growth > 20:
                            fund_score += 12; fund_sigs["net_income"] = f"+{ni_growth:.1f}% QoQ strong (+12)"
                        elif ni_growth > 0:
                            fund_score += 5;  fund_sigs["net_income"] = f"+{ni_growth:.1f}% QoQ (+5)"
                        elif ni_growth < -20:
                            fund_score -= 12; fund_sigs["net_income"] = f"{ni_growth:.1f}% QoQ declining (-12)"

    fund_score = max(0, min(100, fund_score))

    # ── Composite Score ────────────────────────────────────────────────────
    composite = env_score * 0.45 + tech_score * 0.30 + fund_score * 0.25

    # ── Forecast ───────────────────────────────────────────────────────────
    # Load personality baseline for calibration
    pers_data  = {}
    if p_path.exists():
        with open(p_path, encoding="utf-8") as f:
            pers_data = json.load(f)
    baseline_90d = pers_data.get("baseline_90d", {}).get("avg_pct", 6.0) or 6.0
    baseline_win = pers_data.get("baseline_90d", {}).get("win_pct", 60.0) or 60.0

    # Regime-adjusted base return
    rd = pers_data.get("key_signals", {}).get("regime_breakdown", {})
    regime_avg = rd.get(rate_regime, {}).get("avg_90d", baseline_90d) or baseline_90d

    # Score-adjusted return
    if composite > 70:   base_90d = max(regime_avg, baseline_90d) * 1.2
    elif composite > 60: base_90d = regime_avg
    elif composite > 50: base_90d = regime_avg * 0.7
    elif composite > 40: base_90d = regime_avg * 0.3
    else:                base_90d = min(regime_avg * -0.5, -2.0)

    base_30d = base_90d * 0.40
    base_60d = base_90d * 0.70
    spread   = 5.0 + (1 - abs(composite - 50) / 50) * 5.0

    confidence = int(40 + abs(composite - 50) / 50 * 35 + (1 - abs(env_score - tech_score) / 100) * 25)
    confidence = max(20, min(90, confidence))

    price = float(row["close"])

    # Similar historical setups
    valid = master.dropna(subset=["fwd_90d"])
    rsi_match  = (abs(valid["rsi"] - rsi) < 10) if "rsi" in valid else pd.Series(True, index=valid.index)
    rate_match = valid["rate_regime"] == rate_regime if "rate_regime" in valid else pd.Series(True, index=valid.index)
    similar    = valid[rsi_match & rate_match]
    sim_stats  = {}
    if len(similar) >= 5:
        sim_stats = {
            "n": len(similar),
            "avg_90d": round(similar["fwd_90d"].mean() * 100, 1),
            "win_pct": round((similar["fwd_90d"] > 0).mean() * 100, 1),
        }

    state = {
        "symbol":          sym,
        "name":            info["name"],
        "sector":          info["sector"],
        "as_of":           date.today().isoformat(),
        "price":           round(price, 2),
        "price_history":   f"{master.index[0].date()} → {master.index[-1].date()}",
        "trading_days":    len(master),
        "rsi":             round(rsi, 1),
        "bb_pct":          round(bb_pct, 3),
        "ret_20d_pct":     round(ret_20d * 100, 1),
        "ret_5d_pct":      round(ret_5d  * 100, 1),
        "above_ma20":      above_ma20,
        "above_ma50":      above_ma50,
        "above_ma200":     above_ma200,
        "dist_from_52w_high_pct": round(dist_high, 1),
        "dist_from_52w_low_pct":  round(dist_low, 1),
        "vol_ratio":       round(vol_ratio, 2),
        "rate_regime":     rate_regime,
        "repo_rate":       round(repo_rate, 2),
        "vix":             round(vix, 1),
        "oil_price":       round(oil, 1),
        "env_score":       round(env_score, 1),
        "tech_score":      round(tech_score, 1),
        "fund_score":      round(fund_score, 1),
        "composite":       round(composite, 1),
        "env_signals":     env_sigs,
        "tech_signals":    tech_sigs,
        "fund_signals":    fund_sigs,
        "baseline_90d_avg": baseline_90d,
        "baseline_win_pct": baseline_win,
        "regime_90d_avg":   round(regime_avg, 1),
        "forecast": {
            "base_30d_pct":  round(base_30d, 1),
            "base_60d_pct":  round(base_60d, 1),
            "base_90d_pct":  round(base_90d, 1),
            "bull_90d_pct":  round(base_90d + spread, 1),
            "bear_90d_pct":  round(base_90d - spread, 1),
            "target_30d":    round(price * (1 + base_30d/100), 2),
            "target_60d":    round(price * (1 + base_60d/100), 2),
            "target_90d":    round(price * (1 + base_90d/100), 2),
            "bull_target":   round(price * (1 + (base_90d + spread)/100), 2),
            "bear_target":   round(price * (1 + (base_90d - spread)/100), 2),
            "confidence":    confidence,
            "confidence_label": "HIGH" if confidence >= 65 else ("MEDIUM" if confidence >= 45 else "LOW"),
        },
        "similar_setups": sim_stats,
    }
    return state


def main():
    print("Building current state for all stocks...")
    all_states = {}

    for sym, info in STOCKS.items():
        master = load_master(sym)
        if master.empty:
            print(f"  {sym}: no master dataset")
            continue
        state = current_state(sym, info, master)
        all_states[sym] = state

        path = MEMORY_DIR / f"current_state_{sym}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)

        fc = state.get("forecast", {})
        print(f"  {info['name']} ({sym}): price={state['price']} SAR | RSI={state['rsi']} | "
              f"composite={state['composite']}/100 | 90d={fc.get('base_90d_pct',0):+.1f}% | "
              f"conf={fc.get('confidence_label','?')}")

    # Save combined
    with open(MEMORY_DIR / "all_current_states.json", "w", encoding="utf-8") as f:
        json.dump({"generated": datetime.now().isoformat(), "stocks": all_states}, f, indent=2, default=str)

    print(f"\nAll states saved to memory/current_state_{{sym}}.json")


if __name__ == "__main__":
    main()
