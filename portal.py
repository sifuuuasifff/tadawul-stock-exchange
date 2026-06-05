"""
TADAWUL STOCK MEMORY ENGINE — HOME (Summary Dashboard)
Page 1 of 2.  Run:  streamlit run portal.py
"""

import sys, os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd

from shared import STOCKS, load_all_states, build_stock_df, CSS

st.set_page_config(
    page_title="Tadawul Stock Memory Engine",
    page_icon="📊", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)


def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown("### 📊 Tadawul Stock Engine")
        st.markdown(f"*{len(STOCKS)} stocks · {len(set(v['sector'] for v in STOCKS.values()))} sectors*")
        st.markdown("---")
        st.markdown("**🏆 Top Picks Right Now**")
        for _, r in df.sort_values("composite", ascending=False).head(5).iterrows():
            c = r["composite"]
            col = "green" if c >= 70 else "orange"
            st.markdown(f"**:{col}[{r['emoji']} {r['name']}]** {c:.0f}/100 | {r['forecast_90d']:+.1f}%")
        st.markdown("---")
        st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d')}")
        st.caption("Use the page navigation above ↑ to switch pages")


def main():
    df = build_stock_df()
    render_sidebar(df)

    st.markdown('<div class="main-header">📊 Tadawul Opportunity Engine</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{len(df)} stocks · {len(df["sector"].unique())} sectors · {datetime.now().strftime("%Y-%m-%d")}</div>', unsafe_allow_html=True)

    # ── Market Environment ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌍 Current Market Environment")
    first   = df.iloc[0] if not df.empty else pd.Series(dtype=object)
    regime  = first.get("rate_regime", "stable")
    vix_val = first.get("vix", 20)
    oil_val = first.get("oil", 80)
    rate    = first.get("repo_rate", 4.25)
    vix_lbl = "🔴 High Fear" if vix_val > 30 else ("🟡 Elevated" if vix_val > 20 else "🟢 Calm")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Saudi Repo Rate",   f"{rate:.2f}%", delta=regime.capitalize(),
              delta_color="inverse" if regime == "rising" else "normal")
    c2.metric("VIX (Global Risk)", f"{vix_val:.1f}", delta=vix_lbl, delta_color="off")
    c3.metric("Brent Oil",         f"${oil_val:.0f}/bbl")
    c4.metric("Stocks Monitored",  len(df))

    env_msg = {
        "stable":  "✅ **Stable rate environment** — historically the best setup for Saudi equities.",
        "rising":  "⚠️ **Rising rate environment** — historically challenging. Be selective.",
        "falling": "🟡 **Falling rate environment** — mixed signals. Monitor carefully.",
    }.get(regime, "")
    if vix_val > 30:
        env_msg += " VIX is elevated — **fear creates opportunity** based on historical patterns."
    st.info(env_msg)

    # ── Sector Scorecard ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏭 Sector Scorecard")
    sec = df.groupby("sector").agg(Stocks=("symbol","count"),
        Avg_Score=("composite","mean"), Avg_90d=("forecast_90d","mean")).round(1).reset_index()
    best = df.loc[df.groupby("sector")["composite"].idxmax()][["sector","name","composite","forecast_90d"]]
    sec  = sec.merge(best.rename(columns={"name":"Top Pick","composite":"Top Score","forecast_90d":"Top 90d"}), on="sector", how="left")
    sec  = sec.sort_values("Avg_Score", ascending=False)
    sec["Avg_Score"] = sec["Avg_Score"].apply(lambda x: f"{x:.0f}/100")
    sec["Avg_90d"]   = sec["Avg_90d"].apply(lambda x: f"{x:+.1f}%")
    sec["Top Score"] = sec["Top Score"].apply(lambda x: f"{x:.0f}/100")
    sec["Top 90d"]   = sec["Top 90d"].apply(lambda x: f"{x:+.1f}%")
    sec.columns      = ["Sector","# Stocks","Avg Score","Avg 90d","Top Pick","Top Score","Top 90d"]
    st.dataframe(sec, use_container_width=True, hide_index=True)

    # ── Top Picks ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏆 Top Picks — Engine Highest Confidence")
    st.caption("Composite ≥ 68 · positive 90d forecast · ranked by score")
    top = df[(df["composite"] >= 68) & (df["forecast_90d"] > 0)].sort_values("composite", ascending=False).head(12)
    if not top.empty:
        rows = []
        for _, r in top.iterrows():
            rows.append({"": "🟢" if r["composite"] >= 72 else "🟡",
                "Stock": r["name"], "Sector": r["sector"],
                "Price (SAR)": f"{r['price']:.2f}", "RSI": f"{r['rsi']:.1f}",
                "Composite": f"{r['composite']:.0f}/100",
                "90d Forecast": f"{r['forecast_90d']:+.1f}%",
                "90d Target": f"{r['target_90d']:.2f} SAR",
                "Confidence": r["conf_label"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("➡️ Go to **Stock Analysis** page (sidebar) to dive into any stock")
    else:
        st.info("No stocks meet the top picks threshold right now.")

    # ── RSI Alerts ────────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📉 Extreme Oversold (RSI < 25)")
        st.caption("Historically a mean-reversion buying opportunity")
        oversold = df[df["rsi"] < 25].sort_values("rsi")
        if not oversold.empty:
            for _, r in oversold.iterrows():
                st.markdown(f"""<div class="forecast-box"><b>{r['emoji']} {r['name']}</b> — {r['sector']}<br>
                Price: <b>{r['price']:.2f} SAR</b> | RSI: <b style="color:#dc3545">{r['rsi']:.1f}</b> |
                Score: <b>{r['composite']:.0f}/100</b> | 90d: <b>{r['forecast_90d']:+.1f}%</b></div>""",
                unsafe_allow_html=True)
        else:
            st.success("No stocks in extreme oversold territory right now.")

    with c2:
        st.markdown("### 📈 Extreme Overbought (RSI > 78)")
        st.caption("Historically a mean-reversion caution signal")
        overbought = df[df["rsi"] > 78].sort_values("rsi", ascending=False)
        if not overbought.empty:
            for _, r in overbought.iterrows():
                st.markdown(f"""<div class="warning-box"><b>{r['emoji']} {r['name']}</b> — {r['sector']}<br>
                Price: <b>{r['price']:.2f} SAR</b> | RSI: <b style="color:#fd7e14">{r['rsi']:.1f}</b> |
                Score: <b>{r['composite']:.0f}/100</b> | 90d: <b>{r['forecast_90d']:+.1f}%</b></div>""",
                unsafe_allow_html=True)
        else:
            st.success("No stocks in extreme overbought territory right now.")

    # ── Weakest ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚠️ Weakest Signals — Low Confidence")
    weak = df[(df["composite"] < 55) | (df["forecast_90d"] < 0)].sort_values("composite").head(10)
    if not weak.empty:
        st.dataframe(pd.DataFrame([{
            "Stock": r["name"], "Sector": r["sector"],
            "Price": f"{r['price']:.2f}", "RSI": f"{r['rsi']:.1f}",
            "Composite": f"{r['composite']:.0f}/100",
            "90d Forecast": f"{r['forecast_90d']:+.1f}%",
            "Confidence": r["conf_label"],
        } for _, r in weak.iterrows()]), use_container_width=True, hide_index=True)

    # ── Full Table ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 All Stocks — Full Ranking")
    sf   = st.selectbox("Filter sector", ["All"] + sorted(df["sector"].unique()))
    disp = (df if sf == "All" else df[df["sector"] == sf]).sort_values("composite", ascending=False)
    st.dataframe(pd.DataFrame([{
        "": "🟢" if r["composite"] >= 68 else ("🟡" if r["composite"] >= 55 else "🔴"),
        "Stock": r["name"], "Sector": r["sector"],
        "Price": f"{r['price']:.2f}", "RSI": f"{r['rsi']:.1f}",
        "Env": f"{r['env_score']:.0f}", "Tech": f"{r['tech_score']:.0f}",
        "Fund": f"{r['fund_score']:.0f}", "Composite": f"{r['composite']:.0f}/100",
        "90d Fcst": f"{r['forecast_90d']:+.1f}%", "Confidence": r["conf_label"],
    } for _, r in disp.iterrows()]), use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(disp)} stocks | 🟢 Strong  🟡 Neutral  🔴 Weak")


if __name__ == "__main__":
    main()
