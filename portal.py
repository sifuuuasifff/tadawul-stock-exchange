"""
TADAWUL STOCK MEMORY ENGINE — HOME (Summary Dashboard)
Page 1 of 2.  Run:  streamlit run portal.py
"""

import sys, os, json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import anthropic

from shared import STOCKS, load_all_states, build_stock_df, CSS, MEMORY_DIR, REPORTS_DIR, ANTHROPIC_API_KEY

st.set_page_config(
    page_title="Tadawul Stock Memory Engine",
    page_icon="📊", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)


# ── Summary AI ────────────────────────────────────────────────────────────────

def get_summary_api_key() -> str:
    try:
        if hasattr(st, "secrets") and "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY", "") or ANTHROPIC_API_KEY


def build_summary_context(df: pd.DataFrame) -> str:
    states = load_all_states()
    # Top 10 by composite
    top10 = df.sort_values("composite", ascending=False).head(10)
    # Bottom 5
    bot5  = df.sort_values("composite").head(5)
    # Extreme RSI
    oversold   = df[df["rsi"] < 25][["name","rsi","composite","forecast_90d","sector"]].to_dict("records")
    overbought = df[df["rsi"] > 78][["name","rsi","composite","forecast_90d","sector"]].to_dict("records")

    # Load quick hypothesis summaries for top stocks
    first = df.iloc[0] if not df.empty else {}

    return f"""
You are the AI Research Engine for a private Tadawul Stock Memory Engine.
Today: {datetime.now().strftime('%Y-%m-%d')}
You have full engine data (backtests, hypotheses, signals) for {len(STOCKS)} stocks across 10 sectors.

MARKET ENVIRONMENT:
Rate regime: {first.get('rate_regime','?').upper()} @ {first.get('repo_rate','?')}%
VIX: {first.get('vix','?')}  Oil: ${first.get('oil','?')}/bbl

TOP 10 STOCKS BY ENGINE SCORE:
{df.sort_values('composite',ascending=False).head(10)[['name','sector','composite','rsi','forecast_90d','conf_label']].to_string(index=False)}

BOTTOM 5 (weakest signals):
{df.sort_values('composite').head(5)[['name','sector','composite','forecast_90d']].to_string(index=False)}

EXTREME RSI OVERSOLD (<25) — historically strong buy signal:
{json.dumps(oversold, default=str)}

EXTREME RSI OVERBOUGHT (>78) — mean reversion caution:
{json.dumps(overbought, default=str)}

SECTOR AVERAGES:
{df.groupby('sector')[['composite','forecast_90d']].mean().round(1).to_string()}

ALL STOCKS SNAPSHOT:
{df[['name','sector','composite','rsi','forecast_90d','conf_label','price']].sort_values('composite',ascending=False).to_string(index=False)}

INSTRUCTIONS:
- Answer in English always unless user asks for Arabic.
- Plain language — user is non-technical.
- You can answer about individual stocks, sectors, comparisons, or the overall market.
- Always give specific numbers from the data above.
- No financial advice. Research and evidence only.
- Be direct and concise.
"""


def summary_ai_chat(df: pd.DataFrame):
    st.markdown("### 💬 Ask the Engine")
    st.markdown('<div class="sub-header">Ask about any stock, sector, or the overall market — in plain English</div>', unsafe_allow_html=True)

    if "summary_chat" not in st.session_state:
        st.session_state.summary_chat = []

    # Quick question chips
    quick = [
        "Which stocks have the highest confidence right now?",
        "Which sector is strongest today?",
        "Any extreme oversold stocks I should know about?",
        "Compare banking vs petrochemicals",
        "Which stocks does the engine most recommend avoiding?",
        "What is the overall market signal today?",
    ]
    cols = st.columns(3)
    for i, q in enumerate(quick):
        if cols[i % 3].button(q, key=f"sq_{i}", use_container_width=True):
            st.session_state.summary_pending = q

    # Chat history
    for msg in st.session_state.summary_chat[-6:]:
        css  = "chat-user" if msg["role"] == "user" else "chat-ai"
        icon = "🧑" if msg["role"] == "user" else "🤖"
        st.markdown(f'<div class="{css}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)

    with st.form("summary_chat_form", clear_on_submit=True):
        question = st.text_area(
            "Your question",
            placeholder="e.g. Which stocks are the engine's top picks right now and why?\n\nAsk about any stock, sector comparison, or market overview...",
            value=st.session_state.pop("summary_pending", ""),
            height=90,
        )
        submitted = st.form_submit_button("Ask →", use_container_width=True)

    if submitted and question.strip():
        key = get_summary_api_key()
        if not key:
            st.error("API key not set. Add ANTHROPIC_API_KEY to Streamlit Cloud secrets.")
        else:
            with st.spinner("Thinking..."):
                try:
                    client   = anthropic.Anthropic(api_key=key)
                    context  = build_summary_context(df)
                    messages = [{"role": m["role"], "content": m["content"]}
                                for m in st.session_state.summary_chat[-6:]]
                    messages.append({"role": "user", "content": question})
                    resp = client.messages.create(
                        model="claude-sonnet-4-6", max_tokens=1200,
                        system=context, messages=messages,
                    )
                    answer = resp.content[0].text
                    st.session_state.summary_chat.append({"role": "user",      "content": question})
                    st.session_state.summary_chat.append({"role": "assistant",  "content": answer})
                    st.rerun()
                except Exception as e:
                    st.error(f"AI error: {e}")

    if st.session_state.summary_chat:
        if st.button("Clear", key="clear_summary"):
            st.session_state.summary_chat = []
            st.rerun()

    st.markdown("---")


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

        # ── Live refresh button ───────────────────────────────────────────
        last_refresh = "Never"
        lr_path = Path("memory/last_refresh.json")
        if lr_path.exists():
            with open(lr_path) as f:
                lr = json.load(f)
            ts = lr.get("timestamp","")
            if ts:
                last_refresh = ts[:16].replace("T"," ")

        st.caption(f"📅 Last refresh: {last_refresh}")

        if st.button("🔄 Refresh Live Prices", use_container_width=True,
                     help="Pulls latest prices from Yahoo Finance and recalculates all scores"):
            with st.spinner("Pulling live prices for 74 stocks..."):
                import subprocess, sys
                result = subprocess.run(
                    [sys.executable, "daily_refresh.py"],
                    capture_output=True, text=True, timeout=300
                )
            if result.returncode == 0:
                st.success("✅ Prices updated! Reloading...")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(f"Refresh failed: {result.stderr[-200:]}")

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

    # ── AI Chat ───────────────────────────────────────────────────────────────
    st.markdown("---")
    summary_ai_chat(df)

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
