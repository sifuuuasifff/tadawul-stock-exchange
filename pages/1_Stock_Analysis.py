"""
PAGE 2 — STOCK ANALYSIS & AI CHAT
Individual stock deep-dive: AI chat, forecast, personality, signals, hypotheses, backtest.
"""

import sys, json, os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import anthropic

from shared import (STOCKS, load_all_states, load_stock_engine,
                    CSS, MEMORY_DIR, REPORTS_DIR, ANTHROPIC_API_KEY)

st.set_page_config(
    page_title="Stock Analysis — Tadawul Engine",
    page_icon="🔍", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)


# ── AI ─────────────────────────────────────────────────────────────────────────

def load_quarterly_financials(sym: str) -> dict:
    """
    Load quarterly records from all_current_states.json (committed to GitHub).
    Falls back to raw files if running locally.
    """
    # Primary: read from all_current_states.json (always available on cloud)
    states = load_all_states()
    s = states.get(sym, {})
    if s.get("quarterly_records"):
        return {
            "quarterly_records": s["quarterly_records"],
            "ttm_summary":       s.get("ttm_summary", {}),
        }
    # Fallback: try raw files (local only)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from rebuild_engine import load_enriched_financials
        income_df, _, _, _, annual_df = load_enriched_financials(sym)
        records = []
        if not income_df.empty:
            for _, r in income_df.sort_values("report_date").tail(12).iterrows():
                ni  = r.get("net_income")
                rev = r.get("total_revenue")
                records.append({
                    "date":    r["report_date"].date().isoformat(),
                    "quarter": f"Q{int(r.get('fiscal_quarter',0))}" if r.get("fiscal_quarter") else "Annual",
                    "type":    "Full Year" if r["report_date"].month == 12 else "Quarterly",
                    "net_income_mn": round(ni/1e6, 1)  if ni  and not pd.isna(ni)  else None,
                    "revenue_mn":    round(rev/1e6, 1) if rev and not pd.isna(rev) else None,
                })
        ttm = {}
        if not annual_df.empty:
            lat = annual_df.iloc[-1]
            ttm = {
                "ttm_ni_mn":   round(lat.get("net_income_bn", 0) * 1000, 1),
                "ttm_yoy_pct": round(lat.get("ni_yoy", 0), 1),
                "q1_yoy_pct":  round(lat.get("_q_yoy", 0), 1) if lat.get("_q_yoy") else None,
                "is_ttm":      bool(lat.get("_is_ttm", False)),
                "as_of":       lat["report_date"].date().isoformat(),
            }
        return {"quarterly_records": records, "ttm_summary": ttm}
    except Exception as e:
        return {"quarterly_records": [], "ttm_summary": {}, "note": str(e)}


def build_context(sym: str) -> str:
    from config.settings import DATA_RAW, REPORTS_DIR as RPT
    states   = load_all_states()
    engine   = load_stock_engine(sym)
    info     = STOCKS.get(sym, {})
    sector   = info.get("sector", "")
    state    = states.get(sym, {})
    hyp      = engine.get("hypotheses", {})
    bt       = engine.get("backtest", {})
    mem      = engine.get("memory", {})
    pers     = engine.get("personality", engine.get("personality_summary", {}))
    mistakes = engine.get("mistakes", [])
    # Read directly from already-loaded state (avoids double cache issue)
    fin_data = {
        "quarterly_records": state.get("quarterly_records", []),
        "ttm_summary":       state.get("ttm_summary", {}),
        "div_summary":       state.get("div_summary", {}),
    }
    # Fallback to file loader only if state has no quarterly data
    if not fin_data["quarterly_records"]:
        fin_data = load_quarterly_financials(sym)

    # ── All 74 stocks snapshot ────────────────────────────────────────────
    snap = {s: {
        "name":        STOCKS.get(s, {}).get("name", s),
        "sector":      STOCKS.get(s, {}).get("sector", "?"),
        "price":       states[s].get("price"),
        "composite":   states[s].get("composite"),
        "rsi":         states[s].get("rsi"),
        "forecast_90d":states[s].get("forecast", {}).get("base_90d_pct"),
        "conf":        states[s].get("forecast", {}).get("confidence_label"),
    } for s in states}

    # ── Sector ranking ─────────────────────────────────────────────────────
    sector_peers = {s: v for s, v in snap.items()
                    if STOCKS.get(s, {}).get("sector") == sector and v.get("composite")}
    sector_ranked = sorted(sector_peers.items(), key=lambda x: -x[1].get("composite", 0))
    my_rank = next((i+1 for i, (s,_) in enumerate(sector_ranked) if s == sym), None)
    sector_avg_comp = round(sum(v.get("composite",0) for v in sector_peers.values()) / len(sector_peers), 1) if sector_peers else None

    # ── Recent events with sentiment ───────────────────────────────────────
    ev_path = DATA_RAW / ("alrajhi_events_full.json" if sym=="1120" else f"stock_{sym}/events.json")
    recent_events = []
    if ev_path.exists():
        with open(ev_path, encoding="utf-8") as f:
            ev_data = json.load(f)
        events = ev_data.get("events", [])
        for ev in events[:20]:
            if ev.get("event_type") in ("FINANCIAL_REPORT","EARNINGS_SURPRISE","DIVIDEND_ANNOUNCEMENT","REGULATORY_ACTION","ANALYST_RATING_CHANGE"):
                recent_events.append({
                    "date":      ev.get("event_date","")[:10],
                    "type":      ev.get("event_type"),
                    "sentiment": ev.get("sentiment"),
                    "importance":ev.get("importance"),
                    "summary":   ev.get("description","")[:200],
                })

    # ── Regime memory for THIS stock ──────────────────────────────────────
    regime_memory = []
    if mem.get("regime_profiles"):
        for r in mem["regime_profiles"]:
            out90 = r.get("outcomes", {}).get("90d", {})
            if out90.get("n", 0) >= 10:
                regime_memory.append({
                    "regime":    f"{r.get('type')} = {r.get('label')}",
                    "n_days":    r.get("n_days"),
                    "avg_90d":   out90.get("avg_pct"),
                    "win_rate":  out90.get("pct_positive"),
                })

    # ── Recent predictions + outcomes ─────────────────────────────────────
    recent_preds = []
    preds = bt.get("predictions", [])
    for p in preds[-6:]:
        if p.get("actual_90d_pct") is not None:
            recent_preds.append({
                "date":      p.get("prediction_date"),
                "predicted": p.get("base_90d_pct"),
                "actual":    p.get("actual_90d_pct"),
                "correct":   p.get("hit_target_90d"),
                "composite": p.get("composite"),
            })

    # ── Current valuation plain language ──────────────────────────────────
    pe  = state.get("pe_ratio")
    pb  = state.get("price_to_book")
    an_target = state.get("analyst_target_mean")
    an_upside = state.get("analyst_upside_pct")
    n_an = state.get("num_analysts", 0)
    fv   = state.get("fair_price")
    fv_upside = state.get("sahmk_fair_upside_pct")
    price = state.get("price", 0)
    h52  = state.get("dist_from_52w_high_pct")
    valuation_summary = {
        "current_price":        price,
        "pe_ratio":             pe,
        "price_to_book":        pb,
        "analyst_target_mean":  an_target,
        "analyst_upside_pct":   an_upside,
        "num_analysts":         n_an,
        "sahmk_fair_price":     fv,
        "sahmk_fair_upside_pct":fv_upside,
        "dist_from_52w_high_pct": h52,
        "vs_sector_composite":  f"Rank {my_rank} of {len(sector_peers)} in {sector}" if my_rank else None,
        "sector_avg_composite": sector_avg_comp,
    }

    # ── Signal report top signals ─────────────────────────────────────────
    sig_report = engine.get("signals", {})
    top_signals = sorted(
        [s for s in sig_report.get("signals", []) if "avg_return_90d" in s],
        key=lambda x: -x.get("reliability_score", 0)
    )[:8]

    return f"""
You are the AI Research Engine for a private Tadawul Stock Memory Engine.
Today: {datetime.now().strftime('%Y-%m-%d')}
All {len(STOCKS)} stocks are in the engine with full backtests and hypotheses.

SELECTED STOCK: {info.get('name','?')} ({sym}) — {sector}

════════════════════════════════════
CURRENT ENGINE STATE
════════════════════════════════════
Composite score:  {state.get('composite','?')}/100  (30d={state.get('composite_30d','?')} | 60d={state.get('composite_60d','?')} | 90d={state.get('composite_90d','?')})
Environment:      {state.get('env_score','?')}/100
Technical:        {state.get('tech_score','?')}/100
Fundamental:      {state.get('fund_score','?')}/100
Valuation:        {state.get('val_score','?')}/100

Forecast:  30d={state.get('forecast',{}).get('base_30d_pct','?')}%  60d={state.get('forecast',{}).get('base_60d_pct','?')}%  90d={state.get('forecast',{}).get('base_90d_pct','?')}%
Confidence: {state.get('forecast',{}).get('confidence_label','?')}
Rate regime: {state.get('rate_regime','?').upper()} @ {state.get('repo_rate','?')}%
RSI: {state.get('rsi','?')}  |  Price: {price} SAR

Active signals:
  Environment: {json.dumps(state.get('env_signals',{}), default=str)}
  Technical:   {json.dumps(state.get('tech_signals',{}), default=str)}
  Fundamental: {json.dumps(state.get('fund_signals',{}), default=str)}

════════════════════════════════════
VALUATION & ANALYST DATA
════════════════════════════════════
{json.dumps(valuation_summary, indent=2, default=str)}

SECTOR RANKING: {info.get('name','?')} ranks {my_rank} out of {len(sector_peers)} stocks in {sector}
Top 3 in sector by composite:
{json.dumps([{"rank":i+1,"sym":s,"name":STOCKS.get(s,{}).get("name",s),"composite":v.get("composite"),"forecast_90d":v.get("forecast_90d")} for i,(s,v) in enumerate(sector_ranked[:3])], indent=2, default=str)}

════════════════════════════════════
ACTUAL QUARTERLY FINANCIAL DATA
════════════════════════════════════
IMPORTANT: Q4/December records = FULL YEAR totals (not standalone Q4). All Saudi stocks use Dec 31 fiscal year-end.
TTM = Full Year minus prior same quarter plus latest quarter. Not Q1+Q2+Q3+Q4 summed.

Last 12 records:
{json.dumps(fin_data.get('quarterly_records', []), indent=2, default=str)}

TTM Summary:
{json.dumps(fin_data.get('ttm_summary', {}), indent=2, default=str)}

════════════════════════════════════
DIVIDENDS
════════════════════════════════════
{json.dumps(fin_data.get('div_summary', {}), indent=2, default=str)}

════════════════════════════════════
REGIME MEMORY (what historically happened in each environment for THIS stock)
════════════════════════════════════
{json.dumps(regime_memory, indent=2, default=str)}

════════════════════════════════════
RECENT PREDICTIONS vs ACTUAL OUTCOMES (last 6)
════════════════════════════════════
{json.dumps(recent_preds, indent=2, default=str)}
Backtest overall: {bt.get('validation',{}).get('directional_accuracy_pct','?')}% accuracy | {bt.get('validation',{}).get('edge_over_baseline_pct','?'):+}% edge | {bt.get('validation',{}).get('n_predictions','?')} predictions

════════════════════════════════════
RECENT EVENTS & ANNOUNCEMENTS
════════════════════════════════════
{json.dumps(recent_events, indent=2, default=str)}

════════════════════════════════════
TOP SIGNALS FOR THIS STOCK
════════════════════════════════════
{json.dumps([dict(signal=s.get("signal","")[:60],group=s.get("group",""),n=s.get("occurrences",0),avg_90d=s.get("avg_return_90d"),win_pct=s.get("pct_positive_90d"),significant=s.get("significant_90d")) for s in top_signals], indent=2, default=str)}

════════════════════════════════════
PERSONALITY & HYPOTHESES
════════════════════════════════════
Personality rules:
{json.dumps(pers.get('personality_rules', []), indent=2, default=str)}

Hypotheses: {hyp.get('summary',{}).get('total','?')} tested | {hyp.get('summary',{}).get('accepted','?')} accepted | {hyp.get('summary',{}).get('rejected','?')} rejected
KEY ACCEPTED:
{json.dumps([r for r in hyp.get('results',[]) if r.get('verdict')=='ACCEPTED'], indent=2, default=str)}
KEY REJECTED (what does NOT drive this stock):
{json.dumps([dict(id=r.get("id"),hyp=r.get("hypothesis","")[:80],reason=r.get("reason","")[:80]) for r in hyp.get('results',[]) if r.get('verdict')=='REJECTED'][:5], indent=2, default=str)}

Top mistakes (root causes):
{json.dumps(mistakes[:5], indent=2, default=str)}

════════════════════════════════════
ALL 74 STOCKS SNAPSHOT
════════════════════════════════════
{json.dumps(snap, indent=2, default=str)}

════════════════════════════════════
INSTRUCTIONS
════════════════════════════════════
- ALWAYS respond in English. Switch to Arabic only if user explicitly asks.
- Use the ACTUAL DATA above — never say you don't have data if it appears here.
- Q4/December = Full Year. TTM is the correct trailing figure. State this when relevant.
- Give specific numbers. Be direct. No unnecessary disclaimers.
- Do NOT give financial advice — research and evidence only.
- Plain English — user is non-technical.
- When comparing stocks, use the all-stocks snapshot.
- When asked about results, reference the quarterly financial data.
- When asked about history, reference regime memory and recent predictions.
"""


def get_api_key() -> str:
    """Read key at call time — st.secrets is available after app starts."""
    # Try st.secrets first (Streamlit Cloud)
    try:
        if hasattr(st, "secrets"):
            # Try exact key
            if "ANTHROPIC_API_KEY" in st.secrets:
                return st.secrets["ANTHROPIC_API_KEY"]
            # Show available keys for debugging
            available = list(st.secrets.keys()) if st.secrets else []
            if available:
                st.warning(f"DEBUG: st.secrets has keys: {available} — but not ANTHROPIC_API_KEY")
    except Exception as ex:
        st.warning(f"DEBUG: st.secrets error: {ex}")
    # Try environment variable
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if env_key:
        return env_key
    return ""


def ask_claude(question: str, context: str, history: list) -> str:
    try:
        key = get_api_key()
        if not key:
            return "API key not set. Add ANTHROPIC_API_KEY to Streamlit Cloud secrets."
        client   = anthropic.Anthropic(api_key=key)
        messages = [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
        messages.append({"role": "user", "content": question})
        resp = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1500,
            system=context, messages=messages,
        )
        return resp.content[0].text
    except Exception as e:
        return f"AI error: {e}"


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("### 🔍 Stock Analysis")
        st.markdown("---")
        sectors  = ["All"] + sorted(set(v["sector"] for v in STOCKS.values()))
        sel_sec  = st.selectbox("Filter sector", sectors, index=0)
        filtered = STOCKS if sel_sec == "All" else {k:v for k,v in STOCKS.items() if v["sector"]==sel_sec}
        sym = st.selectbox("Select stock", list(filtered.keys()),
                           format_func=lambda s: f"{STOCKS[s]['emoji']} {STOCKS[s]['name']} ({s})")
        st.markdown("---")
        state = load_all_states().get(sym, {})
        if state:
            price = state.get("price", 0)
            comp  = state.get("composite", 0)
            rsi   = state.get("rsi", 50)
            fc    = state.get("forecast", {})
            color = "green" if comp > 62 else ("orange" if comp > 50 else "red")
            c1,c2 = st.columns(2)
            c1.metric("Price", f"{price:.2f} SAR")
            c2.metric("RSI",   f"{rsi:.1f}")
            st.markdown(f"Composite: **:{color}[{comp:.0f}/100]**")
            b90  = fc.get("base_90d_pct", 0)
            conf = fc.get("confidence_label", "?")
            arrow = "🟢" if b90 > 3 else ("🟡" if b90 > 0 else "🔴")
            st.markdown(f"90d: {arrow} **{b90:+.1f}%** ({conf})")
            st.caption(f"Regime: {state.get('rate_regime','?').upper()} @ {state.get('repo_rate','?')}%")
        st.markdown("---")
        st.caption("Use sidebar navigation ↑ to go back to Summary")
    return sym


# ── Tab functions ──────────────────────────────────────────────────────────────

def tab_chat(sym: str):
    info = STOCKS.get(sym, {})
    st.markdown(f'<div class="main-header">💬 AI Chat — {info.get("emoji","")} {info.get("name",sym)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask anything in plain English or Arabic</div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    context = build_context(sym)
    name    = info.get("name", sym)

    quick = [
        f"What is the current forecast for {name}?",
        f"What signals matter most for {name}?",
        f"What is {name}'s personality?",
        f"How does the rate regime affect {name}?",
        f"What are the biggest prediction mistakes for {name}?",
        "Which stock has the best opportunity right now?",
    ]
    st.markdown("**Quick questions:**")
    cols = st.columns(3)
    for i, q in enumerate(quick):
        if cols[i % 3].button(q, key=f"q_{i}"):
            st.session_state.pending_q = q

    st.markdown("---")
    for msg in st.session_state.chat_history:
        css  = "chat-user" if msg["role"] == "user" else "chat-ai"
        icon = "🧑" if msg["role"] == "user" else "🤖"
        st.markdown(f'<div class="{css}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        question = st.text_area("Your question",
            placeholder="e.g. What does the engine say about this stock right now?\n\nYou can ask anything — forecast, signals, history, comparisons...",
            value=st.session_state.pop("pending_q", ""),
            height=100)
        submitted = st.form_submit_button("Ask →", use_container_width=True)

    if submitted and question.strip():
        with st.spinner("Thinking..."):
            answer = ask_claude(question, context, st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "user",     "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()


def tab_forecast(sym: str):
    info   = STOCKS.get(sym, {})
    state  = load_all_states().get(sym, {})
    engine = load_stock_engine(sym)
    st.markdown(f'<div class="main-header">🎯 Forecast — {info.get("emoji","")} {info.get("name",sym)}</div>', unsafe_allow_html=True)
    if not state:
        st.warning("No forecast data found.")
        return

    fc      = state.get("forecast", {})
    fc30    = state.get("forecast_30d", {})
    fc60    = state.get("forecast_60d", {})
    fc90    = state.get("forecast_90d", {})
    price   = state.get("price", 0)
    comp    = state.get("composite", 0)
    conf    = fc.get("confidence", 50)
    conf_label = fc.get("confidence_label", "MEDIUM")
    rsi     = state.get("rsi", 50)
    regime  = state.get("rate_regime", "stable")
    mem     = engine.get("memory", {})

    # ── Confidence Banner ──────────────────────────────────────────────────
    conf_color = "#28a745" if conf_label=="HIGH" else ("#ffc107" if conf_label=="MEDIUM" else "#dc3545")
    conf_desc  = {
        "HIGH":   "Multiple signals agree. Historical setups like this have been reliable.",
        "MEDIUM": "Some signals agree, others are mixed. Treat with caution.",
        "LOW":    "Signals are conflicting or the setup has limited historical backing.",
    }.get(conf_label, "")
    st.markdown(f"""
    <div style="background:{conf_color}22;border-left:5px solid {conf_color};
                padding:1rem;border-radius:8px;margin-bottom:1rem;">
        <span style="font-size:1.3rem;font-weight:700;color:{conf_color}">
            {conf_label} CONFIDENCE — {conf}/100</span><br>
        <span style="color:#444;font-size:0.9rem">{conf_desc}</span>
    </div>""", unsafe_allow_html=True)

    # ── Price Targets ──────────────────────────────────────────────────────
    st.markdown("### 📍 Price Targets")
    c1, c2, c3 = st.columns(3)
    for col, label, pct, tgt in [
        (c1, "30 Days",  fc.get("base_30d_pct", fc30.get("base_pct", 0)),
                          fc.get("target_30d",   fc30.get("target", 0))),
        (c2, "60 Days",  fc.get("base_60d_pct", fc60.get("base_pct", 0)),
                          fc.get("target_60d",   fc60.get("target", 0))),
        (c3, "90 Days",  fc.get("base_90d_pct", fc90.get("base_pct", 0)),
                          fc.get("target_90d",   fc90.get("target", 0))),
    ]:
        arrow = "🟢" if pct > 2 else ("🔴" if pct < -2 else "🟡")
        col.markdown(f"""<div class="forecast-box" style="text-align:center">
        <b>{label}</b><br>
        <span style="font-size:1.5rem;font-weight:700">{tgt:.2f} SAR</span><br>
        <span style="font-size:1.1rem">{arrow} {pct:+.1f}%</span>
        </div>""", unsafe_allow_html=True)

    # ── Probability Distribution ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎲 Probability Assessment (90-day)")

    # Derive probability from regime memory + composite score
    regime_profiles = mem.get("regime_profiles", [])
    win_rate = 55  # default
    avg_return = fc.get("base_90d_pct", 5)
    for rp in regime_profiles:
        if rp.get("type") == "Rate Regime" and rp.get("label", "").lower() == regime.lower():
            win_rate = rp.get("outcomes", {}).get("90d", {}).get("pct_positive", 55) or 55
            break

    # Adjust win rate by composite score
    score_adj = (comp - 60) * 0.3
    adj_win   = min(85, max(30, round(win_rate + score_adj)))
    bull_prob  = adj_win
    bear_prob  = 100 - adj_win
    base_pct   = fc.get("base_90d_pct", 0)
    bull_pct   = fc.get("bull_90d_pct", fc90.get("bull_pct", base_pct + 5))
    bear_pct   = fc.get("bear_90d_pct", fc90.get("bear_pct", base_pct - 5))

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div style="background:#d4edda;border-radius:10px;padding:1rem;text-align:center">
    <b>🐂 Bull Case</b><br>
    <span style="font-size:1.4rem;font-weight:700;color:#28a745">{bull_prob}%</span><br>
    <span style="color:#444">{bull_pct:+.1f}% → {price*(1+bull_pct/100):.2f} SAR</span>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div style="background:#fff3cd;border-radius:10px;padding:1rem;text-align:center">
    <b>📊 Base Case</b><br>
    <span style="font-size:1.4rem;font-weight:700;color:#856404">Most likely</span><br>
    <span style="color:#444">{base_pct:+.1f}% → {price*(1+base_pct/100):.2f} SAR</span>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div style="background:#f8d7da;border-radius:10px;padding:1rem;text-align:center">
    <b>🐻 Bear Case</b><br>
    <span style="font-size:1.4rem;font-weight:700;color:#dc3545">{bear_prob}%</span><br>
    <span style="color:#444">{bear_pct:+.1f}% → {price*(1+bear_pct/100):.2f} SAR</span>
    </div>""", unsafe_allow_html=True)

    # ── Reasons to Believe ─────────────────────────────────────────────────
    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### ✅ Reasons to Believe")

        reasons_for = []
        env_sigs  = state.get("env_signals", {})
        tech_sigs = state.get("tech_signals", {})
        fund_sigs = state.get("fund_signals", {})
        val_sigs  = state.get("val_signals", {})

        # Rate regime
        if "stable" in env_sigs.get("rate", ""):
            reasons_for.append("✅ **Stable rate environment** — historically the best condition for Saudi stocks")
        # RSI
        rsi_sig = tech_sigs.get("rsi", "")
        if "oversold" in rsi_sig.lower():
            reasons_for.append(f"✅ **RSI oversold ({rsi:.0f})** — statistically produces above-average returns")
        if "lower band" in tech_sigs.get("bb", "").lower():
            reasons_for.append("✅ **Price at Bollinger Band lower bound** — historically a strong recovery signal")
        # Momentum
        mom = tech_sigs.get("mom", "")
        if "drop" in mom.lower():
            reasons_for.append(f"✅ **Recent price drop** — mean reversion setup with historically better forward returns")
        # Oil
        if "uptrend" in env_sigs.get("oil", ""):
            reasons_for.append("✅ **Oil in uptrend** — positive for Saudi economy and most Tadawul stocks")
        # Fundamentals
        ni_sig = fund_sigs.get("net_income", "") or fund_sigs.get("ni_growth", "")
        if "strong" in str(ni_sig).lower() or "exceptional" in str(ni_sig).lower():
            reasons_for.append(f"✅ **Strong earnings growth** — {ni_sig}")
        # Analyst
        an_sig = val_sigs.get("analyst", "")
        if "+" in str(an_sig) and "analysts" in str(an_sig):
            reasons_for.append(f"✅ **Analyst consensus positive** — {an_sig}")
        # Composite
        if comp > 68:
            reasons_for.append(f"✅ **High composite score ({comp:.0f}/100)** — signals broadly aligned")
        # VIX fear
        if "fear" in env_sigs.get("vix", ""):
            reasons_for.append("✅ **Market fear elevated** — historically creates buying opportunities")

        if reasons_for:
            for r in reasons_for:
                st.markdown(r)
        else:
            st.markdown("*No strong positive signals at current levels.*")

    with c2:
        st.markdown("### ⚠️ Reasons NOT to Believe")

        reasons_against = []

        # Rising rates
        if "rising" in env_sigs.get("rate", ""):
            reasons_against.append("🔴 **Rising rates** — historically the worst environment for most Saudi stocks")
        # Overbought
        if "overbought" in str(rsi_sig).lower():
            reasons_against.append(f"🔴 **RSI overbought ({rsi:.0f})** — mean reversion risk, historically weak forward returns")
        if "upper band" in tech_sigs.get("bb", "").lower():
            reasons_against.append("🔴 **Price at Bollinger Band upper bound** — historically poor forward returns")
        # Strong rally
        if "rally" in mom.lower() or "strong rally" in mom.lower():
            reasons_against.append("🔴 **Recent strong rally** — momentum historically reverses after large runs")
        # Above all MAs
        if "above all" in tech_sigs.get("ma", "").lower():
            reasons_against.append("🔴 **Price above all moving averages** — historically precedes weaker returns")
        # Analyst downside
        if "-" in str(an_sig) and "analysts" in str(an_sig):
            reasons_against.append(f"🔴 **Analysts cautious** — {an_sig}")
        # Weak fundamentals
        if "declining" in str(ni_sig).lower() or "weak" in str(ni_sig).lower():
            reasons_against.append(f"🔴 **Earnings weakness** — {ni_sig}")
        # Low confidence
        if conf_label == "LOW":
            reasons_against.append("🔴 **Low confidence** — signals are conflicting, limited historical backing")
        # SAHMK overvalued
        fv_sig = val_sigs.get("fair_value", "")
        if "overvalued" in str(fv_sig).lower() or "rich" in str(fv_sig).lower():
            reasons_against.append(f"🔴 **SAHMK fair value model** — {fv_sig}")
        # Short history
        days = state.get("trading_days", 0)
        if days < 1000:
            reasons_against.append(f"⚠️ **Limited history** — only {days} trading days. Engine has less data to learn from.")

        # Model uncertainty note
        reasons_against.append("⚠️ **General model limitation** — analyst consensus is current-only and not backtested")

        if reasons_against:
            for r in reasons_against:
                st.markdown(r)

    # ── Score Breakdown ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 What's Driving the Score")
    c1, c2, c3, c4 = st.columns(4)
    for col, label, score, note in [
        (c1, "🌍 Environment", state.get("env_score",0),
         f"Rate: {regime.capitalize()} | VIX: {state.get('vix',0):.0f} | Oil: {'↑' if state.get('oil_price',0)>80 else '→'}"),
        (c2, "📈 Technical",   state.get("tech_score",0),
         f"RSI: {rsi:.0f} | BB: {state.get('bb_pct',0.5):.2f} | 20d: {state.get('ret_20d_pct',0):+.1f}%"),
        (c3, "💰 Fundamental", state.get("fund_score",0),
         fund_sigs.get("net_income", fund_sigs.get("ni_growth", "See Financials tab"))),
        (c4, "💎 Valuation",   state.get("val_score",0),
         val_sigs.get("analyst", val_sigs.get("pe", "See Investor View tab"))),
    ]:
        bar_color = "#28a745" if score > 65 else ("#ffc107" if score > 50 else "#dc3545")
        col.markdown(f"""<div style="background:#f8f9fa;border-radius:8px;padding:0.8rem;
                         border-left:4px solid {bar_color}">
        <b>{label}</b><br>
        <span style="font-size:1.3rem;font-weight:700;color:{bar_color}">{score:.0f}/100</span><br>
        <span style="font-size:0.78rem;color:#666">{str(note)[:60]}</span>
        </div>""", unsafe_allow_html=True)

    # ── Similar Historical Setups ──────────────────────────────────────────
    if regime_profiles:
        st.markdown("---")
        st.markdown("### 📜 What History Says About This Setup")
        current_regime_data = next(
            (r for r in regime_profiles
             if r.get("type") == "Rate Regime" and r.get("label","").lower() == regime.lower()),
            None
        )
        if current_regime_data:
            out90 = current_regime_data.get("outcomes", {}).get("90d", {})
            n     = current_regime_data.get("n_days", 0)
            avg   = out90.get("avg_pct")
            win   = out90.get("pct_positive")
            if avg is not None:
                st.info(f"📊 In **{regime.upper()} rate environments**, this stock historically returned "
                        f"**{avg:+.1f}%** over 90 days with a **{win:.0f}%** win rate "
                        f"(based on {n:,} trading days of data)")


def tab_personality(sym: str):
    info   = STOCKS.get(sym, {})
    engine = load_stock_engine(sym)
    st.markdown(f'<div class="main-header">🧠 Personality — {info.get("emoji","")} {info.get("name",sym)}</div>', unsafe_allow_html=True)

    pers  = engine.get("personality", engine.get("personality_summary", {}))
    mem   = engine.get("memory", {})
    rules = pers.get("personality_rules", [])

    baseline = mem.get("baseline_90d", {})
    regimes  = [r for r in mem.get("regime_profiles", []) if r.get("type") == "Rate Regime"]
    if baseline:
        st.metric("Baseline 90d avg", f"{baseline.get('avg_pct','?')}%",
                  delta=f"Win rate: {baseline.get('win_pct','?')}%", delta_color="off")
    if regimes:
        cols = st.columns(len(regimes))
        for i, r in enumerate(regimes):
            out = r.get("outcomes",{}).get("90d",{})
            cols[i].metric(f"{r.get('label','?')} Rates",
                           f"{out.get('avg_pct','?')}% avg",
                           f"{out.get('pct_positive','?')}% win")
    st.markdown("---")
    st.markdown("**Evidence-Based Personality Rules:**")
    icons = {"HIGH":"🟢","MEDIUM":"🟡","LOW":"🔴"}
    for rule in rules:
        conf = rule.get("confidence","MEDIUM")
        st.markdown(f"**{icons.get(conf,'⚪')} {rule.get('rule','')}**")
        st.caption(f"Evidence: {rule.get('evidence','N/A')} | Confidence: {conf}")
        st.markdown("")


def tab_signals(sym: str):
    info   = STOCKS.get(sym, {})
    engine = load_stock_engine(sym)
    st.markdown(f'<div class="main-header">📡 Signals — {info.get("emoji","")} {info.get("name",sym)}</div>', unsafe_allow_html=True)

    report  = engine.get("signals", {})
    signals = report.get("signals", [])
    if not signals:
        st.warning("Signal report not found.")
        return

    base_90d = report.get("baseline", {}).get("avg_90d", 6.0) or 6.0
    c1,c2    = st.columns(2)
    group_f  = c1.selectbox("Group", ["All"] + sorted(set(s.get("group","") for s in signals)))
    rel_f    = c2.selectbox("Reliability", ["All","HIGH","MEDIUM","LOW"])

    rows = []
    for s in signals:
        if "avg_return_90d" not in s: continue
        if group_f != "All" and s.get("group") != group_f: continue
        if rel_f   != "All" and s.get("reliability_label") != rel_f: continue
        edge = (s.get("avg_return_90d") or 0) - base_90d
        rows.append({"Signal": s.get("signal","")[:55], "N": s.get("occurrences",0),
                     "Avg 30d %": s.get("avg_return_30d"), "Avg 90d %": s.get("avg_return_90d"),
                     "Win% (90d)": s.get("pct_positive_90d"), "Edge": f"{edge:+.1f}%",
                     "Reliability": s.get("reliability_label",""),
                     "p-value": s.get("p_value_90d"),
                     "Sig": "✓" if s.get("significant_90d") else ""})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"Baseline (buy-and-hold): avg 90d = {base_90d}%")


def tab_hypotheses(sym: str):
    info   = STOCKS.get(sym, {})
    engine = load_stock_engine(sym)
    st.markdown(f'<div class="main-header">🧪 Hypotheses — {info.get("emoji","")} {info.get("name",sym)}</div>', unsafe_allow_html=True)

    hyp     = engine.get("hypotheses", {})
    results = hyp.get("results", [])
    summary = hyp.get("summary", {})
    if not results:
        st.warning("Hypothesis report not found.")
        return

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total",        summary.get("total",0))
    c2.metric("Accepted ✓",   summary.get("accepted",0))
    c3.metric("Rejected ✗",   summary.get("rejected",0))
    c4.metric("Inconclusive", summary.get("inconclusive",0))

    vf = st.selectbox("Filter", ["All","ACCEPTED","REJECTED","INCONCLUSIVE"])
    rows = []
    for r in results:
        if vf != "All" and r.get("verdict") != vf: continue
        icon = {"ACCEPTED":"✓","REJECTED":"✗","INCONCLUSIVE":"~"}.get(r.get("verdict",""),"?")
        rows.append({"ID": r.get("id"),
                     "Verdict": f"{icon} {r.get('verdict','')}",
                     "Hypothesis": r.get("hypothesis","")[:65],
                     "Δ %": r.get("diff"), "p-value": r.get("p_value"),
                     "n": r.get("n_a"), "Reason": r.get("reason","")[:70]})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def tab_backtest(sym: str):
    info   = STOCKS.get(sym, {})
    engine = load_stock_engine(sym)
    st.markdown(f'<div class="main-header">📈 Backtest — {info.get("emoji","")} {info.get("name",sym)}</div>', unsafe_allow_html=True)

    bt = engine.get("backtest", {})
    v  = bt.get("validation", {})
    if not v:
        st.warning("Backtest report not found.")
        return

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Predictions", v.get("n_predictions",0))
    c2.metric("Accuracy",    f"{v.get('directional_accuracy_pct',0)}%",
              delta=f"{v.get('edge_over_baseline_pct',0):+.1f}% vs baseline", delta_color="normal")
    c3.metric("Baseline",    f"{v.get('baseline_always_long_pct',0)}%")
    c4.metric("MAE (90d)",   f"{v.get('mae_pct',0)}%")

    rd = v.get("regime_breakdown", {})
    if rd:
        st.markdown("**By Rate Regime:**")
        cols = st.columns(len(rd))
        for i,(regime,r) in enumerate(rd.items()):
            cols[i].metric(f"{regime.capitalize()} Rates",
                           f"{r.get('dir_accuracy',0)}% acc",
                           f"error {r.get('avg_error',0):+.1f}%")

    st.markdown("---")
    st.markdown("### 🚨 Mistake Vault")
    mistakes = engine.get("mistakes", [])
    if mistakes:
        rows = []
        for m in sorted(mistakes, key=lambda x: -x.get("error",0)):
            rows.append({"Date": m.get("prediction_date"),
                         "Predicted": f"{m.get('predicted_90d',0):+.1f}%",
                         "Actual": f"{m.get('actual_90d',0):+.1f}%",
                         "Error": f"{m.get('error',0):.1f}%",
                         "Dir OK": "✓" if m.get("direction_correct") else "✗",
                         "RSI": m.get("rsi"), "Regime": m.get("rate_regime"),
                         "Root Cause": m.get("root_cause","")[:65]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"{len(mistakes)} large errors catalogued")
    else:
        st.info("No large errors recorded for this stock.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    sym  = render_sidebar()
    info = STOCKS.get(sym, {})
    st.markdown(f'<div class="main-header">{info.get("emoji","")} {info.get("name",sym)} ({sym})</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{info.get("sector","?")} · Full engine analysis</div>', unsafe_allow_html=True)

    t1,t2,t3,t4,t5,t6 = st.tabs(["💬 AI Chat","🎯 Forecast","🧠 Personality","📡 Signals","🧪 Hypotheses","📈 Backtest"])
    with t1: tab_chat(sym)
    with t2: tab_forecast(sym)
    with t3: tab_personality(sym)
    with t4: tab_signals(sym)
    with t5: tab_hypotheses(sym)
    with t6: tab_backtest(sym)


if __name__ == "__main__":
    main()
