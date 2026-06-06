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



# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TECHNICALS
# ══════════════════════════════════════════════════════════════════════════════

def tab_technicals(sym: str):
    info  = STOCKS.get(sym, {})
    state = load_all_states().get(sym, {})
    st.markdown(f'<div class="main-header">📈 Technicals — {info.get("emoji","")} {info.get("name",sym)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Where is the stock right now? What is the price telling us?</div>', unsafe_allow_html=True)

    price  = state.get("price", 0)
    rsi    = state.get("rsi", 50)
    bb     = state.get("bb_pct", 0.5)
    ret20  = state.get("ret_20d_pct", 0)
    h52pct = state.get("dist_from_52w_high_pct", 0)
    tech   = state.get("tech_score", 50)

    if tech >= 70:
        verdict = ("🟢 Bullish Setup", "#28a745", "Technical signals are mostly positive for this stock right now.")
    elif tech >= 55:
        verdict = ("🟡 Neutral", "#ffc107", "Mixed signals — no strong technical direction either way.")
    elif tech >= 40:
        verdict = ("🟠 Mildly Bearish", "#fd7e14", "Some caution signals present.")
    else:
        verdict = ("🔴 Bearish Setup", "#dc3545", "Technical signals suggest caution at current levels.")

    st.markdown(f"""<div style="background:{verdict[1]}22;border-left:5px solid {verdict[1]};
    padding:0.8rem 1rem;border-radius:8px;margin-bottom:1rem">
    <b style="color:{verdict[1]};font-size:1.1rem">{verdict[0]}</b><br>
    <span style="color:#444">{verdict[2]}</span></div>""", unsafe_allow_html=True)

    st.markdown("### 📍 Price Position")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"{price:.2f} SAR")
    c2.metric("20-day Change",  f"{ret20:+.1f}%", delta_color="normal" if ret20 >= 0 else "inverse")
    c3.metric("From 52w High",  f"{h52pct:.1f}%")
    c4.metric("Technical Score", f"{tech:.0f}/100")

    st.markdown("---")
    st.markdown("### 🔮 RSI — Momentum Indicator")
    if rsi < 20:
        rsi_color, rsi_label, rsi_meaning = "#dc3545", f"EXTREME OVERSOLD ({rsi:.0f})", "Historically a very strong buying signal. Engine has seen this setup — average 90-day return in this zone is significantly above baseline."
    elif rsi < 30:
        rsi_color, rsi_label, rsi_meaning = "#fd7e14", f"OVERSOLD ({rsi:.0f})", "Sold down below normal range. This is the zone where mean reversions historically begin."
    elif rsi < 40:
        rsi_color, rsi_label, rsi_meaning = "#ffc107", f"LOW ({rsi:.0f})", "Approaching oversold territory. Not yet a strong signal but worth watching."
    elif rsi < 60:
        rsi_color, rsi_label, rsi_meaning = "#6c757d", f"NEUTRAL ({rsi:.0f})", "No strong signal. Stock is trading in a normal range."
    elif rsi < 70:
        rsi_color, rsi_label, rsi_meaning = "#ffc107", f"HIGH ({rsi:.0f})", "Approaching overbought. Some caution warranted."
    elif rsi < 80:
        rsi_color, rsi_label, rsi_meaning = "#fd7e14", f"OVERBOUGHT ({rsi:.0f})", "Stock has run up significantly. Historically this precedes weaker returns."
    else:
        rsi_color, rsi_label, rsi_meaning = "#dc3545", f"EXTREME OVERBOUGHT ({rsi:.0f})", "Extreme reading. Mean reversion risk is elevated."

    rsi_pct = min(100, max(0, rsi))
    st.markdown(f"""<div style="background:#f8f9fa;border-radius:10px;padding:1rem">
        <b style="color:{rsi_color}">{rsi_label}</b><br>
        <div style="background:#dee2e6;border-radius:5px;height:12px;margin:0.5rem 0">
            <div style="background:{rsi_color};width:{rsi_pct}%;height:12px;border-radius:5px"></div>
        </div>
        <span style="color:#444;font-size:0.9rem">{rsi_meaning}</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Price Range Position (Bollinger Bands)")
    if bb < 0.05:
        bb_color, bb_label, bb_msg = "#28a745", "Near Lower Band", "Price at bottom of normal range. Historically a recovery signal."
    elif bb < 0.25:
        bb_color, bb_label, bb_msg = "#ffc107", "Low in Range", "Lower portion of normal range — mild positive signal."
    elif bb < 0.75:
        bb_color, bb_label, bb_msg = "#6c757d", "Middle of Range", "Normal trading range. No directional signal."
    elif bb < 0.95:
        bb_color, bb_label, bb_msg = "#fd7e14", "High in Range", "Upper portion — mild caution."
    else:
        bb_color, bb_label, bb_msg = "#dc3545", "Near Upper Band", "Top of normal range. Historically precedes consolidation or pullback."

    bb_disp = min(100, max(0, int(bb * 100)))
    st.markdown(f"""<div style="background:#f8f9fa;border-radius:10px;padding:1rem">
        <b style="color:{bb_color}">{bb_label} ({bb_disp}%)</b><br>
        <div style="background:#dee2e6;border-radius:5px;height:12px;margin:0.5rem 0">
            <div style="background:{bb_color};width:{bb_disp}%;height:12px;border-radius:5px"></div>
        </div>
        <span style="color:#444;font-size:0.9rem">{bb_msg}</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏃 20-Day Momentum")
    if ret20 > 15:
        mom_color, mom_label, mom_msg = "#dc3545", f"Strong Rally (+{ret20:.1f}%)", "⚠️ After a strong rally, the engine historically sees WEAKER returns — mean reversion tends to follow."
    elif ret20 > 5:
        mom_color, mom_label, mom_msg = "#ffc107", f"Rising (+{ret20:.1f}%)", "Moderate upward momentum. Monitor closely."
    elif ret20 > -5:
        mom_color, mom_label, mom_msg = "#6c757d", f"Flat ({ret20:.1f}%)", "No strong directional momentum. Neutral signal."
    elif ret20 > -15:
        mom_color, mom_label, mom_msg = "#ffc107", f"Declining ({ret20:.1f}%)", "Stock has pulled back. Potential recovery setup depending on other signals."
    else:
        mom_color, mom_label, mom_msg = "#28a745", f"Sharp Drop ({ret20:.1f}%)", "✅ Large drops create mean reversion opportunity — needs rate regime + RSI to confirm."

    st.markdown(f"""<div style="background:#f8f9fa;border-radius:10px;padding:1rem">
        <b style="color:{mom_color}">{mom_label}</b><br>
        <span style="color:#444;font-size:0.9rem">{mom_msg}</span>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FINANCIALS
# ══════════════════════════════════════════════════════════════════════════════

def tab_financials(sym: str):
    info  = STOCKS.get(sym, {})
    state = load_all_states().get(sym, {})
    st.markdown(f'<div class="main-header">💰 Financials — {info.get("emoji","")} {info.get("name",sym)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">How is the business actually performing?</div>', unsafe_allow_html=True)

    qr   = state.get("quarterly_records", [])
    ttm  = state.get("ttm_summary", {})
    fund = state.get("fund_score", 50)

    if not qr and not ttm:
        st.warning("No financial data loaded for this stock.")
        return

    fund_color = "#28a745" if fund >= 65 else ("#ffc107" if fund >= 45 else "#dc3545")
    fund_label = "Strong" if fund >= 65 else ("Fair" if fund >= 45 else "Weak")
    st.markdown(f"""<div style="background:{fund_color}22;border-left:5px solid {fund_color};
    padding:0.8rem 1rem;border-radius:8px;margin-bottom:1rem">
    <b style="color:{fund_color}">Financial Health: {fund_label} ({fund:.0f}/100)</b><br>
    <span style="color:#444">Based on earnings growth, cash flow quality, and balance sheet strength</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 📊 Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    ttm_ni  = ttm.get("ttm_ni_mn", 0) or 0
    ttm_yoy = ttm.get("ttm_yoy_pct")
    q1_yoy  = ttm.get("q1_yoy_pct") or ttm.get("q1_standalone_yoy")
    cfq     = ttm.get("cash_flow_quality")
    dte     = ttm.get("debt_to_equity", 0) or 0
    roe     = ttm.get("roe_pct")
    roa     = ttm.get("roa_pct")
    assets  = ttm.get("total_assets_mn", 0) or 0
    equity  = ttm.get("equity_mn", 0) or 0

    c1.metric("TTM Net Income", f"{ttm_ni:,.0f}M SAR", delta=f"{ttm_yoy:+.1f}% YoY" if ttm_yoy else None)
    c2.metric("Latest Q Growth", f"{q1_yoy:+.1f}%" if q1_yoy else "N/A", help="Most recent quarter vs same quarter last year")
    c3.metric("Cash Flow Quality", f"{cfq:.2f}" if cfq else "N/A", help="OCF/NI. >1 = earnings backed by cash. <0.5 = low quality")
    c4.metric("Debt / Equity", f"{dte:.2f}x" if dte else "Zero debt", delta="Clean" if dte == 0 else ("OK" if dte < 1 else "High"), delta_color="off")

    if roe or roa:
        c1, c2, c3, c4 = st.columns(4)
        if roe: c1.metric("ROE", f"{roe:.1f}%", help="Return on Equity")
        if roa: c2.metric("ROA", f"{roa:.1f}%", help="Return on Assets")
        if equity: c3.metric("Equity", f"{equity/1000:.1f}B SAR")
        if assets: c4.metric("Total Assets", f"{assets/1000:.1f}B SAR")

    st.markdown("---")
    st.markdown("### 📝 Plain English Summary")
    summaries = []
    if q1_yoy is not None:
        if q1_yoy > 20:    summaries.append(f"✅ **Earnings accelerating** — latest quarter grew **{q1_yoy:.1f}%** vs same quarter last year. Strong momentum.")
        elif q1_yoy > 5:   summaries.append(f"✅ **Earnings growing** — latest quarter up **{q1_yoy:.1f}%** vs same quarter last year.")
        elif q1_yoy > -5:  summaries.append(f"⚠️ **Earnings flat** — latest quarter grew only **{q1_yoy:.1f}%** vs last year. Watch for trend.")
        else:              summaries.append(f"🔴 **Earnings declining** — latest quarter down **{q1_yoy:.1f}%** vs last year.")
    if cfq is not None:
        if cfq > 1.0:      summaries.append(f"✅ **High quality earnings** (CFQ {cfq:.2f}) — profits backed by real cash.")
        elif cfq > 0.5:    summaries.append(f"✅ **Decent cash backing** (CFQ {cfq:.2f}) — earnings reasonably supported.")
        elif cfq >= 0:     summaries.append(f"⚠️ **Low cash backing** (CFQ {cfq:.2f}) — profits may not fully convert to cash.")
        else:              summaries.append("🔴 **Negative operating cash flow** — consuming cash despite reported profits.")
    if dte == 0:           summaries.append("✅ **Zero debt** — very clean balance sheet.")
    elif dte < 0.5:        summaries.append(f"✅ **Low leverage** — D/E {dte:.2f}. Conservative financing.")
    elif dte > 2:          summaries.append(f"⚠️ **High leverage** — D/E {dte:.2f}. Monitor in rising rate environments.")
    for s in summaries:
        st.markdown(s)

    if qr:
        st.markdown("---")
        st.markdown("### 📅 Recent Financial Records")
        st.caption("Q4/December = Full Year total (not just Q4). All other quarters are standalone.")
        rows = []
        for r in reversed(qr):
            rows.append({
                "Period":     f"{r.get('date','?')} ({r.get('quarter','?')})",
                "Type":       r.get("type",""),
                "Net Income": f"{r.get('net_income_mn','?')} M SAR" if r.get("net_income_mn") else "N/A",
                "Revenue":    f"{r.get('revenue_mn','?')} M SAR"    if r.get("revenue_mn")    else "N/A",
                "Assets":     f"{r.get('total_assets_mn','?')} M"   if r.get("total_assets_mn") else "—",
                "OCF":        f"{r.get('ocf_mn','?')} M"            if r.get("ocf_mn")          else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — INVESTOR VIEW
# ══════════════════════════════════════════════════════════════════════════════

def tab_investor_view(sym: str):
    info   = STOCKS.get(sym, {})
    state  = load_all_states().get(sym, {})
    st.markdown(f'<div class="main-header">👁️ Investor View — {info.get("emoji","")} {info.get("name",sym)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">What do professional investors and analysts think?</div>', unsafe_allow_html=True)

    price      = state.get("price", 0)
    an_target  = state.get("analyst_target_mean")
    an_upside  = state.get("analyst_upside_pct")
    n_analysts = state.get("num_analysts", 0) or 0
    fv         = state.get("fair_price")
    fv_upside  = state.get("sahmk_fair_upside_pct")
    pe         = state.get("pe_ratio")
    pb         = state.get("price_to_book")

    st.markdown("### 👥 Professional Analyst View")
    if n_analysts > 0 and an_target:
        upside_color = "#28a745" if (an_upside or 0) > 5 else ("#ffc107" if (an_upside or 0) > -5 else "#dc3545")
        verdict = "BUY" if (an_upside or 0) > 10 else ("HOLD" if (an_upside or 0) > -5 else "CAUTION")
        st.markdown(f"""<div style="background:{upside_color}22;border-left:5px solid {upside_color};
        padding:1rem;border-radius:8px;margin-bottom:1rem">
        <b style="color:{upside_color};font-size:1.1rem">{verdict} — {n_analysts} analysts covering this stock</b><br>
        Average target: <b>{an_target:.2f} SAR</b> vs current {price:.2f} SAR
        → <b style="color:{upside_color}">{an_upside:+.1f}% upside</b>
        </div>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Mean Target", f"{an_target:.2f} SAR", delta=f"{an_upside:+.1f}%" if an_upside else None)
        c2.metric("Analysts Covering", n_analysts)
    else:
        st.info(f"No analyst coverage data available ({n_analysts} analysts found).")

    if fv:
        st.markdown("---")
        st.markdown("### 🎯 SAHMK Model Fair Value")
        fv_color  = "#28a745" if (fv_upside or 0) > 5 else ("#dc3545" if (fv_upside or 0) < -5 else "#ffc107")
        fv_verdict = "UNDERVALUED" if (fv_upside or 0) > 5 else ("OVERVALUED" if (fv_upside or 0) < -5 else "FAIRLY VALUED")
        st.markdown(f"""<div style="background:{fv_color}22;border-left:5px solid {fv_color};
        padding:0.8rem 1rem;border-radius:8px">
        <b style="color:{fv_color}">{fv_verdict}</b> — SAHMK model fair price: <b>{fv:.2f} SAR</b>
        vs current {price:.2f} SAR → <b style="color:{fv_color}">{fv_upside:+.1f}%</b><br>
        <span style="font-size:0.85rem;color:#666">SAHMK's internal model estimate. Not independently backtested.</span>
        </div>""", unsafe_allow_html=True)

    if pe or pb:
        st.markdown("---")
        st.markdown("### 💹 Valuation Ratios")
        c1, c2 = st.columns(2)
        if pe and 0 < pe < 500:
            pe_note = "Cheap" if pe < 8 else ("Fair" if pe < 15 else "Expensive")
            c1.metric("P/E Ratio", f"{pe:.1f}x", delta=pe_note, delta_color="normal" if pe_note=="Cheap" else "off",
                      help="Price-to-Earnings. Lower = cheaper. Saudi banks typically 6-12x.")
        if pb and 0 < pb < 100:
            pb_note = "Below Book" if pb < 1 else ("Fair" if pb < 2.5 else "Premium")
            c2.metric("Price/Book", f"{pb:.2f}x", delta=pb_note, delta_color="normal" if pb_note=="Below Book" else "off")

    div_summary = state.get("div_summary", {})
    recent_divs = div_summary.get("recent_dividends", [])
    yield_pct   = div_summary.get("trailing_12m_yield_pct")
    if recent_divs or yield_pct:
        st.markdown("---")
        st.markdown("### 💵 Dividends")
        if yield_pct:
            st.metric("Trailing 12M Yield", f"{yield_pct:.2f}%")
        if recent_divs:
            rows = []
            for d in recent_divs:
                rows.append({"Announced": d.get("date","?"), "Amount (SAR)": d.get("value_sar","?"),
                             "Period": d.get("period","?"), "Distribution": d.get("dist_date","?")})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    ev_path = Path(__file__).parent.parent / "data" / "raw" / (
        "alrajhi_events_full.json" if sym == "1120" else f"stock_{sym}/events.json"
    )
    if ev_path.exists():
        st.markdown("---")
        st.markdown("### 📢 Recent Announcements")
        with open(ev_path, encoding="utf-8") as f:
            ev_data = json.load(f)
        events = [e for e in ev_data.get("events", [])
                  if e.get("event_type") in ("FINANCIAL_REPORT","EARNINGS_SURPRISE",
                                              "DIVIDEND_ANNOUNCEMENT","REGULATORY_ACTION",
                                              "ANALYST_RATING_CHANGE")][:8]
        if events:
            rows = []
            for e in events:
                sent = e.get("sentiment","neutral")
                icon = {"positive":"✅","negative":"🔴","neutral":"⚪"}.get(sent,"⚪")
                rows.append({"Date": e.get("event_date","?")[:10],
                             "Type": e.get("event_type","").replace("_"," ").title(),
                             "Sentiment": f"{icon} {sent.title()}",
                             "Importance": e.get("importance","").title()})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("Announcement details are in Arabic — use AI Chat for English interpretation")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — TRACK RECORD
# ══════════════════════════════════════════════════════════════════════════════

def tab_track_record(sym: str):
    info   = STOCKS.get(sym, {})
    engine = load_stock_engine(sym)
    st.markdown(f'<div class="main-header">📊 Track Record — {info.get("emoji","")} {info.get("name",sym)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">How well has the engine predicted this stock? Where has it been wrong?</div>', unsafe_allow_html=True)

    bt      = engine.get("backtest", {})
    v       = bt.get("validation", {})
    mistakes= engine.get("mistakes", [])
    mem     = engine.get("memory", {})
    pers    = engine.get("personality", engine.get("personality_summary", {}))

    if not v:
        st.warning("Track record data not found.")
        return

    acc      = v.get("directional_accuracy_pct", 0)
    baseline = v.get("baseline_always_long_pct", 0)
    edge     = v.get("edge_over_baseline_pct", 0)
    n_preds  = v.get("n_predictions", 0)

    acc_color = "#28a745" if edge > 3 else ("#ffc107" if edge > 0 else "#dc3545")
    verdict   = "Better than random buying" if edge > 3 else ("Slightly better than random" if edge > 0 else "Not reliably better than just always buying")

    st.markdown(f"""<div style="background:{acc_color}22;border-left:5px solid {acc_color};
    padding:1rem;border-radius:8px;margin-bottom:1rem">
    <b style="color:{acc_color};font-size:1.1rem">Engine accuracy: {acc:.0f}% correct direction</b><br>
    vs just always buying = {baseline:.0f}% | Edge: <b>{edge:+.1f}%</b><br>
    <span style="color:#444;font-size:0.9rem">Based on {n_preds} quarterly predictions (2016-2026). {verdict}.</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 🎓 What This Means in Plain English")
    st.markdown(f"""
Out of every **10 predictions** the engine made on this stock:
- **{acc/10:.1f} times** it correctly called the direction (up or down)
- **{(100-acc)/10:.1f} times** it was wrong

If you had just bought every time: you would have been right **{baseline/10:.1f} out of 10**.
The engine adds **{edge/10:.1f} extra correct calls** per 10.
""")

    rd = v.get("regime_breakdown", {})
    if rd:
        st.markdown("---")
        st.markdown("### 🌡️ Accuracy by Market Environment")
        for reg, r in rd.items():
            regime_acc = r.get("accuracy", r.get("dir_accuracy", 0))
            r_color = "#28a745" if regime_acc > 65 else ("#ffc107" if regime_acc > 50 else "#dc3545")
            st.markdown(f"**{reg.capitalize()} rates:** <span style='color:{r_color}'>{regime_acc:.0f}% accurate</span> (n={r.get('n',0)} predictions)", unsafe_allow_html=True)

    preds  = bt.get("predictions", [])
    recent = [p for p in preds if p.get("actual_90d_pct") is not None][-8:]
    if recent:
        st.markdown("---")
        st.markdown("### 🔮 Last 8 Predictions vs What Actually Happened")
        rows = []
        for p in reversed(recent):
            pred_val = p.get("base_90d_pct", 0)
            actual   = p.get("actual_90d_pct", 0)
            correct  = (pred_val > 0) == (actual > 0)
            rows.append({"Date": p.get("prediction_date","?"),
                         "Engine said": f"{pred_val:+.1f}%",
                         "Actual": f"{actual:+.1f}%",
                         "Result": f"{'✅ Correct' if correct else '❌ Wrong'}",
                         "Score at time": f"{p.get('composite',0):.0f}/100"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    rules = pers.get("personality_rules", [])
    if rules:
        st.markdown("---")
        st.markdown("### 🧠 What the Engine Has Learned About This Stock")
        st.caption("Rules derived purely from historical evidence — not assumptions.")
        icons = {"HIGH":"🟢","MEDIUM":"🟡","LOW":"🔴"}
        for rule in rules[:5]:
            conf = rule.get("confidence","MEDIUM")
            st.markdown(f"{icons.get(conf,'⚪')} **{rule.get('rule','')}**")
            st.caption(f"Evidence: {rule.get('evidence','N/A')}")

    if mistakes:
        st.markdown("---")
        st.markdown("### ⚠️ Biggest Mistakes — Where the Engine Was Wrong")
        st.caption("Understanding failures is as important as celebrating successes.")
        for m in sorted(mistakes, key=lambda x: -x.get("error",0))[:5]:
            err, actual, pred, cause = m.get("error",0), m.get("actual_90d",0), m.get("predicted_90d",0), m.get("root_cause","Unknown")
            st.markdown(f'''<div class="warning-box">
            <b>{m.get("prediction_date","?")}</b> — predicted {pred:+.1f}%, actual was {actual:+.1f}% (error: {err:.0f}%)<br>
            <span style="font-size:0.85rem">Root cause: {cause}</span>
            </div>''', unsafe_allow_html=True)


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    sym  = render_sidebar()
    info = STOCKS.get(sym, {})
    st.markdown(f'<div class="main-header">{info.get("emoji","")} {info.get("name",sym)} ({sym})</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{info.get("sector","?")} · Engine analysis</div>', unsafe_allow_html=True)

    t1,t2,t3,t4,t5,t6 = st.tabs([
        "💬 AI Chat", "🎯 Forecast", "📈 Technicals",
        "💰 Financials", "👁️ Investor View", "📊 Track Record",
    ])
    with t1: tab_chat(sym)
    with t2: tab_forecast(sym)
    with t3: tab_technicals(sym)
    with t4: tab_financials(sym)
    with t5: tab_investor_view(sym)
    with t6: tab_track_record(sym)


if __name__ == "__main__":
    main()
