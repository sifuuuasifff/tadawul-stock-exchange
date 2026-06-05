"""
PAGE 2 — STOCK ANALYSIS & AI CHAT
==================================
Individual stock deep-dive: AI chat, forecast, personality, signals, hypotheses, backtest.
Select any stock from the sidebar and explore its full engine output.
"""

import sys, json, os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import anthropic

from config.settings import MEMORY_DIR, REPORTS_DIR, DATA_RAW, ANTHROPIC_API_KEY
os.environ.setdefault("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)

# Import STOCKS registry and loaders from home page
from portal import STOCKS, load_all_states, build_stock_df

st.set_page_config(
    page_title="Stock Analysis — Tadawul Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header  { font-size:1.8rem; font-weight:700; color:#1a1a2e; margin-bottom:0.2rem; }
    .sub-header   { font-size:0.95rem; color:#666; margin-bottom:1.5rem; }
    .forecast-box { background:#f0f7ff; border-radius:10px; padding:1.2rem;
                    margin:0.5rem 0; border:1px solid #cce0ff; }
    .warning-box  { background:#fff8e1; border-radius:8px; padding:0.8rem;
                    border-left:3px solid #ffc107; }
    .chat-user    { background:#e8f4fd; border-radius:12px; padding:0.8rem 1rem; margin:0.4rem 0; }
    .chat-ai      { background:#f8f9fa; border-radius:12px; padding:0.8rem 1rem;
                    margin:0.4rem 0; border-left:3px solid #0066cc; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_json(path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def load_stock_state(sym: str) -> dict:
    return load_all_states().get(sym, {})

def load_stock_engine(sym: str) -> dict:
    prefix = f"{sym}_" if sym != "1120" else ""
    data = {}
    for key, fname in [
        ("hypotheses", f"{prefix}HYPOTHESIS_REPORT.json"),
        ("backtest",   f"{prefix}PREDICTION_BACKTEST_REPORT.json"),
        ("signals",    f"{prefix}SIGNAL_DISCOVERY_REPORT.json"),
        ("recal",      f"{prefix}RECALIBRATION_REPORT.json"),
    ]:
        p = REPORTS_DIR / fname
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data[key] = json.load(f)
    for key, fname in [
        ("memory",   f"memory_{sym}.json"),
        ("mistakes", f"mistake_vault_{sym}.json"),
        ("personality", f"personality_{sym}.json"),
        ("personality_summary", "personality_summary.json"),
    ]:
        p = MEMORY_DIR / fname
        if p.exists():
            with open(p, encoding="utf-8") as f:
                raw = json.load(f)
                data[key] = raw.get("mistakes", raw) if key == "mistakes" else raw
    return data


# ── AI context ────────────────────────────────────────────────────────────────

def build_context(sym: str) -> str:
    states  = load_all_states()
    engine  = load_stock_engine(sym)
    info    = STOCKS.get(sym, {})

    hyp     = engine.get("hypotheses", {})
    bt      = engine.get("backtest", {})
    mem     = engine.get("memory", {})
    pers    = engine.get("personality", engine.get("personality_summary", {}))
    mistakes= engine.get("mistakes", [])
    state   = states.get(sym, {})

    all_states_summary = {s: {
        "price": states[s].get("price"),
        "composite": states[s].get("composite"),
        "rsi": states[s].get("rsi"),
        "rate_regime": states[s].get("rate_regime"),
        "forecast_90d": states[s].get("forecast", {}).get("base_90d_pct"),
    } for s in list(states.keys())[:20]}

    return f"""
You are the AI Research Engine for a private Tadawul Stock Memory Engine.
Today: {datetime.now().strftime('%Y-%m-%d')}
All {len(STOCKS)} stocks are in the engine with full backtests and hypotheses.

SELECTED STOCK: {info.get('name','?')} ({sym}) — {info.get('sector','?')}

CURRENT STATE:
{json.dumps(state, indent=2, default=str)}

PERSONALITY RULES:
{json.dumps(pers.get('personality_rules', []), indent=2, default=str)}

HYPOTHESIS RESULTS ({hyp.get('summary',{}).get('total','?')} tested):
Accepted={hyp.get('summary',{}).get('accepted','?')} | Rejected={hyp.get('summary',{}).get('rejected','?')}
KEY ACCEPTED: {json.dumps([r for r in hyp.get('results',[]) if r.get('verdict')=='ACCEPTED'], default=str)}

BACKTEST:
{json.dumps(bt.get('validation',{}), indent=2, default=str)}

MEMORY (regime profiles):
{json.dumps(mem.get('regime_profiles',[])[:6], indent=2, default=str)}

TOP MISTAKES:
{json.dumps(mistakes[:4], indent=2, default=str)}

ALL STOCKS SNAPSHOT (composite scores):
{json.dumps(all_states_summary, indent=2, default=str)}

SIGNAL WEIGHTS: Environment 45% | Technical 30% | Fundamental 25%

INSTRUCTIONS:
- Answer in plain English. User is non-technical.
- Respond in Arabic if asked in Arabic.
- Be direct. Give numbers. No unnecessary disclaimers.
- Do NOT give financial advice — research and evidence only.
- When comparing stocks, use the all-stocks snapshot above.
"""


def ask_claude(question: str, context: str, history: list) -> str:
    try:
        client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
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

        sectors = ["All"] + sorted(set(v["sector"] for v in STOCKS.values()))
        sel_sec = st.selectbox("Filter sector", sectors, index=0)
        filtered = STOCKS if sel_sec == "All" else {k:v for k,v in STOCKS.items() if v["sector"]==sel_sec}

        sym = st.selectbox(
            "Select stock",
            options=list(filtered.keys()),
            format_func=lambda s: f"{STOCKS[s]['emoji']} {STOCKS[s]['name']} ({s})",
        )

        st.markdown("---")
        state = load_stock_state(sym)
        if state:
            price = state.get("price", 0)
            comp  = state.get("composite", 0)
            rsi   = state.get("rsi", 50)
            fc    = state.get("forecast", {})
            color = "green" if comp > 62 else ("orange" if comp > 50 else "red")
            c1, c2 = st.columns(2)
            c1.metric("Price", f"{price:.2f} SAR")
            c2.metric("RSI", f"{rsi:.1f}")
            st.markdown(f"Composite: **:{color}[{comp:.0f}/100]**")
            b90  = fc.get("base_90d_pct", 0)
            conf = fc.get("confidence_label", "?")
            arrow = "🟢" if b90 > 3 else ("🟡" if b90 > 0 else "🔴")
            st.markdown(f"90d: {arrow} **{b90:+.1f}%** ({conf})")
            st.caption(f"Regime: {state.get('rate_regime','?').upper()} @ {state.get('repo_rate','?')}%")

        st.markdown("---")
        st.page_link("portal.py",                 label="← Back to Summary", icon="📊")

    return sym


# ── Tabs ───────────────────────────────────────────────────────────────────────

def tab_chat(sym: str):
    info = STOCKS.get(sym, {})
    st.markdown(f'<div class="main-header">💬 AI Chat — {info.get("emoji","")} {info.get("name",sym)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask anything in plain English or Arabic</div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    context = build_context(sym)

    # Quick questions
    name = info.get("name", sym)
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
        css = "chat-user" if msg["role"] == "user" else "chat-ai"
        icon = "🧑" if msg["role"] == "user" else "🤖"
        st.markdown(f'<div class="{css}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input("Your question",
            placeholder="e.g. What does the engine say about this stock right now?",
            value=st.session_state.pop("pending_q", ""))
        submitted = st.form_submit_button("Ask →", use_container_width=True)

    if submitted and question.strip():
        with st.spinner("Thinking..."):
            answer = ask_claude(question, context, st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "user",      "content": question})
        st.session_state.chat_history.append({"role": "assistant",  "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()


def tab_forecast(sym: str):
    info  = STOCKS.get(sym, {})
    state = load_stock_state(sym)
    st.markdown(f'<div class="main-header">🎯 Forecast — {info.get("emoji","")} {info.get("name",sym)}</div>', unsafe_allow_html=True)

    if not state:
        st.warning("No current state found. Run build_all_forecasts.py.")
        return

    fc = state.get("forecast", {})
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Environment", f"{state.get('env_score',0):.0f}/100")
    c2.metric("Technical",   f"{state.get('tech_score',0):.0f}/100")
    c3.metric("Fundamental", f"{state.get('fund_score',0):.0f}/100")
    c4.metric("Composite",   f"{state.get('composite',0):.0f}/100")

    st.markdown("---")
    c1,c2,c3 = st.columns(3)
    for col, label, pct, tgt in [
        (c1, "30-day target", fc.get("base_30d_pct",0), fc.get("target_30d",0)),
        (c2, "60-day target", fc.get("base_60d_pct",0), fc.get("target_60d",0)),
        (c3, "90-day target", fc.get("base_90d_pct",0), fc.get("target_90d",0)),
    ]:
        color = "signal-good" if pct > 0 else "signal-bad"
        col.markdown(f"""<div class="forecast-box">
        <b>{label}</b><br>
        <span style="font-size:1.4rem;font-weight:700;">{tgt:.2f} SAR</span><br>
        <span class="{color}">{pct:+.1f}%</span>
        </div>""", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    c1.markdown(f'<div class="warning-box">🐂 Bull (90d): {fc.get("bull_target_90d",0):.2f} SAR ({fc.get("bull_90d_pct",0):+.1f}%)</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="warning-box">🐻 Bear (90d): {fc.get("bear_target_90d",0):.2f} SAR ({fc.get("bear_90d_pct",0):+.1f}%)</div>', unsafe_allow_html=True)

    st.markdown(f"**Confidence:** {fc.get('confidence',0)}/100 ({fc.get('confidence_label','?')})")
    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Main drivers:**")
        for d in state.get("env_signals", {}).items():
            st.markdown(f"• {d[0]}: {d[1]}")
        for d in state.get("tech_signals", {}).items():
            st.markdown(f"• {d[0]}: {d[1]}")
    with c2:
        st.markdown("**Key risks:**")
        if state.get("rate_regime") == "rising":
            st.markdown("⚠️ Rising rates — historically bad for most Tadawul stocks")
        if state.get("rsi", 50) > 70:
            st.markdown("⚠️ RSI overbought — mean reversion risk")
        if state.get("rsi", 50) < 30:
            st.markdown("✅ RSI oversold — historically a buying opportunity")


def tab_personality(sym: str):
    info   = STOCKS.get(sym, {})
    engine = load_stock_engine(sym)
    st.markdown(f'<div class="main-header">🧠 Personality — {info.get("emoji","")} {info.get("name",sym)}</div>', unsafe_allow_html=True)

    pers  = engine.get("personality", engine.get("personality_summary", {}))
    mem   = engine.get("memory", {})
    rules = pers.get("personality_rules", [])

    # Baseline and regime
    baseline = mem.get("baseline_90d", {})
    regimes  = mem.get("regime_profiles", [])
    rate_regimes = [r for r in regimes if r.get("type") == "Rate Regime"]
    if baseline:
        st.metric("Baseline 90d avg", f"{baseline.get('avg_pct','?')}%",
                  delta=f"Win rate: {baseline.get('win_pct','?')}%", delta_color="off")
    if rate_regimes:
        cols = st.columns(len(rate_regimes))
        for i, r in enumerate(rate_regimes):
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

    baseline = report.get("baseline", {})
    base_90d = baseline.get("avg_90d", 6.0) or 6.0

    col1,col2 = st.columns(2)
    group_f = col1.selectbox("Group", ["All"] + sorted(set(s.get("group","") for s in signals)))
    rel_f   = col2.selectbox("Reliability", ["All","HIGH","MEDIUM","LOW"])

    filtered = [s for s in signals
                if (group_f == "All" or s.get("group") == group_f)
                and (rel_f == "All" or s.get("reliability_label") == rel_f)
                and "avg_return_90d" in s]

    rows = []
    for s in filtered:
        edge = (s.get("avg_return_90d") or 0) - base_90d
        rows.append({
            "Signal":      s.get("signal","")[:55],
            "N":           s.get("occurrences",0),
            "Avg 30d %":   s.get("avg_return_30d"),
            "Avg 90d %":   s.get("avg_return_90d"),
            "Win% (90d)":  s.get("pct_positive_90d"),
            "Edge":        f"{edge:+.1f}%",
            "Reliability": s.get("reliability_label",""),
            "p-value":     s.get("p_value_90d"),
            "Sig":         "✓" if s.get("significant_90d") else "",
        })
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
        if vf != "All" and r.get("verdict") != vf:
            continue
        icon = {"ACCEPTED":"✓","REJECTED":"✗","INCONCLUSIVE":"~"}.get(r.get("verdict",""),"?")
        rows.append({
            "ID":         r.get("id"),
            "Verdict":    f"{icon} {r.get('verdict','')}",
            "Hypothesis": r.get("hypothesis","")[:65],
            "Δ %":        r.get("diff"),
            "p-value":    r.get("p_value"),
            "n":          r.get("n_a"),
            "Reason":     r.get("reason","")[:70],
        })
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
        for i, (regime, r) in enumerate(rd.items()):
            cols[i].metric(f"{regime.capitalize()} Rates",
                           f"{r.get('dir_accuracy',0)}% acc",
                           f"error {r.get('avg_error',0):+.1f}%")

    st.markdown("---")
    st.markdown("### 🚨 Mistake Vault")
    mistakes = engine.get("mistakes", [])
    if mistakes:
        rows = []
        for m in sorted(mistakes, key=lambda x: -x.get("error",0)):
            rows.append({
                "Date":      m.get("prediction_date"),
                "Predicted": f"{m.get('predicted_90d',0):+.1f}%",
                "Actual":    f"{m.get('actual_90d',0):+.1f}%",
                "Error":     f"{m.get('error',0):.1f}%",
                "Dir OK":    "✓" if m.get("direction_correct") else "✗",
                "RSI":       m.get("rsi"),
                "Regime":    m.get("rate_regime"),
                "Root Cause":m.get("root_cause","")[:65],
            })
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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💬 AI Chat",
        "🎯 Forecast",
        "🧠 Personality",
        "📡 Signals",
        "🧪 Hypotheses",
        "📈 Backtest",
    ])

    with tab1: tab_chat(sym)
    with tab2: tab_forecast(sym)
    with tab3: tab_personality(sym)
    with tab4: tab_signals(sym)
    with tab5: tab_hypotheses(sym)
    with tab6: tab_backtest(sym)


if __name__ == "__main__":
    main()
