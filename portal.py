"""
TADAWUL STOCK MEMORY ENGINE — AI RESEARCH PORTAL
=================================================
A private conversational interface powered by Claude AI.
Ask questions in plain English or Arabic. Get evidence-based answers.

Run:  streamlit run portal.py
"""

import sys
import json
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import anthropic

from config.settings import (
    BASE_DIR, MEMORY_DIR, REPORTS_DIR, DATA_PROCESSED, DATA_RAW, ANTHROPIC_API_KEY,
)
import os
os.environ.setdefault("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tadawul Stock Memory Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.2rem; }
    .sub-header  { font-size: 0.95rem; color: #666; margin-bottom: 1.5rem; }
    .metric-card { background: #f8f9fa; border-radius: 10px; padding: 1rem; border-left: 4px solid #0066cc; }
    .signal-good { color: #28a745; font-weight: 600; }
    .signal-bad  { color: #dc3545; font-weight: 600; }
    .signal-neutral { color: #6c757d; }
    .forecast-box { background: #f0f7ff; border-radius: 10px; padding: 1.2rem; margin: 0.5rem 0; border: 1px solid #cce0ff; }
    .warning-box  { background: #fff8e1; border-radius: 8px; padding: 0.8rem; border-left: 3px solid #ffc107; }
    .chat-msg-user { background: #e8f4fd; border-radius: 12px; padding: 0.8rem 1rem; margin: 0.4rem 0; }
    .chat-msg-ai   { background: #f8f9fa; border-radius: 12px; padding: 0.8rem 1rem; margin: 0.4rem 0; border-left: 3px solid #0066cc; }
</style>
""", unsafe_allow_html=True)

# ── Stock registry ────────────────────────────────────────────────────────────
STOCKS = {
    # Banking
    "1120": {"name": "Al Rajhi Bank",            "sector": "Banking",        "emoji": "🏦"},
    "1180": {"name": "Saudi National Bank (SNB)", "sector": "Banking",        "emoji": "🏦"},
    "1010": {"name": "Riyad Bank",                "sector": "Banking",        "emoji": "🏦"},
    "1060": {"name": "Saudi British Bank (SABB)", "sector": "Banking",        "emoji": "🏦"},
    "1080": {"name": "Arab National Bank",        "sector": "Banking",        "emoji": "🏦"},
    "1050": {"name": "Banque Saudi Fransi",       "sector": "Banking",        "emoji": "🏦"},
    "1150": {"name": "Alinma Bank",               "sector": "Banking",        "emoji": "🏦"},
    "1140": {"name": "Bank Albilad",              "sector": "Banking",        "emoji": "🏦"},
    # Other sectors
    "2230": {"name": "Saudi Chemical",            "sector": "Chemicals",      "emoji": "🧪"},
    "1211": {"name": "Maaden",                    "sector": "Mining",         "emoji": "⛏️"},
    "4015": {"name": "Jamjoom Pharma",            "sector": "Pharmaceuticals","emoji": "💊"},
    # Large Caps
    "2222": {"name": "Saudi Aramco", "sector": "Energy", "emoji": "🛢️"},
    "2010": {"name": "SABIC", "sector": "Petrochemicals", "emoji": "⚗️"},
    "7010": {"name": "STC", "sector": "Telecom", "emoji": "📡"},
    "2270": {"name": "Savola Group", "sector": "Food & Beverages", "emoji": "🥛"},
    "2050": {"name": "Saudi Arabia Fertilizers", "sector": "Food & Beverages", "emoji": "🥛"},
    "6020": {"name": "Halwani Brothers", "sector": "Food & Beverages", "emoji": "🥛"},
    "2100": {"name": "Wafrah for Industry", "sector": "Food & Beverages", "emoji": "🥛"},
    "6010": {"name": "NADEC", "sector": "Food & Beverages", "emoji": "🥛"},
    "2200": {"name": "Arabian Food Industries", "sector": "Food & Beverages", "emoji": "🥛"},
    "3010": {"name": "Yamama Cement", "sector": "Cement", "emoji": "🏗️"},
    "3020": {"name": "Saudi Cement", "sector": "Cement", "emoji": "🏗️"},
    "3030": {"name": "Qassim Cement", "sector": "Cement", "emoji": "🏗️"},
    "3040": {"name": "Southern Province Cement", "sector": "Cement", "emoji": "🏗️"},
    "3050": {"name": "Yanbu Cement", "sector": "Cement", "emoji": "🏗️"},
    "3060": {"name": "City Cement", "sector": "Cement", "emoji": "🏗️"},
    "3080": {"name": "Tabuk Cement", "sector": "Cement", "emoji": "🏗️"},
    "3090": {"name": "Arabian Cement", "sector": "Cement", "emoji": "🏗️"},
    "4110": {"name": "Arriyadh Dev Auth", "sector": "Diversified", "emoji": "📦"},
    "2180": {"name": "Fitaihi Holding", "sector": "Diversified", "emoji": "📦"},
    "4140": {"name": "Saudi Ground Services", "sector": "Transport", "emoji": "✈️"},
    "4030": {"name": "Saudi Airlines Catering", "sector": "Transport", "emoji": "✈️"},
    "2080": {"name": "Saudi Electricity", "sector": "Utilities", "emoji": "⚡"},
    "5110": {"name": "Saudi Telecom Infra", "sector": "Utilities", "emoji": "⚡"},
    "1830": {"name": "Leejam Sports", "sector": "Consumer Services", "emoji": "🏋️"},
    "4280": {"name": "Tabreed", "sector": "Utilities", "emoji": "⚡"},

    # Healthcare
    "4013": {"name": "Dr Sulaiman Al Habib", "sector": "Healthcare", "emoji": "🏥"},
    "4002": {"name": "Mouwasat Medical", "sector": "Healthcare", "emoji": "🏥"},
    "4004": {"name": "Dallah Healthcare", "sector": "Healthcare", "emoji": "🏥"},
    "4345": {"name": "Saudi German Health", "sector": "Healthcare", "emoji": "🏥"},
    "4007": {"name": "Al Hammadi", "sector": "Healthcare", "emoji": "🏥"},
    "4005": {"name": "National Medical Care", "sector": "Healthcare", "emoji": "🏥"},
    "4006": {"name": "Specialized Medical", "sector": "Healthcare", "emoji": "🏥"},
    "2070": {"name": "Saudi Pharmaceutical", "sector": "Healthcare", "emoji": "🏥"},

    # Real Estate
    "4300": {"name": "Dar Al Arkan", "sector": "Real Estate", "emoji": "🏢"},
    "4220": {"name": "Emaar EC", "sector": "Real Estate", "emoji": "🏢"},
    "4322": {"name": "Retal Urban Dev", "sector": "Real Estate", "emoji": "🏢"},
    "4323": {"name": "Roshn Real Estate", "sector": "Real Estate", "emoji": "🏢"},
    "4020": {"name": "Saudi Real Estate", "sector": "Real Estate", "emoji": "🏢"},
    "4250": {"name": "Jabal Omar Dev", "sector": "Real Estate", "emoji": "🏢"},
    "4100": {"name": "Makkah Construction", "sector": "Real Estate", "emoji": "🏢"},
    "4090": {"name": "Arriyadh Development", "sector": "Real Estate", "emoji": "🏢"},

    # Retail & Consumer
    "4190": {"name": "Jarir Bookstore", "sector": "Retail", "emoji": "🛒"},
    "4012": {"name": "Extra (United Electronics)", "sector": "Retail", "emoji": "🛒"},
    "4163": {"name": "Nahdi Medical", "sector": "Retail", "emoji": "🛒"},
    "4161": {"name": "BinDawood Holding", "sector": "Retail", "emoji": "🛒"},
    "4001": {"name": "Al Othaim Markets", "sector": "Retail", "emoji": "🛒"},
    "4082": {"name": "SACO", "sector": "Retail", "emoji": "🛒"},
    "6004": {"name": "Fawaz Abdulaziz Alhokair", "sector": "Retail", "emoji": "🛒"},
    "4003": {"name": "Astra Industrial", "sector": "Retail", "emoji": "🛒"},

    "7020": {"name": "Mobily", "sector": "Telecom", "emoji": "📡"},
    "7030": {"name": "Zain Saudi", "sector": "Telecom", "emoji": "📡"},
    "7040": {"name": "Etihad Atheeb", "sector": "Telecom", "emoji": "📡"},
    "7200": {"name": "Solutions by STC", "sector": "Telecom", "emoji": "📡"},

    "2280": {"name": "Almarai", "sector": "Food & Beverages", "emoji": "🥛"},
    # Petrochemicals
    "2020": {"name": "SABIC Agri-Nutrients", "sector": "Petrochemicals", "emoji": "⚗️"},
    "2290": {"name": "Yanbu National Petro", "sector": "Petrochemicals", "emoji": "⚗️"},
    "2350": {"name": "Saudi Kayan Petrochem", "sector": "Petrochemicals", "emoji": "⚗️"},
    "2310": {"name": "Sipchem", "sector": "Petrochemicals", "emoji": "⚗️"},
    "2380": {"name": "Petro Rabigh", "sector": "Petrochemicals", "emoji": "⚗️"},
    "2210": {"name": "Nama Chemicals", "sector": "Petrochemicals", "emoji": "⚗️"},
    "2060": {"name": "National Industrialization", "sector": "Petrochemicals", "emoji": "⚗️"},
    "2030": {"name": "Advanced Petrochem", "sector": "Petrochemicals", "emoji": "⚗️"},
    "2260": {"name": "Sahara International Petrochem", "sector": "Petrochemicals", "emoji": "⚗️"},
    "2250": {"name": "Saudi Industrial Investment", "sector": "Petrochemicals", "emoji": "⚗️"},

}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS  (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_personality(sym: str) -> dict:
    return load_json(MEMORY_DIR / f"personality_{sym}.json")


@st.cache_data(ttl=300)
def load_current_forecast() -> dict:
    return load_json(MEMORY_DIR / "current_forecast.json")


@st.cache_data(ttl=300)
def load_hypothesis_report() -> dict:
    return load_json(REPORTS_DIR / "HYPOTHESIS_REPORT.json")


@st.cache_data(ttl=300)
def load_personality_summary() -> dict:
    return load_json(MEMORY_DIR / "personality_summary.json")


@st.cache_data(ttl=300)
def load_signal_report() -> dict:
    return load_json(REPORTS_DIR / "SIGNAL_DISCOVERY_REPORT.json")


@st.cache_data(ttl=300)
def load_backtest() -> dict:
    return load_json(REPORTS_DIR / "PREDICTION_BACKTEST_REPORT.json")


@st.cache_data(ttl=300)
def load_mistake_vault() -> dict:
    return load_json(MEMORY_DIR / "mistake_vault.json")


@st.cache_data(ttl=300)
def load_price(sym: str) -> pd.DataFrame:
    # Al Rajhi uses the main file; others use stock subfolder
    if sym == "1120":
        path = DATA_RAW / "alrajhi_price_daily.csv"
    else:
        path = DATA_RAW / f"stock_{sym}" / "price_yahoo.csv"
        if not path.exists():
            path = DATA_RAW / f"stock_{sym}" / "price_sahmk.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna(subset=["close"])


# ══════════════════════════════════════════════════════════════════════════════
# AI CONTEXT BUILDER  — gives Claude everything it needs to answer questions
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def load_all_states() -> dict:
    p = MEMORY_DIR / "all_current_states.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("stocks", {})


@st.cache_data(ttl=300)
def load_stock_full_data(sym: str) -> dict:
    """Load all engine outputs for a stock."""
    prefix = f"{sym}_" if sym != "1120" else ""
    data   = {}
    # Hypothesis report
    p = REPORTS_DIR / f"{prefix}HYPOTHESIS_REPORT.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data["hypotheses"] = json.load(f)
    # Backtest
    p = REPORTS_DIR / f"{prefix}PREDICTION_BACKTEST_REPORT.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data["backtest"] = json.load(f)
    # Memory
    p = MEMORY_DIR / f"memory_{sym}.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data["memory"] = json.load(f)
    # Mistake vault
    p = MEMORY_DIR / f"mistake_vault_{sym}.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data["mistakes"] = json.load(f).get("mistakes", [])
    # Signal report
    p = REPORTS_DIR / f"{prefix}SIGNAL_DISCOVERY_REPORT.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data["signals"] = json.load(f)
    return data


def build_ai_context(selected_sym: str) -> str:
    states = load_all_states()

    # Load full engine data for all stocks
    all_engine_data = {}
    for sym in STOCKS:
        all_engine_data[sym] = load_stock_full_data(sym)

    # Al Rajhi specific
    alr_summary = load_personality_summary()

    def fmt_hyp(sym):
        hyp = all_engine_data[sym].get("hypotheses", {})
        if not hyp:
            return "Not available"
        s   = hyp.get("summary", {})
        acc = [r for r in hyp.get("results", []) if r.get("verdict") == "ACCEPTED"]
        return f"Total={s.get('total','?')} | Accepted={s.get('accepted','?')} | Rejected={s.get('rejected','?')}\nKEY ACCEPTED: {json.dumps(acc, default=str)}"

    def fmt_backtest(sym):
        bt = all_engine_data[sym].get("backtest", {})
        return json.dumps(bt.get("validation", {}), default=str) if bt else "Not available"

    def fmt_mistakes(sym):
        m = all_engine_data[sym].get("mistakes", [])[:4]
        return json.dumps(m, default=str) if m else "None recorded"

    def fmt_memory(sym):
        mem = all_engine_data[sym].get("memory", {})
        return json.dumps({
            "personality_rules": mem.get("personality_rules", []),
            "baseline_90d":      mem.get("baseline_90d", {}),
            "regime_profiles":   mem.get("regime_profiles", []),
            "regime_combos":     mem.get("regime_combos", []),
        }, default=str) if mem else "Not available"

    context = f"""
You are the AI Research Engine for a private Tadawul Stock Memory Engine.
You have COMPLETE, EQUAL-DEPTH data for ALL of the following Saudi stocks.
Never say data is limited — you have full backtests and hypotheses for all stocks.

STOCKS: {json.dumps({k: v['name']+' ('+v['sector']+')' for k,v in STOCKS.items()}, ensure_ascii=False)}
TODAY: {datetime.now().strftime('%Y-%m-%d')}
SIGNAL WEIGHTS: Environment 45% | Technical 30% | Fundamental 25%

════════════════════════════════════════════════
CURRENT STATE — ALL 4 STOCKS (live as of today)
════════════════════════════════════════════════
{json.dumps(states, indent=2, ensure_ascii=False, default=str)}

════════════════════════════════════════════════
AL RAJHI BANK (1120) — FULL ENGINE
════════════════════════════════════════════════
PERSONALITY: {json.dumps(alr_summary.get('personality_rules',[]), default=str)}
HYPOTHESES:  {fmt_hyp('1120')}
BACKTEST:    {fmt_backtest('1120')}
MISTAKES:    {fmt_mistakes('1120')}
MEMORY:      {fmt_memory('1120')}

════════════════════════════════════════════════
SAUDI CHEMICAL (2230) — FULL ENGINE
════════════════════════════════════════════════
HYPOTHESES:  {fmt_hyp('2230')}
BACKTEST:    {fmt_backtest('2230')}
MISTAKES:    {fmt_mistakes('2230')}
MEMORY:      {fmt_memory('2230')}

════════════════════════════════════════════════
MAADEN (1211) — FULL ENGINE
════════════════════════════════════════════════
HYPOTHESES:  {fmt_hyp('1211')}
BACKTEST:    {fmt_backtest('1211')}
MISTAKES:    {fmt_mistakes('1211')}
MEMORY:      {fmt_memory('1211')}

════════════════════════════════════════════════
JAMJOOM PHARMA (4015) — FULL ENGINE
════════════════════════════════════════════════
HYPOTHESES:  {fmt_hyp('4015')}
BACKTEST:    {fmt_backtest('4015')}
MISTAKES:    {fmt_mistakes('4015')}
MEMORY:      {fmt_memory('4015')}

════════════════════════════════════════════════
INSTRUCTIONS
════════════════════════════════════════════════
- Plain English. User is non-technical.
- Respond in Arabic if asked in Arabic.
- Give full current data for ANY stock: price, RSI, scores, forecast, hypotheses, backtest accuracy.
- Always cite evidence: sample size, p-values, number of periods tested.
- NEVER say data is limited or unavailable — all 4 stocks have full engine data above.
- No financial advice. Research and evidence only.
- Be direct. No disclaimers. Just answer.
- When comparing stocks, highlight personality differences clearly with numbers.
"""
    return context


# ══════════════════════════════════════════════════════════════════════════════
# CLAUDE AI CHAT
# ══════════════════════════════════════════════════════════════════════════════

def ask_claude(question: str, context: str, history: list) -> str:
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        messages = []
        for msg in history[-6:]:  # last 6 messages for context
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": question})

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system=context,
            messages=messages,
        )
        return response.content[0].text
    except Exception as e:
        return f"Error connecting to AI: {e}. Check that ANTHROPIC_API_KEY is set."


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Stock overview
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("### 📊 Tadawul Research Engine")
        st.markdown(f"*{len(STOCKS)} stocks | {len(set(v['sector'] for v in STOCKS.values()))} sectors*")
        st.markdown("---")

        # ── Sector filter ─────────────────────────────────────────────────
        sectors = ["All"] + sorted(set(v["sector"] for v in STOCKS.values()))
        selected_sector = st.selectbox("Filter by Sector", options=sectors, index=0)

        # Filter stock list by sector
        if selected_sector == "All":
            filtered_stocks = STOCKS
        else:
            filtered_stocks = {k: v for k, v in STOCKS.items() if v["sector"] == selected_sector}

        # Show sector summary when filtering
        if selected_sector != "All":
            states = load_all_states()
            sector_stocks = list(filtered_stocks.keys())
            sector_composites = [
                states.get(s, {}).get("composite", 0)
                for s in sector_stocks if states.get(s)
            ]
            if sector_composites:
                avg_comp = sum(sector_composites) / len(sector_composites)
                color = "green" if avg_comp > 62 else ("orange" if avg_comp > 50 else "red")
                st.markdown(f"Sector avg composite: **:{color}[{avg_comp:.0f}/100]**")

        st.markdown("---")

        # ── Stock selector (filtered) ──────────────────────────────────────
        selected = st.selectbox(
            "Select Stock",
            options=list(filtered_stocks.keys()),
            format_func=lambda s: f"{STOCKS[s]['emoji']} {STOCKS[s]['name']} ({s})",
        )

        st.markdown("---")

        # Quick metrics for selected stock
        states = load_all_states()
        state  = states.get(selected, {})
        if state:
            price  = state.get("price", 0)
            comp   = state.get("composite", 0)
            rsi    = state.get("rsi", 50)
            fc_d   = state.get("forecast", {})
            color  = "green" if comp > 62 else ("orange" if comp > 50 else "red")

            col1, col2 = st.columns(2)
            col1.metric("Price", f"{price:.2f} SAR")
            col2.metric("RSI", f"{rsi:.1f}")

            st.markdown(f"Composite: **:{color}[{comp:.0f}/100]**")
            b90 = fc_d.get("base_90d_pct", 0)
            conf = fc_d.get("confidence_label", "?")
            arrow = "🟢" if b90 > 3 else ("🟡" if b90 > 0 else "🔴")
            st.markdown(f"90d forecast: {arrow} **{b90:+.1f}%** ({conf})")
            st.caption(f"Regime: {state.get('rate_regime','?').upper()} @ {state.get('repo_rate','?')}%")
            st.caption(f"{state.get('trading_days','?')} trading days of history")
        else:
            price_df = load_price(selected)
            if not price_df.empty:
                latest_price = price_df["close"].iloc[-1]
                prev_price   = price_df["close"].iloc[-2] if len(price_df) > 1 else latest_price
                change_pct   = (latest_price / prev_price - 1) * 100
                col1, col2 = st.columns(2)
                col1.metric("Price", f"{latest_price:.2f} SAR")
                col2.metric("1d", f"{change_pct:+.2f}%", delta_color="normal")

        st.markdown("---")

        # ── Sector Opportunity Ranker ──────────────────────────────────────
        st.markdown("**🏆 Top Opportunities Now**")
        all_states = load_all_states()
        ranked = sorted(
            [(s, all_states[s]) for s in all_states if all_states[s].get("composite")],
            key=lambda x: -x[1]["composite"]
        )
        for sym, st_data in ranked[:5]:
            comp_val = st_data.get("composite", 0)
            fc90     = st_data.get("forecast", {}).get("base_90d_pct", 0)
            color_r  = "green" if comp_val > 65 else "orange"
            st.markdown(
                f"**:{color_r}[{STOCKS.get(sym,{}).get('name', sym)}]** "
                f"— {comp_val:.0f}/100 | {fc90:+.1f}%"
            )

        st.markdown("---")
        st.markdown("**Engine Status**")
        checks = [
            ("Signal Discovery", (REPORTS_DIR / "SIGNAL_DISCOVERY_REPORT.json").exists()),
            ("Memory Engine",    (MEMORY_DIR   / "personality_summary.json").exists()),
            ("Hypotheses",       (REPORTS_DIR  / "HYPOTHESIS_REPORT.json").exists()),
            ("Walk-Forward",     (REPORTS_DIR  / "PREDICTION_BACKTEST_REPORT.json").exists()),
            ("Recalibration",    (REPORTS_DIR  / "RECALIBRATION_REPORT.json").exists()),
        ]
        for label, ok in checks:
            icon = "✅" if ok else "⏳"
            st.markdown(f"{icon} {label}")

        return selected


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

def tab_chat(selected_sym: str):
    st.markdown('<div class="main-header">💬 AI Research Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask anything about these stocks in plain English or Arabic</div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    context = build_ai_context(selected_sym)

    # Suggested questions
    st.markdown("**Quick questions:**")
    cols = st.columns(3)
    quick_questions = [
        "What is Al Rajhi's current forecast?",
        "What signals matter most for Al Rajhi?",
        "How do rising rates affect Al Rajhi?",
        "Compare Al Rajhi and Maaden",
        "What are the biggest mistakes the engine made?",
        "What is Maaden's personality?",
    ]
    for i, q in enumerate(quick_questions):
        if cols[i % 3].button(q, key=f"quick_{i}"):
            st.session_state.pending_question = q

    st.markdown("---")

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-msg-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-msg-ai">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    # Input
    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input(
            "Your question",
            placeholder="e.g. What does the engine say about Al Rajhi right now?",
            value=st.session_state.pop("pending_question", ""),
        )
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


def tab_forecast(selected_sym: str):
    st.markdown('<div class="main-header">🎯 Current Forecast</div>', unsafe_allow_html=True)

    if selected_sym != "1120":
        st.info(f"Full forecast model is currently built for Al Rajhi Bank (1120) only. Signal discovery and personality data for {STOCKS[selected_sym]['name']} is available in the Personality tab.")
        selected_sym = "1120"

    fc = load_current_forecast()
    if not fc:
        st.warning("No forecast found. Run phase4_walkforward.py first.")
        return

    st.markdown(f"**As of:** {fc.get('forecast_date', '?')} | **Price:** {fc.get('price', '?')} SAR")

    # Score gauges
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Environment", f"{fc.get('env_score',0)}/100",  help="Macro + rate regime + VIX + oil")
    col2.metric("Technical",   f"{fc.get('tech_score',0)}/100", help="RSI + BB + momentum + MAs")
    col3.metric("Fundamental", f"{fc.get('fund_score',0)}/100", help="Net income growth, balance sheet")
    col4.metric("Composite",   f"{fc.get('composite',0)}/100",  help="Weighted combination")

    st.markdown("---")

    # Forecast targets
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="forecast-box">
        <b>30-day target</b><br>
        <span style="font-size:1.4rem;font-weight:700;">{fc.get('target_30d','?')} SAR</span><br>
        <span class="signal-good">{fc.get('base_30d_pct',0):+.1f}%</span>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="forecast-box">
        <b>60-day target</b><br>
        <span style="font-size:1.4rem;font-weight:700;">{fc.get('target_60d','?')} SAR</span><br>
        <span class="signal-good">{fc.get('base_60d_pct',0):+.1f}%</span>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="forecast-box">
        <b>90-day target</b><br>
        <span style="font-size:1.4rem;font-weight:700;">{fc.get('target_90d','?')} SAR</span><br>
        <span class="signal-good">{fc.get('base_90d_pct',0):+.1f}%</span>
        </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.markdown(f"""<div class="warning-box">
    🐂 <b>Bull case (90d):</b> {fc.get('bull_target_90d','?')} SAR ({fc.get('bull_90d_pct',0):+.1f}%)
    </div>""", unsafe_allow_html=True)
    col2.markdown(f"""<div class="warning-box">
    🐻 <b>Bear case (90d):</b> {fc.get('bear_target_90d','?')} SAR ({fc.get('bear_90d_pct',0):+.1f}%)
    </div>""", unsafe_allow_html=True)

    st.markdown(f"**Confidence:** {fc.get('confidence',0)}/100 ({fc.get('confidence_label','?')})")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Main drivers:**")
        for d in fc.get("main_drivers", []):
            st.markdown(f"• {d}")
    with col2:
        st.markdown("**Key risks:**")
        for r in fc.get("key_risks", []):
            st.markdown(f"⚠️ {r}")

    sim = fc.get("similar_setups")
    if sim:
        st.markdown("---")
        st.markdown(f"**Similar historical setups:** {sim.get('n_similar_setups')} periods found")
        st.markdown(f"Average 90d outcome: **{sim.get('avg_90d_return')}%** | Win rate: **{sim.get('pct_positive_90d')}%**")


def tab_personality(selected_sym: str):
    st.markdown(f'<div class="main-header">{STOCKS[selected_sym]["emoji"]} {STOCKS[selected_sym]["name"]} — Stock Personality</div>', unsafe_allow_html=True)

    if selected_sym == "1120":
        summary = load_personality_summary()
        rules   = summary.get("personality_rules", [])
    else:
        p     = load_personality(selected_sym)
        rules = p.get("personality_rules", [])

        # Show basics
        col1, col2, col3 = st.columns(3)
        col1.metric("Latest Price", f"{p.get('latest_price','?')} SAR")
        col2.metric("Trading Days", p.get("n_trading_days", "?"))
        col3.metric("Baseline 90d Win%", f"{p.get('baseline_90d',{}).get('win_pct','?')}%")

        sig = p.get("key_signals", {})
        rd  = sig.get("regime_breakdown", {})
        if rd:
            st.markdown("**Rate Regime Performance:**")
            cols = st.columns(len(rd))
            for i, (regime, vals) in enumerate(rd.items()):
                cols[i].metric(
                    f"{regime.capitalize()} Rates",
                    f"{vals.get('avg_90d','?')}% avg",
                    f"{vals.get('win_pct','?')}% win rate",
                )
        st.markdown("---")

    st.markdown("**Evidence-Based Personality Rules:**")
    conf_colors = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}
    for rule in rules:
        conf  = rule.get("confidence", "MEDIUM")
        icon  = conf_colors.get(conf, "⚪")
        st.markdown(f"""
        **{icon} {rule.get('rule', rule.get('label',''))}**
        <div style="color:#666;font-size:0.85rem;margin-left:1.5rem;">
        Evidence: {rule.get('evidence','N/A')} | Confidence: {conf}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")


def tab_signals(selected_sym: str):
    st.markdown('<div class="main-header">📡 Signal Discovery</div>', unsafe_allow_html=True)

    if selected_sym != "1120":
        p = load_personality(selected_sym)
        sig = p.get("key_signals", {})

        st.markdown(f"**Key signals for {STOCKS[selected_sym]['name']}:**")
        signal_labels = {
            "rsi_oversold":       "RSI < 30 (Oversold)",
            "rsi_overbought":     "RSI > 70 (Overbought)",
            "bb_lower":           "Bollinger Band Lower Touch",
            "momentum_strong_up": "Strong Momentum (>10% rally)",
            "stable_rates":       "Stable Rate Regime",
            "rising_rates":       "Rising Rate Regime",
            "vix_fear":           "VIX > 30 (Fear)",
        }
        baseline_avg = sig.get("baseline_90d", {}).get("avg_pct", 6.0)
        rows = []
        for key, label in signal_labels.items():
            s = sig.get(key, {})
            if s and not s.get("insufficient"):
                edge = (s.get("avg_pct", baseline_avg) or 0) - (baseline_avg or 0)
                rows.append({
                    "Signal":      label,
                    "N":           s.get("n", "?"),
                    "Avg 90d %":   s.get("avg_pct"),
                    "Win %":       s.get("win_pct"),
                    "Edge vs base":f"{edge:+.1f}%",
                    "Significant": "✓" if s.get("significant") else "",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        return

    report = load_signal_report()
    signals = report.get("signals", [])
    if not signals:
        st.warning("Run phase1_signal_discovery.py first.")
        return

    # Filter controls
    col1, col2 = st.columns(2)
    group_filter = col1.selectbox("Signal group", ["All"] + sorted(set(s.get("group","") for s in signals)))
    rel_filter   = col2.selectbox("Reliability", ["All", "HIGH", "MEDIUM", "LOW"])

    filtered = signals
    if group_filter != "All":
        filtered = [s for s in filtered if s.get("group") == group_filter]
    if rel_filter != "All":
        filtered = [s for s in filtered if s.get("reliability_label") == rel_filter]

    baselines = report.get("baselines", {})
    base_90d  = baselines.get("always_long_90d", {}).get("avg_return", 6.0)

    rows = []
    for s in filtered:
        if "avg_return_90d" not in s:
            continue
        edge = (s.get("avg_return_90d") or 0) - (base_90d or 0)
        rows.append({
            "Signal":         s.get("signal", "")[:55],
            "Group":          s.get("group", ""),
            "N":              s.get("occurrences", 0),
            "Avg 30d %":      s.get("avg_return_30d"),
            "Avg 90d %":      s.get("avg_return_90d"),
            "Win % (90d)":    s.get("pct_positive_90d"),
            "Edge vs base":   f"{edge:+.1f}%",
            "Reliability":    s.get("reliability_label", ""),
            "p-value":        s.get("p_value_90d"),
            "Sig":            "✓" if s.get("significant_90d") else "",
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"Baseline (buy-and-hold): avg 90d = {base_90d}%")


def tab_hypotheses():
    st.markdown('<div class="main-header">🧪 Hypothesis Engine</div>', unsafe_allow_html=True)

    hyp = load_hypothesis_report()
    if not hyp:
        st.warning("Run phase3_hypothesis_engine.py first.")
        return

    results = hyp.get("results", [])
    summary = hyp.get("summary", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total",       summary.get("total", 0))
    col2.metric("Accepted ✓",  summary.get("accepted", 0))
    col3.metric("Rejected ✗",  summary.get("rejected", 0))
    col4.metric("Inconclusive ~", summary.get("inconclusive", 0))

    verdict_filter = st.selectbox("Filter", ["All", "ACCEPTED", "REJECTED", "INCONCLUSIVE"])

    rows = []
    for r in results:
        if verdict_filter != "All" and r.get("verdict") != verdict_filter:
            continue
        v = r.get("verdict", "?")
        icon = {"ACCEPTED": "✓", "REJECTED": "✗", "INCONCLUSIVE": "~"}.get(v, "?")
        rows.append({
            "ID":         r.get("id"),
            "Verdict":    f"{icon} {v}",
            "Hypothesis": r.get("hypothesis", "")[:70],
            "Δ %":        r.get("diff"),
            "p-value":    r.get("p_value"),
            "n":          r.get("n_a"),
            "Reason":     r.get("reason","")[:80],
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def tab_backtest():
    st.markdown('<div class="main-header">📈 Backtest & Mistake Vault</div>', unsafe_allow_html=True)

    bt = load_backtest()
    if not bt:
        st.warning("Run phase4_walkforward.py first.")
        return

    v = bt.get("validation", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predictions",  v.get("n_predictions", 0))
    col2.metric("Accuracy",     f"{v.get('directional_accuracy_pct',0)}%",
                delta=f"+{v.get('edge_over_baseline_pct',0):.1f}% vs baseline",
                delta_color="normal")
    col3.metric("Baseline",     f"{v.get('baseline_always_long_pct',0)}%")
    col4.metric("MAE (90d)",    f"{v.get('mae_pct',0)}%")

    # Regime breakdown
    rd = v.get("regime_breakdown", {})
    if rd:
        st.markdown("**Performance by Rate Regime:**")
        cols = st.columns(len(rd))
        for i, (regime, r) in enumerate(rd.items()):
            cols[i].metric(
                f"{regime.capitalize()} Rates",
                f"{r.get('dir_accuracy',0)}% accuracy",
                f"avg error {r.get('avg_error',0):+.1f}%",
            )

    st.markdown("---")
    st.markdown("**Confidence Calibration:**")
    cc = v.get("confidence_calibration", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("High confidence", f"{cc.get('high_conf_acc','?')}% acc", f"n={cc.get('high_conf_n','?')}")
    col2.metric("Med confidence",  f"{cc.get('medium_conf_acc','?')}% acc", f"n={cc.get('medium_conf_n','?')}")
    col3.metric("Low confidence",  f"{cc.get('low_conf_acc','?')}% acc", f"n={cc.get('low_conf_n','?')}")

    # Mistake Vault
    st.markdown("---")
    st.markdown("### 🚨 Mistake Vault")
    vault = load_mistake_vault()
    mistakes = vault.get("mistakes", [])
    if mistakes:
        rows = []
        for m in sorted(mistakes, key=lambda x: -x.get("error", 0)):
            rows.append({
                "Date":       m.get("prediction_date"),
                "Predicted":  f"{m.get('predicted_90d',0):+.1f}%",
                "Actual":     f"{m.get('actual_90d',0):+.1f}%",
                "Error":      f"{m.get('error',0):.1f}%",
                "Dir OK":     "✓" if m.get("direction_correct") else "✗",
                "RSI":        m.get("rsi"),
                "VIX":        m.get("vix"),
                "Regime":     m.get("rate_regime"),
                "Root Cause": m.get("root_cause","")[:70],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"{len(mistakes)} large errors (>12% miss) catalogued")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def tab_summary():
    st.markdown('<div class="main-header">📊 Tadawul Opportunity Engine — Summary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">74 stocks · 10 sectors · Updated {datetime.now().strftime("%Y-%m-%d")}</div>', unsafe_allow_html=True)

    states = load_all_states()
    if not states:
        st.warning("No stock data found. Run build_all_forecasts.py first.")
        return

    # Build a flat dataframe of all stocks
    rows = []
    for sym, s in states.items():
        if not s:
            continue
        info = STOCKS.get(sym, {})
        fc   = s.get("forecast", {})
        rows.append({
            "symbol":      sym,
            "name":        info.get("name", sym),
            "sector":      info.get("sector", "Other"),
            "emoji":       info.get("emoji", "📊"),
            "price":       s.get("price", 0),
            "rsi":         s.get("rsi", 50),
            "composite":   s.get("composite", 0),
            "env_score":   s.get("env_score", 0),
            "tech_score":  s.get("tech_score", 0),
            "fund_score":  s.get("fund_score", 0),
            "forecast_90d":fc.get("base_90d_pct", 0),
            "confidence":  fc.get("confidence", 0),
            "conf_label":  fc.get("confidence_label", "?"),
            "rate_regime": s.get("rate_regime", "stable"),
            "repo_rate":   s.get("repo_rate", 4.25),
            "vix":         s.get("vix", 20),
            "oil":         s.get("oil_price", 80),
            "target_90d":  fc.get("target_90d", 0),
        })
    df = pd.DataFrame(rows)

    # ── Section 1: Market Environment ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌍 Current Market Environment")

    first = rows[0] if rows else {}
    regime  = first.get("rate_regime", "stable")
    vix_val = first.get("vix", 20)
    oil_val = first.get("oil", 80)
    rate    = first.get("repo_rate", 4.25)

    regime_color = {"stable": "green", "rising": "red", "falling": "orange"}.get(regime, "gray")
    vix_label    = "🔴 High Fear" if vix_val > 30 else ("🟡 Elevated" if vix_val > 20 else "🟢 Calm")
    oil_label    = f"${oil_val:.0f}/bbl"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Saudi Repo Rate",   f"{rate:.2f}%",  delta=regime.capitalize(),
                delta_color="inverse" if regime == "rising" else "normal")
    col2.metric("VIX (Global Risk)", f"{vix_val:.1f}", delta=vix_label, delta_color="off")
    col3.metric("Brent Oil",         oil_label)
    col4.metric("Stocks Monitored",  f"{len(df)}", delta="74 stocks · 10 sectors", delta_color="off")

    env_msg = {
        "stable":  "✅ **Stable rate environment** — historically the best setup for Saudi equities. Conditions favour accumulation.",
        "rising":  "⚠️ **Rising rate environment** — historically challenging for most Tadawul stocks, especially banks. Be selective.",
        "falling": "🟡 **Falling rate environment** — mixed signals. Monitor carefully.",
    }.get(regime, "")
    if vix_val > 30:
        env_msg += " VIX is elevated — **fear creates opportunity** based on historical patterns."
    st.info(env_msg)

    # ── Section 2: Sector Scorecard ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏭 Sector Scorecard")

    sector_df = df.groupby("sector").agg(
        Stocks    = ("symbol", "count"),
        Avg_Score = ("composite", "mean"),
        Avg_90d   = ("forecast_90d", "mean"),
        Best_Score= ("composite", "max"),
    ).round(1).reset_index()

    # Find best stock per sector
    best_stock = df.loc[df.groupby("sector")["composite"].idxmax()][["sector","name","composite","forecast_90d"]]
    sector_df  = sector_df.merge(best_stock.rename(columns={
        "name":       "Top Pick",
        "composite":  "Top_Composite",
        "forecast_90d":"Top_90d",
    }), on="sector", how="left")

    sector_df = sector_df.sort_values("Avg_Score", ascending=False)
    sector_df["Avg_Score"]    = sector_df["Avg_Score"].apply(lambda x: f"{x:.0f}/100")
    sector_df["Avg_90d"]      = sector_df["Avg_90d"].apply(lambda x: f"{x:+.1f}%")
    sector_df["Top_90d"]      = sector_df["Top_90d"].apply(lambda x: f"{x:+.1f}%")
    sector_df["Top_Composite"]= sector_df["Top_Composite"].apply(lambda x: f"{x:.0f}/100")
    sector_df.columns         = ["Sector","# Stocks","Avg Score","Avg 90d","Best Score","Top Pick","Top Composite","Top 90d Fcst"]

    st.dataframe(sector_df, use_container_width=True, hide_index=True)

    # ── Section 3: Top Picks ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏆 Top Picks — Highest Engine Confidence")
    st.caption("Stocks with composite score above 70 and positive 90d forecast, ranked by confidence")

    top = df[(df["composite"] >= 68) & (df["forecast_90d"] > 0)].sort_values("composite", ascending=False).head(12)
    if not top.empty:
        top_rows = []
        for _, r in top.iterrows():
            comp = r["composite"]
            color_tag = "🟢" if comp >= 72 else "🟡"
            top_rows.append({
                "":           color_tag,
                "Stock":      r["name"],
                "Sector":     r["sector"],
                "Price (SAR)":f"{r['price']:.2f}",
                "RSI":        f"{r['rsi']:.1f}",
                "Composite":  f"{comp:.0f}/100",
                "90d Forecast":f"{r['forecast_90d']:+.1f}%",
                "90d Target": f"{r['target_90d']:.2f} SAR",
                "Confidence": r["conf_label"],
            })
        st.dataframe(pd.DataFrame(top_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No stocks currently meet the top picks criteria.")

    # ── Section 4: Extreme RSI Alerts ────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📉 Extreme Oversold (RSI < 25)")
        st.caption("Historically a mean-reversion buying opportunity")
        oversold = df[df["rsi"] < 25].sort_values("rsi")
        if not oversold.empty:
            for _, r in oversold.iterrows():
                st.markdown(f"""<div class="forecast-box">
                <b>{r['emoji']} {r['name']} ({r['symbol']})</b> — {r['sector']}<br>
                Price: <b>{r['price']:.2f} SAR</b> | RSI: <b style="color:#dc3545">{r['rsi']:.1f}</b> |
                Score: <b>{r['composite']:.0f}/100</b> | 90d: <b>{r['forecast_90d']:+.1f}%</b>
                </div>""", unsafe_allow_html=True)
        else:
            st.success("No stocks in extreme oversold territory right now.")

    with col2:
        st.markdown("### 📈 Extreme Overbought (RSI > 78)")
        st.caption("Historically a mean-reversion caution signal")
        overbought = df[df["rsi"] > 78].sort_values("rsi", ascending=False)
        if not overbought.empty:
            for _, r in overbought.iterrows():
                st.markdown(f"""<div class="warning-box">
                <b>{r['emoji']} {r['name']} ({r['symbol']})</b> — {r['sector']}<br>
                Price: <b>{r['price']:.2f} SAR</b> | RSI: <b style="color:#fd7e14">{r['rsi']:.1f}</b> |
                Score: <b>{r['composite']:.0f}/100</b> | 90d: <b>{r['forecast_90d']:+.1f}%</b>
                </div>""", unsafe_allow_html=True)
        else:
            st.success("No stocks in extreme overbought territory right now.")

    # ── Section 5: Weakest / Avoid ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚠️ Weakest Signals — Engine Has Low Confidence")
    st.caption("Stocks where current signals are mixed or unfavourable")

    weak = df[(df["composite"] < 55) | (df["forecast_90d"] < 0)].sort_values("composite").head(10)
    if not weak.empty:
        weak_rows = []
        for _, r in weak.iterrows():
            weak_rows.append({
                "Stock":       r["name"],
                "Sector":      r["sector"],
                "Price (SAR)": f"{r['price']:.2f}",
                "RSI":         f"{r['rsi']:.1f}",
                "Composite":   f"{r['composite']:.0f}/100",
                "90d Forecast":f"{r['forecast_90d']:+.1f}%",
                "Confidence":  r["conf_label"],
            })
        st.dataframe(pd.DataFrame(weak_rows), use_container_width=True, hide_index=True)

    # ── Section 6: Full Stock Table ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 All Stocks — Full Ranking")
    st.caption("Ranked by composite score. Click column headers to sort.")

    sector_filter = st.selectbox("Filter by sector", ["All"] + sorted(df["sector"].unique()), key="summary_sector")
    display_df = df if sector_filter == "All" else df[df["sector"] == sector_filter]
    display_df = display_df.sort_values("composite", ascending=False)

    table_rows = []
    for _, r in display_df.iterrows():
        comp = r["composite"]
        icon = "🟢" if comp >= 68 else ("🟡" if comp >= 55 else "🔴")
        table_rows.append({
            "":            icon,
            "Stock":       r["name"],
            "Sector":      r["sector"],
            "Price":       f"{r['price']:.2f}",
            "RSI":         f"{r['rsi']:.1f}",
            "Env":         f"{r['env_score']:.0f}",
            "Tech":        f"{r['tech_score']:.0f}",
            "Fund":        f"{r['fund_score']:.0f}",
            "Composite":   f"{comp:.0f}/100",
            "90d Fcst":    f"{r['forecast_90d']:+.1f}%",
            "Confidence":  r["conf_label"],
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(table_rows)} stocks")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    selected_sym = render_sidebar()

    st.markdown(f"""
    <div class="main-header">📊 Tadawul Stock Memory Engine</div>
    <div class="sub-header">Private AI research system — {STOCKS[selected_sym]['emoji']} {STOCKS[selected_sym]['name']} selected</div>
    """, unsafe_allow_html=True)

    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Summary",
        "💬 AI Chat",
        "🎯 Forecast",
        "🧠 Personality",
        "📡 Signals",
        "🧪 Hypotheses",
        "📈 Backtest",
    ])

    with tab0:
        tab_summary()
    with tab1:
        tab_chat(selected_sym)
    with tab2:
        tab_forecast(selected_sym)
    with tab3:
        tab_personality(selected_sym)
    with tab4:
        tab_signals(selected_sym)
    with tab5:
        tab_hypotheses()
    with tab6:
        tab_backtest()


if __name__ == "__main__":
    main()
