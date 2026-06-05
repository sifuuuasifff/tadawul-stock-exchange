"""
TADAWUL STOCK MEMORY ENGINE — HOME (Summary Dashboard)
======================================================
Page 1 of 2: Market overview, sector scorecard, top picks, RSI alerts.
Page 2 (Stock Analysis): Individual stock AI chat, forecast, signals, etc.

Run:  streamlit run portal.py
"""

import sys, json, os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd

from config.settings import MEMORY_DIR, REPORTS_DIR, ANTHROPIC_API_KEY
os.environ.setdefault("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)

st.set_page_config(
    page_title="Tadawul Stock Memory Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header  { font-size:1.8rem; font-weight:700; color:#1a1a2e; margin-bottom:0.2rem; }
    .sub-header   { font-size:0.95rem; color:#666; margin-bottom:1.5rem; }
    .forecast-box { background:#f0f7ff; border-radius:10px; padding:1.2rem;
                    margin:0.5rem 0; border:1px solid #cce0ff; }
    .warning-box  { background:#fff8e1; border-radius:8px; padding:0.8rem;
                    border-left:3px solid #ffc107; }
</style>
""", unsafe_allow_html=True)

# ── Stock registry (single source of truth — imported by both pages) ──────────
STOCKS = {
    # Banking
    "1120": {"name": "Al Rajhi Bank",             "sector": "Banking",           "emoji": "🏦"},
    "1180": {"name": "Saudi National Bank (SNB)",  "sector": "Banking",           "emoji": "🏦"},
    "1010": {"name": "Riyad Bank",                 "sector": "Banking",           "emoji": "🏦"},
    "1060": {"name": "Saudi British Bank (SABB)",  "sector": "Banking",           "emoji": "🏦"},
    "1080": {"name": "Arab National Bank",         "sector": "Banking",           "emoji": "🏦"},
    "1050": {"name": "Banque Saudi Fransi",        "sector": "Banking",           "emoji": "🏦"},
    "1150": {"name": "Alinma Bank",                "sector": "Banking",           "emoji": "🏦"},
    "1140": {"name": "Bank Albilad",               "sector": "Banking",           "emoji": "🏦"},
    # Energy & Large Caps
    "2222": {"name": "Saudi Aramco",               "sector": "Energy",            "emoji": "🛢️"},
    "2010": {"name": "SABIC",                      "sector": "Petrochemicals",    "emoji": "⚗️"},
    # Petrochemicals
    "2020": {"name": "SABIC Agri-Nutrients",       "sector": "Petrochemicals",    "emoji": "⚗️"},
    "2290": {"name": "Yanbu National Petro",       "sector": "Petrochemicals",    "emoji": "⚗️"},
    "2350": {"name": "Saudi Kayan Petrochem",      "sector": "Petrochemicals",    "emoji": "⚗️"},
    "2310": {"name": "Sipchem",                    "sector": "Petrochemicals",    "emoji": "⚗️"},
    "2380": {"name": "Petro Rabigh",               "sector": "Petrochemicals",    "emoji": "⚗️"},
    "2210": {"name": "Nama Chemicals",             "sector": "Petrochemicals",    "emoji": "⚗️"},
    "2060": {"name": "National Industrialization", "sector": "Petrochemicals",    "emoji": "⚗️"},
    "2030": {"name": "Advanced Petrochem",         "sector": "Petrochemicals",    "emoji": "⚗️"},
    "2250": {"name": "Saudi Industrial Investment","sector": "Petrochemicals",    "emoji": "⚗️"},
    # Telecom
    "7010": {"name": "STC",                        "sector": "Telecom",           "emoji": "📡"},
    "7020": {"name": "Mobily",                     "sector": "Telecom",           "emoji": "📡"},
    "7030": {"name": "Zain Saudi",                 "sector": "Telecom",           "emoji": "📡"},
    "7040": {"name": "Etihad Atheeb",              "sector": "Telecom",           "emoji": "📡"},
    "7200": {"name": "Solutions by STC",           "sector": "Telecom",           "emoji": "📡"},
    # Mining & Chemicals
    "1211": {"name": "Maaden",                     "sector": "Mining",            "emoji": "⛏️"},
    "2230": {"name": "Saudi Chemical",             "sector": "Chemicals",         "emoji": "🧪"},
    # Retail
    "4190": {"name": "Jarir Bookstore",            "sector": "Retail",            "emoji": "🛒"},
    "4012": {"name": "Extra (United Electronics)", "sector": "Retail",            "emoji": "🛒"},
    "4163": {"name": "Nahdi Medical",              "sector": "Retail",            "emoji": "🛒"},
    "4161": {"name": "BinDawood Holding",          "sector": "Retail",            "emoji": "🛒"},
    "4001": {"name": "Al Othaim Markets",          "sector": "Retail",            "emoji": "🛒"},
    "4082": {"name": "SACO",                       "sector": "Retail",            "emoji": "🛒"},
    "6004": {"name": "Fawaz Abdulaziz Alhokair",  "sector": "Retail",            "emoji": "🛒"},
    "4003": {"name": "Astra Industrial",           "sector": "Retail",            "emoji": "🛒"},
    # Real Estate
    "4300": {"name": "Dar Al Arkan",               "sector": "Real Estate",       "emoji": "🏢"},
    "4220": {"name": "Emaar EC",                   "sector": "Real Estate",       "emoji": "🏢"},
    "4322": {"name": "Retal Urban Dev",            "sector": "Real Estate",       "emoji": "🏢"},
    "4323": {"name": "Roshn Real Estate",          "sector": "Real Estate",       "emoji": "🏢"},
    "4020": {"name": "Saudi Real Estate",          "sector": "Real Estate",       "emoji": "🏢"},
    "4250": {"name": "Jabal Omar Dev",             "sector": "Real Estate",       "emoji": "🏢"},
    "4100": {"name": "Makkah Construction",        "sector": "Real Estate",       "emoji": "🏢"},
    "4090": {"name": "Arriyadh Development",       "sector": "Real Estate",       "emoji": "🏢"},
    # Healthcare
    "4013": {"name": "Dr Sulaiman Al Habib",      "sector": "Healthcare",        "emoji": "🏥"},
    "4002": {"name": "Mouwasat Medical",           "sector": "Healthcare",        "emoji": "🏥"},
    "4004": {"name": "Dallah Healthcare",          "sector": "Healthcare",        "emoji": "🏥"},
    "4345": {"name": "Saudi German Health",        "sector": "Healthcare",        "emoji": "🏥"},
    "4007": {"name": "Al Hammadi",                 "sector": "Healthcare",        "emoji": "🏥"},
    "4005": {"name": "National Medical Care",      "sector": "Healthcare",        "emoji": "🏥"},
    "4006": {"name": "Specialized Medical",        "sector": "Healthcare",        "emoji": "🏥"},
    "2070": {"name": "Saudi Pharmaceutical",       "sector": "Healthcare",        "emoji": "🏥"},
    "4015": {"name": "Jamjoom Pharma",             "sector": "Pharmaceuticals",   "emoji": "💊"},
    # Food & Beverages
    "2280": {"name": "Almarai",                    "sector": "Food & Beverages",  "emoji": "🥛"},
    "2270": {"name": "Savola Group",               "sector": "Food & Beverages",  "emoji": "🥛"},
    "2050": {"name": "Saudi Arabia Fertilizers",   "sector": "Food & Beverages",  "emoji": "🥛"},
    "6020": {"name": "Halwani Brothers",           "sector": "Food & Beverages",  "emoji": "🥛"},
    "2100": {"name": "Wafrah for Industry",        "sector": "Food & Beverages",  "emoji": "🥛"},
    "6010": {"name": "NADEC",                      "sector": "Food & Beverages",  "emoji": "🥛"},
    "2200": {"name": "Arabian Food Industries",    "sector": "Food & Beverages",  "emoji": "🥛"},
    # Cement
    "3010": {"name": "Yamama Cement",              "sector": "Cement",            "emoji": "🏗️"},
    "3020": {"name": "Saudi Cement",               "sector": "Cement",            "emoji": "🏗️"},
    "3030": {"name": "Qassim Cement",              "sector": "Cement",            "emoji": "🏗️"},
    "3040": {"name": "Southern Province Cement",   "sector": "Cement",            "emoji": "🏗️"},
    "3050": {"name": "Yanbu Cement",               "sector": "Cement",            "emoji": "🏗️"},
    "3060": {"name": "City Cement",                "sector": "Cement",            "emoji": "🏗️"},
    "3080": {"name": "Tabuk Cement",               "sector": "Cement",            "emoji": "🏗️"},
    "3090": {"name": "Arabian Cement",             "sector": "Cement",            "emoji": "🏗️"},
    # Transport & Utilities
    "4140": {"name": "Saudi Ground Services",      "sector": "Transport",         "emoji": "✈️"},
    "4030": {"name": "Saudi Airlines Catering",    "sector": "Transport",         "emoji": "✈️"},
    "2080": {"name": "Saudi Electricity",          "sector": "Utilities",         "emoji": "⚡"},
    "5110": {"name": "Saudi Telecom Infra",        "sector": "Utilities",         "emoji": "⚡"},
    "4280": {"name": "Tabreed",                    "sector": "Utilities",         "emoji": "⚡"},
    "4110": {"name": "Arriyadh Dev Auth",          "sector": "Diversified",       "emoji": "📦"},
    "2180": {"name": "Fitaihi Holding",            "sector": "Diversified",       "emoji": "📦"},
    "1830": {"name": "Leejam Sports",              "sector": "Consumer Services", "emoji": "🏋️"},
}


@st.cache_data(ttl=300)
def load_all_states() -> dict:
    p = MEMORY_DIR / "all_current_states.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("stocks", {})


def build_stock_df() -> pd.DataFrame:
    states = load_all_states()
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
    return pd.DataFrame(rows)


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown("### 📊 Tadawul Stock Engine")
        st.markdown(f"*{len(STOCKS)} stocks · {len(set(v['sector'] for v in STOCKS.values()))} sectors*")
        st.markdown("---")

        # Top 5 always visible
        st.markdown("**🏆 Top Picks Right Now**")
        top5 = df.sort_values("composite", ascending=False).head(5)
        for _, r in top5.iterrows():
            c = r["composite"]
            color = "green" if c >= 70 else "orange"
            st.markdown(f"**:{color}[{r['emoji']} {r['name']}]** {c:.0f}/100 | {r['forecast_90d']:+.1f}%")

        st.markdown("---")
        st.markdown("**Navigate:**")
        st.markdown("📊 Summary — *this page*")
        st.markdown("🔍 Stock Analysis — *use sidebar above*")
        st.markdown("---")
        st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d')}")


# ── Main summary page ──────────────────────────────────────────────────────────
def main():
    df = build_stock_df()
    render_sidebar(df)

    st.markdown('<div class="main-header">📊 Tadawul Opportunity Engine</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{len(df)} stocks monitored · {len(df["sector"].unique())} sectors · Updated {datetime.now().strftime("%Y-%m-%d")}</div>', unsafe_allow_html=True)

    # ── Market Environment ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌍 Current Market Environment")
    first      = df.iloc[0] if not df.empty else {}
    regime     = first.get("rate_regime", "stable")
    vix_val    = first.get("vix", 20)
    oil_val    = first.get("oil", 80)
    rate       = first.get("repo_rate", 4.25)
    vix_label  = "🔴 High Fear" if vix_val > 30 else ("🟡 Elevated" if vix_val > 20 else "🟢 Calm")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saudi Repo Rate",   f"{rate:.2f}%", delta=regime.capitalize(),
              delta_color="inverse" if regime == "rising" else "normal")
    c2.metric("VIX (Global Risk)", f"{vix_val:.1f}", delta=vix_label, delta_color="off")
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
    sec = df.groupby("sector").agg(
        Stocks    =("symbol","count"),
        Avg_Score =("composite","mean"),
        Avg_90d   =("forecast_90d","mean"),
    ).round(1).reset_index()
    best = df.loc[df.groupby("sector")["composite"].idxmax()][["sector","name","composite","forecast_90d"]]
    sec  = sec.merge(best.rename(columns={"name":"Top Pick","composite":"Top Score","forecast_90d":"Top 90d"}), on="sector", how="left")
    sec  = sec.sort_values("Avg_Score", ascending=False)
    sec["Avg_Score"] = sec["Avg_Score"].apply(lambda x: f"{x:.0f}/100")
    sec["Avg_90d"]   = sec["Avg_90d"].apply(lambda x: f"{x:+.1f}%")
    sec["Top Score"] = sec["Top Score"].apply(lambda x: f"{x:.0f}/100")
    sec["Top 90d"]   = sec["Top 90d"].apply(lambda x: f"{x:+.1f}%")
    sec.columns      = ["Sector","# Stocks","Avg Score","Avg 90d Fcst","Top Pick","Top Score","Top 90d"]
    st.dataframe(sec, use_container_width=True, hide_index=True)

    # ── Top Picks ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏆 Top Picks — Engine Highest Confidence")
    st.caption("Composite ≥ 68 · positive 90d forecast · ranked by score")
    top = df[(df["composite"] >= 68) & (df["forecast_90d"] > 0)].sort_values("composite", ascending=False).head(12)
    if not top.empty:
        top_rows = []
        for _, r in top.iterrows():
            top_rows.append({
                "":            "🟢" if r["composite"] >= 72 else "🟡",
                "Stock":       r["name"],
                "Sector":      r["sector"],
                "Price (SAR)": f"{r['price']:.2f}",
                "RSI":         f"{r['rsi']:.1f}",
                "Composite":   f"{r['composite']:.0f}/100",
                "90d Forecast":f"{r['forecast_90d']:+.1f}%",
                "90d Target":  f"{r['target_90d']:.2f} SAR",
                "Confidence":  r["conf_label"],
            })
        st.dataframe(pd.DataFrame(top_rows), use_container_width=True, hide_index=True)
        st.caption("➡️ Click **Stock Analysis** in the sidebar to dive into any of these stocks")
    else:
        st.info("No stocks currently meet the top picks threshold.")

    # ── RSI Alerts ────────────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📉 Extreme Oversold (RSI < 25)")
        st.caption("Historically a mean-reversion buying opportunity")
        oversold = df[df["rsi"] < 25].sort_values("rsi")
        if not oversold.empty:
            for _, r in oversold.iterrows():
                st.markdown(f"""<div class="forecast-box">
                <b>{r['emoji']} {r['name']}</b> — {r['sector']}<br>
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
                <b>{r['emoji']} {r['name']}</b> — {r['sector']}<br>
                Price: <b>{r['price']:.2f} SAR</b> | RSI: <b style="color:#fd7e14">{r['rsi']:.1f}</b> |
                Score: <b>{r['composite']:.0f}/100</b> | 90d: <b>{r['forecast_90d']:+.1f}%</b>
                </div>""", unsafe_allow_html=True)
        else:
            st.success("No stocks in extreme overbought territory right now.")

    # ── Weakest ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚠️ Weakest Signals — Low Confidence")
    weak = df[(df["composite"] < 55) | (df["forecast_90d"] < 0)].sort_values("composite").head(10)
    if not weak.empty:
        weak_rows = []
        for _, r in weak.iterrows():
            weak_rows.append({
                "Stock":       r["name"], "Sector": r["sector"],
                "Price":       f"{r['price']:.2f}", "RSI": f"{r['rsi']:.1f}",
                "Composite":   f"{r['composite']:.0f}/100",
                "90d Forecast":f"{r['forecast_90d']:+.1f}%",
                "Confidence":  r["conf_label"],
            })
        st.dataframe(pd.DataFrame(weak_rows), use_container_width=True, hide_index=True)

    # ── Full Table ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 All Stocks — Full Ranking")
    sf = st.selectbox("Filter by sector", ["All"] + sorted(df["sector"].unique()), key="all_sector")
    disp = df if sf == "All" else df[df["sector"] == sf]
    disp = disp.sort_values("composite", ascending=False)
    table = []
    for _, r in disp.iterrows():
        c = r["composite"]
        table.append({
            "":           "🟢" if c >= 68 else ("🟡" if c >= 55 else "🔴"),
            "Stock":      r["name"], "Sector": r["sector"],
            "Price":      f"{r['price']:.2f}", "RSI": f"{r['rsi']:.1f}",
            "Env":        f"{r['env_score']:.0f}", "Tech": f"{r['tech_score']:.0f}",
            "Fund":       f"{r['fund_score']:.0f}", "Composite": f"{c:.0f}/100",
            "90d Fcst":   f"{r['forecast_90d']:+.1f}%", "Confidence": r["conf_label"],
        })
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(table)} stocks | 🟢 Strong  🟡 Neutral  🔴 Weak")


if __name__ == "__main__":
    main()
