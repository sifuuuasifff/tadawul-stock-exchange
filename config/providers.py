"""
Data Provider Registry — data-provider-agnostic layer.
Every data fetch goes through here. Swapping providers = change only this file.
"""

import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, date
from config.settings import (
    SAHMK_API_KEY, SAHMK_BASE_URL, SAHMK_PLAN, SAHMK_PLAN_LIMITS,
    PRIMARY_STOCK, PEER_BANKS, FRED_API_KEY,
    PRICE_HISTORY_START, FINANCIAL_HISTORY_START, MACRO_HISTORY_START,
)


# ══════════════════════════════════════════════════════════════════════════════
# SAHMK PROVIDER
# ══════════════════════════════════════════════════════════════════════════════

class SAHMKProvider:
    """Wrapper for the SAHMK API (https://app.sahmk.sa/api/v1)."""

    def __init__(self):
        if not SAHMK_API_KEY:
            raise ValueError("SAHMK_API_KEY is not set. Add it to config/settings.py or set the environment variable.")
        self.headers  = {"X-API-Key": SAHMK_API_KEY}
        self.base_url = SAHMK_BASE_URL
        self.plan     = SAHMK_PLAN
        self.limits   = SAHMK_PLAN_LIMITS[self.plan]

    def _get(self, endpoint: str, params: dict = None) -> dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = requests.get(url, headers=self.headers, params=params or {}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def is_available(self) -> bool:
        return bool(SAHMK_API_KEY)

    # ── Quote ──────────────────────────────────────────────────────────────
    def get_quote(self, symbol: str) -> dict:
        return self._get(f"/quote/{symbol}/")

    # ── Historical Price ───────────────────────────────────────────────────
    def get_historical_price(self, symbol: str, from_date: str = None,
                              to_date: str = None, interval: str = "1d") -> pd.DataFrame:
        """
        Returns OHLCV DataFrame indexed by date.
        interval: '1d' | '1w' | '1m'  (daily, weekly, monthly)
        Requires Starter plan or above.
        """
        params = {
            "from":     from_date or PRICE_HISTORY_START,
            "to":       to_date   or date.today().isoformat(),
            "interval": interval,
        }
        data = self._get(f"/historical/{symbol}/", params)
        records = data.get("data", data.get("results", []))
        df = pd.DataFrame(records)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df.columns = [c.lower() for c in df.columns]
        return df

    # ── Financial Statements ───────────────────────────────────────────────
    def get_financials(self, symbol: str, period: str = "quarterly",
                       history: str = "max", stmt_type: str = "all") -> dict:
        """
        period:    'annual' | 'quarterly'
        history:   '1y' | '3y' | '5y' | '10y' | 'max'
        stmt_type: 'income' | 'balance' | 'cashflow' | 'all'
        Quarterly requires Pro plan.
        """
        if period == "quarterly" and not self.limits["quarterly"]:
            raise PermissionError(f"Quarterly financials require Pro plan. Current plan: {self.plan}")
        params = {
            "type":    stmt_type,
            "period":  period,
            "history": history,
            "metrics": "extended",
        }
        return self._get(f"/financials/{symbol}/", params)

    # ── Ratios / Analytics ─────────────────────────────────────────────────
    def get_ratios(self, symbol: str, period: str = "quarterly",
                   history: str = "max") -> dict:
        params = {
            "period":  period,
            "history": history,
            "metrics": "extended",
        }
        return self._get(f"/analytics/ratios/{symbol}/", params)

    # ── Dividends ──────────────────────────────────────────────────────────
    def get_dividends(self, symbol: str, limit: int = 100) -> dict:
        return self._get(f"/dividends/{symbol}/", {"limit": limit})

    # ── Events ────────────────────────────────────────────────────────────
    def get_events(self, symbol: str, event_type: str = None,
                   limit: int = 200) -> dict:
        """
        event_type: 'FINANCIAL_REPORT' | 'DIVIDEND_ANNOUNCEMENT' | etc.
        Requires Pro plan.
        """
        params = {"symbol": symbol, "limit": limit}
        if event_type:
            params["type"] = event_type
        return self._get("/events/", params)

    # ── Company Info ───────────────────────────────────────────────────────
    def get_company(self, symbol: str) -> dict:
        return self._get(f"/company/{symbol}/")

    # ── Sector / Market ────────────────────────────────────────────────────
    def get_market_summary(self, index: str = "TASI") -> dict:
        return self._get("/market/summary/", {"index": index})

    def get_sectors(self, index: str = "TASI") -> dict:
        return self._get("/market/sectors/", {"index": index})

    # ── Peer Comparison ────────────────────────────────────────────────────
    def get_peer_comparison(self, symbols: list, metrics: str = "extended") -> dict:
        return self._get("/analytics/compare/", {
            "symbols": ",".join(symbols),
            "metrics": metrics,
        })

    # ── TASI Historical (via TASI index or ETF) ────────────────────────────
    def get_tasi_historical(self, from_date: str = None, to_date: str = None) -> pd.DataFrame:
        """
        Fetches TASI index historical data.
        Uses the TASI index symbol — check SAHMK docs for exact identifier.
        """
        return self.get_historical_price("TASI", from_date, to_date, interval="1d")


# ══════════════════════════════════════════════════════════════════════════════
# YAHOO FINANCE FALLBACK PROVIDER
# ══════════════════════════════════════════════════════════════════════════════

class YahooProvider:
    """Fallback provider using yfinance for price data."""

    def get_historical_price(self, yahoo_ticker: str, start: str = None,
                              end: str = None) -> pd.DataFrame:
        ticker = yf.Ticker(yahoo_ticker)
        df = ticker.history(
            start=start or PRICE_HISTORY_START,
            end=end or date.today().isoformat(),
            auto_adjust=True,
        )
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]].dropna()

    def get_dividends(self, yahoo_ticker: str) -> pd.Series:
        return yf.Ticker(yahoo_ticker).dividends


# ══════════════════════════════════════════════════════════════════════════════
# MACRO DATA PROVIDER  (free sources — FRED, Yahoo)
# ══════════════════════════════════════════════════════════════════════════════

class MacroProvider:
    """Free macro data: oil, VIX, Fed rate, Saudi repo rate proxy."""

    def get_brent_oil(self, start: str = None) -> pd.DataFrame:
        df = YahooProvider().get_historical_price("BZ=F", start or MACRO_HISTORY_START)
        return df[["close"]].rename(columns={"close": "brent_oil"})

    def get_vix(self, start: str = None) -> pd.DataFrame:
        df = YahooProvider().get_historical_price("^VIX", start or MACRO_HISTORY_START)
        return df[["close"]].rename(columns={"close": "vix"})

    def get_fed_funds_rate(self, start: str = None) -> pd.DataFrame:
        """US Federal Funds Rate via Yahoo (^IRX as proxy, or FRED if key set)."""
        df = YahooProvider().get_historical_price("^IRX", start or MACRO_HISTORY_START)
        return df[["close"]].rename(columns={"close": "fed_rate_proxy"})

    def get_saudi_repo_rate(self) -> pd.DataFrame:
        """
        Saudi Repo Rate — SAMA does not have a public API.
        We use a manually maintained CSV for known rate-change dates.
        File: data/raw/saudi_repo_rate.csv
        Columns: date, rate
        Source: SAMA website (https://www.sama.gov.sa)
        """
        import os
        path = "data/raw/saudi_repo_rate.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["date"])
            df = df.set_index("date").sort_index()
            return df
        else:
            print("WARNING: saudi_repo_rate.csv not found. Using BIS/SAMA data from Phase 0 POC.")
            return pd.DataFrame()

    def get_saibor(self) -> pd.DataFrame:
        """
        SAIBOR 3M — not available in SAHMK or free sources.
        Options:
          1. SAHMK Pro plan (check if they added it)
          2. Bloomberg / Refinitiv (paid)
          3. Manual download from SAMA website
        File: data/raw/saibor_3m.csv  (if manually obtained)
        Columns: date, saibor_3m
        """
        import os
        path = "data/raw/saibor_3m.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["date"])
            return df.set_index("date").sort_index()
        else:
            print("WARNING: saibor_3m.csv not found. Will use Saudi Repo Rate as proxy.")
            return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED DATA GATEWAY  (single entry point for all phases)
# ══════════════════════════════════════════════════════════════════════════════

class DataGateway:
    """
    All phases use this class. Never call providers directly from phases.
    This makes swapping providers a one-file change.
    """

    def __init__(self):
        self.sahmk  = SAHMKProvider() if SAHMK_API_KEY else None
        self.yahoo  = YahooProvider()
        self.macro  = MacroProvider()

    def get_price(self, symbol: str = None, start: str = None, end: str = None) -> pd.DataFrame:
        """Get daily OHLCV. Uses SAHMK if available, falls back to Yahoo."""
        sym = symbol or PRIMARY_STOCK["symbol"]
        if self.sahmk:
            return self.sahmk.get_historical_price(sym, start, end)
        else:
            yahoo_sym = PRIMARY_STOCK["yahoo"] if sym == PRIMARY_STOCK["symbol"] else f"{sym}.SR"
            return self.yahoo.get_historical_price(yahoo_sym, start, end)

    def get_tasi(self, start: str = None) -> pd.DataFrame:
        if self.sahmk:
            return self.sahmk.get_tasi_historical(start)
        else:
            return self.yahoo.get_historical_price("^TASI.SR" , start)

    def get_financials(self, symbol: str = None, period: str = "quarterly") -> dict:
        sym = symbol or PRIMARY_STOCK["symbol"]
        if self.sahmk:
            return self.sahmk.get_financials(sym, period=period, history="max")
        else:
            raise RuntimeError("Quarterly financials require SAHMK Pro plan. No free fallback available.")

    def get_dividends(self, symbol: str = None) -> dict:
        sym = symbol or PRIMARY_STOCK["symbol"]
        if self.sahmk:
            return self.sahmk.get_dividends(sym)
        else:
            yahoo_sym = PRIMARY_STOCK["yahoo"] if sym == PRIMARY_STOCK["symbol"] else f"{sym}.SR"
            return {"history": self.yahoo.get_dividends(yahoo_sym).to_dict()}

    def get_ratios(self, symbol: str = None) -> dict:
        sym = symbol or PRIMARY_STOCK["symbol"]
        if self.sahmk:
            return self.sahmk.get_ratios(sym)
        raise RuntimeError("Ratios require SAHMK.")

    def get_events(self, symbol: str = None) -> dict:
        sym = symbol or PRIMARY_STOCK["symbol"]
        if self.sahmk:
            return self.sahmk.get_events(sym)
        raise RuntimeError("Events require SAHMK Pro plan.")

    def get_macro(self) -> dict:
        return {
            "oil":         self.macro.get_brent_oil(),
            "vix":         self.macro.get_vix(),
            "fed_rate":    self.macro.get_fed_funds_rate(),
            "saudi_repo":  self.macro.get_saudi_repo_rate(),
            "saibor":      self.macro.get_saibor(),
        }

    def get_peers(self) -> dict:
        """Get price data for all peer banks."""
        peers = {}
        for sym, name in PEER_BANKS.items():
            try:
                if self.sahmk:
                    peers[sym] = self.sahmk.get_historical_price(sym)
                else:
                    peers[sym] = self.yahoo.get_historical_price(f"{sym}.SR")
                print(f"  ✓ {name} ({sym})")
            except Exception as e:
                print(f"  ✗ {name} ({sym}): {e}")
        return peers
