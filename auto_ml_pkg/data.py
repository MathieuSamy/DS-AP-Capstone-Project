# src/data.py
import os
import time
from typing import Iterable
import pandas as pd
import yfinance as yf
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Directories for cached and raw CSVs
DATA_DIR = "auto_ml/DS-AP-Capstone-Project/data/cache"
RAW_DIR = "auto_ml/DS-AP-Capstone-Project/data/raw"  # optional: for manually downloaded Yahoo CSVs
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

def _cache_path(ticker: str) -> str:
    """Return path for cached CSV."""
    return os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}.csv")

def _raw_path(ticker: str) -> str:
    """Return path for manually downloaded CSV (Yahoo export)."""
    return os.path.join(RAW_DIR, f"{ticker.replace('.', '_')}.csv")

def _normalize_price_df(df: pd.DataFrame, ticker: str) -> pd.Series | None:
    """
    Clean and extract a single 'Close' price series from a Yahoo DataFrame.
    Handles both single-index and multi-index (ticker, field) structures.
    """
    if df is None or len(df) == 0:
        return None

    # Ensure index is sorted and timezone-free
    df = df.sort_index()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Handle MultiIndex columns (e.g., (TSLA, Close))
    if isinstance(df.columns, pd.MultiIndex):
        # Try to extract by level: ('Close',) or ('Adj Close',)
        for candidate in ["Close", "Adj Close"]:
            try:
                sub = df.xs(key=candidate, level=1, axis=1)
                if ticker in sub.columns:
                    s = sub[ticker].rename(ticker).dropna()
                    s.index.name = "Date"
                    return s
            except Exception:
                pass
        # Alternative attempt: xs by ticker then 'Close'
        try:
            sub = df.xs(key=ticker, level=0, axis=1)
            for col in ["Close", "Adj Close"]:
                if col in sub.columns:
                    s = sub[col].rename(ticker).dropna()
                    s.index.name = "Date"
                    return s
        except Exception:
            pass
        return None

    # Handle simple columns (non-multi-index)
    for col in ["Close", "Adj Close"]:
        if col in df.columns:
            s = df[col].rename(ticker).dropna()
            s.index.name = "Date"
            return s

    return None

def _save_cache(ticker: str, series: pd.Series) -> None:
    """Save a clean price series (Date, Close) to cache."""
    out = series.to_frame(name="Close")
    out.index.name = "Date"
    out.to_csv(_cache_path(ticker), index=True)

def _load_cache(ticker: str, start: str, end: str) -> pd.Series | None:
    """Try to load cached data for the ticker within date range."""
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return None
    try:
        s = pd.read_csv(path, parse_dates=["Date"], index_col="Date")["Close"].rename(ticker)
        s = s.loc[(s.index >= pd.to_datetime(start)) & (s.index <= pd.to_datetime(end))]
        if s.empty:
            return None
        return s.sort_index()
    except Exception:
        return None

def _load_raw_csv_if_available(ticker: str, start: str, end: str) -> pd.Series | None:
    """
    Try to load data from a manually downloaded CSV (from Yahoo),
    located at data/raw/<TICKER>.csv. Accepts either 'Close' or 'Adj Close'.
    """
    path = _raw_path(ticker)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
        col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
        if col is None or df.empty:
            return None
        s = df[col].rename(ticker).dropna()
        s = s.loc[(s.index >= pd.to_datetime(start)) & (s.index <= pd.to_datetime(end))]
        s.index.name = "Date"
        if s.empty:
            return None
        # Save to cache for future offline use
        _save_cache(ticker, s)
        print(f"[INFO] Loaded '{ticker}' from raw CSV and cached it.")
        return s
    except Exception as e:
        print(f"[WARN] Failed to parse raw CSV for '{ticker}': {e}")
        return None

def _download_one(ticker: str, start: str, end: str, retries: int = 4, base_pause: float = 1.5) -> pd.Series | None:
    """
    Robust download strategy:
      1. Try yf.download()
      2. Try yf.Ticker().history()
      3. Fallback to cached data
      4. Fallback to manually downloaded CSV in data/raw
    """
    last_exc = None

    # (1) Try yf.download with exponential backoff
    for i in range(retries):
        try:
            df = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by="ticker",
                timeout=60,
            )
            s = _normalize_price_df(df, ticker)
            if s is not None and len(s) > 0:
                _save_cache(ticker, s)
                return s
            else:
                last_exc = RuntimeError("Empty or unrecognized DataFrame from yf.download")
        except Exception as e:
            last_exc = e
        time.sleep(base_pause * (2 ** i))  # exponential backoff

    # (2) Try yf.Ticker().history
    for i in range(retries):
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(start=start, end=end, interval="1d", auto_adjust=True, actions=False, timeout=60)
            s = _normalize_price_df(df, ticker)
            if s is not None and len(s) > 0:
                _save_cache(ticker, s)
                return s
            else:
                last_exc = RuntimeError("Empty or unrecognized DataFrame from Ticker.history")
        except Exception as e:
            last_exc = e
        time.sleep(base_pause * (2 ** i))

    # (3) Load from cache if available
    cached = _load_cache(ticker, start, end)
    if cached is not None:
        print(f"[INFO] Using cached data for '{ticker}'.")
        return cached

    # (4) Load manually downloaded CSV if available
    raw = _load_raw_csv_if_available(ticker, start, end)
    if raw is not None:
        return raw

    # Log final failure
    if last_exc:
        print(f"[WARN] Failed to fetch '{ticker}': {last_exc}. No cache or raw CSV available.")
    else:
        print(f"[WARN] Failed to fetch '{ticker}' and no cache/raw CSV present.")
    return None

def fetch_prices(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily close prices for a list of tickers.
    Falls back to cache or raw CSVs if network fails.
    """
    series = []
    for t in tickers:
        s = _download_one(t, start, end)
        if s is not None and len(s) > 0:
            series.append(s)
        else:
            print(f"[SKIP] Missing data for '{t}'")

    if not series:
        raise RuntimeError(
            "No price data available (network blocked and no cache). "
            "Upload CSVs to data/raw/<TICKER>.csv or data/cache/<TICKER>.csv with columns Date,Close (or Adj Close)."
        )

    prices = pd.concat(series, axis=1).sort_index()
    return prices

def fetch_benchmark(symbol: str, start: str, end: str, fallback_from: pd.DataFrame | None = None) -> pd.Series:
    """
    Fetch a benchmark index (e.g., S&P 500).
    If unavailable, creates an equal-weight benchmark from provided tickers.
    """
    s = _download_one(symbol, start, end)
    if s is not None and len(s) > 0:
        return s

    if fallback_from is None or fallback_from.empty:
        raise RuntimeError(f"Benchmark '{symbol}' unavailable and no fallback data provided.")

    # Equal-weight synthetic benchmark
    ew = fallback_from.pct_change().mean(axis=1).pipe(lambda r: (1 + r).cumprod())
    ew = ew / ew.iloc[0] * 100.0
    ew.name = "EQUAL_WEIGHT_BENCH"
    print("[INFO] Using equal-weight benchmark fallback.")
    return ew