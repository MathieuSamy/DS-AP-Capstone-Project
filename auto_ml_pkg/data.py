# src/data.py
import os
import time
from typing import Iterable, Tuple
import pandas as pd
import yfinance as yf

DATA_DIR = "data/cache"
os.makedirs(DATA_DIR, exist_ok=True)

def _cache_path(ticker: str) -> str:
    return os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}.csv")

def _save_cache(ticker: str, df: pd.DataFrame) -> None:
    # Save only Date + Close for simplicity and portability
    out = df[["Close"]].copy()
    out.to_csv(_cache_path(ticker), index=True)

def _load_cache(ticker: str, start: str, end: str) -> pd.Series | None:
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return None
    s = pd.read_csv(path, parse_dates=["Date"], index_col="Date")["Close"].rename(ticker)
    s = s.loc[(s.index >= start) & (s.index <= end)]
    if s.empty:
        return None
    return s

def _download_one(ticker: str, start: str, end: str, retries: int = 3, pause: float = 1.5) -> pd.Series | None:
    last_exc = None
    for _ in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                             progress=False, threads=False, group_by="ticker", timeout=60)
            if isinstance(df, pd.DataFrame) and "Close" in df and len(df) > 0:
                s = df["Close"].rename(ticker).sort_index()
                # Cache for offline reuse
                _save_cache(ticker, df)
                return s
        except Exception as e:
            last_exc = e
        time.sleep(pause)
    # Final attempt: load from cache if exists
    cached = _load_cache(ticker, start, end)
    if cached is None and last_exc:
        print(f"[WARN] Failed to fetch '{ticker}': {last_exc}. No cache available.")
    elif cached is None:
        print(f"[WARN] Failed to fetch '{ticker}' and no cache present.")
    return cached  # may be None

def fetch_prices(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
    series = []
    for t in tickers:
        s = _download_one(t, start, end)
        if s is not None:
            series.append(s)
        else:
            print(f"[SKIP] Missing data for '{t}'")
    if not series:
        raise RuntimeError("No price data available (network blocked and no cache). "
                           "Upload CSVs to data/cache/<TICKER>.csv with columns Date,Close.")
    prices = pd.concat(series, axis=1).sort_index()
    return prices

def fetch_benchmark(symbol: str, start: str, end: str, fallback_from: pd.DataFrame | None = None) -> pd.Series:
    """
    Attempts to download benchmark. If not available, builds an equal-weight
    benchmark from the provided price DataFrame (fallback_from).
    """
    s = _download_one(symbol, start, end)
    if s is not None:
        return s
    if fallback_from is None or fallback_from.empty:
        raise RuntimeError(f"Benchmark '{symbol}' unavailable and no fallback universe provided.")
    ew = fallback_from.pct_change().mean(axis=1).pipe(lambda r: (1 + r).cumprod())
    # Normalize to look like a 'price' series
    ew = ew / ew.iloc[0] * 100.0
    ew.name = "EQUAL_WEIGHT_BENCH"
    print("[INFO] Using equal-weight benchmark fallback.")
    return ew