import numpy as np
import pandas as pd

def rsi(px: pd.Series, period: int = 14) -> pd.Series:
    """
    Computes a standard RSI using exponential moving averages of gains/losses.
    """
    delta = px.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - 100 / (1 + rs)

def make_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs technical features per ticker using only past information:
    - Momentum over 5/20/60 days
    - Realized volatility over 20 days
    - Price / MA20 ratio
    - RSI(14)
    Returns a wide DataFrame with feature columns for each ticker.
    """
    feats = []
    for t in prices.columns:
        px = prices[t]
        df = pd.DataFrame(index=prices.index)
        for w in (5, 20, 60):
            df[f"mom_{t}_{w}"] = px.pct_change(w)
        df[f"vol_{t}_20"] = px.pct_change().rolling(20).std()
        df[f"ma_ratio_{t}_20"] = px / px.rolling(20).mean()
        df[f"rsi_{t}_14"] = rsi(px, 14)
        feats.append(df)
    F = pd.concat(feats, axis=1)
    return F

def make_targets_excess(prices: pd.DataFrame, bench: pd.Series, horizon: int) -> pd.DataFrame:
    """
    Creates the regression targets: future log-return (asset) minus future log-return (benchmark)
    aggregated over the next 'horizon' days (t+1 ... t+h).
    Using log-returns ensures additivity over the window.
    """
    lr_assets = np.log1p(prices.pct_change())
    lr_bench  = np.log1p(bench.pct_change())
    y_assets = lr_assets.shift(-1).rolling(horizon).sum()
    y_bench  = lr_bench.shift(-1).rolling(horizon).sum()
    Y = y_assets.subtract(y_bench, axis=0)  # DataFrame with one column per ticker
    return Y