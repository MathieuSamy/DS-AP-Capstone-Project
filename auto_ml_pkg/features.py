import numpy as np
import pandas as pd

def rsi(px: pd.Series, period: int = 14) -> pd.Series: # to compute RSI indicator for a price series
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


def make_features(prices: pd.DataFrame) -> pd.DataFrame: # to create technical features from price data
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

        # Momentum features
        for w in (5, 20, 60):
            df[f"mom_{t}_{w}"] = px.pct_change(w, fill_method=None)

        # Realized volatility (using daily pct_change)
        df[f"vol_{t}_20"] = px.pct_change(fill_method=None).rolling(20).std()

        # Price / moving average ratio
        df[f"ma_ratio_{t}_20"] = px / px.rolling(20).mean()

        # RSI(14)
        df[f"rsi_{t}_14"] = rsi(px, 14)

        # Replace inf / -inf by NaN and drop rows with all-NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        feats.append(df)

    # Combine all tickers’ features
    F = pd.concat(feats, axis=1)

    # Drop rows where everything is NaN (e.g. first few days)
    F.dropna(how="all", inplace=True)
    return F


def make_targets_excess(prices: pd.DataFrame, bench: pd.Series, horizon: int) -> pd.DataFrame:
    """
    Creates the regression targets: future log-return (asset) minus future log-return (benchmark)
    aggregated over the next 'horizon' days (t+1 ... t+h).
    Using log-returns ensures additivity over the window.
    """
    # Compute daily log-returns
    lr_assets = np.log1p(prices.pct_change(fill_method=None))
    lr_bench = np.log1p(bench.pct_change(fill_method=None))

    # Sum of future log-returns over next 'horizon' days
    y_assets = lr_assets.shift(-1).rolling(horizon, min_periods=1).sum()
    y_bench = lr_bench.shift(-1).rolling(horizon, min_periods=1).sum()

    # Excess return = asset - benchmark
    Y = y_assets.subtract(y_bench, axis=0)

    # Align with prices (important!)
    Y = Y.reindex(prices.index)

    # Clean infinities, keep NaN (they’ll be dropped later per-ticker)
    Y.replace([np.inf, -np.inf], np.nan, inplace=True)

    return Y

