import pandas as pd

def equity_curve(pred_scores: pd.DataFrame, future_excess: pd.DataFrame, top_k: int = 5) -> pd.Series:
    """
    Rebalances each date across the top_k tickers by predicted score (equal-weight).
    Uses realized future excess returns as the portfolio return at each date.
    Returns cumulative growth series (×).
    """
    rets = []
    dates = sorted(set(pred_scores.index).intersection(future_excess.index))
    for dt in dates:
        p = pred_scores.loc[dt].dropna()
        y = future_excess.loc[dt].dropna()
        common = p.index.intersection(y.index)
        if len(common) < top_k:
            continue
        picks = p[common].sort_values(ascending=False).head(top_k).index
        rets.append((dt, y[picks].mean()))
    s = pd.Series({d: r for d, r in rets}).sort_index()
    return (1 + s).cumprod()