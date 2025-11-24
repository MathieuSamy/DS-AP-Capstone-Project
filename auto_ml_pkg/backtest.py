import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def equity_curve(                               # to backtest a top-k long strategy based on predicted excess returns
    pred_scores: pd.DataFrame,                  # predicted excess returns
    future_excess: pd.DataFrame,                # realized excess returns
    top_k: int = 5,                             # number of top tickers to hold 
    rebalance_every: int = 5,                   # rebalance frequency (in days)
    transaction_cost_bps: float = 0.0,          # transaction cost in basis points
) -> pd.Series:
    """
    Backtest a top-k long strategy based on predicted EXCESS returns.

    - Rebalances the portfolio every `rebalance_every` dates.
    - At each rebalance date, invests equally in the top_k tickers
      with the highest predicted excess return.
    - Uses realized excess returns in `future_excess` at those dates.
    - Subtracts transaction costs proportional to turnover.

    Returns
    -------
    pd.Series
        Cumulative growth of the strategy (×), in excess of the benchmark.
    """

    # Common dates between predictions and realized excess
    dates = sorted(set(pred_scores.index).intersection(future_excess.index))
    if not dates:
        return pd.Series(dtype=float, name="equity_excess")

    rets = []            # list of (date, net excess return) tuples
    prev_weights = None  # portfolio weights at previous rebalance


    # DEBUG ACCUMULATORS
    turnovers = []     # list of turnover values per rebalance
    costs = []         # list of transaction costs per rebalance

    for i in range(0, len(dates), rebalance_every): # step through rebalance dates
        dt = dates[i]                               # current rebalance date

        # Drop NaNs
        p = pred_scores.loc[dt].dropna()            # predicted scores at date dt
        y = future_excess.loc[dt].dropna()          # realized excess returns at date dt

        # Intersection of tickers
        common = p.index.intersection(y.index)      # tickers with both predictions and realized returns
        if len(common) < top_k:
            continue

        # 1) Select top-k tickers by predicted score
        picks = p[common].sort_values(ascending=False).head(top_k).index

        # 2) Equal weights
        weights = pd.Series(0.0, index=common)      # initialize weights       
        weights[picks] = 1.0 / top_k                # equal weight for selected tickers

        # 3) Turnover vs previous weights
        if prev_weights is None:                                        # first rebalance
            turnover = 0.0
        else:
            w_prev = prev_weights.reindex(weights.index).fillna(0.0)    # align previous weights
            turnover = (weights - w_prev).abs().sum() / 2.0             # turnover calculation

        # 4) Realized excess return of the portfolio
        port_excess = (y[picks] * weights[picks]).sum()                 # portfolio excess return

        # 5) Transaction cost (bps → fraction)
        cost = transaction_cost_bps / 10000.0 * turnover                # transaction cost
        net_excess = port_excess - cost                                 # net excess return after cost

        # store for debug
        turnovers.append(turnover)
        costs.append(cost)

        rets.append((dt, net_excess))
        prev_weights = weights

    # Time series of (net) excess returns
    s = pd.Series({d: r for d, r in rets}).sort_index()
    s.name = "excess_return_net"

    equity = (1 + s).cumprod()
    equity.name = "equity_excess"

    # ==== DEBUG PRINT ====
    if transaction_cost_bps != 0 and len(turnovers) > 0:
        avg_turnover = float(np.mean(turnovers))
        avg_cost = float(np.mean(costs))
        total_cost = float(np.sum(costs))
        print(
            f"[COST DEBUG] avg turnover: {avg_turnover:.3f}, "
            f"avg cost per rebalance: {avg_cost:.5f}, "
            f"total cost over period: {total_cost:.4f}"
        )
        
    return equity