import os
import pandas as pd
from tqdm import tqdm

from auto_ml_pkg.config import Config
from auto_ml_pkg.data import fetch_prices, fetch_benchmark
from auto_ml_pkg.features import make_features, make_targets_excess
from auto_ml_pkg.models import make_ridge
from auto_ml_pkg.evaluate import regression_report, information_coefficient
from auto_ml_pkg.backtest import equity_curve
from auto_ml_pkg.viz import plot_equity, scatter_pred_vs_true

def main():
    cfg = Config()

    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/artifacts", exist_ok=True)

    # 1) Data
    prices = fetch_prices(cfg.tickers, cfg.train_start, cfg.test_end)
    bench  = fetch_benchmark(cfg.benchmark, cfg.train_start, cfg.test_end, fallback_from=prices)
    bench  = bench.reindex(prices.index).ffill().bfill()
    prices = prices.dropna(how="all")
    bench  = bench.reindex(prices.index).ffill().bfill()

    # 2) Features (X) and Targets (Y)
    X = make_features(prices)
    Y = make_targets_excess(prices, bench, cfg.horizon_days)

    # 3) Time split
    train_mask = (X.index >= cfg.train_start) & (X.index < cfg.test_start)
    test_mask  = (X.index >= cfg.test_start) & (X.index <= cfg.test_end)

    # 4) Fit per ticker and predict on test
    preds, reals = {}, {}
    for t in tqdm(cfg.tickers, desc="Per-ticker fit"):
        cols = [c for c in X.columns if any(
            c.startswith(p) for p in (f"mom_{t}_", f"vol_{t}_", f"ma_ratio_{t}_", f"rsi_{t}_")
        )]
        if t not in Y.columns or not cols:
            continue

        Xtr, Xte = X.loc[train_mask, cols], X.loc[test_mask, cols]
        ytr, yte = Y.loc[train_mask, t],    Y.loc[test_mask, t]

        tr = pd.concat([Xtr, ytr], axis=1).dropna()
        te = pd.concat([Xte, yte], axis=1).dropna()
        if len(tr) < 100 or len(te) < 20:
            continue

        model = make_ridge(alpha=2.0)
        model.fit(tr[cols], tr[t])
        p = pd.Series(model.predict(te[cols]), index=te.index, name=t)

        preds[t] = p
        reals[t] = te[t]

    P = pd.DataFrame(preds).sort_index()
    Yf = pd.DataFrame(reals).sort_index()
    if P.empty:
        raise RuntimeError("No predictions created. Check tickers and data availability.")

    # 5) Metrics (stacked across tickers/dates)
    stack_true = Yf.stack()
    stack_pred = P.stack()
    reg = regression_report(stack_true, stack_pred)
    ic  = information_coefficient(stack_true, stack_pred)

    print("\n==== REGRESSION METRICS ====")
    for k, v in reg.items():
        print(f"{k}: {v:.6f}")
    print("IC:", ic)

    # 6) Backtest and plots
    ec = equity_curve(P, Yf, top_k=cfg.top_k)
    ec.to_csv("outputs/artifacts/equity_curve.csv")
    P.to_csv("outputs/artifacts/predictions.csv")
    Yf.to_csv("outputs/artifacts/realized_excess.csv")

    plot_equity(ec, "outputs/figures/equity_curve.png",
                title=f"Top-{cfg.top_k} long — ridge — h={cfg.horizon_days}")
    scatter_pred_vs_true(stack_true, stack_pred, "outputs/figures/pred_vs_realized.png")

if __name__ == "__main__":
    main()