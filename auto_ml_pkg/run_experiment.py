import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# === Import project modules ===
from auto_ml_pkg.config import Config
from auto_ml_pkg.data import fetch_prices
from auto_ml_pkg.features import make_features, make_targets_excess
from auto_ml_pkg.models import make_ridge
from auto_ml_pkg.evaluate import regression_report, information_coefficient
from auto_ml_pkg.backtest import equity_curve
from auto_ml_pkg.viz import plot_equity, scatter_pred_vs_true


def main():
    # === 0) Load configuration ===
    cfg = Config()
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/artifacts", exist_ok=True)

    # === 1) Fetch and inspect data ===
    prices = fetch_prices(cfg.tickers, cfg.train_start, cfg.test_end).dropna(how="all")

    print("\n=== DATA AVAILABILITY CHECK ===")
    print(f"Date range in prices: {prices.index[0]} → {prices.index[-1]}")
    print(f"Number of trading days: {len(prices)}")
    print("Sample preview:")
    print(prices.head(5), "\n")

    # === 2) Build equal-weight benchmark ===
    bench_ret = prices.pct_change(fill_method=None).mean(axis=1)
    bench = (1 + bench_ret.fillna(0)).cumprod()
    bench = bench / bench.iloc[0] * 100.0
    bench.name = "EQUAL_WEIGHT_BENCH"

    print("=== BENCHMARK CHECK ===")
    print(f"Benchmark first/last: {bench.iloc[0]} → {bench.iloc[-1]}")
    print("NaN ratio:", bench.isna().mean(), "\n")

    # === 3) Compute features (X) and targets (Y) ===
    X = make_features(prices)
    Y = make_targets_excess(prices, bench, cfg.horizon_days)

    print("=== TARGETS CHECK ===")
    print(f"Y shape: {Y.shape}")
    print(f"Y NaN ratio: {Y.isna().mean().mean():.2f}")
    print("Overlap between X and Y:", len(X.index.intersection(Y.index)))
    print("First 5 rows of Y:")
    print(Y.head(), "\n")

    # === 4) Time split ===
    train_mask = (X.index >= pd.Timestamp(cfg.train_start)) & (X.index <= pd.Timestamp(cfg.train_end))
    test_mask = (X.index >= pd.Timestamp(cfg.test_start)) & (X.index <= pd.Timestamp(cfg.test_end))

    print("=== SPLIT CHECK ===")
    print("Train days:", train_mask.sum(), "Test days:", test_mask.sum())

    # === 5) Per-ticker model fit ===
    preds, reals = {}, {}

    for t in tqdm(cfg.tickers, desc="Per-ticker fit"):
        # Select relevant feature columns for this ticker
        cols = [c for c in X.columns if any(
            c.startswith(p) for p in (f"mom_{t}_", f"vol_{t}_", f"ma_ratio_{t}_", f"rsi_{t}_")
        )]

        if t not in Y.columns:
            print(f"[SKIP] {t}: target column missing in Y.")
            continue
        if not cols:
            print(f"[SKIP] {t}: no feature columns found.")
            continue

        # Align indexes safely
        Y_t = Y[t].reindex(X.index)
        Y_t.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_clean = X[cols].replace([np.inf, -np.inf], np.nan)

        # Split into train/test
        Xtr, Xte = X_clean.loc[train_mask], X_clean.loc[test_mask]
        ytr, yte = Y_t.loc[train_mask], Y_t.loc[test_mask]

        # Drop missing rows
        tr = pd.concat([Xtr, ytr], axis=1).dropna()
        te = pd.concat([Xte, yte], axis=1).dropna()

        # Check minimal data
        if len(tr) < 100 or len(te) < 20:
            print(f"[SKIP] {t}: insufficient data (train={len(tr)}, test={len(te)}).")
            continue

        # Train model
        model = make_ridge(alpha=2.0)
        model.fit(tr[cols], tr[t])

        # Predict
        p = pd.Series(model.predict(te[cols]), index=te.index, name=t)
        preds[t] = p
        reals[t] = te[t]

    # === 6) Aggregate predictions ===
    P = pd.DataFrame(preds).sort_index()
    Yf = pd.DataFrame(reals).sort_index()

    if P.empty:
        raise RuntimeError("No predictions created. Check tickers, target alignment, or data availability.")

    # === 7) Evaluate ===
    stack_true = Yf.stack()
    stack_pred = P.stack()

    reg = regression_report(stack_true, stack_pred)
    ic = information_coefficient(stack_true, stack_pred)

    print("\n==== REGRESSION METRICS ====")
    for k, v in reg.items():
        print(f"{k}: {v:.6f}")
    print("IC:", ic)

    # === 8) Backtest & export ===
    ec = equity_curve(P, Yf, top_k=cfg.top_k)
    ec.to_csv("outputs/artifacts/equity_curve.csv")
    P.to_csv("outputs/artifacts/predictions.csv")
    Yf.to_csv("outputs/artifacts/realized_excess.csv")

    plot_equity(ec, "outputs/figures/equity_curve.png",
                title=f"Top-{cfg.top_k} long — Ridge — h={cfg.horizon_days}")
    scatter_pred_vs_true(stack_true, stack_pred, "outputs/figures/pred_vs_realized.png")

    print("\n✅ Experiment completed successfully. Results saved in 'outputs/'.\n")


if __name__ == "__main__":
    main()