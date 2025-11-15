import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from auto_ml_pkg.config import Config
from auto_ml_pkg.data import fetch_prices, fetch_benchmark 
from auto_ml_pkg.features import make_features, make_targets_excess
from auto_ml_pkg.models import make_ridge
from auto_ml_pkg.evaluate import regression_report, information_coefficient
from auto_ml_pkg.backtest import equity_curve
from auto_ml_pkg.viz import plot_equity, scatter_pred_vs_true


def build_walkforward_folds(cfg: Config) -> list[dict]:
    """
    Defines an expanding-window walk-forward scheme.
    Folds are non-overlapping in the test period.
    """
    folds = [
        # train_start,     train_end,     test_start,      test_end
        {"train_start": "2016-01-01", "train_end": "2019-12-31",
         "test_start":  "2020-01-01", "test_end":  "2020-12-31"},
        {"train_start": "2016-01-01", "train_end": "2020-12-31",
         "test_start":  "2021-01-01", "test_end":  "2021-12-31"},
        {"train_start": "2016-01-01", "train_end": "2021-12-31",
         "test_start":  "2022-01-01", "test_end":  "2022-12-31"},
        {"train_start": "2016-01-01", "train_end": "2022-12-31",
         "test_start":  "2023-01-01", "test_end":  "2023-12-31"},
        {"train_start": "2016-01-01", "train_end": "2023-12-31",
         "test_start":  "2024-01-01", "test_end":  "2024-12-31"},
        {"train_start": "2016-01-01", "train_end": "2024-12-31",
         "test_start":  "2025-01-01", "test_end":  cfg.test_end},
    ]
    return folds


def main():
    # === 0) Configuration & folders ===
    cfg = Config()
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/artifacts", exist_ok=True)

    # === 1) Prices & benchmark (equal-weight) ===
    prices = fetch_prices(cfg.tickers, cfg.train_start, cfg.test_end).dropna(how="all")

    print("\n=== DATA AVAILABILITY CHECK ===")
    print(f"Date range in prices: {prices.index[0]} → {prices.index[-1]}")
    print(f"Number of trading days: {len(prices)}")
    print("Sample preview:")
    print(prices.head(), "\n")

    # Try to use CARZ as benchmark; if unavailable, fall back to an equal-weight universe index.
    bench = fetch_benchmark(
        symbol=cfg.benchmark,
        start=cfg.train_start,
        end=cfg.test_end,
        fallback_from=prices,
    )

    print("=== BENCHMARK CHECK ===")
    print(f"Benchmark name: {bench.name}")
    print(f"Benchmark first/last: {bench.iloc[0]} → {bench.iloc[-1]}")
    print("NaN ratio:", bench.isna().mean(), "\n")

    # === 2) Features & targets computed once ===
    X = make_features(prices)
    Y = make_targets_excess(prices, bench, cfg.horizon_days)

    print("=== TARGETS CHECK ===")
    print(f"Y shape: {Y.shape}")
    print(f"Y NaN ratio: {Y.isna().mean().mean():.2f}")
    print("Overlap between X and Y:", len(X.index.intersection(Y.index)))
    print("First 5 rows of Y:")
    print(Y.head(), "\n")

    # === 3) Define walk-forward folds ===
    folds = build_walkforward_folds(cfg)

    all_P = []         # predictions for test periods (all folds)
    all_Y = []         # realized excess returns for test periods (all folds)
    fold_metrics = []  # per-fold metrics

    # === 4) Loop over folds ===
    for i, f in enumerate(folds, start=1):
        train_start = pd.Timestamp(f["train_start"])
        train_end   = pd.Timestamp(f["train_end"])
        test_start  = pd.Timestamp(f["test_start"])
        test_end    = pd.Timestamp(f["test_end"])

        train_mask = (X.index >= train_start) & (X.index <= train_end)
        test_mask  = (X.index >= test_start)  & (X.index <= test_end)

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print(f"[INFO] Fold {i}: empty train or test segment, skipped.")
            continue

        print(f"\n==== Fold {i} ====")
        print(f"Train: {train_start.date()} → {train_end.date()}")
        print(f"Test : {test_start.date()} → {test_end.date()}")
        print(f"Train days: {train_mask.sum()}, Test days: {test_mask.sum()}")

        preds_fold = {}
        reals_fold = {}

        # Per-ticker ridge regression for this fold
        for t in tqdm(cfg.tickers, desc=f"Per-ticker fit (fold {i})"):
            # Feature columns corresponding to this ticker
            cols = [c for c in X.columns if any(
                c.startswith(p) for p in (f"mom_{t}_", f"vol_{t}_", f"ma_ratio_{t}_", f"rsi_{t}_")
            )]

            if t not in Y.columns:
                print(f"[SKIP] {t}: target column missing in Y.")
                continue
            if not cols:
                print(f"[SKIP] {t}: no feature columns found.")
                continue

            # Align target with X index
            Y_t = Y[t].reindex(X.index)
            Y_t.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_clean = X[cols].replace([np.inf, -np.inf], np.nan)

            # Split for this fold
            Xtr, Xte = X_clean.loc[train_mask], X_clean.loc[test_mask]
            ytr, yte = Y_t.loc[train_mask],     Y_t.loc[test_mask]

            tr = pd.concat([Xtr, ytr], axis=1).dropna()
            te = pd.concat([Xte, yte], axis=1).dropna()

            if len(tr) < 100 or len(te) < 20:
                print(f"[SKIP] {t} (fold {i}): insufficient data (train={len(tr)}, test={len(te)}).")
                continue

            model = make_ridge(alpha=2.0)
            model.fit(tr[cols], tr[t])

            p = pd.Series(model.predict(te[cols]), index=te.index, name=t)
            preds_fold[t] = p
            reals_fold[t] = te[t]

        if not preds_fold:
            print(f"[INFO] Fold {i}: no predictions created, skipped.")
            continue

        P_fold = pd.DataFrame(preds_fold).sort_index()
        Y_fold = pd.DataFrame(reals_fold).sort_index()

        # Store test predictions only
        all_P.append(P_fold)
        all_Y.append(Y_fold)

        # Per-fold metrics (on test only)
        stack_true = Y_fold.stack()
        stack_pred = P_fold.stack()

        reg = regression_report(stack_true, stack_pred)
        ic = information_coefficient(stack_true, stack_pred)

        fold_record = {
            "fold": i,
            "train_start": str(train_start.date()),
            "train_end":   str(train_end.date()),
            "test_start":  str(test_start.date()),
            "test_end":    str(test_end.date()),
            "MSE": reg["MSE"],
            "MAE": reg["MAE"],
            "R2":  reg["R2"],
            "IC_pearson":  ic["pearson"],
            "IC_spearman": ic["spearman"],
        }
        fold_metrics.append(fold_record)
        print("Fold metrics:", fold_record)

    if not all_P:
        raise RuntimeError("No predictions created in any fold. Check folds or data availability.")

    # === 5) Aggregate all out-of-sample predictions (all test periods) ===
    P_all = pd.concat(all_P, axis=0).sort_index()
    Y_all = pd.concat(all_Y, axis=0).sort_index()

    # Align indices just in case
    Y_all = Y_all.reindex(P_all.index).ffill().bfill()

    # Global metrics over all test predictions
    stack_true_all = Y_all.stack()
    stack_pred_all = P_all.stack()

    reg_global = regression_report(stack_true_all, stack_pred_all)
    ic_global = information_coefficient(stack_true_all, stack_pred_all)

    print("\n==== GLOBAL OUT-OF-SAMPLE METRICS (All folds) ====")
    for k, v in reg_global.items():
        print(f"{k}: {v:.6f}")
    print("IC:", ic_global)

    # Save per-fold metrics
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv("outputs/artifacts/walkforward_metrics.csv", index=False)

    # === 6) Backtest & plots on the full walk-forward OOS period ===
    ec = equity_curve(
    P_all,
    Y_all,
    top_k=cfg.top_k,
    rebalance_every=cfg.horizon_days,
    transaction_cost_bps=cfg.transaction_cost_bps,
)
    ec.to_csv("outputs/artifacts/equity_curve_walkforward.csv")

    plot_equity(
        ec,
        "outputs/figures/equity_curve_walkforward.png",
        title=f"Walk-forward Top-{cfg.top_k} long — Ridge — h={cfg.horizon_days}"
    )
    scatter_pred_vs_true(
        stack_true_all,
        stack_pred_all,
        "outputs/figures/pred_vs_realized_walkforward.png"
    )

    P_all.to_csv("outputs/artifacts/predictions_walkforward.csv")
    Y_all.to_csv("outputs/artifacts/realized_excess_walkforward.csv")

    print("\n✅ Walk-forward experiment completed. Results saved in 'outputs/'.\n")


if __name__ == "__main__":
    main()