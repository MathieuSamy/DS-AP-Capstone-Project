import os  # to handle file system operations
import numpy as np  # to handle numerical operations and arrays
import pandas as pd  # for data manipulation and analysis
from tqdm import tqdm  # to display progress bars during iterations
import sys  # to modify Python path for imports

# ============================================================
# 0) Set up project root and Python path
# ============================================================

# CURRENT_DIR = directory of this file: auto_ml_pkg/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# PROJECT_ROOT = parent directory of auto_ml_pkg/ → auto_ml/
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Add project root (/files/auto_ml) to sys.path so that
# "import auto_ml_pkg.xxx" works even when running from outside.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Define output directories relative to the project root
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")         # auto_ml/outputs
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")          # auto_ml/outputs/figures
ARTIFACTS_DIR = os.path.join(OUTPUTS_DIR, "artifacts")      # auto_ml/outputs/artifacts

# Ensure output folders exist for saving figures and CSV artifacts
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ============================================================
# 1) Import project modules
# ============================================================

# === Import project modules ===
from auto_ml_pkg.config import Config  # to load experiment configuration
from auto_ml_pkg.data import fetch_prices, fetch_benchmark  # to fetch historical price data and benchmark data
from auto_ml_pkg.features import make_features, make_targets_excess  # to create features and targets
from auto_ml_pkg.models import make_ridge  # to create Ridge regression model
from auto_ml_pkg.evaluate import regression_report, information_coefficient  # to evaluate model performance
from auto_ml_pkg.backtest import equity_curve  # to compute equity curve for backtesting
from auto_ml_pkg.viz import plot_equity, scatter_pred_vs_true, plot_multi_equity  # to visualize results


def main():
    # 0) Load configuration (tickers, dates, top_k, transaction costs, etc.)
    cfg = Config()

    # 1) Fetch and inspect data (Download or load cached daily close prices for the whole universe)
    prices = fetch_prices(cfg.tickers, cfg.train_start, cfg.test_end).dropna(how="all")
    # Basic sanity checks on the raw price data
    print("\n=== DATA AVAILABILITY CHECK ===")
    print(f"Date range in prices: {prices.index[0]} → {prices.index[-1]}")
    print(f"Number of trading days: {len(prices)}")
    print("Sample preview:")
    print(prices.head(5), "\n")

    # === 2) Build benchmark (CARZ with equal-weight fallback) ===
    # First try to fetch the benchmark specified in the Config (e.g. 'CARZ').
    # If it is not available (e.g. in this environment), fall back to an equal-weight index of the universe.
    bench = fetch_benchmark(
        symbol=cfg.benchmark,        # e.g. 'CARZ'
        start=cfg.train_start,       # e.g. '2016-01-01'
        end=cfg.test_end,            # e.g. '2025-09-30'
        fallback_from=prices,        # equal-weight from universe if CARZ unavailable
    )

    print("=== BENCHMARK CHECK ===")
    print(f"Benchmark name: {bench.name}")
    print(f"Benchmark first/last: {bench.iloc[0]} → {bench.iloc[-1]}")
    print("NaN ratio:", bench.isna().mean(), "\n")

    # === 3) Compute features (X) and targets (Y) ===
    X = make_features(prices)                                   # to create technical features from price data
    Y = make_targets_excess(prices, bench, cfg.horizon_days)    # to create excess return targets

    print("=== TARGETS CHECK ===")
    print(f"Y shape: {Y.shape}")
    print(f"Y NaN ratio: {Y.isna().mean().mean():.2f}")
    print("Overlap between X and Y:", len(X.index.intersection(Y.index)))
    print("First 5 rows of Y:")
    print(Y.head(), "\n")

    # === 4) Time split ===
    train_mask = (X.index >= pd.Timestamp(cfg.train_start)) & (X.index <= pd.Timestamp(cfg.train_end))   # to create training data mask
    test_mask = (X.index >= pd.Timestamp(cfg.test_start)) & (X.index <= pd.Timestamp(cfg.test_end))      # to create testing data mask

    print("=== SPLIT CHECK ===")
    print("Train days:", train_mask.sum(), "Test days:", test_mask.sum())

    # === 5) Per-ticker model fit ===
    preds, reals = {}, {}                               # to store predictions and realized values

    for t in tqdm(cfg.tickers, desc="Per-ticker fit"):  # to iterate over each ticker with progress bar
        # Select relevant feature columns for this ticker
        cols = [
            c
            for c in X.columns
            if any(
                c.startswith(p)
                for p in (f"mom_{t}_", f"vol_{t}_", f"ma_ratio_{t}_", f"rsi_{t}_")
            )
        ]  # to filter feature columns for the ticker

        if t not in Y.columns:                          # to check if target column exists
            print(f"[SKIP] {t}: target column missing in Y.")
            continue
        if not cols:                                    # to check if any feature columns were found
            print(f"[SKIP] {t}: no feature columns found.")
            continue

        # Align indexes safely and clean data
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

    # === 6) Aggregate predictions === (from dict to DataFrame)
    P = pd.DataFrame(preds).sort_index()                          # to create DataFrame of predictions
    Yf = pd.DataFrame(reals).sort_index()                         # to create DataFrame of realized values

    if P.empty:
        raise RuntimeError("No predictions created. Check tickers, target alignment, or data availability.")

    # === 7) Evaluate === (regression metrics and IC)
    stack_true = Yf.stack()                                    # to stack realized values
    stack_pred = P.stack()                                     # to stack predicted values

    reg = regression_report(stack_true, stack_pred)            # to compute regression metrics
    ic = information_coefficient(stack_true, stack_pred)       # to compute information coefficient

    print("\n==== REGRESSION METRICS ====")
    for k, v in reg.items():
        print(f"{k}: {v:.6f}")
    print("IC:", ic)

    # === 8) Backtest & export === (Top-k long strategy based on predicted excess returns)
    ec = equity_curve(                                      # to compute equity curve for backtesting
        P,                                                  # predicted scores
        Yf,                                                 # realized excess returns
        top_k=cfg.top_k,                                    # e.g. 5
        rebalance_every=cfg.horizon_days,                   # normally 5
        transaction_cost_bps=cfg.transaction_cost_bps,      # for example 10 bps
    )

    # Save main artifacts (using global OUTPUTS_DIR structure)
    ec.to_csv(os.path.join(ARTIFACTS_DIR, "equity_curve.csv"))          # to save equity curve to CSV
    P.to_csv(os.path.join(ARTIFACTS_DIR, "predictions.csv"))            # to save predictions to CSV
    Yf.to_csv(os.path.join(ARTIFACTS_DIR, "realized_excess.csv"))       # to save realized excess returns to CSV

    # Plot main diagnostic figures
    plot_equity(
        ec,
        os.path.join(FIGURES_DIR, "equity_curve.png"),                  # to plot equity curve
        title=f"Top-{cfg.top_k} long — Ridge — h={cfg.horizon_days}",   # plot title
    )
    scatter_pred_vs_true(
        stack_true,
        stack_pred,
        os.path.join(FIGURES_DIR, "pred_vs_realized.png"),              # to plot scatter of predictions vs realized
    )

    # === 9) Benchmark comparison: CARZ vs Equal-Weight universe ===
    # We rebuild an equal-weight benchmark from the same automotive universe
    # to compare it with the CARZ ETF (sector benchmark).

    bench_ew_ret = prices.pct_change(fill_method=None).mean(axis=1)  # equal-weight daily returns
    bench_ew = (1 + bench_ew_ret.fillna(0)).cumprod()                # cumulative returns
    bench_ew = bench_ew / bench_ew.iloc[0] * 100.0                   # rescale to base 100
    bench_ew.name = "EQUAL_WEIGHT_BENCH"

    # The 'bench' variable used earlier in make_targets_excess corresponds to CARZ (via fetch_benchmark).
    # We rescale it to base 100 for a fair visual comparison.
    bench_carz = bench / bench.iloc[0] * 100.0
    bench_carz.name = "CARZ"

    bench_df = pd.concat([bench_carz, bench_ew], axis=1).dropna()  # to combine both benchmarks into a DataFrame
    bench_df.to_csv(
        os.path.join(ARTIFACTS_DIR, "benchmarks_CARZ_vs_EW_single_split.csv")
    )  # to save benchmarks comparison to CSV

    plot_multi_equity(  # to plot multiple equity curves
        bench_df,
        os.path.join(FIGURES_DIR, "benchmarks_CARZ_vs_EW_single_split.png"),
        title="CARZ vs Equal-Weight Automotive Benchmark (Single Split)",
    )

    # === 10) Strategy comparison: excess vs CARZ vs excess vs Equal-Weight ===
    # We keep the same predictions P (trained with CARZ-based targets),
    # but we recompute future excess returns using the equal-weight benchmark
    # and see how the same signals behave under a different definition of "excess return".

    # 10.1 Recompute excess returns vs equal-weight benchmark
    Y_ew = make_targets_excess(prices, bench_ew, cfg.horizon_days)  # to create excess return targets vs equal-weight benchmark

    # Align Y_ew with the dates where we have predictions (Yf).
    Y_ew_aligned = Y_ew.reindex(Yf.index).ffill().bfill()  # to align excess returns with predictions

    # 10.2 Backtest strategy using excess vs CARZ (already done as 'ec')
    ec_carz = ec.rename("Strategy_excess_vs_CARZ")  # to rename equity curve for CARZ benchmark

    # 10.3 Backtest strategy using excess vs Equal-Weight
    ec_ew = equity_curve(  # to compute equity curve for equal-weight benchmark
        P,
        Y_ew_aligned,
        top_k=cfg.top_k,
        rebalance_every=cfg.horizon_days,
        transaction_cost_bps=cfg.transaction_cost_bps,
    ).rename("Strategy_excess_vs_EW")

    # 10.4 Save and plot both curves
    strat_df = pd.concat([ec_carz, ec_ew], axis=1).dropna()  # to combine both strategy equity curves into a DataFrame
    strat_df.to_csv(
        os.path.join(ARTIFACTS_DIR, "strategy_CARZ_vs_EW_single_split.csv")
    )

    plot_multi_equity(  # to plot multiple equity curves for strategies
        strat_df,
        os.path.join(FIGURES_DIR, "strategy_CARZ_vs_EW_single_split.png"),
        title=f"Strategy vs Two Benchmarks (Top-{cfg.top_k}, h={cfg.horizon_days}, Single Split)",
    )

    print(f"\n✅ Experiment completed successfully. Results saved in '{OUTPUTS_DIR}/'.\n")


if __name__ == "__main__":
    main()