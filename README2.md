This repository contains the full implementation of my capstone project for the Data Science & Advanced Programming course at HEC Lausanne.
The objective was to build a clean, reproducible and fully automated Python pipeline to test whether technical indicators + machine learning can predict short-horizon excess returns in the global automobile industry.

The project includes:
	•	A complete data engineering pipeline (Yahoo Finance API + caching system)
	•	Feature engineering based only on past-dependent technical indicators
	•	Per-ticker Ridge Regression models
	•	Single-split and Walk-Forward expanding-window evaluations
	•	A realistic Top-K long-only trading strategy with transaction costs
	•	A full backtesting engine including turnover computation

⸻

1. Project Goal

This project predicts 5-day ahead excess returns of 13 major global automakers:
	•	USA: TSLA, F, GM
	•	Europe: BMW.DE, RNO.PA, STLA, MBG.DE
	•	Japan: TM, 7269.T (Suzuki), 7270.T (Subaru), 7261.T (Mazda)
	•	Korea: 005380.KS (Hyundai)

Targets are defined relative to the CARZ automotive ETF (or equal-weight fallback).

The final strategy selects the Top-5 predicted stocks, invests equally, and rebalances every 5 days.

⸻

2. Key Features

Feature Engineering

All features use past-only information:
	•	Momentum (5, 20, 60 days)
	•	Realized volatility (20 days)
	•	Price / MA20 ratio
	•	RSI(14)

Models
	•	Ridge Regression (one model per ticker)

Chosen for:
	•	High stability
	•	Low risk of overfitting
	•	Ability to extract weak but meaningful signals

Evaluation Metrics
	•	MSE, MAE, R²
	•	Pearson & Spearman Information Coefficient (IC)
	•	Cumulative performance (equity curve)
	•	Turnover diagnostics & transaction-cost-adjusted returns

⸻

3. Backtesting

The trading strategy:
	•	Long-only
	•	Equal-weighted Top-5
	•	Rebalanced every h = 5 days
	•	Includes 10 bps per turnover transaction costs
	•	Produces both gross and net equity curves

Two evaluation modes:
	1.	Single Train-Test Split (2016–2022 → 2023–2025)
	2.	Walk-Forward (2016 → … → 2025), 6 folds

Outputs are automatically exported:

auto_ml/
│
├── outputs/
│   ├── figures/      # PNG charts (equity curves, scatter plots, benchmarks)
│   └── artifacts/    # CSV predictions, realized excess returns, metrics

4. Project Structure

auto_ml/
│
├── auto_ml_pkg/
│   ├── __init__.py
│   ├── config.py              # Global config (tickers, dates, backtest params)
│   ├── data.py                # Yahoo Finance download + cache system
│   ├── features.py            # Technical indicators + target creation
│   ├── models.py              # Ridge model creation
│   ├── evaluate.py            # Regression metrics + IC
│   ├── backtest.py            # Top-K strategy + turnover + costs + equity
│   ├── viz.py                 # Visualization utilities
│   ├── run_experiment_single_split.py
│   └── run_experiment_walkforward.py
│
├── data/
│   ├── cache/                 # Cached daily prices
│   └── raw/                   # Manual Yahoo CSVs (optional)
│
├── outputs/
│   ├── figures/
│   └── artifacts/
│
├── environment.yml
├── README.md
└── requirements.txt

5. Installation & Execution

Clone the Repository 

git clone https://github.com/MathieuSamy/auto_ml.git
cd auto_ml

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

6. Run Experiments

Single Train–Test Split

python auto_ml_pkg/run_experiment_single_split.py

Walk-Forward Evaluation

python auto_ml_pkg/run_experiment_walkforward.py

All outputs will appear under:

auto_ml/outputs/figures/
auto_ml/outputs/artifacts/

7. Exported Results

Figures
	•	equity_curve.png
	•	equity_curve_walkforward.png
	•	pred_vs_realized.png
	•	pred_vs_realized_walkforward.png
	•	benchmarks_CARZ_vs_EW_single_split.png
	•	strategy_CARZ_vs_EW_single_split.png

Artifacts (CSV)
	•	equity_curve.csv
	•	predictions.csv
	•	realized_excess.csv
	•	walkforward_metrics.csv
	•	predictions_walkforward.csv
	•	realized_excess_walkforward.csv

⸻

Author

Mathieu SAMY
Master in Finance — HEC Lausanne
mathieu.samy@unil.ch
Lausanne, Switzerland

⸻

License

MIT License — free to use, modify and distribute.

⸻

Academic Context

This project was completed as the final capstone for the Data Science & Advanced Programming course (HEC Lausanne, Fall 2025).
The methodology, code design, walk-forward evaluation and backtesting follow the expected academic standards for quantitative finance research.