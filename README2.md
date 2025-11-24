## AutoML for Excess Return Prediction in the Car Industry
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Last Update](https://img.shields.io/badge/Updated-Oct_2025-lightgrey.svg)

### Capstone Project — Data Science & Advanced Programming (HEC Lausanne)

This repository contains the full implementation of my capstone project for the Data Science & Advanced Programming course at HEC Lausanne.  
The objective is to build a clean, reproducible, and fully automated Python pipeline to test whether technical indicators + machine learning can predict **short-horizon excess returns in the global automobile industry**.

---

###  Project Overview

The project includes:

- A complete **data engineering pipeline** (Yahoo Finance API + caching system)  
- **Feature engineering** based solely on past-dependent technical indicators  
- **Per-ticker Ridge Regression models**  
- **Single-split** and **Walk-Forward expanding-window** evaluations  
- A realistic **Top-K long-only trading strategy** with transaction costs  
- A full **backtesting engine** including turnover computation  

---

## 1. Project Goal

This project predicts **5-day ahead excess returns** for 13 major global automakers:

- **USA**: TSLA, F, GM  
- **Europe**: BMW.DE, RNO.PA, STLA, MBG.DE  
- **Japan**: TM, 7269.T, 7270.T, 7261.T  
- **Korea**: 005380.KS (Hyundai)

Targets are defined relative to the **CARZ automotive ETF** (or equal-weight fallback).

The final strategy selects the **Top-5 predicted stocks**, invests equally, and rebalances every 5 days.

---

## 2. Key Features

### Feature Engineering

All features are strictly past-dependent:

- Momentum (5, 20, 60 days)
- Realized volatility (20 days)
- Price / MA20 ratio
- RSI(14)

### Models

- **Ridge Regression** (one model per ticker)

Chosen for its:

- High stability  
- Low overfitting risk  
- Ability to extract weak but meaningful signals  

### Evaluation Metrics

- MSE, MAE, R²  
- Pearson & Spearman Information Coefficient (IC)  
- Cumulative performance (equity curve)  
- Turnover diagnostics & transaction-cost-adjusted returns  

---

## 3. Backtesting

The trading strategy:

- Long-only  
- Equal-weighted **Top-5**  
- Rebalanced every 5 days  
- Includes **10 bps per turnover** transaction costs  
- Produces **gross and net** equity curves  

### Evaluation Modes

1. **Single Train–Test Split** (2016–2022 → 2023–2025)  
2. **Walk-Forward** (2016 → … → 2025), 6 folds  

### Output Structure
```
auto_ml/
│
├── outputs/
│   ├── figures/      # PNG charts (equity curves, scatter plots, benchmarks)
│   └── artifacts/    # CSV predictions, realized excess returns, metrics
```
---

## 4. Project Structure

```
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
```

---

## 5. Installation & Execution

### Clone the Repository

```bash
git clone https://github.com/MathieuSamy/auto_ml.git
cd auto_ml

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
``` 
## 6. Run Experiments

Choose and run one of the experiments below from the project root.

Single Train–Test Split

```bash
# Single Train–Test Split
python auto_ml_pkg/run_experiment_single_split.py
```

Walk‑Forward Evaluation

```bash
# Walk‑Forward Evaluation (6 folds)
python auto_ml_pkg/run_experiment_walkforward.py
```

All outputs will be written to:

auto_ml/outputs/figures/  
auto_ml/outputs/artifacts/

## 7. Exported Results

Figures
- equity_curve.png
- equity_curve_walkforward.png
- pred_vs_realized.png
- pred_vs_realized_walkforward.png
- benchmarks_CARZ_vs_EW_single_split.png
- strategy_CARZ_vs_EW_single_split.png

Artifacts (CSV)
- equity_curve.csv
- predictions.csv
- realized_excess.csv
- walkforward_metrics.csv
- predictions_walkforward.csv
- realized_excess_walkforward.csv

--- 

**Mathieu SAMY**  
Master in Finance — HEC Lausanne  
Email: mathieu.samy@unil.ch  
Location: Lausanne, Switzerland

---

### Academic Context

This project was completed as the final capstone for the
Data Science & Advanced Programming course (HEC Lausanne, Fall 2025).

The methodology, code design, walk‑forward evaluation and backtesting adhere to academic standards in quantitative finance research.

--- 

### License

MIT License — free to use, modify, and distribute.

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Last Update](https://img.shields.io/badge/Updated-Oct_2025-lightgrey.svg)