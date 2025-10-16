# Predicting Excess Returns in the Car Industry
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Last Update](https://img.shields.io/badge/Updated-Oct_2025-lightgrey.svg)

## Overview
This project is part of the final capstone project for the Data Science and Advanced Programming course at HEC Lausanne.  
It aims to **predict the short-term excess returns** of major car industry companies using publicly available financial data from Yahoo Finance.

The goal is to build and evaluate machine learning models (Linear Regression, Random Forest) that can forecast whether a stock will outperform or underperform the automotive sector benchmark over a short horizon (e.g. one week).

---

Mathieu le boss

## Objectives
- Retrieve and preprocess historical price data for a selection of global car manufacturers (e.g., Tesla, Toyota, Ford, BMW, Renault).
- Engineer time-series based financial features (momentum, volatility, RSI, moving averages).
- Train regression models to predict **excess returns** relative to a benchmark (CARZ ETF or equal-weight sector index).
- Evaluate predictive performance using metrics such as **MSE**, **R²**, and **Information Coefficient (IC)**.
- Backtest a simple **Top-K long strategy** based on model predictions.

---

## Data Sources
All data is obtained from **Yahoo Finance API** via the Python library `yfinance`.  
Main data types include:
- Daily closing prices (`Close`)
- Trading volumes (`Volume`)
- Benchmark index or ETF (`CARZ`, or custom equal-weight benchmark)

---

## Methods
1. **Feature Engineering**  
   - Momentum indicators (5, 20, 60 days)  
   - Realized volatility (20 days)  
   - Moving average ratio (Price / MA20)  
   - Relative Strength Index (RSI14)

2. **Modeling**  
   - Ridge Regression (linear baseline)  
   - Random Forest Regressor (nonlinear baseline)

3. **Evaluation**  
   - Mean Absolute Error (MAE), Mean Squared Error (MSE), R²  
   - Information Coefficient (IC) between predicted and realized excess returns  
   - Cumulative performance (equity curve) of a Top-K predicted portfolio

---

## Experimental Setup
- **Training period:** 2016–2021  
- **Testing period:** 2022–2025  
- **Prediction horizon:** 5 trading days (1 week)  
- **Universe:** 10–15 global automotive stocks  
- **Benchmark:** CARZ ETF (or custom equal-weight auto index)

---

## Project Structure
auto-ml/
```
├── README.md
├── requirements.txt
├── src/
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── backtest.py
│   ├── evaluate.py
│   └── viz.py
├── scripts/
│   └── run_experiment.py
├── notebooks/
│   └── 01_eda.ipynb
└── outputs/
├── figures/
└── artifacts/
```

---

## How to Run

### 1️⃣ Environment Setup
```bash
git clone https://github.com/MathieuSamy/DS-AP-Capstone-Project.git
cd auto-ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣ Run Experiment

```
python scripts/run_experiment.py \
  --model ridge \
  --horizon 5 \
  --train-start 2016-01-01 \
  --test-start 2022-01-01 \
  --test-end 2025-09-30
```

Results are saved automatically under:
```
outputs/figures/
outputs/artifacts/
```

---


**Mathieu Samy**  
Master in Finance — HEC Lausanne  
mathieu.samy@unil.ch  
Lausanne, Switzerland

