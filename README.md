# Predicting Excess Returns in the Car Industry

## Overview
This project is part of the *Data Science and Advanced Programming* capstone final prject at HEC Lausanne.  
It aims to **predict the short-term excess returns** of major car industry companies using publicly available financial data from Yahoo Finance.

The goal is to build and evaluate machine learning models (Linear Regression, Random Forest) that can forecast whether a stock will outperform or underperform the automotive sector benchmark over a short horizon (e.g. one week).

---

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

## 🗂️ Project Structure
