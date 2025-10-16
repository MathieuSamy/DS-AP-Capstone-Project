# scripts/evaluate_model.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load results ---
P = pd.read_csv("outputs/artifacts/predictions.csv", index_col=0, parse_dates=True)
Yf = pd.read_csv("outputs/artifacts/realized_excess.csv", index_col=0, parse_dates=True)
ec = pd.read_csv("outputs/artifacts/equity_curve.csv", index_col=0, parse_dates=True)

# --- Summary statistics ---
print("Predictions shape:", P.shape)
print("Correlation (mean):", P.corrwith(Yf).mean())

# --- Plot per-ticker prediction accuracy ---
corrs = {t: P[t].corr(Yf[t]) for t in P.columns if t in Yf}
sns.barplot(x=list(corrs.keys()), y=list(corrs.values()))
plt.title("Prediction correlation per automaker")
plt.ylabel("Pearson correlation")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Plot equity curve ---
ec.plot(figsize=(8,4), title="Equity Curve (Top-k Portfolio)")
plt.ylabel("Cumulative performance")
plt.xlabel("Date")
plt.grid()
plt.show()