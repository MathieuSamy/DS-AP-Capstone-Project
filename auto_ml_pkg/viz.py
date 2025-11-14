import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_equity(ec: pd.Series, path: str, title: str = "Top-K long (excess)"):
    """
    Saves a cumulative growth plot of the backtested strategy.
    """
    plt.figure(figsize=(9, 4))
    ax = ec.plot(figsize=(9, 4))
    ax.set_title(title)
    ax.set_ylabel("Cumulative (×)")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def scatter_pred_vs_true(y_true: pd.Series, y_pred: pd.Series, path: str):
    """
    Saves a scatter of predicted vs realized excess returns with a fitted line.
    """
    df = pd.concat([y_true, y_pred], axis=1, keys=["y", "p"]).dropna()
    plt.figure(figsize=(5, 5))
    plt.scatter(df["p"], df["y"], alpha=0.4)
    m, b = np.polyfit(df["p"], df["y"], 1)
    xs = np.linspace(df["p"].min(), df["p"].max(), 100)
    plt.plot(xs, m * xs + b)
    plt.xlabel("Predicted excess")
    plt.ylabel("Realized excess")
    plt.tight_layout()
    plt.savefig(path)