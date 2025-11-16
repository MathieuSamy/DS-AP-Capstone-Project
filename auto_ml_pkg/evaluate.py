import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_report(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Computes MSE, MAE and R^2 on aligned non-missing pairs of (y_true, y_pred).
    """
    df = pd.concat([y_true, y_pred], axis=1).dropna()
    if df.empty:
        return {"MSE": np.nan, "MAE": np.nan, "R2": np.nan}
    return {
        "MSE": float(mean_squared_error(df.iloc[:, 0], df.iloc[:, 1])),
        "MAE": float(mean_absolute_error(df.iloc[:, 0], df.iloc[:, 1])),
        "R2":  float(r2_score(df.iloc[:, 0], df.iloc[:, 1])),
    }

def information_coefficient(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Computes Pearson and Spearman correlations between predictions and realized targets.
    """
    df = pd.concat([y_true, y_pred], axis=1).dropna()
    if len(df) < 3:
        return {"pearson": np.nan, "spearman": np.nan}
    return {
        "pearson": float(df.corr(method="pearson").iloc[0, 1]),
        "spearman": float(df.corr(method="spearman").iloc[0, 1]),
    }