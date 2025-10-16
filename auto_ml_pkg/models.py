from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def make_ridge(alpha=2.0):
    """
    Ridge regression baseline wrapped in a pipeline with standardization.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha, random_state=42))
    ])