from dataclasses import dataclass, field # to define configuration class
from typing import List # to type-hint list of strings
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class Config:
    """Central configuration for universe, dates, and backtest settings."""
    tickers: List[str] = field(default_factory=lambda: [
        "TSLA",        # Tesla
        "F",           # Ford
        "GM",          # General Motors
        "TM",          # Toyota
        "BMW.DE",      # BMW
        "RNO.PA",      # Renault
        "STLA",        # Stellantis
        "HMC",         # Honda
        "MBG.DE",      # Mercedes-Benz Group
        "005380.KS",   # Hyundai Motor
        "7269.T",      # Suzuki Motor
        "7270.T",      # Subaru
        "7261.T",      # Mazda  
    ])
    benchmark: str = "CARZ"          # Sector ETF for automotive
    horizon_days: int = 5            # Prediction horizon (trading days)
    
    # Date ranges
    train_start = "2016-01-01"
    train_end   = "2022-12-31"
    test_start  = "2023-01-01"
    test_end    = "2025-09-30"

    # Backtest settings
    top_k: int = 5                     # Number of top predicted tickers to hold
    transaction_cost_bps: float = 10.0 # 10 basis points = 0.10%
    seed: int = 42                     # Random seed for reproducibility