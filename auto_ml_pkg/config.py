from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    """Central configuration for universe, dates, and backtest settings."""
    tickers: List[str] = field(default_factory=lambda: [
        "TSLA", "F", "GM", "TM", "BMW.DE", "RNO.PA", "STLA", "HMC"
    ])
    benchmark: str = "CARZ"          # Sector ETF; if unavailable, use equal-weight
    horizon_days: int = 5            # Prediction horizon (trading days)
    
    # Updated date ranges for your current dataset
    train_start = "2016-01-01"
    train_end   = "2022-12-31"
    test_start  = "2023-01-01"
    test_end    = "2025-09-30"

    top_k: int = 5
    seed: int = 42