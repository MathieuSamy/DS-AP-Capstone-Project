from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    """Central configuration for universe, dates, and backtest settings."""
    tickers: List[str] = field(default_factory=lambda: [
        "TSLA", "F", "GM", "TM", "BMW.DE", "RNO.PA", "STLA", "HMC"
    ])
    benchmark: str = "CARZ"          # Sector ETF; if unavailable, use an equal-weight index
    horizon_days: int = 5            # Prediction horizon (trading days)
    train_start: str = "2016-01-01"
    test_start:  str = "2022-01-01"
    test_end:    str = "2025-09-30"
    top_k: int = 5                   # Portfolio size for backtest
    seed: int = 42