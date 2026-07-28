"""Data loading and preprocessing module."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Configuration for data loading."""
    start_date: str = "2015-01-01"
    end_date: Optional[str] = None
    test_start: str = "2025-01-01"
    assets: List[str] = None

    def __post_init__(self):
        if self.assets is None:
            self.assets = ["TSLA", "SPY", "BND"]


def load_data(tickers: List[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
    """
    Download price data from Yahoo Finance.

    Args:
        tickers: List of stock symbols.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD). Defaults to today.

    Returns:
        DataFrame with adjusted close prices.
    """
    import yfinance as yf
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    data = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end, progress=False)
        data[t] = df["Close"]
    return pd.DataFrame(data)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns."""
    return prices.pct_change().dropna()


def train_test_split(data: pd.DataFrame, test_start: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data chronologically."""
    train = data.loc[data.index < test_start]
    test = data.loc[data.index >= test_start]
    return train, test