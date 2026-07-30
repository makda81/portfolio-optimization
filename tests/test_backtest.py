"""Unit tests for backtest module."""

import pytest
import pandas as pd
import numpy as np
from src.backtest import backtest_strategy, BacktestConfig


def test_backtest_strategy():
    """Test backtesting functionality."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    prices = pd.DataFrame({
        "A": 100 + np.cumsum(np.random.randn(100) * 0.5),
        "B": 200 + np.cumsum(np.random.randn(100) * 0.3)
    }, index=dates)

    weights = np.array([0.5, 0.5])
    result = backtest_strategy(prices, weights)

    assert "returns" in result
    assert "cumulative" in result
    assert "total_return" in result
    assert "sharpe_ratio" in result
    assert "max_drawdown" in result