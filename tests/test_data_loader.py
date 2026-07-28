"""Unit tests for data_loader module."""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import compute_returns, train_test_split


def test_compute_returns():
    """Test that returns are computed correctly."""
    prices = pd.DataFrame({"A": [100, 101, 102, 101, 100]})
    returns = compute_returns(prices)
    expected = [0.01, 0.00990099, -0.00980392, -0.00990099]
    # Use full array, not sliced
    np.testing.assert_almost_equal(returns["A"].values, expected, decimal=5)


def test_train_test_split():
    """Test chronological train/test split."""
    dates = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({"A": range(10)}, index=dates)
    train, test = train_test_split(data, "2020-01-05")
    assert len(train) == 4
    assert len(test) == 6
    assert train.index[-1] < test.index[0]


def test_train_test_split_with_index():
    """Test that split uses the correct date boundary."""
    dates = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({"A": range(10)}, index=dates)
    train, test = train_test_split(data, "2020-01-06")
    assert len(train) == 5
    assert len(test) == 5