"""Unit tests for portfolio module."""

import pytest
import numpy as np
from src.portfolio import portfolio_performance, optimize_max_sharpe


def test_portfolio_performance():
    """Test portfolio performance calculations."""
    weights = np.array([0.5, 0.5])
    expected_returns = np.array([0.10, 0.15])
    cov_matrix = np.array([[0.04, 0.02], [0.02, 0.09]])

    ret, risk, sharpe = portfolio_performance(weights, expected_returns, cov_matrix)

    assert ret == 0.125
    assert risk > 0
    assert sharpe > 0


def test_optimize_max_sharpe():
    """Test maximum Sharpe ratio optimization."""
    expected_returns = np.array([0.10, 0.15, 0.12])
    cov_matrix = np.array([
        [0.04, 0.02, 0.01],
        [0.02, 0.09, 0.02],
        [0.01, 0.02, 0.04]
    ])

    result = optimize_max_sharpe(expected_returns, cov_matrix, 3)

    assert "weights" in result
    assert "return" in result
    assert "risk" in result
    assert "sharpe" in result
    assert abs(sum(result["weights"]) - 1.0) < 0.01