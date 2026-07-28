"""Portfolio optimization using Modern Portfolio Theory."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from scipy.optimize import minimize


def portfolio_performance(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate portfolio return, risk, and Sharpe ratio.

    Args:
        weights: Asset weights.
        expected_returns: Expected returns for each asset.
        cov_matrix: Covariance matrix.

    Returns:
        Tuple of (return, risk, sharpe_ratio).
    """
    port_return = np.dot(weights, expected_returns)
    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = port_return / port_risk if port_risk > 0 else 0
    return port_return, port_risk, sharpe


def optimize_max_sharpe(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    num_assets: int
) -> Dict[str, Any]:
    """
    Find portfolio with maximum Sharpe ratio.

    Args:
        expected_returns: Expected returns for each asset.
        cov_matrix: Covariance matrix.
        num_assets: Number of assets.

    Returns:
        Dictionary with optimal weights and performance metrics.
    """
    def neg_sharpe(w):
        _, _, sharpe = portfolio_performance(w, expected_returns, cov_matrix)
        return -sharpe

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = [1/num_assets] * num_assets

    result = minimize(
        neg_sharpe,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    weights = result.x
    port_return, port_risk, sharpe = portfolio_performance(weights, expected_returns, cov_matrix)

    return {
        "weights": weights,
        "return": port_return,
        "risk": port_risk,
        "sharpe": sharpe
    }