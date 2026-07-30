"""Backtesting module for portfolio strategies."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    rebalance_frequency: str = "monthly"  # "monthly" or "hold"
    transaction_cost: float = 0.001  # 0.1%


def backtest_strategy(
    prices: pd.DataFrame,
    weights: np.ndarray,
    config: BacktestConfig = None
) -> Dict[str, Any]:
    """
    Backtest a portfolio strategy.

    Args:
        prices: Price data for all assets.
        weights: Target portfolio weights.
        config: Backtest configuration.

    Returns:
        Dictionary with results: returns, cumulative returns, metrics.
    """
    if config is None:
        config = BacktestConfig()

    # Calculate daily returns
    returns = prices.pct_change().dropna()

    # Strategy returns (weighted average)
    strategy_returns = (returns * weights).sum(axis=1)

    # Cumulative returns
    cumulative = (1 + strategy_returns).cumprod()

    # Metrics
    total_return = cumulative.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
    volatility = strategy_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

    # Max drawdown
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    return {
        "returns": strategy_returns,
        "cumulative": cumulative,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "config": config
    }


def benchmark_60_40(
    prices: pd.DataFrame,
    spy_col: str = "SPY",
    bnd_col: str = "BND"
) -> pd.Series:
    """
    Create a 60/40 benchmark portfolio.

    Args:
        prices: Price data.
        spy_col: Column name for SPY.
        bnd_col: Column name for BND.

    Returns:
        Benchmark returns series.
    """
    weights = np.array([0.6, 0.4])
    returns = prices[[spy_col, bnd_col]].pct_change().dropna()
    return (returns * weights).sum(axis=1)