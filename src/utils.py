"""Utility functions for portfolio optimization."""

import pandas as pd
import numpy as np
from typing import List, Optional


def calculate_annualized_returns(
    returns: pd.Series,
    periods: int = 252
) -> float:
    """Calculate annualized return from daily returns."""
    return ((1 + returns).prod()) ** (periods / len(returns)) - 1


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods: int = 252
) -> float:
    """Calculate Sharpe ratio."""
    excess_returns = returns - risk_free_rate / periods
    return excess_returns.mean() * np.sqrt(periods) / excess_returns.std()


def calculate_max_drawdown(cumulative: pd.Series) -> float:
    """Calculate maximum drawdown."""
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def format_currency(value: float) -> str:
    """Format a number as currency."""
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format a number as percentage."""
    return f"{value:.2%}"