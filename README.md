# Portfolio Optimization for GMF Investments

## Business Problem

GMF Investments, a financial advisory firm, needed a data-driven approach to optimize client portfolios. Traditional methods rely on historical averages, which fail to account for changing market conditions. This project combines time series forecasting (LSTM) with Modern Portfolio Theory to recommend optimal asset allocations.

## Solution Overview

The system delivers:

- **Forecasting:** LSTM model predicts TSLA stock prices (RMSE: 19.92)
- **Optimization:** Efficient Frontier finds optimal portfolio weights
- **Backtesting:** Strategy validation against 60/40 benchmark
- **Dashboard:** Interactive Streamlit app for stakeholders

## Key Results

| Metric | Value |
| -------- | ------- |
| LSTM Forecast RMSE | 19.92 |
| Optimal Sharpe Ratio | 0.709 |
| Backtest Total Return | 3.94% |

## Quick Start

```bash
git clone https://github.com/makda81/portfolio-optimization.git
cd portfolio-optimization
pip install -r requirements.txt
streamlit run dashboard/app.py
