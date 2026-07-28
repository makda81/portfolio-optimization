"""Interactive dashboard for portfolio optimization results."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Portfolio Optimization Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Portfolio Optimization Dashboard")
st.markdown("### Interactive exploration of TSLA, SPY, and BND portfolio analysis")

# Load data
@st.cache_data
def load_data():
    """Load processed data and ensure numeric columns."""
    try:
        tsla = pd.read_csv("data/processed/tsla_processed.csv", index_col=0, parse_dates=True)
        spy = pd.read_csv("data/processed/spy_processed.csv", index_col=0, parse_dates=True)
        bnd = pd.read_csv("data/processed/bnd_processed.csv", index_col=0, parse_dates=True)
        
        # Convert 'Close' to numeric (coerce errors to NaN)
        for df in [tsla, spy, bnd]:
            if 'Close' in df.columns:
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            # Drop rows with NaN in 'Close'
            df.dropna(subset=['Close'], inplace=True)
        
        return tsla, spy, bnd
    except FileNotFoundError:
        # Use sample data if files not found
        dates = pd.date_range("2015-01-01", "2024-12-31", freq="D")
        tsla = pd.DataFrame({"Close": 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)}, index=dates)
        spy = pd.DataFrame({"Close": 200 + np.cumsum(np.random.randn(len(dates)) * 0.3)}, index=dates)
        bnd = pd.DataFrame({"Close": 80 + np.cumsum(np.random.randn(len(dates)) * 0.1)}, index=dates)
        return tsla, spy, bnd

tsla, spy, bnd = load_data()

# Combine into one DataFrame
prices = pd.DataFrame({
    "TSLA": tsla["Close"],
    "SPY": spy["Close"],
    "BND": bnd["Close"]
}).dropna()

# Ensure all columns are numeric
for col in prices.columns:
    prices[col] = pd.to_numeric(prices[col], errors='coerce')
prices.dropna(inplace=True)

# ============================================================
# Metric cards with safe calculation
# ============================================================
col1, col2, col3 = st.columns(3)

def safe_pct_change(series):
    """Calculate 1‑year percentage change safely."""
    if len(series) < 252:
        return 0.0
    latest = series.iloc[-1]
    year_ago = series.iloc[-252]
    if year_ago == 0:
        return 0.0
    return ((latest / year_ago) - 1) * 100

with col1:
    latest_tsla = prices['TSLA'].iloc[-1]
    pct_tsla = safe_pct_change(prices['TSLA'])
    st.metric("📊 TSLA", f"${latest_tsla:.2f}", delta=f"{pct_tsla:.2f}%")

with col2:
    latest_spy = prices['SPY'].iloc[-1]
    pct_spy = safe_pct_change(prices['SPY'])
    st.metric("📊 SPY", f"${latest_spy:.2f}", delta=f"{pct_spy:.2f}%")

with col3:
    latest_bnd = prices['BND'].iloc[-1]
    pct_bnd = safe_pct_change(prices['BND'])
    st.metric("📊 BND", f"${latest_bnd:.2f}", delta=f"{pct_bnd:.2f}%")

# Price chart
st.markdown("---")
st.subheader("Historical Prices")

fig = go.Figure()
for col in prices.columns:
    fig.add_trace(go.Scatter(
        x=prices.index,
        y=prices[col],
        name=col,
        mode="lines"
    ))
fig.update_layout(
    height=400,
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit | Data: Yahoo Finance")