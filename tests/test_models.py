"""Unit tests for models module."""

import pytest
import pandas as pd
import numpy as np
from src.models import LSTMConfig


def test_lstm_config():
    """Test LSTM configuration defaults."""
    config = LSTMConfig()
    assert config.window_size == 60
    assert config.epochs == 20
    assert config.batch_size == 32
    assert len(config.lstm_units) == 2


def test_lstm_config_custom():
    """Test custom LSTM configuration."""
    config = LSTMConfig(
        window_size=30,
        epochs=10,
        lstm_units=(64, 64, 32)
    )
    assert config.window_size == 30
    assert config.epochs == 10
    assert len(config.lstm_units) == 3