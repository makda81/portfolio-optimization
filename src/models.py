"""Time series forecasting models."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler


@dataclass
class LSTMConfig:
    """Configuration for LSTM model."""
    window_size: int = 60
    epochs: int = 20
    batch_size: int = 32
    lstm_units: Tuple[int, ...] = (50, 50)
    dropout_rate: float = 0.2


def train_lstm(
    train_data: pd.Series,
    config: Optional[LSTMConfig] = None
) -> Dict[str, Any]:
    """
    Train an LSTM model for time series forecasting.

    Args:
        train_data: Training data (single time series).
        config: Model configuration.

    Returns:
        Dictionary with trained model, history, and scaler.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    if config is None:
        config = LSTMConfig()

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(train_data.values.reshape(-1, 1))

    # Create sequences
    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(scaled_data, config.window_size)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # Build model
    model = Sequential()
    for i, units in enumerate(config.lstm_units):
        return_seq = i < len(config.lstm_units) - 1
        model.add(LSTM(units, return_sequences=return_seq, input_shape=(config.window_size, 1)))
        model.add(Dropout(config.dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=0.1,
        verbose=0
    )

    return {
        "model": model,
        "history": history,
        "scaler": scaler,
        "config": config
    }


def forecast_lstm(
    model_state: Dict[str, Any],
    last_sequence: np.ndarray,
    steps: int = 126
) -> np.ndarray:
    """
    Generate multi-step forecasts using an LSTM model.

    Args:
        model_state: Output from train_lstm().
        last_sequence: The last window_size days of data (scaled).
        steps: Number of days to forecast.

    Returns:
        Array of forecasted prices.
    """
    model = model_state["model"]
    scaler = model_state["scaler"]
    window_size = model_state["config"].window_size

    current_sequence = last_sequence.reshape(1, window_size, 1)
    predictions_scaled = []

    for _ in range(steps):
        next_pred_scaled = model.predict(current_sequence, verbose=0)
        predictions_scaled.append(next_pred_scaled[0, 0])
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = next_pred_scaled[0, 0]

    return scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()