"""Model explainability using SHAP."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, List
import shap


def explain_lstm_model(
    model_state: Dict[str, Any],
    X_background: np.ndarray,
    X_explain: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate SHAP explanations for an LSTM model.

    Args:
        model_state: Output from train_lstm().
        X_background: Background data for SHAP (100-200 samples).
        X_explain: Data to explain. If None, uses background.
        feature_names: Names of features.

    Returns:
        Dictionary with SHAP values and explanation object.
    """
    model = model_state["model"]

    if feature_names is None:
        feature_names = [f"Day_{i}" for i in range(X_background.shape[1])]

    def predict(X: np.ndarray) -> np.ndarray:
        """Wrapper for LSTM prediction."""
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        return model.predict(X_reshaped, verbose=0).flatten()

    if X_explain is None:
        X_explain = X_background[:10]

    explainer = shap.KernelExplainer(predict, X_background[:100])
    shap_values = explainer.shap_values(X_explain)

    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "X_background": X_background,
        "X_explain": X_explain,
        "feature_names": feature_names
    }


def plot_shap_summary(shap_result: Dict[str, Any]) -> plt.Figure:
    """Create a SHAP summary plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_result["shap_values"],
        shap_result["X_explain"],
        feature_names=shap_result["feature_names"],
        show=False
    )
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    return fig


def plot_shap_waterfall(
    shap_result: Dict[str, Any],
    index: int = 0
) -> plt.Figure:
    """Create a SHAP waterfall plot for a single prediction."""
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_result["shap_values"][index],
            base_values=shap_result["explainer"].expected_value,
            data=shap_result["X_explain"][index],
            feature_names=shap_result["feature_names"]
        ),
        show=False
    )
    plt.title("SHAP Waterfall Plot - Single Prediction")
    plt.tight_layout()
    return fig