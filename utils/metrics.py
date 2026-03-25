"""
电力系统常用评价指标
Power System Evaluation Metrics

Metrics:
    - MAPE  : Mean Absolute Percentage Error
    - SMAPE : Symmetric Mean Absolute Percentage Error
    - RMSE  : Root Mean Square Error
    - MAE   : Mean Absolute Error
    - R2    : Coefficient of Determination
"""

import numpy as np
import pandas as pd
from typing import Union

ArrayLike = Union[np.ndarray, pd.Series, list]


def _to_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float)


def mape(actual: ArrayLike, predicted: ArrayLike, epsilon: float = 1e-6) -> float:
    """Mean Absolute Percentage Error (%).

    Args:
        actual: Ground-truth values.
        predicted: Predicted values.
        epsilon: Small constant to avoid division by zero.

    Returns:
        MAPE value in percentage.
    """
    y_true = _to_array(actual)
    y_pred = _to_array(predicted)
    mask = np.abs(y_true) > epsilon
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(actual: ArrayLike, predicted: ArrayLike, epsilon: float = 1e-6) -> float:
    """Symmetric Mean Absolute Percentage Error (%).

    Args:
        actual: Ground-truth values.
        predicted: Predicted values.
        epsilon: Small constant to avoid division by zero.

    Returns:
        SMAPE value in percentage.
    """
    y_true = _to_array(actual)
    y_pred = _to_array(predicted)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + epsilon
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100)


def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Root Mean Square Error.

    Args:
        actual: Ground-truth values.
        predicted: Predicted values.

    Returns:
        RMSE value.
    """
    y_true = _to_array(actual)
    y_pred = _to_array(predicted)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean Absolute Error.

    Args:
        actual: Ground-truth values.
        predicted: Predicted values.

    Returns:
        MAE value.
    """
    y_true = _to_array(actual)
    y_pred = _to_array(predicted)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Coefficient of Determination (R²).

    Args:
        actual: Ground-truth values.
        predicted: Predicted values.

    Returns:
        R² value (1.0 is perfect prediction).
    """
    y_true = _to_array(actual)
    y_pred = _to_array(predicted)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)


def evaluate_all(actual: ArrayLike, predicted: ArrayLike) -> dict:
    """Compute all evaluation metrics at once.

    Args:
        actual: Ground-truth values.
        predicted: Predicted values.

    Returns:
        Dictionary with keys: mape, smape, rmse, mae, r2.
    """
    return {
        "mape": mape(actual, predicted),
        "smape": smape(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mae": mae(actual, predicted),
        "r2": r2_score(actual, predicted),
    }
