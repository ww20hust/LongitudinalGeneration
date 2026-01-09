"""
Evaluation metrics for PEAG framework.

This module implements evaluation metrics: Pearson correlation coefficient,
Mean Absolute Error (MAE), and Mean Squared Error (MSE).

Reference: Evaluation task description
"""

import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient.
    
    Args:
        y_true: Ground-truth values of shape (n_samples, n_features) or (n_samples,)
        y_pred: Predicted values of shape (n_samples, n_features) or (n_samples,)
    
    Returns:
        Pearson correlation coefficient (scalar for 1D, mean across features for 2D)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) == 0:
        return 0.0
    
    corr, _ = pearsonr(y_true, y_pred)
    
    # Handle NaN (occurs when std is 0)
    if np.isnan(corr):
        return 0.0
    
    return corr


def mean_absolute_error_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error (MAE).
    
    Args:
        y_true: Ground-truth values of shape (n_samples, n_features) or (n_samples,)
        y_pred: Predicted values of shape (n_samples, n_features) or (n_samples,)
    
    Returns:
        Mean absolute error
    """
    return mean_absolute_error(y_true, y_pred)


def mean_squared_error_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error (MSE).
    
    Args:
        y_true: Ground-truth values of shape (n_samples, n_features) or (n_samples,)
        y_pred: Predicted values of shape (n_samples, n_features) or (n_samples,)
    
    Returns:
        Mean squared error
    """
    return mean_squared_error(y_true, y_pred)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: Ground-truth values of shape (n_samples, n_features) or (n_samples,)
        y_pred: Predicted values of shape (n_samples, n_features) or (n_samples,)
    
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        "pearson_r": pearson_correlation(y_true, y_pred),
        "mae": mean_absolute_error_metric(y_true, y_pred),
        "mse": mean_squared_error_metric(y_true, y_pred),
    }
    
    return metrics


def compute_metrics_batch(
    y_true: torch.Tensor,
    y_pred: torch.Tensor
) -> dict[str, float]:
    """
    Compute metrics on a batch of predictions.
    
    Args:
        y_true: Ground-truth values of shape (batch_size, n_features)
        y_pred: Predicted values of shape (batch_size, n_features)
    
    Returns:
        Dictionary containing all metrics
    """
    # Convert to numpy
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    
    return compute_all_metrics(y_true_np, y_pred_np)

