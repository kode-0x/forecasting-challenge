"""Consistent error metrics for temporal validation and holdout diagnostics."""

from __future__ import annotations

import numpy as np


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    training_scale: float,
    peak_threshold: float | None = None,
) -> dict[str, float | int | None]:
    """Compute MAE, RMSE, fold-scaled MAE, bias, and optional peak errors."""
    truth = np.asarray(actual, dtype=float)
    forecast = np.asarray(predicted, dtype=float)
    if truth.shape != forecast.shape or truth.size == 0:
        raise ValueError("actual and predicted must have the same non-empty shape")
    if not np.isfinite(truth).all() or not np.isfinite(forecast).all():
        raise ValueError("metric inputs must be finite")
    errors = forecast - truth
    absolute = np.abs(errors)
    scale = float(training_scale)
    normalized_mae = float(absolute.mean() / scale) if np.isfinite(scale) and scale > 0 else None
    peak_mask = truth >= peak_threshold if peak_threshold is not None else np.zeros_like(truth, dtype=bool)
    return {
        "n": int(truth.size),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "normalized_mae": normalized_mae,
        "mean_error": float(errors.mean()),
        "peak_threshold": float(peak_threshold) if peak_threshold is not None else None,
        "peak_sample_count": int(peak_mask.sum()),
        "peak_mae": float(absolute[peak_mask].mean()) if peak_mask.any() else None,
    }


def relative_rmse_improvement(baseline_rmse: float, model_rmse: float) -> float | None:
    """Return percent RMSE reduction; undefined when the baseline error is zero."""
    if baseline_rmse == 0:
        return None
    return 100 * (baseline_rmse - model_rmse) / baseline_rmse