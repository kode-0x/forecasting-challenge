"""Fixed-origin, multi-step baseline forecast methods."""

from __future__ import annotations

import numpy as np


def persistence_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observation across the full forecast horizon."""
    values = _history(history, horizon)
    return np.full(horizon, values[-1], dtype=float)


def moving_average_forecast(
    history: np.ndarray, horizon: int, window: int = 12
) -> np.ndarray:
    """Repeat a fixed trailing-window average without using future observations."""
    values = _history(history, horizon)
    if window < 1:
        raise ValueError("window must be positive")
    return np.full(horizon, values[-window:].mean(), dtype=float)


def _history(history: np.ndarray, horizon: int) -> np.ndarray:
    values = np.asarray(history, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("history must be a non-empty one-dimensional series")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not np.isfinite(values).all():
        raise ValueError("history must contain only finite values")
    return values