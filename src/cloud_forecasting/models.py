"""Small, deterministic fixed-origin forecasting candidates."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from cloud_forecasting.baselines import moving_average_forecast, persistence_forecast

ForecastFunction = Callable[..., np.ndarray]


def damped_trend_forecast(
    history: np.ndarray,
    horizon: int,
    window: int = 12,
    damping: float = 0.85,
) -> np.ndarray:
    """Extrapolate a recent linear trend with geometrically damped increments."""
    values = np.asarray(history, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("history must be a non-empty finite one-dimensional series")
    if horizon < 1 or window < 2:
        raise ValueError("horizon must be positive and trend window must be at least two")
    if not 0 <= damping <= 1:
        raise ValueError("damping must be between zero and one")
    recent = values[-window:]
    slope = float(np.polyfit(np.arange(recent.size), recent, deg=1)[0])
    steps = np.arange(1, horizon + 1, dtype=float)
    damped_steps = (1 - damping**steps) / (1 - damping) if damping < 1 else steps
    return values[-1] + slope * damped_steps


def fixed_forecast(
    name: str,
    history: np.ndarray,
    horizon: int,
    moving_average_window: int = 12,
    trend_window: int = 12,
    trend_damping: float = 0.85,
) -> np.ndarray:
    """Dispatch one predeclared method and return a finite forecast vector."""
    methods: dict[str, ForecastFunction] = {
        "persistence": persistence_forecast,
        "moving_average_12": lambda values, steps: moving_average_forecast(
            values, steps, window=moving_average_window
        ),
        "damped_trend_12": lambda values, steps: damped_trend_forecast(
            values, steps, window=trend_window, damping=trend_damping
        ),
    }
    try:
        prediction = methods[name](history, horizon)
    except KeyError as exc:
        raise ValueError(f"Unknown forecasting method: {name}") from exc
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != (horizon,) or not np.isfinite(prediction).all():
        raise ValueError(f"{name} returned an invalid forecast")
    return prediction


MODEL_NAMES = ("persistence", "moving_average_12", "damped_trend_12")