"""Expanding-origin validation with fixed multi-step forecasts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from cloud_forecasting.metrics import regression_metrics, relative_rmse_improvement
from cloud_forecasting.models import MODEL_NAMES, fixed_forecast

TARGETS = ("active_vm_count", "aggregate_cpu_utilization")


def evaluate_validation(
    training_targets: pd.DataFrame,
    origins: Sequence[int] = (62, 74, 86),
    horizon: int = 24,
    moving_average_window: int = 12,
    trend_window: int = 12,
    trend_damping: float = 0.85,
    peak_quantile: float = 0.9,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate all methods at identical expanding origins and full horizons.

    Origins that do not fit inside ``training_targets`` (insufficient history, or
    the validation window would overrun the training data) are skipped so the
    pipeline also runs on a short swapped-in dataset. When no configured origin
    fits, validation is undefined, so :func:`select_models` falls back to the
    leakage-free ``persistence`` method for every target (see that function).
    """
    if not 0 < peak_quantile < 1:
        raise ValueError("peak_quantile must be strictly between zero and one")
    minimum_history = max(moving_average_window, trend_window, 2)
    usable_origins = [
        int(origin)
        for origin in origins
        if origin >= minimum_history and origin + horizon <= len(training_targets)
    ]
    if not usable_origins:
        return _empty_validation_outputs()
    rows: list[dict[str, float | int | str]] = []
    for origin in usable_origins:
        history_frame = training_targets.iloc[:origin]
        validation_frame = training_targets.iloc[origin : origin + horizon]
        origin_timestamp = int(history_frame["timestamp"].iloc[-1])
        for target in TARGETS:
            history = history_frame[target].to_numpy(dtype=float)
            actual = validation_frame[target].to_numpy(dtype=float)
            if not np.isfinite(history).all() or not np.isfinite(actual).all():
                raise ValueError(f"Non-finite values in validation series for {target}")
            training_scale = float(np.mean(np.abs(history)))
            peak_threshold = float(np.quantile(history, peak_quantile))
            for model_name in MODEL_NAMES:
                prediction = _constrain(
                    target,
                    fixed_forecast(
                        model_name,
                        history,
                        horizon,
                        moving_average_window,
                        trend_window,
                        trend_damping,
                    ),
                )
                for step, (timestamp, actual_value, predicted_value) in enumerate(
                    zip(validation_frame["timestamp"], actual, prediction), start=1
                ):
                    rows.append(
                        {
                            "origin_index": int(origin),
                            "origin_timestamp": origin_timestamp,
                            "timestamp": int(timestamp),
                            "target": target,
                            "model": model_name,
                            "horizon_step": step,
                            "actual": float(actual_value),
                            "predicted": float(predicted_value),
                            "training_scale": training_scale,
                            "peak_threshold": peak_threshold,
                        }
                    )

    forecasts = pd.DataFrame(rows)
    forecasts["error"] = forecasts["predicted"] - forecasts["actual"]
    forecasts["absolute_error"] = forecasts["error"].abs()
    forecasts["is_peak"] = forecasts["actual"] >= forecasts["peak_threshold"]

    metric_rows: list[dict[str, float | int | str | None]] = []
    for (target, model), group in forecasts.groupby(["target", "model"], sort=True):
        metric = regression_metrics(
            group["actual"].to_numpy(),
            group["predicted"].to_numpy(),
            float(group["training_scale"].mean()),
            peak_threshold=None,
        )
        peak_errors = group.loc[group["is_peak"], "absolute_error"]
        residual_low, residual_high = np.quantile(group["error"], [0.1, 0.9])
        residual_coverage = np.mean(
            (group["actual"] >= group["predicted"] - residual_high)
            & (group["actual"] <= group["predicted"] - residual_low)
        )
        metric_rows.append(
            {
                "target": target,
                "model": model,
                **metric,
                "normalized_mae": float(
                    np.mean(group["absolute_error"].to_numpy() / group["training_scale"].to_numpy())
                ),
                "peak_threshold_policy": f"per-origin training {peak_quantile:.0%} quantile; ties included",
                "peak_sample_count": int(len(peak_errors)),
                "peak_mae": float(peak_errors.mean()) if not peak_errors.empty else None,
                "validation_residual_p10": float(residual_low),
                "validation_residual_p90": float(residual_high),
                "validation_interval_coverage": float(residual_coverage),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    persistence = metrics.loc[metrics["model"] == "persistence", ["target", "rmse"]].rename(
        columns={"rmse": "persistence_rmse"}
    )
    metrics = metrics.merge(persistence, on="target", how="left")
    metrics["relative_rmse_improvement_pct"] = metrics.apply(
        lambda row: relative_rmse_improvement(row["persistence_rmse"], row["rmse"]), axis=1
    )

    horizon_rows: list[dict[str, float | int | str | None]] = []
    for (target, model, step), group in forecasts.groupby(
        ["target", "model", "horizon_step"], sort=True
    ):
        horizon_rows.append(
            {
                "target": target,
                "model": model,
                "horizon_step": int(step),
                "n": int(len(group)),
                "mae": float(group["absolute_error"].mean()),
                "rmse": float(np.sqrt(np.mean(group["error"] ** 2))),
                "mean_error": float(group["error"].mean()),
            }
        )
    return forecasts, metrics, pd.DataFrame(horizon_rows)


def select_models(metrics: pd.DataFrame) -> dict[str, str]:
    """Choose each target's minimum-validation-RMSE method, with fixed tie-breaks.

    When ``metrics`` is empty (no validation origin fit the series, e.g. a tiny
    smoke dataset) fall back to the leakage-free ``persistence`` method for every
    target rather than raising.
    """
    if metrics.empty:
        return {target: "persistence" for target in TARGETS}
    selected: dict[str, str] = {}
    for target, group in metrics.groupby("target", sort=True):
        ordered = group.assign(
            _tie=group["model"].map({name: index for index, name in enumerate(MODEL_NAMES)})
        ).sort_values(["rmse", "_tie"], kind="stable")
        selected[str(target)] = str(ordered.iloc[0]["model"])
    return selected


def _empty_validation_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return empty forecast/metric/horizon frames when no origin fits the series."""
    forecasts = pd.DataFrame(
        columns=[
            "origin_index",
            "origin_timestamp",
            "timestamp",
            "target",
            "model",
            "horizon_step",
            "actual",
            "predicted",
            "training_scale",
            "peak_threshold",
            "error",
            "absolute_error",
            "is_peak",
        ]
    )
    metrics = pd.DataFrame(columns=["target", "model", "rmse"])
    horizon_metrics = pd.DataFrame(
        columns=["target", "model", "horizon_step", "n", "mae", "rmse", "mean_error"]
    )
    return forecasts, metrics, horizon_metrics


def _constrain(target: str, prediction: np.ndarray) -> np.ndarray:
    if target == "active_vm_count":
        return np.maximum(prediction, 0)
    if target == "aggregate_cpu_utilization":
        return np.clip(prediction, 0, 100)
    raise ValueError(f"Unsupported target: {target}")