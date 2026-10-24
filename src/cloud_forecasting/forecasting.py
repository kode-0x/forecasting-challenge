"""Final fixed-origin forecasts and post-freeze holdout evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from cloud_forecasting.metrics import regression_metrics, relative_rmse_improvement
from cloud_forecasting.models import MODEL_NAMES, fixed_forecast

SUBMISSION_COLUMNS = [
    "timestamp",
    "active_vm_count",
    "aggregate_cpu_utilization",
]


def make_final_forecasts(
    training_targets: pd.DataFrame,
    holdout_timestamps: np.ndarray,
    selected_models: Mapping[str, str],
    validation_forecasts: pd.DataFrame,
    horizon: int = 24,
    moving_average_window: int = 12,
    trend_window: int = 12,
    trend_damping: float = 0.85,
    peak_quantile: float = 0.9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate forecasts using training history and holdout timestamps only."""
    timestamps = np.asarray(holdout_timestamps)
    if timestamps.ndim != 1 or len(timestamps) != horizon:
        raise ValueError(f"Expected exactly {horizon} holdout timestamps")
    if len(np.unique(timestamps)) != horizon or not np.all(np.diff(timestamps) > 0):
        raise ValueError("Holdout timestamps must be unique and chronological")
    expected_targets = {"active_vm_count", "aggregate_cpu_utilization"}
    if set(selected_models) != expected_targets:
        raise ValueError(f"selected_models must contain exactly {sorted(expected_targets)}")

    prediction_frame = pd.DataFrame({"timestamp": timestamps.astype("int64")})
    diagnostic_rows: list[dict[str, float | int | str]] = []
    for target in sorted(expected_targets):
        history = training_targets[target].to_numpy(dtype=float)
        if history.size == 0 or not np.isfinite(history).all():
            raise ValueError(f"Invalid training history for {target}")
        selected_name = selected_models[target]
        selected = _constrain(
            target,
            fixed_forecast(
                selected_name,
                history,
                horizon,
                moving_average_window,
                trend_window,
                trend_damping,
            ),
        )
        persistence = _constrain(target, fixed_forecast("persistence", history, horizon))
        moving_average = _constrain(
            target,
            fixed_forecast(
                "moving_average_12", history, horizon, moving_average_window, trend_window, trend_damping
            ),
        )
        threshold = float(np.quantile(history, peak_quantile))
        residuals = validation_forecasts.loc[
            (validation_forecasts["target"] == target)
            & (validation_forecasts["model"] == selected_name),
            "error",
        ].to_numpy(dtype=float)
        lower_residual = float(np.quantile(residuals, 0.1)) if residuals.size else 0.0
        upper_residual = float(np.quantile(residuals, 0.9)) if residuals.size else 0.0
        prediction_frame[target] = selected
        for index, timestamp in enumerate(timestamps.astype("int64")):
            diagnostic_rows.append(
                {
                    "timestamp": int(timestamp),
                    "target": target,
                    "selected_model": selected_name,
                    "prediction": float(selected[index]),
                    "persistence_prediction": float(persistence[index]),
                    "moving_average_prediction": float(moving_average[index]),
                    "training_p90_threshold": threshold,
                    "empirical_residual_p10": lower_residual,
                    "empirical_residual_p90": upper_residual,
                    "approx_lower": float(
                        _constrain(target, np.array([selected[index] - upper_residual]))[0]
                    ),
                    "approx_upper": float(
                        _constrain(target, np.array([selected[index] - lower_residual]))[0]
                    ),
                }
            )
    validate_submission(prediction_frame, timestamps, horizon)
    return prediction_frame[SUBMISSION_COLUMNS], pd.DataFrame(diagnostic_rows)


def validate_submission(
    predictions: pd.DataFrame, expected_timestamps: np.ndarray, horizon: int = 24
) -> None:
    """Raise a useful error unless predictions satisfy the submission contract."""
    if list(predictions.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"Expected columns in order: {SUBMISSION_COLUMNS}")
    if len(predictions) != horizon:
        raise ValueError(f"Expected {horizon} prediction rows, found {len(predictions)}")
    timestamps = predictions["timestamp"].to_numpy()
    if not np.array_equal(timestamps, np.asarray(expected_timestamps)):
        raise ValueError("Prediction timestamps do not exactly match the holdout window")
    if predictions["timestamp"].duplicated().any() or not predictions["timestamp"].is_monotonic_increasing:
        raise ValueError("Prediction timestamps must be unique and chronological")
    numeric = predictions[SUBMISSION_COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Predictions must contain only finite numeric values")
    if (predictions["active_vm_count"] < 0).any():
        raise ValueError("active_vm_count predictions must be nonnegative")
    if not predictions["aggregate_cpu_utilization"].between(0, 100).all():
        raise ValueError("aggregate_cpu_utilization predictions must be in [0, 100]")


def evaluate_holdout(
    diagnostic_forecasts: pd.DataFrame,
    holdout_targets: pd.DataFrame,
    training_targets: pd.DataFrame,
    peak_quantile: float = 0.9,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retrospectively score frozen forecasts; this function never feeds model selection."""
    actual_long = holdout_targets.melt(
        id_vars=["timestamp"],
        value_vars=["active_vm_count", "aggregate_cpu_utilization"],
        var_name="target",
        value_name="actual",
    )
    scored = diagnostic_forecasts.merge(
        actual_long, on=["timestamp", "target"], how="left", validate="one_to_one"
    )
    if scored["actual"].isna().any():
        raise ValueError("Holdout actuals do not align with frozen forecast timestamps")
    scored["selected_error"] = scored["prediction"] - scored["actual"]
    scored["selected_absolute_error"] = scored["selected_error"].abs()
    scored["is_peak"] = scored["actual"] >= scored["training_p90_threshold"]

    rows: list[dict[str, float | int | str | None]] = []
    for target, group in scored.groupby("target", sort=True):
        selected_name = str(group["selected_model"].iloc[0])
        training_scale = float(training_targets[target].abs().mean())
        models = {selected_name: "prediction"}
        if selected_name != "persistence":
            models["persistence"] = "persistence_prediction"
        if selected_name != "moving_average_12":
            models["moving_average_12"] = "moving_average_prediction"
        for model, column in models.items():
            metric = regression_metrics(
                group["actual"].to_numpy(),
                group[column].to_numpy(),
                training_scale,
            )
            peak = group["is_peak"].to_numpy(dtype=bool)
            absolute = np.abs(group[column].to_numpy(dtype=float) - group["actual"].to_numpy(dtype=float))
            metric.update(
                {
                    "target": str(target),
                    "model": model,
                    "normalized_mae": float(metric["mae"] / training_scale) if training_scale > 0 else None,
                    "peak_sample_count": int(peak.sum()),
                    "peak_mae": float(absolute[peak].mean()) if peak.any() else None,
                    "peak_threshold_policy": f"first-110 training {peak_quantile:.0%} quantile; ties included",
                }
            )
            rows.append(metric)
    metrics = pd.DataFrame(rows)
    persistence_rmse = metrics.loc[metrics["model"] == "persistence", ["target", "rmse"]].rename(
        columns={"rmse": "persistence_rmse"}
    )
    metrics = metrics.merge(persistence_rmse, on="target", how="left")
    metrics["relative_rmse_improvement_pct"] = metrics.apply(
        lambda row: relative_rmse_improvement(row["persistence_rmse"], row["rmse"]), axis=1
    )

    horizon_rows: list[dict[str, float | int | str]] = []
    for target, group in scored.groupby("target", sort=True):
        for step, row in enumerate(group.itertuples(index=False), start=1):
            horizon_rows.append(
                {
                    "target": str(target),
                    "horizon_step": step,
                    "timestamp": int(row.timestamp),
                    "actual": float(row.actual),
                    "predicted": float(row.prediction),
                    "error": float(row.selected_error),
                    "absolute_error": float(row.selected_absolute_error),
                }
            )
    return scored, metrics, pd.DataFrame(horizon_rows)


def _constrain(target: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if target == "active_vm_count":
        return np.maximum(array, 0)
    if target == "aggregate_cpu_utilization":
        return np.clip(array, 0, 100)
    raise ValueError(f"Unsupported target: {target}")