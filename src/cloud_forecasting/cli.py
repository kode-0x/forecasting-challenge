"""Command-line workflows for analysis, temporal validation, and forecasts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from cloud_forecasting.analysis import (
    analyze_workload,
    plot_holdout_forecasts,
    plot_validation_errors,
    plot_validation_residuals,
)
from cloud_forecasting.config import load_config
from cloud_forecasting.data import audit_dataset, chronological_split, construct_targets, load_dataset
from cloud_forecasting.forecasting import evaluate_holdout, make_final_forecasts
from cloud_forecasting.validation import evaluate_validation, select_models

LOGGER = logging.getLogger("cloud_forecasting")


def _prepare(config: dict[str, Any]):
    readings, metadata = load_dataset(config["data"]["readings"], config["data"]["metadata"])
    quality = audit_dataset(readings, metadata, int(config["cadence_seconds"]))
    targets, target_diagnostics = construct_targets(readings, int(config["cadence_seconds"]))
    configured_train_size = config.get("train_size")
    train, holdout = chronological_split(
        targets,
        int(configured_train_size) if configured_train_size is not None else None,
        int(config["horizon"]),
    )
    quality["target_series"] = target_diagnostics
    return readings, metadata, targets, train, holdout, quality


def _validate(config: dict[str, Any], train):
    return evaluate_validation(
        train,
        origins=config["validation_origins"],
        horizon=int(config["horizon"]),
        moving_average_window=int(config["moving_average_window"]),
        trend_window=int(config["trend_window"]),
        trend_damping=float(config["trend_damping"]),
        peak_quantile=float(config["peak_quantile"]),
    )


def _analyze(config: dict[str, Any]) -> None:
    readings, metadata, targets, _, _, quality = _prepare(config)
    diagnostics = analyze_workload(
        readings,
        metadata,
        targets,
        quality,
        config["outputs"]["directory"],
        config["outputs"]["figures"],
    )
    LOGGER.info(
        "Analysis complete: %s rows, %s VMs, %s timestamps",
        diagnostics["readings"]["rows"],
        diagnostics["readings"]["unique_vms"],
        diagnostics["readings"]["unique_timestamps"],
    )


def _validate_command(config: dict[str, Any]) -> None:
    _, _, _, train, _, _ = _prepare(config)
    forecasts, metrics, horizon_metrics = _validate(config, train)
    output = Path(config["outputs"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(output / "validation_forecasts.csv", index=False)
    metrics.to_csv(output / "validation_metrics.csv", index=False)
    horizon_metrics.to_csv(output / "validation_horizon_metrics.csv", index=False)
    plot_validation_errors(horizon_metrics, config["outputs"]["figures"] / "validation_horizon_errors.png")
    plot_validation_residuals(forecasts, config["outputs"]["figures"] / "validation_residuals.png")
    selected = select_models(metrics)
    with (output / "model_selection.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "selected_models": selected,
                "selection_rule": "Minimum aggregate RMSE across the predeclared expanding-origin validation forecasts; fixed method-order tie-break.",
                "validation_origins": list(config["validation_origins"]),
                "horizon": int(config["horizon"]),
            },
            stream,
            indent=2,
        )
    LOGGER.info("Validation complete; selected models: %s", selected)


def _forecast_command(config: dict[str, Any], output_override: str | None) -> None:
    _, _, _, train, holdout, _ = _prepare(config)
    validation_forecasts, validation_metrics, horizon_metrics = _validate(config, train)
    selected = select_models(validation_metrics)
    predictions, diagnostics = make_final_forecasts(
        train,
        holdout["timestamp"].to_numpy(),
        selected,
        validation_forecasts,
        horizon=int(config["horizon"]),
        moving_average_window=int(config["moving_average_window"]),
        trend_window=int(config["trend_window"]),
        trend_damping=float(config["trend_damping"]),
        peak_quantile=float(config["peak_quantile"]),
    )
    output_path = Path(output_override).expanduser() if output_override else config["outputs"]["root_predictions"]
    if not output_path.is_absolute():
        output_path = config["_project_root"] / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False, float_format="%.10g")

    scored, holdout_metrics, horizon_holdout = evaluate_holdout(
        diagnostics, holdout, train, float(config["peak_quantile"])
    )
    outputs = Path(config["outputs"]["directory"])
    figures = Path(config["outputs"]["figures"])
    outputs.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    plot_validation_errors(horizon_metrics, figures / "validation_horizon_errors.png")
    plot_validation_residuals(validation_forecasts, figures / "validation_residuals.png")
    plot_holdout_forecasts(scored, train, figures / "holdout_forecasts.png")
    validation_forecasts.to_csv(outputs / "validation_forecasts.csv", index=False)
    validation_metrics.to_csv(outputs / "validation_metrics.csv", index=False)
    horizon_metrics.to_csv(outputs / "validation_horizon_metrics.csv", index=False)
    diagnostics_with_actual = scored.rename(columns={"prediction": "selected_prediction"})
    diagnostics_with_actual.to_csv(outputs / "holdout_forecasts.csv", index=False)
    holdout_metrics.to_csv(outputs / "holdout_metrics.csv", index=False)
    horizon_holdout.to_csv(outputs / "holdout_horizon_metrics.csv", index=False)
    with (outputs / "model_selection.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "selected_models": selected,
                "selection_rule": "Minimum aggregate RMSE across internal validation; final 24 targets were not used in selection.",
                "holdout_was_used_for": "Retrospective evaluation after predictions.csv was written.",
                "validation_origins": list(config["validation_origins"]),
            },
            stream,
            indent=2,
        )
    LOGGER.info("Wrote %s (%s rows); holdout scored after forecast freeze", output_path, len(predictions))


def main() -> None:
    parser = argparse.ArgumentParser(description="Azure VM short-horizon workload forecasting")
    parser.add_argument("command", choices=("analyze", "validate", "forecast"))
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", help="Optional output path for the required predictions.csv")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(arguments.config)
    if arguments.command == "analyze":
        _analyze(config)
    elif arguments.command == "validate":
        _validate_command(config)
    else:
        _forecast_command(config, arguments.output)


if __name__ == "__main__":
    main()