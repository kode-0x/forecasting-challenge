"""Descriptive workload analysis and reproducible figure generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_COLUMNS = ("active_vm_count", "aggregate_cpu_utilization")
SEGMENT_COLUMNS = ("vm_category", "virtual_core_bucket", "memory_gb_bucket")


def analyze_workload(
    readings: pd.DataFrame,
    metadata: pd.DataFrame,
    targets: pd.DataFrame,
    quality_diagnostics: dict[str, Any],
    output_dir: str | Path,
    figure_dir: str | Path,
) -> dict[str, Any]:
    """Write target, VM, and segment summaries plus descriptive figures."""
    output_path = Path(output_dir)
    figures_path = Path(figure_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    per_vm = readings.groupby("vm_id", dropna=False).agg(
        observations=("avg_cpu", "size"),
        mean_cpu=("avg_cpu", "mean"),
        std_cpu=("avg_cpu", "std"),
        median_cpu=("avg_cpu", "median"),
        p95_cpu=("avg_cpu", lambda values: values.quantile(0.95)),
        zero_reading_share=("avg_cpu", lambda values: float(values.eq(0).mean())),
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
    ).reset_index()
    clean_metadata = metadata.loc[~metadata["vm_id"].duplicated(keep=False)].copy()
    available_segments = [column for column in SEGMENT_COLUMNS if column in clean_metadata.columns]
    per_vm = per_vm.merge(
        clean_metadata[["vm_id", *available_segments]],
        on="vm_id",
        how="left",
        validate="one_to_one",
    )
    per_vm.to_csv(output_path / "vm_summary.csv", index=False)

    segment_rows: list[dict[str, Any]] = []
    for segment in available_segments:
        values = per_vm.copy()
        values[segment] = values[segment].fillna("<missing>").astype(str)
        for category, group in values.groupby(segment, sort=True):
            segment_rows.append(
                {
                    "segment": segment,
                    "category": str(category),
                    "vm_count": int(group["vm_id"].nunique()),
                    "row_count": int(group["observations"].sum()),
                    "mean_vm_cpu_mean": float(group["mean_cpu"].mean()),
                    "median_vm_cpu_mean": float(group["mean_cpu"].median()),
                    "mean_vm_p95_cpu": float(group["p95_cpu"].mean()),
                    "mean_vm_cpu_std": float(group["std_cpu"].fillna(0).mean()),
                    "mean_zero_reading_share": float(group["zero_reading_share"].mean()),
                }
            )
    segments = pd.DataFrame(segment_rows)
    segments.to_csv(output_path / "segment_summary.csv", index=False)
    targets.to_csv(output_path / "target_series.csv", index=False)

    dynamics = targets[["timestamp", *TARGET_COLUMNS]].copy()
    for target in TARGET_COLUMNS:
        dynamics[f"{target}_first_difference"] = dynamics[target].diff()
        dynamics[f"{target}_past_12_mean"] = dynamics[target].rolling(12, min_periods=1).mean()
        dynamics[f"{target}_past_12_std"] = dynamics[target].rolling(12, min_periods=2).std()
    dynamics.to_csv(output_path / "target_dynamics.csv", index=False)

    global_cpu = readings["avg_cpu"].to_numpy(dtype=float)
    target_stats = {
        target: {
            "mean": float(targets[target].mean()),
            "standard_deviation": float(targets[target].std(ddof=1)),
            "minimum": float(targets[target].min()),
            "maximum": float(targets[target].max()),
            "first_value": float(targets[target].iloc[0]),
            "last_value": float(targets[target].iloc[-1]),
            "maximum_absolute_step_change": float(targets[target].diff().abs().max()),
            "mean_absolute_step_change": float(targets[target].diff().abs().mean()),
            "minimum_timestamp_seconds": int(targets.loc[targets[target].idxmin(), "timestamp"]),
            "maximum_timestamp_seconds": int(targets.loc[targets[target].idxmax(), "timestamp"]),
        }
        for target in TARGET_COLUMNS
    }
    cpu_quantiles = {
        str(q): float(np.quantile(global_cpu, q)) for q in (0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1)
    }
    autocorrelation = {
        str(lag): _autocorrelation(targets["aggregate_cpu_utilization"].to_numpy(), lag)
        for lag in (1, 2, 3, 6, 12, 24, 48)
    }
    diagnostics: dict[str, Any] = {
        **quality_diagnostics,
        "target_series": {
            "target_statistics": target_stats,
            "avg_cpu_reading_quantiles": cpu_quantiles,
            "active_vm_count_distribution": {
                str(int(value)): int(count)
                for value, count in targets["active_vm_count"].value_counts().sort_index().items()
            },
            "aggregate_cpu_autocorrelation": autocorrelation,
            "elapsed_hours": float(
                (targets["timestamp"].iloc[-1] - targets["timestamp"].iloc[0]) / 3600
            ),
            "acf_scope": "Descriptive whole-trace statistics; not used for model selection.",
        },
        "vm_behavior": {
            "vm_summary_rows": int(len(per_vm)),
            "observed_lifetime_seconds_distribution": {
                str(int(value)): int(count)
                for value, count in (per_vm["last_timestamp"] - per_vm["first_timestamp"])
                .value_counts()
                .sort_index()
                .items()
            },
            "vm_mean_cpu_quantiles": {
                str(q): float(per_vm["mean_cpu"].quantile(q))
                for q in (0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1)
            },
            "vm_std_cpu_quantiles": {
                str(q): float(per_vm["std_cpu"].quantile(q))
                for q in (0, 0.5, 0.9, 0.95, 0.99, 1)
            },
            "vm_zero_reading_share_quantiles": {
                str(q): float(per_vm["zero_reading_share"].quantile(q))
                for q in (0, 0.5, 0.9, 0.99, 1)
            },
            "metadata_rows_excluded_from_segment_join_due_to_duplicate_key": int(
                metadata["vm_id"].duplicated(keep=False).sum()
            ),
            "lifecycle_limit": "Observed first/last readings in this complete curated trace are not infrastructure create/delete events; lifecycle timestamp units and epoch are not defined by the supplied data card.",
        },
        "segment_summary": segment_rows,
        "interpretation_limits": [
            "The trace spans only about 11.1 hours, so recurring daily or weekly seasonality cannot be established.",
            "aggregate_cpu_utilization is mean VM utilization, not total CPU demand or CPU-core-weighted utilization.",
            "The selected VM population is a curated stratified sample; observed population counts describe this challenge subset, not the full Azure fleet.",
        ],
    }
    with (output_path / "diagnostics.json").open("w", encoding="utf-8") as stream:
        json.dump(diagnostics, stream, indent=2, allow_nan=False)

    _plot_overview(targets, figures_path / "workload_overview.png")
    _plot_cpu_distribution(global_cpu, per_vm, figures_path / "cpu_distribution.png")
    if "vm_category" in per_vm:
        _plot_category_cpu(per_vm, figures_path / "vm_category_cpu.png")
    return diagnostics


def plot_validation_errors(
    horizon_metrics: pd.DataFrame, destination: str | Path
) -> None:
    """Plot validation MAE by forecast step for each target and method."""
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    colors = {"persistence": "#176b5b", "moving_average_12": "#cf5a35", "damped_trend_12": "#4267a5"}
    for axis, target in zip(axes, TARGET_COLUMNS):
        frame = horizon_metrics.loc[horizon_metrics["target"] == target]
        for model, group in frame.groupby("model", sort=False):
            axis.plot(
                group["horizon_step"],
                group["mae"],
                label=model.replace("_", " "),
                color=colors.get(model),
                linewidth=1.6,
            )
        unit = "VMs" if target == "active_vm_count" else "percentage points"
        axis.set_ylabel(f"MAE ({unit})")
        axis.set_title(target.replace("_", " "))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, ncols=3)
    axes[-1].set_xlabel("Forecast step (5-minute increments)")
    figure.suptitle("Expanding-origin validation error by forecast horizon", fontsize=13)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def plot_validation_residuals(
    forecasts: pd.DataFrame, destination: str | Path
) -> None:
    """Plot forecast error distributions and error-by-step diagnostics."""
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, target in enumerate(TARGET_COLUMNS):
        group = forecasts.loc[forecasts["target"] == target].copy()
        unit = "VMs" if target == "active_vm_count" else "percentage points"
        for model, model_group in group.groupby("model", sort=False):
            axes[row, 0].hist(
                model_group["error"], bins=20, alpha=0.42, label=model.replace("_", " ")
            )
        axes[row, 0].axvline(0, color="#263b35", linewidth=1, linestyle="--")
        axes[row, 0].set_xlabel(f"Forecast error ({unit}; prediction - actual)")
        axes[row, 0].set_ylabel("Validation forecast rows")
        axes[row, 0].set_title(f"{target.replace('_', ' ')} error distribution")
        axes[row, 0].grid(axis="y", alpha=0.2)
        axes[row, 0].legend(frameon=False, fontsize=8)

        for model, model_group in group.groupby("model", sort=False):
            axes[row, 1].scatter(
                model_group["horizon_step"],
                model_group["error"],
                s=16,
                alpha=0.45,
                label=model.replace("_", " "),
            )
        axes[row, 1].axhline(0, color="#263b35", linewidth=1, linestyle="--")
        axes[row, 1].set_xlabel("Forecast step (5-minute increments)")
        axes[row, 1].set_ylabel(f"Forecast error ({unit})")
        axes[row, 1].set_title("Residual by horizon step")
        axes[row, 1].grid(alpha=0.2)
    figure.suptitle("Expanding-origin validation residual diagnostics", fontsize=13)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def plot_holdout_forecasts(
    scored: pd.DataFrame, train_targets: pd.DataFrame, destination: str | Path
) -> None:
    """Plot frozen 24-step forecasts against holdout actuals after scoring."""
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    training_end = float(train_targets["timestamp"].iloc[-1]) / 3600
    for axis, target in zip(axes, TARGET_COLUMNS):
        group = scored.loc[scored["target"] == target].sort_values("timestamp")
        hours = group["timestamp"].to_numpy(dtype=float) / 3600
        axis.plot(hours, group["actual"], label="Observed", color="#263b35", linewidth=1.8)
        axis.plot(hours, group["prediction"], label="Frozen forecast", color="#cf5a35", linewidth=1.8)
        axis.fill_between(
            hours,
            group["approx_lower"].to_numpy(dtype=float),
            group["approx_upper"].to_numpy(dtype=float),
            color="#e1a74c",
            alpha=0.2,
            label="Validation residual p10-p90 band",
        )
        axis.axvline(training_end, color="#4267a5", linestyle="--", linewidth=1)
        axis.set_ylabel("VMs" if target == "active_vm_count" else "Mean CPU (%)")
        axis.set_title(target.replace("_", " "))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, ncols=3)
    axes[-1].set_xlabel("Trace elapsed time (hours; timestamp is seconds from trace start)")
    figure.suptitle("Final 2-hour holdout | retrospective comparison", fontsize=13)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _autocorrelation(values: np.ndarray, lag: int) -> float | None:
    if values.size <= lag or np.std(values) == 0:
        return None
    return float(np.corrcoef(values[:-lag], values[lag:])[0, 1])


def _plot_overview(targets: pd.DataFrame, destination: Path) -> None:
    hours = (targets["timestamp"] - targets["timestamp"].iloc[0]) / 3600
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].step(hours, targets["active_vm_count"], where="mid", color="#176b5b", linewidth=1.7)
    axes[0].set_ylabel("Represented VMs")
    axes[0].set_title("Active VM population")
    axes[1].plot(hours, targets["aggregate_cpu_utilization"], color="#cf5a35", linewidth=1.7)
    axes[1].set_ylabel("Mean VM CPU utilization (%)")
    axes[1].set_xlabel("Elapsed time from trace start (hours)")
    axes[1].set_title("Aggregate CPU utilization (mean across represented VMs)")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Azure VM workload | 5-minute observations", fontsize=14)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _plot_cpu_distribution(values: np.ndarray, per_vm: pd.DataFrame, destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].hist(values, bins=60, color="#cf5a35", alpha=0.88)
    axes[0].set_xlabel("VM average CPU utilization (%)")
    axes[0].set_ylabel("Five-minute VM readings")
    axes[0].set_title("Reading-level distribution")
    axes[1].hist(per_vm["mean_cpu"], bins=50, color="#176b5b", alpha=0.88)
    axes[1].set_xlabel("Per-VM mean CPU utilization (%)")
    axes[1].set_ylabel("VMs")
    axes[1].set_title("Across-VM mean distribution")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("CPU utilization is right-skewed across readings and VMs", fontsize=13)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _plot_category_cpu(per_vm: pd.DataFrame, destination: Path) -> None:
    frame = per_vm.copy()
    frame["vm_category"] = frame["vm_category"].fillna("<missing>").astype(str)
    categories = sorted(frame["vm_category"].unique())
    samples = [frame.loc[frame["vm_category"] == category, "mean_cpu"].to_numpy() for category in categories]
    labels = [f"{category}\n(n={len(sample):,})" for category, sample in zip(categories, samples)]
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.boxplot(samples, tick_labels=labels, showfliers=False, patch_artist=True,
                 boxprops={"facecolor": "#e1a74c", "alpha": 0.7},
                 medianprops={"color": "#243a36", "linewidth": 1.5})
    axis.set_ylabel("Per-VM mean CPU utilization (%)")
    axis.set_title("Per-VM utilization by documented workload category")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(destination, dpi=160)
    plt.close(figure)