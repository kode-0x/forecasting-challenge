"""Data loading, quality diagnostics, and target construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


READING_COLUMNS = ["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"]
CPU_COLUMNS = ["min_cpu", "max_cpu", "avg_cpu"]


def load_dataset(
    readings_path: str | Path, metadata_path: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two challenge CSVs without modifying the source files."""
    readings = pd.read_csv(
        readings_path,
        compression="infer",
        dtype={"vm_id": "string"},
    )
    metadata = pd.read_csv(
        metadata_path,
        compression="infer",
        dtype={"vm_id": "string"},
    )
    missing_reading_columns = set(READING_COLUMNS) - set(readings.columns)
    if missing_reading_columns:
        raise ValueError(f"Readings file is missing columns: {sorted(missing_reading_columns)}")
    if "vm_id" not in metadata.columns:
        raise ValueError("Metadata file is missing required column: vm_id")
    readings["timestamp"] = pd.to_numeric(readings["timestamp"], errors="coerce")
    for column in CPU_COLUMNS:
        readings[column] = pd.to_numeric(readings[column], errors="coerce")
    return readings, metadata


def audit_dataset(
    readings: pd.DataFrame, metadata: pd.DataFrame, cadence_seconds: int = 300
) -> dict[str, Any]:
    """Summarize schema, coverage, missingness, duplicates, and metadata integrity."""
    if readings.empty:
        raise ValueError("The CPU readings file contains no records")
    numeric_timestamps = readings["timestamp"].dropna().astype("int64")
    if numeric_timestamps.empty:
        raise ValueError("No valid timestamps were found")

    timestamp_values = np.sort(numeric_timestamps.unique())
    expected = np.arange(
        timestamp_values[0], timestamp_values[-1] + cadence_seconds, cadence_seconds
    )
    per_vm_counts = readings.groupby("vm_id", dropna=False).size()
    timestamp_counts = readings.groupby("timestamp").size()
    duplicate_mask = readings.duplicated(["vm_id", "timestamp"], keep=False)
    metadata_ids = set(metadata["vm_id"].dropna().astype(str))
    reading_ids = set(readings["vm_id"].dropna().astype(str))
    cpu_quality: dict[str, dict[str, int | float | None]] = {}
    for column in CPU_COLUMNS:
        values = readings[column]
        finite = values.notna() & np.isfinite(values)
        in_range = finite & values.between(0, 100)
        cpu_quality[column] = {
            "missing": int(values.isna().sum()),
            "non_finite": int((values.notna() & ~np.isfinite(values)).sum()),
            "out_of_range": int((finite & ~values.between(0, 100)).sum()),
            "minimum": float(values[finite].min()) if finite.any() else None,
            "maximum": float(values[finite].max()) if finite.any() else None,
            "valid_for_range": int(in_range.sum()),
        }

    interval_counts = pd.Series(np.diff(timestamp_values)).value_counts().sort_index()
    metadata_duplicates = int(metadata["vm_id"].duplicated(keep=False).sum())
    diagnostics: dict[str, Any] = {
        "readings": {
            "rows": int(len(readings)),
            "columns": list(readings.columns),
            "column_dtypes_after_numeric_parsing": {
                str(column): str(dtype) for column, dtype in readings.dtypes.items()
            },
            "unique_vms": int(readings["vm_id"].nunique(dropna=True)),
            "unique_timestamps": int(len(timestamp_values)),
            "source_rows_chronological_by_timestamp": bool(
                readings["timestamp"].dropna().is_monotonic_increasing
            ),
            "timestamp_min_seconds": int(timestamp_values[0]),
            "timestamp_max_seconds": int(timestamp_values[-1]),
            "observed_interval_counts_seconds": {
                str(int(interval)): int(count) for interval, count in interval_counts.items()
            },
            "expected_cadence_seconds": cadence_seconds,
            "missing_cadence_timestamps": int(len(set(expected) - set(timestamp_values))),
            "timestamps_off_cadence": int(
                np.sum((timestamp_values - timestamp_values[0]) % cadence_seconds != 0)
            ),
            "rows_per_vm_distribution": {
                str(int(count)): int(freq)
                for count, freq in per_vm_counts.value_counts().sort_index().items()
            },
            "timestamp_rows_distribution": {
                str(int(count)): int(freq)
                for count, freq in timestamp_counts.value_counts().sort_index().items()
            },
            "duplicate_vm_timestamp_rows": int(duplicate_mask.sum()),
            "duplicate_vm_timestamp_keys": int(
                readings.loc[duplicate_mask, ["vm_id", "timestamp"]].drop_duplicates().shape[0]
            ),
            "cpu_columns": cpu_quality,
            "vm_observations_share_timestamp_index": bool(
                per_vm_counts.nunique() == 1
                and per_vm_counts.iloc[0] == len(timestamp_values)
                and len(timestamp_counts) == len(timestamp_values)
                and timestamp_counts.nunique() == 1
            ),
        },
        "metadata": {
            "rows": int(len(metadata)),
            "columns": list(metadata.columns),
            "duplicate_vm_id_rows": metadata_duplicates,
            "reading_ids_without_metadata": len(reading_ids - metadata_ids),
            "metadata_ids_without_readings": len(metadata_ids - reading_ids),
            "missing_values": {
                str(column): int(count)
                for column, count in metadata.isna().sum().items()
                if count
            },
            "category_counts": (
                metadata["vm_category"].fillna("<missing>").astype(str).value_counts().to_dict()
                if "vm_category" in metadata.columns
                else {}
            ),
        },
        "policies": {
            "activity": "A VM is represented/active at a timestamp when at least one row exists for its vm_id and timestamp, independent of CPU value.",
            "duplicate_readings": "Average valid avg_cpu values within each vm_id/timestamp before timestamp-level aggregation; count each VM once for active_vm_count.",
            "invalid_cpu": "Missing, non-finite, or out-of-range CPU readings are excluded from CPU means and are never converted to zero; their source rows still represent VM activity.",
            "timestamp_gaps": "Create the complete fixed-cadence index and leave targets missing for timestamps with no represented observations; do not interpolate or forward-fill.",
        },
    }
    return diagnostics


def construct_targets(
    readings: pd.DataFrame, cadence_seconds: int = 300
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate VM readings onto the complete chronological challenge index."""
    valid_timestamp = readings["timestamp"].notna()
    if not valid_timestamp.any():
        raise ValueError("Cannot construct targets without valid timestamps")
    frame = readings.loc[valid_timestamp, ["timestamp", "vm_id", "avg_cpu"]].copy()
    frame["timestamp"] = frame["timestamp"].astype("int64")
    frame["_valid_avg_cpu"] = frame["avg_cpu"].where(
        np.isfinite(frame["avg_cpu"]) & frame["avg_cpu"].between(0, 100)
    )

    vm_timestamp = frame.groupby(["timestamp", "vm_id"], as_index=False, dropna=False).agg(
        vm_avg_cpu=("_valid_avg_cpu", "mean"),
        source_rows=("vm_id", "size"),
    )
    aggregated = vm_timestamp.groupby("timestamp").agg(
        active_vm_count=("vm_id", "nunique"),
        aggregate_cpu_utilization=("vm_avg_cpu", "mean"),
        valid_cpu_vm_count=("vm_avg_cpu", "count"),
        represented_vm_count=("vm_id", "nunique"),
        source_row_count=("source_rows", "sum"),
    )
    start = int(frame["timestamp"].min())
    end = int(frame["timestamp"].max())
    if (end - start) % cadence_seconds:
        raise ValueError(
            f"Timestamp range {start}..{end} is not divisible by cadence {cadence_seconds}"
        )
    index = pd.Index(
        np.arange(start, end + cadence_seconds, cadence_seconds, dtype="int64"),
        name="timestamp",
    )
    targets = aggregated.reindex(index)
    count_columns = ["valid_cpu_vm_count", "represented_vm_count", "source_row_count"]
    targets[count_columns] = targets[count_columns].fillna(0).astype("int64")
    targets = targets.reset_index()
    if targets["timestamp"].duplicated().any() or not targets["timestamp"].is_monotonic_increasing:
        raise AssertionError("Target timestamps must be unique and chronological")
    if len(targets) > 1 and not (targets["timestamp"].diff().dropna() == cadence_seconds).all():
        raise AssertionError("Target index does not match the configured cadence")
    valid_cpu = targets["aggregate_cpu_utilization"].dropna()
    if not valid_cpu.between(0, 100).all():
        raise AssertionError("Aggregate CPU utilization must remain within [0, 100]")
    target_diagnostics = {
        "canonical_timestamp_count": int(len(targets)),
        "timestamp_min_seconds": int(start),
        "timestamp_max_seconds": int(end),
        "cadence_seconds": cadence_seconds,
        "timestamps_without_rows": int((targets["represented_vm_count"] == 0).sum()),
        "timestamps_without_valid_cpu": int((targets["valid_cpu_vm_count"] == 0).sum()),
        "duplicate_source_rows_collapsed": int(len(frame) - len(vm_timestamp)),
        "active_count_is_integer_valued": bool(
            targets["active_vm_count"].dropna().mod(1).eq(0).all()
        ),
    }
    return targets, target_diagnostics


def chronological_split(
    targets: pd.DataFrame, train_size: int | None = 110, horizon: int = 24
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a fully constructed series into the public train and holdout windows.

    The holdout is always the final ``horizon`` observations (the contract's
    public holdout). ``train_size`` is optional: when it is ``None`` it is
    derived as ``len(targets) - horizon`` so a swapped-in dataset of any length
    (for example a reviewer's smoke dataset) splits correctly. When supplied it
    must equal ``len(targets) - horizon``.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if train_size is None:
        train_size = len(targets) - horizon
    if train_size < 1:
        raise ValueError(
            f"Series of length {len(targets)} is too short for a {horizon}-step holdout"
        )
    if len(targets) != train_size + horizon:
        raise ValueError(
            f"Expected {train_size + horizon} canonical timestamps, found {len(targets)}"
        )
    if not targets["timestamp"].is_monotonic_increasing:
        raise ValueError("Targets must be sorted before the chronological split")
    return targets.iloc[:train_size].copy(), targets.iloc[train_size:].copy()