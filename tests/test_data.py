import pandas as pd

from cloud_forecasting.data import audit_dataset, construct_targets


def test_duplicate_and_missing_readings_preserve_activity_and_not_zero_fill():
    readings = pd.DataFrame(
        [
            (0, "vm-a", 10.0),
            (0, "vm-a", 20.0),
            (0, "vm-b", 30.0),
            (300, "vm-a", float("nan")),
        ],
        columns=["timestamp", "vm_id", "avg_cpu"],
    )
    targets, diagnostics = construct_targets(readings)

    assert targets.loc[0, "active_vm_count"] == 2
    assert targets.loc[0, "aggregate_cpu_utilization"] == 22.5
    assert targets.loc[1, "active_vm_count"] == 1
    assert pd.isna(targets.loc[1, "aggregate_cpu_utilization"])
    assert diagnostics["duplicate_source_rows_collapsed"] == 1


def test_quality_audit_counts_duplicates_and_metadata_mismatch():
    readings = pd.DataFrame(
        [(0, "vm-a", 1.0, 1.0, 1.0), (0, "vm-a", 2.0, 2.0, 2.0)],
        columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"],
    )
    metadata = pd.DataFrame({"vm_id": ["vm-b"]})

    diagnostics = audit_dataset(readings, metadata)

    assert diagnostics["readings"]["duplicate_vm_timestamp_rows"] == 2
    assert diagnostics["metadata"]["reading_ids_without_metadata"] == 1
    assert diagnostics["metadata"]["metadata_ids_without_readings"] == 1