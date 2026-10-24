import pandas as pd
import pytest

from cloud_forecasting.data import chronological_split, construct_targets


def test_complete_targets_and_chronological_split():
    readings = pd.DataFrame(
        [(timestamp, "vm-a", float(timestamp % 100)) for timestamp in range(0, 40200, 300)],
        columns=["timestamp", "vm_id", "avg_cpu"],
    )
    targets, _ = construct_targets(readings)
    train, holdout = chronological_split(targets)

    assert len(targets) == 134
    assert len(train) == 110
    assert len(holdout) == 24
    assert train["timestamp"].iloc[-1] == 32700
    assert holdout["timestamp"].iloc[0] == 33000
    assert targets["timestamp"].diff().dropna().eq(300).all()


def test_split_rejects_unexpected_series_length():
    targets = pd.DataFrame({"timestamp": [0, 300]})
    with pytest.raises(ValueError, match="Expected 134"):
        chronological_split(targets)


def test_timestamp_gap_is_preserved_with_missing_targets_not_zero_filled():
    readings = pd.DataFrame(
        [(0, "vm-a", 10.0), (600, "vm-a", 20.0)],
        columns=["timestamp", "vm_id", "avg_cpu"],
    )

    targets, diagnostics = construct_targets(readings)

    assert targets["timestamp"].tolist() == [0, 300, 600]
    assert pd.isna(targets.loc[1, "active_vm_count"])
    assert pd.isna(targets.loc[1, "aggregate_cpu_utilization"])
    assert targets.loc[1, "represented_vm_count"] == 0
    assert diagnostics["timestamps_without_rows"] == 1