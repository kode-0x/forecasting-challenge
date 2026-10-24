import numpy as np
import pandas as pd
import pytest

from cloud_forecasting.forecasting import (
    SUBMISSION_COLUMNS,
    make_final_forecasts,
    validate_submission,
)


def test_prediction_contract_accepts_exact_chronological_schema():
    timestamps = np.arange(24, dtype=np.int64) * 300 + 33_000
    predictions = pd.DataFrame(
        {
            "timestamp": timestamps,
            "active_vm_count": np.full(24, 10_000.0),
            "aggregate_cpu_utilization": np.full(24, 9.0),
        }
    )

    validate_submission(predictions, timestamps)
    assert list(predictions.columns) == SUBMISSION_COLUMNS


def test_prediction_contract_rejects_out_of_range_values():
    timestamps = np.arange(24, dtype=np.int64) * 300
    predictions = pd.DataFrame(
        {
            "timestamp": timestamps,
            "active_vm_count": np.full(24, 10_000.0),
            "aggregate_cpu_utilization": np.full(24, 101.0),
        }
    )

    with pytest.raises(ValueError, match=r"in \[0, 100\]"):
        validate_submission(predictions, timestamps)


def test_residual_quantiles_form_actual_value_bounds_in_correct_direction():
    training = pd.DataFrame(
        {
            "active_vm_count": np.full(110, 100.0),
            "aggregate_cpu_utilization": np.full(110, 10.0),
        }
    )
    validation = pd.DataFrame(
        {
            "target": ["aggregate_cpu_utilization"] * 4,
            "model": ["persistence"] * 4,
            "error": [-2.0, -1.0, 1.0, 2.0],
        }
    )

    _, diagnostics = make_final_forecasts(
        training,
        np.arange(24) * 300,
        {"active_vm_count": "persistence", "aggregate_cpu_utilization": "persistence"},
        validation,
    )
    cpu = diagnostics.loc[diagnostics["target"] == "aggregate_cpu_utilization"].iloc[0]

    assert cpu["approx_lower"] < cpu["prediction"] < cpu["approx_upper"]