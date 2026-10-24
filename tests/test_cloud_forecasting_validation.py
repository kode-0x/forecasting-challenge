import numpy as np
import pandas as pd

from cloud_forecasting.validation import evaluate_validation, select_models


def test_validation_uses_identical_fixed_horizons_for_all_models():
    targets = pd.DataFrame(
        {
            "timestamp": np.arange(110) * 300,
            "active_vm_count": np.full(110, 10_000.0),
            "aggregate_cpu_utilization": 9 + np.sin(np.arange(110) / 8),
        }
    )

    forecasts, metrics, horizon_metrics = evaluate_validation(targets)

    assert len(forecasts) == 3 * 2 * 3 * 24
    assert len(metrics) == 6
    assert len(horizon_metrics) == 2 * 3 * 24
    assert forecasts.groupby(["origin_index", "target", "model"]).size().eq(24).all()
    assert select_models(metrics)["active_vm_count"] == "persistence"
    cpu_coverage = metrics.loc[
        (metrics["target"] == "aggregate_cpu_utilization")
        & (metrics["model"] == "persistence"),
        "validation_interval_coverage",
    ].iloc[0]
    assert 0.75 <= cpu_coverage <= 0.85