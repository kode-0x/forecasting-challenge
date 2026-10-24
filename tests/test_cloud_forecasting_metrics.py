import numpy as np

from cloud_forecasting.metrics import regression_metrics, relative_rmse_improvement


def test_metrics_and_zero_baseline_behavior():
    metrics = regression_metrics(np.array([1.0, 3.0]), np.array([2.0, 3.0]), 2.0)

    assert metrics["mae"] == 0.5
    assert metrics["rmse"] == np.sqrt(0.5)
    assert metrics["normalized_mae"] == 0.25
    assert relative_rmse_improvement(0.0, 1.0) is None
    assert relative_rmse_improvement(2.0, 1.5) == 25.0