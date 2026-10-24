import numpy as np

from cloud_forecasting.models import fixed_forecast


def test_fixed_origin_forecast_depends_only_on_history_not_future_targets():
    history = np.array([2.0, 3.0, 5.0, 4.0])
    first_unobserved_future = np.array([50.0, 60.0, 70.0])
    unrelated_future = np.array([-500.0, 0.0, 500.0])

    forecast_a = fixed_forecast("persistence", history, len(first_unobserved_future))
    forecast_b = fixed_forecast("persistence", history, len(unrelated_future))

    np.testing.assert_array_equal(forecast_a, [4.0, 4.0, 4.0])
    np.testing.assert_array_equal(forecast_b, [4.0, 4.0, 4.0])


def test_moving_average_is_fixed_for_the_entire_horizon():
    forecast = fixed_forecast("moving_average_12", np.arange(1.0, 13.0), 24)
    assert forecast.shape == (24,)
    assert np.all(forecast == 6.5)