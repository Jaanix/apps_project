"""
Unit tests for forecasting and threshold calibration (src/forecast.py).
"""

import pytest
import numpy as np
from src.forecast import forecast_intensity, calibrate_threshold, should_alert

def test_forecast_intensity_horizons():
    params = {"mu": 1.0, "alpha": 0.5, "beta": 1.0}
    event_times = np.array([10.0, 10.2, 10.4, 10.5])

    # 2h forecast
    fc_2h = forecast_intensity(params, event_times, forecast_horizon_hours=2.0, num_points=50)
    assert len(fc_2h["intensities"]) == 50
    assert fc_2h["time_grid"][0] == 10.5
    assert np.isclose(fc_2h["time_grid"][-1], 12.5)
    assert fc_2h["peak_intensity"] >= params["mu"]

    # 4h forecast
    fc_4h = forecast_intensity(params, event_times, forecast_horizon_hours=4.0, num_points=50)
    assert np.isclose(fc_4h["time_grid"][-1], 14.5)

def test_calibrate_threshold():
    params = {"mu": 1.0, "alpha": 0.5, "beta": 1.0}
    event_times = np.linspace(0, 100, 200)

    th_95 = calibrate_threshold(params, event_times, percentile=95.0)
    th_50 = calibrate_threshold(params, event_times, percentile=50.0)

    assert th_95 > th_50
    assert th_95 > params["mu"]

def test_should_alert():
    assert should_alert({"peak_intensity": 15.0}, threshold=10.0) is True
    assert should_alert({"peak_intensity": 5.0}, threshold=10.0) is False
    assert should_alert(np.array([2.0, 8.0, 12.0]), threshold=10.0) is True
