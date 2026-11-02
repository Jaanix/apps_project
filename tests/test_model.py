"""
Unit tests for Hawkes MLE fitting and stationarity (src/model.py).
"""

import pytest
import numpy as np
from src.model import fit_hawkes, compute_conditional_intensity, hawkes_neg_log_likelihood

def test_fit_hawkes_synthetic():
    np.random.seed(123)
    # Generate synthetic arrival times
    event_times = np.sort(np.cumsum(np.random.exponential(scale=0.2, size=150)))
    end_time = float(event_times[-1] + 1.0)

    res = fit_hawkes(event_times, end_time)

    assert "mu" in res
    assert "alpha" in res
    assert "beta" in res
    assert "branching_ratio" in res
    assert "log_likelihood" in res

    # Strict Stationarity Verification: alpha must be strictly less than beta
    assert res["alpha"] < res["beta"]
    assert res["branching_ratio"] < 1.0
    assert res["mu"] > 0
    assert res["alpha"] > 0
    assert res["beta"] > 0

def test_compute_conditional_intensity():
    event_times = np.array([1.0, 2.0, 3.0])
    t_points = np.array([0.5, 1.5, 3.5])
    mu, alpha, beta = 1.0, 0.5, 1.0

    intensities = compute_conditional_intensity(t_points, event_times, mu, alpha, beta)

    assert len(intensities) == 3
    # At t=0.5, no events have occurred yet, lambda = mu
    assert np.isclose(intensities[0], mu)
    # At t=1.5, event at 1.0 contributes: mu + alpha * exp(-beta * (1.5 - 1.0))
    expected_1_5 = mu + alpha * np.exp(-beta * 0.5)
    assert np.isclose(intensities[1], expected_1_5)
    # At t=3.5, all 3 events contribute
    assert intensities[2] > mu

def test_hawkes_empty_events():
    with pytest.raises(ValueError):
        fit_hawkes(np.array([]), 10.0)
