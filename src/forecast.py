"""
Intensity Forecasting and Threshold Calibration for Hawkes Point Processes.
Projects forward conditional intensity lambda(t + Delta) for 2h and 4h horizons,
calibrates historical percentile-based alert thresholds, and determines SOC alert conditions.
"""

import numpy as np
from typing import Dict, Any, Union, Optional, Tuple

def compute_historical_intensity_series(
    model_params: Dict[str, Any],
    event_times: np.ndarray,
    grid_points: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes intensity lambda(t) along a continuous uniform time grid over the historical event sequence.
    Returns: (time_grid, intensity_grid)
    """
    mu = float(model_params["mu"])
    alpha = float(model_params["alpha"])
    beta = float(model_params["beta"])
    
    event_times = np.sort(np.asarray(event_times, dtype=np.float64))
    if len(event_times) == 0:
        return np.array([0.0]), np.array([mu])

    start_t = float(event_times[0])
    end_t = float(event_times[-1])
    time_grid = np.linspace(start_t, end_t, max(grid_points, 100))
    
    intensities = np.full_like(time_grid, mu)
    for i, t in enumerate(time_grid):
        past = event_times[event_times < t]
        if len(past) > 0:
            intensities[i] += alpha * np.sum(np.exp(-beta * (t - past)))
            
    return time_grid, intensities


def calibrate_threshold(
    model_params: Dict[str, Any],
    event_times: np.ndarray,
    percentile: float = 95.0,
    window_hours: Optional[float] = None
) -> float:
    """
    Calibrates an alert threshold by evaluating historical lambda(t) and taking the
    specified percentile (default: 95th percentile).
    """
    event_times = np.sort(np.asarray(event_times, dtype=np.float64))
    if len(event_times) == 0:
        return float(model_params["mu"] * 2.0)

    if window_hours is not None and window_hours > 0:
        cutoff = event_times[-1] - window_hours
        window_events = event_times[event_times >= cutoff]
    else:
        window_events = event_times

    if len(window_events) == 0:
        window_events = event_times

    _, hist_intensities = compute_historical_intensity_series(
        model_params=model_params,
        event_times=window_events,
        grid_points=1000
    )

    threshold = float(np.percentile(hist_intensities, percentile))
    # Safety floor: must be strictly above baseline
    floor = float(model_params["mu"] * 1.5)
    return max(threshold, floor)


def forecast_intensity(
    model_params: Dict[str, Any],
    event_times: np.ndarray,
    forecast_horizon_hours: float = 2.0,
    num_points: int = 100,
    include_branching_expectation: bool = True
) -> Dict[str, Any]:
    """
    Forecasts conditional intensity lambda(t) over a future grid:
    from t_last to (t_last + forecast_horizon_hours).
    
    If include_branching_expectation is True, evaluates expected intensity accounting for future offspring:
      E[lambda(t) | H_{t_n}] = mu + (lambda(t_n^+) - mu) * exp(-(beta - alpha) * (t - t_n))
    Otherwise, evaluates deterministic relaxation of past events:
      lambda_decay(t | H_{t_n}) = mu + (lambda(t_n^+) - mu) * exp(-beta * (t - t_n))
      
    Returns dictionary with:
      - 'time_grid': np.ndarray of future time offsets (hours)
      - 'intensities': np.ndarray of forecasted intensities
      - 'peak_intensity': float
      - 'mean_intensity': float
      - 'expected_event_count': float (integrated intensity over horizon)
    """
    mu = float(model_params["mu"])
    alpha = float(model_params["alpha"])
    beta = float(model_params["beta"])

    event_times = np.sort(np.asarray(event_times, dtype=np.float64))
    t_last = float(event_times[-1]) if len(event_times) > 0 else 0.0

    # Calculate intensity right after the last event: lambda(t_last^+)
    past_events = event_times[event_times <= t_last]
    if len(past_events) > 0:
        excitation_sum = np.sum(np.exp(-beta * (t_last - past_events)))
        lambda_last_plus = mu + alpha * excitation_sum
    else:
        lambda_last_plus = mu

    # Create forward time grid
    future_offsets = np.linspace(0.0, float(forecast_horizon_hours), num_points)
    future_times = t_last + future_offsets

    decay_rate = (beta - alpha) if (include_branching_expectation and beta > alpha) else beta
    excess_intensity = max(lambda_last_plus - mu, 0.0)

    # Forecast values: lambda(t) = mu + excess * exp(-decay_rate * offset)
    forecast_values = mu + excess_intensity * np.exp(-decay_rate * future_offsets)

    # Analytical integral of intensity over [t_last, t_last + horizon]
    # Integral = mu * H + excess * (1 - exp(-decay_rate * H)) / decay_rate
    if decay_rate > 1e-6:
        expected_events = mu * forecast_horizon_hours + excess_intensity * (1.0 - np.exp(-decay_rate * forecast_horizon_hours)) / decay_rate
    else:
        expected_events = mu * forecast_horizon_hours + excess_intensity * forecast_horizon_hours

    return {
        "t_last": t_last,
        "time_grid": future_times,
        "future_offsets_hours": future_offsets,
        "intensities": forecast_values,
        "peak_intensity": float(np.max(forecast_values)),
        "mean_intensity": float(np.mean(forecast_values)),
        "expected_event_count": float(expected_events)
    }


def should_alert(
    forecast: Union[np.ndarray, Dict[str, Any]],
    threshold: float
) -> bool:
    """
    Determines if forecasted intensity exceeds the calibrated alert threshold.
    """
    if isinstance(forecast, dict):
        peak = forecast.get("peak_intensity", 0.0)
    else:
        peak = float(np.max(forecast)) if len(forecast) > 0 else 0.0
        
    return bool(peak > threshold)


if __name__ == "__main__":
    test_params = {"mu": 1.5, "alpha": 0.8, "beta": 1.2}
    sim_events = np.array([1.0, 1.2, 1.3, 1.35, 1.38])
    th = calibrate_threshold(test_params, sim_events, percentile=90.0)
    fc = forecast_intensity(test_params, sim_events, forecast_horizon_hours=2.0)
    alert = should_alert(fc, th)
    print(f"Calibrated Threshold: {th:.2f}")
    print(f"Peak Forecast Intensity: {fc['peak_intensity']:.2f}")
    print(f"Should Alert: {alert}")
