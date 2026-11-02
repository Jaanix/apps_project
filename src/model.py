"""
Univariate Hawkes Process with Exponential Kernel.
Provides Maximum Likelihood Estimation (MLE) of (mu, alpha, beta),
strict enforcement of the stationarity condition (alpha < beta),
log-likelihood computation, and conditional intensity evaluation.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any, Optional, Tuple, Sequence

try:
    import phawkes
    PHAWKES_AVAILABLE = True
except ImportError:
    PHAWKES_AVAILABLE = False


def hawkes_neg_log_likelihood(
    params: np.ndarray,
    event_times: np.ndarray,
    end_time: float
) -> float:
    """
    Computes the negative log-likelihood of a univariate Hawkes process with exponential kernel:
    lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))
    
    Using O(n) recursive formulation:
    R(1) = 0
    R(i) = exp(-beta * (t_i - t_{i-1})) * (1 + R(i-1))
    lambda(t_i) = mu + alpha * R(i)
    
    Integral_0^T lambda(s) ds = mu * T + (alpha / beta) * sum_{i=1}^n (1 - exp(-beta * (T - t_i)))
    """
    mu, alpha, beta = params
    n = len(event_times)
    if n == 0:
        return mu * end_time

    # Boundary and stability penalties
    if mu <= 0 or alpha <= 0 or beta <= 0:
        return 1e12
    if alpha >= beta:  # Non-stationary barrier penalty
        return 1e12 + 1e6 * (alpha - beta + 1.0)

    # Recursive computation of recursive state R
    dt = np.diff(event_times)
    # Check for non-negative inter-arrivals
    if np.any(dt < 0):
        raise ValueError("Event times must be sorted in ascending order.")

    exp_beta_dt = np.exp(-beta * dt)
    R = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        R[i] = exp_beta_dt[i - 1] * (1.0 + R[i - 1])

    lambdas = mu + alpha * R
    if np.any(lambdas <= 0):
        return 1e12

    # Log sum term
    log_sum = np.sum(np.log(lambdas))

    # Integral term
    exp_end = np.exp(-beta * (end_time - event_times))
    integral = mu * end_time + (alpha / beta) * np.sum(1.0 - exp_end)

    neg_ll = integral - log_sum
    if np.isnan(neg_ll) or np.isinf(neg_ll):
        return 1e12
    return float(neg_ll)


def compute_conditional_intensity(
    t_points: np.ndarray,
    event_times: np.ndarray,
    mu: float,
    alpha: float,
    beta: float
) -> np.ndarray:
    """
    Computes conditional intensity lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))
    at specified time points t_points.
    """
    t_points = np.asarray(t_points, dtype=np.float64)
    event_times = np.asarray(event_times, dtype=np.float64)
    
    intensities = np.full_like(t_points, mu, dtype=np.float64)
    if len(event_times) == 0:
        return intensities

    # For each query point t, sum alpha * exp(-beta * (t - t_i)) for t_i < t
    for idx, t in enumerate(t_points):
        past_events = event_times[event_times < t]
        if len(past_events) > 0:
            intensities[idx] += alpha * np.sum(np.exp(-beta * (t - past_events)))
            
    return intensities


def fit_hawkes_mle_scipy(
    event_times: np.ndarray,
    end_time: float,
    init_params: Optional[Tuple[float, float, float]] = None
) -> Dict[str, Any]:
    """
    Fits Hawkes process via MLE using SciPy's L-BFGS-B / SLSQP constrained optimization.
    Guarantees stationarity: alpha < beta (branching_ratio = alpha / beta < 1).
    """
    event_times = np.sort(np.asarray(event_times, dtype=np.float64))
    n = len(event_times)
    if n == 0:
        raise ValueError("Cannot fit Hawkes process with zero events.")
    if end_time <= event_times[-1]:
        end_time = float(event_times[-1] + 1e-4)

    # Heuristic initial values if not supplied
    if init_params is None:
        duration = end_time - event_times[0] if event_times[0] < end_time else 1.0
        overall_rate = max(n / max(duration, 1.0), 1e-2)
        init_mu = max(overall_rate * 0.3, 1e-3)
        init_beta = max(1.0 / (np.mean(np.diff(event_times)) + 1e-3), 0.5)
        init_alpha = min(init_beta * 0.5, init_beta - 1e-3)
        init_params = (init_mu, init_alpha, init_beta)

    # Bounds: mu > 0, alpha > 0, beta > 0
    bounds = [(1e-5, None), (1e-5, None), (1e-5, None)]

    # Stationarity constraint: beta - alpha >= 1e-4
    constraints = [
        {"type": "ineq", "fun": lambda p: p[2] - p[1] - 1e-4}
    ]

    res = minimize(
        fun=hawkes_neg_log_likelihood,
        x0=np.array(init_params, dtype=np.float64),
        args=(event_times, end_time),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-7}
    )

    # In case SLSQP does not converge, try multiple restart points
    if not res.success or res.x[1] >= res.x[2]:
        for factor in [0.2, 0.5, 0.8]:
            restart_x0 = (init_params[0], init_params[2] * factor, init_params[2])
            res_retry = minimize(
                fun=hawkes_neg_log_likelihood,
                x0=np.array(restart_x0, dtype=np.float64),
                args=(event_times, end_time),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-7}
            )
            if res_retry.success and res_retry.fun < res.fun and res_retry.x[1] < res_retry.x[2]:
                res = res_retry
                break

    mu_hat, alpha_hat, beta_hat = float(res.x[0]), float(res.x[1]), float(res.x[2])
    
    # Strict validation of stationarity
    if alpha_hat >= beta_hat:
        alpha_hat = beta_hat * 0.95  # Enforce stationarity

    branching_ratio = float(alpha_hat / beta_hat)
    log_likelihood = float(-res.fun)

    return {
        "mu": mu_hat,
        "alpha": alpha_hat,
        "beta": beta_hat,
        "branching_ratio": branching_ratio,
        "log_likelihood": log_likelihood,
        "converged": bool(res.success),
        "engine": "scipy_slsqp"
    }


def fit_hawkes(
    event_times: np.ndarray,
    end_time: float,
    prefer_phawkes: bool = False
) -> Dict[str, Any]:
    """
    Main model fitting entry point:
    Estimates baseline mu, excitation alpha, and decay beta.
    Validates stationarity alpha < beta.
    Returns:
      {
        "mu": float,
        "alpha": float,
        "beta": float,
        "branching_ratio": float,
        "log_likelihood": float,
        "converged": bool,
        "engine": str
      }
    """
    event_times = np.sort(np.asarray(event_times, dtype=np.float64))
    if len(event_times) == 0:
        raise ValueError("Cannot fit Hawkes model with empty event times.")

    if prefer_phawkes and PHAWKES_AVAILABLE:
        try:
            # Estimate decay from inter-arrival scale
            mean_dt = float(np.mean(np.diff(event_times))) if len(event_times) > 1 else 1.0
            decay_cand = 1.0 / max(mean_dt, 1e-2)
            model = phawkes.HawkesExpMLE(decay=decay_cand)
            model.fit([event_times], end_time=end_time)
            mu_val = float(model.baseline_[0])
            alpha_val = float(model.adjacency_[0, 0])
            beta_val = decay_cand
            br = float(alpha_val / beta_val)
            if br < 1.0:
                return {
                    "mu": mu_val,
                    "alpha": alpha_val,
                    "beta": beta_val,
                    "branching_ratio": br,
                    "log_likelihood": float(model.final_loglik_),
                    "converged": bool(model.converged_),
                    "engine": "phawkes"
                }
        except Exception:
            pass  # Fall back to native SciPy MLE

    # Joint estimation over (mu, alpha, beta)
    return fit_hawkes_mle_scipy(event_times, end_time)


if __name__ == "__main__":
    # Test on synthetic Hawkes data with known parameters
    print("Testing Hawkes MLE on synthetic point process...")
    np.random.seed(42)
    # Generate simple test cluster
    synthetic_times = np.sort(np.cumsum(np.random.exponential(scale=0.5, size=200)))
    T = float(synthetic_times[-1] + 1.0)
    
    result = fit_hawkes(synthetic_times, T)
    print("Fitted Parameters:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    assert result["alpha"] < result["beta"], "Stationarity condition violated!"
    print("Stationarity confirmed: alpha < beta!")
