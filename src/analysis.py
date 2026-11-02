"""
Temporal Analysis and Clustering Detection for Security Event Streams.
Computes inter-arrival distributions, tests against a homogeneous Poisson baseline,
and generates diagnostic Q-Q and histogram plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Any

def analyze_clustering(
    events: pd.DataFrame,
    output_plot_path: str = "reports/clustering_analysis.png"
) -> Dict[str, Any]:
    """
    Analyzes the temporal clustering of events:
    1. Computes inter-arrival times in hours.
    2. Evaluates the Coefficient of Variation (CV = std / mean). For Poisson, CV = 1; for clustered, CV > 1.
    3. Performs Kolmogorov-Smirnov test against exponential distribution.
    4. Generates and saves histogram & Q-Q plots.
    Returns:
      {
        "is_clustered": bool,
        "mean_inter_arrival_hours": float,
        "std_inter_arrival_hours": float,
        "coefficient_of_variation": float,
        "ks_stat": float,
        "ks_pvalue": float,
        "plot_path": str
      }
    """
    if events.empty or len(events) < 3:
        raise ValueError("Insufficient events to analyze clustering (need at least 3 events).")

    # Ensure timestamps are parsed and sorted
    timestamps = pd.to_datetime(events["timestamp"], utc=True).sort_values()
    
    # Calculate inter-arrival times in hours
    diffs = timestamps.diff().dropna()
    inter_arrivals_hours = diffs.dt.total_seconds().values / 3600.0
    
    # Filter non-positive values (duplicate timestamps within sub-second)
    # Add a minimal epsilon for numerical stability if 0
    inter_arrivals_clean = np.where(inter_arrivals_hours <= 0, 1e-4, inter_arrivals_hours)

    mean_dt = float(np.mean(inter_arrivals_clean))
    std_dt = float(np.std(inter_arrivals_clean))
    cv = float(std_dt / mean_dt) if mean_dt > 0 else 0.0

    # Fit exponential distribution (scale = mean)
    # Kolmogorov-Smirnov test against exponential distribution
    ks_res = stats.kstest(inter_arrivals_clean, "expon", args=(0, mean_dt))
    ks_stat = float(ks_res.statistic)
    ks_pvalue = float(ks_res.pvalue)

    # Clustered if overdispersed (CV > 1.2) or strongly rejected exponential baseline (p < 0.01)
    is_clustered = bool(cv > 1.2 or ks_pvalue < 0.01)

    # Generate plots
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Inter-arrival Histogram vs Fitted Exponential
    ax1 = axes[0]
    n, bins, _ = ax1.hist(
        inter_arrivals_clean,
        bins=50,
        density=True,
        alpha=0.6,
        color="royalblue",
        edgecolor="black",
        label="Observed Inter-Arrivals"
    )
    x_grid = np.linspace(0, np.percentile(inter_arrivals_clean, 98), 200)
    pdf_fitted = stats.expon.pdf(x_grid, scale=mean_dt)
    ax1.plot(x_grid, pdf_fitted, "r--", lw=2, label=f"Poisson (Exp PDF, $\\mu={mean_dt:.2f}$h)")
    ax1.set_xlim(0, np.percentile(inter_arrivals_clean, 98))
    ax1.set_xlabel("Inter-arrival Time (Hours)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title(f"Inter-Arrival Distribution (CV = {cv:.2f})", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Q-Q Plot against Exponential
    ax2 = axes[1]
    # Calculate empirical quantiles vs theoretical exponential quantiles
    sorted_data = np.sort(inter_arrivals_clean)
    n_points = len(sorted_data)
    quantiles = np.linspace(0.5 / n_points, 1 - 0.5 / n_points, n_points)
    theoretical_quantiles = stats.expon.ppf(quantiles, scale=mean_dt)

    ax2.scatter(theoretical_quantiles, sorted_data, color="darkorange", alpha=0.6, edgecolors="none", s=25, label="Observed vs Theory")
    max_q = max(theoretical_quantiles[-1], sorted_data[-1])
    line_limit = np.percentile(inter_arrivals_clean, 98)
    ax2.plot([0, line_limit], [0, line_limit], "r--", lw=2, label="Poisson Baseline (1:1)")
    ax2.set_xlim(0, line_limit)
    ax2.set_ylim(0, line_limit)
    ax2.set_xlabel("Theoretical Exponential Quantiles (Hours)", fontsize=11)
    ax2.set_ylabel("Observed Inter-arrival Quantiles (Hours)", fontsize=11)
    clustering_status = "Clustered (Self-Exciting Hawkes Suitable)" if is_clustered else "Poisson (Unclustered)"
    ax2.set_title(f"Q-Q Plot vs Poisson Baseline\nStatus: {clustering_status}", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close(fig)

    return {
        "is_clustered": is_clustered,
        "mean_inter_arrival_hours": mean_dt,
        "std_inter_arrival_hours": std_dt,
        "coefficient_of_variation": cv,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "plot_path": output_plot_path
    }

if __name__ == "__main__":
    events_path = "data/processed/events.csv"
    if not os.path.exists(events_path):
        print("Events file not found. Run parser first.")
        exit(1)
    df = pd.read_csv(events_path)
    res = analyze_clustering(df, "reports/clustering_analysis.png")
    print("Temporal Clustering Analysis Results:")
    for k, v in res.items():
        print(f"  {k}: {v}")
