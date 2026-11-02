"""
Streamlit SOC Early Warning Dashboard for Ransomware Attack Waves.
Visualizes real-time Hawkes conditional intensity, forward 2h and 4h forecasting horizons,
calibrated alert thresholds, SOC containment playbooks, and audit logs.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timezone

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import fit_hawkes, compute_conditional_intensity
from src.forecast import forecast_intensity, calibrate_threshold, should_alert, compute_historical_intensity_series
from src.alerting import build_alert_payload, send_alert, RECOMMENDED_SOC_ACTIONS

# Set Page Config
st.set_page_config(
    page_title="Hawkes Ransomware Wave Forecaster",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ Hawkes Process Attack Wave Forecaster for Ransomware")
st.markdown(
    "**Proactive SOC Early-Warning System**: Continuous temporal point process modeling "
    "of precursor logons, lateral SMB/RDP movement, and privilege abuse to forecast ransomware encryption waves."
)

# Sidebar Configuration
st.sidebar.header("⚙️ Forecaster Configuration")
events_file = st.sidebar.text_input("Events CSV Path", value="data/processed/events.csv")
forecast_horizon = st.sidebar.selectbox("Forecast Horizon", options=[2.0, 4.0, 6.0], index=0)
threshold_p = st.sidebar.slider("Alert Threshold Percentile (P)", min_value=85.0, max_value=99.0, value=97.0, step=0.5)
replay_window_hours = st.sidebar.slider("Historical View Window (Hours)", min_value=12.0, max_value=168.0, value=48.0, step=6.0)

# Load Data
@st.cache_data
def load_events(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(by="timestamp_dt").reset_index(drop=True)
    t0 = df["timestamp_dt"].min()
    df["hours"] = (df["timestamp_dt"] - t0).dt.total_seconds() / 3600.0
    return df

df_events = load_events(events_file)

if df_events.empty:
    st.error(f"⚠️ Events dataset not found at `{events_file}`. Please execute `python src/parser.py` first.")
    st.stop()

# Replay Time Slider
max_h = float(df_events["hours"].max())
min_h = float(df_events["hours"].min())
current_sim_hour = st.sidebar.slider(
    "Observation Time (Replay Mode)",
    min_value=min_h + 12.0,
    max_value=max_h,
    value=max_h,
    step=1.0,
    help="Select point in time to evaluate Hawkes forecast"
)

# Filter events up to current observation time
obs_events = df_events[df_events["hours"] <= current_sim_hour].reset_index(drop=True)
active_window_events = obs_events[obs_events["hours"] >= (current_sim_hour - replay_window_hours)].reset_index(drop=True)

# Model Fitting
@st.cache_data
def get_fitted_model(times_tuple: tuple, t_end: float) -> dict:
    times_arr = np.array(times_tuple)
    return fit_hawkes(times_arr, end_time=t_end)

obs_times = tuple(obs_events["hours"].values)
with st.spinner("Calibrating Hawkes Process Intensity Parameters..."):
    model_params = get_fitted_model(obs_times, current_sim_hour)
    threshold = calibrate_threshold(model_params, np.array(obs_times), percentile=threshold_p)

# Forecast Future Intensity
forecast_res = forecast_intensity(
    model_params=model_params,
    event_times=np.array(obs_times),
    forecast_horizon_hours=forecast_horizon,
    num_points=120
)
alert_triggered = should_alert(forecast_res, threshold)

# KPI Metrics Banner
col1, col2, col3, col4, col5 = st.columns(5)
curr_intensity = forecast_res["intensities"][0]
peak_intensity = forecast_res["peak_intensity"]

col1.metric("Current Intensity λ(t)", f"{curr_intensity:.2f} /hr")
col2.metric(f"Peak Forecast (Δ={forecast_horizon}h)", f"{peak_intensity:.2f} /hr", delta=f"{peak_intensity - threshold:+.1f} vs Thresh")
col3.metric(f"Threshold (P{threshold_p:.0f})", f"{threshold:.2f} /hr")
col4.metric("Branching Ratio (η = α/β)", f"{model_params['branching_ratio']:.3f}", delta="Stationary (<1.0)")
col5.metric("Baseline Noise (μ)", f"{model_params['mu']:.3f} /hr")

# Alert Banner & Containment Actions
st.markdown("---")
if alert_triggered:
    st.error(f"🚨 **CRITICAL SOC ADVISORY: RANSOMWARE ATTACK WAVE IMMINENT** (Next {forecast_horizon} Hours)")
    alert_payload = build_alert_payload(
        predicted_intensity=peak_intensity,
        threshold=threshold,
        forecast_horizon=f"{forecast_horizon}h",
        model_params=model_params
    )
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("📋 Recommended SOC Containment Playbook:")
        for action in alert_payload["recommended_actions"]:
            st.markdown(f"- ✅ **{action}**")
    with col_b:
        st.subheader("⚠️ Alert Telemetry:")
        st.json(alert_payload)
        if st.button("📤 Dispatch Alert to SOC Webhook / Log"):
            res = send_alert(alert_payload)
            st.success(f"Dispatched: {res['delivery_status']} (Logged: {res['logged_to']})")
else:
    st.success(f"🟢 **NORMAL SECURITY TELEMETRY**: Forecasted intensity stays below P{threshold_p:.0f} threshold ({threshold:.2f}/hr).")

# Intensity Visualization (Matplotlib figure)
st.subheader("📈 Hawkes Process Conditional Intensity & Forward Forecast")

# Generate visualization
fig, ax = plt.subplots(figsize=(12, 4.5))

# Plot historical intensity curve over active window
view_start = max(current_sim_hour - replay_window_hours, 0.0)
t_hist_grid = np.linspace(view_start, current_sim_hour, 300)
hist_lambda = compute_conditional_intensity(
    t_hist_grid,
    np.array(obs_times),
    model_params["mu"],
    model_params["alpha"],
    model_params["beta"]
)

ax.plot(t_hist_grid, hist_lambda, color="#1f77b4", lw=2, label="Historical Intensity $\\lambda(t)$")

# Plot forecast curve
ax.plot(
    forecast_res["time_grid"],
    forecast_res["intensities"],
    color="#d62728",
    lw=2.5,
    linestyle="--",
    label=f"Forecast $\\mathbb{{E}}[\\lambda(t+\\Delta)]$ ({forecast_horizon}h Horizon)"
)

# Plot alert threshold
ax.axhline(
    y=threshold,
    color="#ff7f0e",
    linestyle=":",
    lw=2,
    label=f"Alert Threshold (P{threshold_p:.0f} = {threshold:.2f}/hr)"
)

# Vertical observation boundary
ax.axvline(x=current_sim_hour, color="gray", linestyle="-.", alpha=0.7, label="Current Observation (T_obs)")

# Plot precursor event ticks on x-axis
win_events = active_window_events["hours"].values
if len(win_events) > 0:
    ax.scatter(win_events, np.zeros_like(win_events), marker="|", color="darkred", s=60, alpha=0.5, label="Precursor Events")

ax.set_xlim(view_start, current_sim_hour + forecast_horizon)
ax.set_ylim(bottom=0, top=max(float(np.max(hist_lambda)), float(peak_intensity), threshold) * 1.25)
ax.set_xlabel("Elapsed Simulation Time (Hours)", fontsize=11)
ax.set_ylabel("Event Intensity $\\lambda(t)$ (events/hour)", fontsize=11)
ax.set_title(f"Hawkes Self-Exciting Intensity with Forward Horizon $\\Delta = {forecast_horizon}$h", fontsize=12, fontweight="bold")
ax.legend(loc="upper left", frameon=True)
ax.grid(True, alpha=0.3)

st.pyplot(fig)
plt.close(fig)

# Precursor Event Breakdown Table
st.subheader("🔍 Precursor Event Log Telemetry")
col_t1, col_t2 = st.columns([1, 2])
with col_t1:
    st.write(f"**Observed Events in Window:** {len(active_window_events)}")
    type_counts = active_window_events["event_type"].value_counts()
    st.bar_chart(type_counts)
with col_t2:
    st.dataframe(
        active_window_events[["timestamp", "event_type", "source_ip", "severity", "hours"]]
        .sort_values(by="hours", ascending=False)
        .head(100),
        use_container_width=True
    )

# Footer
st.markdown("---")
st.caption("Hawkes Process Attack Wave Forecaster | Pure-Python MLE Stationarity Engine | SOC Advisory Prototype")
