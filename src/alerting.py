"""
SOC Alert Generation and Webhook Dispatcher.
Constructs structured, actionable alert payloads with containment recommendations
and delivers alerts to webhooks or local alert audit logs.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import requests

RECOMMENDED_SOC_ACTIONS = [
    "Isolate source and target host subnets exhibiting anomalous logon bursts",
    "Block SMB (TCP 445) and RDP (TCP 3389) laterally across internal network segments",
    "Revoke active Kerberos TGTs and reset compromised account credentials",
    "Enable emergency immutable snapshot locks and monitor VSS deletion commands",
    "Pre-stage endpoint containment via EDR sensor host isolation"
]

def determine_confidence(
    predicted_intensity: float,
    threshold: float,
    branching_ratio: Optional[float] = None
) -> str:
    """
    Computes alert confidence level (high, medium, low) based on severity ratio
    and model branching ratio.
    """
    ratio = predicted_intensity / max(threshold, 1e-4)
    br = branching_ratio if branching_ratio is not None else 0.5
    
    if ratio >= 1.5 or (ratio >= 1.25 and br >= 0.5):
        return "high"
    elif ratio >= 1.15:
        return "medium"
    else:
        return "low"


def build_alert_payload(
    predicted_intensity: float,
    threshold: float,
    forecast_horizon: str = "2h",
    model_params: Optional[Dict[str, Any]] = None,
    recommended_actions: Optional[List[str]] = None,
    timestamp: Optional[str] = None
) -> Dict[str, Any]:
    """
    Constructs an actionable SOC alert payload.
    """
    if timestamp is None:
        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        now_ts = timestamp

    br = float(model_params.get("branching_ratio", 0.5)) if model_params else None
    confidence = determine_confidence(predicted_intensity, threshold, br)

    actions = recommended_actions if recommended_actions is not None else RECOMMENDED_SOC_ACTIONS

    payload = {
        "timestamp": now_ts,
        "forecast_horizon": forecast_horizon,
        "predicted_intensity": round(float(predicted_intensity), 3),
        "threshold": round(float(threshold), 3),
        "intensity_ratio": round(float(predicted_intensity / max(threshold, 1e-4)), 2),
        "confidence": confidence,
        "model_parameters": {
            "mu": round(float(model_params.get("mu", 0.0)), 4) if model_params else None,
            "alpha": round(float(model_params.get("alpha", 0.0)), 4) if model_params else None,
            "beta": round(float(model_params.get("beta", 0.0)), 4) if model_params else None,
            "branching_ratio": round(float(br), 4) if br is not None else None
        } if model_params else {},
        "recommended_actions": actions,
        "advisory": "EARLY WARNING ONLY: Pre-stage containment. Manual analyst review advised."
    }
    return payload


def send_alert(
    payload: Dict[str, Any],
    webhook_url: Optional[str] = None,
    log_file: str = "alerts.log"
) -> Dict[str, Any]:
    """
    Dispatches alert payload to configured webhook (Slack, Teams, PagerDuty, SIEM).
    If no webhook is configured or webhook fails, logs securely to log_file.
    """
    delivered = False
    delivery_status = "logged_locally"
    error_msg = None

    # Always persist locally to log_file for SOC audit trail
    try:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception as e:
        print(f"Failed to append to log file {log_file}: {e}")

    # Webhook delivery if URL provided
    if webhook_url and webhook_url.strip():
        try:
            headers = {"Content-Type": "application/json"}
            resp = requests.post(webhook_url, json=payload, headers=headers, timeout=5)
            if resp.status_code in (200, 201, 202, 204):
                delivered = True
                delivery_status = f"webhook_success_http_{resp.status_code}"
            else:
                delivery_status = f"webhook_error_http_{resp.status_code}"
                error_msg = resp.text[:200]
        except Exception as e:
            delivery_status = "webhook_exception"
            error_msg = str(e)

    return {
        "delivered": delivered,
        "delivery_status": delivery_status,
        "logged_to": log_file,
        "timestamp": payload.get("timestamp"),
        "error": error_msg
    }


if __name__ == "__main__":
    sample_params = {"mu": 0.85, "alpha": 0.62, "beta": 0.95, "branching_ratio": 0.65}
    sample_payload = build_alert_payload(
        predicted_intensity=18.4,
        threshold=10.0,
        forecast_horizon="2h",
        model_params=sample_params
    )
    result = send_alert(sample_payload)
    print("Generated Alert Payload:")
    print(json.dumps(sample_payload, indent=2))
    print("\nDispatch Result:")
    print(result)
