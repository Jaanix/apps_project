"""
Unit tests for alert payload and dispatcher (src/alerting.py).
"""

import os
import json
import pytest
from src.alerting import build_alert_payload, determine_confidence, send_alert

def test_build_alert_payload():
    model_params = {"mu": 1.0, "alpha": 0.6, "beta": 1.2, "branching_ratio": 0.5}
    payload = build_alert_payload(
        predicted_intensity=25.0,
        threshold=10.0,
        forecast_horizon="2h",
        model_params=model_params
    )

    assert payload["forecast_horizon"] == "2h"
    assert payload["predicted_intensity"] == 25.0
    assert payload["threshold"] == 10.0
    assert payload["confidence"] in ["high", "medium", "low"]
    assert len(payload["recommended_actions"]) > 0
    # Must include actionable containment recommendations
    actions_text = " ".join(payload["recommended_actions"]).lower()
    assert "isolate" in actions_text
    assert "smb" in actions_text

def test_determine_confidence():
    assert determine_confidence(predicted_intensity=20.0, threshold=10.0) == "high"
    assert determine_confidence(predicted_intensity=12.0, threshold=10.0) == "medium"
    assert determine_confidence(predicted_intensity=10.5, threshold=10.0) == "low"

def test_send_alert_logging(tmp_path):
    log_file = str(tmp_path / "test_alerts.log")
    payload = {"timestamp": "2026-10-01T12:00:00Z", "alert": "test"}

    result = send_alert(payload, webhook_url=None, log_file=log_file)

    assert result["delivery_status"] == "logged_locally"
    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        line = json.loads(f.readline())
        assert line["alert"] == "test"
