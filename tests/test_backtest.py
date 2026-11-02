"""
Unit tests for chronological backtesting engine (src/backtest.py).
"""

import os
import pytest
from src.backtest import run_backtest

def test_run_backtest_acceptance_criteria(tmp_path):
    events_path = "data/processed/events.csv"
    if not os.path.exists(events_path):
        pytest.skip("Processed events not generated yet.")

    report_file = str(tmp_path / "test_backtest.md")
    results = run_backtest(
        events_path=events_path,
        train_ratio=0.80,
        forecast_horizon_hours=2.0,
        eval_step_hours=1.0,
        threshold_percentile=97.0,
        report_path=report_file
    )

    # Acceptance Criteria Check
    assert results["precision"] > 0.75, f"Precision {results['precision']} not > 0.75"
    assert results["recall"] > 0.70, f"Recall {results['recall']} not > 0.70"
    assert results["average_lead_time_hours"] > 2.0, f"Lead time {results['average_lead_time_hours']} not > 2.0h"
    assert os.path.exists(report_file)
    assert results["model_params"]["alpha"] < results["model_params"]["beta"]
