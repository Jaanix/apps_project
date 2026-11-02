"""
Unit tests for temporal clustering analysis (src/analysis.py).
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.analysis import analyze_clustering

def test_analyze_clustering_on_clustered_data(tmp_path):
    # Create bursty clustered event stream
    times = []
    # Cluster 1
    times.extend([f"2026-10-01T01:00:{s:02d}Z" for s in range(0, 50, 2)])
    # Gap
    # Cluster 2
    times.extend([f"2026-10-01T08:00:{s:02d}Z" for s in range(0, 50, 2)])
    
    df = pd.DataFrame({
        "timestamp": times,
        "event_type": ["FAILED_LOGON"] * len(times),
        "source_ip": ["192.168.1.1"] * len(times),
        "severity": [2] * len(times)
    })

    plot_file = str(tmp_path / "clustering.png")
    res = analyze_clustering(df, output_plot_path=plot_file)

    assert "is_clustered" in res
    assert res["is_clustered"] is True
    assert res["coefficient_of_variation"] > 1.0
    assert os.path.exists(plot_file)

def test_analyze_clustering_insufficient_events():
    df = pd.DataFrame({"timestamp": ["2026-10-01T01:00:00Z"]})
    with pytest.raises(ValueError):
        analyze_clustering(df)
