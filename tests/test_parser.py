"""
Unit tests for SIEM & EDR log parser (src/parser.py).
"""

import os
import json
import tempfile
import pytest
import pandas as pd
from src.parser import (
    parse_windows_event,
    parse_edr_event,
    parse_firewall_event,
    is_suspicious_command,
    parse_logs
)

def test_is_suspicious_command():
    assert is_suspicious_command("vssadmin delete shadows /all /quiet") is True
    assert is_suspicious_command("bcdedit /set {default} recoveryenabled no") is True
    assert is_suspicious_command("powershell.exe -enc SQBFAFgAIAA=") is True
    assert is_suspicious_command("wbadmin delete catalog -quiet") is True
    assert is_suspicious_command("notepad.exe test.txt") is False
    assert is_suspicious_command(None) is False

def test_parse_windows_event():
    # Test 4625 failed logon
    raw_4625 = {
        "TimeCreated": "2026-10-01T12:00:00Z",
        "EventID": 4625,
        "IpAddress": "10.0.0.50"
    }
    parsed = parse_windows_event(raw_4625)
    assert parsed is not None
    assert parsed["event_type"] == "FAILED_LOGON"
    assert parsed["severity"] == 2
    assert parsed["source_ip"] == "10.0.0.50"

    # Test 4688 suspicious command
    raw_4688 = {
        "TimeCreated": "2026-10-01T12:05:00Z",
        "EventID": 4688,
        "CommandLine": "vssadmin.exe delete shadows /all",
        "IpAddress": "10.0.0.50"
    }
    parsed_cmd = parse_windows_event(raw_4688)
    assert parsed_cmd is not None
    assert parsed_cmd["event_type"] == "SUSPICIOUS_CMD"
    assert parsed_cmd["severity"] == 3

    # Test 4624 normal logon ignored
    raw_4624 = {
        "TimeCreated": "2026-10-01T12:06:00Z",
        "EventID": 4624
    }
    assert parse_windows_event(raw_4624) is None

def test_parse_edr_event():
    # Test file encryption event
    raw_edr = {
        "timestamp": "2026-10-01T12:10:00Z",
        "file_path": "C:\\Data\\file1.locked",
        "entropy_score": 7.9,
        "action": "ENCRYPT_WRITE",
        "source_ip": "10.0.0.50"
    }
    parsed = parse_edr_event(raw_edr)
    assert parsed is not None
    assert parsed["event_type"] == "MASS_ENCRYPTION_BURST"
    assert parsed["severity"] == 4

def test_parse_firewall_event():
    # Test SMB
    raw_smb = {
        "timestamp": "2026-10-01T12:15:00Z",
        "dest_port": 445,
        "src_ip": "10.0.0.50"
    }
    parsed_smb = parse_firewall_event(raw_smb)
    assert parsed_smb is not None
    assert parsed_smb["event_type"] == "LATERAL_SMB"

    # Test RDP
    raw_rdp = {
        "timestamp": "2026-10-01T12:16:00Z",
        "dest_port": 3389,
        "src_ip": "10.0.0.50"
    }
    parsed_rdp = parse_firewall_event(raw_rdp)
    assert parsed_rdp is not None
    assert parsed_rdp["event_type"] == "LATERAL_RDP"

def test_parse_logs_end_to_end(tmp_path):
    test_json = tmp_path / "win.json"
    events = [
        {"TimeCreated": "2026-10-01T10:00:00Z", "EventID": 4625, "IpAddress": "192.168.1.10"},
        {"TimeCreated": "2026-10-01T10:01:00Z", "EventID": 4688, "CommandLine": "vssadmin delete shadows"}
    ]
    test_json.write_text(json.dumps(events))

    df = parse_logs(str(tmp_path))
    assert len(df) == 2
    assert list(df.columns) == ["timestamp", "event_type", "source_ip", "severity"]
    assert df["event_type"].tolist() == ["FAILED_LOGON", "SUSPICIOUS_CMD"]
