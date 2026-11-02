"""
SIEM & EDR Log Parser for Ransomware Precursor Events.
Ingests raw logs (Windows Events, EDR, Network/Firewall) in CSV and JSON formats,
extracts precursor telemetry, normalizes into a unified schema, and writes clean event streams.
"""

import os
import glob
import json
import re
import pandas as pd
from typing import List, Dict, Any, Optional

# Precursor keywords indicating ransomware preparation or evasion
SUSPICIOUS_CMD_PATTERNS = [
    re.compile(r"vssadmin(\.exe)?\s+delete\s+shadows", re.IGNORECASE),
    re.compile(r"bcdedit(\.exe)?\s+/set.*recoveryenabled\s+no", re.IGNORECASE),
    re.compile(r"bcdedit(\.exe)?\s+/set.*bootstatuspolicy\s+ignoreallfailures", re.IGNORECASE),
    re.compile(r"wbadmin(\.exe)?\s+delete", re.IGNORECASE),
    re.compile(r"powershell(\.exe)?\s+.*(-enc|-encodedcommand)", re.IGNORECASE),
    re.compile(r"certutil(\.exe)?\s+.*-urlcache", re.IGNORECASE),
    re.compile(r"wmic(\.exe)?\s+shadowcopy\s+delete", re.IGNORECASE),
    re.compile(r"cipher(\.exe)?\s+/w:", re.IGNORECASE)
]

def is_suspicious_command(cmd: str) -> bool:
    if not isinstance(cmd, str):
        return False
    return any(p.search(cmd) for p in SUSPICIOUS_CMD_PATTERNS)

def parse_windows_event(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extracts precursor indicators from Windows Security/System event logs."""
    event_id = item.get("EventID")
    time_str = item.get("TimeCreated") or item.get("timestamp") or item.get("TimeGenerated")
    if not time_str or not event_id:
        return None

    try:
        eid = int(event_id)
    except (ValueError, TypeError):
        return None

    src_ip = item.get("IpAddress") or item.get("source_ip") or "127.0.0.1"

    # Event 4625: Failed Logon (Credential stuffing / brute-force)
    if eid == 4625:
        return {
            "timestamp": time_str,
            "event_type": "FAILED_LOGON",
            "source_ip": str(src_ip),
            "severity": 2
        }

    # Event 4688: Process Creation (Command-line auditing)
    if eid == 4688:
        cmd_line = item.get("CommandLine", "")
        if is_suspicious_command(cmd_line):
            return {
                "timestamp": time_str,
                "event_type": "SUSPICIOUS_CMD",
                "source_ip": str(src_ip),
                "severity": 3
            }

    # Event 7045: New Service Installed (Persistence / Driver manipulation)
    if eid == 7045:
        return {
            "timestamp": time_str,
            "event_type": "SERVICE_INSTALL",
            "source_ip": str(src_ip),
            "severity": 3
        }

    return None

def parse_edr_event(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extracts precursor indicators from EDR file/process events."""
    action = str(item.get("action", "")).upper()
    entropy = item.get("entropy_score")
    file_path = str(item.get("file_path", "")).lower()
    time_str = item.get("timestamp") or item.get("TimeCreated")
    src_ip = item.get("source_ip") or item.get("target_host") or "127.0.0.1"

    # Check for rapid mass encryption or high entropy writes
    is_encrypted_ext = any(file_path.endswith(ext) for ext in [".locked", ".crypt", ".enc", ".ransom", ".crypto"])
    is_high_entropy = entropy is not None and float(entropy) >= 7.5
    is_encryption_action = "ENCRYPT" in action or action == "FILE_ENCRYPTION"

    if (is_encrypted_ext or is_high_entropy or is_encryption_action) and time_str:
        return {
            "timestamp": time_str,
            "event_type": "MASS_ENCRYPTION_BURST",
            "source_ip": str(src_ip),
            "severity": 4
        }
    return None

def parse_firewall_event(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extracts lateral movement precursor indicators from firewall/network logs."""
    dest_port = item.get("dest_port")
    time_str = item.get("timestamp")
    src_ip = item.get("src_ip") or item.get("source_ip") or "127.0.0.1"
    
    if not dest_port or not time_str:
        return None

    try:
        port = int(dest_port)
    except (ValueError, TypeError):
        return None

    # Lateral movement ports: SMB (445) and RDP (3389)
    if port in (445, 3389):
        proto = "SMB" if port == 445 else "RDP"
        return {
            "timestamp": time_str,
            "event_type": f"LATERAL_{proto}",
            "source_ip": str(src_ip),
            "severity": 2
        }
    return None

def parse_logs(input_dir: str = "data/raw") -> pd.DataFrame:
    """
    Modular parser that ingests all JSON and CSV logs in input_dir,
    extracts ransomware precursor events, normalizes into unified schema:
    { timestamp, event_type, source_ip, severity }
    and returns a chronologically sorted DataFrame.
    """
    normalized_events: List[Dict[str, Any]] = []

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    # Process all JSON files
    json_files = glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True)
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
                elif not isinstance(data, list):
                    continue

                for item in data:
                    if not isinstance(item, dict):
                        continue
                    # Check for windows event
                    we = parse_windows_event(item)
                    if we:
                        normalized_events.append(we)
                        continue
                    # Check for EDR event
                    ee = parse_edr_event(item)
                    if ee:
                        normalized_events.append(ee)
                        continue
                    # Check for firewall event
                    fe = parse_firewall_event(item)
                    if fe:
                        normalized_events.append(fe)
        except Exception as e:
            print(f"Warning: Failed reading {jf}: {e}")

    # Process all CSV files
    csv_files = glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True)
    for cf in csv_files:
        try:
            df_csv = pd.read_csv(cf)
            records = df_csv.to_dict(orient="records")
            for item in records:
                # Check firewall
                fe = parse_firewall_event(item)
                if fe:
                    normalized_events.append(fe)
                    continue
                # Check windows
                we = parse_windows_event(item)
                if we:
                    normalized_events.append(we)
                    continue
                # Check EDR
                ee = parse_edr_event(item)
                if ee:
                    normalized_events.append(ee)
        except Exception as e:
            print(f"Warning: Failed reading {cf}: {e}")

    if not normalized_events:
        # Return empty DataFrame with defined columns
        return pd.DataFrame(columns=["timestamp", "event_type", "source_ip", "severity"])

    df = pd.DataFrame(normalized_events)
    # Parse timestamp to ISO8601 UTC and sort
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return df

if __name__ == "__main__":
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)
    df = parse_logs("data/raw")
    target_path = os.path.join(out_dir, "events.csv")
    df.to_csv(target_path, index=False)
    print(f"Parsed {len(df)} ransomware-precursor events and saved to {target_path}")
    print("Event breakdown:")
    print(df["event_type"].value_counts())
