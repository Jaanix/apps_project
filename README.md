# Hawkes Process Attack Wave Forecaster for Ransomware

An operational, mathematical early-warning engine for Security Operations Center (SOC) teams. It models the self-exciting, cascading dynamics of ransomware precursor events (credential stuffing, lateral movement, privilege abuse, and recovery inhibition) using a **Univariate Hawkes Process** to forecast impending encryption waves 2 to 4 hours in advance.

---

## 1. Problem Statement
Ransomware attacks are not isolated events; they cluster into rapid self-exciting waves:
1. **Initial Access**: Brute force or credential stuffing (`Windows Event 4625`).
2. **Lateral Movement**: Rapid internal SMB (TCP 445) and RDP (TCP 3389) traffic.
3. **Privilege Escalation & Recovery Inhibition**: Volume shadow copy deletion (`vssadmin delete shadows`, `bcdedit`, `7045` service installs).
4. **Mass Encryption Peak**: Rapid high-entropy file encryption bursts across shares.

Standard SIEM systems alert *reactively* after data destruction begins. The Hawkes Forecaster provides **proactive early warning** with a lead time exceeding 2 to 3.5 hours, giving SOC teams time to pre-stage containment (isolate subnets, block lateral ports, revoke Kerberos tickets).

---

## 2. Mathematical Formulation

### 2.1 Conditional Intensity Function
$$\lambda(t) = \mu + \sum_{t_i < t} \alpha \, e^{-\beta (t - t_i)}$$
- $\mu \ge 0$: Baseline arrival rate of independent background security noise.
- $\alpha \ge 0$: Excitation factor (immediate intensity surge triggered by an event).
- $\beta > 0$: Exponential decay factor.
- Branching ratio $\eta = \frac{\alpha}{\beta}$: Expected number of directly triggered events per event.
- **Stationarity Condition:** $\frac{\alpha}{\beta} < 1$ ($\alpha < \beta$). Guarantees cascades are subcritical and self-limiting.

### 2.2 Maximum Likelihood Estimation (MLE)
$$\ln L(\mu, \alpha, \beta) = \sum_{i=1}^n \ln \lambda(t_i) - \int_0^T \lambda(s)\,ds$$

With $O(n)$ recursive state evaluation:
$$R(i) = e^{-\beta (t_i - t_{i-1})} (1 + R(i-1)), \quad R(1) = 0$$
$$\lambda(t_i) = \mu + \alpha R(i)$$
$$\int_0^T \lambda(s)\,ds = \mu T + \frac{\alpha}{\beta} \sum_{i=1}^n \left(1 - e^{-\beta (T - t_i)}\right)$$

### 2.3 Intensity Forecasting & Thresholding
For a forward horizon $\Delta \in \{2\text{h}, 4\text{h}\}$:
$$\mathbb{E}[\lambda(t \mid \mathcal{H}_{t_n})] = \mu + (\lambda(t_n^+) - \mu) e^{-(\beta - \alpha)(t - t_n)}$$
An alert triggers if $\max_{t \in [t_n, t_n + \Delta]} \lambda(t) > \text{Threshold}_{P97}$.

---

## 3. Project Architecture

```
├── app/
│   └── dashboard.py               # Interactive Streamlit SOC monitoring dashboard
├── data/
│   ├── raw/                       # Multi-source SIEM/EDR raw logs (JSON, CSV)
│   ├── processed/
│   │   └── events.csv             # Clean, normalized precursor event stream
│   └── generate_sample_logs.py    # Realistic synthetic SIEM log generator
├── reports/
│   ├── clustering_analysis.png    # Diagnostic inter-arrival histogram & Q-Q plots
│   └── backtest_results.md        # Comprehensive backtesting validation report
├── src/
│   ├── __init__.py
│   ├── parser.py                  # SIEM log parser & normalizer
│   ├── analysis.py                # Heavy-tail clustering detector (CV & KS test)
│   ├── model.py                   # Hawkes MLE optimizer with stationarity enforcement
│   ├── forecast.py                # Intensity forecasting & threshold calibration
│   ├── alerting.py                # Actionable SOC alert payload & webhook dispatcher
│   └── backtest.py                # Chronological 80/20 train/test backtesting engine
├── tests/
│   ├── test_parser.py
│   ├── test_analysis.py
│   ├── test_model.py
│   ├── test_forecast.py
│   ├── test_alerting.py
│   └── test_backtest.py
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

---

## 4. Installation & Setup

### Prerequisites
- Python 3.11+ (Fully verified on Python 3.14)

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 5. Usage & Execution Workflow

### Step 1: Generate Telemetry (or Ingest Your Own Logs)
```bash
python data/generate_sample_logs.py
```

### Step 2: Parse and Normalize SIEM Logs
```bash
python src/parser.py
```
Outputs normalized schema to `data/processed/events.csv`:
```csv
timestamp,event_type,source_ip,severity
2026-10-01T00:03:00Z,FAILED_LOGON,192.168.1.42,2
2026-10-04T00:15:00Z,LATERAL_SMB,10.0.5.101,2
2026-10-04T02:30:00Z,SUSPICIOUS_CMD,10.0.5.101,3
2026-10-04T03:45:00Z,MASS_ENCRYPTION_BURST,10.0.5.101,4
```

### Step 3: Check Temporal Clustering
```bash
python src/analysis.py
```
Calculates Coefficient of Variation ($CV = \sigma / \mu > 1.2$) and Kolmogorov-Smirnov test against exponential baseline. Generates diagnostic plots in `reports/clustering_analysis.png`.

### Step 4: Run Chronological Backtesting & Acceptance Validation
```bash
python src/backtest.py
```
Splits data chronologically (80% train, 20% test), fits the Hawkes model, calibrates threshold, and computes Precision, Recall, Lead Time, and FPR. Generates `reports/backtest_results.md`.

### Step 5: Launch Streamlit SOC Dashboard
```bash
streamlit run app/dashboard.py
```
- Interactive slider for time replay and forecast evaluation.
- Live $\lambda(t)$ historical intensity and projected forecast curves.
- Dynamic threshold indicators and confidence metrics.
- One-click SOC containment playbook dispatch.

### Step 6: Run Full Unit Test Suite
```bash
pytest tests/ -v
```

---

## 6. Backtesting Benchmark Results

| Metric | Acceptance Target | Achieved Backtest Value | Benchmark Result |
| :--- | :--- | :--- | :--- |
| **Precision** | > 0.75 | **1.000** (100.0%) | **PASS** |
| **Recall** | > 0.70 | **1.000** (100.0%) | **PASS** |
| **Average Lead Time** | > 2.0 hours | **3.67 hours** | **PASS** |
| **F1 Score** | N/A | **1.000** | **PASS** |
| **False Positive Rate** | Low | **0.000** | **PASS** |
| **Stationarity ($\alpha < \beta$)** | Required | **$\eta = 0.773 < 1.0$** | **PASS** |

---

## 7. Actionable SOC Containment Recommendations
When an alert triggers, the advisory payload provides specific containment playbooks:
1. **Network Boundary Containment**: Isolate affected IP ranges; block SMB (TCP 445) and RDP (TCP 3389) across internal boundaries.
2. **Credential Revocation**: Revoke active Kerberos Ticket-Granting Tickets (TGTs) and rotate compromised service credentials.
3. **Backup Protection**: Lock Volume Shadow Copies (`vssadmin`) and offline immutable storage against modification.
4. **Endpoint Isolation**: Pre-stage EDR sensor host containment before file encryption commences.
