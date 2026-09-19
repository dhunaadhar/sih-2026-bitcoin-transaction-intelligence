# SIH 2026 — AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Final implementation for Smart India Hackathon 2026 — Problem Statement 26146.

An offline-first Bitcoin transaction intelligence platform combining machine learning, anomaly detection, behavioural analysis, graph/entity intelligence, network intelligence, explainable AI, unified risk scoring, ranked alerts, and investigator-oriented link analysis.

---

## Problem Statement

PS 26146 — AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

The system analyses bulk Bitcoin transaction and network metadata and correlates blockchain-layer information with network-layer observations where available.

The platform is designed for offline investigative analysis and produces explainable investigative intelligence rather than automatic identity or criminality determinations.

---

## Solution Overview

The platform implements the following end-to-end pipeline:

Source Data
↓
Canonical Data Layer
↓
Ingestion & Validation
↓
Feature Engineering
↓
Graph & Entity Intelligence
↓
Temporal-Safe ML
↓
Anomaly Detection
↓
Behavioural Intelligence
↓
Network Intelligence
↓
Unified Risk Engine
↓
SHAP Explainability
↓
Ranked Alerts
↓
Investigation Graph
↓
FastAPI Backend
↓
Investigation Dashboard
↓
Monitoring / Reporting / Notification
↓
Offline Linux / AppImage

---

## Core Capabilities

### Data

- CSV, JSON and XML ingestion
- Schema validation
- Field normalization and aliases
- Exact duplicate detection
- SHA-256 provenance tracking
- Canonical transaction representation
- Missing-value handling
- Investigator data import

### Transaction Intelligence

- Transaction-level statistical features
- Temporal activity features
- Wallet-level structural features
- Point-in-time wallet history
- Transaction behavioural indicators

### Graph & Entity Intelligence

- Address transaction graph
- Address-to-transaction relationships
- Input co-spend analysis
- Structural entity evidence
- Temporal-safe entity features
- Unified transaction-wallet-IP investigation graph

### AI / ML

- Logistic Regression benchmark
- Random Forest benchmark
- HistGradientBoosting benchmark
- XGBoost benchmark
- Temporal-safe chronological evaluation
- Production XGBoost classifier
- Isolation Forest anomaly detection
- SHAP-based explainability

### Behavioural Intelligence

- Peeling-chain detection
- Mixing-pattern detection
- Behavioural evidence fusion
- Strict temporal safeguards
- Deterministic chain reconstruction

### Network Intelligence

- IP correlation
- Port correlation
- Timing context
- Optional ASN/country intelligence
- Optional VPN/proxy/Tor/hosting intelligence
- Offline intelligence-feed support
- No fabricated network attribution

### Investigation

- Unified risk scoring
- Ranked alerts
- Confidence and evidence
- Severity bands
- Transaction investigation
- Wallet investigation
- Graph neighbourhood exploration
- Link analysis
- Controlled investigative reporting

### Reliability & Security

- Structured JSONL logging
- Health monitoring
- Watchdog monitoring
- Incident management
- External heartbeat monitoring
- Role-based access control
- Integrity verification
- Credential protection
- Offline-first operation
- Controlled notifications

### Deployment

- Ubuntu Linux deployment
- Offline installation workflow
- Terminal-based startup
- Deployment validation
- x86_64 AppImage distribution

---

## Dataset

The implementation uses the Elliptic++ dataset as a research/training source.

The original large source dataset is maintained outside this repository and is not duplicated into Git.

### Canonical Dataset

- 203,769 unique transactions
- 49 temporal steps
- 19 canonical columns
- 965 transactions with incomplete address relationships
- 202,804 transactions with both input/output address relationships

The canonical dataset is generated reproducibly from the source files.

### Source Graph

The address graph contains:

- 822,942 unique addresses
- 2,868,964 raw edge observations
- 2,784,344 unique directed edges after deduplication
- 8,707 unique self-loop edges
- 6,326 reciprocal directed edges

### Transaction-Address Relationships

- 477,117 input-address/transaction observations
- 837,124 transaction/output-address observations
- 202,804 transactions with both input and output relationships

### Labels

| Class | Count |
|---:|---:|
| 1 | 4,545 |
| 2 | 42,019 |
| 3 | 157,205 |

The implementation preserves the numeric class identifiers and does not assign unsupported semantic names.

---

## Feature Engineering

The unified feature matrix contains:

- 203,769 transactions
- 93 columns
- 92 analytical features + txid
- 0 duplicate TXIDs
- 0 infinite values

Feature groups include:

1. Transaction statistics
2. Temporal activity
3. Wallet structure
4. Point-in-time wallet history
5. Graph/entity evidence

Temporal history and graph features used for ML are constructed using only information available before the transaction's current time step.

Same-time-step and future-data leakage are explicitly prevented in the temporal-safe pipeline.

---

## Machine Learning Benchmark

The ML benchmark uses chronological splits:

| Split | Time steps | Transactions |
|---|---:|---:|
| Train | 1–29 | 120,804 |
| Validation | 30–39 | 36,318 |
| Test | 40–49 | 46,647 |

Candidate models were evaluated quantitatively rather than selecting a model in advance.

### Temporal-Safe Test Results

| Model | Accuracy | Macro F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.2266 | 0.2414 | 0.6840 | 0.4456 |
| Random Forest | 0.8005 | 0.4544 | 0.7856 | 0.5113 |
| HistGradientBoosting | 0.7367 | 0.4639 | 0.7662 | 0.5009 |
| XGBoost | 0.8007 | 0.4435 | 0.8019 | 0.5206 |

XGBoost is used as the production supervised classifier following the documented temporal-safe benchmark.

---

## Production ML Model

The production XGBoost package:

- Uses 87 model features
- Excludes leakage-sensitive temporal activity features
- Excludes identifiers
- Uses median imputation
- Adds missingness indicators
- Produces multiclass probabilities
- Stores model metadata and feature schema
- Is evaluated against the frozen temporal test set

The production model is an investigative classification component, not an automatic determination of illicit activity.

---

## Anomaly Detection

Isolation Forest provides an independent unsupervised anomaly signal.

Its output is treated as supporting evidence alongside supervised ML, graph, behavioural, temporal, and network evidence.

It is not interpreted as a direct classification of criminal or illicit behaviour.

---

## Behavioural Intelligence

### Peeling Chains

The detector uses strict causal reconstruction:

- Previous transaction output address reused as a later input
- Strictly earlier transaction time
- Bounded history window
- Transaction-level output-value reduction
- Deterministic longest-chain reconstruction

Validated result:

- 12,018 reconstructed peeling-chain candidates

### Mixing Patterns

Mixing detection combines structural evidence including:

- Fan-in
- Fan-out
- Balance
- Participant count
- Extreme structural evidence
- Interaction evidence

Validated result:

- 1,497 mixing candidates

Behavioural evidence represents observable transaction structure and does not establish ownership or criminal intent.

---

## Entity Intelligence

Input co-spend relationships are analysed at multiple thresholds rather than treating one threshold as a definitive identity boundary.

Repeated co-spend evidence includes:

- 14,532,245 unique address pairs
- Maximum shared transaction count: 39
- 10,518 addresses with repeated co-spend evidence

Entity evidence is incorporated into transaction-level features and investigation workflows.

A graph cluster is a structural association, not a confirmed real-world entity.

---

## Network Intelligence

The source Elliptic++ dataset does not contain IP, port, country, or ASN information.

Therefore the system treats network intelligence as an optional correlated layer.

Supported network evidence includes:

- Source/destination IP
- Source/destination port
- Timing
- Country
- ASN
- VPN classification
- Proxy classification
- Tor classification
- Hosting/datacenter classification

Network classifications are applied only when supported by an explicit offline intelligence feed.

Unknown information is never fabricated.

---

## Unified Risk Engine

The risk engine combines multiple evidence channels:

1. Production ML confidence
2. Isolation Forest anomaly evidence
3. Behavioural evidence
4. Temporal-safe entity/graph evidence
5. Network evidence

The result is an investigative prioritisation score from 0–100.

Alert bands:

| Score | Band |
|---:|---|
| 0–19 | LOW |
| 20–39 | GUARDED |
| 40–59 | MODERATE |
| 60–79 | HIGH |
| 80–100 | VERY_HIGH |

The score represents investigative prioritisation and must not be interpreted as a probability of criminality.

---

## Explainable AI

SHAP is integrated with the production XGBoost model.

For ranked alerts, the system can provide:

- Predicted class
- Model confidence
- Feature contributions
- Strongest positive/negative evidence
- Human-readable investigative context

SHAP explanations describe model behaviour and evidence contribution; they do not establish identity, ownership, or guilt.

---

## Ranked Alerts

The alert subsystem provides:

- Deterministic ranking
- Risk score
- Priority
- Severity
- Confidence
- Evidence-channel information
- Behavioural indicators
- Anomaly evidence
- Entity/graph evidence
- Network evidence
- Explainability

The ranked queue supports analyst-oriented investigation rather than automatic enforcement.

---

## Investigation Graph

The unified investigation graph links:

- Transactions
- Wallets
- IP observations
- Address-reuse continuity

Current graph scale:

- 1,026,715 nodes
- 1,379,970 edges

The graph supports transaction context, wallet context, neighbourhood exploration, shortest-path investigation, and top-risk investigation views.

---

## Dashboard

The investigation dashboard provides:

- System overview
- Alert exploration
- Transaction investigation
- Wallet investigation
- Temporal analysis
- Graph investigation
- Evidence inspection
- Report workflows

Cross-workspace transaction selection is synchronized so an investigated TXID can be carried between investigation and graph views without manual copy/paste.

---

## Backend API

The backend uses FastAPI.

The API exposes investigation services for:

- Summary
- Alerts
- Transaction investigation
- Wallet investigation
- Temporal analysis
- Graph investigation
- Reporting
- Import workflows
- Health and monitoring functions

The dashboard communicates with the backend through the local API.

---

## Monitoring & Reliability

Implemented reliability components include:

- Health checks
- Structured logging
- Watchdog monitoring
- Incident management
- External heartbeat
- Failure notification workflow

Complete host failure is handled through independent monitoring rather than relying solely on the failed application process.

---

## Security & Privacy

The platform is designed for offline investigative use.

Security controls include:

- Offline-first analytical operation
- Data minimisation
- Source-data separation
- SHA-256 integrity verification
- Provenance tracking
- Role-based access control
- Credential protection
- Structured security auditing
- Controlled report distribution
- Notification auditing

Sensitive credentials are not hard-coded into source files.

---

## Investigative Safeguards

The platform explicitly does not establish solely from its outputs:

- Real-world identity
- Wallet ownership
- Illicit activity
- Criminal guilt

Blockchain addresses are pseudonymous identifiers.

Graph relationships represent structural evidence.

Network association does not independently establish wallet ownership.

Risk scores and alerts are investigative leads requiring appropriate independent evidence and human review.

---

## Testing & Validation

The project contains unit, integration, data, ML, API, graph, security, and end-to-end tests.

Current validated status:

- 62 tests passed
- 3 warnings
- Graph regression tests: 8 passed
- Linux deployment validated
- Dashboard validated on Ubuntu
- AppImage built and executed successfully

The warnings are dependency deprecation warnings and do not represent test failures.

---

## Offline Linux Deployment

The repository contains the Linux deployment scripts and AppImage builder under deployment/.

The application can be deployed on Ubuntu Linux and distributed as an offline x86_64 AppImage.

---

## Quick Start — Development

Install dependencies:

pip install -r requirements.txt

Start the API:

uvicorn src.api.app:app --host 127.0.0.1 --port 8000

Open the dashboard:

http://127.0.0.1:8000/dashboard/

Run tests:

python -m pytest -q

---

## Quick Start — Linux Deployment

Run:

bash deployment/linux/scripts/validate_deployment.sh

Then:

bash deployment/linux/scripts/start.sh

The application runs locally and does not require a live Bitcoin network connection.

---

## AppImage

The project provides an x86_64 offline AppImage distribution.

Build using:

bash deployment/appimage/build_appimage.sh

The resulting executable is generated under:

deployment/appimage/dist/

---

## Repository Structure

.
├── data/
├── deployment/
├── docs/
├── evaluation/
├── models/
├── reports/
├── src/
│   ├── api/
│   ├── dashboard/
│   ├── graph/
│   ├── ingestion/
│   ├── ml/
│   ├── monitoring/
│   └── risk/
├── tests/
├── requirements.txt
└── README.md

The Bitcoin intelligence implementation is centred in the src/, data/, models/, evaluation/, reports/, deployment/, docs/, and tests/ components.

---

## Documentation

Detailed documentation is available under docs/.

Key documents include:

- FINAL_SYSTEM_SPECIFICATION.md
- DATASET_SPECIFICATION.md
- CANONICAL_DATASET_BUILD_SPECIFICATION.md
- INGESTION_SPECIFICATION.md
- ENTITY_CLUSTERING.md
- SECURITY_PRIVACY.md
- M22_TECHNICAL_DOCUMENTATION.md

---

## Limitations

1. Elliptic++ does not contain live network interception metadata.
2. Network intelligence therefore depends on optional offline feeds.
3. The source does not provide individual address-level BTC amounts for all address relationships.
4. Address relationships provide structural evidence rather than verified ownership.
5. Behavioural patterns are indicators and not proof of intent.
6. ML confidence and risk scores depend on the evaluated dataset and may change under distribution shift.
7. The system requires independent evidence for real-world attribution and enforcement decisions.

---

## Future Work

Potential extensions include:

- Expanded validated network-intelligence feeds
- Additional temporal graph models
- Broader independent datasets
- Improved investigator workflows
- Additional network and entity correlation sources
- Extended deployment targets
- Further robustness and distribution-shift evaluation

---

## Future Scope

The following enhancements are planned as the next stage of the platform beyond the current prototype implementation:

- **M26 — Native Desktop Application:** Extend the AppImage deployment so the application launches directly as a desktop application without requiring the user to manually open a browser or access a localhost URL.
- **M27 — Secure Investigator Data Dumps:** Treat every investigator-supplied CSV, JSON, and XML dump as untrusted input. Add file-type validation, size and filename checks, integrity hashing, malware scanning, schema validation, and safe parsing before data enters the intelligence pipeline.
- **M28 — High-Risk Transaction Drill-Down:** Make the high-risk transaction summary interactive so investigators can open the complete list of high-risk transactions and inspect individual transactions directly.
- **M29 — Authentication:** Provide an optional authentication layer for secured or operational deployments while retaining an authentication-disabled mode for controlled offline demonstrations.
- **M30 — Large-Monitor Dashboard Optimization:** Optimize the investigation dashboard for large monitoring displays and operational control-room environments.
- **M31 — Automatic Process Recovery:** Integrate process supervision and automatic restart mechanisms, including **Tron**, to improve application availability and recover automatically from unexpected process failures.
- **M32 — Synthetic Investigator Datasets:** Provide clearly labelled synthetic CSV, JSON, and XML datasets for demonstrating the complete investigator-data import workflow without relying on real sensitive network observations.

These enhancements are intended to improve operational security, deployment usability, resilience, investigator workflow, and demonstration coverage while preserving the platform's offline-first architecture.

## Project Status

### Completed

- M0 — Project foundation
- M1 — Source-data reconnaissance
- M2 — Canonical data layer
- M3 — Ingestion & validation
- M4 — Feature engineering
- M5 — Graph & entity intelligence
- M6 — ML benchmark & selection
- M7 — Production ML intelligence
- M8 — Anomaly detection
- M9 — Behavioural intelligence
- M10 — Network intelligence & explainability
- M11 — Unified risk & ranked alerts
- M12 — Investigation graph
- M13 — VPN/proxy/Tor intelligence
- M14 — Dashboard
- M15 — Backend/API
- M16 — Monitoring & reliability
- M17 — Authority notification
- M18 — Security & privacy
- M19 — Testing & validation
- M20 — Linux deployment
- M21 — AppImage

Current milestone: **M22 — Technical Documentation**

---

## Accountability Principle

The platform is designed to support investigators with structured, explainable intelligence.

It does not replace human investigation or legal processes.

Bitcoin transparency is not equivalent to real-world identity transparency.

System-generated clusters, behavioural indicators, network associations, model predictions, risk scores, and alerts must therefore be interpreted as evidence for investigation rather than as definitive attribution.

---

## License / Usage

This repository is developed as part of Smart India Hackathon 2026 Problem Statement 26146.

Refer to the project documentation for dataset provenance, usage limitations, deployment requirements, and security considerations.