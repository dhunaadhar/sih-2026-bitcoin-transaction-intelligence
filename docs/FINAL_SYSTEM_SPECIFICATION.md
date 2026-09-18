# SIH 2026 - Final System Specification

## Problem Statement
Smart India Hackathon 2026 - PS 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic.

## Production Model Status
VALIDATED — XGBoost.

The production supervised classifier was selected after quantitative temporal-safe benchmarking. XGBoost is the production classification model, while Isolation Forest provides an independent unsupervised anomaly channel. The benchmark and selection evidence are documented in the M6 ML evaluation reports.

## Core Requirements

- Offline Linux deployment.
- Bulk Bitcoin transaction and network metadata ingestion.
- CSV, JSON, and XML input support.
- Schema validation, normalization, deduplication, and provenance tracking.
- Correlation of network-layer IP, port, and timing information with blockchain-layer wallet and transaction information.
- Unified graph linking IPs, wallets, transactions, and relevant network entities.
- Entity resolution and entity clustering.
- Transaction, wallet, temporal, network, and graph feature engineering.
- AI/ML-based detection rather than rules alone.
- Quantitative comparison of multiple ML models.
- Logistic Regression must be included as a benchmark candidate.
- XGBoost is the validated production supervised classifier; Isolation Forest is the production unsupervised anomaly detector.
- Model selection is supported by the completed temporal-safe quantitative benchmark.
- Anomaly detection.
- Peeling-chain detection.
- Mixing-pattern detection.
- Behavioral-pattern detection.
- Risk scoring and graph-based risk propagation.
- Explainable ranked alerts with confidence and supporting evidence.
- Dashboard and link-analysis visualization.
- Structured logging.
- Health monitoring and failure detection.
- Immediate failure/crash notification to the concerned authority or configured recipient.
- VPN/network intelligence analysis where applicable.
- Linux deployment and Ubuntu VM startup through the terminal.
- Final distributable AppImage.

## Data Requirements

The system must support the required transaction and network fields specified by PS 26146, including timestamp, source/destination IP and ports, transaction ID, input/output wallet addresses and amounts, transaction fee, script type, geographic country, and ASN where available.

The final SIH dataset will use synthetic/modelled data based on the specified Bitcoin P2P and transaction fields because the official problem statement does not provide real seized or live-intercept data.

The existing Elliptic++ dataset is retained outside the final repository as a large read-only source dataset and must not be duplicated into the final project.

## Accountability and Transparency

Bitcoin blockchain transparency does not automatically provide real-world identity transparency. The system therefore converts pseudonymous transaction, network, temporal, and graph evidence into investigative intelligence through entity clustering, graph analysis, behavioral analysis, anomaly detection, and explainable risk ranking.

System outputs are investigative leads and risk indicators, not automatic proof of identity, ownership, or criminal guilt.

## Model Selection Principle

No model shall be considered superior without quantitative verification on the project's evaluation data. Candidate models will be compared using appropriate metrics including precision, recall, F1, PR-AUC, ROC-AUC, ranking quality, calibration where applicable, computational cost, and robustness to temporal drift.

## Implementation Principle

Every major milestone shall follow:
1. Implement.
2. Test.
3. Document.
4. Quantitatively validate.
5. Commit to Git.


## 22. Implemented Production Status

The implementation has progressed from the initial specification to a validated end-to-end investigative platform.

### Data and intelligence pipeline

- Canonical transaction dataset: 203,769 transactions across 49 time steps.
- Transaction feature matrix: 93 columns including `txid`.
- Temporal-safe graph/entity evidence is used for ML evaluation.
- Address graph: 822,942 unique addresses and 2,784,344 unique directed edges.
- Unified investigation graph: 1,026,715 nodes and 1,379,970 edges.

### Machine learning

Multiple models were benchmarked using chronological train/validation/test splits. The frozen temporal-safe test set contains 46,647 transactions.

The production supervised classifier is XGBoost, trained after temporal-safe benchmarking. The production package uses 87 model features and excludes leakage-sensitive temporal activity features and identifiers.

The temporal-safe XGBoost benchmark achieved test accuracy 0.8007, Macro F1 0.4435, ROC-AUC 0.8019, PR-AUC 0.5206, and LogLoss 0.4977.

Isolation Forest provides an independent unsupervised anomaly channel.

### Behavioural intelligence

Peeling-chain and mixing-pattern detectors are implemented with explicit temporal and structural safeguards. The behavioural layer reconstructed 12,018 valid peeling-chain candidates and 1,497 mixing candidates from the available source relationships.

### Explainability and risk intelligence

SHAP-based explanations are generated for ranked alerts. The unified risk engine combines supervised ML, anomaly, behavioural, temporal-safe entity/graph, and network evidence into an investigative prioritisation score from 0 to 100.

Ranked alerts provide priority, confidence, severity, supporting evidence, and analyst-facing explanations.

### Network intelligence

Network intelligence supports optional offline correlation of IP and related network metadata. VPN, proxy, Tor, and hosting classifications are only applied when supported by an explicit intelligence feed. Unknown network intelligence is not fabricated.

### Application and deployment

The system provides a FastAPI backend, investigation dashboard, monitoring and reliability components, controlled reporting/notification workflows, Linux deployment scripts, and an offline x86_64 AppImage.

### Validation status

The current automated test suite contains 62 passing tests with 3 warnings. Graph-specific regression tests contain 8 passing tests. Linux deployment and AppImage execution have also been manually validated.

The production system remains an investigative intelligence platform. Its outputs do not establish real-world identity, wallet ownership, illicit activity, or criminal guilt without independent evidence.
