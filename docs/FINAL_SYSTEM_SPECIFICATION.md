# SIH 2026 - Final System Specification

## Problem Statement
Smart India Hackathon 2026 - PS 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic.

## Production Model Status
UNDECIDED.

The final production machine-learning model shall not be selected in advance. Candidate models will be quantitatively benchmarked and the final model will be selected based on documented evaluation results.

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
- XGBoost and Isolation Forest are candidate techniques, not predetermined final choices.
- Automated experimentation/model selection shall be evaluated as part of the project.
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
