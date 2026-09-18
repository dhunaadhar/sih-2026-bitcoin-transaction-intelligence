# Bitcoin Transaction Intelligence Platform

## SIH 2026 - Problem Statement 26146

## 1. System Overview

The Bitcoin Transaction Intelligence Platform is an offline-first investigative analytics system for monitoring and analysing Bitcoin transaction traffic. It combines transaction-level machine learning, temporal anomaly detection, behavioural analysis, graph/entity intelligence, network intelligence, explainability, unified risk scoring, ranked alerts, and investigator-oriented graph exploration.

The platform is designed for offline Linux execution and does not claim to establish real-world identity, ownership, illicit activity, or guilt from blockchain or network evidence alone.

## 2. End-to-End Architecture

The system follows the pipeline:

Source Data -> Canonical Data -> Validation -> Feature Engineering -> Graph/Entity Intelligence -> Temporal-Safe ML -> Anomaly Detection -> Behavioural Intelligence -> Network Intelligence -> Unified Risk -> Explainability -> Ranked Alerts -> Investigation Graph -> Dashboard/API -> Monitoring/Notification -> Offline Deployment

## 3. Data Layer

The canonical transaction dataset contains 203,769 transactions across 49 temporal steps. The source Elliptic++ data provides transaction, address, wallet-feature, and class information. The source dataset does not provide IP, port, country, or ASN fields; therefore network-layer intelligence is implemented as an optional correlated layer rather than fabricated source data.

The canonical dataset contains transaction-level aggregate BTC and fee statistics and address relationships. Individual address-level BTC amounts are not available in the source data.

## 4. Feature Engineering

Feature groups include transaction statistics, temporal activity, wallet statistics, point-in-time wallet history, and temporal-safe graph/entity evidence.

Temporal history features are constructed without allowing future transactions or same-time-step transactions to influence the current transaction.

## 5. Graph and Entity Intelligence

Address relationships are represented using directed address graphs and transaction-address relationships. Co-spend relationships provide structural association evidence. Connected components and repeated co-spend structures are treated as investigative evidence only; a graph cluster is not interpreted as a real-world identity.

## 6. Machine Learning

Model benchmarking uses chronological train, validation, and frozen test periods. Candidate models include Logistic Regression, Random Forest, HistGradientBoosting, and XGBoost.

The production classifier is XGBoost selected using temporal-safe benchmark results. The production model uses 87 model features after excluding leakage-sensitive features and identifiers.

## 7. Anomaly Detection

Isolation Forest provides an independent unsupervised anomaly channel. It is used as supporting evidence and is not treated as a direct classification of illicit activity.

## 8. Behavioural Intelligence

Behavioural detection includes hardened peeling-chain and mixing-pattern indicators. Peeling chains require strict temporal predecessor relationships and transaction-level output-value reduction evidence. Mixing candidates require multiple structural conditions including fan-in, fan-out, balance, and participant evidence.

## 9. Network Intelligence

Network intelligence correlates optional IP/port/timing/ASN/country information with transaction intelligence. VPN, proxy, Tor, and hosting classifications are only applied when supported by an offline intelligence feed. Unknown network intelligence is not fabricated.

## 10. Explainability

SHAP explanations provide feature-level contributions for ranked investigative alerts. Explanations describe model evidence contributing to prioritisation and are not interpreted as proof of criminal activity or identity.

## 11. Unified Risk and Alerts

The unified risk engine combines independent evidence channels into a 0-100 investigative prioritisation score. Alert bands are LOW, GUARDED, MODERATE, HIGH, and VERY_HIGH.

The ranked alert queue provides priority, confidence, severity, evidence channels, behavioural indicators, anomaly evidence, graph/entity evidence, network evidence, and explainability.

## 12. Investigation Graph

The unified graph links transactions, wallets, IP observations, and address-reuse continuity relationships. Investigators can move from alerts to transaction context, wallet context, graph neighbourhoods, and linked evidence.

## 13. Dashboard and API

The dashboard provides investigation-oriented views for alerts, transactions, wallets, temporal intelligence, graph exploration, and reporting. FastAPI exposes the backend data and investigation services.

## 14. Monitoring and Reliability

The platform includes health checks, structured logging, watchdog monitoring, incident management, and external heartbeat support.

## 15. Controlled Notification

Authority reporting and notification workflows are controlled and evidence-based. Reports explicitly avoid unsupported identity, ownership, illicitness, or guilt conclusions.

## 16. Security and Privacy

The platform is designed for offline operation, data minimisation, provenance tracking, access control, integrity verification, and controlled reporting. See SECURITY_PRIVACY.md for the detailed security baseline.

## 17. Deployment

Linux deployment is supported through the deployment/linux scripts. The application can also be distributed as an offline AppImage for x86_64 Linux systems.

## 18. Validation

The project includes unit, integration, data, ML, API, graph, security, and end-to-end tests. The validated Windows test suite currently contains 62 passing tests with 3 warnings.

## 19. Limitations

The source dataset is historical and does not contain real-time network interception fields. Network intelligence is therefore optional and offline-feed driven. Blockchain addresses are pseudonymous identifiers, not verified real-world identities. Structural graph relationships do not establish ownership. Behavioural patterns and risk scores are investigative signals rather than legal or criminal determinations.

## 20. Reproducibility

All major processing stages produce deterministic artifacts, validation reports, model metadata, feature manifests, and provenance information. Chronological evaluation protects the primary ML benchmark from future-data leakage.

## 21. Future Work

Future development can extend network intelligence coverage, integrate additional validated external intelligence feeds, improve temporal graph modelling, expand investigator workflows, and evaluate the system against additional independently sourced datasets.
