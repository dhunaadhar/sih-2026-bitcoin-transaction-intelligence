SIH 2026 — AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Final implementation for Smart India Hackathon 2026 — Problem Statement 26146.

An offline-first Bitcoin transaction intelligence platform combining machine learning, anomaly detection, behavioural analysis, graph/entity intelligence, network intelligence, explainable AI, unified risk scoring, ranked alerts, investigator-oriented link analysis, secure local authentication, process supervision, reproducible investigator-data ingestion, and an offline Investigation Assistant.

Problem Statement

PS 26146 — AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

The system analyses bulk Bitcoin transaction and network metadata and correlates blockchain-layer information with network-layer observations where available.

The platform is designed for offline investigative analysis and produces explainable investigative intelligence rather than automatic identity or criminality determinations.

Solution Overview

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
Investigator Data Import
↓
Imported Case Intelligence
↓
Offline Investigation Assistant
↓
Monitoring / Reporting / Notification
↓
Offline Linux / AppImage

Core Capabilities

Data

CSV, JSON and XML ingestion

Schema validation

Field normalization and aliases

Exact duplicate detection

SHA-256 provenance tracking

Canonical transaction representation

Missing-value handling

Investigator data import

Transaction Intelligence

Transaction-level statistical features

Temporal activity features

Wallet-level structural features

Point-in-time wallet history

Transaction behavioural indicators

Graph & Entity Intelligence

Address transaction graph

Address-to-transaction relationships

Input co-spend analysis

Structural entity evidence

Temporal-safe entity features

Unified transaction-wallet-IP investigation graph

AI / ML

Logistic Regression benchmark

Random Forest benchmark

HistGradientBoosting benchmark

XGBoost benchmark

Temporal-safe chronological evaluation

Production XGBoost classifier

Isolation Forest anomaly detection

SHAP-based explainability

Behavioural Intelligence

Peeling-chain detection

Mixing-pattern detection

Behavioural evidence fusion

Strict temporal safeguards

Deterministic chain reconstruction

Network Intelligence

IP correlation

Port correlation

Timing context

Optional ASN/country intelligence

Optional VPN/proxy/Tor/hosting intelligence

Offline intelligence-feed support

No fabricated network attribution

Investigation

Unified risk scoring

Ranked alerts

Confidence and evidence

Severity bands

Transaction investigation

Wallet investigation

Graph neighbourhood exploration

Link analysis

Controlled investigative reporting

Investigator Data Import

CSV, JSON and XML investigator-data ingestion

Schema-aware validation and normalization

Duplicate detection

SHA-256 provenance tracking

Deterministic import identifiers

Persisted import manifests

Imported-case intelligence analysis

Explicit evidence-availability tracking

Investigation Assistant

Dedicated dashboard Assistant workspace

Conversational investigator interaction

Case-level and transaction-level questions

Risk and behavioural evidence queries

Evidence-availability queries

Transaction lookup

Evidence-based next-step guidance

Deterministic local NLP

No cloud model dependency

No fabricated evidence

Reliability & Security

Structured JSONL logging

Health monitoring

Watchdog monitoring

Incident management

External heartbeat monitoring

Role-based access control

Local session authentication

PBKDF2 password protection

Session expiry

Integrity verification

Credential protection

Bounded process restart supervision

Heartbeat and incident state tracking

Offline-first operation

Controlled notifications

Deployment

Ubuntu Linux deployment

Offline installation workflow

Terminal-based startup

Deployment validation

x86_64 AppImage distribution

Dataset

The implementation uses the Elliptic++ dataset as a research/training source.

The original large source dataset is maintained outside this repository and is not duplicated into Git.

Canonical Dataset

203,769 unique transactions

49 temporal steps

19 canonical columns

965 transactions with incomplete address relationships

202,804 transactions with both input/output address relationships

The canonical dataset is generated reproducibly from the source files.

Source Graph

The address graph contains:

822,942 unique addresses

2,868,964 raw edge observations

2,784,344 unique directed edges after deduplication

8,707 unique self-loop edges

6,326 reciprocal directed edges

Transaction-Address Relationships

477,117 input-address/transaction observations

837,124 transaction/output-address observations

202,804 transactions with both input and output relationships

Labels

Class

Count

1

4,545

2

42,019

3

157,205

The implementation preserves the numeric class identifiers and does not assign unsupported semantic names.

Feature Engineering

The unified feature matrix contains:

203,769 transactions

93 columns

92 analytical features + txid

0 duplicate TXIDs

0 infinite values

Feature groups include:

Transaction statistics

Temporal activity

Wallet structure

Point-in-time wallet history

Graph/entity evidence

Temporal history and graph features used for ML are constructed using only information available before the transaction's current time step.

Same-time-step and future-data leakage are explicitly prevented in the temporal-safe pipeline.

Machine Learning Benchmark

The ML benchmark uses chronological splits:

Split

Time steps

Transactions

Train

1–29

120,804

Validation

30–39

36,318

Test

40–49

46,647

Candidate models were evaluated quantitatively rather than selecting a model in advance.

Temporal-Safe Test Results

Model

Accuracy

Macro F1

ROC-AUC

PR-AUC

Logistic Regression

0.2266

0.2414

0.6840

0.4456

Random Forest

0.8005

0.4544

0.7856

0.5113

HistGradientBoosting

0.7367

0.4639

0.7662

0.5009

XGBoost

0.8007

0.4435

0.8019

0.5206

XGBoost is used as the production supervised classifier following the documented temporal-safe benchmark.

Production ML Model

The production XGBoost package:

Uses 87 model features

Excludes leakage-sensitive temporal activity features

Excludes identifiers

Uses median imputation

Adds missingness indicators

Produces multiclass probabilities

Stores model metadata and feature schema

Is evaluated against the frozen temporal test set

The production model is an investigative classification component, not an automatic determination of illicit activity.

Anomaly Detection

Isolation Forest provides an independent unsupervised anomaly signal.

Its output is treated as supporting evidence alongside supervised ML, graph, behavioural, temporal, and network evidence.

It is not interpreted as a direct classification of criminal or illicit behaviour.

Behavioural Intelligence

Peeling Chains

The detector uses strict causal reconstruction:

Previous transaction output address reused as a later input

Strictly earlier transaction time

Bounded history window

Transaction-level output-value reduction

Deterministic longest-chain reconstruction

Validated result:

12,018 reconstructed peeling-chain candidates

Mixing Patterns

Mixing detection combines structural evidence including:

Fan-in

Fan-out

Balance

Participant count

Extreme structural evidence

Interaction evidence

Validated result:

1,497 mixing candidates

Behavioural evidence represents observable transaction structure and does not establish ownership or criminal intent.

Entity Intelligence

Input co-spend relationships are analysed at multiple thresholds rather than treating one threshold as a definitive identity boundary.

Repeated co-spend evidence includes:

14,532,245 unique address pairs

Maximum shared transaction count: 39

10,518 addresses with repeated co-spend evidence

Entity evidence is incorporated into transaction-level features and investigation workflows.

A graph cluster is a structural association, not a confirmed real-world entity.

Network Intelligence

The source Elliptic++ dataset does not contain IP, port, country, or ASN information.

Therefore the system treats network intelligence as an optional correlated layer.

Supported network evidence includes:

Source/destination IP

Source/destination port

Timing

Country

ASN

VPN classification

Proxy classification

Tor classification

Hosting/datacenter classification

Network classifications are applied only when supported by an explicit offline intelligence feed.

Unknown information is never fabricated.

Unified Risk Engine

The risk engine combines multiple evidence channels:

Production ML confidence

Isolation Forest anomaly evidence

Behavioural evidence

Temporal-safe entity/graph evidence

Network evidence

The result is an investigative prioritisation score from 0–100.

Alert bands:

Score

Band

0–19

LOW

20–39

GUARDED

40–59

MODERATE

60–79

HIGH

80–100

VERY_HIGH

The score represents investigative prioritisation and must not be interpreted as a probability of criminality.

Explainable AI

SHAP is integrated with the production XGBoost model.

For ranked alerts, the system can provide:

Predicted class

Model confidence

Feature contributions

Strongest positive/negative evidence

Human-readable investigative context

SHAP explanations describe model behaviour and evidence contribution; they do not establish identity, ownership, or guilt.

Ranked Alerts

The alert subsystem provides:

Deterministic ranking

Risk score

Priority

Severity

Confidence

Evidence-channel information

Behavioural indicators

Anomaly evidence

Entity/graph evidence

Network evidence

Explainability

The ranked queue supports analyst-oriented investigation rather than automatic enforcement.

Investigation Graph

The unified investigation graph links:

Transactions

Wallets

IP observations

Address-reuse continuity

Current graph scale:

1,026,715 nodes

1,379,970 edges

The graph supports transaction context, wallet context, neighbourhood exploration, shortest-path investigation, and top-risk investigation views.

Dashboard

The investigation dashboard provides:

System overview

Alert exploration

Transaction investigation

Wallet investigation

Temporal analysis

Graph investigation

Evidence inspection

Report workflows

Offline Investigation Assistant

Cross-workspace transaction selection is synchronized so an investigated TXID can be carried between investigation and graph views without manual copy/paste.

Backend API

The backend uses FastAPI.

The API exposes investigation services for:

Summary

Alerts

Transaction investigation

Wallet investigation

Temporal analysis

Graph investigation

Reporting

Import workflows

Health and monitoring functions

The dashboard communicates with the backend through the local API. Protected API routes require an authenticated local session.

Monitoring & Reliability

Implemented reliability components include:

Health checks

Structured logging

Watchdog monitoring

Incident management

External heartbeat

Failure notification workflow

Complete host failure is handled through independent monitoring rather than relying solely on the failed application process.

Security & Privacy

The platform is designed for offline investigative use.

Security controls include:

Offline-first analytical operation

Data minimisation

Source-data separation

SHA-256 integrity verification

Provenance tracking

Role-based access control

Credential protection

Structured security auditing

Controlled report distribution

Notification auditing

Sensitive credentials are not hard-coded into source files.

Investigative Safeguards

The platform explicitly does not establish solely from its outputs:

Real-world identity

Wallet ownership

Illicit activity

Criminal guilt

Blockchain addresses are pseudonymous identifiers.

Graph relationships represent structural evidence.

Network association does not independently establish wallet ownership.

Risk scores and alerts are investigative leads requiring appropriate independent evidence and human review.

Testing & Validation

The project contains unit, integration, data, ML, API, graph, security, import, case-intelligence, and end-to-end tests.

Current validated regression status:

65 tests passed

0 test failures

3 warnings

API endpoint tests: 8 passed

End-to-end tests: 4 passed

Investigator import tests validated against CSV, JSON and XML

Synthetic investigator dataset generation validated

Imported case intelligence validated

Authentication-aware API and end-to-end tests validated

The three warnings are existing FastAPI lifecycle deprecation warnings related to on_event and do not represent test failures.

Offline Linux Deployment

The repository contains the Linux deployment scripts and AppImage builder under deployment/.

The application can be deployed on Ubuntu Linux and distributed as an offline x86_64 AppImage.

Quick Start — Development

Install dependencies:

pip install -r requirements.txt

Start the API:

uvicorn src.api.app --host 127.0.0.1 --port 8000

Open the dashboard:

http://127.0.0.1:8000/dashboard/

Run tests:

python -m pytest -q

Quick Start — Linux Deployment

Run:

bash deployment/linux/scripts/validate_deployment.sh

Then:

bash deployment/linux/scripts/start.sh

The application runs locally and does not require a live Bitcoin network connection.

AppImage

The project provides an x86_64 offline AppImage distribution.

Build using:

bash deployment/appimage/build_appimage.sh

The resulting executable is generated under:

deployment/appimage/dist/

Repository Structure

.
├── data/
├── deployment/
├── docs/
├── evaluation/
├── models/
├── reports/
├── src/
│   ├── api/
│   │   ├── app.py
│   │   └── assistant_routes.py
│   ├── dashboard/
│   │   ├── index.html
│   │   ├── assistant.js
│   │   └── ...
│   ├── graph/
│   ├── ingestion/
│   ├── ml/
│   ├── monitoring/
│   │   └── process_supervisor.py
│   └── risk/
├── tests/
├── requirements.txt
└── README.md

The Bitcoin intelligence implementation is centred in the src/, data/, models/, evaluation/, reports/, deployment/, docs/, and tests/ components.

Documentation

Detailed documentation is available under docs/.

Key documents include:

FINAL_SYSTEM_SPECIFICATION.md

DATASET_SPECIFICATION.md

CANONICAL_DATASET_BUILD_SPECIFICATION.md

INGESTION_SPECIFICATION.md

ENTITY_CLUSTERING.md

SECURITY_PRIVACY.md

M22_TECHNICAL_DOCUMENTATION.md

Limitations

Elliptic++ does not contain live network interception metadata.

Network intelligence therefore depends on optional offline feeds.

The source does not provide individual address-level BTC amounts for all address relationships.

Address relationships provide structural evidence rather than verified ownership.

Behavioural patterns are indicators and not proof of intent.

ML confidence and risk scores depend on the evaluated dataset and may change under distribution shift.

The system requires independent evidence for real-world attribution and enforcement decisions.

Future Work

Potential extensions include:

Expanded validated network-intelligence feeds

Additional temporal graph models

Broader independent datasets

Improved investigator workflows

Additional network and entity correlation sources

Extended deployment targets

Further robustness and distribution-shift evaluation

Current Roadmap & Future Scope

The project has moved beyond the original research/prototype milestones and is now in finalisation and operational-hardening work.

Completed Milestones

M0–M25 — Core platform, intelligence pipeline, dashboard, deployment, validation and final SIH package preparation

M29 — Local authentication and protected API sessions

M31 — Process supervision, bounded restart handling and heartbeat support

M32 — Reproducible synthetic CSV/JSON/XML investigator datasets

M34 — Offline Local Investigation Assistant and dashboard integration (UNDER PROGRESS)

Current Work

M33 — Final Linux/AppImage validation

Final offline packaging and deployment validation remains part of the finalisation cycle.

M34 — Offline Local Investigation Assistant (UNDER PROGRESS)

Backend and dashboard integration are implemented.

Deterministic local NLP is used.

Assistant responses are constrained to imported case evidence.

Current regression suite: 65 passed, 0 failed.

M34 Assistant Architecture

Investigator
     │
     ▼
Dashboard Assistant
     │
     ▼
Local FastAPI /api/assistant/query
     │
     ▼
Imported Case Artifact
     │
     ▼
Case Intelligence Analysis
     │
     ├── Risk
     ├── Behavioural Evidence
     ├── Entity Evidence
     ├── Network Context
     └── Evidence Availability
     │
     ▼
Deterministic Local Response

The assistant is intentionally not implemented as an unrestricted cloud generative-AI interface. It operates locally and provides evidence-grounded investigative responses.

Remaining Finalisation

Complete final Ubuntu/AppImage validation

Replace temporary development credentials before final deployment

Final security and operational audit

Final SIH demonstration package

Final repository cleanup and release checkpoint

Current Implementation Snapshot

The current platform includes:

Offline Bitcoin transaction intelligence pipeline

Temporal-safe ML and anomaly detection

Behavioural, entity and network evidence

Unified risk scoring and ranked alerts

SHAP explainability

Investigation graph

FastAPI backend

Investigator CSV/JSON/XML import

Imported-case intelligence

Local authentication and protected API routes

Process supervision and heartbeat support

Native dashboard Assistant workspace

Deterministic offline Investigation Assistant

Reproducible synthetic investigator datasets

65 automated tests passing with 0 failures

The Investigation Assistant is deliberately evidence-constrained: when a case lacks a particular evidence channel, the assistant reports that limitation instead of inventing an interpretation.

Accountability Principle

The platform is designed to support investigators with structured, explainable intelligence.

It does not replace human investigation or legal processes.

Bitcoin transparency is not equivalent to real-world identity transparency.

System-generated clusters, behavioural indicators, network associations, model predictions, risk scores, and alerts must therefore be interpreted as evidence for investigation rather than as definitive attribution.

License / Usage

This repository is developed as part of Smart India Hackathon 2026 Problem Statement 26146.

Refer to the project documentation for dataset provenance, usage limitations, deployment requirements, and security considerations.