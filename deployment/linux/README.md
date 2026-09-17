\# SIH 2026 Bitcoin Transaction Intelligence

\## Offline Linux Deployment



This directory contains the deployment assets required to run the SIH 2026 Bitcoin Transaction Intelligence system on an offline Linux environment.



\## Deployment Principles



\- Offline-first execution

\- No external API dependency

\- No internet requirement during runtime

\- Existing trained models and derived artifacts are reused

\- Configuration is externalized

\- Secrets are never stored in source code

\- The deployment must preserve the existing investigation pipeline, API, dashboard, monitoring, security, and notification workflows



\## Runtime Components



The deployed system consists of:



1\. Investigation data and derived artifacts

2\. Production XGBoost classifier

3\. Isolation Forest anomaly detector

4\. Behavioral detection engine

5\. Unified risk scoring

6\. SHAP explainability

7\. Investigation graph

8\. FastAPI backend

9\. Investigation dashboard

10\. Monitoring and health checks

11\. Controlled notification workflow



\## Linux Runtime



The primary runtime environment is Ubuntu/Linux with Python 3.11.



The application is designed to operate without internet connectivity after deployment.



\## Required Runtime Structure



```text

SIH-2026/

├── data/

├── models/

├── reports/

├── src/

├── tests/

├── deployment/

└── requirements.txt

