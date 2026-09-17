\# Security \& Privacy



\## 1. Security Objectives



The system is designed for offline Bitcoin transaction intelligence and investigative analysis.



Primary objectives:



\- Preserve source-data integrity.

\- Minimize unnecessary data exposure.

\- Prevent credential disclosure.

\- Maintain auditable investigative actions.

\- Separate raw, derived, model, and reporting data.

\- Restrict privileged system operations.

\- Preserve provenance of imported and generated data.

\- Avoid treating model output as proof of identity, ownership, illicit activity, or guilt.



\## 2. Offline-First Security



The core analytical pipeline operates without requiring internet connectivity.



External network intelligence is optional and must be supplied through controlled offline feeds.



No live Bitcoin network connection is required for the analytical pipeline.



\## 3. Data Minimization



The system processes only fields required for:



\- transaction analysis,

\- temporal analysis,

\- wallet/entity analysis,

\- network correlation,

\- anomaly detection,

\- behavioral analysis,

\- risk scoring,

\- explainability,

\- investigation graph construction.



Raw source data is kept separate from derived analytical artifacts.



\## 4. Credential Protection



SMTP and notification credentials are not hard-coded into source files.



Supported configuration variables include:



\- `SMTP\_HOST`

\- `SMTP\_PORT`

\- `SMTP\_USERNAME`

\- `SMTP\_PASSWORD`

\- `SMTP\_SENDER`

\- `NOTIFICATION\_RECIPIENTS`



Sensitive values must be supplied through protected deployment configuration.



Passwords and credentials must never be written to logs or reports.



\## 5. Access Control



The system defines three operational roles:



\### ANALYST



Can:



\- read alerts,

\- read graph information,

\- read risk information,

\- export investigative reports,

\- manage incidents.



Cannot:



\- manage system configuration.



\### AUTHORITY



Can:



\- read alerts,

\- read graph information,

\- read risk information,

\- export investigative reports.



Cannot:



\- manage incidents,

\- manage system configuration.



\### ADMIN



Can:



\- access analytical information,

\- export reports,

\- manage incidents,

\- manage system configuration.



Unknown roles and undefined permissions are denied by default.



\## 6. Provenance \& Integrity



Source provenance is maintained through the dataset source manifest.



SHA-256 hashing is supported for file-integrity verification.



The pipeline does not modify original source datasets.



Derived datasets and audit reports are stored separately.



\## 7. Audit Logging



Security-relevant events are recorded with:



\- UTC timestamp,

\- event,

\- actor,

\- action,

\- relevant non-sensitive details,

\- execution mode.



Secrets are explicitly excluded from security audit records.



\## 8. Notification Security



Notifications are controlled and auditable.



Every notification attempt records:



\- notification ID,

\- timestamp,

\- recipient,

\- channel,

\- subject,

\- status,

\- error information where applicable.



Notification delivery is not represented as guaranteed.



For complete host failure, notification must be performed by an independent monitoring system rather than relying on the failed host itself.



\## 9. Investigative Safeguards



The system generates investigative intelligence rather than establishing real-world identity.



The following are explicitly not established solely by system output:



\- real-world identity,

\- wallet ownership,

\- illicit activity,

\- guilt.



Graph clustering represents structural association and does not establish that multiple addresses belong to the same person or organization.



Risk scores represent investigative prioritization signals.



\## 10. Threat Model



The system considers:



| Threat | Mitigation |

|---|---|

| Credential leakage | Environment-based configuration |

| Source-data tampering | SHA-256 provenance/integrity checks |

| Unauthorized privileged operations | Role-based access control |

| Sensitive information in logs | Structured audit logging without secrets |

| Accidental source modification | Raw/derived data separation |

| Invalid investigator input | Schema validation and normalization |

| Application failure | Health monitor + watchdog |

| Host failure | Independent heartbeat monitoring |

| Notification failure | Auditable notification-attempt records |

| Misinterpretation of risk | Explicit investigative limitations |

| Uncontrolled report distribution | Controlled export workflow |



\## 11. Security Validation



Implemented security validation components:



\- Security baseline

\- Integrity verification

\- Access-control validation

\- Security audit logging

\- Credential exposure checks

\- Provenance verification



All security validation must pass before final deployment.

