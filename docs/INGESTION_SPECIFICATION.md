\# Ingestion Specification



\## 1. Purpose



The ingestion layer accepts Bitcoin transaction and network metadata supplied in CSV, JSON, or XML format and converts it into a validated canonical record representation.



The ingestion pipeline is:



Input File

→ Format Reader

→ Field Normalization

→ Schema Validation

→ Exact Duplicate Removal

→ Provenance Generation

→ Validated Records



\## 2. Supported Formats



\- CSV

\- JSON

\- XML



All three formats have been tested using representative fixtures.



\## 3. Required Canonical Fields



Each validated record must contain:



\- timestamp

\- src\_ip

\- src\_port

\- dst\_ip

\- dst\_port

\- txid

\- input\_wallets

\- output\_wallets

\- input\_amount

\- output\_amount

\- fee

\- script\_type

\- geo\_country

\- asn



\## 4. Validation Rules



\### Network fields



\- `src\_ip` and `dst\_ip` must contain valid IPv4 or IPv6 addresses.

\- `src\_port` and `dst\_port` must be integers in the range 0–65535.



\### Transaction fields



\- `txid` must be a non-empty integer or string.

\- `input\_amount`, `output\_amount`, and `fee` must be numeric.

\- `input\_wallets` and `output\_wallets` must be lists of non-empty strings.



\### Timestamp



\- `timestamp` must be a non-empty ISO-8601 timestamp.



\### Metadata



The following fields must be non-empty strings:



\- `script\_type`

\- `geo\_country`

\- `asn`



\## 5. Normalization



The ingestion layer maps supported aliases to canonical field names.



Examples include:



\- `tx\_id` → `txid`

\- `transaction\_id` → `txid`

\- `source\_ip` → `src\_ip`

\- `destination\_ip` → `dst\_ip`

\- `source\_port` → `src\_port`

\- `destination\_port` → `dst\_port`

\- `fee\_btc` → `fee`

\- `input\_amount\_btc` → `input\_amount`

\- `output\_amount\_btc` → `output\_amount`



Wallet fields are normalized into list representations.



\## 6. Deduplication



Deduplication is based on the SHA-256 fingerprint of the complete normalized record.



Only exact canonical-record duplicates are removed.



Records sharing the same transaction ID are \*\*not automatically considered duplicates\*\*, because the same blockchain transaction may legitimately appear in multiple network observations.



\## 7. Provenance



Each ingestion operation records:



\- source file path

\- source format

\- source SHA-256

\- UTC ingestion timestamp

\- records read

\- valid records

\- invalid records

\- duplicate records removed



This preserves traceability between generated intelligence and its source data.



\## 8. Error Handling



Invalid records are rejected and reported rather than silently entering downstream processing.



Malformed or unreadable input files produce structured ingestion errors.



The ingestion layer therefore acts as a data-integrity boundary before canonicalization and feature engineering.



\## 9. Verification



M3 validation results:



\### Clean multi-format integration test



\- CSV: 2 read, 2 valid, 0 invalid, 0 duplicates

\- JSON: 2 read, 2 valid, 0 invalid, 0 duplicates

\- XML: 2 read, 2 valid, 0 invalid, 0 duplicates



\### Failure-case tests



\- Missing required field: PASSED

\- Invalid IP/port: PASSED

\- Exact duplicate removal: PASSED



Automated test result:



`3 passed`



\## 10. Design Boundary



The ingestion layer does not perform:



\- entity clustering

\- transaction graph construction

\- anomaly detection

\- ML prediction

\- risk scoring

\- identity attribution



Those functions belong to downstream components.



In particular, an IP address associated with a transaction must not automatically be interpreted as proof of ownership or identity of a wallet.

