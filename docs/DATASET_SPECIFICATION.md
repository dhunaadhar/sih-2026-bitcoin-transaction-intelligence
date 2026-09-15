# M1 - Dataset Specification

## 1. Purpose

This document defines the canonical data contract for the SIH 2026 PS 26146 implementation. It specifies the fields, formats, validation requirements, provenance, and representation rules that must be satisfied before data enters the analysis pipeline.

## 2. Supported Input Formats

The final system shall accept bulk transaction and network metadata in:

- CSV
- JSON
- XML

All three formats shall be normalized into the same canonical internal representation.

## 3. Required Blockchain-Layer Fields

Each transaction record shall support:

- timestamp
- txid
- input wallet address
- input amount
- output wallet address
- output amount
- transaction fee
- script type

Multiple inputs and outputs shall be representable for a single transaction.

## 4. Required Network-Layer Fields

Network metadata shall support:

- timestamp
- source IP
- source port
- destination IP
- destination port
- txid
- geo_country
- asn

Network observations shall be correlated with blockchain transactions through the transaction identifier and temporal context.

## 5. Canonical Entity Types

The system shall represent at minimum:

- Transaction
- Wallet
- IP
- ASN
- Country

Entity identifiers shall be deterministic within a dataset and shall not be interpreted as proof of real-world identity.

## 6. Labels

Where ground-truth labels are available, they shall be stored separately from raw observations and clearly identified as labels. Label semantics and provenance shall be documented with the dataset.

The production pipeline shall also support unlabeled data because operational Bitcoin traffic may not contain ground-truth labels.

## 7. Temporal Representation

Timestamps shall be normalized to a documented canonical representation. The original timestamp shall be preserved where possible for provenance.

Derived temporal features may include transaction ordering, activity frequency, burst behaviour, time gaps, and other validated temporal characteristics.

## 8. Data Validation

The ingestion layer shall validate:

- Required fields
- Data types
- Timestamp validity
- IP address validity
- Port ranges
- Transaction identifier format
- Wallet/address presence
- Numeric amount and fee values
- Country and ASN representation
- Duplicate records
- Referential consistency

Invalid records shall not silently enter the analytical dataset.

## 9. Missing Values

Missing values shall be explicitly represented and recorded. The pipeline shall distinguish between:

- genuinely unavailable information
- malformed information
- information not applicable to a record

Imputation, when used, shall be documented and shall not create false network-to-wallet attribution.

## 10. Deduplication

Duplicate observations shall be detected using deterministic keys appropriate to the input format and record type. Deduplication shall preserve provenance and shall not merge distinct transactions merely because they share similar attributes.

## 11. Provenance

Every imported dataset shall retain provenance information sufficient to identify:

- source dataset
- source format
- ingestion time
- processing version
- transformation steps
- validation status

Synthetic or derived records shall be explicitly marked as such.

## 12. Synthetic SIH Dataset

The official problem statement specifies a synthetic dataset modelled on real Bitcoin P2P and transaction fields and does not provide real seized or live-intercept data.

The final implementation shall therefore provide a reproducible synthetic SIH dataset containing the required blockchain and network fields. Synthetic records must be clearly distinguished from captured traffic.

The existing Elliptic++ dataset may be used as a research/training source where appropriate, but it shall remain a read-only external source and shall not be duplicated into this repository.

## 13. Network Attribution Constraint

A network observation associated with a transaction shall be described as an observed or associated network event. The system shall not automatically claim that an IP address owns or controls a wallet solely because both were associated with the same transaction event.

## 14. Format Equivalence

CSV, JSON, and XML versions of equivalent records shall represent the same canonical data contract. Format conversion shall preserve identifiers, timestamps, amounts, relationships, and provenance semantics.

## 15. Dataset Quality Gate

A dataset shall be accepted for downstream feature engineering only after:

1. Schema validation passes.
2. Required-field validation passes.
3. Type and range validation passes.
4. Duplicate handling is completed.
5. Referential consistency is verified.
6. Provenance is recorded.
7. Synthetic/derived data is explicitly identified.

## 16. Model Independence

This specification does not select or prefer any machine-learning algorithm. Logistic Regression, XGBoost, Isolation Forest, and other appropriate candidates shall be evaluated quantitatively in the later ML benchmark milestone.
