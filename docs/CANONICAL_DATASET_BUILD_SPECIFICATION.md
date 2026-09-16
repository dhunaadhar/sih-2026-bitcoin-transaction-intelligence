# Canonical Dataset Build Specification

## Purpose

Define the transaction-centric canonical representation constructed from the Elliptic++ source for SIH 26146.

## Source

The source dataset is stored outside the final repository at data/external/elipticpp.
The source contains blockchain and actor relationships only. It does not contain captured IP, port, ASN, country, or packet-level network metadata.

## Transaction identity

- txid: integer transaction identifier.
- time_step: integer source time step, 1 through 49.

## Transaction amounts

The following source transaction-level fields are preserved:
- total_btc
- fee_btc
- input_btc_total
- output_btc_total
- input_btc_min
- input_btc_max
- input_btc_mean
- input_btc_median
- output_btc_min
- output_btc_max
- output_btc_mean
- output_btc_median

## Wallet relationships

Input and output wallet relationships are constructed from:
- AddrTx_edgelist.csv
- TxAddr_edgelist.csv

The relationship files contain identifiers only and do not contain individual wallet amounts.
Individual wallet-to-amount mappings must therefore not be fabricated.

## Network layer

Network observations are maintained as a separate canonical layer.
Elliptic++ does not provide IP addresses, ports, ASN, country, or packet-level timestamps.
Any supplementary network data must carry explicit provenance and must not be represented as captured traffic unless supported by source evidence.

## Missing relationships

Transactions without address-edge representation must be retained in the transaction table and explicitly marked rather than silently discarded.

## Source-specific model features

Local_feature_* and Aggregate_feature_* columns remain available for model experimentation but are not treated as human-interpretable canonical facts.

## Provenance

Every derived artifact must identify its source files, transformation stage, and generation metadata.

## Model independence

The canonical dataset must not encode a preferred machine-learning model. Model selection is a later benchmark stage.
