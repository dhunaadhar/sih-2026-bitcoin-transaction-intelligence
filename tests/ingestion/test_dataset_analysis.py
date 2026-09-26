from __future__ import annotations

from src.analysis.dataset_analysis import analyze_dataset


def test_analyze_dataset_with_financial_and_wallet_fields():
    records = [
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "txid": "tx1",
            "input_wallets": ["wallet_in_1"],
            "output_wallets": ["wallet_out_1"],
            "input_amount": 2.5,
            "output_amount": 2.49,
            "fee": 0.01,
            "src_ip": "192.0.2.1",
            "dst_ip": "198.51.100.1",
            "src_port": 1000,
            "dst_port": 8333,
            "geo_country": "IN",
            "asn": "AS12345",
            "script_type": "P2WPKH",
        },
        {
            "timestamp": "2026-01-02T00:00:00Z",
            "txid": "tx2",
            "input_wallets": ["wallet_in_2"],
            "output_wallets": ["wallet_out_2"],
            "input_amount": 1.5,
            "output_amount": 1.48,
            "fee": 0.02,
            "src_ip": "192.0.2.2",
            "dst_ip": "198.51.100.2",
            "src_port": 1001,
            "dst_port": 8333,
            "geo_country": "US",
            "asn": "AS64500",
            "script_type": "P2PKH",
        },
    ]

    result = analyze_dataset(
        records,
        records_read=2,
        invalid_records=0,
        duplicates_removed=0,
    )

    assert result["scope"] == "imported_dataset"

    assert result["records"]["analyzed"] == 2
    assert result["records"]["records_read"] == 2
    assert result["records"]["invalid"] == 0
    assert result["records"]["duplicates_removed"] == 0

    assert result["transactions"]["unique_txids"] == 2

    assert result["wallets"]["available"] is True
    assert result["wallets"]["unique_input_wallets"] == 2
    assert result["wallets"]["unique_output_wallets"] == 2
    assert result["wallets"]["unique_wallets"] == 4

    assert result["financial"]["available"] is True

    assert result["financial"]["input"]["total"] == 4.0
    assert result["financial"]["output"]["total"] == 3.97
    assert result["financial"]["fees"]["total"] == 0.03

    assert result["financial"]["conservation_delta"] == 0.0

    assert result["time"]["records_with_valid_timestamp"] == 2
    assert result["time"]["start"] == "2026-01-01T00:00:00+00:00"
    assert result["time"]["end"] == "2026-01-02T00:00:00+00:00"

    assert result["network"]["unique_source_ips"] == 2
    assert result["network"]["unique_destination_ips"] == 2
    assert result["network"]["unique_source_ports"] == 2
    assert result["network"]["unique_destination_ports"] == 1

    assert len(result["distributions"]["countries"]) == 2
    assert len(result["distributions"]["asns"]) == 2
    assert len(result["distributions"]["script_types"]) == 2

    assert result["availability"]["risk_analysis"] is False
    assert result["availability"]["behavioral_analysis"] is False
    assert result["availability"]["graph_analysis"] is False


def test_analyze_dataset_does_not_treat_missing_financial_fields_as_zero():
    records = [
        {
            "timestamp": "2026-09-17T10:00:00Z",
            "txid": "4304541",
            "src_ip": "192.0.2.10",
            "dst_ip": "198.51.100.20",
            "src_port": 8333,
            "dst_port": 8333,
            "geo_country": "IN",
            "asn": "AS12345",
        },
        {
            "timestamp": "2026-09-17T10:00:01Z",
            "txid": "4304542",
            "src_ip": "203.0.113.10",
            "dst_ip": "198.51.100.30",
            "src_port": 8333,
            "dst_port": 8333,
            "geo_country": "US",
            "asn": "AS64500",
        },
    ]

    result = analyze_dataset(
        records,
        records_read=2,
        invalid_records=0,
        duplicates_removed=0,
    )

    assert result["records"]["analyzed"] == 2
    assert result["transactions"]["unique_txids"] == 2

    assert result["financial"]["available"] is False

    assert result["financial"]["input"]["total"] is None
    assert result["financial"]["output"]["total"] is None
    assert result["financial"]["fees"]["total"] is None

    assert result["financial"]["conservation_delta"] is None

    assert result["wallets"]["available"] is False
    assert result["wallets"]["unique_wallets"] is None
    assert result["wallets"]["top_wallets"] == []

    assert result["availability"]["financial_analysis"] is False
    assert result["availability"]["wallet_analysis"] is False

    assert result["network"]["unique_source_ips"] == 2
    assert result["network"]["unique_destination_ips"] == 2
    assert result["network"]["unique_source_ports"] == 1
    assert result["network"]["unique_destination_ports"] == 1

    assert len(result["distributions"]["countries"]) == 2
    assert len(result["distributions"]["asns"]) == 2


def test_analyze_dataset_preserves_import_statistics():
    records = [
        {
            "timestamp": "2026-09-17T10:00:00Z",
            "txid": "tx1",
        }
    ]

    result = analyze_dataset(
        records,
        records_read=3,
        invalid_records=1,
        duplicates_removed=1,
    )

    assert result["records"]["analyzed"] == 1
    assert result["records"]["records_read"] == 3
    assert result["records"]["invalid"] == 1
    assert result["records"]["duplicates_removed"] == 1