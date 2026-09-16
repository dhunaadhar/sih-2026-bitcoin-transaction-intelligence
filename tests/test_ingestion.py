from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.deduplicator import deduplicate_records
from src.ingestion.ingest import ingest
from src.ingestion.schema_validator import validate_record


def test_missing_required_field_is_rejected():
    record = {
        "timestamp": "2026-01-01T00:00:00Z",
        "src_ip": "10.0.0.1",
        "src_port": 8333,
        "dst_ip": "10.0.0.2",
        "dst_port": 8333,
        "txid": 12345,
        "input_wallets": ["wallet_in"],
        "output_wallets": ["wallet_out"],
        "input_amount": 1.0,
        "output_amount": 0.99,
        "fee": 0.01,
        "script_type": "P2PKH",
        "geo_country": "IN",
    }

    result = validate_record(record)

    assert result.valid is False
    assert "asn" in result.missing_fields


def test_invalid_ip_and_port_are_rejected():
    record = {
        "timestamp": "2026-01-01T00:00:00Z",
        "src_ip": "not-an-ip",
        "src_port": 70000,
        "dst_ip": "10.0.0.2",
        "dst_port": 8333,
        "txid": 12345,
        "input_wallets": ["wallet_in"],
        "output_wallets": ["wallet_out"],
        "input_amount": 1.0,
        "output_amount": 0.99,
        "fee": 0.01,
        "script_type": "P2PKH",
        "geo_country": "IN",
        "asn": "AS12345",
    }

    result = validate_record(record)

    assert result.valid is False
    assert "src_ip" in result.invalid_fields
    assert "src_port" in result.invalid_fields


def test_exact_duplicate_is_removed():
    record = {
        "timestamp": "2026-01-01T00:00:00Z",
        "src_ip": "10.0.0.1",
        "src_port": 8333,
        "dst_ip": "10.0.0.2",
        "dst_port": 8333,
        "txid": 12345,
        "input_wallets": ["wallet_in"],
        "output_wallets": ["wallet_out"],
        "input_amount": 1.0,
        "output_amount": 0.99,
        "fee": 0.01,
        "script_type": "P2PKH",
        "geo_country": "IN",
        "asn": "AS12345",
    }

    unique, duplicates = deduplicate_records(
        [record, record.copy()]
    )

    assert len(unique) == 1
    assert duplicates == 1