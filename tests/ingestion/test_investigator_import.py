from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.ingestion.investigator_import import import_investigator_data


NETWORK_OBSERVATION = "NETWORK_OBSERVATION"
IP_INTELLIGENCE = "IP_INTELLIGENCE"
BLOCKCHAIN_TRANSACTION = "BLOCKCHAIN_TRANSACTION"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_network_csv_import_and_deduplication():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "network.csv"
        output = root / "output"

        write_text(
            source,
            """timestamp,src_ip,src_port,dst_ip,dst_port,txid,geo_country,asn
2026-09-17T10:00:00Z,192.0.2.10,8333,198.51.100.20,8333,4304541,IN,AS12345
2026-09-17T10:00:00Z,192.0.2.10,8333,198.51.100.20,8333,4304541,IN,AS12345
2026-09-17T10:00:01Z,203.0.113.10,8333,198.51.100.30,8333,4304542,US,AS64500
""",
        )

        result = import_investigator_data(
            source,
            NETWORK_OBSERVATION,
            output,
        )

        stats = result["statistics"]

        assert result["status"] == "PASS"
        assert stats["records_read"] == 3
        assert stats["valid_records_before_deduplication"] == 3
        assert stats["valid_records"] == 2
        assert stats["invalid_records"] == 0
        assert stats["duplicates_removed"] == 1
        assert len(result["records"]) == 2


def test_ip_intelligence_json_import():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "ip_intelligence.json"
        output = root / "output"

        payload = [
            {
                "ip": "192.0.2.10",
                "network_type": "VPN",
                "provider": "Example VPN Provider",
                "confidence": 0.98,
                "source": "test",
                "valid_from": "2026-01-01T00:00:00Z",
                "valid_to": "2026-12-31T23:59:59Z",
                "asn": "AS12345",
                "country": "IN",
            },
            {
                "ip": "2001:db8::10",
                "network_type": "PROXY",
                "provider": "Example Proxy Provider",
                "confidence": 0.91,
                "source": "test",
                "valid_from": "2026-01-01T00:00:00Z",
                "valid_to": "2026-12-31T23:59:59Z",
                "asn": "AS12345",
                "country": "IN",
            },
        ]

        write_text(source, json.dumps(payload, indent=2))

        result = import_investigator_data(
            source,
            IP_INTELLIGENCE,
            output,
        )

        stats = result["statistics"]

        assert result["status"] == "PASS"
        assert stats["records_read"] == 2
        assert stats["valid_records_before_deduplication"] == 2
        assert stats["valid_records"] == 2
        assert stats["invalid_records"] == 0
        assert stats["duplicates_removed"] == 0
        assert len(result["records"]) == 2


def test_blockchain_xml_import():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "blockchain.xml"
        output = root / "output"

        write_text(
            source,
            """<?xml version="1.0" encoding="UTF-8"?>
<records>
    <record>
        <txid>4304541</txid>
        <timestamp>2026-09-17T10:00:00Z</timestamp>
        <input_wallets>bc1input001|bc1input002</input_wallets>
        <output_wallets>bc1output001|bc1output002</output_wallets>
        <input_amount>1.25</input_amount>
        <output_amount>1.24</output_amount>
        <fee>0.01</fee>
    </record>
    <record>
        <txid>4304542</txid>
        <timestamp>2026-09-17T10:00:05Z</timestamp>
        <input_wallets>bc1input003</input_wallets>
        <output_wallets>bc1output003</output_wallets>
        <input_amount>2.50</input_amount>
        <output_amount>2.48</output_amount>
        <fee>0.02</fee>
    </record>
</records>
""",
        )

        result = import_investigator_data(
            source,
            BLOCKCHAIN_TRANSACTION,
            output,
        )

        stats = result["statistics"]

        assert result["status"] == "PASS"
        assert stats["records_read"] == 2
        assert stats["valid_records_before_deduplication"] == 2
        assert stats["valid_records"] == 2
        assert stats["invalid_records"] == 0
        assert stats["duplicates_removed"] == 0
        assert len(result["records"]) == 2


def test_malformed_network_data_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "malformed.csv"
        output = root / "output"

        write_text(
            source,
            """timestamp,src_ip,src_port,dst_ip,dst_port,txid,geo_country,asn
2026-09-17T10:00:00Z,999.999.999.999,8333,198.51.100.20,8333,4304541,IN,AS12345
2026-09-17T10:00:01Z,192.0.2.10,99999,198.51.100.20,8333,,IN,AS12345
2026-09-17T10:00:02Z,192.0.2.10,8333,198.51.100.20,8333,4304543,IN,AS12345
""",
        )

        result = import_investigator_data(
            source,
            NETWORK_OBSERVATION,
            output,
        )

        stats = result["statistics"]

        assert result["status"] == "WARNING"
        assert stats["records_read"] == 3
        assert stats["valid_records_before_deduplication"] == 1
        assert stats["valid_records"] == 1
        assert stats["invalid_records"] == 2
        assert stats["duplicates_removed"] == 0
        assert len(result["records"]) == 1
        assert len(result["invalid_records"]) == 2


def test_manifest_and_artifacts_are_created():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "network.csv"
        output = root / "output"

        write_text(
            source,
            """timestamp,src_ip,src_port,dst_ip,dst_port,txid,geo_country,asn
2026-09-17T10:00:00Z,192.0.2.10,8333,198.51.100.20,8333,4304541,IN,AS12345
""",
        )

        result = import_investigator_data(
            source,
            NETWORK_OBSERVATION,
            output,
        )

        import_id = result["import_id"]

        records_path = (
            output
            / f"{import_id}_normalized_records.json"
        )
        errors_path = (
            output
            / f"{import_id}_validation_errors.json"
        )
        manifest_path = (
            output
            / f"{import_id}_manifest.json"
        )

        assert records_path.exists()
        assert errors_path.exists()
        assert manifest_path.exists()

        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        assert manifest["import_id"] == import_id
        assert manifest["data_type"] == NETWORK_OBSERVATION
        assert manifest["records_read"] == 1
        assert manifest["valid_records"] == 1
        assert manifest["invalid_records"] == 0
        assert manifest["duplicate_records_removed"] == 0
        assert manifest["unique_valid_records"] == 1
        assert manifest["offline_processing"] is True
        assert manifest["provenance_available"] is True


def test_import_id_is_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "network.csv"
        output1 = root / "output1"
        output2 = root / "output2"

        write_text(
            source,
            """timestamp,src_ip,src_port,dst_ip,dst_port,txid,geo_country,asn
2026-09-17T10:00:00Z,192.0.2.10,8333,198.51.100.20,8333,4304541,IN,AS12345
""",
        )

        result1 = import_investigator_data(
            source,
            NETWORK_OBSERVATION,
            output1,
        )

        result2 = import_investigator_data(
            source,
            NETWORK_OBSERVATION,
            output2,
        )

        assert result1["status"] == "PASS"
        assert result2["status"] == "PASS"
        assert result1["import_id"] == result2["import_id"]


if __name__ == "__main__":
    print(
        "Run with: "
        "python -m pytest "
        "tests\\ingestion\\test_investigator_import.py -v"
    )