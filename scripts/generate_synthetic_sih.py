import csv
import json
import random
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEED = 20260925
COUNT = 100

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "raw" / "synthetic_sih"

random.seed(SEED)

countries = ["US", "DE", "NL", "GB", "FR", "IN", "CA", "SG"]
asns = ["AS14061", "AS16276", "AS24940", "AS9009", "AS16509"]
script_types = ["P2PKH", "P2SH", "P2WPKH", "P2WSH"]

start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

records = []

for i in range(COUNT):
    timestamp = start_time + timedelta(
        minutes=i * 37,
        seconds=random.randint(0, 59)
    )

    src_ip = f"192.0.2.{(i % 254) + 1}"
    dst_ip = f"198.51.100.{((i * 7) % 254) + 1}"

    input_amount = round(random.uniform(0.01, 5.0), 8)
    fee = round(random.uniform(0.00001, 0.001), 8)
    output_amount = round(input_amount - fee, 8)

    txid = f"{i + 1:064x}"

    record = {
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "src_ip": src_ip,
        "src_port": random.randint(1024, 65535),
        "dst_ip": dst_ip,
        "dst_port": 8333,
        "txid": txid,
        "input_wallets": [f"wallet_in_{i + 1:04d}"],
        "output_wallets": [f"wallet_out_{i + 1:04d}"],
        "input_amount": input_amount,
        "output_amount": output_amount,
        "fee": fee,
        "script_type": random.choice(script_types),
        "geo_country": random.choice(countries),
        "asn": random.choice(asns),
        "is_synthetic": True
    }

    records.append(record)


def write_csv():
    path = OUTPUT_DIR / "synthetic_sih.csv"

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()

        for record in records:
            row = record.copy()
            row["input_wallets"] = json.dumps(row["input_wallets"])
            row["output_wallets"] = json.dumps(row["output_wallets"])
            writer.writerow(row)


def write_json():
    path = OUTPUT_DIR / "synthetic_sih.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def write_xml():
    path = OUTPUT_DIR / "synthetic_sih.xml"

    root = ET.Element("transactions")

    for record in records:
        transaction = ET.SubElement(root, "transaction")

        for key, value in record.items():
            element = ET.SubElement(transaction, key)

            if isinstance(value, list):
                for item in value:
                    wallet = ET.SubElement(element, "wallet")
                    wallet.text = str(item)
            else:
                element.text = str(value)

    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def write_metadata():
    path = OUTPUT_DIR / "metadata.json"

    metadata = {
        "dataset": "synthetic_sih",
        "description": "Reproducible synthetic Bitcoin transaction intelligence dataset",
        "synthetic": True,
        "seed": SEED,
        "record_count": COUNT,
        "formats": ["CSV", "JSON", "XML"],
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "SIH 2026 synthetic dataset generator"
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

write_csv()
write_json()
write_xml()
write_metadata()

print(f"Generated {COUNT} synthetic records.")
print(f"Output: {OUTPUT_DIR}")