# Usage:
# python create_incident_json.py data/logs/predictive-maintenance-incident-log.csv

import json
import argparse
from pathlib import Path
import pandas as pd


def clean_value(v):
    if pd.isna(v):
        return None

    if isinstance(v, str):
        v = v.strip()
        if v == "":
            return None

    return v


def to_rfc3339_utc(value):
    if value is None:
        return None

    dt = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(dt):
        return None

    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def to_float_or_none(value):
    if value is None:
        return None
    try:
        val = float(value)
        if pd.isna(val):
            return None
        return val
    except Exception:
        return None


def to_int_or_none(value):
    if value is None:
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def build_incident_text(row: dict) -> str:

    sentences = []

    intro_parts = []

    if row.get("incident_id"):
        intro_parts.append(f"Incident {row['incident_id']}")

    if row.get("incident_datetime"):
        intro_parts.append(f"occurred on {row['incident_datetime']}")

    if row.get("location"):
        intro_parts.append(f"at {row['location']}")

    machine_phrase = []

    if row.get("machine_id"):
        machine_phrase.append(str(row["machine_id"]))

    if row.get("machine_type"):
        machine_phrase.append(f"a {row['machine_type']}")

    if machine_phrase:
        intro_parts.append(f"on machine {', '.join(machine_phrase)}")

    if intro_parts:
        sentences.append(" ".join(intro_parts) + ".")

    if row.get("incident_type"):
        sentences.append(f"Incident type: {row['incident_type']}.")

    if row.get("failure_code"):
        sentences.append(f"Failure code: {row['failure_code']}.")

    if row.get("failure_description"):
        sentences.append(f"Failure description: {row['failure_description']}.")

    if row.get("sensor_id") or row.get("sensor_type") or row.get("sensor_value") is not None:

        sensor_parts = ["Sensor"]

        if row.get("sensor_id"):
            sensor_parts.append(str(row["sensor_id"]))

        if row.get("sensor_type"):
            sensor_parts.append(f"of type {row['sensor_type']}")

        if row.get("sensor_value") is not None:
            sensor_parts.append(f"recorded value {row['sensor_value']}")

        sentences.append(" ".join(sensor_parts) + ".")

    if row.get("maintenance_type"):
        sentences.append(f"Maintenance type: {row['maintenance_type']}.")

    if row.get("maintenance_action"):
        sentences.append(f"Maintenance action: {row['maintenance_action']}.")

    if row.get("downtime_minutes") is not None:
        sentences.append(f"Downtime was {row['downtime_minutes']} minutes.")

    if row.get("reported_by"):
        sentences.append(f"Reported by {row['reported_by']}.")

    resolution_parts = ["The incident"]
    resolution_has_content = False

    if row.get("resolved_datetime"):
        resolution_parts.append(f"was resolved on {row['resolved_datetime']}")
        resolution_has_content = True

    if row.get("resolution_status"):
        resolution_parts.append(f"with status {row['resolution_status']}")
        resolution_has_content = True

    if resolution_has_content:
        sentences.append(" ".join(resolution_parts) + ".")

    if row.get("cost_estimate") is not None:
        sentences.append(f"Estimated cost was {row['cost_estimate']}.")

    if row.get("root_cause"):
        sentences.append(f"Root cause: {row['root_cause']}.")

    return " ".join(sentences)


def csv_to_incident_json(csv_path: Path, json_path: Path, delimiter: str):

    df = pd.read_csv(csv_path, sep=delimiter)

    df = df.map(clean_value)

    records = []

    for _, row in df.iterrows():

        row_dict = row.to_dict()

        record = {

            "chunk_id": f"incident_{row_dict['incident_id']}",
            "source": "incident_log",
            "record_type": "maintenance_incident",

            "incident_id": row_dict.get("incident_id"),
            "machine_id": row_dict.get("machine_id"),
            "machine_type": row_dict.get("machine_type"),
            "location": row_dict.get("location"),

            "incident_datetime": to_rfc3339_utc(
                row_dict.get("incident_datetime")
            ),

            "resolved_datetime": to_rfc3339_utc(
                row_dict.get("resolved_datetime")
            ),

            "incident_type": row_dict.get("incident_type"),
            "failure_code": row_dict.get("failure_code"),
            "failure_description": row_dict.get("failure_description"),

            "sensor_id": row_dict.get("sensor_id"),
            "sensor_type": row_dict.get("sensor_type"),
            "sensor_value": to_float_or_none(
                row_dict.get("sensor_value")
            ),

            "maintenance_type": row_dict.get("maintenance_type"),
            "maintenance_action": row_dict.get("maintenance_action"),

            "downtime_minutes": to_int_or_none(
                row_dict.get("downtime_minutes")
            ),

            "reported_by": row_dict.get("reported_by"),
            "resolution_status": row_dict.get("resolution_status"),
            "cost_estimate": to_float_or_none(
                row_dict.get("cost_estimate")
            ),

            "root_cause": row_dict.get("root_cause"),
        }

        record["text"] = build_incident_text(record)

        records.append(record)

    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} records to {json_path}")


def main():

    parser = argparse.ArgumentParser(
        description="Convert incident CSV into chunked JSON records."
    )

    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to incident CSV file"
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default="data/logs/incident_chunks.json",
        help="Output JSON path"
    )

    parser.add_argument(
        "--delimiter",
        type=str,
        default=",",
        help="CSV delimiter (default: comma)"
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    json_path = Path(args.output_json)

    csv_to_incident_json(
        csv_path=csv_path,
        json_path=json_path,
        delimiter=args.delimiter
    )


if __name__ == "__main__":
    main()