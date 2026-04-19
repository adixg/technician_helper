import os
from pathlib import Path
from typing import Optional

import pandas as pd
import weaviate
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


DEFAULT_INCIDENT_COLLECTION = "IncidentLogs"
DEFAULT_INCIDENT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

load_dotenv()
hf_token = os.getenv("HF_TOKEN")


def clean_value(v):
    if v is None:
        return None

    if isinstance(v, str):
        v = v.strip()
        if v == "":
            return None

    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    return v


def to_rfc3339_utc(value):
    if value is None:
        return None

    dt = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(dt):
        return None

    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def to_float_or_none(value):
    value = clean_value(value)
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
    value = clean_value(value)
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


def build_incident_record_from_form(form_data: dict) -> dict:
    record = {
        "chunk_id": f"incident_{clean_value(form_data.get('incident_id'))}",
        "source": "incident_log",
        "record_type": "maintenance_incident",

        "incident_id": clean_value(form_data.get("incident_id")),
        "machine_id": clean_value(form_data.get("machine_id")),
        "machine_type": clean_value(form_data.get("machine_type")),
        "location": clean_value(form_data.get("location")),

        "incident_datetime": to_rfc3339_utc(form_data.get("incident_datetime")),
        "resolved_datetime": to_rfc3339_utc(form_data.get("resolved_datetime")),

        "incident_type": clean_value(form_data.get("incident_type")),
        "failure_code": clean_value(form_data.get("failure_code")),
        "failure_description": clean_value(form_data.get("failure_description")),

        "sensor_id": clean_value(form_data.get("sensor_id")),
        "sensor_type": clean_value(form_data.get("sensor_type")),
        "sensor_value": to_float_or_none(form_data.get("sensor_value")),

        "maintenance_type": clean_value(form_data.get("maintenance_type")),
        "maintenance_action": clean_value(form_data.get("maintenance_action")),

        "downtime_minutes": to_int_or_none(form_data.get("downtime_minutes")),
        "reported_by": clean_value(form_data.get("reported_by")),
        "resolution_status": clean_value(form_data.get("resolution_status")),
        "cost_estimate": to_float_or_none(form_data.get("cost_estimate")),
        "root_cause": clean_value(form_data.get("root_cause")),
    }

    record["text"] = build_incident_text(record)
    return record


def upload_single_incident_to_weaviate(
    record: dict,
    collection_name: str = DEFAULT_INCIDENT_COLLECTION,
    embed_model_name: str = DEFAULT_INCIDENT_EMBED_MODEL,
):
    model = SentenceTransformer(
        embed_model_name,
        token=hf_token if hf_token else None
    )

    vector = model.encode(
        record["text"],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).tolist()

    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051
    )

    try:
        collection = client.collections.get(collection_name)
        collection.data.insert(
            properties=record,
            vector={"incident_vector": vector}
        )
    finally:
        client.close()


def append_incident_to_csv(record: dict, csv_path: str):
    csv_columns = [
        "incident_id",
        "machine_id",
        "machine_type",
        "location",
        "incident_datetime",
        "incident_type",
        "failure_code",
        "failure_description",
        "sensor_id",
        "sensor_type",
        "sensor_value",
        "maintenance_type",
        "maintenance_action",
        "downtime_minutes",
        "reported_by",
        "resolved_datetime",
        "resolution_status",
        "cost_estimate",
        "root_cause",
    ]

    row = {col: record.get(col) for col in csv_columns}
    df_new = pd.DataFrame([row])

    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    if csv_file.exists():
        df_old = pd.read_csv(csv_file)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
        df_out.to_csv(csv_file, index=False)
    else:
        df_new.to_csv(csv_file, index=False)