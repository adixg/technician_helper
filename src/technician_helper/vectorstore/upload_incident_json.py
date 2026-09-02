"""Embed incident JSON records and upload them to the Weaviate incident collection.

python -m technician_helper.vectorstore.upload_incident_json data/logs/incident_chunks.json
"""

import argparse
import json
from pathlib import Path

from technician_helper.clients import weaviate_client
from technician_helper.config import settings
from technician_helper.embeddings import get_embedding_model

DEFAULT_JSON_PATH = "data/logs/incident_chunks.json"


def load_records(json_path: Path) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def embed_and_upload(client, collection_name: str, records: list[dict], model_name: str):
    from weaviate.util import generate_uuid5

    collection = client.collections.get(collection_name)

    model = get_embedding_model(model_name)

    texts = [r["text"] for r in records]
    vectors = model.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
    ).tolist()

    # Deterministic UUIDs keyed on chunk_id -> re-running upserts instead of
    # inserting duplicates.
    with collection.batch.dynamic() as batch:
        for record, vector in zip(records, vectors, strict=False):
            batch.add_object(
                properties=record,
                vector={"incident_vector": vector},
                uuid=generate_uuid5(record["chunk_id"]),
            )

    failed = collection.batch.failed_objects
    if failed:
        print(f"Upload finished with {len(failed)} failed objects")
        for obj in failed[:5]:
            print(obj)
    else:
        print(f"Successfully uploaded {len(records)} objects to {collection_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Embed incident records and upload them to a Weaviate collection."
    )

    parser.add_argument(
        "json_path", nargs="?", default=DEFAULT_JSON_PATH, help="Path to incident JSON file"
    )

    parser.add_argument(
        "--collection_name",
        type=str,
        default=settings.incident_collection,
        help="Weaviate collection name",
    )

    parser.add_argument(
        "--embed_model",
        type=str,
        default=settings.embed_model,
        help="SentenceTransformer embedding model name",
    )

    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    records = load_records(json_path)

    embed_and_upload(
        client=weaviate_client(),
        collection_name=args.collection_name,
        records=records,
        model_name=args.embed_model,
    )


if __name__ == "__main__":
    main()
