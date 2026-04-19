# Usage:
# python upload_incident_json.py data/logs/incident_chunks.json

import json
import argparse
from pathlib import Path
import os

import weaviate
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


DEFAULT_COLLECTION_NAME = "IncidentLogs"
DEFAULT_JSON_PATH = "data/logs/incident_chunks.json"
DEFAULT_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


load_dotenv()
hf_token = os.getenv("HF_TOKEN")


def load_records(json_path: Path) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_and_upload(client, collection_name: str, records: list[dict], model_name: str):
    collection = client.collections.get(collection_name)

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, token=hf_token if hf_token else None)

    texts = [r["text"] for r in records]
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    ).tolist()

    with collection.batch.dynamic() as batch:
        for record, vector in zip(records, vectors):
            batch.add_object(
                properties=record,
                vector={"incident_vector": vector}
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
        "json_path",
        nargs="?",
        default=DEFAULT_JSON_PATH,
        help="Path to incident JSON file"
    )

    parser.add_argument(
        "--collection_name",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help="Weaviate collection name"
    )

    parser.add_argument(
        "--embed_model",
        type=str,
        default=DEFAULT_EMBED_MODEL_NAME,
        help="SentenceTransformer embedding model name"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Weaviate host"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Weaviate HTTP port"
    )

    parser.add_argument(
        "--grpc_port",
        type=int,
        default=50051,
        help="Weaviate gRPC port"
    )

    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    records = load_records(json_path)

    client = weaviate.connect_to_local(
        host=args.host,
        port=args.port,
        grpc_port=args.grpc_port
    )

    try:
        embed_and_upload(
            client=client,
            collection_name=args.collection_name,
            records=records,
            model_name=args.embed_model
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()