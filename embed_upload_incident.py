import json
import weaviate
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

COLLECTION_NAME = "IncidentLogs"
JSON_PATH = "data/logs/incident_chunks.json"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

def load_records(json_path: str) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_and_upload(client, collection_name: str, records: list[dict], model_name: str):
    collection = client.collections.get(collection_name)
    model = SentenceTransformer(model_name)

    texts = [r["text"] for r in records]
    vectors = model.encode(texts, normalize_embeddings=True).tolist()

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
    records = load_records(JSON_PATH)

    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051
    )

    try:
        embed_and_upload(client, COLLECTION_NAME, records, EMBED_MODEL_NAME)
    finally:
        client.close()


if __name__ == "__main__":
    main()