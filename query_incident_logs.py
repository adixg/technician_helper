# Usage:
# python query_incident_logs.py --query "bearing vibration on pump" --top_k 3

import argparse
import json
import os
from typing import Callable, Dict, List, Optional

import weaviate
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "IncidentLogs"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

load_dotenv()
hf_token = os.getenv("HF_TOKEN")


def _update_stage(stage_callback: Optional[Callable[[str], None]], message: str) -> None:
    if stage_callback:
        stage_callback(message)


def semantic_query(
    query_text: str,
    top_k: int = 5,
    stage_callback: Optional[Callable[[str], None]] = None,
) -> List[Dict]:
    """
    Semantic search over IncidentLogs collection.
    Returns top-k incident records as a list of property dicts.
    """
    _update_stage(stage_callback, "Connecting to incident database")

    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051,
    )

    try:
        _update_stage(stage_callback, "Loading incident embedding model")

        collection = client.collections.get(COLLECTION_NAME)

        model = SentenceTransformer(
            EMBED_MODEL_NAME,
            token=hf_token,
        )

        _update_stage(stage_callback, "Encoding incident query")

        query_vector = model.encode(query_text).tolist()

        _update_stage(stage_callback, "Searching incident vectors")

        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            target_vector="incident_vector",
        )

        _update_stage(stage_callback, "Processing incident retrieval results")

        results = []
        for obj in response.objects:
            results.append(obj.properties)

        _update_stage(stage_callback, "Incident retrieval complete")
        return results

    finally:
        client.close()


def print_results(results: List[Dict]) -> None:
    print("\nTop incident matches:\n")

    if not results:
        print("No matching incident records found.")
        return

    for i, r in enumerate(results, 1):
        print(f"Result {i}")
        print("-" * 50)
        for key, value in r.items():
            print(f"{key}: {value}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the IncidentLogs collection.")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query text for semantic search.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top results to return.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results as JSON instead of formatted text.",
    )
    args = parser.parse_args()

    results = semantic_query(args.query, top_k=args.top_k)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print_results(results)


if __name__ == "__main__":
    main()