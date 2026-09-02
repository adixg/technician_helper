"""Semantic search over the incident log collection.

python -m technician_helper.retrieval.incidents --query "bearing vibration on pump" --top_k 3
"""

import argparse
import json
from collections.abc import Callable

from technician_helper.clients import weaviate_client
from technician_helper.config import settings
from technician_helper.embeddings import get_embedding_model


def _update_stage(stage_callback: Callable[[str], None] | None, message: str) -> None:
    if stage_callback:
        stage_callback(message)


def semantic_query(
    query_text: str,
    top_k: int = 5,
    stage_callback: Callable[[str], None] | None = None,
) -> list[dict]:
    """
    Semantic search over IncidentLogs collection.
    Returns top-k incident records as a list of property dicts.
    """
    _update_stage(stage_callback, "Connecting to incident database")

    collection = weaviate_client().collections.get(settings.incident_collection)

    _update_stage(stage_callback, "Loading incident embedding model")

    model = get_embedding_model()

    _update_stage(stage_callback, "Encoding incident query")

    query_vector = model.encode(query_text).tolist()

    _update_stage(stage_callback, "Searching incident vectors")

    response = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k,
        target_vector="incident_vector",
    )

    _update_stage(stage_callback, "Processing incident retrieval results")

    results = [obj.properties for obj in response.objects]

    _update_stage(stage_callback, "Incident retrieval complete")
    return results


def print_results(results: list[dict]) -> None:
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
