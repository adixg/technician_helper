# Usage
# python query_manuals.py --query "motor grounding procedure" --top_k 3


import argparse
import json
import os
from typing import Callable, Dict, List, Optional

import torch
import weaviate
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "ManualChunk"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

load_dotenv()
hf_token = os.getenv("HF_TOKEN")


def _update_stage(stage_callback: Optional[Callable[[str], None]], message: str) -> None:
    if stage_callback:
        stage_callback(message)


def semantic_query(
    question: str,
    top_k: int = 5,
    stage_callback: Optional[Callable[[str], None]] = None,
) -> List[Dict]:
    """
    Semantic search over ManualChunk collection.
    Returns top-k manual chunk records as a list of property dicts.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _update_stage(stage_callback, f"Loading manual embedding model on {device}")

    model = SentenceTransformer(
        EMBED_MODEL_NAME,
        trust_remote_code=True,
        device=device,
        token=hf_token,
    )

    _update_stage(stage_callback, "Encoding manual query")

    qvec = model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    _update_stage(stage_callback, "Connecting to manual database")

    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051,
    )

    try:
        collection = client.collections.get(COLLECTION_NAME)

        _update_stage(stage_callback, "Searching manual vectors")

        response = collection.query.near_vector(
            near_vector=qvec,
            limit=top_k,
            return_properties=[
                "chunk_id",
                "section_title",
                "chunk_text",
                "images",
                "source_pdf_file",
                "manufacturer",
                "machine",
            ],
        )

        _update_stage(stage_callback, "Processing manual retrieval results")

        results = []
        for obj in response.objects:
            results.append(obj.properties)

        _update_stage(stage_callback, "Manual retrieval complete")
        return results

    finally:
        client.close()


def print_results(results: List[Dict]) -> None:
    print("\nTop manual matches:\n")

    if not results:
        print("No matching manual chunks found.")
        return

    for i, props in enumerate(results, start=1):
        print("=" * 100)
        print(f"Rank: {i}")
        print("chunk_id:", props.get("chunk_id"))
        print("section_title:", props.get("section_title"))
        print("source_pdf_file:", props.get("source_pdf_file"))
        print("manufacturer:", props.get("manufacturer"))
        print("machine:", props.get("machine"))
        print("images:", props.get("images"))
        print("text:")
        print(props.get("chunk_text"))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the ManualChunk collection.")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Question/query text for manual retrieval.",
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