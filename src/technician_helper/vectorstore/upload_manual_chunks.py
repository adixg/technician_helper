"""Embed manual chunk JSON and upload it to the Weaviate manual collection."""

import argparse
import json
from pathlib import Path

from technician_helper.clients import weaviate_client
from technician_helper.config import settings
from technician_helper.embeddings import get_embedding_model, resolve_device


def load_chunks(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def upload_manual_chunks(
    chunks_json_path: Path,
    collection_name: str | None = None,
    embed_model: str | None = None,
    batch_size: int = 2,
    progress_callback=None,
):
    collection_name = collection_name or settings.manual_collection
    embed_model = embed_model or settings.embed_model

    def update(message: str, pct: int):
        if progress_callback is not None:
            progress_callback("upload", message, pct)

    if not chunks_json_path.exists():
        raise FileNotFoundError(f"Chunks JSON not found: {chunks_json_path}")

    doc = load_chunks(chunks_json_path)
    chunks = doc["chunks"]

    device = resolve_device()
    update(f"Loading embedding model {embed_model} on {device}...", 72)

    model = get_embedding_model(embed_model, device)

    texts = [c["chunk_text"] for c in chunks]
    total = len(texts)

    update("Generating embeddings...", 75)

    vectors = []
    for i in range(0, total, batch_size):
        batch_texts = texts[i : i + batch_size]

        batch_vecs = model.encode(
            batch_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )

        vectors.extend(batch_vecs)

        done = min(i + batch_size, total)
        pct = 75 + int((done / max(total, 1)) * 15)
        update(f"Embedded {done}/{total} chunks...", pct)

    update("Connecting to Weaviate...", 91)

    from weaviate.util import generate_uuid5

    collection = weaviate_client().collections.get(collection_name)

    update("Uploading chunks to Weaviate...", 92)

    # Deterministic UUIDs keyed on chunk_id make re-runs idempotent (upsert,
    # not duplicate).
    with collection.batch.dynamic() as batch:
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors, strict=False), start=1):
            batch.add_object(
                properties={
                    "chunk_id": chunk["chunk_id"],
                    "source_pdf_file": doc.get("source_pdf_file"),
                    "source_md_file": doc.get("source_md_file"),
                    "machine": doc.get("machine"),
                    "manufacturer": doc.get("manufacturer"),
                    "manual_type": doc.get("manual_type"),
                    "section_id": chunk["section_id"],
                    "section_title": chunk["section_title"],
                    "chunk_index_within_section": chunk["chunk_index_within_section"],
                    "chunk_text": chunk["chunk_text"],
                    "images": chunk["images"],
                },
                vector=vec.tolist(),
                uuid=generate_uuid5(chunk["chunk_id"]),
            )

            if idx % 5 == 0 or idx == len(chunks):
                pct = 92 + int((idx / max(len(chunks), 1)) * 8)
                update(f"Uploaded {idx}/{len(chunks)} chunks...", pct)

    update(f"Upload complete. Uploaded {len(chunks)} chunks.", 100)
    print(f"Uploaded {len(chunks)} chunks to '{collection_name}'")


def main():
    parser = argparse.ArgumentParser(
        description="Embed chunk JSON and upload to Weaviate collection."
    )

    parser.add_argument("chunks_json_path", type=str, help="Path to chunks JSON file")

    parser.add_argument(
        "--collection_name",
        type=str,
        default=settings.manual_collection,
        help="Weaviate collection name",
    )

    parser.add_argument(
        "--embed_model", type=str, default=settings.embed_model, help="Embedding model name"
    )

    parser.add_argument("--batch_size", type=int, default=2, help="Embedding batch size")

    args = parser.parse_args()

    upload_manual_chunks(
        chunks_json_path=Path(args.chunks_json_path),
        collection_name=args.collection_name,
        embed_model=args.embed_model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
