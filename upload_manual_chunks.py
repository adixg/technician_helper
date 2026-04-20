import json
import argparse
from pathlib import Path
import os

from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer
import weaviate


DEFAULT_COLLECTION_NAME = "ManualChunk"
# DEFAULT_EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


load_dotenv()
hf_token = os.getenv("HF_TOKEN")


def load_chunks(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def upload_manual_chunks(
    chunks_json_path: Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embed_model: str = DEFAULT_EMBED_MODEL_NAME,
    batch_size: int = 2,
    progress_callback=None,
):
    def update(message: str, pct: int):
        if progress_callback is not None:
            progress_callback("upload", message, pct)

    if not chunks_json_path.exists():
        raise FileNotFoundError(f"Chunks JSON not found: {chunks_json_path}")

    doc = load_chunks(chunks_json_path)
    chunks = doc["chunks"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading embedding model: {embed_model}")

    update(f"Loading embedding model on {device}...", 72)

    model = SentenceTransformer(
        embed_model,
        trust_remote_code=True,
        device=device,
        token=hf_token
    )

    texts = [c["chunk_text"] for c in chunks]
    total = len(texts)

    update("Generating embeddings...", 75)

    vectors = []
    for i in range(0, total, batch_size):
        batch_texts = texts[i:i + batch_size]

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

    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051
    )

    try:
        collection = client.collections.get(collection_name)

        update("Uploading chunks to Weaviate...", 92)

        with collection.batch.dynamic() as batch:
            for idx, (chunk, vec) in enumerate(zip(chunks, vectors), start=1):
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
                )

                if idx % 5 == 0 or idx == len(chunks):
                    pct = 92 + int((idx / max(len(chunks), 1)) * 8)
                    update(f"Uploaded {idx}/{len(chunks)} chunks...", pct)

        update(f"Upload complete. Uploaded {len(chunks)} chunks.", 100)

        print(f"Uploaded {len(chunks)} chunks to '{collection_name}'")

    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Embed chunk JSON and upload to Weaviate collection."
    )

    parser.add_argument(
        "chunks_json_path",
        type=str,
        help="Path to chunks JSON file"
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
        help="Embedding model name"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Embedding batch size"
    )

    args = parser.parse_args()

    upload_manual_chunks(
        chunks_json_path=Path(args.chunks_json_path),
        collection_name=args.collection_name,
        embed_model=args.embed_model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()