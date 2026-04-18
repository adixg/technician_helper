import json
from pathlib import Path
import os
from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer
import weaviate

CHUNKS_JSON_PATH = Path(r"./data/manuals_chunks/Century NEMA 42-140 Frame Motor Instruction Leaflet-chunks.json")
COLLECTION_NAME = "ManualChunk"

EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
# print(hf_token)


def load_chunks(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    doc = load_chunks(CHUNKS_JSON_PATH)
    chunks = doc["chunks"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SentenceTransformer(
        EMBED_MODEL_NAME,
        trust_remote_code=True,
        device=device,
    )

    texts = [c["chunk_text"] for c in chunks]
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    client = weaviate.connect_to_local()

    try:
        collection = client.collections.get(COLLECTION_NAME)

        with collection.batch.dynamic() as batch:
            for chunk, vec in zip(chunks, vectors):
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

        print(f"Uploaded {len(chunks)} chunks to '{COLLECTION_NAME}'")

    finally:
        client.close()

if __name__ == "__main__":
    main()