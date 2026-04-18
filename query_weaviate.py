import os
from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer
import weaviate

COLLECTION_NAME = "ManualChunk"
EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
load_dotenv()

hf_token = os.getenv("HF_TOKEN")


def main():
    question = "How should the motor be grounded?"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SentenceTransformer(
        EMBED_MODEL_NAME,
        trust_remote_code=True,
        device=device,
    )

    qvec = model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    client = weaviate.connect_to_local()

    try:
        collection = client.collections.get(COLLECTION_NAME)

        response = collection.query.near_vector(
            near_vector=qvec,
            limit=5,
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

        for i, obj in enumerate(response.objects, start=1):
            props = obj.properties
            print("=" * 100)
            print(f"Rank: {i}")
            print("chunk_id:", props["chunk_id"])
            print("section_title:", props["section_title"])
            print("source_pdf_file:", props["source_pdf_file"])
            print("manufacturer:", props["manufacturer"])
            print("machine:", props["machine"])
            print("images:", props["images"])
            print("text:")
            print(props["chunk_text"])

    finally:
        client.close()

if __name__ == "__main__":
    main()
