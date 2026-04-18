import weaviate
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "IncidentLogs"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def semantic_query(query_text, top_k=5):
    """
    Semantic search over IncidentLogs collection.
    Returns top-k incident records.
    """

    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051
    )

    try:
        collection = client.collections.get(COLLECTION_NAME)

        model = SentenceTransformer(EMBED_MODEL_NAME)
        query_vector = model.encode(query_text).tolist()

        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            target_vector="incident_vector"
        )

        results = []

        for obj in response.objects:
            results.append(obj.properties)

        return results

    finally:
        client.close()


def main():

    query = "spindle overheating on CNC lathe"

    results = semantic_query(query, top_k=5)

    print("\nTop incident matches:\n")

    for i, r in enumerate(results, 1):
        print(f"Result {i}")
        print("-" * 50)

        for key, value in r.items():
            print(f"{key}: {value}")

        print()


if __name__ == "__main__":
    main()