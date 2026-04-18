import weaviate
from weaviate.classes.config import Configure, Property, DataType

COLLECTION_NAME = "ManualChunk"

def main():
    client = weaviate.connect_to_local()

    try:
        if client.collections.exists(COLLECTION_NAME):
            print(f"Collection '{COLLECTION_NAME}' already exists")
            return

        client.collections.create(
            name=COLLECTION_NAME,
            vector_config=Configure.Vectors.self_provided(),
            properties=[
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="source_pdf_file", data_type=DataType.TEXT),
                Property(name="source_md_file", data_type=DataType.TEXT),
                Property(name="machine", data_type=DataType.TEXT),
                Property(name="manufacturer", data_type=DataType.TEXT),
                Property(name="manual_type", data_type=DataType.TEXT),
                Property(name="section_id", data_type=DataType.TEXT),
                Property(name="section_title", data_type=DataType.TEXT),
                Property(name="chunk_index_within_section", data_type=DataType.INT),
                Property(name="chunk_text", data_type=DataType.TEXT),
                Property(name="images", data_type=DataType.TEXT_ARRAY),
            ],
        )

        print(f"Created collection '{COLLECTION_NAME}'")

    finally:
        client.close()

if __name__ == "__main__":
    main()