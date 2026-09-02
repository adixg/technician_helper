"""Create the manual chunk collection schema in Weaviate."""

from weaviate.classes.config import Configure, DataType, Property

from technician_helper.clients import weaviate_client
from technician_helper.config import settings


def main():
    client = weaviate_client()
    collection_name = settings.manual_collection

    if client.collections.exists(collection_name):
        print(f"Collection '{collection_name}' already exists")
        return

    client.collections.create(
        name=collection_name,
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
    print(f"Created collection '{collection_name}'")


if __name__ == "__main__":
    main()
