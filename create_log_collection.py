import weaviate
from weaviate.classes.config import Configure, Property, DataType

COLLECTION_NAME = "IncidentLogs"
DELETE_IF_EXISTS = False


def create_incidentlogs_collection(client, collection_name: str):
    existing = set(client.collections.list_all().keys())

    if collection_name in existing:
        if DELETE_IF_EXISTS:
            client.collections.delete(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        else:
            print(f"Collection already exists: {collection_name}")
            return

    client.collections.create(
        name=collection_name,
        vector_config=[
            Configure.Vectors.self_provided(name="incident_vector")
        ],
        properties=[
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="record_type", data_type=DataType.TEXT),

            Property(name="incident_id", data_type=DataType.TEXT),
            Property(name="machine_id", data_type=DataType.TEXT),
            Property(name="machine_type", data_type=DataType.TEXT),
            Property(name="location", data_type=DataType.TEXT),
            Property(name="incident_datetime", data_type=DataType.DATE),
            Property(name="incident_type", data_type=DataType.TEXT),
            Property(name="failure_code", data_type=DataType.TEXT),
            Property(name="failure_description", data_type=DataType.TEXT),

            Property(name="sensor_id", data_type=DataType.TEXT),
            Property(name="sensor_type", data_type=DataType.TEXT),
            Property(name="sensor_value", data_type=DataType.NUMBER),

            Property(name="maintenance_type", data_type=DataType.TEXT),
            Property(name="maintenance_action", data_type=DataType.TEXT),

            Property(name="downtime_minutes", data_type=DataType.INT),
            Property(name="reported_by", data_type=DataType.TEXT),
            Property(name="resolved_datetime", data_type=DataType.DATE),
            Property(name="resolution_status", data_type=DataType.TEXT),
            Property(name="cost_estimate", data_type=DataType.NUMBER),
            Property(name="root_cause", data_type=DataType.TEXT),

            Property(name="text", data_type=DataType.TEXT),
        ]
    )
    print(f"Created collection: {collection_name}")


def main():
    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051
    )

    try:
        create_incidentlogs_collection(client, COLLECTION_NAME)
    finally:
        client.close()


if __name__ == "__main__":
    main()