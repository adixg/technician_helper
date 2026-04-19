# technician_helper
Software that helps technicians using manuals and historical logs

# Docker startup:
docker run -d --name weaviate -p 8080:8080 -p 50051:50051 -v "${PWD}\weaviate_data:/var/lib/weaviate" -e QUERY_DEFAULTS_LIMIT=20 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true -e DEFAULT_VECTORIZER_MODULE=none semitechnologies/weaviate:latest
