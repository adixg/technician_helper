# technician_helper
Software that helps technicians using manuals and historical logs

# Docker startup:
docker run -d --name weaviate -p 8080:8080 -p 50051:50051 -v "${PWD}\weaviate_data:/var/lib/weaviate" -e QUERY_DEFAULTS_LIMIT=20 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true -e DEFAULT_VECTORIZER_MODULE=none semitechnologies/weaviate:latest

# File Overview

This repository contains scripts for ingesting OEM manuals and incident logs into a vector database (Weaviate) and running a Retrieval-Augmented Generation (RAG) troubleshooting assistant through a Streamlit interface.

---

## `app.py`

Main Streamlit application.

Provides UI for:

* ingesting PDF manuals into Weaviate
* adding new incident log entries
* running manual retrieval
* running incident log retrieval
* executing the full troubleshooting pipeline

### Run

```bash
streamlit run app.py
```

---

## `chunks_json_gen.py`

Splits manual **sections JSON** into smaller embedding-ready chunks.

Used before uploading manuals to Weaviate.

**Input**

```
sections JSON
```

**Output**

```
chunks JSON
```

### Run

```bash
python chunks_json_gen.py "data/manuals_sections/manual-sections.json"
```

Optional arguments:

```bash
python chunks_json_gen.py "data/manuals_sections/manual-sections.json" \
    --output_dir data/manuals_chunks \
    --max_chars 2000 \
    --min_chars 200
```

---

## `create_incident_json.py`

Converts incident log CSV into structured JSON records suitable for embedding.

Performs:

* datetime normalization
* numeric cleaning
* text field generation for semantic search

**Input**

```
incident CSV
```

**Output**

```
incident JSON
```

### Run

```bash
python create_incident_json.py data/logs/predictive-maintenance-incident-log.csv
```

Optional:

```bash
python create_incident_json.py \
    data/logs/predictive-maintenance-incident-log.csv \
    --output_json data/logs/incident_chunks.json
```

---

## `create_log_collection.py`

Creates the **IncidentLogs** collection schema inside Weaviate.

Run this before uploading incident JSON.

### Run

```bash
python create_log_collection.py
```

---

## `create_manual_collection.py`

Creates the **ManualChunk** collection schema inside Weaviate.

Run this before uploading manual chunks.

### Run

```bash
python create_manual_collection.py
```

---

## `docling_code.py`

Converts PDF manuals into Markdown using Docling.

Also extracts:

* figures
* tables
* image references

**Input**

```
PDF manual
```

**Output**

```
Markdown with image references
```

### Run

```bash
python docling_code.py "data/manuals/manual.pdf"
```

Optional:

```bash
python docling_code.py "data/manuals/manual.pdf" --output_dir data/manuals_converted
```

---

## `incident_ingest.py`

Utility module for inserting **single incident records** into Weaviate.

Used by:

```
app.py
```

Handles:

* record construction
* embedding
* upload to IncidentLogs collection
* optional CSV persistence

Example usage inside Python:

```python
from incident_ingest import build_incident_record_from_form
from incident_ingest import upload_single_incident_to_weaviate
```

---

## `query_incident_logs.py`

Performs semantic search over the **IncidentLogs** collection.

Returns similar historical incidents.

Example:

```python
from query_incident_logs import semantic_query

results = semantic_query(
    query_text="fault code E102 vibration",
    top_k=5
)
```

---

## `query_manuals.py`

Performs semantic search over the **ManualChunk** collection.

Returns relevant manual sections.

Example:

```python
from query_manuals import semantic_query

results = semantic_query(
    question="How should the motor be grounded?",
    top_k=5
)
```

---

## `rag_fusion.py`

Runs the full troubleshooting pipeline.

Pipeline steps:

1. retrieve manual evidence
2. retrieve incident evidence
3. construct prompt
4. call LLM
5. return structured JSON response

Used by:

```
app.py
```

Example:

```python
from rag_fusion import run_rag_fusion

result = run_rag_fusion(
    query="Pump vibration after restart"
)
```

---

## `sections_json_gen.py`

Splits Markdown manuals into structured sections JSON.

Each section contains:

* title
* text
* image references

**Input**

```
Markdown manual
```

**Output**

```
sections JSON
```

### Run

```bash
python sections_json_gen.py \
    "data/manuals_converted/manual-with-image-refs.md"
```

Optional:

```bash
python sections_json_gen.py \
    "data/manuals_converted/manual-with-image-refs.md" \
    --output_dir data/manuals_sections
```

---

## `test_ollama.py`

Test script for verifying local Ollama model availability.

Useful for confirming:

* model installation
* inference connectivity
* prompt response behavior

### Run

```bash
python test_ollama.py
```

---

## `upload_incident_json.py`

Embeds incident JSON records and uploads them into the **IncidentLogs** collection.

**Input**

```
incident JSON
```

**Output**

```
Weaviate incident embeddings
```

### Run

```bash
python upload_incident_json.py data/logs/incident_chunks.json
```

Optional:

```bash
python upload_incident_json.py \
    data/logs/incident_chunks.json \
    --collection_name IncidentLogs \
    --embed_model sentence-transformers/all-MiniLM-L6-v2
```

---

## `upload_manual_chunks.py`

Embeds manual chunk JSON and uploads them into the **ManualChunk** collection.

**Input**

```
manual chunks JSON
```

**Output**

```
Weaviate manual embeddings
```

### Run

```bash
python upload_manual_chunks.py data/manuals_chunks/manual-chunks.json
```

Optional:

```bash
python upload_manual_chunks.py \
    data/manuals_chunks/manual-chunks.json \
    --collection_name ManualChunk \
    --embed_model Qwen/Qwen3-Embedding-0.6B \
    --batch_size 2
```

---

# Typical Workflow

## Manual ingestion pipeline

```bash
python docling_code.py "data/manuals/manual.pdf"

python sections_json_gen.py \
    "data/manuals_converted/manual-with-image-refs.md"

python chunks_json_gen.py \
    "data/manuals_sections/manual-sections.json" \
    --max_chars 2000 \
    --min_chars 200

python create_manual_collection.py

python upload_manual_chunks.py \
    "data/manuals_chunks/manual-chunks.json"
```

---

## Incident log ingestion pipeline

```bash
python create_incident_json.py \
    data/logs/predictive-maintenance-incident-log.csv

python create_log_collection.py

python upload_incident_json.py \
    data/logs/incident_chunks.json
```

---

## Run the assistant

```bash
streamlit run app.py
```
