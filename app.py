import json
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import List

import streamlit as st

from query_manuals import semantic_query as query_manuals
from query_incident_logs import semantic_query as query_incident_logs
from rag_fusion import run_rag_fusion

from docling_code import convert_pdf
from sections_json_gen import generate_sections_json
from chunks_json_gen import generate_chunks_json
from upload_manual_chunks import upload_manual_chunks

from incident_ingest import (
    build_incident_record_from_form,
    upload_single_incident_to_weaviate,
    append_incident_to_csv,
)


st.set_page_config(
    page_title="Technician Helper",
    page_icon="🛠️",
    layout="wide",
)

st.title("Technician Helper")
st.caption(
    "Upload manuals and logs, ingest manuals into Weaviate, add incidents, run retrieval, and complete the troubleshooting pipeline."
)

DEFAULT_EXAMPLES = [
    "Machine M01 has fault code E102 with vibration and abnormal bearing noise. What should I inspect first?",
    "Pump M01 has recurring vibration after restart with fault code E102. What are the likely causes?",
    "How should the motor be grounded?",
]


if "query_text" not in st.session_state:
    st.session_state.query_text = DEFAULT_EXAMPLES[0]

if "pipeline_stage" not in st.session_state:
    st.session_state.pipeline_stage = "Idle"

if "pipeline_message" not in st.session_state:
    st.session_state.pipeline_message = "Waiting for input."

if "manual_results" not in st.session_state:
    st.session_state.manual_results = None

if "log_results" not in st.session_state:
    st.session_state.log_results = None

if "result_json" not in st.session_state:
    st.session_state.result_json = None

if "last_error" not in st.session_state:
    st.session_state.last_error = None

if "ingest_error" not in st.session_state:
    st.session_state.ingest_error = None

if "ingest_logs" not in st.session_state:
    st.session_state.ingest_logs = []

if "ingest_done" not in st.session_state:
    st.session_state.ingest_done = False

if "ingest_result_paths" not in st.session_state:
    st.session_state.ingest_result_paths = None

if "incident_add_error" not in st.session_state:
    st.session_state.incident_add_error = None

if "incident_add_success" not in st.session_state:
    st.session_state.incident_add_success = None

if "last_added_incident" not in st.session_state:
    st.session_state.last_added_incident = None


def set_stage(stage: str, message: str, placeholder) -> None:
    st.session_state.pipeline_stage = stage
    st.session_state.pipeline_message = message
    with placeholder.container():
        st.info(f"**Stage:** {stage}\n\n{message}")


def retrieval_stage_callback(prefix: str, placeholder):
    def _callback(message: str):
        set_stage(prefix, message, placeholder)
    return _callback


def fusion_stage_callback(placeholder):
    def _callback(stage: str, message: str):
        set_stage(stage, message, placeholder)
    return _callback


def save_uploaded_files(uploaded_files, target_dir: Path) -> List[str]:
    saved_paths: List[str] = []
    target_dir.mkdir(parents=True, exist_ok=True)

    if not uploaded_files:
        return saved_paths

    for uploaded_file in uploaded_files:
        file_path = target_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(str(file_path))

    return saved_paths


def run_manual_retrieval(query: str, placeholder):
    set_stage("Manual Retrieval", "Starting manual retrieval...", placeholder)
    results = query_manuals(
        question=query,
        top_k=5,
        stage_callback=retrieval_stage_callback("Manual Retrieval", placeholder),
    )
    set_stage("Manual Retrieval", "Manual retrieval finished.", placeholder)
    return results


def run_log_retrieval(query: str, placeholder):
    set_stage("Log Retrieval", "Starting log retrieval...", placeholder)
    results = query_incident_logs(
        query_text=query,
        top_k=5,
        stage_callback=retrieval_stage_callback("Log Retrieval", placeholder),
    )
    set_stage("Log Retrieval", "Log retrieval finished.", placeholder)
    return results


def run_full_pipeline(query: str, manual_files, log_files, placeholder):
    tmp_root = Path(tempfile.mkdtemp(prefix="technician_helper_"))
    manuals_dir = tmp_root / "manuals"
    logs_dir = tmp_root / "logs"

    try:
        set_stage("Preparation", "Saving uploaded files...", placeholder)
        manual_paths = save_uploaded_files(manual_files, manuals_dir)
        log_paths = save_uploaded_files(log_files, logs_dir)

        pipeline_output = run_rag_fusion(
            query=query,
            manual_paths=manual_paths,
            log_paths=log_paths,
            stage_callback=fusion_stage_callback(placeholder),
        )

        result_json = pipeline_output["result"]
        manual_results = pipeline_output["manual_results"]
        log_results = pipeline_output["incident_results"]

        return result_json, manual_results, log_results, None

    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        set_stage("Pipeline Error", "The troubleshooting pipeline failed.", placeholder)
        return None, None, None, error_text

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def ingest_manual_pdf(
    uploaded_pdf,
    collection_name: str,
    embed_model: str,
    max_chars: int,
    min_chars: int,
    batch_size: int,
    machine: str | None,
    manufacturer: str | None,
    manual_type: str | None,
    progress_bar,
    status_placeholder,
    log_placeholder,
):
    st.session_state.ingest_logs = []
    st.session_state.ingest_error = None
    st.session_state.ingest_done = False

    def progress_callback(stage: str, message: str, pct: int):
        pct = max(0, min(100, int(pct)))
        progress_bar.progress(pct)
        status_placeholder.info(f"**Manual ingestion stage:** {stage}\n\n{message}")
        st.session_state.ingest_logs.append(f"[{pct:>3}%] [{stage}] {message}")
        log_placeholder.text("\n".join(st.session_state.ingest_logs[-12:]))

    tmp_root = Path(tempfile.mkdtemp(prefix="manual_ingest_"))

    try:
        input_dir = tmp_root / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = input_dir / uploaded_pdf.name
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        progress_callback("preparation", "Saved uploaded PDF to temporary directory.", 1)

        md_path = convert_pdf(
            source=pdf_path,
            output_dir=Path("data/manuals_converted"),
            progress_callback=progress_callback,
        )

        sections_json_path = generate_sections_json(
            md_path=md_path,
            output_dir=Path("data/manuals_sections"),
            machine=machine,
            manufacturer=manufacturer,
            manual_type=manual_type,
            progress_callback=progress_callback,
        )

        chunks_json_path = generate_chunks_json(
            sections_json_path=sections_json_path,
            output_dir=Path("data/manuals_chunks"),
            max_chars=max_chars,
            min_chars=min_chars,
            progress_callback=progress_callback,
        )

        upload_manual_chunks(
            chunks_json_path=chunks_json_path,
            collection_name=collection_name,
            embed_model=embed_model,
            batch_size=batch_size,
            progress_callback=progress_callback,
        )

        st.session_state.ingest_done = True
        return {
            "md_path": str(md_path),
            "sections_json_path": str(sections_json_path),
            "chunks_json_path": str(chunks_json_path),
        }

    except Exception as exc:
        st.session_state.ingest_error = (
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        )
        return None

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("1. Upload evidence")

    manual_files = st.file_uploader(
        "Upload OEM manuals",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        key="manual_files_for_rag",
    )

    log_files = st.file_uploader(
        "Upload incident / maintenance logs",
        type=["csv", "json", "txt", "md"],
        accept_multiple_files=True,
        key="log_files_for_rag",
    )

    with st.expander("Uploaded files", expanded=True):
        st.write(f"Manuals: **{len(manual_files) if manual_files else 0}**")
        if manual_files:
            for f in manual_files:
                st.write(f"- {f.name}")

        st.write(f"Logs: **{len(log_files) if log_files else 0}**")
        if log_files:
            for f in log_files:
                st.write(f"- {f.name}")

with right_col:
    st.subheader("2. Technician query")

    example_cols = st.columns(len(DEFAULT_EXAMPLES))
    for i, example in enumerate(DEFAULT_EXAMPLES):
        with example_cols[i]:
            if st.button(f"Example {i + 1}", use_container_width=True):
                st.session_state.query_text = example

    st.text_area(
        "Query",
        key="query_text",
        height=180,
        placeholder="Enter a troubleshooting question...",
    )

    b1, b2, b3 = st.columns(3)
    with b1:
        run_manual_button = st.button("Run manual retrieval", use_container_width=True)
    with b2:
        run_log_button = st.button("Run log retrieval", use_container_width=True)
    with b3:
        run_pipeline_button = st.button(
            "Run troubleshooting pipeline",
            type="primary",
            use_container_width=True,
        )

st.divider()

st.subheader("3. Ingest a manual PDF into Weaviate")

ingest_pdf = st.file_uploader(
    "Upload a PDF manual to ingest",
    type=["pdf"],
    accept_multiple_files=False,
    key="manual_pdf_ingest",
)

ingest_col1, ingest_col2, ingest_col3 = st.columns(3)
with ingest_col1:
    collection_name = st.text_input("Collection name", value="ManualChunk")
with ingest_col2:
    max_chars = st.number_input(
        "Max chars per chunk", min_value=100, value=2000, step=100
    )
with ingest_col3:
    min_chars = st.number_input(
        "Min chars merge threshold", min_value=1, value=200, step=50
    )

ingest_col4, ingest_col5, ingest_col6 = st.columns(3)
with ingest_col4:
    batch_size = st.number_input(
        "Embedding batch size", min_value=1, value=2, step=1
    )
with ingest_col5:
    machine = st.text_input("Machine metadata (optional)", value="")
with ingest_col6:
    manufacturer = st.text_input("Manufacturer metadata (optional)", value="")

manual_type = st.text_input("Manual type metadata (optional)", value="")
embed_model = st.text_input("Embedding model", value="Qwen/Qwen3-Embedding-0.6B")

run_ingest_button = st.button(
    "Process PDF and upload to Weaviate",
    use_container_width=True,
)

ingest_progress_bar = st.progress(0)
ingest_status_placeholder = st.empty()
ingest_log_placeholder = st.empty()

if run_ingest_button:
    st.session_state.ingest_result_paths = None

    if ingest_pdf is None:
        ingest_status_placeholder.warning("Please upload a PDF first.")
    else:
        result_paths = ingest_manual_pdf(
            uploaded_pdf=ingest_pdf,
            collection_name=collection_name,
            embed_model=embed_model,
            max_chars=int(max_chars),
            min_chars=int(min_chars),
            batch_size=int(batch_size),
            machine=machine.strip() or None,
            manufacturer=manufacturer.strip() or None,
            manual_type=manual_type.strip() or None,
            progress_bar=ingest_progress_bar,
            status_placeholder=ingest_status_placeholder,
            log_placeholder=ingest_log_placeholder,
        )

        if result_paths is not None:
            st.session_state.ingest_result_paths = result_paths
            st.success("Manual PDF processed and uploaded successfully.")
            st.json(result_paths, expanded=True)
        elif st.session_state.ingest_error:
            st.error("Manual ingestion failed.")
            st.code(st.session_state.ingest_error, language="text")

if st.session_state.ingest_result_paths is not None:
    md_file_path = st.session_state.ingest_result_paths["md_path"]

    with st.expander("View generated markdown", expanded=False):
        try:
            with open(md_file_path, "r", encoding="utf-8") as f:
                md_text = f.read()

            st.text_area(
                "Generated markdown",
                value=md_text,
                height=500,
            )

            st.download_button(
                label="Download markdown file",
                data=md_text,
                file_name=Path(md_file_path).name,
                mime="text/markdown",
            )

        except Exception as e:
            st.error(f"Could not load markdown file: {e}")

if st.session_state.ingest_logs:
    with st.expander("Manual ingestion log", expanded=True):
        st.text("\n".join(st.session_state.ingest_logs))

st.divider()

st.subheader("4. Add a new incident log entry")

with st.form("add_incident_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        incident_id = st.text_input("incident_id")
        machine_id = st.text_input("machine_id")
        machine_type = st.text_input("machine_type")
        location = st.text_input("location")
        incident_datetime = st.text_input(
            "incident_datetime",
            placeholder="2024-05-09 10:50:12",
        )
        incident_type = st.text_input("incident_type")
        failure_code = st.text_input("failure_code")

    with c2:
        failure_description = st.text_area("failure_description", height=120)
        sensor_id = st.text_input("sensor_id")
        sensor_type = st.text_input("sensor_type")
        sensor_value = st.text_input("sensor_value")
        maintenance_type = st.text_input("maintenance_type")
        maintenance_action = st.text_area("maintenance_action", height=100)

    with c3:
        downtime_minutes = st.text_input("downtime_minutes")
        reported_by = st.text_input("reported_by")
        resolved_datetime = st.text_input(
            "resolved_datetime",
            placeholder="2024-05-09 12:10:00",
        )
        resolution_status = st.text_input("resolution_status")
        cost_estimate = st.text_input("cost_estimate")
        root_cause = st.text_area("root_cause", height=100)

    persist_to_csv = st.checkbox("Also append this incident to CSV", value=True)

    csv_save_path = st.text_input(
        "CSV path",
        value="data/logs/predictive-maintenance-incident-log.csv",
    )

    incident_embed_model = st.text_input(
        "Incident embedding model",
        value="sentence-transformers/all-MiniLM-L6-v2",
    )

    incident_collection = st.text_input(
        "Incident collection name",
        value="IncidentLogs",
    )

    submit_incident = st.form_submit_button(
        "Add incident and upload to Weaviate"
    )

if submit_incident:
    st.session_state.incident_add_error = None
    st.session_state.incident_add_success = None
    st.session_state.last_added_incident = None

    try:
        form_data = {
            "incident_id": incident_id,
            "machine_id": machine_id,
            "machine_type": machine_type,
            "location": location,
            "incident_datetime": incident_datetime,
            "incident_type": incident_type,
            "failure_code": failure_code,
            "failure_description": failure_description,
            "sensor_id": sensor_id,
            "sensor_type": sensor_type,
            "sensor_value": sensor_value,
            "maintenance_type": maintenance_type,
            "maintenance_action": maintenance_action,
            "downtime_minutes": downtime_minutes,
            "reported_by": reported_by,
            "resolved_datetime": resolved_datetime,
            "resolution_status": resolution_status,
            "cost_estimate": cost_estimate,
            "root_cause": root_cause,
        }

        record = build_incident_record_from_form(form_data)

        if not record.get("incident_id"):
            raise ValueError("incident_id is required")

        upload_single_incident_to_weaviate(
            record=record,
            collection_name=incident_collection,
            embed_model_name=incident_embed_model,
        )

        if persist_to_csv:
            append_incident_to_csv(record, csv_save_path)

        st.session_state.last_added_incident = record
        st.session_state.incident_add_success = (
            f"Incident {record['incident_id']} uploaded to Weaviate successfully."
        )

    except Exception as exc:
        st.session_state.incident_add_error = (
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        )

if st.session_state.incident_add_success:
    st.success(st.session_state.incident_add_success)

if st.session_state.incident_add_error:
    st.error("Incident ingestion failed.")
    st.code(st.session_state.incident_add_error, language="text")

if st.session_state.last_added_incident is not None:
    with st.expander("Last added incident record", expanded=False):
        st.json(st.session_state.last_added_incident, expanded=True)

st.divider()

st.subheader("5. Pipeline status")
stage_placeholder = st.empty()
with stage_placeholder.container():
    st.info(
        f"**Stage:** {st.session_state.pipeline_stage}\n\n"
        f"{st.session_state.pipeline_message}"
    )

query = st.session_state.query_text.strip()

if run_manual_button:
    st.session_state.last_error = None
    st.session_state.result_json = None

    if not query:
        set_stage("Validation", "Please enter a query first.", stage_placeholder)
    else:
        try:
            st.session_state.manual_results = run_manual_retrieval(
                query, stage_placeholder
            )
        except Exception as exc:
            st.session_state.manual_results = None
            st.session_state.last_error = (
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            )
            set_stage(
                "Manual Retrieval Error",
                "Manual retrieval failed.",
                stage_placeholder,
            )

if run_log_button:
    st.session_state.last_error = None
    st.session_state.result_json = None

    if not query:
        set_stage("Validation", "Please enter a query first.", stage_placeholder)
    else:
        try:
            st.session_state.log_results = run_log_retrieval(
                query, stage_placeholder
            )
        except Exception as exc:
            st.session_state.log_results = None
            st.session_state.last_error = (
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            )
            set_stage(
                "Log Retrieval Error",
                "Log retrieval failed.",
                stage_placeholder,
            )

if run_pipeline_button:
    st.session_state.last_error = None

    if not query:
        set_stage("Validation", "Please enter a query first.", stage_placeholder)
    else:
        result_json, manual_results, log_results, error_text = run_full_pipeline(
            query=query,
            manual_files=manual_files,
            log_files=log_files,
            placeholder=stage_placeholder,
        )

        st.session_state.result_json = result_json
        st.session_state.manual_results = manual_results
        st.session_state.log_results = log_results
        st.session_state.last_error = error_text

st.divider()

manual_col, log_col = st.columns(2)

with manual_col:
    st.subheader("6. Manual retrieval output")
    if st.session_state.manual_results is not None:
        st.json(st.session_state.manual_results, expanded=False)
    else:
        st.write("No manual retrieval output yet.")

with log_col:
    st.subheader("7. Log retrieval output")
    if st.session_state.log_results is not None:
        st.json(st.session_state.log_results, expanded=False)
    else:
        st.write("No log retrieval output yet.")

st.divider()

st.subheader("8. Final JSON output")

if st.session_state.result_json is not None:
    st.json(st.session_state.result_json, expanded=True)
elif st.session_state.last_error:
    st.error("An error occurred.")
    st.code(st.session_state.last_error, language="text")
else:
    st.code(
        json.dumps(
            {
                "likely_causes": [{"cause": "", "why": ""}],
                "recommended_checks": [""],
                "manual_references": [
                    {"section_title": "", "source_pdf": "", "reason": ""}
                ],
                "similar_incidents": [
                    {"machine_id": "", "fault_code": "", "summary": ""}
                ],
                "clarifying_questions": [""],
                "escalation_needed": False,
                "escalation_reason": "",
                "confidence": "low|medium|high",
                "evidence_gaps": [""],
            },
            indent=2,
        ),
        language="json",
    )

st.divider()

if st.button("Clear all outputs"):
    st.session_state.manual_results = None
    st.session_state.log_results = None
    st.session_state.result_json = None
    st.session_state.last_error = None
    st.session_state.pipeline_stage = "Idle"
    st.session_state.pipeline_message = "Waiting for input."
    st.session_state.ingest_error = None
    st.session_state.ingest_logs = []
    st.session_state.ingest_done = False
    st.session_state.ingest_result_paths = None
    st.session_state.incident_add_error = None
    st.session_state.incident_add_success = None
    st.session_state.last_added_incident = None
    st.rerun()