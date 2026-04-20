from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from query_manuals import semantic_query as query_manuals
from query_incident_logs import semantic_query as query_incident_logs

load_dotenv()


SCHEMA_EXAMPLE = {
    "likely_causes": [
        {
            "cause": "bearing wear",
            "why": "A similar historical incident with the same fault pattern identified bearing wear as the root cause."
        }
    ],
    "recommended_checks": [
        "Inspect the bearing housing for signs of wear or damage",
        "Check shaft alignment and coupling condition"
    ],
    "manual_references": [
        {
            "section_title": "Bearing Inspection",
            "source_pdf": "OEM Manual.pdf",
            "reason": "This section contains bearing-related inspection guidance relevant to the observed symptoms."
        }
    ],
    "similar_incidents": [
        {
            "machine_id": "M01",
            "fault_code": "E102",
            "summary": "Abnormal bearing noise and vibration were resolved by replacing worn bearings and relubricating."
        }
    ],
    "clarifying_questions": [
        "Is the vibration continuous or only after restart?"
    ],
    "escalation_needed": False,
    "escalation_reason": "",
    "confidence": "medium",
    "evidence_gaps": []
}


def _update_stage(
    stage_callback: Optional[Callable[[str, str], None]],
    stage: str,
    message: str,
) -> None:
    if stage_callback:
        stage_callback(stage, message)


def sanitize_retrieval_text(text: str) -> str:
    """
    Removes obvious prompt leakage or formatting that could confuse the fusion model.
    """
    bad_patterns = [
        r"You are a helpful AI assistant\..*?I don't know\.'?",
        r"Please answer the user's question in the same language as the user's question\..*",
        r"Do not output reasoning\..*",
        r"Do not output <think>\..*",
    ]

    cleaned = text
    for pat in bad_patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    return cleaned.strip()


def stringify_manual_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "None"

    blocks = []
    for i, item in enumerate(results, start=1):
        block = [
            f"[MANUAL {i}]",
            f"chunk_id: {item.get('chunk_id', '')}",
            f"section_title: {item.get('section_title', '')}",
            f"source_pdf_file: {item.get('source_pdf_file', '')}",
            f"manufacturer: {item.get('manufacturer', '')}",
            f"machine: {item.get('machine', '')}",
            f"images: {item.get('images', '')}",
            "chunk_text:",
            str(item.get("chunk_text", "")),
        ]
        blocks.append("\n".join(block))

    return "\n\n".join(blocks)


def stringify_incident_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "None"

    blocks = []
    for i, item in enumerate(results, start=1):
        lines = [f"[LOG {i}]"]
        for key, value in item.items():
            lines.append(f"{key}: {value}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def build_prompt(user_query: str, manual_output: str, incident_output: str) -> str:
    return f"""
/set nothink
/no_think
Return ONLY one valid JSON object with EXACTLY these keys and no others:

- likely_causes
- recommended_checks
- manual_references
- similar_incidents
- clarifying_questions
- escalation_needed
- escalation_reason
- confidence
- evidence_gaps

Schema example:
{json.dumps(SCHEMA_EXAMPLE, indent=2)}

Rules:
1. Use ONLY the retrieved evidence below.
2. Do NOT invent causes, incidents, manual sections, or procedures.
3. Rank likely_causes from most likely to least likely.
4. recommended_checks must be concise and actionable.
5. manual_references must be empty if no relevant manual evidence exists.
6. similar_incidents must include only incidents supported by the retrieved incident output.
7. escalation_needed should be true only if there is a clear safety or operational reason from the evidence.
8. confidence must be exactly one of: low, medium, high
9. evidence_gaps should list ambiguity, missing details, or lack of fault-specific manual guidance.
10. Do NOT output keys such as: query, response, answer, notes, explanation, metadata.
11. If evidence is insufficient, keep the required keys and use empty lists where appropriate.
12. Do not output markdown.
13. Do not output reasoning.
14. Do not output <think>.

User troubleshooting query:
{user_query}

Retrieved incident results:
{incident_output}

Retrieved manual results:
{manual_output}
""".strip()


def create_client() -> InferenceClient:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is not set.")

    return InferenceClient(api_key=hf_token)


def call_llm(
    client: InferenceClient,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 900,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "/set nothink\n"
                    "/no_think\n"
                    "You are a maintenance troubleshooting assistant. "
                    "Return only one valid JSON object matching the required schema. "
                    "Do not add markdown or extra text. "
                    "Do not output reasoning. "
                    "Do not output <think>. "
                    "Do not output explanations. "
                    "Do not output extra keys."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content

    if content is None:
        raise RuntimeError("Model returned empty content (None).")

    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    content = str(content).strip()

    if not content:
        raise RuntimeError("Model returned blank content after normalization.")

    return content


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()

    # Remove fenced code blocks if present
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    # Remove <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    # Try direct JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract first JSON object from remaining text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        preview = text[:1000] if text else "<EMPTY OUTPUT>"
        raise ValueError(f"No JSON object found in model output. Raw output preview:\n{preview}")

    json_text = match.group(0)
    return json.loads(json_text)


def validate_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    required_keys = {
        "likely_causes",
        "recommended_checks",
        "manual_references",
        "similar_incidents",
        "clarifying_questions",
        "escalation_needed",
        "escalation_reason",
        "confidence",
        "evidence_gaps",
    }

    missing = required_keys - set(obj.keys())
    extra = set(obj.keys()) - required_keys

    if missing:
        raise ValueError(f"Model output missing required keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"Model output contains unexpected keys: {sorted(extra)}")

    if obj["confidence"] not in {"low", "medium", "high"}:
        raise ValueError("confidence must be one of: low, medium, high")

    if not isinstance(obj["likely_causes"], list):
        raise ValueError("likely_causes must be a list")
    if not isinstance(obj["recommended_checks"], list):
        raise ValueError("recommended_checks must be a list")
    if not isinstance(obj["manual_references"], list):
        raise ValueError("manual_references must be a list")
    if not isinstance(obj["similar_incidents"], list):
        raise ValueError("similar_incidents must be a list")
    if not isinstance(obj["clarifying_questions"], list):
        raise ValueError("clarifying_questions must be a list")
    if not isinstance(obj["evidence_gaps"], list):
        raise ValueError("evidence_gaps must be a list")
    if not isinstance(obj["escalation_needed"], bool):
        raise ValueError("escalation_needed must be a boolean")
    if not isinstance(obj["escalation_reason"], str):
        raise ValueError("escalation_reason must be a string")

    for item in obj["likely_causes"]:
        if not isinstance(item, dict):
            raise ValueError("Each item in likely_causes must be an object")
        if "cause" not in item or "why" not in item:
            raise ValueError("Each likely_causes item must contain cause and why")

    for item in obj["manual_references"]:
        if not isinstance(item, dict):
            raise ValueError("Each item in manual_references must be an object")
        for key in ("section_title", "source_pdf", "reason"):
            if key not in item:
                raise ValueError(f"Each manual_references item must contain {key}")

    for item in obj["similar_incidents"]:
        if not isinstance(item, dict):
            raise ValueError("Each item in similar_incidents must be an object")
        for key in ("machine_id", "fault_code", "summary"):
            if key not in item:
                raise ValueError(f"Each similar_incidents item must contain {key}")

    return obj


def run_rag_fusion(
    query: str,
    manual_paths: Optional[List[str]] = None,
    log_paths: Optional[List[str]] = None,
    stage_callback: Optional[Callable[[str, str], None]] = None,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    temperature: float = 0.0,
    max_tokens: int = 900,
    top_k_manual: int = 3,
    top_k_logs: int = 3,
    return_debug: bool = False,
) -> Dict[str, Any]:
    """
    Full troubleshooting pipeline:
    1. Retrieve manual evidence
    2. Retrieve incident evidence
    3. Build prompt
    4. Call HF model
    5. Extract + validate JSON
    """

    _update_stage(stage_callback, "Manual Retrieval", "Starting manual retrieval...")

    manual_results = query_manuals(
        question=query,
        top_k=top_k_manual,
        stage_callback=lambda msg: _update_stage(stage_callback, "Manual Retrieval", msg),
    )

    _update_stage(stage_callback, "Log Retrieval", "Starting log retrieval...")

    incident_results = query_incident_logs(
        query_text=query,
        top_k=top_k_logs,
        stage_callback=lambda msg: _update_stage(stage_callback, "Log Retrieval", msg),
    )

    _update_stage(stage_callback, "Prompt Build", "Formatting retrieved evidence...")

    manual_output = sanitize_retrieval_text(stringify_manual_results(manual_results))
    incident_output = sanitize_retrieval_text(stringify_incident_results(incident_results))

    prompt = build_prompt(
        user_query=query,
        manual_output=manual_output,
        incident_output=incident_output,
    )

    _update_stage(stage_callback, "LLM Call", f"Calling model: {model}")

    client = create_client()
    llm_text = call_llm(
        client=client,
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    _update_stage(stage_callback, "Post-Processing", "Extracting JSON from model output...")

    result_obj = extract_json_object(llm_text)

    _update_stage(stage_callback, "Validation", "Validating schema...")

    result_obj = validate_output(result_obj)

    _update_stage(stage_callback, "Complete", "Troubleshooting pipeline finished successfully.")

    output = {
        "result": result_obj,
        "manual_results": manual_results,
        "incident_results": incident_results,
    }

    if return_debug:
        output.update({
            "manual_output": manual_output,
            "incident_output": incident_output,
            "prompt": prompt,
            "raw_llm_text": llm_text,
        })

    return output

if __name__ == "__main__":
    import argparse
    import json
    import traceback

    parser = argparse.ArgumentParser(description="Run troubleshooting RAG pipeline from terminal.")

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Technician troubleshooting query",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Return debug payload including prompt and raw LLM output",
    )

    args = parser.parse_args()

    def terminal_stage_callback(stage, message):
        print(f"[{stage}] {message}")

    try:
        result = run_rag_fusion(
            query=args.query,
            stage_callback=terminal_stage_callback,
            return_debug=args.debug,
        )

        print("\nFINAL RESULT:\n")

        print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        print("\nPIPELINE FAILED\n")
        print(str(e))
        print(traceback.format_exc())