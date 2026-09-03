"""Generate pipeline outputs (live) and turn golden + runs into a Report."""

from __future__ import annotations

import logging
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from technician_helper.config import settings
from technician_helper.evals import metrics
from technician_helper.evals.schema import (
    GoldenRecord,
    Report,
    RunRecord,
    read_jsonl,
    write_jsonl,
)

log = logging.getLogger(__name__)

_METRIC_KEYS = (
    "recall_at_k",
    "hit_rate",
    "mrr",
    "schema_valid",
    "groundedness",
    "field_match",
    "completeness",
    "error_rate",
    "latency_ms_total",
)


def load_golden(path: str | Path) -> list[GoldenRecord]:
    return [GoldenRecord.from_dict(d) for d in read_jsonl(path)]


def load_runs(path: str | Path) -> list[RunRecord]:
    return [RunRecord.from_dict(d) for d in read_jsonl(path)]


def save_runs(runs: list[RunRecord], path: str | Path) -> None:
    write_jsonl([r.to_dict() for r in runs], path)


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def run_live(
    golden: list[GoldenRecord],
    *,
    limit: int | None = None,
    top_k_manual: int = 3,
    top_k_logs: int = 3,
) -> list[RunRecord]:
    """Run the real pipeline for each golden query. Needs Weaviate + HF_TOKEN."""
    from technician_helper.pipeline.rag_fusion import run_rag_fusion

    runs: list[RunRecord] = []
    for g in golden[: limit or len(golden)]:
        log.info("eval: running %s", g.id)
        try:
            out = run_rag_fusion(
                query=g.query,
                top_k_manual=top_k_manual,
                top_k_logs=top_k_logs,
            )
            manual = out.get("manual_results") or []
            incident = out.get("incident_results") or []
            runs.append(
                RunRecord(
                    id=g.id,
                    query=g.query,
                    result=out.get("result"),
                    retrieved_manual_ids=[m.get("chunk_id") for m in manual if m.get("chunk_id")],
                    retrieved_incident_ids=[
                        i.get("chunk_id") for i in incident if i.get("chunk_id")
                    ],
                    evidence_texts=(
                        [str(m.get("chunk_text", "")) for m in manual]
                        + [str(i.get("text", "")) for i in incident]
                    ),
                    timings_ms=out.get("timings", {}),
                    repair_attempts=int(out.get("repair_attempts", 0)),
                )
            )
        except Exception as exc:
            log.warning("eval: %s failed: %s", g.id, exc)
            runs.append(
                RunRecord(id=g.id, query=g.query, result=None, error=f"{type(exc).__name__}: {exc}")
            )
    return runs


def _score_one(g: GoldenRecord, r: RunRecord, k: int) -> dict[str, float]:
    return {
        "recall_at_k": metrics.recall_at_k(r.retrieved_ids, g.expected_chunk_ids, k),
        "hit_rate": metrics.hit_rate_at_k(r.retrieved_ids, g.expected_chunk_ids, k),
        "mrr": metrics.mrr(r.retrieved_ids, g.expected_chunk_ids),
        "schema_valid": metrics.schema_valid(r.result),
        "groundedness": metrics.groundedness(r.result, r.evidence_texts),
        "field_match": metrics.field_match(r.result, g.expected),
        "completeness": metrics.completeness(r.result),
        "error_rate": 1.0 if r.error else 0.0,
        "latency_ms_total": float(r.timings_ms.get("total_ms", 0.0)),
    }


def _mean_over(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {key: 0.0 for key in _METRIC_KEYS}
    return {
        key: round(sum(row.get(key, 0.0) for row in rows) / len(rows), 4) for key in _METRIC_KEYS
    }


def evaluate(
    golden: list[GoldenRecord],
    runs: list[RunRecord],
    *,
    dataset: str = "golden",
    k: int | None = None,
) -> Report:
    k = k or settings.eval_k
    by_id = {r.id: r for r in runs}

    per_record: dict[str, dict[str, float]] = {}
    slices: dict[str, list[dict[str, float]]] = defaultdict(list)

    for g in golden:
        r = by_id.get(g.id)
        if r is None:
            log.warning("eval: no run for golden id %s — skipping", g.id)
            continue
        scored = _score_one(g, r, k)
        per_record[g.id] = scored
        for dim, val in g.slice.items():
            slices[f"{dim}={val}"].append(scored)

    return Report(
        dataset=dataset,
        n=len(per_record),
        aggregate=_mean_over(list(per_record.values())),
        per_record=per_record,
        per_slice={name: _mean_over(rows) for name, rows in sorted(slices.items())},
        meta={
            "k": k,
            "git_sha": _git_sha(),
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "llm_model": settings.llm_model,
            "embed_model": settings.embed_model,
        },
    )
