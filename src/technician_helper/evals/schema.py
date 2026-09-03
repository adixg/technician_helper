"""Typed records for the eval harness."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GoldenRecord:
    """One labelled evaluation case."""

    id: str
    query: str
    expected_chunk_ids: list[str] = field(default_factory=list)
    expected: dict[str, Any] = field(default_factory=dict)
    slice: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GoldenRecord:
        return cls(
            id=str(d["id"]),
            query=d["query"],
            expected_chunk_ids=list(d.get("expected_chunk_ids", [])),
            expected=dict(d.get("expected", {})),
            slice=dict(d.get("slice", {})),
        )


@dataclass
class RunRecord:
    """The pipeline's output for one golden case (freshly run or replayed)."""

    id: str
    query: str
    result: dict[str, Any] | None = None
    retrieved_manual_ids: list[str] = field(default_factory=list)
    retrieved_incident_ids: list[str] = field(default_factory=list)
    evidence_texts: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    repair_attempts: int = 0
    error: str | None = None

    @property
    def retrieved_ids(self) -> list[str]:
        return [*self.retrieved_manual_ids, *self.retrieved_incident_ids]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "result": self.result,
            "retrieved_manual_ids": self.retrieved_manual_ids,
            "retrieved_incident_ids": self.retrieved_incident_ids,
            "evidence_texts": self.evidence_texts,
            "timings_ms": self.timings_ms,
            "repair_attempts": self.repair_attempts,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunRecord:
        return cls(
            id=str(d["id"]),
            query=d["query"],
            result=d.get("result"),
            retrieved_manual_ids=list(d.get("retrieved_manual_ids", [])),
            retrieved_incident_ids=list(d.get("retrieved_incident_ids", [])),
            evidence_texts=list(d.get("evidence_texts", [])),
            timings_ms=dict(d.get("timings_ms", {})),
            repair_attempts=int(d.get("repair_attempts", 0)),
            error=d.get("error"),
        )


@dataclass
class Report:
    dataset: str
    n: int
    aggregate: dict[str, float]
    per_record: dict[str, dict[str, float]]
    per_slice: dict[str, dict[str, float]]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "n": self.n,
            "aggregate": self.aggregate,
            "per_record": self.per_record,
            "per_slice": self.per_slice,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Report:
        return cls(
            dataset=d["dataset"],
            n=int(d["n"]),
            aggregate=dict(d.get("aggregate", {})),
            per_record=dict(d.get("per_record", {})),
            per_slice=dict(d.get("per_slice", {})),
            meta=dict(d.get("meta", {})),
        )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
