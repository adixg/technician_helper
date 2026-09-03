"""Render, persist, and regression-gate evaluation reports."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from technician_helper.evals.schema import Report

# Lower-is-better metrics are excluded from the "higher is better" gate logic.
_LOWER_IS_BETTER = {"error_rate", "latency_ms_total"}


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(report: Report) -> str:
    lines: list[str] = []
    lines.append(f"# Eval report — `{report.dataset}` (n={report.n})")
    lines.append("")
    meta = report.meta
    lines.append(
        f"_git `{meta.get('git_sha', '?')}` · k={meta.get('k', '?')} · "
        f"model `{meta.get('llm_model', '?')}` · {meta.get('generated_at', '')}_"
    )
    lines.append("")

    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    for key, val in report.aggregate.items():
        lines.append(f"| {key} | {_fmt(val)} |")
    lines.append("")

    if report.per_slice:
        metric_cols = list(report.aggregate.keys())
        lines.append("## By slice")
        lines.append("")
        lines.append("| Slice | " + " | ".join(metric_cols) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in metric_cols) + " |")
        for name, vals in report.per_slice.items():
            row = " | ".join(_fmt(vals.get(m, 0.0)) for m in metric_cols)
            lines.append(f"| {name} | {row} |")
        lines.append("")

    worst = sorted(
        report.per_record.items(),
        key=lambda kv: (
            kv[1].get("schema_valid", 0.0),
            kv[1].get("groundedness", 0.0),
            kv[1].get("recall_at_k", 0.0),
        ),
    )[:5]
    if worst:
        lines.append("## Lowest-scoring cases")
        lines.append("")
        lines.append("| id | recall@k | schema_valid | groundedness | field_match |")
        lines.append("| --- | --- | --- | --- | --- |")
        for rid, m in worst:
            lines.append(
                f"| {rid} | {_fmt(m['recall_at_k'])} | {_fmt(m['schema_valid'])} | "
                f"{_fmt(m['groundedness'])} | {_fmt(m['field_match'])} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_report(report: Report, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"{report.dataset}-{stamp}.json"
    md_path = out_dir / f"{report.dataset}-{stamp}.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def compare(baseline: dict[str, float], current: dict[str, float]) -> dict[str, float]:
    """Per-metric ``current - baseline`` over the aggregate section."""
    keys = set(baseline) | set(current)
    return {k: round(current.get(k, 0.0) - baseline.get(k, 0.0), 4) for k in sorted(keys)}


def gate(
    current: dict[str, float],
    deltas: dict[str, float],
    thresholds: dict,
) -> tuple[bool, list[str]]:
    """Return (passed, reasons). Fails on an absolute floor breach or a regression.

    ``thresholds`` shape::

        {"max_regression": 0.05, "floors": {"recall_at_k": 0.5, ...}}
    """
    reasons: list[str] = []
    floors: dict[str, float] = thresholds.get("floors", {})
    max_regression = float(thresholds.get("max_regression", 0.05))

    for metric, floor in floors.items():
        value = current.get(metric)
        if value is None:
            reasons.append(f"{metric}: missing from report")
        elif value < floor:
            reasons.append(f"{metric}: {value:.3f} below floor {floor:.3f}")

    for metric, delta in deltas.items():
        if metric in _LOWER_IS_BETTER:
            continue
        if delta < -max_regression:
            reasons.append(
                f"{metric}: regressed by {abs(delta):.3f} (> {max_regression:.3f} allowed)"
            )

    return (not reasons), reasons


def load_thresholds(path: str | Path) -> dict:
    import tomllib

    with open(path, "rb") as f:
        return tomllib.load(f)
