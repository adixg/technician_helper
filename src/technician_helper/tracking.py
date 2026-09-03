"""Lightweight local experiment tracking.

Every run is one JSON line in ``runs/index.jsonl`` (timestamp, git SHA, params,
metrics); artifacts are copied under ``runs/<run_id>/``. When ``mlflow_enabled``
is set the same params/metrics/artifacts are also sent to MLflow.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from technician_helper.config import settings

log = logging.getLogger(__name__)


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


def _index_path() -> Path:
    return settings.runs_dir / "index.jsonl"


def log_run(
    name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: list[str | Path] = (),
) -> str:
    """Append one run record and return its run id."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{name}-{stamp}"

    record = {
        "run_id": run_id,
        "name": name,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "params": params,
        "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
    }

    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    with _index_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if artifacts:
        dest = settings.runs_dir / run_id
        dest.mkdir(parents=True, exist_ok=True)
        for art in artifacts:
            art = Path(art)
            if art.exists():
                shutil.copy2(art, dest / art.name)

    if settings.mlflow_enabled:
        _log_mlflow(run_id, params, metrics, artifacts)

    log.info("logged run %s", run_id)
    return run_id


def _log_mlflow(
    run_id: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: list[str | Path],
) -> None:
    try:
        import mlflow

        with mlflow.start_run(run_name=run_id):
            mlflow.log_params(params)
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            for art in artifacts:
                if Path(art).exists():
                    mlflow.log_artifact(str(art))
    except Exception as exc:
        log.warning("mlflow logging skipped: %s", exc)


def load_runs() -> list[dict[str, Any]]:
    path = _index_path()
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def history(metric: str, name: str | None = None) -> list[tuple[str, float]]:
    """``(timestamp, value)`` pairs for a metric, oldest first."""
    out: list[tuple[str, float]] = []
    for run in load_runs():
        if name and run.get("name") != name:
            continue
        value = run.get("metrics", {}).get(metric)
        if value is not None:
            out.append((run["timestamp"], float(value)))
    return out


def main() -> None:
    """`th-runs` — print recent eval runs as a table."""
    import argparse

    parser = argparse.ArgumentParser(description="Show tracked experiment runs.")
    parser.add_argument("--name", default=None, help="Filter by run name.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--metrics",
        default="recall_at_k,schema_valid,groundedness,field_match",
        help="Comma-separated metric columns to show.",
    )
    args = parser.parse_args()

    cols = [c.strip() for c in args.metrics.split(",") if c.strip()]
    runs = [r for r in load_runs() if not args.name or r.get("name") == args.name]
    runs = runs[-args.limit :]

    if not runs:
        print("No runs recorded yet.")
        return

    header = ["timestamp", "git", *cols]
    widths = [len(h) for h in header]
    rows: list[list[str]] = []
    for r in runs:
        m = r.get("metrics", {})
        row = [
            r["timestamp"],
            str(r.get("git_sha") or "-"),
            *[f"{m[c]:.3f}" if c in m else "-" for c in cols],
        ]
        rows.append(row)
        widths = [max(w, len(cell)) for w, cell in zip(widths, row, strict=False)]

    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


if __name__ == "__main__":
    main()
