"""``th-eval`` — run / score / regression-gate the RAG evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from technician_helper.evals import report as report_mod
from technician_helper.evals import runner
from technician_helper.evals.schema import Report
from technician_helper.logging_config import configure_logging

_DATASETS = "evals/datasets"
_FIXTURES = "evals/fixtures/pipeline_runs"
_REPORTS = "evals/reports"
_THRESHOLDS = "evals/thresholds.toml"


def _golden_path(dataset: str) -> Path:
    return Path(_DATASETS) / f"{dataset}.jsonl"


def _fixture_path(dataset: str) -> Path:
    return Path(_FIXTURES) / f"{dataset}.jsonl"


def _print_aggregate(rep: Report) -> None:
    print(f"\n{rep.dataset}: n={rep.n}")
    for key, val in rep.aggregate.items():
        print(f"  {key:<18} {val:.3f}")
    print()


def _cmd_run(args: argparse.Namespace) -> int:
    golden = runner.load_golden(_golden_path(args.dataset))
    runs = runner.run_live(golden, limit=args.limit)
    fixture = _fixture_path(args.dataset)
    runner.save_runs(runs, fixture)
    print(f"wrote {len(runs)} runs -> {fixture}")

    rep = runner.evaluate(golden, runs, dataset=args.dataset)
    json_path, md_path = report_mod.write_report(rep, _REPORTS)
    _print_aggregate(rep)
    print(f"report -> {md_path}")

    from technician_helper.tracking import log_run

    log_run(
        "rag_eval",
        params={"dataset": args.dataset, "k": rep.meta.get("k"), "n": rep.n, "mode": "live"},
        metrics=rep.aggregate,
        artifacts=[md_path, json_path],
    )
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    golden = runner.load_golden(_golden_path(args.dataset))
    runs = runner.load_runs(_fixture_path(args.dataset))
    rep = runner.evaluate(golden, runs, dataset=args.dataset)
    json_path, md_path = report_mod.write_report(rep, _REPORTS)
    _print_aggregate(rep)
    print(f"report -> {md_path}")

    if not args.baseline:
        return 0

    baseline = Report.from_dict(json.loads(Path(args.baseline).read_text(encoding="utf-8")))
    deltas = report_mod.compare(baseline.aggregate, rep.aggregate)
    print("deltas vs baseline:")
    for metric, delta in deltas.items():
        print(f"  {metric:<18} {delta:+.3f}")

    if args.gate:
        thresholds = report_mod.load_thresholds(args.thresholds)
        passed, reasons = report_mod.gate(rep.aggregate, deltas, thresholds)
        if passed:
            print("\ngate: PASS")
            return 0
        print("\ngate: FAIL")
        for reason in reasons:
            print(f"  - {reason}")
        return 1
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    baseline = Report.from_dict(json.loads(Path(args.baseline).read_text(encoding="utf-8")))
    current = Report.from_dict(json.loads(Path(args.current).read_text(encoding="utf-8")))
    deltas = report_mod.compare(baseline.aggregate, current.aggregate)
    for metric, delta in deltas.items():
        print(f"  {metric:<18} {delta:+.3f}")
    if args.gate:
        thresholds = report_mod.load_thresholds(args.thresholds)
        passed, reasons = report_mod.gate(current.aggregate, deltas, thresholds)
        print(f"\ngate: {'PASS' if passed else 'FAIL'}")
        for reason in reasons:
            print(f"  - {reason}")
        return 0 if passed else 1
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(prog="th-eval", description=__doc__)
    parser.add_argument("--dataset", default="golden", help="Dataset name (default: golden).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the live pipeline and score it (needs Weaviate + HF).")
    p_run.add_argument("--limit", type=int, default=None)
    p_run.set_defaults(func=_cmd_run)

    p_score = sub.add_parser(
        "score", help="Recompute the report from committed fixtures (offline)."
    )
    p_score.add_argument("--baseline", default=None, help="Baseline report JSON to diff against.")
    p_score.add_argument("--gate", action="store_true", help="Exit non-zero on regression.")
    p_score.add_argument("--thresholds", default=_THRESHOLDS)
    p_score.set_defaults(func=_cmd_score)

    p_cmp = sub.add_parser("compare", help="Diff two report JSON files.")
    p_cmp.add_argument("--baseline", required=True)
    p_cmp.add_argument("--current", required=True)
    p_cmp.add_argument("--gate", action="store_true")
    p_cmp.add_argument("--thresholds", default=_THRESHOLDS)
    p_cmp.set_defaults(func=_cmd_compare)

    args = parser.parse_args(argv)
    Path(_REPORTS).mkdir(parents=True, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
