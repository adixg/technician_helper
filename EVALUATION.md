# Evaluation

The `th-eval` harness scores the retrieval-augmented troubleshooting pipeline
against a labelled **golden dataset** and gates regressions.

## Layout

```
evals/
├── datasets/golden.jsonl            # labelled cases: query, expected_chunk_ids, expected fields, slice tags
├── fixtures/pipeline_runs/golden.jsonl   # cached pipeline outputs — lets `score` run offline / in CI
├── reports/baseline.json            # committed baseline for the regression gate
└── thresholds.toml                  # metric floors + max allowed regression
```

The code is in `src/technician_helper/evals/` (`metrics.py` is pure and
dependency-free) and `src/technician_helper/tracking.py`.

## Commands

```bash
# Recompute metrics from committed fixtures — no Weaviate / HF needed. Used in CI.
th-eval score

# Same, plus diff against the baseline and fail on regression.
th-eval score --baseline evals/reports/baseline.json --gate

# Run the real pipeline for every golden query, refresh fixtures, write a report,
# and log the run to runs/index.jsonl. Needs Weaviate up and HF_TOKEN set.
th-eval run

# Diff two saved report JSON files.
th-eval compare --baseline a.json --current b.json --gate

# Show tracked eval runs over time.
th-runs
```

Every `score`/`run` writes a timestamped `evals/reports/<dataset>-<ts>.{json,md}`
(git-ignored); `run` also appends a row to `runs/index.jsonl`.

## Metrics

All are means over the dataset, in `[0, 1]` unless noted, higher is better.

| Metric | Definition |
| --- | --- |
| `recall_at_k` | Fraction of a case's `expected_chunk_ids` found in the top-`k` retrieved chunk ids (`k` = `eval_k`, default 5). |
| `hit_rate` | 1 if any expected chunk id is in the top-`k`, else 0. |
| `mrr` | Reciprocal rank of the first expected chunk id in the retrieved list. |
| `schema_valid` | 1 if the answer object passes the pipeline's schema validation. |
| `groundedness` | Fraction of generated claims (`likely_causes`, `manual_references`, `similar_incidents`) whose salient tokens appear in the retrieved evidence text. |
| `field_match` | Fraction of a case's `expected` assertions satisfied — `confidence`, `escalation_needed`, `must_mention`, `manual_refs_expected`. |
| `completeness` | 1 when both `likely_causes` and `recommended_checks` are non-empty. |
| `error_rate` | Fraction of cases where the pipeline raised (lower is better). |
| `latency_ms_total` | Mean end-to-end pipeline time in ms (informational, lower is better). |

A case with no labels for a given metric scores 1.0 for it (vacuously satisfied),
so partially labelled golden cases still contribute.

Reports also break every metric down **by slice** (`machine_type`, `category`) —
the hook for fairness / disparity analysis.

## The regression gate

`th-eval score --gate` fails (exit 1) when, against `evals/reports/baseline.json`:

- any aggregate metric is **below its floor** in `evals/thresholds.toml`, or
- any higher-is-better metric **regressed by more than `max_regression`**.

CI runs this on the committed fixtures, so a change that quietly degrades answer
quality fails the build.

## Working with the dataset

- **Add cases**: append lines to `evals/datasets/golden.jsonl`. `expected_chunk_ids`
  and `expected` are optional — an empty expectation just doesn't constrain that
  metric.
- **Refresh fixtures against real data**: with Weaviate populated, run
  `th-eval run`; commit the updated `evals/fixtures/pipeline_runs/golden.jsonl`.
- **Move the baseline**: after an intentional improvement, copy the fresh report
  JSON over `evals/reports/baseline.json` and commit it.

The fixtures shipped in the repo are hand-written seeds so the offline path and CI
work out of the box; replace them with `th-eval run` output once your Weaviate
instance is loaded.

## Experiment tracking

`th-eval run` logs each evaluation to `runs/index.jsonl` (timestamp, git SHA,
params, aggregate metrics; the Markdown/JSON report is copied under
`runs/<run_id>/`). `th-runs` prints the history. Set `MLFLOW_ENABLED=true` (and
`pip install -e ".[tracking]"`) to also log params/metrics/artifacts to MLflow.
