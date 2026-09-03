from technician_helper.evals import report as report_mod
from technician_helper.evals.runner import evaluate
from technician_helper.evals.schema import GoldenRecord, Report, RunRecord

VALID_RESULT = {
    "likely_causes": [{"cause": "bearing wear", "why": "prior incident"}],
    "recommended_checks": ["Inspect bearing housing"],
    "manual_references": [{"section_title": "Bearing", "source_pdf": "m.pdf", "reason": "r"}],
    "similar_incidents": [
        {"machine_id": "M01", "fault_code": "E102", "summary": "bearing replaced"}
    ],
    "clarifying_questions": [],
    "escalation_needed": False,
    "escalation_reason": "",
    "confidence": "medium",
    "evidence_gaps": [],
}


def test_evaluate_aggregates_and_slices():
    golden = [
        GoldenRecord("g1", "q1", ["c1"], {"confidence": ["medium"]}, {"machine_type": "pump"}),
        GoldenRecord("g2", "q2", ["c9"], {}, {"machine_type": "motor"}),
    ]
    runs = [
        RunRecord(
            "g1",
            "q1",
            result=VALID_RESULT,
            retrieved_manual_ids=["c1"],
            evidence_texts=["bearing wear bearing replaced"],
            timings_ms={"total_ms": 100.0},
        ),
        RunRecord("g2", "q2", result=None, error="boom"),
    ]

    rep = evaluate(golden, runs, dataset="unit", k=5)

    assert rep.n == 2
    assert rep.aggregate["schema_valid"] == 0.5  # one valid, one error
    assert rep.aggregate["error_rate"] == 0.5
    assert rep.aggregate["recall_at_k"] == 0.5  # g1 hit, g2 miss
    assert set(rep.per_slice) == {"machine_type=pump", "machine_type=motor"}
    assert rep.per_slice["machine_type=pump"]["schema_valid"] == 1.0
    assert rep.per_slice["machine_type=motor"]["error_rate"] == 1.0


def test_evaluate_skips_missing_runs():
    golden = [GoldenRecord("g1", "q1"), GoldenRecord("g2", "q2")]
    runs = [RunRecord("g1", "q1", result=VALID_RESULT)]
    rep = evaluate(golden, runs, dataset="unit")
    assert rep.n == 1


def test_compare_computes_deltas():
    base = {"recall_at_k": 0.8, "schema_valid": 0.9}
    cur = {"recall_at_k": 0.7, "schema_valid": 0.95}
    deltas = report_mod.compare(base, cur)
    assert deltas["recall_at_k"] == -0.1
    assert deltas["schema_valid"] == 0.05


def test_gate_passes_when_within_bounds():
    current = {"recall_at_k": 0.8, "schema_valid": 0.9}
    deltas = {"recall_at_k": -0.02, "schema_valid": 0.0}
    thresholds = {"max_regression": 0.05, "floors": {"recall_at_k": 0.5, "schema_valid": 0.85}}
    passed, reasons = report_mod.gate(current, deltas, thresholds)
    assert passed and reasons == []


def test_gate_fails_on_floor_breach():
    current = {"recall_at_k": 0.4}
    passed, reasons = report_mod.gate(current, {}, {"floors": {"recall_at_k": 0.5}})
    assert not passed
    assert "below floor" in reasons[0]


def test_gate_fails_on_regression():
    current = {"schema_valid": 0.9}
    deltas = {"schema_valid": -0.2}
    passed, reasons = report_mod.gate(current, deltas, {"max_regression": 0.05, "floors": {}})
    assert not passed
    assert "regressed" in reasons[0]


def test_gate_ignores_regression_on_lower_is_better():
    passed, reasons = report_mod.gate(
        {"error_rate": 0.2, "latency_ms_total": 5000.0},
        {"error_rate": 0.15, "latency_ms_total": 2000.0},
        {"max_regression": 0.05, "floors": {}},
    )
    assert passed and reasons == []


def test_render_markdown_smoke():
    rep = Report(
        dataset="unit",
        n=1,
        aggregate={"recall_at_k": 1.0, "schema_valid": 1.0},
        per_record={
            "g1": {"recall_at_k": 1.0, "schema_valid": 1.0, "groundedness": 1.0, "field_match": 1.0}
        },
        per_slice={"machine_type=pump": {"recall_at_k": 1.0, "schema_valid": 1.0}},
        meta={"git_sha": "abc123", "k": 5},
    )
    md = report_mod.render_markdown(rep)
    assert "# Eval report" in md
    assert "By slice" in md
    assert "abc123" in md
