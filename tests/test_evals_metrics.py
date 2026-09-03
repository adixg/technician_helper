from technician_helper.evals import metrics

VALID = {
    "likely_causes": [{"cause": "bearing wear", "why": "matches a prior incident"}],
    "recommended_checks": ["Inspect the bearing housing"],
    "manual_references": [
        {"section_title": "Bearing Inspection", "source_pdf": "m.pdf", "reason": "relevant"}
    ],
    "similar_incidents": [
        {"machine_id": "M01", "fault_code": "E102", "summary": "bearing replaced"}
    ],
    "clarifying_questions": [],
    "escalation_needed": False,
    "escalation_reason": "",
    "confidence": "medium",
    "evidence_gaps": [],
}


class TestRetrievalMetrics:
    def test_recall_at_k(self):
        assert metrics.recall_at_k(["a", "b", "c"], ["a", "d"], k=3) == 0.5
        assert metrics.recall_at_k(["a", "b"], ["a", "b"], k=5) == 1.0
        assert metrics.recall_at_k(["x"], ["a"], k=3) == 0.0

    def test_recall_respects_k(self):
        assert metrics.recall_at_k(["x", "y", "a"], ["a"], k=2) == 0.0
        assert metrics.recall_at_k(["x", "y", "a"], ["a"], k=3) == 1.0

    def test_empty_expected_is_vacuous(self):
        assert metrics.recall_at_k([], [], k=5) == 1.0
        assert metrics.hit_rate_at_k([], [], k=5) == 1.0
        assert metrics.mrr([], []) == 1.0

    def test_hit_rate(self):
        assert metrics.hit_rate_at_k(["a", "b"], ["b"], k=2) == 1.0
        assert metrics.hit_rate_at_k(["a", "b"], ["z"], k=2) == 0.0

    def test_mrr(self):
        assert metrics.mrr(["a", "b", "c"], ["c"]) == 1 / 3
        assert metrics.mrr(["a", "b"], ["a"]) == 1.0
        assert metrics.mrr(["a", "b"], ["z"]) == 0.0


class TestSchemaValid:
    def test_valid_object(self):
        assert metrics.schema_valid(VALID) == 1.0

    def test_none(self):
        assert metrics.schema_valid(None) == 0.0

    def test_broken(self):
        broken = dict(VALID)
        del broken["confidence"]
        assert metrics.schema_valid(broken) == 0.0


class TestGroundedness:
    def test_fully_supported(self):
        evidence = ["The bearing shows wear. Bearing Inspection section. Bearing was replaced."]
        assert metrics.groundedness(VALID, evidence) == 1.0

    def test_unsupported_claims(self):
        assert metrics.groundedness(VALID, ["completely unrelated hydraulic text"]) == 0.0

    def test_no_claims_is_one(self):
        empty = dict(VALID, likely_causes=[], manual_references=[], similar_incidents=[])
        assert metrics.groundedness(empty, []) == 1.0

    def test_none_result(self):
        assert metrics.groundedness(None, ["x"]) == 0.0


class TestFieldMatch:
    def test_all_pass(self):
        expected = {
            "confidence": ["medium", "high"],
            "escalation_needed": False,
            "must_mention": ["bearing"],
            "manual_refs_expected": True,
        }
        assert metrics.field_match(VALID, expected) == 1.0

    def test_partial(self):
        expected = {"confidence": "high", "must_mention": ["bearing"]}
        assert metrics.field_match(VALID, expected) == 0.5

    def test_empty_expected(self):
        assert metrics.field_match(VALID, {}) == 1.0

    def test_none_result(self):
        assert metrics.field_match(None, {"confidence": "low"}) == 0.0

    def test_manual_refs_expected_false(self):
        no_refs = dict(VALID, manual_references=[])
        assert metrics.field_match(no_refs, {"manual_refs_expected": False}) == 1.0


class TestCompleteness:
    def test_full(self):
        assert metrics.completeness(VALID) == 1.0

    def test_half(self):
        assert metrics.completeness(dict(VALID, recommended_checks=[])) == 0.5

    def test_none(self):
        assert metrics.completeness(None) == 0.0
