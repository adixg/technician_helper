import copy
import json

import pytest

from technician_helper.pipeline.rag_fusion import (
    SCHEMA_EXAMPLE,
    answer_with_repair,
    extract_json_object,
    sanitize_retrieval_text,
    validate_output,
)


def _valid_json_text() -> str:
    obj = copy.deepcopy(SCHEMA_EXAMPLE)
    obj["clarifying_questions"] = list(obj["clarifying_questions"])
    return json.dumps(obj)


class TestExtractJsonObject:
    def test_plain_json(self):
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_fenced_json_block(self):
        text = '```json\n{"a": 1, "b": [2, 3]}\n```'
        assert extract_json_object(text) == {"a": 1, "b": [2, 3]}

    def test_strips_think_block(self):
        text = '<think>reasoning here</think>\n{"a": 1}'
        assert extract_json_object(text) == {"a": 1}

    def test_extracts_object_embedded_in_prose(self):
        text = 'Here is the answer: {"a": 1} — hope it helps'
        assert extract_json_object(text) == {"a": 1}

    def test_raises_when_no_object_present(self):
        with pytest.raises(ValueError):
            extract_json_object("no json at all")


class TestValidateOutput:
    def _valid(self):
        obj = copy.deepcopy(SCHEMA_EXAMPLE)
        obj["clarifying_questions"] = list(obj["clarifying_questions"])
        return obj

    def test_accepts_schema_example(self):
        assert validate_output(self._valid()) == self._valid()

    def test_rejects_missing_key(self):
        obj = self._valid()
        del obj["confidence"]
        with pytest.raises(ValueError, match="missing required keys"):
            validate_output(obj)

    def test_rejects_extra_key(self):
        obj = self._valid()
        obj["notes"] = "surprise"
        with pytest.raises(ValueError, match="unexpected keys"):
            validate_output(obj)

    def test_rejects_bad_confidence(self):
        obj = self._valid()
        obj["confidence"] = "very-high"
        with pytest.raises(ValueError, match="confidence"):
            validate_output(obj)

    def test_rejects_non_bool_escalation(self):
        obj = self._valid()
        obj["escalation_needed"] = "yes"
        with pytest.raises(ValueError, match="escalation_needed"):
            validate_output(obj)

    def test_rejects_likely_cause_without_why(self):
        obj = self._valid()
        obj["likely_causes"] = [{"cause": "bearing wear"}]
        with pytest.raises(ValueError, match="cause and why"):
            validate_output(obj)


class TestSanitizeRetrievalText:
    def test_passthrough_for_clean_text(self):
        assert sanitize_retrieval_text("  bearing vibration  ") == "bearing vibration"

    def test_strips_prompt_leakage(self):
        dirty = "Do not output reasoning. Something else entirely"
        assert "Do not output reasoning" not in sanitize_retrieval_text(dirty)


class TestAnswerWithRepair:
    def test_succeeds_first_try(self):
        calls = []

        def call_fn(prompt):
            calls.append(prompt)
            return _valid_json_text()

        obj, raw = answer_with_repair(call_fn, "base prompt", repair_attempts=2)
        assert obj["confidence"] == "medium"
        assert len(calls) == 1

    def test_repairs_after_invalid_output(self):
        outputs = iter(["not json at all", "{}", _valid_json_text()])
        prompts = []

        def call_fn(prompt):
            prompts.append(prompt)
            return next(outputs)

        obj, _ = answer_with_repair(call_fn, "base prompt", repair_attempts=2)
        assert obj["confidence"] == "medium"
        assert len(prompts) == 3
        assert "rejected" in prompts[1]  # error fed back into the retry prompt

    def test_gives_up_after_repair_budget(self):
        def call_fn(_prompt):
            return "still not json"

        with pytest.raises(ValueError, match="did not return valid JSON after 2"):
            answer_with_repair(call_fn, "base prompt", repair_attempts=1)
