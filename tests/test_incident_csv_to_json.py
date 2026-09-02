import numpy as np

from technician_helper.ingestion.incident_csv_to_json import (
    build_incident_text,
    clean_value,
    to_float_or_none,
    to_int_or_none,
    to_rfc3339_utc,
)


class TestCleanValue:
    def test_trims_strings(self):
        assert clean_value("  hi  ") == "hi"

    def test_empty_string_becomes_none(self):
        assert clean_value("   ") is None

    def test_nan_becomes_none(self):
        assert clean_value(np.nan) is None

    def test_keeps_numbers(self):
        assert clean_value(0) == 0


class TestToRfc3339Utc:
    def test_parses_naive_datetime(self):
        assert to_rfc3339_utc("2024-05-09 10:50:12") == "2024-05-09T10:50:12Z"

    def test_none_passthrough(self):
        assert to_rfc3339_utc(None) is None

    def test_garbage_becomes_none(self):
        assert to_rfc3339_utc("not a date") is None


class TestNumericCoercion:
    def test_float_ok(self):
        assert to_float_or_none("3.5") == 3.5

    def test_float_garbage_none(self):
        assert to_float_or_none("abc") is None

    def test_int_truncates_float_string(self):
        assert to_int_or_none("7.9") == 7

    def test_int_none_passthrough(self):
        assert to_int_or_none(None) is None


class TestBuildIncidentText:
    def test_includes_core_fields(self):
        text = build_incident_text(
            {
                "incident_id": "INC-1",
                "machine_id": "M01",
                "machine_type": "pump",
                "failure_code": "E102",
                "root_cause": "bearing wear",
            }
        )
        assert "Incident INC-1" in text
        assert "Failure code: E102." in text
        assert "Root cause: bearing wear." in text

    def test_empty_row_is_empty_string(self):
        assert build_incident_text({}) == ""
