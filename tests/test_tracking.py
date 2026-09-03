import pytest

from technician_helper import tracking
from technician_helper.config import settings


@pytest.fixture(autouse=True)
def _isolated_runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "runs_dir", tmp_path / "runs")
    monkeypatch.setattr(settings, "mlflow_enabled", False)


def test_log_run_appends_and_loads_back():
    rid1 = tracking.log_run("rag_eval", {"k": 5}, {"recall_at_k": 0.8, "schema_valid": 0.9})
    rid2 = tracking.log_run("rag_eval", {"k": 5}, {"recall_at_k": 0.85, "schema_valid": 0.92})

    runs = tracking.load_runs()
    assert [r["run_id"] for r in runs] == [rid1, rid2]
    assert runs[0]["params"] == {"k": 5}
    assert runs[1]["metrics"]["recall_at_k"] == 0.85
    assert "git_sha" in runs[0]


def test_load_runs_empty_when_no_file():
    assert tracking.load_runs() == []


def test_history_filters_by_metric_and_name():
    tracking.log_run("rag_eval", {}, {"recall_at_k": 0.7})
    tracking.log_run("rag_eval", {}, {"recall_at_k": 0.75})
    tracking.log_run("other", {}, {"recall_at_k": 0.1})

    hist = tracking.history("recall_at_k", name="rag_eval")
    assert [v for _, v in hist] == [0.7, 0.75]

    assert tracking.history("nonexistent") == []


def test_log_run_copies_artifacts(tmp_path):
    art = tmp_path / "report.md"
    art.write_text("# report", encoding="utf-8")

    rid = tracking.log_run("rag_eval", {}, {"x": 1.0}, artifacts=[art])

    copied = settings.runs_dir / rid / "report.md"
    assert copied.exists()
    assert copied.read_text(encoding="utf-8") == "# report"
