import pytest

from technician_helper.retry import retry_call


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("technician_helper.retry.time.sleep", lambda _: None)


def test_returns_on_first_success():
    calls = []
    assert retry_call(lambda: calls.append(1) or "ok", attempts=3) == "ok"
    assert len(calls) == 1


def test_retries_then_succeeds():
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("transient")
        return "recovered"

    assert retry_call(flaky, attempts=5) == "recovered"
    assert attempts["n"] == 3


def test_raises_last_error_after_exhausting():
    attempts = {"n": 0}

    def always_fails():
        attempts["n"] += 1
        raise TimeoutError(f"fail {attempts['n']}")

    with pytest.raises(TimeoutError, match="fail 3"):
        retry_call(always_fails, attempts=3)
    assert attempts["n"] == 3


def test_does_not_retry_unlisted_exception():
    attempts = {"n": 0}

    def bad():
        attempts["n"] += 1
        raise KeyError("nope")

    with pytest.raises(KeyError):
        retry_call(bad, attempts=5, exceptions=(ConnectionError,))
    assert attempts["n"] == 1


def test_rejects_zero_attempts():
    with pytest.raises(ValueError):
        retry_call(lambda: None, attempts=0)
