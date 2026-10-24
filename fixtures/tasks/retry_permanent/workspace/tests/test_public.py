import pytest

from retry import PermanentError, TransientError, call_with_retry


def test_recovers_from_transient_error():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise TransientError("try again")
        return "ok"

    assert call_with_retry(flaky, max_attempts=3) == "ok"
    assert len(calls) == 2


def test_permanent_error_is_not_retried():
    calls = []

    def broken():
        calls.append(1)
        raise PermanentError("bad input")

    with pytest.raises(PermanentError):
        call_with_retry(broken, max_attempts=3)
    assert len(calls) == 1
