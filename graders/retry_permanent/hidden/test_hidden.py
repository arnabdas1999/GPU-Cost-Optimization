import pytest

from retry import PermanentError, TransientError, call_with_retry


def scripted(outcomes):
    """Return a callable that raises or returns ``outcomes`` in order, plus its call log."""
    calls = []

    def fn():
        item = outcomes[len(calls)]
        calls.append(item)
        if isinstance(item, BaseException):
            raise item
        return item

    return fn, calls


def test_first_success_is_returned_without_retry():
    fn, calls = scripted(["value"])
    assert call_with_retry(fn) == "value"
    assert len(calls) == 1


def test_transient_failures_then_success():
    fn, calls = scripted([TransientError("a"), TransientError("b"), "ok"])
    assert call_with_retry(fn, max_attempts=3) == "ok"
    assert len(calls) == 3


def test_attempt_limit_is_exact_and_last_error_raised():
    fn, calls = scripted([TransientError(str(i)) for i in range(5)])
    with pytest.raises(TransientError) as info:
        call_with_retry(fn, max_attempts=3)
    assert len(calls) == 3
    assert str(info.value) == "2"


def test_permanent_error_after_transient_stops_immediately():
    fn, calls = scripted([TransientError("t"), PermanentError("p"), "never"])
    with pytest.raises(PermanentError):
        call_with_retry(fn, max_attempts=5)
    assert len(calls) == 2


def test_unclassified_builtin_error_propagates_immediately():
    fn, calls = scripted([ValueError("x"), "never"])
    with pytest.raises(ValueError):
        call_with_retry(fn, max_attempts=3)
    assert len(calls) == 1


def test_custom_classifier_allows_retry():
    fn, calls = scripted([KeyError("k"), "ok"])
    result = call_with_retry(fn, max_attempts=2, is_transient=lambda e: isinstance(e, KeyError))
    assert result == "ok"
    assert len(calls) == 2


def test_custom_classifier_can_reject_transient_error_class():
    fn, calls = scripted([TransientError("t"), "ok"])
    with pytest.raises(TransientError):
        call_with_retry(fn, max_attempts=3, is_transient=lambda e: False)
    assert len(calls) == 1


def test_single_attempt():
    fn, calls = scripted([TransientError("t"), "ok"])
    with pytest.raises(TransientError):
        call_with_retry(fn, max_attempts=1)
    assert len(calls) == 1


def test_invalid_max_attempts():
    with pytest.raises(ValueError):
        call_with_retry(lambda: 1, max_attempts=0)
