import pytest

from backoff import retry_with_backoff


class Boom(Exception):
    pass


def run(outcomes, **kwargs):
    calls, sleeps = [], []

    def fn():
        item = outcomes[len(calls)]
        calls.append(item)
        if isinstance(item, BaseException):
            raise item
        return item

    try:
        result = retry_with_backoff(fn, sleep=sleeps.append, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the tests inspect it
        return None, calls, sleeps, exc
    return result, calls, sleeps, None


def test_delays_double_from_base_delay():
    _, calls, sleeps, error = run([Boom()] * 4, max_attempts=4, base_delay=0.1, max_delay=10)
    assert isinstance(error, Boom)
    assert len(calls) == 4
    assert sleeps == pytest.approx([0.1, 0.2, 0.4])


def test_delay_is_capped():
    _, _, sleeps, _ = run([Boom()] * 5, max_attempts=5, base_delay=1, max_delay=3)
    assert sleeps == pytest.approx([1, 2, 3, 3])


def test_no_sleep_after_final_failure():
    _, calls, sleeps, _ = run([Boom()] * 2, max_attempts=2)
    assert len(calls) == 2
    assert len(sleeps) == 1


def test_success_on_second_attempt_uses_default_base_delay():
    result, calls, sleeps, error = run([Boom(), "ok"], max_attempts=5)
    assert error is None
    assert result == "ok"
    assert len(calls) == 2
    assert sleeps == pytest.approx([0.1])


def test_single_attempt_never_sleeps():
    _, calls, sleeps, error = run([Boom()], max_attempts=1)
    assert isinstance(error, Boom)
    assert len(calls) == 1
    assert sleeps == []


def test_last_exception_is_raised():
    _, _, _, error = run([Boom("1"), Boom("2"), Boom("3")], max_attempts=3)
    assert str(error) == "3"


def test_invalid_max_attempts_does_not_call():
    _, calls, _, error = run(["ok"], max_attempts=0)
    assert isinstance(error, ValueError)
    assert calls == []
