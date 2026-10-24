import pytest

from backoff import retry_with_backoff


class Boom(Exception):
    pass


def test_makes_exactly_max_attempts_calls():
    calls = []

    def always_fails():
        calls.append(1)
        raise Boom()

    with pytest.raises(Boom):
        retry_with_backoff(always_fails, max_attempts=3, sleep=lambda seconds: None)
    assert len(calls) == 3


def test_returns_value_on_success():
    assert retry_with_backoff(lambda: 42, max_attempts=3, sleep=lambda seconds: None) == 42
