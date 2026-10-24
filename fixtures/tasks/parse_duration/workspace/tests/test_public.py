import pytest

from duration import parse_duration


def test_hours_and_minutes():
    assert parse_duration("1h30m") == 5400


def test_missing_unit_is_rejected():
    with pytest.raises(ValueError):
        parse_duration("1h30")
