import pytest

from ports import parse_port_range


def test_range():
    assert parse_port_range("8000-8010") == (8000, 8010)


def test_reversed_range_is_rejected():
    with pytest.raises(ValueError):
        parse_port_range("9000-8000")
