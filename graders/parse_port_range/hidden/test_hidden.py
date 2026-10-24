import pytest

from ports import parse_port_range


@pytest.mark.parametrize(
    "text, expected",
    [
        ("80", (80, 80)),
        ("1", (1, 1)),
        ("65535", (65535, 65535)),
        ("1-65535", (1, 65535)),
        ("8080-8080", (8080, 8080)),
        ("8000-8010", (8000, 8010)),
    ],
)
def test_valid_specs(text, expected):
    assert parse_port_range(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "0",
        "65536",
        "0-10",
        "10-65536",
        "9000-8000",
        "",
        "-",
        "80-",
        "-80",
        " 80",
        "80 ",
        "80 - 90",
        "+80",
        "8_0",
        "80-90-100",
        "abc",
        "٨٠",
    ],
)
def test_invalid_specs(text):
    with pytest.raises(ValueError):
        parse_port_range(text)
