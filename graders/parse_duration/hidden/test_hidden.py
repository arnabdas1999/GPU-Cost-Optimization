import pytest

from duration import parse_duration


@pytest.mark.parametrize(
    "text, seconds",
    [
        ("45s", 45),
        ("2m", 120),
        ("1h", 3600),
        ("2h5s", 7205),
        ("1h2m3s", 3723),
        ("0s", 0),
        ("90m", 5400),
        ("007s", 7),
    ],
)
def test_valid_durations(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize(
    "text",
    ["", "10", "h", "1x", "abc", "1m1h", "1h1h", "1h 30m", " 1h", "-5s", "+5s", "1.5h", "5s\n", "1H", "m5", "1h30"],
)
def test_invalid_durations(text):
    with pytest.raises(ValueError):
        parse_duration(text)
