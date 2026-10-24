"""Parsing of human-readable durations."""

import re

_UNITS = {"h": 3600, "m": 60, "s": 1}


def parse_duration(text):
    """Parse a duration such as "1h30m", "45s" or "2h5s" into whole seconds.

    Rules:
    - The text is one or more ``<digits><unit>`` parts, where digits are ASCII
      0-9 and the unit is ``h``, ``m`` or ``s`` (lowercase).
    - Each unit appears at most once, in the order h, m, s.
    - Nothing else is allowed: no spaces, signs, decimals or trailing text.
    - "0s" is valid and returns 0.

    Anything else raises ``ValueError``.
    """
    total = 0
    for number, unit in re.findall(r"(\d+)([hms])", text):
        total += int(number) * _UNITS[unit]
    return total
