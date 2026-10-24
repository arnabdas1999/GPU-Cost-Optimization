"""Parsing of human-readable durations."""

import re

_PATTERN = re.compile(r"(?:([0-9]+)h)?(?:([0-9]+)m)?(?:([0-9]+)s)?")


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
    match = _PATTERN.fullmatch(text)
    if not text or match is None:
        raise ValueError(f"invalid duration: {text!r}")
    hours, minutes, seconds = (int(part) if part else 0 for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds
