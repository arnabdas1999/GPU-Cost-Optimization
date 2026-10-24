"""Parsing of firewall port specifications."""

import re

_DIGITS = re.compile(r"[0-9]+")


def _parse_port(text):
    if not _DIGITS.fullmatch(text):
        raise ValueError(f"invalid port: {text!r}")
    port = int(text)
    if not 1 <= port <= 65535:
        raise ValueError(f"port out of range: {port}")
    return port


def parse_port_range(text):
    """Parse "PORT" or "START-END" into a ``(start, end)`` tuple of ints.

    - Each port is written with ASCII digits 0-9 only: no signs, spaces,
      underscores or other characters anywhere in the text.
    - Ports must be in the range 1..65535.
    - A single port "P" returns ``(P, P)``.
    - In a range, start must be less than or equal to end.

    Invalid input raises ``ValueError``.
    """
    start_text, separator, end_text = text.partition("-")
    if not separator:
        end_text = start_text
    start, end = _parse_port(start_text), _parse_port(end_text)
    if start > end:
        raise ValueError(f"reversed range: {text!r}")
    return start, end
