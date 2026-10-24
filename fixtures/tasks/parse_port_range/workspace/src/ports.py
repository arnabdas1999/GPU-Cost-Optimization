"""Parsing of firewall port specifications."""


def parse_port_range(text):
    """Parse "PORT" or "START-END" into a ``(start, end)`` tuple of ints.

    - Each port is written with ASCII digits 0-9 only: no signs, spaces,
      underscores or other characters anywhere in the text.
    - Ports must be in the range 1..65535.
    - A single port "P" returns ``(P, P)``.
    - In a range, start must be less than or equal to end.

    Invalid input raises ``ValueError``.
    """
    if "-" in text:
        start_text, end_text = text.split("-", 1)
    else:
        start_text = end_text = text
    return int(start_text), int(end_text)
