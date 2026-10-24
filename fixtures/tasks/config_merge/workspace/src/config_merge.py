"""Merging of default settings with user overrides."""


def merge_config(defaults, overrides):
    """Return a new dict combining ``defaults`` with ``overrides``.

    - A key present in ``overrides`` always wins, including falsy values such
      as ``False``, ``0``, ``""``, ``[]`` and ``None``.
    - When both values for a key are dicts, they are merged recursively by the
      same rules.
    - Keys missing from ``overrides`` keep their default value.
    - Neither input is modified.
    """
    result = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        elif value:
            result[key] = value
    return result
