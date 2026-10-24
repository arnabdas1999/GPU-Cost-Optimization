"""Environment-variable overrides for service settings."""


def apply_env_overrides(config, environ, prefix="APP_"):
    """Return a copy of ``config`` with values overridden from ``environ``.

    For each top-level key in ``config``, the variable ``prefix + key.upper()``
    overrides it. The string value is converted to the type of the default:

    - bool: "1", "true", "yes", "on" give True and "0", "false", "no", "off"
      give False (case-insensitive, surrounding whitespace ignored); any other
      value raises ``ValueError``. The result is always a ``bool``.
    - int: parsed with ``int()``; invalid values raise ``ValueError``.
    - float: parsed with ``float()``; invalid values raise ``ValueError``.
    - anything else: the string is used as-is.

    Variables without a matching config key are ignored. ``config`` is not
    modified.
    """
    result = dict(config)
    for key, default in config.items():
        name = prefix + key.upper()
        if name not in environ:
            continue
        raw = environ[name]
        if isinstance(default, int):
            result[key] = int(raw)
        elif isinstance(default, bool):
            result[key] = bool(raw)
        elif isinstance(default, float):
            result[key] = float(raw)
        else:
            result[key] = raw
    return result
