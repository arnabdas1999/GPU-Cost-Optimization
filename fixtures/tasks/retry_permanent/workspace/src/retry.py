"""Retry helper used by the job runner."""


class TransientError(Exception):
    """An error that may succeed if the operation is attempted again."""


class PermanentError(Exception):
    """An error that will not succeed on retry."""


def call_with_retry(fn, max_attempts=3, is_transient=None):
    """Call ``fn()`` and retry it when it fails with a transient error.

    ``is_transient(exc)`` decides whether an exception is worth retrying; by
    default only ``TransientError`` is. Any other exception propagates
    immediately without further attempts. At most ``max_attempts`` calls are
    made in total; when they are used up, the last exception is raised.
    Raises ``ValueError`` if ``max_attempts`` is less than 1.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if is_transient is None:
        is_transient = lambda exc: isinstance(exc, TransientError)  # noqa: E731

    last_exc = None
    for _ in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
    raise last_exc
