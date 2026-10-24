"""Exponential backoff for calls to flaky services."""

import time


def retry_with_backoff(fn, max_attempts, base_delay=0.1, max_delay=2.0, sleep=time.sleep):
    """Call ``fn()`` and retry on any exception, sleeping between attempts.

    - At most ``max_attempts`` calls are made in total.
    - Before the n-th retry (n = 1, 2, ...) it sleeps
      ``min(base_delay * 2 ** (n - 1), max_delay)`` seconds, using ``sleep``.
    - It never sleeps after the final failed attempt.
    - When every attempt fails, the last exception is re-raised.
    - Raises ``ValueError`` without calling ``fn`` if ``max_attempts`` < 1.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    attempt = 0
    while True:
        try:
            return fn()
        except Exception:
            attempt += 1
            if attempt >= max_attempts:
                raise
            sleep(min(base_delay * 2 ** (attempt - 1), max_delay))
