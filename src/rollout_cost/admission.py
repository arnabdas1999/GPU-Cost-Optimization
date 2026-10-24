"""Admission control primitives.

Episode admission is a fixed set of FIFO workers in the runner (one worker per
``max_active_episodes``). Model requests are bounded separately by a
``FifoSemaphore`` sized by ``max_inflight_llm_requests``: an episode holds a
request slot only while its model call is outstanding, so episodes running
tools release GPU access to others.
"""

from __future__ import annotations

import asyncio
from collections import deque


class FifoSemaphore:
    """A counting semaphore that grants waiters strictly in arrival order.

    Cancellation-safe: a waiter cancelled after being granted passes the slot on.
    """

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.in_use = 0
        self.max_in_use = 0
        self._waiters: deque[asyncio.Future[None]] = deque()

    @property
    def waiting(self) -> int:
        return sum(1 for f in self._waiters if not f.done())

    async def acquire(self) -> None:
        if self.in_use < self.capacity and not self._waiters:
            self._grant()
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.append(future)
        try:
            await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                # Granted, then cancelled before we could use it: hand it on.
                self.release()
            else:
                self._waiters.remove(future)
            raise

    def _grant(self) -> None:
        self.in_use += 1
        self.max_in_use = max(self.max_in_use, self.in_use)

    def release(self) -> None:
        if self.in_use <= 0:
            raise RuntimeError("release() without acquire()")
        self.in_use -= 1
        while self._waiters:
            future = self._waiters.popleft()
            if not future.done():
                self._grant()
                future.set_result(None)
                return
