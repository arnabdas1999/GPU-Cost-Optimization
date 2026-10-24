"""Allocated-GPU cost accounting.

Cost is allocation time multiplied by the hourly price of the allocation. It is
never derived by summing overlapping episode durations or by scaling the price
by utilization. An unknown price yields an unknown cost, never zero.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CostLine:
    label: str
    seconds: float
    billed_seconds: float
    hourly_rate: float | None
    currency: str
    granularity_seconds: int | None
    cost: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def billed_seconds(seconds: float, granularity_seconds: int | None) -> float:
    if seconds < 0:
        raise ValueError("seconds must be non-negative")
    if not granularity_seconds:
        return seconds
    return math.ceil(seconds / granularity_seconds) * granularity_seconds


def allocation_cost(label: str, seconds: float, hourly_rate: float | None, currency: str,
                    granularity_seconds: int | None = None) -> CostLine:
    billed = billed_seconds(seconds, granularity_seconds)
    cost = None if hourly_rate is None else billed / 3600.0 * hourly_rate
    return CostLine(label, seconds, billed, hourly_rate, currency, granularity_seconds, cost)


def per_unit(cost: float | None, count: int) -> float | None:
    """Cost per episode/success; undefined (None) when cost is unknown or count is zero."""
    if cost is None or count <= 0:
        return None
    return cost / count
