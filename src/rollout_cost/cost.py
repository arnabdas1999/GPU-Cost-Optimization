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


def cost_summary(window_s: float, run_s: float, valid: int, successes: int, config: dict) -> dict:
    """Price measured allocation intervals under an explicitly recorded assumption."""
    rate, currency = config["allocation_hourly_rate"], config["currency"]
    window = allocation_cost("steady-state window", window_s, rate, currency)
    inclusive = allocation_cost("run-inclusive (reset, warm-up, cohort, drain)", run_s, rate, currency,
                                config.get("billing_granularity_seconds"))
    return {"window": window.to_dict(), "run_inclusive": inclusive.to_dict(),
            "per_valid_episode": per_unit(window.cost, valid), "per_success": per_unit(window.cost, successes),
            "rate_source": config.get("rate_source"), "rate_kind": config.get("rate_kind", "recorded")}


ESTIMATED_COST_PREFIX = "Estimated equivalent cost:"
UNKNOWN_COST_WARNING = "Allocation price unknown: costs are unavailable (timings are still valid)."


def cost_notice(cost: dict) -> str | None:
    if cost["window"]["cost"] is None:
        return UNKNOWN_COST_WARNING
    if cost.get("rate_kind") == "estimated_equivalent":
        window = cost["window"]
        return (f"{ESTIMATED_COST_PREFIX} {window['hourly_rate']:.6f} {window['currency']}/hour; "
                f"{cost.get('rate_source') or 'source not recorded'} This is a reference valuation, not a Colab bill.")
    return None
