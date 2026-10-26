"""Turn a run's event stream into spans, state change points and concurrency series.

Spans (per episode, from paired events):

* queued         cohort_start -> episode_admitted
* llm_wait       llm_wait_start -> request_start (or episode_stop if never granted)
* model_request  request_start -> request_end
* tool_wait      tool_start(run_tests) -> tool_slot_acquired
* tool           tool_slot_acquired -> tool_end (whole tool_start..tool_end for older runs)
* file_op        tool_start(read/edit/write) -> tool_end
* grading_wait   grading_start -> grading_slot_acquired
* grading        grading_slot_acquired -> episode_terminal

Gaps while an episode is admitted are harness overhead (state ``idle``). After
grading, the episode's state is its outcome category.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contract import CONCURRENCY_SERIES, STATE_CODES

EPSILON = 1e-6


@dataclass(frozen=True)
class Span:
    episode_id: str
    kind: str
    start: float  # run-relative monotonic seconds
    end: float
    request_id: str | None = None
    detail: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


def build_spans(events: list[dict]) -> list[Span]:
    spans: list[Span] = []
    open_: dict[tuple[str, str], tuple[float, str | None, str | None]] = {}
    cohort_start = next((e["t"] for e in events if e["event_type"] == "cohort_start"), None)
    # Runs recorded before slot events existed: a whole run_tests call / grading counts as busy time.
    kinds_seen = {e["event_type"] for e in events}
    tool_slot_events = "tool_slot_acquired" in kinds_seen
    grading_slot_events = "grading_slot_acquired" in kinds_seen

    def begin(eid: str, kind: str, t: float, request_id=None, detail=None) -> None:
        open_[(eid, kind)] = (t, request_id, detail)

    def end(eid: str, kind: str, t: float) -> None:
        started = open_.pop((eid, kind), None)
        if started is not None:
            spans.append(Span(eid, kind, started[0], max(t, started[0]), started[1], started[2]))

    for e in events:
        eid, kind, t = e["episode_id"], e["event_type"], e["t"]
        data = e.get("data") or {}
        if eid is None:
            continue
        if kind == "episode_admitted" and cohort_start is not None:
            spans.append(Span(eid, "queued", cohort_start, max(t, cohort_start)))
        elif kind == "llm_wait_start":
            begin(eid, "llm_wait", t, e["request_id"])
        elif kind == "request_start":
            end(eid, "llm_wait", t)
            begin(eid, "model_request", t, e["request_id"])
        elif kind == "request_end":
            end(eid, "model_request", t)
        elif kind == "tool_start":
            action = data.get("action")
            if action == "run_tests":
                begin(eid, "tool_wait" if tool_slot_events else "tool", t, detail=action)
            else:
                begin(eid, "file_op", t, detail=action)
        elif kind == "tool_slot_acquired":
            end(eid, "tool_wait", t)
            begin(eid, "tool", t, detail="run_tests")
        elif kind == "tool_end":
            if data.get("action") == "run_tests":
                end(eid, "tool_wait", t)  # still waiting: the wall-time backstop expired before a slot
                end(eid, "tool", t)
            else:
                end(eid, "file_op", t)
        elif kind == "episode_stop":
            for open_kind in ("llm_wait", "model_request", "tool_wait", "tool", "file_op"):
                end(eid, open_kind, t)
        elif kind == "grading_start":
            begin(eid, "grading_wait" if grading_slot_events else "grading", t)
        elif kind == "grading_slot_acquired":
            end(eid, "grading_wait", t)
            begin(eid, "grading", t)
        elif kind == "episode_terminal":
            end(eid, "grading_wait", t)  # grading never got a slot
            end(eid, "grading", t)
    spans.sort(key=lambda s: (s.episode_id, s.start, s.end))
    return spans


def episode_states(events: list[dict], spans: list[Span]) -> list[tuple[str, float, int]]:
    """State change points ``(episode_id, t, code)`` for Grafana's state timeline."""
    by_episode: dict[str, list[Span]] = {}
    for span in spans:
        by_episode.setdefault(span.episode_id, []).append(span)
    terminal = {e["episode_id"]: (e["t"], e["status"]) for e in events if e["event_type"] == "episode_terminal"}
    points: list[tuple[str, float, int]] = []
    for eid, items in by_episode.items():
        items.sort(key=lambda s: (s.start, s.end))
        for i, span in enumerate(items):
            points.append((eid, span.start, STATE_CODES[span.kind]))
            following = items[i + 1].start if i + 1 < len(items) else None
            if following is None or following - span.end > EPSILON:
                points.append((eid, span.end, STATE_CODES["idle"]))
        if eid in terminal and terminal[eid][1] in STATE_CODES:
            points.append((eid, terminal[eid][0], STATE_CODES[terminal[eid][1]]))
    # Later points win at equal times (a span start replaces the previous idle point).
    deduped: dict[tuple[str, float], int] = {}
    for eid, t, code in points:
        deduped[(eid, round(t, 6))] = code
    return sorted(((eid, t, code) for (eid, t), code in deduped.items()), key=lambda p: (p[0], p[1]))


def concurrency(spans: list[Span]) -> dict[str, list[tuple[float, int]]]:
    """Step series: value after each change. Built from spans so it matches the timeline."""
    members = {
        "llm_inflight": {"model_request"},
        "llm_waiting": {"llm_wait"},
        "tool_busy": {"tool", "grading"},
        "tool_waiting": {"tool_wait", "grading_wait"},
    }
    deltas: dict[str, list[tuple[float, int]]] = {name: [] for name in CONCURRENCY_SERIES}
    active: dict[str, list[float]] = {}
    for span in spans:
        for name, kinds in members.items():
            if span.kind in kinds:
                deltas[name] += [(span.start, 1), (span.end, -1)]
        if span.kind != "queued" and not span.kind.startswith("grading"):
            bounds = active.setdefault(span.episode_id, [span.start, span.end])
            bounds[0], bounds[1] = min(bounds[0], span.start), max(bounds[1], span.end)
    for start, end in active.values():
        deltas["active_episodes"] += [(start, 1), (end, -1)]
    series: dict[str, list[tuple[float, int]]] = {}
    for name, items in deltas.items():
        items.sort(key=lambda d: (d[0], d[1]))  # ends before starts at equal times
        value, points = 0, []
        for t, delta in items:
            value += delta
            if points and abs(points[-1][0] - t) <= EPSILON:
                points[-1] = (points[-1][0], value)
            else:
                points.append((t, value))
        series[name] = points
    return series


def totals_by_kind(spans: list[Span]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for span in spans:
        out.setdefault(span.episode_id, {}).setdefault(span.kind, 0.0)
        out[span.episode_id][span.kind] += span.duration
    return out
