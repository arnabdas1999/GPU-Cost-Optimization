"""One agent episode: prompts, model calls, action execution and budgets.

Budgets and their boundaries are identical for every configuration:

* Step budget: at most ``max_steps`` model replies, malformed ones included.
  A malformed or truncated reply consumes a step and produces feedback.
* Context budget: before each request the prompt size is the server-reported
  usage of the previous request plus a conservative estimate for the text
  appended since. If it would exceed ``context_window_tokens -
  max_output_tokens_per_call`` the episode stops with ``context_exhausted``.
  History is never compacted or rewritten. Tool output is truncated to
  ``max_observation_chars`` by a fixed head/tail rule.
* Wall-time backstop: ``episode_wall_timeout_s`` runs from admission (episode
  slot acquired) until the agent loop ends. It includes waiting for a model
  request slot, model requests, waiting for a tool slot and tool execution.
  It excludes FIFO queueing before admission and grading. On expiry the
  outstanding request is cancelled (closing the stream aborts it in vLLM) or
  the tool process is killed, and the stop reason is ``wall_timeout``.
* Per-call limits: a model request is bounded by ``request_timeout_s``; its
  expiry is an infrastructure error (``model_timeout``). A test run is
  bounded by ``test_timeout_s``; its expiry is feedback to the model, because
  generated code can hang. The smaller of the call limit and the remaining
  wall time applies, and the result records which one fired.
"""

from __future__ import annotations

import asyncio
import math
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .admission import FifoSemaphore
from .config import Config
from .fixtures import EpisodeSpec
from .model_client import ModelError, RequestContext
from .protocol import Action, ProtocolError, parse_action, system_prompt
from .telemetry import EventRecorder
from .tools import ExecutorError, ToolError, ToolPool, edit_file, list_files, read_file, truncate, write_file

MESSAGE_OVERHEAD_TOKENS = 8
TEST_OUTPUT_HEAD_FRACTION = 0.2  # keep mostly the tail, where pytest prints its summary


@dataclass
class EpisodeEnv:
    run_id: str
    cfg: Config
    client: Any
    llm_sem: FifoSemaphore
    tool_pool: ToolPool
    executor: Any
    recorder: EventRecorder


@dataclass
class EpisodeResult:
    stop_reason: str = "infra_error"
    stop_detail: str | None = None
    infrastructure_status: str = "ok"
    infrastructure_detail: str | None = None
    steps: int = 0
    actions: dict[str, int] = field(default_factory=dict)
    protocol_errors: dict[str, int] = field(default_factory=dict)
    tool_errors: int = 0
    test_timeouts: int = 0
    observation_truncations: int = 0
    prompt_tokens_known: int = 0
    completion_tokens_known: int = 0
    usage_missing: int = 0
    llm_wait_s: float = 0.0
    request_s: float = 0.0
    ttft_s: list[float] = field(default_factory=list)
    tool_wait_s: float = 0.0
    tool_s: float = 0.0
    file_op_s: float = 0.0
    cancelled_requests: int = 0
    requests: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        data = asdict(self)
        data.pop("requests")
        data.pop("messages")
        complete = self.usage_missing == 0
        # Missing usage is unknown, never zero.
        data["prompt_tokens"] = self.prompt_tokens_known if complete else None
        data["completion_tokens"] = self.completion_tokens_known if complete else None
        data["protocol_error_count"] = sum(self.protocol_errors.values())
        return data


class _WallTimeout(Exception):
    pass


class _CallTimeout(Exception):
    pass


class _Stop(Exception):
    def __init__(self, reason: str, detail: str | None = None, infra: str = "ok", infra_detail: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail
        self.infra = infra
        self.infra_detail = infra_detail


class Episode:
    def __init__(self, env: EpisodeEnv, spec: EpisodeSpec, workspace: Path):
        self.env = env
        self.spec = spec
        self.workspace = workspace
        self.cfg = env.cfg
        self.result = EpisodeResult()
        self.messages: list[dict] = []
        self.deadline = 0.0
        w = self.cfg.workload
        self.max_steps = w.max_steps
        self.max_output = self.cfg.generation.max_output_tokens_per_call
        self.input_budget = w.context_window_tokens - self.max_output
        self._last_usage: tuple[int, int] | None = None
        self._last_sent = 0

    # -- helpers -----------------------------------------------------------

    def emit(self, event_type: str, **kwargs: Any) -> None:
        self.env.recorder.emit(event_type, self.spec.episode_id, **kwargs)

    def _estimate(self, message: dict) -> int:
        return math.ceil(len(message["content"]) / self.cfg.workload.chars_per_token_estimate) + MESSAGE_OVERHEAD_TOKENS

    def estimate_prompt_tokens(self) -> int:
        if self._last_usage is None:
            return sum(self._estimate(m) for m in self.messages)
        prompt, completion = self._last_usage
        appended = self.messages[self._last_sent + 1 :]  # everything after the counted assistant reply
        return prompt + completion + MESSAGE_OVERHEAD_TOKENS + sum(self._estimate(m) for m in appended)

    async def _bounded(self, awaitable, limit_s: float | None):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            if asyncio.iscoroutine(awaitable):
                awaitable.close()
            raise _WallTimeout()
        if limit_s is not None and limit_s < remaining:
            try:
                return await asyncio.wait_for(awaitable, limit_s)
            except TimeoutError:
                raise _CallTimeout() from None
        try:
            return await asyncio.wait_for(awaitable, remaining)
        except TimeoutError:
            raise _WallTimeout() from None

    def _feedback(self, body: str) -> None:
        left = self.max_steps - self.result.steps
        note = f"\n\n[{self.result.steps} of {self.max_steps} replies used; {left} remaining.]"
        if left == 1:
            note += " Your next reply is the last one; the repository is graded as it stands after it."
        self.messages.append({"role": "user", "content": body + note})

    def _initial_message(self) -> str:
        task = self.spec.task
        max_chars = self.cfg.workload.max_observation_chars
        parts = [f"Task: {task.title}", "", task.description, "", "Repository files:"]
        parts += [f"- {f}" for f in list_files(self.workspace)]
        for rel in task.initial_files:
            text, truncated = read_file(self.workspace, rel, max_chars)
            self.result.observation_truncations += int(truncated)
            parts += ["", f"=== {rel} ===", text]
        parts += ["", "Reply with one JSON action."]
        return "\n".join(parts)

    # -- model calls -------------------------------------------------------

    async def _model_call(self, estimate: int):
        r = self.result
        request_id = f"{self.spec.episode_id}-s{r.steps}"
        ready = time.monotonic()
        self.emit("llm_wait_start", request_id=request_id)
        try:
            await self._bounded(self.env.llm_sem.acquire(), None)
        except _WallTimeout:
            r.llm_wait_s += time.monotonic() - ready
            raise _Stop("wall_timeout", "expired while waiting for a model-request slot") from None
        waited = time.monotonic() - ready
        r.llm_wait_s += waited
        sent = list(self.messages)
        attempts: list[dict] = []
        record: dict[str, Any] = {"request_id": request_id, "step": r.steps, "message_count": len(sent),
                                  "estimated_prompt_tokens": estimate, "wait_s": waited}
        ctx = RequestContext(self.env.run_id, self.spec.episode_id, self.spec.task.task_id, request_id, r.steps, self.spec.seed)
        started = time.monotonic()
        self.emit("request_start", request_id=request_id, wait_s=round(waited, 6), estimated_prompt_tokens=estimate)
        status = "ok"
        try:
            response = await self._bounded(
                self.env.client.complete(sent, self.max_output, ctx, attempts), self.cfg.model.request_timeout_s
            )
        except _WallTimeout:
            status = "cancelled_wall_timeout"
            r.cancelled_requests += 1
            raise _Stop("wall_timeout", "expired during a model request; request cancelled") from None
        except _CallTimeout:
            status = "cancelled_request_timeout"
            r.cancelled_requests += 1
            raise _Stop("infra_error", "model request timed out", "model_timeout",
                        f"no response within {self.cfg.model.request_timeout_s}s") from None
        except ModelError as exc:
            status = f"error_{exc.kind}"
            if exc.kind == "context_overflow":
                raise _Stop("context_exhausted", "server rejected the prompt as too long") from None
            raise _Stop("infra_error", f"model error: {exc.kind}", "model_server_error", str(exc)[:500]) from None
        finally:
            self.env.llm_sem.release()
            duration = time.monotonic() - started
            r.request_s += duration
            record.update(status=status, duration_s=duration, attempts=attempts)
            r.requests.append(record)
            if status != "ok":
                self.emit("request_end", request_id=request_id, status=status, duration_s=round(duration, 6))
        record.update(finish_reason=response.finish_reason, ttft_s=response.ttft_s,
                      prompt_tokens=response.prompt_tokens, completion_tokens=response.completion_tokens)
        self.emit("request_end", request_id=request_id, status="ok", duration_s=round(duration, 6),
                  ttft_s=response.ttft_s, prompt_tokens=response.prompt_tokens,
                  completion_tokens=response.completion_tokens, finish_reason=response.finish_reason)
        if response.ttft_s is not None:
            r.ttft_s.append(response.ttft_s)
        self._last_sent = len(sent)
        if response.prompt_tokens is not None and response.completion_tokens is not None:
            self._last_usage = (response.prompt_tokens, response.completion_tokens)
            r.prompt_tokens_known += response.prompt_tokens
            r.completion_tokens_known += response.completion_tokens
        else:
            self._last_usage = None
            r.usage_missing += 1
        return response

    # -- tools -------------------------------------------------------------

    async def _run_tests(self) -> str:
        r = self.result
        waiting = time.monotonic()
        try:
            slot = await self._bounded(self.env.tool_pool.acquire(), None)
        except _WallTimeout:
            r.tool_wait_s += time.monotonic() - waiting
            raise _Stop("wall_timeout", "expired while waiting for a tool slot") from None
        r.tool_wait_s += time.monotonic() - waiting
        self.emit("tool_slot_acquired", slot=slot.index, cpu=slot.cpu, wait_s=round(time.monotonic() - waiting, 6))
        started = time.monotonic()
        test_timeout = self.cfg.tools.test_timeout_s
        remaining = self.deadline - started
        limit = min(test_timeout, remaining)
        try:
            if limit <= 0:
                raise _Stop("wall_timeout", "expired before tests could start")
            label = f"{self.spec.episode_id}-tests"
            result = await self.env.executor.run(self.workspace, self.spec.task.test_command, limit, slot, label)
        except ExecutorError as exc:
            raise _Stop("infra_error", "tool executor failed", "tool_executor_error", str(exc)[:500]) from None
        finally:
            self.env.tool_pool.release(slot)
            r.tool_s += time.monotonic() - started
        if result.timed_out:
            if remaining <= test_timeout:
                raise _Stop("wall_timeout", "expired during run_tests; process killed")
            r.test_timeouts += 1
            return f"run_tests timed out after {test_timeout:.0f}s and the process was killed."
        output, truncated = truncate(result.output, self.cfg.workload.max_observation_chars, TEST_OUTPUT_HEAD_FRACTION)
        r.observation_truncations += int(truncated)
        return f"run_tests finished with exit code {result.exit_code}:\n```\n{output}\n```"

    async def _execute(self, action: Action) -> None:
        r = self.result
        path = action.args.get("path")
        self.emit("tool_start", action=action.name, path=path)
        started = time.monotonic()
        status = "ok"
        try:
            if action.name == "read_file":
                text, truncated = read_file(self.workspace, path, self.cfg.workload.max_observation_chars)
                r.observation_truncations += int(truncated)
                body = f"Contents of {path}:\n```\n{text}\n```"
            elif action.name == "write_file":
                written = write_file(self.workspace, path, action.args["content"])
                body = f"Wrote {written} bytes to {path}."
            elif action.name == "edit_file":
                edit_file(self.workspace, path, action.args["old"], action.args["new"])
                body = f"Edited {path}."
            elif action.name == "run_tests":
                body = await self._run_tests()
            else:  # pragma: no cover - parse_action rejects unknown actions
                raise ToolError(f"unsupported action {action.name}")
        except ToolError as exc:
            r.tool_errors += 1
            status = "tool_error"
            body = f"Error: {exc}"
        finally:
            if action.name != "run_tests":
                r.file_op_s += time.monotonic() - started
            self.emit("tool_end", action=action.name, status=status, duration_s=round(time.monotonic() - started, 6))
        self._feedback(body)

    # -- main loop ---------------------------------------------------------

    async def run(self) -> EpisodeResult:
        r = self.result
        self.deadline = time.monotonic() + self.cfg.workload.episode_wall_timeout_s
        actions: Counter[str] = Counter()
        protocol_errors: Counter[str] = Counter()
        self.emit("episode_start", task_id=self.spec.task.task_id, rep=self.spec.rep, seed=self.spec.seed)
        try:
            self.messages = [
                {"role": "system", "content": system_prompt(self.max_steps, self.max_output)},
                {"role": "user", "content": self._initial_message()},
            ]
            while True:
                if r.steps >= self.max_steps:
                    raise _Stop("max_steps")
                estimate = self.estimate_prompt_tokens()
                if estimate > self.input_budget:
                    raise _Stop("context_exhausted", f"estimated prompt {estimate} tokens > input budget {self.input_budget}")
                response = await self._model_call(estimate)
                r.steps += 1
                self.messages.append({"role": "assistant", "content": response.content})
                try:
                    action = parse_action(response.content, response.finish_reason, self.cfg.workload.max_write_bytes)
                except ProtocolError as exc:
                    protocol_errors[exc.kind] += 1
                    self.emit("protocol_error", kind=exc.kind)
                    self._feedback(f"Protocol error: {exc}")
                    continue
                actions[action.name] += 1
                if action.name == "finish":
                    raise _Stop("finished")
                await self._execute(action)
        except _Stop as stop:
            r.stop_reason, r.stop_detail = stop.reason, stop.detail
            r.infrastructure_status, r.infrastructure_detail = stop.infra, stop.infra_detail
        except Exception as exc:  # noqa: BLE001 - a harness bug must not look like a model failure
            r.stop_reason, r.stop_detail = "infra_error", "harness exception"
            r.infrastructure_status = "harness_error"
            r.infrastructure_detail = "".join(traceback.format_exception_only(exc)).strip()[:1000]
        r.actions = dict(actions)
        r.protocol_errors = dict(protocol_errors)
        r.messages = self.messages
        self.emit("episode_stop", status=r.stop_reason, detail=r.stop_detail, infrastructure=r.infrastructure_status)
        return r


async def run_episode(env: EpisodeEnv, spec: EpisodeSpec, workspace: Path) -> EpisodeResult:
    return await Episode(env, spec, workspace).run()
