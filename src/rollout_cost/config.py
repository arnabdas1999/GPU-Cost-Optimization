"""Configuration files: loading, ``extends`` inheritance, validation and fingerprints."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Keys a sweep may change between baseline and candidate. Every other setting is
# protected: it must be identical for a baseline/candidate comparison to be valid.
TUNABLE_KEYS = frozenset(
    {
        "execution.max_active_episodes",
        "execution.max_inflight_llm_requests",
        "server.args.max-num-seqs",
        "server.args.max-num-batched-tokens",
        "server.args.enable-prefix-caching",
    }
)
# Descriptive keys that never affect behavior.
DESCRIPTIVE_KEYS = frozenset({"name", "description"})


class ConfigError(ValueError):
    pass


@dataclass
class ModelConfig:
    # "openai": an OpenAI-compatible server such as vLLM.
    # "scripted": a deterministic fake for exercising the harness without a GPU.
    kind: str = "openai"
    base_url: str = "http://127.0.0.1:8000/v1"
    served_name: str = "coding-model"
    repository: str | None = None
    revision: str | None = None
    tokenizer_revision: str | None = None
    api_key_env: str | None = None
    request_timeout_s: float = 120.0
    connect_timeout_s: float = 10.0
    # Retries apply only to connection failures and HTTP 502/503/504 that occur
    # before any content was streamed. Each attempt is recorded.
    max_request_retries: int = 1
    retry_backoff_s: float = 1.0
    scripted_policy: str = "reference"
    scripted_base_latency_s: float = 0.2
    scripted_per_token_latency_s: float = 0.002
    scripted_contention: float = 0.3


@dataclass
class GenerationConfig:
    temperature: float = 0.2
    top_p: float = 0.95
    max_output_tokens_per_call: int = 512
    seed: int = 42


@dataclass
class WorkloadConfig:
    fixtures_dir: str = "fixtures/tasks"
    graders_dir: str = "graders"
    splits_file: str = "fixtures/splits.yaml"
    # Independent attempts per task. The cohort is tasks x repetitions, so a
    # concurrency sweep has a backlog larger than the episode cap.
    repetitions: int = 1
    max_steps: int = 6
    # Must equal the server's max-model-len. Prompt + max output must fit.
    context_window_tokens: int = 8192
    # Conservative characters-per-token estimate, used only for text whose
    # exact token count is not yet known from server usage.
    chars_per_token_estimate: float = 3.0
    max_observation_chars: int = 6000
    max_write_bytes: int = 65536
    # Safety backstop measured from admission to the end of the agent loop.
    # Step and token budgets are the primary limits; episodes that hit this
    # are reported separately because the limit is scheduling-sensitive.
    episode_wall_timeout_s: float = 300.0


@dataclass
class ExecutionConfig:
    max_active_episodes: int = 4
    max_inflight_llm_requests: int = 4


@dataclass
class ToolsConfig:
    executor: str = "docker"  # docker | local (local = no isolation, development only)
    image: str = "rollout-cost-tools:py312"
    # One pinned CPU per tool worker. The aggregate tool CPU budget is fixed by
    # this list, independent of how many episodes are active.
    worker_cpus: list[int] = field(default_factory=list)
    # Used only when worker_cpus is empty (no pinning).
    workers: int = 2
    memory_mb: int = 512
    pids_limit: int = 128
    test_timeout_s: float = 60.0
    grader_timeout_s: float = 120.0


@dataclass
class ServerConfig:
    # Cache procedure before each measured run:
    #   restart  - stop and start the server with start_command (cleanest)
    #   endpoint - POST /reset_prefix_cache (needs VLLM_SERVER_DEV_MODE=1)
    #   none     - no reset; the run is labelled exploratory
    cache_reset: str = "restart"
    # "{args}" in the command is replaced by the rendered ``args`` mapping, so
    # the recorded arguments and the launched arguments cannot drift apart.
    start_command: list[str] | None = None
    stop_command: list[str] | None = None
    stop_after_run: bool = True
    ready_timeout_s: float = 900.0
    args: dict[str, Any] = field(default_factory=dict)
    warmup_file: str | None = "fixtures/warmup.jsonl"
    # CPUs reserved for the server and the runner; must not overlap worker_cpus.
    reserved_cpus: list[int] = field(default_factory=list)
    drain_timeout_s: float = 60.0


@dataclass
class TelemetryConfig:
    gpu_sample_interval_s: float = 1.0
    server_metrics_interval_s: float = 2.0
    nvidia_smi: str = "nvidia-smi"


@dataclass
class CostConfig:
    # Unknown price produces timing results with cost marked unavailable, never $0.
    allocation_hourly_rate: float | None = None
    currency: str = "USD"
    rate_source: str | None = None
    billing_granularity_seconds: int | None = None


_SECTIONS: dict[str, type] = {
    "model": ModelConfig,
    "generation": GenerationConfig,
    "workload": WorkloadConfig,
    "execution": ExecutionConfig,
    "tools": ToolsConfig,
    "server": ServerConfig,
    "telemetry": TelemetryConfig,
    "cost": CostConfig,
}


@dataclass
class Config:
    name: str = "unnamed"
    description: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    # Base directory for relative paths; not part of the serialized config.
    root: Path = field(default_factory=Path.cwd, compare=False)

    def path(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else self.root / p

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data.pop("root")
        return data

    @property
    def synthetic(self) -> bool:
        return self.model.kind == "scripted"

    def validate(self) -> list[str]:
        """Raise on invalid settings; return non-fatal warnings."""
        errors: list[str] = []
        warnings: list[str] = []
        if self.model.kind not in ("openai", "scripted"):
            errors.append(f"model.kind must be 'openai' or 'scripted', not {self.model.kind!r}")
        if self.tools.executor not in ("docker", "local"):
            errors.append(f"tools.executor must be 'docker' or 'local', not {self.tools.executor!r}")
        if self.server.cache_reset not in ("restart", "endpoint", "none"):
            errors.append("server.cache_reset must be 'restart', 'endpoint' or 'none'")
        for name in ("max_active_episodes", "max_inflight_llm_requests"):
            if getattr(self.execution, name) < 1:
                errors.append(f"execution.{name} must be >= 1")
        w = self.workload
        if w.repetitions < 1 or w.max_steps < 1:
            errors.append("workload.repetitions and workload.max_steps must be >= 1")
        if self.generation.max_output_tokens_per_call >= w.context_window_tokens:
            errors.append("generation.max_output_tokens_per_call must be smaller than workload.context_window_tokens")
        if w.chars_per_token_estimate <= 0:
            errors.append("workload.chars_per_token_estimate must be positive")
        if not self.tools.worker_cpus and self.tools.workers < 1:
            errors.append("tools.workers must be >= 1 when tools.worker_cpus is empty")
        if len(set(self.tools.worker_cpus)) != len(self.tools.worker_cpus):
            errors.append("tools.worker_cpus contains duplicates")
        overlap = set(self.tools.worker_cpus) & set(self.server.reserved_cpus)
        if overlap:
            errors.append(f"tools.worker_cpus overlaps server.reserved_cpus: {sorted(overlap)}")
        rate = self.cost.allocation_hourly_rate
        if rate is not None and rate < 0:
            errors.append("cost.allocation_hourly_rate must be non-negative")
        if errors:
            raise ConfigError("; ".join(errors))

        if not self.synthetic:
            if self.server.cache_reset == "restart" and not self.server.start_command:
                warnings.append("server.cache_reset is 'restart' but server.start_command is not set")
            if self.server.cache_reset == "none":
                warnings.append("server.cache_reset is 'none': prefix-cache state is uncontrolled; label results exploratory")
            if self.tools.executor == "local":
                warnings.append("tools.executor is 'local': generated code runs without isolation")
            if not self.tools.worker_cpus:
                warnings.append("tools.worker_cpus is empty: tool CPU is not pinned apart from the server")
            if self.model.revision is None:
                warnings.append("model.revision is not pinned")
        if rate is None:
            warnings.append("cost.allocation_hourly_rate is unknown: cost will be reported as unavailable")
        return warnings

    @property
    def tool_worker_count(self) -> int:
        return len(self.tools.worker_cpus) or self.tools.workers


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict where ``override`` wins, merging nested dicts recursively.

    Every key present in ``override`` wins, including False, 0 and None.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _find_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


def load_raw(path: Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in _seen:
        raise ConfigError(f"'extends' cycle through {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping")
    parent = data.pop("extends", None)
    if parent:
        data = deep_merge(load_raw(path.parent / parent, (*_seen, path)), data)
    return data


def build_config(data: dict[str, Any], root: Path) -> Config:
    unknown = set(data) - {f.name for f in dataclasses.fields(Config)} - {"root"}
    if unknown or "root" in data:
        raise ConfigError(f"unknown top-level config keys: {sorted(unknown | ({'root'} & set(data)))}")
    kwargs: dict[str, Any] = {"root": root}
    for key, value in data.items():
        section = _SECTIONS.get(key)
        if section is None:
            kwargs[key] = value
            continue
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ConfigError(f"config section '{key}' must be a mapping")
        allowed = {f.name for f in dataclasses.fields(section)}
        bad = set(value) - allowed
        if bad:
            raise ConfigError(f"unknown keys in '{key}': {sorted(bad)}")
        kwargs[key] = section(**value)
    return Config(**kwargs)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    path = Path(path)
    data = load_raw(path)
    if overrides:
        data = deep_merge(data, overrides)
    cfg = build_config(data, _find_root(path.resolve().parent))
    cfg.validate()
    return cfg


def flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def protected_settings(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Flattened config entries that must match between baseline and candidate."""
    return {
        key: value
        for key, value in flatten(config_dict).items()
        if key not in TUNABLE_KEYS and key not in DESCRIPTIVE_KEYS
        # The rendered launch command follows server.args; args are compared instead.
        and key not in ("server.start_command", "server.stop_command")
    }


def fingerprint(data: Any) -> str:
    encoded = json.dumps(data, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def render_args(args: dict[str, Any]) -> list[str]:
    """Render ``{"max-model-len": 8192, "enable-prefix-caching": False}`` as CLI flags."""
    rendered: list[str] = []
    for key, value in args.items():
        if value is None:
            continue
        flag = f"--{key}"
        if value is True:
            rendered.append(flag)
        elif value is False:
            rendered.append(f"--no-{key}")
        elif isinstance(value, (list, tuple)):
            rendered.append(flag)
            rendered.extend(str(item) for item in value)
        else:
            rendered.extend([flag, str(value)])
    return rendered


def render_command(template: list[str], args: dict[str, Any]) -> list[str]:
    command: list[str] = []
    for part in template:
        if part == "{args}":
            command.extend(render_args(args))
        else:
            command.append(part)
    return command
