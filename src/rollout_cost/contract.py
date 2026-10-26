"""Names and versions shared by the runner, exporter, export schema and dashboards.

Everything that crosses a component boundary is defined here once. Tests check
the live exporter, ``spans.py``, ``observability/schema.sql`` and every Grafana
dashboard against these definitions, so a rename in one place cannot silently
break another.
"""

from __future__ import annotations

# Version of the run-directory layout (run.json, episodes.jsonl, events.jsonl, ...).
# Bump when a field that readers depend on changes meaning or disappears.
RUN_FORMAT_VERSION = 1
SUPPORTED_RUN_FORMATS = frozenset({1})

# Version of observability/schema.sql. export refuses a database at another version.
EXPORT_SCHEMA_VERSION = 1

# Span kinds derived from events, with the numeric codes used by Grafana's state
# timeline (SQL long-to-wide conversion needs numeric values; the dashboard maps
# codes back to names and colors).
SPAN_KINDS: dict[str, int] = {
    "idle": 0,
    "queued": 1,
    "llm_wait": 2,
    "model_request": 3,
    "tool_wait": 4,
    "tool": 5,
    "file_op": 6,
    "grading": 7,
    "grading_wait": 8,
}
# Terminal states shown after an episode is graded.
TERMINAL_STATES: dict[str, int] = {
    "success": 10,
    "failed_after_finish": 11,
    "failed_budget_exhausted": 12,
    "infra_error": 13,
}
STATE_CODES: dict[str, int] = {**SPAN_KINDS, **TERMINAL_STATES}

# Concurrency step series derived from events.
CONCURRENCY_SERIES = ("active_episodes", "llm_inflight", "llm_waiting", "tool_busy", "tool_waiting")

# Live metrics exported by the runner while a run is in progress. Exposed names
# (counters carry the _total suffix). Every metric has run_id and config labels.
LIVE_LABELS = ("run_id", "config")
LIVE_METRICS: dict[str, dict] = {
    "rc_run_info": {"type": "gauge", "labels": ("synthetic", "split"), "help": "1 while a run is exported"},
    "rc_run_phase": {"type": "gauge", "labels": ("phase",), "help": "1 for the current phase (prep, cohort, drain, done)"},
    "rc_episodes_submitted": {"type": "gauge", "labels": (), "help": "Episodes in the cohort"},
    "rc_episodes_active": {"type": "gauge", "labels": (), "help": "Episodes holding an episode slot"},
    "rc_episode_cap": {"type": "gauge", "labels": (), "help": "max_active_episodes"},
    "rc_episodes_terminal_total": {"type": "counter", "labels": ("category",), "help": "Graded episodes by category"},
    "rc_llm_requests_inflight": {"type": "gauge", "labels": (), "help": "Model requests holding a request slot"},
    "rc_llm_requests_waiting": {"type": "gauge", "labels": (), "help": "Ready model requests waiting for a slot"},
    "rc_llm_request_cap": {"type": "gauge", "labels": (), "help": "max_inflight_llm_requests"},
    "rc_llm_requests_total": {"type": "counter", "labels": ("status",), "help": "Finished model requests by status"},
    "rc_tool_slots_busy": {"type": "gauge", "labels": (), "help": "Tool slots in use (tests and grading)"},
    "rc_tool_slots_waiting": {"type": "gauge", "labels": (), "help": "Tool jobs waiting for a slot"},
    "rc_tool_slots_total": {"type": "gauge", "labels": (), "help": "Tool slots (fixed CPU budget)"},
    "rc_prompt_tokens_total": {"type": "counter", "labels": (), "help": "Prompt tokens reported by the server"},
    "rc_completion_tokens_total": {"type": "counter", "labels": (), "help": "Completion tokens reported by the server"},
    "rc_window_seconds": {"type": "gauge", "labels": (), "help": "Elapsed steady-state window"},
    "rc_allocation_cost": {"type": "gauge", "labels": ("currency",), "help": "Window cost so far (absent if price unknown)"},
    "rc_gpu_utilization_percent": {"type": "gauge", "labels": ("gpu",), "help": "nvidia-smi utilization.gpu"},
    "rc_gpu_memory_used_mib": {"type": "gauge", "labels": ("gpu",), "help": "nvidia-smi memory.used"},
    "rc_gpu_memory_total_mib": {"type": "gauge", "labels": ("gpu",), "help": "nvidia-smi memory.total"},
}

# vLLM metric names the dashboards may query (in addition to telemetry.SERVER_SIGNALS).
VLLM_DASHBOARD_METRICS = frozenset(
    {
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:gpu_cache_usage_perc",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:num_preemptions_total",
        "vllm:time_to_first_token_seconds_bucket",
        "vllm:e2e_request_latency_seconds_bucket",
    }
)

# Fixed Grafana datasource UIDs used by provisioning and dashboards.
DATASOURCE_UIDS = {"prometheus": "rc-prometheus", "postgres": "rc-postgres"}
