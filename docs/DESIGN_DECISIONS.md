# Design decisions

This records how the gaps found in review of the development plan were resolved,
plus decisions made while building. Each entry names where it is implemented and
which test covers it.

## 1. Timeout and cancellation boundaries

| Limit | Clock covers | On expiry | Outcome fields |
| --- | --- | --- | --- |
| `episode_wall_timeout_s` (backstop) | Admission (episode slot acquired) → end of agent loop: waiting for a request slot, requests, waiting for a tool slot, tools | Outstanding request cancelled (stream closed → vLLM aborts it) or tool process/container killed | `stop_reason=wall_timeout`, infra `ok` |
| FIFO queueing before admission | Not counted against any budget | — | recorded as `admission_wait_s` |
| `request_timeout_s` (per call) | One model request, including retries | Request cancelled | `stop_reason=infra_error`, `infrastructure_status=model_timeout` |
| `test_timeout_s` (per call) | One `run_tests` | Process killed; feedback to the model (generated code can hang) | episode continues |
| `grader_timeout_s` | Grading, after the agent loop | Process killed | `grader_outcome=fail`, detail `grader_timeout` |

The smaller of a per-call limit and the remaining wall time applies, and the record says
which fired. **Step and token budgets are the primary limits**, because they don't depend
on scheduling. The wall clock is only a backstop. Any episode that hits it is reported as
"budgets were not scheduling-neutral", and the acceptance rule refuses an increase in
backstop hits.

**Closing a run:** once the cohort is graded, the runner kills leftover tool processes
(`docker ps --filter label=rollout_cost.run_id=…`) and waits until the vLLM gauges
`num_requests_running` and `num_requests_waiting` both read 0. If idleness can't be
verified, the run carries a warning.

*Code:* `harness.py` module docstring and `_bounded`, `runner.py` drain. *Tests:*
`test_wall_timeout_cancels_outstanding_request`, `test_request_timeout_is_an_infrastructure_error`.

## 2. Comparable cache conditions

The same procedure runs before **every** measured run (`server.py`):

1. Reset the cache. `cache_reset: restart` (the default) stops and restarts the server; it
   works the same way on any vLLM version. `endpoint` calls `POST /reset_prefix_cache`,
   which needs `VLLM_SERVER_DEV_MODE=1`. `none` labels the run exploratory.
2. Wait for `/health`.
3. Warm up with `fixtures/warmup.jsonl`. These prompts use a different system prompt, so
   the measured task prefixes start uncached in every condition. The warm-up file's hash
   is a protected setting.
4. Confirm the server is idle.

Prefix-cache counters are snapshotted after warm-up and after the cohort, and the report
shows the hit rate inside the window. Restart and warm-up time is recorded as overhead.
It's excluded from the steady-state window but included in the run-inclusive cost.

The server command is rendered from `server.args`, so the recorded arguments and the
launched arguments can't drift apart. The server log is saved per run as `server.log`.
It contains vLLM's own view of the effective configuration.

## 3. Aggregate CPU budget

Tool work goes through a `ToolPool` with exactly `len(tools.worker_cpus)` slots. Each slot
is a container pinned to one CPU (`--cpuset-cpus`). Raising `max_active_episodes` adds
waiting for a tool slot, never more CPU. The pool size and CPU list are protected
settings. The vLLM process is pinned to `server.reserved_cpus` through the start command
(`taskset`), and the config is rejected if the two sets overlap. Grading shares the same
pool.

The report breaks out `tool_wait` separately. In the synthetic sweep, going from 3 to 6
active episodes with 2 tool workers didn't shorten the window, because tool slots were
the bottleneck. A per-container CPU cap would have hidden that.

## 4. Backlog size versus concurrency

The cohort is **tasks × repetitions**, and each repetition is an independent attempt with
its own seed and fresh workspace. The measured baseline uses 6 × 4 = 24 episodes, three
times the largest proposed episode cap of 8. Every run reports:

- `steady_state_fraction`: how much of the window the episode cap was actually full.
- `tool_time_fraction`: whether tools take enough time for model/tool overlap to matter.

Warnings fire when the cap exceeds the cohort, when the cohort is under 3× the cap, or
when tool time is under 10% of episode time. **The pilot must check `tool_time_fraction`
before any concurrency claim.** If the tiny tasks barely use tools, the overlap
hypothesis has nothing to act on.

## 5. Truncation and context overflow

- **Truncated replies** (`finish_reason == "length"`) are rejected without parsing, even
  if a prefix happens to decode. The reply consumes a step and the feedback suggests a
  smaller action.
- **`edit_file` was added to the protocol** (exact single-occurrence replacement). The
  plan's protocol only had `write_file`, which makes the model reproduce the whole file
  inside a 512-token reply.
- **Context:** before each request, the prompt size is the server-reported usage of the
  previous request plus a conservative estimate (3 characters per token) for the text
  appended since. If it would exceed `context_window_tokens − max_output_tokens_per_call`,
  the episode stops with `context_exhausted`. A server-side 400 for over-length prompts
  maps to the same stop reason, not an infrastructure error. History is never compacted.
- **Observations** are cut to `max_observation_chars`. `read_file` keeps the head; test
  output keeps 20% head and 80% tail, where pytest prints its summary.
- A protocol-error rate above 20% triggers a warning that the run may be measuring
  protocol failures.

*Tests:* `test_truncated_reply_consumes_a_step`, `test_context_exhaustion_stops_before_overflowing`.

## 6. Separate outcome fields

Each episode records `stop_reason`, `infrastructure_status` and `grader_outcome`
independently (`outcomes.py`). The four report categories are derived from these three
fields and never recorded directly:

- **infra_error**: any infrastructure failure, or the grader couldn't assess the final
  state. These are excluded from valid denominators. They're still graded for
  diagnostics, but never counted as model quality.
- **success**: the grader passed. A correct patch that ran out of budget still counts
  here; the stop-reason × grade cross-tab shows how many did.
- **failed_after_finish** / **failed_budget_exhausted**: the rest, split by stop reason.

*Test:* `test_derive_category` (all combinations).

## 7. Acceptance rule

Predeclared in the experiment file (`analysis.DEFAULT_ACCEPTANCE`) and evaluated over all
paired trials:

- Pooled successes may not drop (`max_pooled_success_drop: 0`).
- **Task-level:** for each task, pass counts are pooled over trials and repetitions. A drop
  beyond `task_regression_tolerance` counts as a regression, and no regressions are
  allowed. The episode transition table (pass→fail, fail→pass, …) is part of every
  comparison.
- Infrastructure errors and wall-time backstop hits may not increase.
- The pooled allocated window must shrink by at least `min_window_reduction_pct`, the
  window must be shorter in every paired trial, and the reduction must **exceed the
  trial-to-trial spread** of either side (see below).
- Protected settings, the cohort and the synthetic flag must match. Otherwise the
  comparison is invalid.
- Each run must be complete and contain exactly one terminal record for every declared
  episode. Summary values used by the decision are checked against values recomputed
  from episode records and the manifest timeline. Missing, duplicate or unexpected
  records and inconsistent summaries cannot pass acceptance.
- Real-server idleness must be verified both before and after the cohort. Missing
  metrics or a busy server make the comparison invalid while preserving run artifacts
  for inspection. Scripted and mock runs remain explicitly synthetic.

Restart mode checks that the endpoint is unused after stop, before launching a child.
Readiness also checks the child's exit status before and after each health request, so
an existing healthy server cannot conceal a failed launch. Regression tests for these
guards are in `tests/test_run_validation.py` and `tests/test_server.py`.

`doctor` requires usable running/waiting request gauges even when optional telemetry
is absent. Preparation stops before starting the cohort if idleness is unverified.
On Linux the server starts in its own session, and shutdown signals the whole process
group, including workers whose API parent has already exited. Surviving workers are
force-stopped and the parent is reaped before another run can start.

*Test:* `test_comparison_detects_task_swaps_hidden_by_equal_totals`. The baseline solves
A/B and the candidate solves B/C: equal totals, and the comparison is rejected.

**Found during the build:** an A/A run (identical configs, 2 trials) on the development
machine showed a 7.5% "reduction" from noise alone and passed every original check.
Three checks were added:

- An identical pair is labeled **A/A calibration**, not accepted.
- At least `min_trials: 2` paired trials are required.
- The reduction must exceed the relative spread `(max − min) / mean` of either side's
  windows.

Single-trial dev selection in `sweep` relaxes the last two and says so. **Run one A/A
pair on the GPU host first** (`paired --baseline X --candidate X`) to measure the noise
floor.

## 8. Objective and latency guardrail

The objective is **batch completion**. At a fixed rate, steady-state cost is
proportional to makespan, so the redundant 10% makespan guardrail was dropped. An
optional `max_p90_service_time_increase_pct` guards per-episode responsiveness, measured
from admission to the end of the agent loop. Queue-inclusive `episode_latency_s` (cohort
start → graded) is reported, but no decision depends on it.

## Other decisions

- **Hidden-test integrity.** Grading starts from a pristine copy of the fixture. Only the
  task's `graded_paths` (`src/`) come from the agent's workspace, so editing public tests
  can't produce a pass (`test_editing_public_tests_does_not_count_as_a_fix`). Hidden tests
  are never mounted into a container the agent's code runs in.
- **Read-only test containers.** Containers for `run_tests` mount the workspace read-only,
  so only the trusted runner writes files. That rules out symlink tricks against later
  `read_file` calls. The path checks still reject `..`, absolute paths, backslashes and
  symlinks.
- **Retries.** Only connection errors and HTTP 502/503/504 that occur before any content
  is streamed are retried, at most `max_request_retries` times. Every attempt is recorded,
  and an unsuccessful but valid episode is never rerun.
- **What counts as a protected setting.** The behavior hash covers only modules that
  change what a run does. Reports can be re-rendered with newer analysis code without
  invalidating comparisons.
- **The local executor has no isolation.** It exists for scripted development runs on
  machines without Docker. `doctor` warns about it, and the measured configs use Docker.

## Observability and compatibility

**Two stores, one source of values.** Live metrics go to Prometheus and history goes to
PostgreSQL, because they answer different questions. "Is the run healthy right now?"
needs scraping. "Compare these runs" needs SQL over episodes, spans and comparisons.
Neither store recomputes anything. The exporter derives its numbers from the run's own
event stream, plus read-only views of the semaphore and tool pool. `export` stores the
same summaries, comparisons and acceptance results the Markdown reports use, through
`analysis.evaluate_experiment_file` and `evaluate_sweep_selection`.

**Observation only.** The exporter can't change scheduling, and a busy port is recorded
rather than failing the run. Its settings are descriptive, not protected, so turning
live metrics on doesn't invalidate a comparison.

**No plugins, no dcgm-exporter.** Dashboards use built-in Grafana panels only. GPU
gauges come from the runner's own nvidia-smi sampler, so nothing on the GPU host needs
Docker.

**Single sources of truth, enforced by tests** (`tests/test_observability.py`):

| Shared item | Defined in | Checked against |
| --- | --- | --- |
| Live metric names | `contract.LIVE_METRICS` | Exporter output during a run; every PromQL query |
| vLLM metric names | `contract.VLLM_DASHBOARD_METRICS`, `telemetry.SERVER_SIGNALS` | Dashboard queries; mock server output |
| Timeline state codes | `contract.STATE_CODES` | `spans.py`; the timeline's value mappings |
| Export tables | `src/rollout_cost/schema.sql` (versioned) | Row builders; every dashboard SQL query (and executed against Postgres when `RC_TEST_PG` is set) |
| Datasource UIDs | `contract.DATASOURCE_UIDS` | Provisioning files and every dashboard reference |
| Image/binary versions | `observability/versions.env` | `compose.yaml` defaults; the native script |
| Dashboard JSON | `observability/build_dashboards.py` | Committed JSON (stale files fail) |
| Run artifacts | `contract.RUN_FORMAT_VERSION` | `load_run` rejects unknown versions; stored paths are relative POSIX |

**Mock vLLM server.** `rollout-cost mock-server` serves streaming chat completions
(driven by the scripted policies), `/health`, `/version` (`"mock": true`), `/metrics` with
vLLM's names, and `/reset_prefix_cache`. It emulates `max-num-seqs` queueing, prefix-cache
reuse and client-disconnect aborts. That exercises the real HTTP client, the server
restart/warm-up/drain lifecycle and the vLLM dashboard panels without a GPU. The runner
detects the mock and labels the run SYNTHETIC.

**Compatibility matrix** (as verified on 2026-10-25)

| Environment | Executor | Observability | Verified by |
| --- | --- | --- | --- |
| Windows laptop | local (unpinned) | Docker Compose (Grafana 11.6, Prometheus 3.1, PostgreSQL 16.4) | Full suite including Postgres integration; mock-vLLM run scraped by Prometheus in Docker; every panel query of all four dashboards returns data through Grafana (`scripts/verify_dashboards.py`) |
| Linux without Docker (Lightning Studio) | local, CPU-pinned | `observability-native.sh` | `scripts/rehearse-lightning.sh` in `ubuntu:24.04`: setup `--rehearsal`, `doctor --start-server`, smoke and full pinned runs, Prometheus storing runner and vLLM metrics, runner pinned to the reserved CPUs, the Linux-only affinity test, and the Live dashboard checked from the laptop |
| Linux to Windows artifacts | — | — | Rehearsal run directories exported and loaded on Windows unchanged |
| Linux with Docker | docker | Docker Compose | The setup script detects the daemon. Prometheus in compose reaches host services through `host-gateway`, so on Linux the model server must listen on an address the bridge can reach. vLLM's `host: 127.0.0.1` can't be reached that way, so prefer the native script there |
| Real GPU + vLLM | either | either | Only in the Lightning session (see the runbook's list) |

**Runner CPU pinning.** When `server.reserved_cpus` is set on Linux, the runner pins itself
to those CPUs for the run. The tool-worker CPUs then do nothing but tests and grading.
The applied affinity is recorded in `run.json`.

**Thread-safe local tool pinning.** Local tool commands are launched through `taskset`,
which applies the slot's CPU before executing Python. The runner does not use Python's
`preexec_fn`, which can deadlock when the live exporter or a subprocess watcher has
started threads. If `taskset` is missing, pinned execution fails as an infrastructure
error instead of silently running without affinity. `tests/test_lightning_setup.py`
checks command launch and missing-tool behavior; the Linux affinity test checks the
child's actual CPU set.

**Lightning preparation and model selection.** `lightning-setup.sh --prepare` installs
dependencies and downloads the selected model on CPU without starting GPU services.
Normal setup always synchronizes the requested vLLM release and verifies its installed
metadata; the default environment path includes the version. The script passes
`MODEL_REPO` to `pin --model-repo`, so the downloaded model, served model and recorded
tokenizer revision agree. Shell regression tests exercise preparation, reused
environments, version mismatch failures and model selection without downloading vLLM.

**Local PostgreSQL servers.** The compose database publishes on host port 55432, because
5432 and 5433 are often taken by local installations (including on the development
machine). Override it with `POSTGRES_HOST_PORT`.

## Effort estimate (revised)

The six-task milestone is the delivery target. The software built so far is roughly the
plan's sessions 1–3, minus anything that needs a GPU. Remaining work:

- **GPU pilot and baseline:** 3–5 hours on the host. That covers verifying vLLM flags
  for the pinned release, calibrating budgets, measuring tool fraction and an A/A noise
  floor, and running the first sweep.
- **Paired validation:** depends on the A/A spread. If noise is large relative to the
  effect, more trials are needed.
- **24-task held-out expansion:** a separate effort, about 8–12 hours. That's 18 new
  independently designed bugs, each with hidden tests validated against starting code
  and a reference fix. It shouldn't be counted inside the original 12–18 hours.
