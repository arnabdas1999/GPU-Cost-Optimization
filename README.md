# Rollout Cost Lab

Rollout Cost Lab measures what it costs, in rented GPU time, to run a batch of AI coding
agents against a self-hosted model. It then tests whether a different serving
configuration can do the same work for less without solving fewer tasks.

> For this model, GPU and workload, which configuration completes the same set of
> multi-step coding-agent episodes at lower allocated-GPU cost, while keeping task
> outcomes intact?

The project has four parts:

- a command-line tool that runs the agent episodes;
- a set of small coding tasks with hidden tests;
- cost and outcome accounting that compares configurations fairly;
- Grafana dashboards, live and historical.

The model server is [vLLM](https://docs.vllm.ai/); this project builds the workload,
the scheduling controls, the measurements and the experiment logic around it.

**Status: working prototype; GPU task-solving quality still needs calibration.** Initial
Colab T4 runs completed without infrastructure errors but solved zero tasks. No GPU cost
savings have been established. The software also runs and is tested on a laptop without
a GPU, using a scripted model or a mock vLLM server.
Results from those runs are labeled **SYNTHETIC** everywhere. They show that the
software works, not that any configuration saves money. Start real-GPU testing with
the Colab/Kaggle T4 notebook: a one-task preflight, six-episode smoke test, optional pilot, and result download.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arnabdas1999/GPU-Cost-Optimization/blob/main/notebooks/gpu_smoke_test.ipynb)

Choose a GPU runtime, then **Run all**. For Kaggle, import
[`notebooks/gpu_smoke_test.ipynb`](notebooks/gpu_smoke_test.ipynb), select **T4** and enable
**Internet**. See the [notebook runbook](docs/NOTEBOOK_RUNBOOK.md) for exact steps and
troubleshooting. The calibration profile uses one GPU, the 7B AWQ coder model, FP16
activations, an 8,192-token context and 12 replies per episode. The longer pilot and A/A
runs are off by default; review the new smoke result before enabling them.
The notebook uses complete-file replacements with constrained JSON actions, avoiding
the literal-edit loops seen in the first GPU runs. Grading remains independent; the
preflight must pass on the actual model before running the full cohort.

The [Lightning runbook](docs/LIGHTNING_RUNBOOK.md) remains available for an L4 host.
As of October 26, 2026, Lightning reports Studio access unavailable during its
[incident investigation](https://lightning.ai/security-incident/); use the notebook route first.

---

## Contents

1. [Why this exists](#1-why-this-exists)
2. [How it works](#2-how-it-works)
3. [Quick start (no GPU, about 5 minutes)](#3-quick-start-no-gpu-about-5-minutes)
4. [Dashboards](#4-dashboards)
5. [Running on a real GPU](#5-running-on-a-real-gpu)
6. [Finding a cheaper configuration: the workflow](#6-finding-a-cheaper-configuration-the-workflow)
7. [Configuration guide](#7-configuration-guide)
8. [Reading the results](#8-reading-the-results)
9. [Command reference](#9-command-reference)
10. [Adding tasks](#10-adding-tasks)
11. [Testing and verification](#11-testing-and-verification)
12. [Project layout](#12-project-layout)
13. [Status, limitations and scope](#13-status-limitations-and-scope)

---

## 1. Why this exists

An AI coding agent doesn't send one request and stop. It works in a loop:

```
ask the model → model picks an action → run it (read a file, edit, run tests) → feed the result back → ask again
```

While one agent waits for its tests to run, the GPU could be serving another agent. But
running too many agents at once makes them queue for the GPU and compete for memory and
CPU. So the usual serving benchmark, requests per second, doesn't tell you whether a
whole batch of agent tasks finishes sooner or more cheaply.

This project measures what you actually pay for: **allocated GPU time**, meaning the
seconds the GPU host is rented, multiplied by its hourly price. It measures that for a
fixed batch of agent tasks and checks that a cheaper configuration doesn't solve fewer
tasks. That's useful anywhere agent trajectories are generated in bulk: agent
evaluation, benchmark runs, rollout generation for post-training.

---

## 2. How it works

```
                 configs/*.yaml  (model, server flags, budgets, concurrency, price)
                        │
                        ▼
  ┌──────────────── rollout-cost run ──────────────────────────────────────────────┐
  │ 1. Manifest: record the config, task/grader hashes, harness version, host and GPU │
  │ 2. Server prep: restart vLLM (clears its prefix cache), fixed warm-up, wait idle  │
  │ 3. Cohort: tasks × repetitions, admitted strictly in order (FIFO)                 │
  │                                                                                    │
  │    episode cap ──► [episode][episode][episode][episode]   max_active_episodes      │
  │                        │ each episode loops:                                       │
  │                        ├─► model request ── request cap ──► vLLM (GPU)              │
  │                        └─► run tests ────── tool pool ─────► pinned CPU workers     │
  │                                                                                    │
  │ 4. Grading: hidden tests on a clean copy of the final code                         │
  │ 5. Drain: wait until vLLM reports no running or waiting requests                   │
  └────────────────────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
   runs/<run-id>/  events, transcripts, outcomes, summary.json, report.md
                        │
        ┌───────────────┼───────────────────────────┐
        ▼               ▼                           ▼
   compare / sweep / paired             rollout-cost export ──► PostgreSQL ──► Grafana
   (acceptance decision)                live exporter :9400 ──► Prometheus ──► Grafana
```

### The workload

There are six small, original Python repositories, two in each of three families. Each
contains a planted bug:

| Family | Tasks | The bug |
| --- | --- | --- |
| Retry logic | `retry_permanent`, `retry_backoff` | Permanent errors are retried; off-by-one attempt limit and backoff schedule |
| Configuration | `config_merge`, `config_env` | Explicit `false`/`0` overrides ignored; `"false"` parsed as true |
| Parsing | `parse_duration`, `parse_port_range` | Malformed durations and out-of-range or reversed port ranges silently accepted |

Each task has public tests the agent can run, plus **hidden tests** that only the grader
sees. `validate-fixtures` checks that every starting version fails and every reference
fix passes. Grading copies only the agent's `src/` onto a clean copy of the task, so
editing the public tests can't produce a pass.

### One episode

The model replies with one JSON action per turn:

```json
{"action": "read_file", "path": "src/retry.py"}
{"action": "edit_file", "path": "src/retry.py", "old": "exact text", "new": "replacement"}
{"action": "write_file", "path": "src/retry.py", "content": "..."}
{"action": "run_tests"}
{"action": "finish"}
```

The harness runs the action and feeds the result back. Each episode has a fixed budget.
The baseline profile uses:

- 6 model replies;
- 512 output tokens per reply;
- an 8,192-token context window;
- a 300-second wall-clock backstop.

The notebook calibration profile uses 12 replies and 1,024 output tokens per reply,
with the same 8,192-token context and 300-second backstop. Python edits receive syntax
feedback without executing the generated code; invalid edits remain applied for repair.
A malformed or cut-off reply still uses up one reply. When the budget runs out, the
final code is graded as it stands.

### Three controls on concurrency

| Control | Setting | What it bounds |
| --- | --- | --- |
| Episode cap | `execution.max_active_episodes` | How many tasks are in progress at once |
| Request cap | `execution.max_inflight_llm_requests` | How many model requests are in flight. An episode running tests frees its request slot for others |
| Tool pool | `tools.worker_cpus` | A **fixed** set of CPU cores for running tests and grading, each worker pinned to one core. More episodes means more waiting for a core, never more CPU |

The fixed tool pool keeps comparisons honest. If raising the episode cap could also grab
more CPU, a "faster" result might come from extra CPU rather than better GPU scheduling.

### What gets recorded

Every episode records three outcome fields separately:

| Field | Values |
| --- | --- |
| `stop_reason` | `finished`, `max_steps`, `context_exhausted`, `wall_timeout`, `infra_error` |
| `infrastructure_status` | `ok`, `model_server_error`, `model_timeout`, `tool_executor_error`, `harness_error` |
| `grader_outcome` | `pass`, `fail`, `grader_error`, `not_graded` |

From those, each episode lands in exactly one category:

- **success**: the hidden tests passed, even if the episode ran out of budget.
- **failed_after_finish**: the model said it was done, but the tests failed.
- **failed_budget_exhausted**: the budget ran out and the tests failed.
- **infra_error**: something outside the model broke. These are excluded from quality
  numbers.

The run also records timings for every request and tool call, token counts, GPU and
vLLM samples, and an event-by-event trace.

### How cost is calculated

```
allocation_cost = billed_seconds / 3600 × hourly_rate
```

- **Steady-state window:** from the first episode admitted to the last episode graded.
  Comparisons use this.
- **Run-inclusive:** also counts the server restart, warm-up and drain, with billing
  rounding (`billing_granularity_seconds`).
- **Per valid episode** and **per success**: if there are no successes, cost per success
  is shown as undefined, never as 0.

Overlapping episode times are never added up, and the price is never scaled by GPU
utilization. **An unknown price shows as "unavailable", never as $0.** Free credits still
have a price, so record the on-demand rate.

### Fair comparisons

A comparison is only valid if everything except a short list of tunable settings is
identical:

- the same tasks, repetitions and budgets;
- the same model revision, prompts and harness code (hashed);
- the same CPU layout, tool pool and grader;
- the same cache-reset procedure.

Before each measured run, the server is restarted (clearing vLLM's prefix cache) and
warmed up identically. Restart refuses an endpoint that is still occupied after stop,
and readiness requires the launched process to remain alive. Server idleness must be
verified before and after the cohort for a comparison to be valid. `doctor` requires
usable running/waiting request gauges, and preparation stops before the measured
workload if idleness cannot be verified. A failed check after the cohort leaves run
artifacts available for inspection and blocks acceptance. Each run must also be
complete, contain exactly one terminal record per declared episode, and have summary
decision values consistent with its records and timeline.

A configuration is **accepted** only if it passes a rule you set in advance (see
[section 8](#8-reading-the-results)). That rule includes a noise check, because
identical configurations can differ by several percent between runs.

The rationale for each of these choices is in
[docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md).

---

## 3. Quick start (no GPU, about 5 minutes)

**Requirements:** [uv](https://docs.astral.sh/uv/) and Git. Python is installed by uv if
needed. Docker is optional; you need it for the dashboards.

```bash
git clone <repo-url> rollout-cost && cd rollout-cost
uv sync --extra dashboard

# 1. The six tasks behave as intended: starting code fails, reference fixes pass.
uv run rollout-cost validate-fixtures --config configs/synthetic.yaml

# 2. A full run with a scripted (fake) model, in-process. Takes under a minute.
uv run rollout-cost run --config configs/synthetic.yaml --split dev

# 3. The same through the real HTTP path: the runner starts a mock vLLM server,
#    warms it up, streams every request, reads its metrics and checks it drains.
uv run rollout-cost run --config configs/mock-vllm.yaml --split dev
```

Each run prints a one-line summary and writes a folder under `runs/`. Open
`runs/<run-id>/report.md` for the full report: outcomes, cost, where episode time went,
latency and warnings.

You can also try a sweep and a comparison:

```bash
uv run rollout-cost sweep --config configs/synthetic-sweep.yaml        # 3 configurations; picks one
uv run rollout-cost compare --baseline runs/<id-a> --candidate runs/<id-b>
```

Synthetic numbers show the software works. Don't read them as GPU results.

---

## 4. Dashboards

There are four Grafana dashboards over the same data the Markdown reports use:

| Dashboard | Shows |
| --- | --- |
| **Live run** | While a run is in progress: phase, progress, episodes against the cap, request slots, tool slots, the vLLM queue and KV cache, tokens per second, prefix-cache hit rate, time to first token, GPU, and cost so far |
| **Runs** | Every exported run, synthetic or measured, with outcomes, allocated time and cost. Click a run id to open its detail |
| **Run detail** | **Episode timeline**: one row per episode, coloured by what it was doing (queued, waiting for the GPU, generating, waiting for a CPU, running tests, grading, outcome). Below it: concurrency against each cap, where the time went, latency, per-task results, GPU and server samples, warnings |
| **Experiments and sweeps** | The decision, allocated window, cost and successes per trial or variant, every acceptance check, which episodes flipped between pass and fail, per-task passes |

### On your laptop (Docker)

```bash
docker compose -f observability/compose.yaml up -d     # Grafana, Prometheus, PostgreSQL
uv run rollout-cost run --config configs/mock-vllm.yaml # watch "Live run" while it runs
uv run rollout-cost export                              # load runs/ and experiments/ into PostgreSQL
```

Then open **http://127.0.0.1:3000** (anonymous viewing; admin login `admin`/`admin`).
Start from **Runs** and click a run.

| Service | Address | Notes |
| --- | --- | --- |
| Grafana | `127.0.0.1:3000` | Dashboards are provisioned automatically |
| Prometheus | `127.0.0.1:9090` | Scrapes the runner (`:9400`) and the model server (`:8000`) |
| PostgreSQL | `127.0.0.1:55432` | History database. Not 5432, so a locally installed PostgreSQL is left alone |

- `rollout-cost export` is safe to rerun; it replaces each run's rows. Run it after
  every session.
- Stop the stack with `docker compose -f observability/compose.yaml down`.
- To change logins or ports, copy `observability/.env.example` to `observability/.env`.

### On a GPU host without Docker

```bash
bash scripts/observability-native.sh start   # downloads pinned Prometheus + Grafana, no root; live view only
bash scripts/observability-native.sh stop    # stop when done: a running server keeps a Lightning Studio awake
```

Historical dashboards run on the laptop: copy the `runs/` folder home and `export` it.
More detail is in [observability/README.md](observability/README.md).

To share historical results at a public HTTPS URL without keeping your laptop on,
follow the [free Render + Neon hosting guide](docs/FREE_HOSTING.md). The site URL
stays the same when you import new runs. The deployment includes read-only viewing
and persistent Grafana state; the GPU experiments continue running separately.

---

## 5. Running on a real GPU

### Colab / Kaggle (start here)

Open [the GPU test notebook](https://colab.research.google.com/github/arnabdas1999/GPU-Cost-Optimization/blob/main/notebooks/gpu_smoke_test.ipynb)
in Colab, select a GPU, and run all cells. On Kaggle, import the same notebook, select
T4 and enable Internet. The [notebook runbook](docs/NOTEBOOK_RUNBOOK.md) covers setup,
the smoke test, optional longer runs and downloading diagnostics. GPU execution still
needs to be verified on the assigned runtime; CPU rehearsals establish software behavior.

### Lightning AI (alternative when Studio access returns)

The whole session is scripted and has been rehearsed in a plain Ubuntu container without
a GPU. Follow [docs/LIGHTNING_RUNBOOK.md](docs/LIGHTNING_RUNBOOK.md). It's a timed
checklist with a rough GPU-time budget and a stop/go check at every step, so free credits
aren't spent debugging.

```bash
# On the CPU Studio, set VLLM_VERSION to an exact release first:
bash scripts/lightning-setup.sh --prepare

# After switching to the L4 GPU, restore VLLM_VERSION and set RATE and RATE_SOURCE:
bash scripts/lightning-setup.sh
uv run rollout-cost run --config configs/smoke.pinned.yaml --split dev       # 6 episodes, one at a time
uv run rollout-cost run --config configs/lightning.pinned.yaml --split dev   # pilot baseline (24 episodes)
```

`lightning-setup.sh` does the following:

- checks the GPU;
- installs uv and the project;
- synchronizes and verifies your exact vLLM version, even when its environment exists;
- uses a version-specific environment by default (`~/vllm-venv-$VLLM_VERSION`);
- resolves the selected model's exact commit and downloads it;
- splits the CPUs between vLLM and the tool workers;
- writes `configs/lightning.pinned.yaml`, `smoke.pinned.yaml` and `sweep.pinned.yaml`;
- validates the tasks;
- runs `doctor --start-server`.

`--prepare` installs the dependencies and downloads the model on CPU, then exits without
checking the GPU, building containers, writing pinned configs or starting a server.
After switching machines, restore any `MODEL_REPO` and `VLLM_VENV` overrides as well as
`VLLM_VERSION`. The selected repository is used for both the download and serving config.
The GPU setup records the vLLM package inventory in `vllm-packages.txt` and checks CUDA
availability before attempting to start the server.

Docker isn't required. Without it, tests run as local processes pinned to their CPU.
That keeps the CPU budget fixed but doesn't isolate generated code, and the report says
so.

To rehearse the whole Lightning session on your laptop (needs Docker):

```bash
bash scripts/rehearse-lightning.sh
```

### Any other Linux GPU host

1. Clone the repo, then run `uv sync --extra dashboard` and install vLLM.
2. Build the test image:
   `docker build -t rollout-cost-tools:py312 -f docker/tool-runner.Dockerfile docker`.
3. Edit `configs/local.yaml`:
   - the model revision;
   - `server.start_command` and `server.args`, checking each flag against
     `vllm serve --help` for your version;
   - `server.reserved_cpus` and `tools.worker_cpus`;
   - the price.

   Or run `bash scripts/lightning-setup.sh` on a Linux GPU host.
4. Run `uv run rollout-cost doctor --config configs/baseline.yaml --start-server` and fix
   anything it flags.
5. Follow the workflow below.

**Model and hardware assumptions:** `Qwen/Qwen2.5-Coder-7B-Instruct` in BF16, on one
NVIDIA GPU with 24–48 GB of memory. A smaller GPU, such as a 16 GB T4, needs a smaller or
quantized model. That's a different experiment, so record it as one.

---

## 6. Finding a cheaper configuration: the workflow

The process is deliberately conservative: a small sample can easily produce a "saving"
that is really noise.

| Step | Command | What you learn |
| --- | --- | --- |
| **1. Smoke** | `run --config <smoke config>` | The model follows the protocol and solves at least something. Read two transcripts in `runs/<id>/transcripts/` |
| **2. Pilot baseline** | `run --config <base config>` | Whether the budgets are sensible (see the pilot checks in section 8). Then **freeze** the settings and commit them. Don't change tasks or budgets after this point |
| **3. A/A noise floor** | `paired --baseline X --candidate X --trials 2 --name aa` | The same config run twice. The spread between the runs is the smallest saving you can believe |
| **4. Sweep** | `sweep --config <sweep config>` | Up to 4 alternative configurations, one run each. Writes `experiments/<sweep>/chosen.yaml`, the cheapest variant that passed the rule |
| **5. Paired validation** | `paired --baseline <base> --candidate experiments/<sweep>/chosen.yaml --trials 2` | Alternating baseline/candidate runs, scored by the full acceptance rule. Writes `experiments/<name>.yaml` and `reports/<name>.md` |
| **6. Look and share** | `export`, then the dashboards, plus `reports/` | The decision, and the evidence behind it |

**What a sweep varies by default** (`configs/sweep.yaml`):

| Variant | Episode cap | Request cap |
| --- | --- | --- |
| `active-2` | 2 | 2 |
| `active-4` (baseline) | 4 | 4 |
| `active-8` | 8 | 4 |

The fixed tool pool shows whether extra concurrency helps the GPU, or just creates
queues for CPU.

**Ideas to test next**, one change at a time:

- a vLLM scheduler limit: `server.args.max-num-seqs` or `max-num-batched-tokens`;
- prefix caching on or off: `server.args.enable-prefix-caching`. The mock server shows
  heavy prefix reuse in multi-turn histories, so this is worth measuring for real;
- a different ratio of request cap to episode cap.

**If nothing is cheaper, report that.** "The baseline is already efficient" or "tool
execution dominates" are useful results for an infrastructure discussion.

---

## 7. Configuration guide

### How config files fit together

Configs are YAML files. They can `extends:` another file, in which case the child's
values override the parent's, merged key by key. Unknown keys are rejected, so a typo
fails loudly instead of being ignored.

```
local.yaml            hardware: model, vLLM command and flags, CPU layout, price
  └─ baseline.yaml    budgets, repetitions, concurrency for measured runs
       ├─ lightning.yaml          Lightning Studio specifics (local executor, live metrics)
       │    └─ lightning.pinned.yaml, smoke.pinned.yaml   generated per session by `rollout-cost pin`
       └─ (your variants)
synthetic.yaml        scripted model, no GPU   └─ mock-vllm.yaml   real HTTP path against the mock server
sweep.yaml / synthetic-sweep.yaml   sweep definitions (a base config plus variants)
```

### The settings that matter most

| Section | Key | Default (baseline) | Meaning |
| --- | --- | --- | --- |
| `model` | `kind` | `openai` | `openai` (vLLM or any OpenAI-compatible server) or `scripted` (fake model, no server) |
| | `base_url`, `served_name` | `http://127.0.0.1:8000/v1`, `coding-model` | Where the model is served |
| | `repository`, `revision` | Qwen2.5-Coder-7B-Instruct, *pin it* | Recorded in every run. `doctor` fails on `REPLACE_*` placeholders |
| | `request_timeout_s`, `max_request_retries` | 120, 1 | Per-request limit. Only connection errors and 502/503/504 are retried |
| `generation` | `temperature`, `top_p`, `seed` | 0.2, 0.95, 42 | Sampling. Each episode gets its own seed derived from this one |
| | `max_output_tokens_per_call` | 512 | Output cap per reply. A cut-off reply is rejected and uses up a reply |
| `workload` | `repetitions` | 4 | Independent attempts per task (cohort = 6 tasks × 4 = 24 episodes) |
| | `max_steps` | 6 | Model replies per episode |
| | `context_window_tokens` | 8192 | Must equal vLLM's `max-model-len` |
| | `episode_wall_timeout_s` | 300 | Safety backstop. Hits are reported, and comparisons refuse an increase |
| | `max_observation_chars` | 6000 | Tool output is cut to this before it goes to the model |
| `execution` | `max_active_episodes` | 4 | Episode cap (**tunable**) |
| | `max_inflight_llm_requests` | 4 | Request cap (**tunable**) |
| `tools` | `executor` | `docker` | `docker` (isolated containers) or `local` (subprocesses, pinned on Linux) |
| | `worker_cpus` | `[4,5,6,7]` | Fixed CPU budget for tests and grading |
| `server` | `cache_reset` | `restart` | `restart`, `endpoint` (`/reset_prefix_cache`) or `none` (marked exploratory) |
| | `start_command` | `taskset -c 0-3 vllm serve … {args}` | `{args}` is replaced by the `args` mapping below |
| | `args` | `max-model-len: 8192`, `dtype: bfloat16`, … | vLLM flags as `key: value`. `true` gives `--flag`, `false` gives `--no-flag`, `null` drops it. `max-num-seqs`, `max-num-batched-tokens` and `enable-prefix-caching` are **tunable** |
| | `reserved_cpus` | `[0,1,2,3]` | CPUs for vLLM and the runner. Must not overlap `worker_cpus` |
| `telemetry` | `live_metrics_port` | `null` (9400 in Lightning and mock configs) | Turns on the live exporter for Grafana |
| `cost` | `allocation_hourly_rate` | `null` | Your price per hour. `null` means cost is shown as unavailable |
| | `rate_source`, `billing_granularity_seconds` | `null` | Where the price came from, and how billing rounds (e.g. 1, 60, 3600) |

### Tunable versus protected settings

Only these settings may differ between a baseline and a candidate:

- `execution.max_active_episodes`
- `execution.max_inflight_llm_requests`
- `server.args.max-num-seqs`
- `server.args.max-num-batched-tokens`
- `server.args.enable-prefix-caching`

**Everything else is protected.** If any protected setting differs, `compare` marks the
comparison **INVALID** and lists the differences. Protected settings include budgets,
model revision, prompts, CPU layout, task hashes, harness version and cache procedure.

### Common changes

**Set your GPU price.** In `configs/local.yaml`, or in a pinned config:

```yaml
cost:
  allocation_hourly_rate: 0.80
  rate_source: "Lightning AI L4 on-demand, 2026-10-25"
  billing_granularity_seconds: 1
```

**Try a different concurrency.** Make a small variant file, for example `configs/active-8.yaml`:

```yaml
extends: baseline.yaml          # or lightning.pinned.yaml on Lightning
name: active-8
execution:
  max_active_episodes: 8
  max_inflight_llm_requests: 4
```

Run it, then compare it with the baseline:

```bash
uv run rollout-cost run --config configs/active-8.yaml --split dev
uv run rollout-cost compare --baseline runs/<baseline-id> --candidate runs/<active-8-id>
```

**Test a vLLM setting.** Add a variant to `configs/sweep.yaml`:

```yaml
  - name: active-4-no-prefix-cache
    overrides: {server: {args: {enable-prefix-caching: false}}}
```

A sweep holds at most 4 candidates plus the baseline, and its variants may only change
tunable settings.

**Make the backlog bigger.** Raise `workload.repetitions` so the cohort is at least 3×
the largest episode cap you test. Otherwise most of the run is spent filling up and
draining, and the report warns about it.

**Change the decision rule.** Pass `--acceptance my-rule.yaml` to `paired` or `compare`,
or set `acceptance:` in a sweep file. The keys are listed in
[section 8](#8-reading-the-results). Decide the rule *before* looking at results.

**Use a different model or server.** Change `model.repository`, `model.served_name`,
`server.start_command` and `server.args`, then rerun `doctor --start-server`. Any
OpenAI-compatible streaming server works if it reports token usage. Server metrics are
optional, and missing ones are reported as unavailable.

---

## 8. Reading the results

### A run report (`runs/<id>/report.md`, or the Run detail dashboard)

- **Outcomes:** every episode in exactly one category, with all denominators shown.
- **Allocation time and cost:** the steady-state window, the run-inclusive total, cost
  per valid episode and cost per success.
- **Where episode time went:** the share of time spent waiting for a request slot, in
  model requests, waiting for a tool slot, running tools and doing file operations.
  **Episode slots full** tells you how long the episode cap was actually reached. **Tool
  share** tells you whether overlapping model and tool time could help at all.
- **Warnings:** read them first.

### Pilot checks

| If you see | Do this, before freezing the settings |
| --- | --- |
| 0 successes | Simplify the tasks or use a stronger model. Comparisons are meaningless here |
| Every task solved | Add harder hidden tests |
| Protocol errors over 20%, or many truncated replies | Raise the output cap or adjust the prompt |
| Many `context_exhausted` stops | Lower `max_observation_chars` |
| Tool share under 10% | Concurrency has little room to help. Report it as a finding |
| Wall-time backstop hits | Raise `episode_wall_timeout_s`, so budgets don't depend on scheduling |
| "Server idleness not verified" | The vLLM version doesn't expose the request gauges. Check `/metrics` |

### The acceptance decision (`paired`, the Experiments dashboard)

These defaults are set in advance. A candidate is **ACCEPTED** only if every check
passes:

| Check | Default |
| --- | --- |
| Comparison is valid | Protected settings, cohort and synthetic flag all match |
| Candidate actually differs | Otherwise the result is an **A/A CALIBRATION**, not a decision |
| Enough paired trials | `min_trials: 2` |
| Pooled successes | No drop (`max_pooled_success_drop: 0`) |
| Task-level regressions | No task loses passes (`task_regression_tolerance: 0`, `max_task_regressions: 0`). This catches the case where the baseline solved A and B and the candidate solved B and C: equal totals, still a regression |
| Infrastructure errors / backstop hits | No increase |
| Allocated-window reduction | At least `min_window_reduction_pct: 5.0` pooled, and lower in every trial |
| Above noise | The reduction must exceed the trial-to-trial spread of either side (`require_reduction_above_spread`) |
| Responsiveness (optional) | `max_p90_service_time_increase_pct`, off by default |

Sweeps run one trial per variant, so they relax the trial-count and noise checks and
label the choice "development selection". Validate the chosen variant with `paired`.

Why both "enough trials" and "above noise": in testing, two runs of the **same**
configuration showed a 7.5% "saving" from noise alone. Without these checks, that would
have passed.

---

## 9. Command reference

Run every command as `uv run rollout-cost <command> …`.

| Command | What it does |
| --- | --- |
| `validate-fixtures --config C` | Checks each task: starting code fails its public and hidden tests, the reference fix passes both |
| `doctor --config C [--start-server]` | Checks the config, placeholders, tasks, CPU sets, executor, GPU, live exporter, model server, streaming and usage. `--start-server` launches the server with `server.start_command` for the checks |
| `pin --revision SHA --vllm PATH [--profile lightning\|notebook] [--model-repo REPO] [--rate R --rate-source S] [--rehearsal]` | Writes resolved configs for the selected GPU profile |
| `run --config C [--split dev] [--label L]` | Runs one cohort under one config and writes `runs/<id>/` |
| `sweep --config S` | Runs each variant in a sweep file, compares each to the baseline variant, writes `experiments/<sweep>-<time>/` with `selection.md` and `chosen.yaml` |
| `paired --baseline A --candidate B [--trials N] [--name X] [--split dev]` | Runs alternating paired trials, writes `experiments/X.yaml` and `reports/X.md` with the decision |
| `compare --baseline RUN --candidate RUN [--acceptance F]` | Compares two existing runs: transitions, per-task results, decision |
| `report --run RUN` / `report --experiment F` | Re-renders a report from saved data |
| `export [--db URL] [--reset]` | Loads runs, sweeps and experiments into PostgreSQL for Grafana |
| `mock-server [--port 8000 --policy mixed …]` | Serves a mock vLLM endpoint (SYNTHETIC) |

`paired` uses the `heldout` split by default, and that split is empty. Pass
`--split dev` until held-out tasks exist.

**Helper scripts** (`scripts/`):

- `lightning-setup.sh`: GPU-host setup.
- `observability-native.sh`: Grafana and Prometheus without Docker.
- `rehearse-lightning.sh`: rehearses the Lightning session in Docker.
- `verify_dashboards.py`: runs every dashboard query through Grafana and reports empty
  or failing panels.

**What a run folder contains** (`runs/<id>/`):

| File | Contents |
| --- | --- |
| `run.json` | Manifest: config, protected settings and their fingerprint, task and grader hashes, harness version, host, GPU, server command, timeline |
| `events.jsonl` | Every event with a per-episode sequence number, monotonic time and UTC time |
| `episodes.jsonl` | One record per episode: outcome fields, timings and tokens |
| `transcripts/` | The exact messages, requests and attempts for each episode |
| `workspaces/` | The final code of each episode |
| `summary.json`, `report.md` | Computed results |
| `server.log`, `gpu_samples.jsonl`, `server_metrics.jsonl` | Server output and telemetry samples |

---

## 10. Adding tasks

1. Create `fixtures/tasks/<task_id>/`:
   - `task.yaml`, with `id`, `family`, `title`, `description`, `initial_files` and
     `graded_paths`;
   - a `workspace/` containing `src/`, `tests/test_public.py` and a `pytest.ini` with
     `pythonpath = src`.
2. Create `graders/<task_id>/`:
   - `hidden/test_hidden.py`;
   - `reference/src/...` with the fixed files.
3. Add the task id to one split in `fixtures/splits.yaml`. A held-out task must use a
   genuinely different bug, not renamed variables.
4. Run `uv run rollout-cost validate-fixtures --config configs/synthetic.yaml --split <split>`.

Task and grader contents are hashed into every run. Changing a task invalidates
comparisons with older runs, by design.

---

## 11. Testing and verification

```bash
uv run pytest                                                              # 2-3 minutes, no GPU or Docker needed
RC_TEST_PG=postgresql://rollout:rollout@127.0.0.1:55432/rollout uv run pytest   # adds the PostgreSQL integration tests
uv run python scripts/verify_dashboards.py                                 # every dashboard panel through Grafana
scripts/rehearse-lightning.sh                                              # the Lightning session in an Ubuntu container
```

The tests cover:

- **Accounting:** billing rounding, unknown prices, zero successes.
- **Outcome logic:** every combination of the three outcome fields.
- **Protocol:** truncated and malformed replies.
- **Workspace safety:** path escapes and symlinks.
- **Admission:** the FIFO semaphore and cancellation.
- **Timeouts:** cancelling an in-flight request, and the difference between a
  request-timeout and a wall-time stop.
- **Context:** running out of context.
- **Grading:** editing public tests doesn't count as a pass.
- **Acceptance:** task swaps hidden by equal totals; A/A detection.
- **Mock server:** streaming, retries, context overflow, client-disconnect aborts.
- **Live exporter:** the metrics it serves mid-run.
- **Compatibility checks:** dashboards, exporter, schema, mock server, pinned configs
  and run portability all agree with one shared definition (`src/rollout_cost/contract.py`).

| Environment | Verified |
| --- | --- |
| Windows laptop with Docker Desktop | Full suite, Compose stack, all dashboards |
| Ubuntu container without Docker (stand-in for a Lightning Studio) | Setup, `doctor`, runs, native observability stack, CPU pinning |
| Linux to Windows | Run folders produced on Linux export and load on Windows unchanged |

---

## 12. Project layout

```text
README.md                  this file
pyproject.toml             package, CLI entry point, optional "dashboard" extra
configs/                   local, baseline, sweep, lightning, mock-vllm, synthetic (see section 7)
fixtures/                  six tasks, splits.yaml, warm-up prompts
graders/                   hidden tests and reference fixes (never visible to the agent)
src/rollout_cost/
  cli.py                   all commands
  runner.py                run lifecycle: manifest, server prep, FIFO admission, grading, drain
  harness.py               one episode: action protocol, budgets, timeouts, cancellation
  admission.py             FIFO request semaphore
  tools.py                 safe file actions, pinned tool pool, Docker and local executors
  grading.py               hidden-test grading on a clean copy
  model_client.py          streaming OpenAI-compatible client and scripted fake model
  mock_server.py           mock vLLM server
  server.py                server restart, cache reset, warm-up, idle checks
  telemetry.py             event trace, nvidia-smi and vLLM metric sampling
  live.py                  live Prometheus exporter
  spans.py                 events turned into timeline spans and concurrency series
  export.py, schema.sql    PostgreSQL export for Grafana
  analysis.py, cost.py     summaries, comparisons, acceptance rule, cost accounting
  report.py                Markdown reports
  contract.py              names and versions shared by exporter, schema and dashboards
  pinning.py               session config generation
observability/             Docker Compose stack, Prometheus and Grafana provisioning, dashboard generator
scripts/                   Lightning setup, native observability, rehearsal, dashboard verification
docker/                    test-runner image
docs/DESIGN_DECISIONS.md   why each measurement and fairness rule is the way it is
docs/LIGHTNING_RUNBOOK.md  the GPU session, step by step
tests/                     unit, end-to-end, compatibility and integration tests
runs/, experiments/, reports/   generated output (runs/ and experiment folders are git-ignored)
```

---

## 13. Status, limitations and scope

| Works today (tested without a GPU) | Not done yet |
| --- | --- |
| The six tasks with hidden tests; scripted and mock-vLLM runs end to end; separate episode and request caps; fixed, pinned tool CPU pool; timeouts with cancellation and drain checks; truncation and context handling; outcome fields; cost accounting with billing rounding; sweeps, paired trials, transition tables and acceptance with an A/A noise check; live exporter; PostgreSQL export; four Grafana dashboards; scripted Lightning setup and rehearsal | **A useful GPU task solve rate and validated cost savings** (initial T4 runs completed with zero solves); the Lightning Port Viewer; held-out tasks (the `heldout` split is empty); vLLM-setting ablations; replaying recorded workloads to isolate serving performance |

**Limitations to keep in mind**

- Six small tasks are a demonstration, not a benchmark. Results will be specific to one
  model, one GPU and these tasks.
- The local executor (used when Docker isn't available) doesn't isolate generated code.
  Only use it with your own fixtures.
- The same seed doesn't guarantee identical model outputs when batching differs. That's
  why outcomes are always re-measured rather than assumed unchanged.
- Saving GPU time saves money only if the allocation is released sooner, or you rent
  less. With hourly billing, or hardware you already own, the gain may be capacity rather
  than a smaller bill.

**Out of scope:** multi-GPU or multi-node serving, model routing or quantization
comparisons, adaptive scheduling during a run, and training. The tool generates and
measures agent trajectories, but it doesn't update model weights.

**Further reading:**

- [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md): timeout boundaries, the cache
  procedure, the CPU budget, the acceptance rule, observability design, and the
  compatibility matrix.
- [docs/LIGHTNING_RUNBOOK.md](docs/LIGHTNING_RUNBOOK.md): the GPU session.
- [observability/README.md](observability/README.md): dashboards in depth.
