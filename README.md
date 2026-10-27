# Rollout Cost Lab

**Find the cheapest way to run AI coding agents on your own GPU without making them worse
at their job.**

[![Live dashboard](https://img.shields.io/badge/Live%20dashboard-Grafana-orange)](https://rollout-cost-dashboard.onrender.com)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arnabdas1999/GPU-Cost-Optimization/blob/main/notebooks/gpu_smoke_test.ipynb)

---

## What this project is for

Teams that run AI agents in bulk on self-hosted open models (vLLM on rented GPUs) pay by
the GPU-hour. Typical cases:

- agent evaluations;
- benchmark suites;
- generating rollouts for post-training.

How many agents you run at once, and how the model server is tuned, change how long the
same batch takes, and so what it costs. Get it wrong and you either leave the GPU idle
or make agents queue for it.

**Rollout Cost Lab answers that with measurements, not guesses.** It runs a fixed batch
of real multi-step coding-agent tasks against your model server, under two or more
serving configurations. For each configuration it:

1. **measures the rented GPU time** for the whole batch and converts it to dollars;
2. **re-grades every task with hidden tests**, so a "cheaper" setting that solves fewer
   tasks is caught;
3. **shows where the time went** for every agent: waiting for the GPU, generating,
   waiting for a CPU, running tests;
4. **makes a decision you set the rules for in advance**, such as ACCEPTED, NOT ACCEPTED
   or A/A CALIBRATION, with a noise check so a lucky run doesn't look like a saving;
5. **publishes the evidence** to a shareable Grafana dashboard.

**Example question it settles.** *"We run 4 agents at a time on one GPU. Would 8 finish
the same batch sooner, and therefore cheaper, without solving fewer tasks?"* The lab runs
the identical batch both ways, alternating the order, and reports:

- the cost difference;
- the per-task pass changes;
- the trial-to-trial noise;
- a verdict.

Other questions it's built for:

- Does turning on vLLM's prefix caching pay off for multi-turn agents?
- Is the bottleneck the GPU or the tool execution?
- What does one solved task cost on this GPU?

### See it

**Live dashboard: https://rollout-cost-dashboard.onrender.com**

Public and read-only. It's on a free tier and sleeps when idle, so the first load can
take about a minute.

| Dashboard | What you'll find |
| --- | --- |
| **Runs** | Every imported run: real T4 runs from Colab and synthetic test runs, clearly labeled. Click any run id |
| **Run detail** | An **episode timeline**: one row per agent, coloured by what it was doing every second. Also concurrency against the caps, where the time went, outcomes per task, GPU and server samples, and the cost basis |
| **Experiments and sweeps** | The decision for a comparison, cost and successes per trial, every acceptance check, and which tasks flipped between pass and fail |

### Where it stands (October 2026)

The pipeline works end to end on a real GPU: a Colab T4 running
`Qwen2.5-Coder-7B-Instruct-AWQ` under vLLM, with the results on the public dashboard.
Model quality is still being calibrated, so **no cost saving has been claimed yet.**

| Real T4 runs (all on the dashboard) | Result |
| --- | --- |
| First smoke tests (6 tasks each, literal text edits) | 0 of 6 solved. The model got stuck in edit loops or broke the code |
| One-task preflight after switching to whole-file edits | Passed (twice) |
| Latest smoke test (6 tasks, whole-file edits) | **2 of 6 solved**, 574.8 s of measured GPU time. That's about **$0.086** at the documented T4 reference rate of $0.54/h ([estimate basis](docs/COST_ESTIMATES.md)). It's a valuation, not a bill |

Next: raise the solve rate, run the 24-episode pilot, measure the A/A noise floor, then
the concurrency sweep and paired validation. The workflow is in
[section 6](#6-finding-a-cheaper-configuration-the-workflow).

### How the pieces fit together

```
 Colab / Kaggle T4 notebook          your laptop                          free hosting
 ┌─────────────────────────┐   download   ┌────────────────────┐  export   ┌──────────────────────┐
 │ vLLM + Qwen 7B (GPU)    │  results.tgz │ rollout-cost export│ ───────►  │ Neon PostgreSQL      │
 │ rollout-cost run        │ ───────────► │  (reports, compare)│           │  rollout-history     │
 │ preflight, smoke, pilot │              └────────────────────┘           └──────────┬───────────┘
 └─────────────────────────┘                                              read-only   │
                                                                                      ▼
                                                                          ┌──────────────────────┐
                                     anyone with the link ──────────────► │ Grafana on Render    │
                                                                          │ (state in a second   │
                                                                          │  Neon project)       │
                                                                          └──────────────────────┘
```

There are three steps to publish a new result:

1. Run the [Colab notebook](https://colab.research.google.com/github/arnabdas1999/GPU-Cost-Optimization/blob/main/notebooks/gpu_smoke_test.ipynb)
   and download `notebook-results.tgz`.
2. On your laptop, run `rollout-cost export` into the Neon history database.
3. Refresh the dashboard.

The public site's URL never changes. Setup is in [docs/FREE_HOSTING.md](docs/FREE_HOSTING.md).

---

## Contents

1. [Why measuring this is harder than it looks](#1-why-measuring-this-is-harder-than-it-looks)
2. [How it works](#2-how-it-works)
3. [Quick start (no GPU, about 5 minutes)](#3-quick-start-no-gpu-about-5-minutes)
4. [Dashboards: local, live and hosted](#4-dashboards-local-live-and-hosted)
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

## 1. Why measuring this is harder than it looks

An AI coding agent doesn't send one request and stop. It works in a loop:

```
ask the model → model picks an action → run it (read a file, edit, run tests) → feed the result back → ask again
```

While one agent waits for its tests, the GPU could be serving another. But too many
agents at once queue for the GPU and compete for memory and CPU. The usual serving
benchmark, requests per second, can't tell you whether a whole batch of agent tasks
finishes sooner or more cheaply.

There are also several easy ways to fool yourself:

- A setting that looks faster may just solve fewer tasks.
- Two runs of the same setting differ by several percent.
- A warm cache from the previous run can make the next one look better.
- Extra concurrency can quietly borrow more CPU.

So the project measures the thing you pay for, **allocated GPU time × hourly price**,
for a fixed batch, re-grades every task, and controls for each of these effects.

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
   compare / sweep / paired             rollout-cost export ──► PostgreSQL (local or Neon) ──► Grafana (local or Render)
   (acceptance decision)                live exporter :9400 ──► Prometheus ──► Grafana (live view)
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

The harness runs the action and feeds the result back. There are two edit modes
(`workload.edit_mode`):

- **`literal`** (baseline): small exact-text replacements with `edit_file`.
- **`whole_file`** (notebook profile): the model writes the complete module, and later
  corrections replace the whole file too. Replies are constrained to valid actions with
  vLLM's JSON-schema support. This removed the edit loops seen in the first T4 runs.
  Python syntax errors are reported back without running the code. The constraint
  controls the *shape* of a reply, not whether its code is correct; that's still
  decided by the hidden tests.

Each episode has a fixed budget:

| Budget | Baseline profile | Notebook (T4) profile |
| --- | --- | --- |
| Model replies | 6 | 12 |
| Output tokens per reply | 512 | 1,024 |
| Context window | 8,192 | 8,192 |
| Wall-clock backstop | 300 s | 300 s |

A malformed or cut-off reply still uses up a reply. When the budget runs out, the final
code is graded as it stands.

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
utilization. **An unknown price shows as "unavailable", never as $0.**

There are two kinds of rate, and reports label which one was used:

- **`recorded`:** a price you actually pay, such as a cloud on-demand rate.
- **`estimated_equivalent`:** a documented reference valuation for hardware with no bill,
  such as Colab. The notebook profile values a T4 at **$0.539999/h**: Google Cloud's
  us-central1 rate for one T4 plus an n1-standard-4 host, checked 2026-10-26
  ([details](docs/COST_ESTIMATES.md)).

Completed runs can be revalued without rerunning the GPU. Use
`report --run … --cost-config configs/notebook.yaml`, or pass `--cost-config` to
`export`.

### Fair comparisons

A comparison is only valid if everything except a short list of tunable settings is
identical:

- the same tasks, repetitions and budgets;
- the same model revision, prompts and harness code (hashed);
- the same CPU layout, tool pool and grader;
- the same cache-reset procedure.

Before each measured run, the server is restarted (clearing vLLM's prefix cache) and
warmed up identically:

- Restart refuses an endpoint that is still occupied after the stop.
- Readiness requires the launched process to stay alive.
- The server must be verified idle before and after the cohort. If it can't be verified
  beforehand, preparation stops before any measurement. If the check after the cohort
  fails, the artifacts are kept for inspection but acceptance is blocked.
- Each run must be complete, with exactly one terminal record per declared episode, and
  its summary must agree with those records.

A configuration is **accepted** only if it passes a rule you set in advance (see
[section 8](#8-reading-the-results)). That rule includes a noise check, because
identical configurations can differ by several percent between runs.

The rationale for each of these choices is in
[docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md).

---

## 3. Quick start (no GPU, about 5 minutes)

**Requirements:** [uv](https://docs.astral.sh/uv/) and Git. Python is installed by uv if
needed. Docker is optional; you need it for the local dashboards.

```bash
git clone https://github.com/arnabdas1999/GPU-Cost-Optimization.git rollout-cost && cd rollout-cost
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

Synthetic runs, scripted or mock, are labeled **SYNTHETIC** everywhere. They show that
the software works, not GPU results.

---

## 4. Dashboards: local, live and hosted

There are four Grafana dashboards over the same data the Markdown reports use:

| Dashboard | Shows |
| --- | --- |
| **Live run** | While a run is in progress: phase, progress, episodes against the cap, request slots, tool slots, the vLLM queue and KV cache, tokens per second, prefix-cache hit rate, time to first token, GPU, and cost so far |
| **Runs** | Every exported run, synthetic or measured, with outcomes, allocated time, cost and cost basis. Click a run id to open its detail |
| **Run detail** | **Episode timeline**: one row per episode, coloured by what it was doing (queued, waiting for the GPU, generating, waiting for a CPU, running tests, grading, outcome). Below it: concurrency against each cap, where the time went, latency, per-task results, GPU and server samples, cost basis, warnings |
| **Experiments and sweeps** | The decision, allocated window, cost and successes per trial or variant, every acceptance check, which episodes flipped between pass and fail, per-task passes |

The history dashboards (the last three) read from PostgreSQL, filled by
`rollout-cost export`. The live dashboard reads from Prometheus while a run is in
progress.

### Hosted and public: Render + Neon (free)

This is the deployment behind https://rollout-cost-dashboard.onrender.com. It needs no
GPU, no Docker and no running laptop.

| Piece | Service | Role |
| --- | --- | --- |
| History database | Neon PostgreSQL, project `rollout-history` | Holds the exported results. Grafana reads it through a **read-only** `grafana_reader` role |
| Grafana state | Neon PostgreSQL, project `grafana-state` | Grafana's own settings and users, kept separate so public queries can't reach them |
| Dashboard site | Render free web service (`render.yaml`, `observability/render/`) | Grafana 13.2.4 with the three history dashboards provisioned from Git. Anonymous viewers, TLS to both databases |

To publish new results, download the results archive from Colab, extract it in the
repository, and export into the history database:

```powershell
# Windows PowerShell; paste the rollout-history OWNER connection string at the hidden prompt.
$secretUrl = Read-Host 'rollout-history owner connection string' -AsSecureString
$env:RC_DB_URL = [System.Net.NetworkCredential]::new('', $secretUrl).Password
uv run --extra dashboard --locked rollout-cost export --cost-config configs/notebook.yaml
Remove-Item Env:RC_DB_URL
```

Refresh the site and the new run appears. Re-exporting updates rows without creating
duplicates. **Never commit or paste the owner connection string anywhere.** Keep it in a
password manager, and give Grafana only the read-only role. The full setup is in
[docs/FREE_HOSTING.md](docs/FREE_HOSTING.md): the Neon projects, the SQL for the reader
role, the Render Blueprint, verification and the free-tier limits.

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

- `rollout-cost export` is safe to rerun; it replaces each run's rows.
- Stop the stack with `docker compose -f observability/compose.yaml down`.
- To change logins or ports, copy `observability/.env.example` to `observability/.env`.

### On a GPU host without Docker (live view)

```bash
bash scripts/observability-native.sh start   # downloads pinned Prometheus + Grafana, no root; live view only
bash scripts/observability-native.sh stop    # stop when done: a running server keeps a Lightning Studio awake
```

More detail is in [observability/README.md](observability/README.md).

---

## 5. Running on a real GPU

### Colab / Kaggle (start here)

Open [the GPU notebook](https://colab.research.google.com/github/arnabdas1999/GPU-Cost-Optimization/blob/main/notebooks/gpu_smoke_test.ipynb),
choose **Runtime > Change runtime type > GPU** (a T4), and **Run all**. On Kaggle, import
`notebooks/gpu_smoke_test.ipynb`, select **T4** and enable **Internet**.

The notebook:

1. checks the GPU, and rejects CPU or P100 sessions before any large download;
2. installs the project and an exact vLLM release (default `0.31.0`);
3. resolves and downloads `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`;
4. writes pinned configs, validates the tasks and runs `doctor`;
5. runs a **one-task preflight**, then a **six-episode smoke test**, one at a time;
6. packages everything into `notebook-results.tgz` for download.

The longer pilot and A/A runs are off by default; enable them after the smoke test looks
healthy. If a cell fails, the archive is still built, so you can inspect what happened.
The [notebook runbook](docs/NOTEBOOK_RUNBOOK.md) covers every step, the profile defaults
and troubleshooting.

**The notebook profile** (`configs/notebook.yaml`):

- one T4, using AWQ 4-bit weights with FP16 activations;
- an 8,192-token context;
- 12 replies of up to 1,024 tokens each;
- greedy sampling;
- whole-file edits with JSON-schema-constrained actions;
- 2 server request slots;
- local tool execution;
- T4 reference pricing.

### Lightning AI (alternative)

The whole session is scripted and has been rehearsed in a plain Ubuntu container. Follow
[docs/LIGHTNING_RUNBOOK.md](docs/LIGHTNING_RUNBOOK.md).

```bash
# On the CPU Studio, set VLLM_VERSION to an exact release first:
bash scripts/lightning-setup.sh --prepare
# After switching to the L4 GPU, restore VLLM_VERSION and set RATE and RATE_SOURCE:
bash scripts/lightning-setup.sh
uv run rollout-cost run --config configs/smoke.pinned.yaml --split dev       # 6 episodes, one at a time
uv run rollout-cost run --config configs/lightning.pinned.yaml --split dev   # pilot baseline (24 episodes)
```

`lightning-setup.sh` does the following:

- checks the GPU and CUDA;
- installs and verifies your exact vLLM version, in its own version-specific environment;
- resolves the model's exact commit and downloads it;
- splits the CPUs between vLLM and the tool workers;
- writes pinned configs;
- validates the tasks;
- runs `doctor --start-server`.

`--prepare` does the downloads on a CPU machine first. As of October 26, 2026, Lightning
reported Studio access unavailable during an
[incident investigation](https://lightning.ai/security-incident/); use the notebook
route first. To rehearse the Lightning session locally (needs Docker), run
`bash scripts/rehearse-lightning.sh`.

### Any other Linux GPU host

1. Clone the repo, then run `uv sync --extra dashboard` and install vLLM.
2. Build the test image:
   `docker build -t rollout-cost-tools:py312 -f docker/tool-runner.Dockerfile docker`.
3. Edit `configs/local.yaml`:
   - the model revision;
   - `server.start_command` and `server.args`, checking each flag against
     `vllm serve --help`;
   - `server.reserved_cpus` and `tools.worker_cpus`;
   - the price.

   Or run `bash scripts/lightning-setup.sh` on the host.
4. Run `uv run rollout-cost doctor --config configs/baseline.yaml --start-server` and fix
   anything it flags.
5. Follow the workflow below.

**Model and hardware assumptions:** the baseline profile uses
`Qwen/Qwen2.5-Coder-7B-Instruct` in BF16, on one GPU with 24–48 GB. A 16 GB T4 uses the
AWQ build in the notebook profile instead. Comparisons are only valid within one model,
GPU and profile.

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
| **6. Publish** | `export` (local or Neon), then the dashboards, plus `reports/` | The decision, and the evidence behind it, at a shareable link |

**What a sweep varies by default:**

| Profile | Variants (episode cap / request cap) | Baseline |
| --- | --- | --- |
| Baseline / Lightning (`configs/sweep.yaml`) | `active-2` 2/2, `active-4` 4/4, `active-8` 8/4 | `active-4` |
| Notebook T4 (`notebook-sweep.pinned.yaml`) | `active-1` 1/1, `active-2` 2/2, `active-4` 4/2 | `active-2` |

**Ideas to test next**, one change at a time:

- a vLLM scheduler limit: `server.args.max-num-seqs` or `max-num-batched-tokens`;
- prefix caching on or off: `server.args.enable-prefix-caching`. Multi-turn agents reuse
  their own conversation prefixes heavily;
- a different ratio of request cap to episode cap.

**If nothing is cheaper, report that.** "The baseline is already efficient" or "tool
execution dominates" are useful infrastructure results.

---

## 7. Configuration guide

### How config files fit together

Configs are YAML files. They can `extends:` another file, in which case the child's
values override the parent's, merged key by key. Unknown keys are rejected, so a typo
fails loudly instead of being ignored.

```
local.yaml            hardware: model, vLLM command and flags, CPU layout, price
  └─ baseline.yaml    budgets, repetitions, concurrency for measured runs
       ├─ notebook.yaml           Colab/Kaggle T4: AWQ model, whole-file edits, T4 reference price
       │    └─ notebook.pinned.yaml, notebook-smoke.pinned.yaml, notebook-sweep.pinned.yaml   (generated)
       ├─ lightning.yaml          Lightning Studio specifics (local executor, live metrics)
       │    └─ lightning.pinned.yaml, smoke.pinned.yaml, sweep.pinned.yaml                     (generated)
       └─ (your variants)
synthetic.yaml        scripted model, no GPU   └─ mock-vllm.yaml   real HTTP path against the mock server
sweep.yaml / synthetic-sweep.yaml   sweep definitions (a base config plus variants)
```

`*.pinned.yaml` files are written per GPU session by `rollout-cost pin`, which the setup
scripts call. They record the resolved model commit, vLLM path, CPU split and price.

### The settings that matter most

| Section | Key | Default (baseline) | Meaning |
| --- | --- | --- | --- |
| `model` | `kind` | `openai` | `openai` (vLLM or any OpenAI-compatible server) or `scripted` (fake model, no server) |
| | `base_url`, `served_name` | `http://127.0.0.1:8000/v1`, `coding-model` | Where the model is served |
| | `repository`, `revision` | Qwen2.5-Coder-7B-Instruct, *pin it* | Recorded in every run. `doctor` fails on `REPLACE_*` placeholders |
| | `request_timeout_s`, `max_request_retries` | 120, 1 | Per-request limit. Only connection errors and 502/503/504 are retried |
| `generation` | `temperature`, `top_p`, `seed` | 0.2, 0.95, 42 | Sampling (the notebook profile uses greedy: 0 and 1.0) |
| | `max_output_tokens_per_call` | 512 | Output cap per reply. A cut-off reply is rejected and uses up a reply |
| `workload` | `repetitions` | 4 | Independent attempts per task (cohort = 6 tasks × 4 = 24 episodes) |
| | `max_steps` | 6 | Model replies per episode |
| | `edit_mode` | `literal` | `literal` (exact-text `edit_file`) or `whole_file` (complete-file writes, schema-constrained) |
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
| `cost` | `allocation_hourly_rate` | `null` (notebook: 0.539999) | Price per hour. `null` means cost is shown as unavailable |
| | `rate_kind` | `recorded` | `estimated_equivalent` labels a reference valuation ([assumptions](docs/COST_ESTIMATES.md)) |
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
model revision, prompts, edit mode, CPU layout, task hashes, harness version and cache
procedure.

### Common changes

**Set your GPU price.** In `configs/local.yaml`, or in a pinned config:

```yaml
cost:
  allocation_hourly_rate: 0.80
  rate_kind: recorded
  rate_source: "Lightning AI L4 on-demand, 2026-10-25"
  billing_granularity_seconds: 1
```

**Try a different concurrency.** Make a small variant file, for example `configs/active-8.yaml`:

```yaml
extends: baseline.yaml          # or notebook.pinned.yaml / lightning.pinned.yaml on a GPU session
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

**Test a vLLM setting.** Add a variant to the sweep file:

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

**Revalue past runs.** Run `report --run runs/<id> --cost-config configs/notebook.yaml`.
That writes `report.estimated.md` and leaves the original artifacts untouched. For the
dashboards, use `export --cost-config configs/notebook.yaml`.

**Use a different model or server.** Change `model.repository`, `model.served_name`,
`server.start_command` and `server.args`, then rerun `doctor --start-server`. Any
OpenAI-compatible streaming server works if it reports token usage. Server metrics are
optional, and missing ones are reported as unavailable.

---

## 8. Reading the results

### A run report (`runs/<id>/report.md`, or the Run detail dashboard)

- **Outcomes:** every episode in exactly one category, with all denominators shown.
- **Allocation time and cost:** the steady-state window, the run-inclusive total, cost
  per valid episode and cost per success, with the cost basis (recorded or estimated
  rate, and its source).
- **Where episode time went:** the share of time spent waiting for a request slot, in
  model requests, waiting for a tool slot, running tools and doing file operations.
  **Episode slots full** tells you how long the episode cap was actually reached. **Tool
  share** tells you whether overlapping model and tool time could help at all.
- **Warnings:** read them first.

### Pilot checks

| If you see | Do this, before freezing the settings |
| --- | --- |
| 0 successes | Simplify the tasks, adjust the edit mode or budgets, or use a stronger model. Comparisons are meaningless here |
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
| Comparison is valid | Protected settings, cohort and synthetic flag all match; both runs complete and drain-verified |
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
have passed. At a fixed rate, a dollar reduction is exactly proportional to the
measured-time reduction. A saving at an estimated reference rate shows the time saved;
it doesn't prove a specific cloud bill.

---

## 9. Command reference

Run every command as `uv run rollout-cost <command> …`.

| Command | What it does |
| --- | --- |
| `validate-fixtures --config C` | Checks each task: starting code fails its public and hidden tests, the reference fix passes both |
| `doctor --config C [--start-server]` | Checks the config, placeholders, tasks, CPU sets, executor, GPU, live exporter, model server, streaming, usage and request gauges. `--start-server` launches the server with `server.start_command` for the checks |
| `pin --revision SHA --vllm PATH [--profile lightning\|notebook] [--model-repo REPO] [--rate R --rate-source S] [--rehearsal]` | Writes the pinned configs for a GPU session (called by the setup scripts) |
| `run --config C [--split dev] [--label L]` | Runs one cohort under one config and writes `runs/<id>/` |
| `sweep --config S` | Runs each variant in a sweep file, compares each to the baseline variant, writes `experiments/<sweep>-<time>/` with `selection.md` and `chosen.yaml` |
| `paired --baseline A --candidate B [--trials N] [--name X] [--split dev]` | Runs alternating paired trials, writes `experiments/X.yaml` and `reports/X.md` with the decision |
| `compare --baseline RUN --candidate RUN [--acceptance F]` | Compares two existing runs: transitions, per-task results, decision |
| `report --run RUN [--cost-config C]` / `report --experiment F` | Re-renders a report. With `--cost-config`, writes a revalued `report.estimated.md` |
| `export [--db URL] [--cost-config C] [--runs-dir D] [--experiments-dir D] [--reset]` | Loads runs, sweeps and experiments into PostgreSQL: local Docker, or Neon via `$RC_DB_URL` |
| `mock-server [--port 8000 --policy mixed …]` | Serves a mock vLLM endpoint (SYNTHETIC) |

`paired` uses the `heldout` split by default, and that split is empty. Pass
`--split dev` until held-out tasks exist.

**Helper scripts** (`scripts/`):

- `notebook-setup.sh`: Colab/Kaggle setup, called by the notebook.
- `lightning-setup.sh`: setup for a Lightning or other Linux GPU host.
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
| `summary.json`, `report.md` (`report.estimated.md`) | Computed results, and an optional revalued report |
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
bash scripts/rehearse-lightning.sh                                         # the Lightning session in an Ubuntu container
```

The tests cover:

- **Accounting:** billing rounding, unknown prices, zero successes, estimated rates and
  revaluation.
- **Outcome logic:** every combination of the three outcome fields.
- **Protocol:** truncated and malformed replies; whole-file mode and the JSON schemas.
- **Workspace safety:** path escapes and symlinks.
- **Admission:** the FIFO semaphore and cancellation.
- **Timeouts:** cancelling an in-flight request, and the difference between a
  request-timeout and a wall-time stop.
- **Context:** running out of context.
- **Grading:** editing public tests doesn't count as a pass.
- **Acceptance:** task swaps hidden by equal totals; A/A detection; run-validity checks.
- **Server lifecycle:** restart and readiness.
- **Notebook helpers:** GPU checks and archiving.
- **Mock server:** streaming, retries, context overflow, client-disconnect aborts.
- **Live exporter:** the metrics it serves mid-run.
- **Compatibility checks:** dashboards, exporter, schema, mock server, pinned configs
  and run portability all agree with one shared definition (`src/rollout_cost/contract.py`).

| Environment | Verified |
| --- | --- |
| Colab T4 (real GPU) | Notebook setup, preflight and smoke runs with vLLM; results exported to Neon and visible on the Render dashboard |
| Windows laptop with Docker Desktop | Full suite, Compose stack, all dashboards |
| Ubuntu container without Docker (stand-in for a Lightning Studio) | Setup, `doctor`, runs, native observability stack, CPU pinning |
| Linux to Windows | Run folders produced on Linux export and load on Windows unchanged |

---

## 12. Project layout

```text
README.md                  this file
pyproject.toml             package, CLI entry point, optional "dashboard" extra
render.yaml                Render Blueprint for the public Grafana site
configs/                   local, baseline, notebook, lightning, sweep, mock-vllm, synthetic (see section 7)
notebooks/                 gpu_smoke_test.ipynb: the Colab/Kaggle GPU session
fixtures/                  six tasks, splits.yaml, warm-up prompts
graders/                   hidden tests and reference fixes (never visible to the agent)
src/rollout_cost/
  cli.py                   all commands
  runner.py                run lifecycle: manifest, server prep, FIFO admission, grading, drain
  harness.py               one episode: action protocol, edit modes, budgets, timeouts, cancellation
  protocol.py              JSON actions and the per-request JSON schemas
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
  analysis.py, cost.py     summaries, comparisons, acceptance rule, cost accounting and valuation
  report.py                Markdown reports
  contract.py              names and versions shared by exporter, schema and dashboards
  pinning.py               session config generation (lightning and notebook profiles)
  notebook.py              Colab/Kaggle helpers: GPU check, logged commands, result archive
observability/             Docker Compose stack, Prometheus and Grafana provisioning, dashboard generator
observability/render/      Grafana image, config and provisioning for the Render deployment
scripts/                   notebook/Lightning setup, native observability, rehearsal, dashboard verification
docker/                    test-runner image
docs/                      NOTEBOOK_RUNBOOK, FREE_HOSTING, COST_ESTIMATES, LIGHTNING_RUNBOOK, DESIGN_DECISIONS
tests/                     unit, end-to-end, compatibility and integration tests
runs/, experiments/, reports/   generated output (runs/ and experiment folders are git-ignored)
```

---

## 13. Status, limitations and scope

| Works today | Not done yet |
| --- | --- |
| The six tasks with hidden tests; real T4 runs through the Colab notebook; scripted and mock-vLLM runs; separate episode and request caps; fixed, pinned tool CPU pool; timeouts with cancellation and drain checks; truncation and context handling; literal and whole-file edit modes; outcome fields; cost accounting with recorded or estimated rates; sweeps, paired trials, transition tables and acceptance with an A/A noise check; live exporter; PostgreSQL export; four Grafana dashboards, three of them hosted publicly on Render + Neon | **A useful solve rate and a validated cost saving.** The latest T4 smoke test solved 2 of 6; the pilot, A/A, sweep and paired validation haven't been run yet. Also: held-out tasks (the `heldout` split is empty); vLLM-setting ablations; replaying recorded workloads to isolate serving performance |

**Limitations to keep in mind**

- Six small tasks are a demonstration, not a benchmark. Results will be specific to one
  model, one GPU and these tasks.
- Colab and Kaggle runtimes are shared, and their CPU, clocks and neighbours vary.
  Measure the A/A noise floor there before trusting a difference.
- The estimated T4 rate values the time; it isn't anyone's invoice.
- The local executor (used when Docker isn't available) doesn't isolate generated code.
  Only use it with your own fixtures, in a fresh runtime.
- The same seed doesn't guarantee identical model outputs when batching differs. That's
  why outcomes are always re-measured rather than assumed unchanged.
- Saving GPU time saves money only if the allocation is released sooner, or you rent
  less. With hourly billing, or hardware you already own, the gain may be capacity rather
  than a smaller bill.
- The public dashboard runs on free tiers. Render sleeps when idle, and Neon has monthly
  compute and storage limits. Anything exported to the history database is public.

**Out of scope:** multi-GPU or multi-node serving, model routing or quantization
comparisons, adaptive scheduling during a run, and training. The tool generates and
measures agent trajectories, but it doesn't update model weights.

**Further reading:**

- [docs/NOTEBOOK_RUNBOOK.md](docs/NOTEBOOK_RUNBOOK.md): the Colab/Kaggle GPU session.
- [docs/FREE_HOSTING.md](docs/FREE_HOSTING.md): the public dashboard on Render + Neon.
- [docs/COST_ESTIMATES.md](docs/COST_ESTIMATES.md): the T4 reference rate and revaluing
  runs.
- [docs/LIGHTNING_RUNBOOK.md](docs/LIGHTNING_RUNBOOK.md): the Lightning L4 session.
- [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md): timeout boundaries, the cache
  procedure, the CPU budget, the acceptance rule, observability design, and the
  compatibility matrix.
- [observability/README.md](observability/README.md): dashboards in depth.
