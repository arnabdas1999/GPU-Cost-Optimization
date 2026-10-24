# Rollout Cost Lab

> For this model, GPU and workload, which serving configuration completes the same set of
> multi-step coding-agent episodes at lower allocated-GPU cost, without worse task outcomes?

A command-line tool that runs a fixed cohort of agent episodes (model → tool → feedback →
model → final artifact) against a self-hosted vLLM server. It records where episode time
goes, grades each final workspace with hidden tests, and compares serving configurations
on allocated GPU time and outcomes.

**Status: early prototype. No GPU measurements yet, so no performance claim is made.**
The harness, grading, accounting and reporting run end to end with a scripted model on a
CPU-only machine. The next step is a real baseline on a single rented GPU.

## What it measures

- **Allocated GPU time and cost.** Allocation time multiplied by the hourly rate. Costs
  are never summed over overlapping episodes or scaled by utilization. An unknown price
  is reported as unavailable, never as zero.
- **Outcomes.** `stop_reason`, `infrastructure_status` and `grader_outcome` are recorded
  separately. Every submitted episode appears in exactly one category: success, failed
  after finish, failed with budget exhausted, or infrastructure error.
- **Time attribution.** Shares of time spent waiting for a model-request slot, in model
  requests, waiting for a tool slot, and running tools. Also latency percentiles, whether
  the episode cap stayed full, prefix-cache reuse, and GPU and server samples.
- **Comparisons.** Baseline and candidate must match on every protected setting. The
  comparison includes a task-level transition table and a predeclared acceptance rule
  with a noise-floor check.

vLLM does the model serving. This repository provides the workload, the admission
controls (separate caps on active episodes and in-flight model requests, plus a fixed
pinned-CPU tool pool), the measurements and the experiment logic.

## Quick start without a GPU (synthetic)

```bash
uv sync
uv run rollout-cost validate-fixtures --config configs/synthetic.yaml   # starting code fails, reference passes
uv run rollout-cost run --config configs/synthetic.yaml --split dev     # scripted model, local executor
uv run rollout-cost sweep --config configs/synthetic-sweep.yaml         # bounded sweep + selection
uv run pytest
```

Synthetic runs are labeled **SYNTHETIC** throughout. They verify the software, not GPU
efficiency.

## On the GPU host

1. Build the tool image:
   `docker build -t rollout-cost-tools:py312 -f docker/tool-runner.Dockerfile docker`.
2. In `configs/local.yaml`, pin the model and tokenizer revisions, CPU sets, price and
   billing granularity. Check every `server.args` flag against the CLI reference for the
   vLLM release you pin.
3. Run `uv run rollout-cost doctor --config configs/baseline.yaml`. It must pass. It
   checks for leftover placeholders, the Docker image, CPU sets, the served model, that
   streaming reports usage, and which metrics are available.
4. **Pilot.** Run one baseline and confirm the budgets allow real attempts. Check that
   tool time is a meaningful share of episode time and that protocol errors are rare.
   Then freeze the configs.
5. **Noise floor.** Run
   `uv run rollout-cost paired --baseline configs/baseline.yaml --candidate configs/baseline.yaml --split dev --trials 2 --name aa`.
6. **Sweep.** Run `uv run rollout-cost sweep --config configs/sweep.yaml`. It runs at most
   4 candidates on the dev split and writes `chosen.yaml`.
7. **Validate.** Run
   `uv run rollout-cost paired --baseline configs/baseline.yaml --candidate experiments/<sweep>/chosen.yaml --split heldout --trials 3`.
   Run order alternates between trials.
8. Export `runs/`, `experiments/` and `reports/`, then **release the allocation**.

Every measured run first restarts the server (`server.cache_reset: restart`), warms it up
with prompts that don't share the task prefix, and refuses to close until the server
reports no running or waiting requests.

## Repository layout

```text
configs/      local (hardware/server/price), baseline, sweep, synthetic
fixtures/     six original tasks (retry, config, parsing families) and splits.yaml
graders/      hidden tests and reference fixes; never visible to the agent
src/rollout_cost/
  runner.py       cohort lifecycle: manifest, server prep, FIFO admission, grading, drain
  harness.py      one episode: JSON action protocol, budgets, timeouts, cancellation
  admission.py    FIFO request semaphore
  tools.py        path-safe file actions, pinned tool pool, Docker/local executors
  grading.py      pristine-copy grading with hidden tests
  model_client.py streaming OpenAI-compatible client, scripted fake
  server.py       restart/reset, warm-up, idle checks
  telemetry.py    JSONL events, nvidia-smi and vLLM metrics sampling
  cost.py analysis.py report.py cli.py
docs/DESIGN_DECISIONS.md   timeout boundaries, cache procedure, CPU budget, acceptance rule
```

Each run directory holds:

- `run.json`: the manifest, including protected settings, hashes, host/GPU, server
  command and timeline.
- `events.jsonl`: the per-episode event sequence.
- `episodes.jsonl`
- `transcripts/`: the exact messages and requests for each episode.
- `workspaces/`: final state.
- `summary.json` and `report.md`.

## What works, what is preliminary, what is not built

| Works (tested without a GPU) | Not yet done |
| --- | --- |
| Six validated fixtures; scripted end-to-end runs; FIFO admission with independent episode and request caps; fixed tool CPU pool; wall-time and per-call timeouts with cancellation; truncation and context handling; separate outcome fields; cost accounting with billing rounding; sweep, paired trials, transition tables and acceptance with an A/A noise check | Any measurement on a real GPU; Docker executor run against a live daemon; vLLM flag and metric names verified for a pinned release; held-out tasks (the `heldout` split is empty); server-setting ablations; performance replay |

Out of scope for this prototype: multi-GPU or multi-node serving, model routing or
quantization comparisons, online adaptive scheduling, and policy updates. It generates and
measures trajectories but does not train on them. Results will be specific to one model,
one GPU and these small tasks.
