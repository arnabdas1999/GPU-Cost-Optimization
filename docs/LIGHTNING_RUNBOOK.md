# Lightning AI session runbook

This is one GPU session on a Lightning AI Studio with a single L4 (24 GB). The workflow
has a CPU-only Linux rehearsal using the mock vLLM server. Rerun it with
`bash scripts/rehearse-lightning.sh`; it needs Docker on the laptop. Setup regression
tests also check CPU preparation, version synchronization and model selection without
downloading vLLM. A rehearsal does not establish real-GPU performance or compatibility.

**Before you switch the GPU on**
- Push the repository and clone it into the Studio's persistent storage
  (`/teamspace/studios/this_studio`).
- Pick an exact vLLM release (`VLLM_VERSION`), at least two weeks old.
- Note the L4 on-demand price shown in the Lightning console (`RATE`). Free credits
  still have a price, and the report needs it.
- Plan the credit budget below, and stop at the first failed stop/go check.

**Prepare on CPU**

Commit and push the new scripts, configs, source modules and tests from the laptop first.
`git commit -am` does not include untracked files: stage the intended files explicitly.
Then, in the CPU Studio:

```bash
cd /teamspace/studios/this_studio
git clone https://github.com/arnabdas1999/GPU-Cost-Optimization.git rollout-cost
cd rollout-cost

read -r -p "Exact vLLM release to install: " VLLM_VERSION
export VLLM_VERSION
bash scripts/lightning-setup.sh --prepare
uv run rollout-cost validate-fixtures --config configs/synthetic.yaml
uv run rollout-cost doctor --config configs/mock-vllm.yaml --start-server
bash scripts/observability-native.sh install  # optional dashboard downloads on CPU
```

`--prepare` does not require a GPU or start a server. It installs the locked project
dependencies on Python 3.12, synchronizes the exact vLLM release, verifies its installed
metadata, records `vllm-packages.txt`, and downloads the selected model. It does not build
Docker images because Studio images may need rebuilding after switching machines.

The default vLLM environment is `~/vllm-venv-$VLLM_VERSION`. An explicit `VLLM_VENV`
override is supported; setup synchronizes and checks its version even if a vLLM binary
already exists. `MODEL_REPO` selects both the downloaded and served model; the default is
`Qwen/Qwen2.5-Coder-7B-Instruct`. Restore the same version and any overrides if switching
machines opens a new terminal.

**After switching to one L4**

```bash
cd /teamspace/studios/this_studio/rollout-cost
# Restore/export VLLM_VERSION and any MODEL_REPO or VLLM_VENV overrides first.
nvidia-smi
read -r -p "Lightning L4 hourly rate in USD: " RATE
export RATE
export RATE_SOURCE="Lightning L4 on-demand, console quote for this session"
bash scripts/lightning-setup.sh
```

Setup verifies CUDA through the installed vLLM environment before downloading uncached
model files or starting the server. It records the CLI help in `vllm-serve-help.txt`.
Docker is preferred when available; otherwise local tools use `taskset` for CPU pinning,
without Python fork callbacks. Local execution still has no container isolation. The
selected executor and applied CPU sets are recorded in the pinned config and run manifest.
`doctor` fails if running/waiting request gauges are missing or invalid. Preparation
also stops before the cohort if the server cannot be verified idle. Linux shutdown
terminates the server's process group, including workers left by an exited API process.

| # | Step | Command | Rough GPU time | Stop/go |
| --- | --- | --- | --- | --- |
| 0 | Prepare the CPU Studio | `bash scripts/lightning-setup.sh --prepare` | 0 | Dependencies and model downloaded; CPU checks pass |
| 1 | Switch to the L4 machine and run setup | `bash scripts/lightning-setup.sh` with the same exported version and the session rate | Startup-dependent; cold setup can add 15–25 min | `doctor` passes. If vLLM rejects a flag, fix `configs/local.yaml` against `vllm-serve-help.txt` and rerun |
| 2 | Live view (optional) | `bash scripts/observability-native.sh start`, then forward port 3000 over SSH | About 1 min after CPU downloads | Live dashboard shows runner and vLLM panels |
| 3 | Smoke: 6 episodes, one at a time | `uv run rollout-cost run --config configs/smoke.pinned.yaml --split dev` | 5–10 min | Read 2 transcripts in `runs/<id>/transcripts/`. Protocol errors under 20%, at least one task solved |
| 4 | Pilot baseline | `uv run rollout-cost run --config configs/lightning.pinned.yaml --split dev` | 8–15 min | Check the report against the table below, then explicitly stage and commit the generated configs using the commands below |
| 5 | A/A noise floor | `uv run rollout-cost paired --baseline configs/lightning.pinned.yaml --candidate configs/lightning.pinned.yaml --split dev --trials 2 --name aa` | 30–50 min | Note the window spread; it's the smallest effect you can claim |
| 6 | Sweep (3 variants) | `uv run rollout-cost sweep --config configs/sweep.pinned.yaml` | 25–45 min | Writes `experiments/<sweep>/chosen.yaml` |
| 7 | Paired validation | `uv run rollout-cost paired --baseline configs/lightning.pinned.yaml --candidate "experiments/REPLACE_WITH_SWEEP_DIRECTORY/chosen.yaml" --split dev --trials 2 --name first-comparison` | 30–50 min | Replace the path with the sweep's printed path. Skip if the sweep kept the baseline |
| 8 | Stop and copy home | Use the stop/archive commands below and download the archive | About 2 min | **Switch the Studio back to CPU or stop it** |

These are planning estimates, not measured GPU results: allow about 2–3.5 GPU-hours for
the complete workflow and use the actual pilot duration to revise that budget. Smoke
plus pilot is a useful first-session stopping point. If the pilot is slower than expected, drop step 7
or use 1 trial in step 5 and label the result exploratory.

**Freeze after the pilot**

```bash
git add configs/lightning.pinned.yaml configs/smoke.pinned.yaml configs/sweep.pinned.yaml
git commit -m "Freeze Lightning pilot settings"
```

Also stage any source, fixture or base-config changes made during calibration. Do not
change those settings between comparison trials. The held-out split is still empty;
the commands here use `--split dev` and their results must be labeled development results.

**Stop and save**

```bash
bash scripts/observability-native.sh stop
tar -czf ../lightning-results.tgz runs experiments reports configs/*.pinned.yaml \
  vllm-serve-help.txt vllm-packages.txt
```

Download `/teamspace/studios/this_studio/lightning-results.tgz`, then switch the Studio
back to CPU or stop it. The CLI stops vLLM after each run; that does not release the GPU
allocation. CLI help and the package inventory are ignored by Git but included in this archive.

**Pilot checks (step 4)**

| If the report shows | Do (before freezing) |
| --- | --- |
| 0 successes | Simplify the tasks or try another model. Don't run comparisons |
| All successes | Add edge cases to the hidden tests. Don't run comparisons |
| Protocol errors over 20%, or truncated replies | Raise `max_output_tokens_per_call` or tighten the system prompt |
| Many `context_exhausted` stops | Lower `max_observation_chars` |
| Tool share of episode time under 10% | Concurrency has little room to help. Report that as a finding |
| Wall-time backstop hits | Raise `episode_wall_timeout_s`. Budgets must not depend on scheduling |
| Idleness before or after the cohort not verified | Stop comparisons and check that `/metrics` exposes both running and waiting request gauges on this vLLM release; these runs cannot pass acceptance |

**Afterwards, on the laptop**

```bash
tar xzf lightning-results.tgz
uv sync --extra dashboard --locked
docker compose -f observability/compose.yaml up -d
uv run rollout-cost export
uv run python scripts/verify_dashboards.py --skip-live
# Grafana: http://127.0.0.1:3000 → Runs → click the run → Run detail; Experiments for the A/A, sweep and paired results
```

**Only a real session can check:**
- That the vLLM flags in `configs/local.yaml` work on the installed release.
- Which metric names that release exposes. Missing ones show as "unavailable", never as 0.
- The model's protocol-error and solve rates.
- Whether the Studio's Port Viewer serves Grafana at the root path or under a sub-path.
  The default Grafana address is loopback for SSH forwarding. For Port Viewer, restart
  with `GRAFANA_ADDR=0.0.0.0` and configure its access controls. For a sub-path, also set
  `GF_SERVER_ROOT_URL` to the displayed URL and `GF_SERVER_SERVE_FROM_SUB_PATH=true`.
- Whether Docker works in the Studio. The setup script detects it and switches to the
  Docker executor if it does.
