# Colab / Kaggle: first GPU session

[Open in Colab](https://colab.research.google.com/github/arnabdas1999/GPU-Cost-Optimization/blob/main/notebooks/gpu_smoke_test.ipynb)

The same notebook is at `notebooks/gpu_smoke_test.ipynb` for importing into Kaggle.
No laptop GPU is needed. Keep the notebook interactive; the harness calls vLLM on
loopback inside the runtime. Use the notebook cells for testing and downloading results.

## Start here

1. **Colab:** open the link, choose **Runtime > Change runtime type > GPU**, and connect.
   **Kaggle:** create/import the notebook, choose **T4** in its accelerator settings,
   and enable **Internet**. Availability and quotas depend on your account.
2. Select **Run all**. The first cell clones the public repository into `/content/rollout-cost`
   or `/kaggle/working/rollout-cost`, then checks GPU 0. P100 and CPU sessions are rejected
   before model downloads. No kernel PyTorch replacement or kernel restart is required.
3. Setup installs the locked project on Python 3.12 and a configurable exact vLLM release
   (default `0.31.0`) in `/tmp/rollout-cost-vllm-<version>`. It resolves the model commit,
   downloads it, detects available CPUs, generates configs, validates fixtures and runs doctor.
4. The smoke cell runs **six episodes, one at a time**. It shows the report and checks real
   GPU status, no infrastructure errors, at least one success, protocol errors below 20%,
   no wall timeouts and verified drain. Review two transcripts as well.
5. The final cell downloads `notebook-results.tgz` on Colab or displays its path/link on Kaggle.
   Download it, then end the GPU session with the platform's controls. Stopping vLLM does
   not end your allocated GPU session.

**If a cell fails:** the command log is retained. Run the final download cell manually;
it works after setup or smoke failures. Setup and smoke also build an archive in their
`finally` blocks. The archive contains configs, run artifacts, command logs, hardware
details and vLLM package/help inventories when available. It excludes environments,
model caches and `.env` files.

## Defaults and compatibility

| Setting | Notebook profile |
| --- | --- |
| Model | `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`, exact revision resolved at setup |
| Precision | AWQ 4-bit weights; FP16 (`float16`) activations |
| Context | 8192 for both server and harness |
| Episode budget | 12 replies; 1024 output tokens per reply |
| Sampling | Greedy (`temperature: 0`, `top_p: 1`) |
| GPUs | One; notebook selects GPU 0, tensor parallel size 1 |
| Server request slots | 2 |
| CUDA graphs | Disabled with `enforce-eager` for quicker startup; fixed across comparisons |
| Smoke | 6 episodes, active/request caps 1/1 |
| Pilot | 24 episodes, active/request caps 2/2 |
| Tools | Local executor; available CPU set detected, unpinned below 4 CPUs |
| Price | Unknown, reported unavailable; not an invented $0 allocation rate |

Kaggle T4 x2 does not automatically pool memory. This first-session profile uses one
of those GPUs. No Docker daemon or Grafana service is required. Local tools execute
generated code as the notebook user; run in a fresh runtime with only the project data
needed for testing.

The earlier six-reply, 4096-context T4 runs completed without infrastructure errors but
solved zero tasks. In the 7B AWQ run, five tasks ended with invalid Python and one missed
behavior required by its docstring. Four exhausted the reply budget. The current defaults
are a calibration proposal: literal-edit indentation is explained, file writes/edits get
an immediate syntax check without executing generated code, and unchanged edits are rejected.
Invalid code remains applied and hidden grading stays independent. These changes do not
guarantee a solve rate; review the new smoke run before comparisons. The harness behavior
hash distinguishes runs with the updated prompt and tools from older runs.

Actual installation depends on the runtime driver and vLLM wheel. Free runtime
limits and CPU contention also affect timing. Results use `dev`: the held-out split is empty.

## Update an existing 7B AWQ session

If the model and vLLM are already installed, keep the current runtime, saved results,
model cache and pinned revision. In a code cell, update the checkout:

```python
subprocess.run(["git", "pull", "--ff-only"], cwd=ROOT, check=True)
```

Then rerun the smoke cell. The existing pinned configs extend `notebook.yaml`, so they
inherit the new budgets and server context. The CLI starts a fresh process and loads the
updated prompt and tools. No setup rerun or model download is needed for this case.
If Git reports conflicting local edits, preserve them and resolve the conflict before
continuing. Use the new settings/setup cells for a fresh runtime or a different model.

Do not compare an older six-reply run with a new run as evidence of a serving optimization.
After calibration, freeze the model, prompt, harness, budgets and sampling for both variants.

## Optional pilot and comparisons

After the smoke check passes, change `RUN_PILOT = True` in the settings cell, execute
that cell, then execute the pilot cell. Session labels are preserved when settings are
rerun. Review budgets, protocol errors and solve counts, then freeze all non-tunable settings.

After a healthy pilot, set `RUN_AA = True` in the settings cell, execute that cell,
then execute the A/A cell. It runs four cohorts across two
paired trials. Expected verdict: **A/A CALIBRATION**. Use its spread to judge timing noise.

For an admission sweep, run in a notebook code cell after the pilot:

```python
run_logged(ROOT, RC + ["sweep", "--config", "configs/notebook-sweep.pinned.yaml"],
           "notebook-sweep.log")
archive_results(ROOT)
```

If the sweep selects a candidate, copy its printed `chosen.yaml` path into this cell:

```python
CHOSEN = "experiments/REPLACE_WITH_PRINTED_SWEEP_DIRECTORY/chosen.yaml"
run_logged(ROOT, RC + ["paired", "--baseline", "configs/notebook.pinned.yaml",
                      "--candidate", CHOSEN, "--split", "dev", "--trials", "2",
                      "--name", "notebook-comparison-" + SESSION_TAG], "notebook-comparison.log")
archive_results(ROOT)
```

Skip paired validation if the sweep kept the baseline. Download the updated archive afterward.

## Troubleshooting

- **No GPU / P100:** select a supported GPU runtime (T4 or newer). If unavailable, wait
  for capacity or use another supported host. The selected vLLM release requires compute
  capability 7.5 or higher.
- **Download/network failure:** enable Kaggle Internet and inspect `notebook-setup.log`.
- **CUDA/driver or wheel errors:** save `notebook-hardware.txt`, `notebook-setup.log` and
  `vllm-serve-help.txt` if present. Do not upgrade the kernel's Torch blindly: the server
  has its own environment. Adjust `VLLM_VERSION` only after checking the driver/wheel match.
- **Zero solves / protocol failures:** inspect transcripts before longer tests. The gate
  prints the actual model, reply/context budgets and stop-reason counts. `max_steps` and
  `context_exhausted` need different calibration. Passing public tests does not guarantee
  that a solution follows the full docstring. The archive remains available after a failure.
- **Out of memory:** verify other model processes are stopped. Reduce context in both
  `server.args.max-model-len` and `workload.context_window_tokens`, then regenerate configs
  and recalibrate before comparisons. The baseline and candidate must use the same budgets.
- **Interrupted cell:** the logging helper sends SIGINT and allows cleanup. Doctor cleans
  up even if interrupted during startup; Linux shutdown terminates the managed server group.
- **New runtime:** rerun from the top. Model caches and environments are ephemeral; download
  the results archive before ending the current session.

Official references: [vLLM GPU requirements](https://docs.vllm.ai/en/v0.31.0/getting_started/installation/gpu/),
[Qwen model](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-AWQ),
[Colab limits](https://research.google.com/colaboratory/faq.html),
[Kaggle notebooks](https://www.kaggle.com/docs/notebooks).
