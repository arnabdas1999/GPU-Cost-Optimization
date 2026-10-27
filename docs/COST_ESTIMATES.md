# Estimated equivalent cost for notebook runs

The Colab/Kaggle T4 profile uses a documented **reference compute allocation** to
value measured time. This is an estimate, not your Colab bill or a prediction of
performance on the reference VM. It does not provision anything or incur charges.

Reference checked **October 26, 2026**, Google Cloud **us-central1 (Iowa)**,
on-demand, USD:

| Component | Hourly rate |
| --- | ---: |
| 1 NVIDIA T4, 16 GB GPU memory | $0.35 |
| n1-standard-4, 4 vCPU and 15 GiB RAM | $0.189999 |
| Reference compute allocation | **$0.539999 (about $0.54)** |

Sources: [GPU pricing](https://cloud.google.com/products/compute/gpus-pricing),
[N1 machine pricing](https://cloud.google.com/products/compute/pricing/general-purpose?hl=en).
Google charges for the GPU in addition to the VM, so using the GPU-only price would
leave out the CPU and RAM allocation used by tools and the server.

The calculation is `measured_seconds / 3600 * 0.539999`. For example, 600 seconds
is approximately $0.09. Both the steady-state window and the run-inclusive interval
are valued. Setup, downloads, idle runtime outside those intervals, storage,
network, taxes and provider minimum charges are excluded. The estimate is prorated
over measured time; it is not a full hypothetical provider invoice.

Colab's CPU, RAM, GPU clocks and resource sharing can differ from the reference
host. Dollar reductions at a fixed reference rate are exactly proportional to
measured time reductions; they do not independently prove real-world cloud savings.
Keep the same rate and workload across a comparison. Use a suitable reference or
the actual provider rate when testing another GPU or a multi-GPU allocation.

## New sessions

The assumption lives in `configs/notebook.yaml` as `cost.rate_kind:
estimated_equivalent`, with the amount, region, components, source URLs and checked
date recorded in the configuration. Fresh setup/pin runs inherit it automatically.
Other profiles still use their own prices or an unknown rate.

An explicit `pin --rate ... --rate-source ...` overrides the reference and records
that supplied allocation rate. The reference source is not carried over to an
unrelated price.

If your Colab session already has an older generated `notebook.pinned.yaml`, its
explicit `allocation_hourly_rate: null` overrides the new default. Pull the latest
code, then remove the old `cost` block from that generated config to inherit the
updated notebook profile. The smoke/sweep configs inherit the same pinned base.
No model download is required. Previously completed runs retain their original rate.

Reports label the assumption **Estimated equivalent cost**. Grafana's Runs table
has a **cost_basis** column, and Run detail has a **Cost basis** panel containing
the rate and source. The existing warnings table retains this provenance, so no
database schema reset or migration is necessary. Refresh/redeploy Grafana after
updating its dashboard definitions.

## Completed runs: no GPU rerun required

After pulling the latest code, render a separate valuation report:

```powershell
uv run --extra dashboard --locked rollout-cost report `
  --run runs/YOUR_RUN_ID --cost-config configs/notebook.yaml
```

This writes `runs/YOUR_RUN_ID/report.estimated.md`. It preserves `run.json`,
`summary.json`, episode records and the original `report.md`. Recorded timing,
successes, task grading, fingerprints and acceptance decisions remain unchanged.
The override applies to reporting/export, not to the comparison's measured inputs.

In the existing Colab notebook, after updating the clone, you can use:

```python
run_logged(ROOT, RC + ["report", "--run", str(PILOT_DIR),
                      "--cost-config", "configs/notebook.yaml"], "notebook-cost.log")
display(Markdown((PILOT_DIR / "report.estimated.md").read_text()))
archive_results(ROOT)
```

Replace `PILOT_DIR` with your actual run directory if it was not assigned in this
session. Use `SMOKE_DIR` for the smoke run. No vLLM restart is needed.

For Grafana, export with an explicit valuation:

```powershell
# RC_DB_URL points to your local database or the hosted rollout-history owner URL.
uv run --extra dashboard --locked rollout-cost export --cost-config configs/notebook.yaml
```

This values **all runs and experiment/sweep trial rows selected by the export** at
the reference rate. Export only the T4 collection you intend to value; use
`--runs-dir` and `--experiments-dir` to select it. Existing database rows are
updated by run ID without duplicates. The source archives remain unchanged.

Re-exporting without `--cost-config` restores the prices recorded in the artifacts.
For older artifacts that recorded an unknown rate, that restores unavailable cost.
Keep the override in your publishing command when you want reference valuations.
Cost per success remains undefined when a run has zero successful episodes.
