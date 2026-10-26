# Dashboards

Grafana shows two views of the same data:

- **Live**: Prometheus scrapes the runner's exporter (`:9400`, metrics `rc_*`) and the
  model server's own `/metrics` while a run is in progress.
- **History**: `rollout-cost export` loads run directories, sweeps and paired experiments
  into PostgreSQL. Grafana reads them from there.

| Dashboard | Data | Shows |
| --- | --- | --- |
| Live run | Prometheus | Phase, progress, episodes vs cap, request slots, tool slots, vLLM queue and KV cache, tokens/s, prefix-cache hit rate, TTFT, GPU, cost so far |
| Runs | PostgreSQL | Every run, synthetic or measured, with outcomes, window and cost. Click a run to open its detail |
| Run detail | PostgreSQL | Episode timeline, concurrency vs caps, time attribution, latency, outcomes, per-task results, GPU/server samples, warnings |
| Experiments and sweeps | PostgreSQL | Decision, window/cost/successes per trial or variant, acceptance checks, transitions, per-task passes |

Values from the scripted model or the mock server are marked **SYNTHETIC**.

## On the laptop (Docker)

```bash
uv sync --extra dashboard
docker compose -f observability/compose.yaml up -d        # Grafana :3000, Prometheus :9090, Postgres :55432
uv run rollout-cost run --config configs/mock-vllm.yaml   # watch it in "Live run" (SYNTHETIC)
uv run rollout-cost export                                # then open "Runs"
```

Grafana allows anonymous viewing. The admin login is `admin`/`admin` unless overridden
in `observability/.env` (copy `.env.example`). Export is idempotent: rerun it after every
session. `--reset` recreates the tables after a schema change.

## On a GPU host without Docker (Lightning AI)

```bash
bash scripts/observability-native.sh start    # pinned Prometheus + Grafana binaries, no root; live view only
bash scripts/observability-native.sh stop     # do this when finished, or the Studio stays awake
```

## Checking that every panel works

```bash
uv run python scripts/verify_dashboards.py                                  # laptop stack: all dashboards
uv run python scripts/verify_dashboards.py --grafana http://127.0.0.1:3300 --live-only   # a GPU host's Grafana via a tunnel
```

The script sends each panel query through Grafana's own query API. It reports panels that
error, and panels that return nothing when they should (GPU panels may be empty without a
GPU).

## Changing dashboards

The dashboard JSON is generated. Edit `observability/build_dashboards.py`, then run:

```bash
uv run python observability/build_dashboards.py
```

`tests/test_observability.py` fails if the committed JSON is stale. It also fails if a
query uses a metric outside `rollout_cost.contract`, a table missing from
`src/rollout_cost/schema.sql`, or a datasource UID that isn't provisioned. Pinned image
and binary versions live in `versions.env`.
