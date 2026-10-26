#!/usr/bin/env bash
# Runs inside the rehearsal container (see rehearse-lightning.sh) as a non-root user,
# following docs/LIGHTNING_RUNBOOK.md with the mock vLLM server in place of a GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RATE=0.80 RATE_SOURCE="rehearsal placeholder, not a quote" bash scripts/lightning-setup.sh --rehearsal
export PATH="$HOME/.local/bin:$PATH"
GRAFANA_ADDR=0.0.0.0 bash scripts/observability-native.sh start
uv run rollout-cost run --config configs/smoke.pinned.yaml --split dev
uv run rollout-cost run --config configs/lightning.pinned.yaml --split dev

echo "== checks"
check() {  # name, then a command that must succeed
  local name="$1"
  shift
  if "$@" >/dev/null 2>&1; then echo "check ok: $name"; else echo "check FAILED: $name"; exit 1; fi
}
prom_has_value() {
  curl -fsSg "http://127.0.0.1:9090/api/v1/query" --data-urlencode "query=$1" | grep -q '"value"'
}
check "grafana serves the live dashboard" bash -c "curl -fsS 'http://127.0.0.1:3000/api/search?query=Live' | grep -q rc-live"
check "prometheus stored runner metrics" prom_has_value "max_over_time(rc_episodes_submitted[30m])"
check "prometheus stored vllm metrics" prom_has_value "max_over_time(vllm:num_requests_running[30m])"
check "runner pinned to the reserved cpus" uv run python -c "
import glob, json
m = json.load(open(sorted(glob.glob('runs/*lightning*/run.json'))[-1]))
assert m['host']['runner_cpu_affinity'] == m['config']['server']['reserved_cpus'], m['host']['runner_cpu_affinity']"
check "local executor pins test processes (Linux-only test)" uv run pytest -q tests/test_observability.py -k pins_to_slot
bash scripts/observability-native.sh status

cp -r runs /out/runs
cp configs/*.pinned.yaml /out/
if [[ -n "${REHEARSAL_HOLD:-}" ]]; then
  echo "holding for $REHEARSAL_HOLD s (Grafana on the host at http://127.0.0.1:3300)"
  sleep "$REHEARSAL_HOLD"
fi
bash scripts/observability-native.sh stop
