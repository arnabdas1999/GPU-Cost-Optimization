#!/usr/bin/env bash
# Prometheus + Grafana as plain processes (no Docker, no root), for a Lightning AI
# Studio or any Linux host. Same provisioning files as observability/compose.yaml,
# live dashboard only: historical dashboards run on the laptop after `export`.
#
#   scripts/observability-native.sh start | stop | status
#
# IMPORTANT on Lightning: a running server keeps the Studio awake and uses
# credits. Run `stop` as soon as the measured session is over.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../observability/versions.env
source "$ROOT/observability/versions.env"
OBS_HOME="${OBS_HOME:-$HOME/.cache/rollout-cost-obs}"
GRAFANA_PORT="${GRAFANA_PORT:-3000}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
# 127.0.0.1 keeps Grafana private; use an SSH tunnel or the Studio's Port Viewer to open it.
GRAFANA_ADDR="${GRAFANA_ADDR:-127.0.0.1}"

case "$(uname -m)" in
  x86_64 | amd64) ARCH=amd64 ;;
  aarch64 | arm64) ARCH=arm64 ;;
  *) echo "unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac

PROM_DIR="$OBS_HOME/prometheus-$PROMETHEUS_VERSION"
GRAFANA_DIR="$OBS_HOME/grafana-$GRAFANA_VERSION"
RUN_DIR="$OBS_HOME/run"

fetch() {  # url dest_dir
  local url="$1" dest="$2" tmp
  [[ -d "$dest" ]] && return 0
  tmp="$(mktemp -d)"
  echo "downloading $url"
  curl -fsSL --retry 3 "$url" -o "$tmp/pkg.tar.gz"
  mkdir -p "$dest"
  tar -xzf "$tmp/pkg.tar.gz" -C "$dest" --strip-components=1
  rm -rf "$tmp"
}

install() {
  mkdir -p "$OBS_HOME"
  fetch "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-${ARCH}.tar.gz" "$PROM_DIR"
  fetch "https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-${ARCH}.tar.gz" "$GRAFANA_DIR"
}

is_running() { [[ -f "$RUN_DIR/$1.pid" ]] && kill -0 "$(cat "$RUN_DIR/$1.pid")" 2>/dev/null; }

wait_http() {  # url name
  for _ in $(seq 1 60); do
    if curl -fsS "$1" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  echo "$2 did not become ready; see $RUN_DIR/$2.log" >&2
  return 1
}

start() {
  install
  mkdir -p "$RUN_DIR/prometheus/targets" "$RUN_DIR/provisioning/datasources" "$RUN_DIR/provisioning/dashboards" \
           "$RUN_DIR/prometheus-data" "$RUN_DIR/grafana-data" "$RUN_DIR/grafana-logs" "$RUN_DIR/grafana-plugins"
  cp "$ROOT/observability/prometheus/prometheus.yml" "$RUN_DIR/prometheus/prometheus.yml"
  cp "$ROOT"/observability/prometheus/targets/native/*.json "$RUN_DIR/prometheus/targets/"
  # Live view only on the GPU host.
  cp "$ROOT/observability/grafana/provisioning/datasources/prometheus.yaml" "$RUN_DIR/provisioning/datasources/"
  cp "$ROOT/observability/grafana/provisioning/dashboards/live.yaml" "$RUN_DIR/provisioning/dashboards/"

  if is_running prometheus; then
    echo "prometheus already running"
  else
    nohup "$PROM_DIR/prometheus" --config.file="$RUN_DIR/prometheus/prometheus.yml" \
      --storage.tsdb.path="$RUN_DIR/prometheus-data" --web.listen-address="127.0.0.1:$PROMETHEUS_PORT" \
      >"$RUN_DIR/prometheus.log" 2>&1 &
    echo $! >"$RUN_DIR/prometheus.pid"
  fi
  if is_running grafana; then
    echo "grafana already running"
  else
    env GF_PATHS_HOME="$GRAFANA_DIR" GF_PATHS_DATA="$RUN_DIR/grafana-data" GF_PATHS_LOGS="$RUN_DIR/grafana-logs" \
      GF_PATHS_PLUGINS="$RUN_DIR/grafana-plugins" GF_PATHS_PROVISIONING="$RUN_DIR/provisioning" \
      GF_SERVER_HTTP_ADDR="$GRAFANA_ADDR" GF_SERVER_HTTP_PORT="$GRAFANA_PORT" \
      GF_AUTH_ANONYMOUS_ENABLED=true GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
      GF_SECURITY_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-admin}" \
      GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH="$ROOT/observability/grafana/dashboards/live/live-run.json" \
      GF_ANALYTICS_REPORTING_ENABLED=false GF_ANALYTICS_CHECK_FOR_UPDATES=false \
      ${GF_SERVER_ROOT_URL:+GF_SERVER_ROOT_URL="$GF_SERVER_ROOT_URL"} \
      ${GF_SERVER_SERVE_FROM_SUB_PATH:+GF_SERVER_SERVE_FROM_SUB_PATH="$GF_SERVER_SERVE_FROM_SUB_PATH"} \
      RC_DASHBOARDS_DIR="$ROOT/observability/grafana/dashboards" PROMETHEUS_URL="http://127.0.0.1:$PROMETHEUS_PORT" \
      nohup "$GRAFANA_DIR/bin/grafana" server --homepath "$GRAFANA_DIR" >"$RUN_DIR/grafana.log" 2>&1 &
    echo $! >"$RUN_DIR/grafana.pid"
  fi
  wait_http "http://127.0.0.1:$PROMETHEUS_PORT/-/ready" prometheus
  wait_http "http://127.0.0.1:$GRAFANA_PORT/api/health" grafana
  cat <<EOF
Observability is running (live view).
  Grafana:    http://127.0.0.1:$GRAFANA_PORT   (anonymous viewer; admin password from \$GRAFANA_ADMIN_PASSWORD)
  Prometheus: http://127.0.0.1:$PROMETHEUS_PORT (scrapes the runner on :9400 and vLLM on :8000)
Open it from your laptop with an SSH tunnel:  ssh -L $GRAFANA_PORT:127.0.0.1:$GRAFANA_PORT <studio-ssh-host>
or, on Lightning, the Port Viewer plugin on port $GRAFANA_PORT (if it serves under a sub-path, restart with
GF_SERVER_ROOT_URL=<url> GF_SERVER_SERVE_FROM_SUB_PATH=true).
REMEMBER: run '$0 stop' when finished so the Studio can sleep.
EOF
}

stop() {
  for name in grafana prometheus; do
    if is_running "$name"; then
      kill "$(cat "$RUN_DIR/$name.pid")" && echo "stopped $name"
    fi
    rm -f "$RUN_DIR/$name.pid"
  done
}

status() {
  for name in prometheus grafana; do
    if is_running "$name"; then echo "$name: running (pid $(cat "$RUN_DIR/$name.pid"))"; else echo "$name: stopped"; fi
  done
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  install) install ;;
  *) echo "usage: $0 start|stop|status|install" >&2; exit 2 ;;
esac
