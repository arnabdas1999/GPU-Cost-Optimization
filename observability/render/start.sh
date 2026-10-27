#!/bin/sh
set -eu

# Grafana's own state and its query datasource deliberately use different users
# and separate Neon projects. Never give the public datasource an owner password.
: "${GF_DATABASE_HOST:?Set the grafana-state Neon host with :5432}"
: "${GF_DATABASE_NAME:?Set the grafana-state database name}"
: "${GF_DATABASE_USER:?Set the grafana-state owner role}"
: "${GF_DATABASE_PASSWORD:?Set the grafana-state owner password}"
: "${GF_SECURITY_ADMIN_PASSWORD:?Set a strong Grafana admin password}"
: "${GF_SECURITY_SECRET_KEY:?Set a stable Grafana encryption key}"
: "${RC_HISTORY_HOST:?Set the rollout-history Neon host with :5432}"
: "${RC_HISTORY_DATABASE:?Set the history database name}"
: "${RC_HISTORY_USER:?Set the history read-only role}"
: "${RC_HISTORY_PASSWORD:?Set the history read-only password}"

export GF_SERVER_HTTP_PORT="${PORT:-10000}"
RC_PUBLIC_URL="${RENDER_EXTERNAL_URL:-${GF_SERVER_ROOT_URL:-http://localhost:$GF_SERVER_HTTP_PORT}}"
export GF_SERVER_ROOT_URL="${RC_PUBLIC_URL%/}/"
export RC_HISTORY_SSLMODE="${RC_HISTORY_SSLMODE:-verify-full}"

exec /run.sh "$@"
