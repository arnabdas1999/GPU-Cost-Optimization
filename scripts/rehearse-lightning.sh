#!/usr/bin/env bash
# Rehearse the Lightning session on the laptop: a plain Ubuntu container (no Docker
# inside, no GPU) runs scripts/rehearsal-inside.sh as a normal user. That script runs
# lightning-setup.sh --rehearsal, the native observability stack and pinned runs
# against the mock vLLM server. Artifacts are copied to rehearsal-out/.
#
#   scripts/rehearse-lightning.sh                      # run and exit
#   REHEARSAL_HOLD=300 scripts/rehearse-lightning.sh   # keep the container's Grafana on http://127.0.0.1:3300
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/rehearsal-out"
IMAGE="${REHEARSAL_IMAGE:-ubuntu:24.04}"
rm -rf "$OUT" && mkdir -p "$OUT"

# Copy the repository without host-specific state (the Windows .venv, previous runs).
tar -C "$ROOT" --exclude=./.venv --exclude=./runs --exclude=./experiments --exclude=./rehearsal-out \
    --exclude=./.git --exclude='*/__pycache__' -czf "$OUT/repo.tgz" .

MOUNT="$(cygpath -w "$OUT" 2>/dev/null || echo "$OUT")"
docker run --rm --name rc-rehearsal -p 127.0.0.1:3300:3000 -e REHEARSAL_HOLD="${REHEARSAL_HOLD:-}" \
  -v "$MOUNT:/out" "$IMAGE" bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq curl ca-certificates git >/dev/null
    useradd -m -s /bin/bash studio
    mkdir -p /home/studio/rollout-cost && tar -xzf /out/repo.tgz -C /home/studio/rollout-cost
    chown -R studio:studio /home/studio /out
    su studio -s /bin/bash -c "REHEARSAL_HOLD=$REHEARSAL_HOLD bash /home/studio/rollout-cost/scripts/rehearsal-inside.sh"
  '
echo "rehearsal artifacts in $OUT"
