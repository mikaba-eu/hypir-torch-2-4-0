#!/usr/bin/env bash
set -Eeuo pipefail

JOB_PID=""

log() {
  echo "[$(date -Is)] [entrypoint] $*"
}

terminate_pod_best_effort() {
  local pod_id="${RUNPOD_POD_ID:-}"
  if [[ -z "${pod_id}" ]]; then
    log "RUNPOD_POD_ID not set; cannot terminate pod via runpodctl."
    return 0
  fi

  if ! command -v runpodctl >/dev/null 2>&1; then
    log "runpodctl not found; cannot terminate pod via runpodctl."
    return 0
  fi

  # Always try remove (terminate). If that fails, try stop as fallback.
  # Retry a few times because network/API can be flaky during shutdown.
  local attempt=1
  local max_attempts=8
  local sleep_s=2

  while (( attempt <= max_attempts )); do
    log "Attempt ${attempt}/${max_attempts}: runpodctl remove pod ${pod_id}"
    if runpodctl remove pod "${pod_id}"; then
      log "Pod terminated successfully."
      return 0
    fi

    log "WARNING: remove failed; trying stop as fallback."
    runpodctl stop pod "${pod_id}" || true

    log "Retrying in ${sleep_s}s..."
    sleep "${sleep_s}"
    attempt=$((attempt + 1))
    sleep_s=$((sleep_s * 2))
    if (( sleep_s > 30 )); then
      sleep_s=30
    fi
  done

  log "ERROR: Could not terminate pod after ${max_attempts} attempts."
  return 1
}

cleanup() {
  local exit_code=$?
  set +e

  log "Cleanup triggered (exit_code=${exit_code})."

  # If job still running, try to stop it (best effort).
  if [[ -n "${JOB_PID}" ]] && kill -0 "${JOB_PID}" >/dev/null 2>&1; then
    log "Job still running (pid=${JOB_PID}); sending TERM."
    kill -TERM "${JOB_PID}" >/dev/null 2>&1 || true
    sleep 3
    if kill -0 "${JOB_PID}" >/dev/null 2>&1; then
      log "Job still running; sending KILL."
      kill -KILL "${JOB_PID}" >/dev/null 2>&1 || true
    fi
  fi

  terminate_pod_best_effort || true

  log "Cleanup finished. Exiting with original exit_code=${exit_code}."
  exit "${exit_code}"
}

trap cleanup EXIT
trap 'log "SIGINT received"; exit 130' INT
trap 'log "SIGTERM received"; exit 143' TERM
trap 'log "SIGHUP received"; exit 129' HUP
trap 'log "SIGQUIT received"; exit 131' QUIT

main() {
  log "Starting python job..."
  python /workspace/run.py "$@" &
  JOB_PID=$!
  wait "${JOB_PID}"
  log "Python job finished."
}

main "$@"