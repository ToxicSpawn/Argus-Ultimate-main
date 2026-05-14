#!/bin/bash
# Push 72 — Argus Ultimate entrypoint script
# Handles:
#   - Wait for Redis + Postgres readiness
#   - Optional DB migration
#   - Mode selection: paper | live | backtest
#   - Graceful SIGTERM shutdown

set -euo pipefail

ARGUS_MODE=${ARGUS_MODE:-paper}
REDIS_URL=${REDIS_URL:-redis://redis:6379/0}
POSTGRES_URL=${POSTGRES_URL:-postgresql://argus:argus@postgres:5432/argus}
MAX_WAIT=${MAX_WAIT_SECS:-60}
LOG_LEVEL=${LOG_LEVEL:-INFO}

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] [entrypoint] $*"; }

# -----------------------------------------------------------------------
# Trap SIGTERM / SIGINT for graceful shutdown
# -----------------------------------------------------------------------
_ARGUS_PID=""
shutdown() {
    log "SIGTERM received — shutting down Argus (pid=${_ARGUS_PID})"
    if [ -n "${_ARGUS_PID}" ]; then
        kill -TERM "${_ARGUS_PID}" 2>/dev/null || true
        wait "${_ARGUS_PID}" 2>/dev/null || true
    fi
    log "Shutdown complete."
    exit 0
}
trap shutdown SIGTERM SIGINT

# -----------------------------------------------------------------------
# Wait for Redis
# -----------------------------------------------------------------------
wait_redis() {
    log "Waiting for Redis at ${REDIS_URL} ..."
    local host port
    host=$(echo "${REDIS_URL}" | sed 's|redis://||' | cut -d: -f1)
    port=$(echo "${REDIS_URL}" | sed 's|redis://||' | cut -d: -f2 | cut -d/ -f1)
    local elapsed=0
    until redis-cli -h "${host}" -p "${port}" ping 2>/dev/null | grep -q PONG; do
        if [ ${elapsed} -ge ${MAX_WAIT} ]; then
            log "ERROR: Redis not ready after ${MAX_WAIT}s. Exiting."
            exit 1
        fi
        sleep 2; elapsed=$((elapsed + 2))
    done
    log "Redis ready."
}

# -----------------------------------------------------------------------
# Wait for Postgres
# -----------------------------------------------------------------------
wait_postgres() {
    log "Waiting for Postgres ..."
    local elapsed=0
    until pg_isready -d "${POSTGRES_URL}" -q 2>/dev/null; do
        if [ ${elapsed} -ge ${MAX_WAIT} ]; then
            log "ERROR: Postgres not ready after ${MAX_WAIT}s. Exiting."
            exit 1
        fi
        sleep 2; elapsed=$((elapsed + 2))
    done
    log "Postgres ready."
}

# -----------------------------------------------------------------------
# Start health endpoint (background)
# -----------------------------------------------------------------------
start_health_server() {
    log "Starting health endpoint on :8000 ..."
    python docker/healthcheck.py &
}

# -----------------------------------------------------------------------
# Mode dispatch
# -----------------------------------------------------------------------
log "Argus Ultimate v8.8.0 — mode=${ARGUS_MODE} log_level=${LOG_LEVEL}"

# Only wait for infra in non-test modes
if [ "${ARGUS_MODE}" != "test" ]; then
    wait_redis  || log "WARN: Redis wait skipped (redis-cli not found)"
    wait_postgres || log "WARN: Postgres wait skipped (pg_isready not found)"
fi

start_health_server

case "${ARGUS_MODE}" in
    paper)
        log "Starting paper trading session ..."
        exec python -m core.paper_trading.session_manager "$@" &
        _ARGUS_PID=$!
        ;;
    live)
        log "Starting LIVE trading session — CAUTION: real orders will be placed"
        exec python -m core.live.live_order_manager "$@" &
        _ARGUS_PID=$!
        ;;
    backtest)
        log "Starting backtest runner ..."
        exec python -m core.backtest.backtest_runner "$@" &
        _ARGUS_PID=$!
        ;;
    healthcheck)
        python docker/healthcheck.py
        exit 0
        ;;
    *)
        log "ERROR: Unknown ARGUS_MODE='${ARGUS_MODE}'. Use: paper | live | backtest"
        exit 1
        ;;
esac

wait "${_ARGUS_PID}"
