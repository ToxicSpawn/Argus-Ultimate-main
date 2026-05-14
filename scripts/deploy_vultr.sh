#!/usr/bin/env bash
# =============================================================================
# deploy_vultr.sh — One-command deploy from Sydney workstation to Vultr VPS
#
# Usage (from your workstation):
#   bash scripts/deploy_vultr.sh <VPS_IP>
#   bash scripts/deploy_vultr.sh <VPS_IP> --sync-models
#
# Prerequisites:
#   - SSH key configured for root@<VPS_IP>
#   - install_vultr.sh already run on the VPS
#   - Docker installed locally (for optional local build+push)
# =============================================================================
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
VPS_IP="${1:-}"
SYNC_MODELS=false
APP_DIR="/opt/argus"
SSH_USER="root"
API_PORT="8080"
HEALTH_TIMEOUT=120

# ── Parse args ────────────────────────────────────────────────────────────────
for arg in "$@"; do
    case $arg in
        --sync-models) SYNC_MODELS=true ;;
    esac
done

if [ -z "${VPS_IP}" ]; then
    echo "Usage: $0 <VPS_IP> [--sync-models]"
    exit 1
fi

SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 ${SSH_USER}@${VPS_IP}"
SCP="scp -o StrictHostKeyChecking=no"

echo "====================================================="
echo " Argus Deploy → Vultr ${VPS_IP}"
echo " $(date)"
echo "====================================================="

# ── 1. Pull latest code on VPS ────────────────────────────────────────────────
echo "[1/5] Pulling latest code on VPS..."
${SSH} "cd ${APP_DIR} && git pull --ff-only"

# ── 2. Sync config override ───────────────────────────────────────────────────
echo "[2/5] Syncing unified_config.yaml..."
${SCP} unified_config.yaml ${SSH_USER}@${VPS_IP}:${APP_DIR}/unified_config.yaml

# ── 3. Optionally sync model artifacts from local ModelManager store ──────────
if [ "${SYNC_MODELS}" = true ]; then
    LOCAL_MODELS="${HOME}/.argus/models"
    if [ -d "${LOCAL_MODELS}" ]; then
        echo "[3/5] Syncing model artifacts to VPS..."
        rsync -avz --progress \
            --include='*.pkl' --include='*.pt' --include='*.json' \
            --exclude='*' \
            "${LOCAL_MODELS}/" \
            "${SSH_USER}@${VPS_IP}:${APP_DIR}/data/models/"
    else
        echo "[3/5] No local models found at ${LOCAL_MODELS} — skipping model sync"
    fi
else
    echo "[3/5] Skipping model sync (pass --sync-models to enable)"
fi

# ── 4. Build and restart with zero-downtime rolling update ────────────────────
echo "[4/5] Building and restarting Argus execution container..."
${SSH} bash << 'REMOTE'
set -euo pipefail
cd /opt/argus

# Build new image
docker compose -f docker-compose.vultr.yml build --no-cache argus-execution

# Rolling restart: bring up new container before stopping old one
docker compose -f docker-compose.vultr.yml up -d --no-deps --force-recreate argus-execution

echo "Container restarted. Waiting for health check..."
REMOTE

# ── 5. Health check poll ──────────────────────────────────────────────────────
echo "[5/5] Polling /health endpoint (timeout ${HEALTH_TIMEOUT}s)..."
ELAPSED=0
INTERVAL=5
HEALTHY=false

while [ $ELAPSED -lt $HEALTH_TIMEOUT ]; do
    HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" \
        "http://${VPS_IP}:${API_PORT}/health" 2>/dev/null || echo "000")
    if [ "${HTTP_CODE}" = "200" ]; then
        HEALTHY=true
        break
    fi
    echo "  Waiting... (${ELAPSED}s elapsed, HTTP ${HTTP_CODE})"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ "${HEALTHY}" = true ]; then
    echo ""
    echo "====================================================="
    echo " Deploy successful!"
    echo " Health: http://${VPS_IP}:${API_PORT}/health"
    echo " Metrics: http://${VPS_IP}:${API_PORT}/metrics"
    echo " Positions: http://${VPS_IP}:${API_PORT}/positions"
    echo "====================================================="
else
    echo ""
    echo "ERROR: Health check timed out after ${HEALTH_TIMEOUT}s"
    echo "Check logs with:"
    echo "  ssh ${SSH_USER}@${VPS_IP} 'docker compose -f /opt/argus/docker-compose.vultr.yml logs --tail=50 argus-execution'"
    exit 1
fi
