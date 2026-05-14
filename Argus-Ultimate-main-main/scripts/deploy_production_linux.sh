#!/bin/bash
# =============================================================================
# Deploy Argus Ultimate to /opt/argus (R730 or any Linux trading host)
# =============================================================================
# Prereqs: run scripts/setup-linux.sh first (creates argus user, venv, systemd).
# Usage: sudo ./scripts/deploy_production_linux.sh [repo_url]
# Default repo: https://github.com/ToxicSpawn/Argus-Ultimate.git
# =============================================================================

set -e

REPO_URL="${1:-https://github.com/ToxicSpawn/Argus-Ultimate.git}"
INSTALL_DIR="/opt/argus"
REPO_DIR="$INSTALL_DIR/repo"

if [[ $EUID -ne 0 ]]; then
   echo "[ERROR] Run as root: sudo $0"
   exit 1
fi

echo "[INFO] Deploying Argus to $REPO_DIR"

# Clone or pull
if [[ -d "$REPO_DIR/.git" ]]; then
   echo "[INFO] Updating existing clone..."
   sudo -u argus git -C "$REPO_DIR" fetch origin
   sudo -u argus git -C "$REPO_DIR" checkout main
   sudo -u argus git -C "$REPO_DIR" pull --rebase origin main
else
   mkdir -p "$INSTALL_DIR"
   chown argus:argus "$INSTALL_DIR"
   echo "[INFO] Cloning $REPO_URL..."
   sudo -u argus git clone "$REPO_URL" "$REPO_DIR"
fi

# Install Python deps into existing venv
if [[ -f "$INSTALL_DIR/venv/bin/pip" ]]; then
   echo "[INFO] Installing requirements..."
   sudo -u argus "$INSTALL_DIR/venv/bin/pip" install -q --upgrade pip
   sudo -u argus "$INSTALL_DIR/venv/bin/pip" install -q -r "$REPO_DIR/requirements.txt"
else
   echo "[WARN] No venv at $INSTALL_DIR/venv. Run scripts/setup-linux.sh first."
   exit 1
fi

# .env from example if missing
if [[ ! -f "$REPO_DIR/.env" ]]; then
   if [[ -f "$REPO_DIR/.env.example" ]]; then
      sudo -u argus cp "$REPO_DIR/.env.example" "$REPO_DIR/.env"
      echo "[INFO] Created $REPO_DIR/.env from .env.example - edit with your API keys."
   fi
fi

# Validate
echo "[INFO] Running validation..."
sudo -u argus bash -c "cd $REPO_DIR && $INSTALL_DIR/venv/bin/python main.py validate" || true

# Optional infra preflight (fail-closed only when enabled explicitly).
if [[ "${ARGUS_ENFORCE_INFRA_PREFLIGHT:-0}" == "1" ]]; then
   echo "[INFO] Running infra verification + preflight (ARGUS_ENFORCE_INFRA_PREFLIGHT=1)..."
   sudo -u argus bash -c "cd $REPO_DIR && $INSTALL_DIR/venv/bin/python scripts/infra_verify_host.py --iface ${ARGUS_IFACE:-eth0} --output reports/infra/verification_latest.json --strict"
   sudo -u argus bash -c "cd $REPO_DIR && $INSTALL_DIR/venv/bin/python scripts/infra_preflight.py --report reports/infra/verification_latest.json --output reports/infra/infra_preflight_latest.json --max-clock-offset-us ${ARGUS_MAX_CLOCK_OFFSET_US:-250}"
fi

# Update systemd to use repo dir and unified config
SVC_FILE="/etc/systemd/system/argus.service"
if [[ -f "$SVC_FILE" ]]; then
   if ! grep -q "WorkingDirectory=$REPO_DIR" "$SVC_FILE" 2>/dev/null; then
      sed -i "s|WorkingDirectory=.*|WorkingDirectory=$REPO_DIR|" "$SVC_FILE"
      sed -i "s|ExecStart=.*|ExecStart=$INSTALL_DIR/venv/bin/python -O main.py paper --capital 1000 --config unified_config.yaml|" "$SVC_FILE"
      systemctl daemon-reload
      echo "[INFO] Updated argus.service to use $REPO_DIR and unified_config.yaml"
   fi
fi

echo ""
echo "[DONE] Deploy complete. Next:"
echo "  1. Edit $REPO_DIR/.env with API keys (for live) or leave for paper."
echo "  2. Start: systemctl start argus"
echo "  3. Logs:  journalctl -u argus -f"
echo "  4. Optional: unified_config.production.yaml and --config for overrides."
echo ""
