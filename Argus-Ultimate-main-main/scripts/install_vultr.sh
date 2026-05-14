#!/usr/bin/env bash
# =============================================================================
# install_vultr.sh — Bootstrap Argus on a fresh Vultr Ubuntu 24.04 bare metal
#
# Run as root on the VPS:
#   curl -fsSL https://raw.githubusercontent.com/ToxicSpawn/Argus-Ultimate-main/main/scripts/install_vultr.sh | bash
# OR after cloning:
#   sudo bash scripts/install_vultr.sh
# =============================================================================
set -euo pipefail

REPO_URL="https://github.com/ToxicSpawn/Argus-Ultimate-main.git"
APP_DIR="/opt/argus"
ARGUS_USER="argus"
API_PORT="8080"

echo "====================================================="
echo " Argus Vultr Bootstrap — $(date)"
echo "====================================================="

# ── 1. System update ──────────────────────────────────────────────────────────
echo "[1/8] Updating system packages..."
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq \
    curl wget git unzip \
    ethtool net-tools iproute2 \
    ufw fail2ban \
    htop iotop sysstat \
    build-essential \
    ca-certificates gnupg lsb-release

# ── 2. Docker install ─────────────────────────────────────────────────────────
echo "[2/8] Installing Docker..."
if ! command -v docker &>/dev/null; then
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
        > /etc/apt/sources.list.d/docker.list
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
    systemctl enable docker
    systemctl start docker
    echo "Docker installed: $(docker --version)"
else
    echo "Docker already installed: $(docker --version)"
fi

# ── 3. Kernel network tuning ──────────────────────────────────────────────────
echo "[3/8] Applying kernel network tunables..."
cat > /etc/sysctl.d/99-argus-hft.conf << 'EOF'
# Argus HFT network optimisations
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.core.rmem_default = 67108864
net.core.wmem_default = 67108864
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 250000
net.core.somaxconn = 65535
net.ipv4.tcp_low_latency = 1
net.ipv4.tcp_timestamps = 0
net.ipv4.tcp_sack = 1
net.ipv4.tcp_no_delay_ack = 1
net.ipv4.tcp_thin_linear_timeouts = 1
vm.swappiness = 1
vm.dirty_ratio = 10
vm.dirty_background_ratio = 5
EOF
sysctl -p /etc/sysctl.d/99-argus-hft.conf

# ── 4. UFW firewall ───────────────────────────────────────────────────────────
echo "[4/8] Configuring UFW firewall..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    comment 'SSH'
ufw allow ${API_PORT}/tcp comment 'Argus API dashboard'
# Redis is internal only — no public port
ufw --force enable
ufw status verbose

# ── 5. Create argus user + app dir ───────────────────────────────────────────
echo "[5/8] Creating argus user and app directory..."
id -u ${ARGUS_USER} &>/dev/null || useradd -r -m -d ${APP_DIR} -s /bin/bash ${ARGUS_USER}
usermod -aG docker ${ARGUS_USER}
mkdir -p ${APP_DIR}
chown -R ${ARGUS_USER}:${ARGUS_USER} ${APP_DIR}

# ── 6. Clone / update repo ───────────────────────────────────────────────────
echo "[6/8] Cloning Argus repository..."
if [ -d "${APP_DIR}/.git" ]; then
    echo "  Repo exists — pulling latest..."
    sudo -u ${ARGUS_USER} git -C ${APP_DIR} pull --ff-only
else
    sudo -u ${ARGUS_USER} git clone ${REPO_URL} ${APP_DIR}
fi

# ── 7. Create .env.vultr template ────────────────────────────────────────────
echo "[7/8] Creating .env.vultr template..."
ENV_FILE="${APP_DIR}/.env.vultr"
if [ ! -f "${ENV_FILE}" ]; then
    cat > ${ENV_FILE} << 'EOF'
# ============================================================
# Argus Vultr Execution Environment
# Fill in your exchange API keys and settings before starting
# ============================================================

# --- Exchange API Keys ---
KRAKEN_API_KEY=your_kraken_api_key_here
KRAKEN_API_SECRET=your_kraken_api_secret_here
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_API_SECRET=your_coinbase_api_secret_here
COINBASE_API_PASSPHRASE=your_coinbase_passphrase_here

# --- Discord Webhook (regime alerts) ---
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_webhook_here

# --- Trading Mode ---
# Options: live | paper | dry_run
ARGUS_MODE=paper

# --- API Security ---
ARGUS_API_SECRET=change_this_to_a_random_secret
ARGUS_API_PORT=8080

# --- Model sync (set to R7525 IP for model push) ---
MODEL_SOURCE_HOST=
MODEL_SOURCE_PATH=/path/to/models
EOF
    chown ${ARGUS_USER}:${ARGUS_USER} ${ENV_FILE}
    chmod 600 ${ENV_FILE}
    echo "  Created ${ENV_FILE} — EDIT THIS FILE before starting Argus!"
else
    echo "  ${ENV_FILE} already exists — skipping"
fi

# ── 8. systemd service ────────────────────────────────────────────────────────
echo "[8/8] Installing argus systemd service..."
cat > /etc/systemd/system/argus.service << EOF
[Unit]
Description=Argus HFT Execution Engine
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=simple
User=${ARGUS_USER}
WorkingDirectory=${APP_DIR}
ExecStartPre=docker compose -f docker-compose.vultr.yml pull --quiet
ExecStart=docker compose -f docker-compose.vultr.yml up
ExecStop=docker compose -f docker-compose.vultr.yml down
Restart=on-failure
RestartSec=30
TimeoutStartSec=120
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable argus.service

echo ""
echo "====================================================="
echo " Bootstrap complete!"
echo "====================================================="
echo ""
echo " NEXT STEPS:"
echo "  1. Edit API keys:  nano ${ENV_FILE}"
echo "  2. Start Argus:    cd ${APP_DIR} && docker compose -f docker-compose.vultr.yml up -d"
echo "  3. Check health:   curl http://localhost:${API_PORT}/health"
echo "  4. View logs:      docker compose -f docker-compose.vultr.yml logs -f argus-execution"
echo ""
echo " Argus API will be available at: http://$(curl -sf ifconfig.me):${API_PORT}"
echo ""
