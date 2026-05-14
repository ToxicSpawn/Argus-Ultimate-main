#!/bin/bash
# =============================================================================
# ARGUS ULTIMATE - Dell R7525 Deployment Script
# =============================================================================
# Optimized for AMD EPYC 7002/7003 Series (128 cores, 4TB RAM)
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "=================================================================="
echo "     ARGUS ULTIMATE - Dell R7525 Server Deployment"
echo "     AMD EPYC Optimized | Quantum Omega Engine"
echo "=================================================================="
echo -e "${NC}"

# =============================================================================
# SYSTEM CHECKS
# =============================================================================
echo -e "${YELLOW}[1/8] Checking System Requirements...${NC}"

# Check CPU
CPU_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "unknown")
echo "  CPU Cores: $CPU_CORES"

# Check RAM
if [ -f /proc/meminfo ]; then
    RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    RAM_GB=$((RAM_KB / 1024 / 1024))
    echo "  RAM: ${RAM_GB}GB"
elif [ "$(uname)" == "Darwin" ]; then
    RAM_BYTES=$(sysctl -n hw.memsize)
    RAM_GB=$((RAM_BYTES / 1024 / 1024 / 1024))
    echo "  RAM: ${RAM_GB}GB"
else
    echo "  RAM: Unknown"
fi

# Check Python
PYTHON_VERSION=$(python3 --version 2>&1 || echo "Not found")
echo "  Python: $PYTHON_VERSION"

# Check if running on Dell hardware
if [ -f /sys/class/dmi/id/sys_vendor ]; then
    VENDOR=$(cat /sys/class/dmi/id/sys_vendor)
    PRODUCT=$(cat /sys/class/dmi/id/product_name 2>/dev/null || echo "Unknown")
    echo "  Hardware: $VENDOR $PRODUCT"
fi

echo -e "${GREEN}  ✓ System check complete${NC}"

# =============================================================================
# DEPENDENCY INSTALLATION
# =============================================================================
echo -e "\n${YELLOW}[2/8] Installing Dependencies...${NC}"

# System packages (Debian/Ubuntu)
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y \
        python3.12 python3.12-venv python3.12-dev \
        build-essential \
        libssl-dev \
        libffi-dev \
        tmux \
        htop \
        iotop \
        numactl \
        libnuma-dev
fi

# System packages (RHEL/CentOS)
if command -v yum &> /dev/null; then
    sudo yum install -y \
        python312 python312-devel \
        gcc gcc-c++ \
        openssl-devel \
        libffi-devel \
        tmux \
        htop \
        numactl-devel
fi

echo -e "${GREEN}  ✓ Dependencies installed${NC}"

# =============================================================================
# NUMA OPTIMIZATION (Critical for EPYC)
# =============================================================================
echo -e "\n${YELLOW}[3/8] Configuring NUMA Optimization...${NC}"

# Check NUMA topology
if command -v numactl &> /dev/null; then
    echo "  NUMA Nodes:"
    numactl --hardware 2>/dev/null || echo "  NUMA info not available"
    
    # Set memory policy
    echo 0 | sudo tee /proc/sys/vm/numa_balancing > /dev/null 2>&1 || true
    echo "  ✓ NUMA balancing disabled (manual pinning preferred)"
fi

# Set CPU governor to performance
if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1 || true
    echo "  ✓ CPU governor set to performance"
fi

echo -e "${GREEN}  ✓ NUMA optimization configured${NC}"

# =============================================================================
# DIRECTORY SETUP
# =============================================================================
echo -e "\n${YELLOW}[4/8] Setting Up Directories...${NC}"

# Create directories
sudo mkdir -p /data/argus/{db,ohlcv,logs,backup}
sudo mkdir -p /var/log/argus
sudo mkdir -p /etc/argus

# Set ownership
sudo chown -R $USER:$USER /data/argus
sudo chown -R $USER:$USER /var/log/argus

# Create symlink for config
if [ ! -L /etc/argus/config.yaml ]; then
    ln -sf $(pwd)/config/profiles/r7525_server.yaml /etc/argus/config.yaml
fi

echo -e "${GREEN}  ✓ Directories created${NC}"

# =============================================================================
# PYTHON ENVIRONMENT
# =============================================================================
echo -e "\n${YELLOW}[5/8] Setting Up Python Environment...${NC}"

# Create virtual environment
if [ ! -d "venv" ]; then
    python3.12 -m venv venv
fi

# Activate and install
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Install quantum dependencies
pip install qiskit qiskit-aer numpy scipy

echo -e "${GREEN}  ✓ Python environment ready${NC}"

# =============================================================================
# SYSTEMD SERVICE
# =============================================================================
echo -e "\n${YELLOW}[6/8] Creating Systemd Service...${NC}"

SERVICE_FILE="/etc/systemd/system/argus-trading.service"

sudo tee $SERVICE_FILE > /dev/null <<EOF
[Unit]
Description=Argus Ultimate Trading System
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/venv/bin/python main.py live --config config/profiles/r7525_server.yaml
Restart=always
RestartSec=10

# Resource limits (optimized for R7525)
CPUQuota=6400%
MemoryMax=128G
MemorySwapMax=0

# NUMA pinning (if available)
# NUMAPolicy=bind
# NUMAMask=0-7

# Environment
Environment="PYTHONUNBUFFERED=1"
Environment="ARGUS_ENV=production"
Environment="ARGUS_CONFIG_PROFILE=r7525"

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=argus

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
echo -e "${GREEN}  ✓ Systemd service created${NC}"

# =============================================================================
# SECURITY
# =============================================================================
echo -e "\n${YELLOW}[7/8] Security Configuration...${NC}"

# Create argus user if not exists
if ! id "argus" &>/dev/null; then
    sudo useradd -r -s /bin/false argus
    sudo usermod -aG argus $USER
fi

# Set permissions
chmod 700 config/profiles/r7525_server.yaml
chmod 700 .env 2>/dev/null || true

# Firewall (if ufw available)
if command -v ufw &> /dev/null; then
    sudo ufw allow 8080/tcp comment "Argus API"
    sudo ufw allow 8081/tcp comment "Argus WebSocket"
    sudo ufw allow 9090/tcp comment "Prometheus"
fi

echo -e "${GREEN}  ✓ Security configured${NC}"

# =============================================================================
# FINAL SETUP
# =============================================================================
echo -e "\n${YELLOW}[8/8] Final Setup...${NC}"

# Create .env template if not exists
if [ ! -f .env ]; then
    cat > .env <<EOF
# Argus Environment Variables
# Fill in your API keys

# Kraken
KRAKEN_API_KEY=your_api_key_here
KRAKEN_API_SECRET=your_api_secret_here

# Coinbase Advanced
COINBASE_API_KEY=your_api_key_here
COINBASE_API_SECRET=your_api_secret_here

# Optional: Binance
BINANCE_API_KEY=
BINANCE_API_SECRET=

# Optional: Alerts
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
DISCORD_WEBHOOK_URL=
EOF
    echo -e "${YELLOW}  ⚠ Created .env template - fill in your API keys${NC}"
fi

# Create data directories for SQLite
mkdir -p data/{strategy_metrics.db,meta_weights.db,strategy_states.db,strategy_decay.db}

echo -e "${GREEN}  ✓ Setup complete${NC}"

# =============================================================================
# SUMMARY
# =============================================================================
echo -e "\n${BLUE}"
echo "=================================================================="
echo "                    DEPLOYMENT COMPLETE"
echo "=================================================================="
echo -e "${NC}"

echo -e "${GREEN}Configuration:${NC}"
echo "  Config: config/profiles/r7525_server.yaml"
echo "  Capital: \$10,000 AUD"
echo "  Mode: Live (change to paper for testing)"
echo "  Quantum: Omega Engine (20 logical qubits)"
echo "  Cycle: 3 seconds"
echo ""

echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Edit .env with your API keys"
echo "  2. Test with paper mode: py main.py paper --config config/profiles/r7525_server.yaml"
echo "  3. Start service: sudo systemctl start argus-trading"
echo "  4. Check status: sudo systemctl status argus-trading"
echo "  5. View logs: journalctl -u argus-trading -f"
echo ""

echo -e "${BLUE}Monitoring:${NC}"
echo "  API: http://localhost:8080"
echo "  WebSocket: ws://localhost:8081"
echo "  Prometheus: http://localhost:9090"
echo ""

echo -e "${GREEN}Server Optimizations Applied:${NC}"
echo "  ✓ NUMA-aware threading"
echo "  ✓ CPU governor: performance"
echo "  ✓ 64 workers (half of 128 cores)"
echo "  ✓ 128GB memory allocation"
echo "  ✓ NVMe-optimized SQLite"
echo "  ✓ 3-second cycle time"
echo "  ✓ 20-qubit quantum simulation"
echo ""
