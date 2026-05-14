#!/bin/bash
# =============================================================================
# Argus Trading System - Linux Performance Setup
# =============================================================================
# This script optimizes a Linux system for low-latency trading.
# Run with: sudo ./scripts/setup-linux.sh
#
# Designed for: Ubuntu 22.04 LTS / Debian 12
# Hardware: Intel Core Ultra 9 + Solarflare NIC
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check root
if [[ $EUID -ne 0 ]]; then
   log_error "This script must be run as root (sudo)"
   exit 1
fi

log_info "=========================================="
log_info "Argus Trading System - Linux Setup"
log_info "=========================================="

# =============================================================================
# 1. Install Dependencies
# =============================================================================
log_info "Installing system dependencies..."

apt-get update
apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    build-essential \
    linux-tools-common \
    linux-tools-generic \
    cpufrequtils \
    tuned \
    irqbalance \
    numactl \
    htop \
    iotop \
    net-tools

# =============================================================================
# 2. Install Low-Latency Kernel (Optional)
# =============================================================================
log_info "Checking for low-latency kernel..."

if ! dpkg -l | grep -q linux-lowlatency; then
    read -p "Install low-latency kernel? (recommended) [y/N]: " install_lowlat
    if [[ "$install_lowlat" =~ ^[Yy]$ ]]; then
        apt-get install -y linux-lowlatency
        log_info "Low-latency kernel installed. Reboot required."
    fi
else
    log_info "Low-latency kernel already installed."
fi

# =============================================================================
# 3. CPU Performance Tuning
# =============================================================================
log_info "Configuring CPU for performance..."

# Set CPU governor to performance
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "performance" > "$cpu" 2>/dev/null || true
done

# Disable CPU frequency scaling
systemctl disable ondemand 2>/dev/null || true

# Create persistent config
cat > /etc/default/cpufrequtils << 'EOF'
GOVERNOR="performance"
EOF

# Disable turbo boost variance (more consistent latency)
# Note: Comment out if you prefer max speed over consistency
# echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true

log_info "CPU governor set to performance mode."

# =============================================================================
# 4. Kernel Parameters for Low Latency
# =============================================================================
log_info "Applying kernel parameters..."

cat > /etc/sysctl.d/99-argus-trading.conf << 'EOF'
# =============================================================================
# Argus Trading System - Kernel Tuning
# =============================================================================

# Network performance
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.core.rmem_default = 16777216
net.core.wmem_default = 16777216
net.core.netdev_max_backlog = 300000
net.core.somaxconn = 65535

# TCP tuning
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 87380 134217728
net.ipv4.tcp_congestion_control = bbr
net.ipv4.tcp_fastopen = 3
net.ipv4.tcp_slow_start_after_idle = 0
net.ipv4.tcp_no_metrics_save = 1
net.ipv4.tcp_timestamps = 1
net.ipv4.tcp_sack = 1
net.ipv4.tcp_window_scaling = 1

# Reduce latency
net.ipv4.tcp_low_latency = 1

# Memory
vm.swappiness = 1
vm.dirty_ratio = 10
vm.dirty_background_ratio = 5

# Disable transparent huge pages (reduces latency variance)
# Note: Also handled via service below

# File handles
fs.file-max = 2097152
fs.nr_open = 2097152
EOF

sysctl -p /etc/sysctl.d/99-argus-trading.conf

log_info "Kernel parameters applied."

# =============================================================================
# 5. Disable Transparent Huge Pages (THP)
# =============================================================================
log_info "Disabling Transparent Huge Pages..."

cat > /etc/systemd/system/disable-thp.service << 'EOF'
[Unit]
Description=Disable Transparent Huge Pages (THP)
DefaultDependencies=no
After=sysinit.target local-fs.target
Before=basic.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'echo never > /sys/kernel/mm/transparent_hugepage/enabled'
ExecStart=/bin/sh -c 'echo never > /sys/kernel/mm/transparent_hugepage/defrag'

[Install]
WantedBy=basic.target
EOF

systemctl daemon-reload
systemctl enable disable-thp
systemctl start disable-thp

log_info "THP disabled."

# =============================================================================
# 6. IRQ Affinity (for Solarflare NIC)
# =============================================================================
log_info "Configuring IRQ affinity..."

# Stop irqbalance (we'll manually pin IRQs)
systemctl stop irqbalance
systemctl disable irqbalance

# Create IRQ affinity script for Solarflare
cat > /usr/local/bin/set-solarflare-affinity.sh << 'EOF'
#!/bin/bash
# Pin Solarflare NIC interrupts to specific CPU cores
# Adjust CPU cores based on your setup

# Find Solarflare interfaces
for iface in $(ls /sys/class/net/ | grep -E '^(eth|enp|ens)'); do
    driver=$(readlink /sys/class/net/$iface/device/driver 2>/dev/null | xargs basename 2>/dev/null)
    if [[ "$driver" == "sfc" ]]; then
        echo "Found Solarflare interface: $iface"

        # Get IRQs for this interface
        irqs=$(grep "$iface" /proc/interrupts | awk '{print $1}' | tr -d ':')

        core=2  # Start pinning from core 2 (leave 0-1 for OS)
        for irq in $irqs; do
            echo $core > /proc/irq/$irq/smp_affinity_list
            echo "Pinned IRQ $irq to core $core"
            ((core++))
        done
    fi
done
EOF

chmod +x /usr/local/bin/set-solarflare-affinity.sh

# Create systemd service
cat > /etc/systemd/system/solarflare-affinity.service << 'EOF'
[Unit]
Description=Set Solarflare NIC IRQ Affinity
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/set-solarflare-affinity.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable solarflare-affinity

log_info "IRQ affinity configured."

# =============================================================================
# 7. CPU Isolation (for trading process)
# =============================================================================
log_info "Setting up CPU isolation..."

# Reserve cores 4-7 for Argus (adjust based on your CPU)
# This requires GRUB modification

GRUB_FILE="/etc/default/grub"
if ! grep -q "isolcpus" "$GRUB_FILE"; then
    log_warn "To isolate CPUs, add to GRUB_CMDLINE_LINUX in $GRUB_FILE:"
    log_warn '  isolcpus=4-7 nohz_full=4-7 rcu_nocbs=4-7'
    log_warn "Then run: update-grub && reboot"
else
    log_info "CPU isolation already configured in GRUB."
fi

# =============================================================================
# 8. Create Argus User and Directories
# =============================================================================
log_info "Creating argus user and directories..."

# Create user if doesn't exist
if ! id -u argus &>/dev/null; then
    useradd -m -s /bin/bash argus
    log_info "Created user: argus"
fi

# Create directories
mkdir -p /opt/argus/{data,logs,config}
chown -R argus:argus /opt/argus

# Set capabilities for low-latency (avoid running as root)
# Allows setting process priority without root
setcap cap_sys_nice+ep /usr/bin/python3.11 2>/dev/null || log_warn "Could not set capabilities on Python"

log_info "Directories created at /opt/argus/"

# =============================================================================
# 9. Systemd Service for Argus
# =============================================================================
log_info "Creating systemd service..."

cat > /etc/systemd/system/argus.service << 'EOF'
[Unit]
Description=Argus Trading System
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=argus
Group=argus
WorkingDirectory=/opt/argus
Environment=PYTHONUNBUFFERED=1
Environment=ARGUS_MODE=paper

# CPU affinity - run on isolated cores
CPUAffinity=4 5 6 7

# Process priority
Nice=-10
IOSchedulingClass=realtime
IOSchedulingPriority=0

# Memory locking (prevent swapping)
LimitMEMLOCK=infinity

ExecStart=/opt/argus/venv/bin/python -O main.py paper --capital 1000

Restart=always
RestartSec=5

# Logging
StandardOutput=append:/opt/argus/logs/argus.log
StandardError=append:/opt/argus/logs/argus-error.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
log_info "Systemd service created: argus.service"

# =============================================================================
# 10. Setup Python Virtual Environment
# =============================================================================
log_info "Setting up Python virtual environment..."

sudo -u argus bash << 'EOF'
cd /opt/argus
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel
EOF

log_info "Virtual environment created at /opt/argus/venv/"

# =============================================================================
# Summary
# =============================================================================
echo ""
log_info "=========================================="
log_info "Setup Complete!"
log_info "=========================================="
echo ""
echo "Next steps:"
echo "  1. Copy Argus code to /opt/argus/"
echo "  2. Install dependencies:"
echo "     sudo -u argus /opt/argus/venv/bin/pip install -r requirements.txt"
echo "  3. Configure environment variables in /opt/argus/.env"
echo "  4. Start with: systemctl start argus"
echo "  5. View logs: journalctl -u argus -f"
echo ""
echo "Optional:"
echo "  - Add 'isolcpus=4-7 nohz_full=4-7 rcu_nocbs=4-7' to GRUB for CPU isolation"
echo "  - Install Solarflare OpenOnload for kernel bypass"
echo "  - Reboot to apply all changes"
echo ""
