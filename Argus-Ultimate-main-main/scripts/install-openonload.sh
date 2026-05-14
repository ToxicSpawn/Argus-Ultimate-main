#!/bin/bash
# =============================================================================
# Solarflare OpenOnload Installation Script
# =============================================================================
# Kernel bypass networking for ultra-low latency trading
#
# Requirements:
# - Solarflare SFN8522-PLUS or compatible NIC
# - Ubuntu 22.04 LTS / Debian 12
# - Root access
#
# Run with: sudo ./scripts/install-openonload.sh
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check root
if [[ $EUID -ne 0 ]]; then
   log_error "This script must be run as root (sudo)"
   exit 1
fi

ONLOAD_VERSION="8.1.2.26"  # Update to latest stable
WORK_DIR="/tmp/openonload-install"

log_info "=========================================="
log_info "Solarflare OpenOnload Installation"
log_info "Version: $ONLOAD_VERSION"
log_info "=========================================="

# =============================================================================
# 1. Check for Solarflare NIC
# =============================================================================
log_info "Checking for Solarflare NIC..."

if ! lspci | grep -i "solarflare\|xilinx" > /dev/null; then
    log_error "No Solarflare/Xilinx NIC detected!"
    log_error "Please ensure your SFN8522-PLUS is properly installed."
    exit 1
fi

log_info "Solarflare NIC detected:"
lspci | grep -i "solarflare\|xilinx"

# =============================================================================
# 2. Install Build Dependencies
# =============================================================================
log_info "Installing build dependencies..."

apt-get update
apt-get install -y \
    build-essential \
    linux-headers-$(uname -r) \
    libc6-dev \
    libcap-dev \
    libnl-3-dev \
    libnl-route-3-dev \
    python3-dev \
    autoconf \
    automake \
    libtool \
    pkg-config \
    libpcap-dev \
    ethtool \
    wget \
    tar

# =============================================================================
# 3. Download OpenOnload
# =============================================================================
log_info "Downloading OpenOnload $ONLOAD_VERSION..."

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# AMD/Xilinx now owns Solarflare - download from their site
ONLOAD_URL="https://www.xilinx.com/content/dam/xilinx/publications/solarflare/onload/openonload-${ONLOAD_VERSION}.tgz"

# Alternative: GitHub releases
# ONLOAD_URL="https://github.com/Xilinx-CNS/onload/archive/refs/tags/v${ONLOAD_VERSION}.tar.gz"

if ! wget -q "$ONLOAD_URL" -O openonload.tgz; then
    log_warn "Could not download from Xilinx, trying GitHub..."
    wget "https://github.com/Xilinx-CNS/onload/archive/refs/tags/v${ONLOAD_VERSION}.tar.gz" -O openonload.tgz
fi

tar -xzf openonload.tgz
cd openonload-* || cd onload-*

# =============================================================================
# 4. Build OpenOnload
# =============================================================================
log_info "Building OpenOnload (this may take several minutes)..."

cd scripts
./onload_build --kernel
./onload_install --kernelfiles

# Install user-space libraries
./onload_install --userfiles

# =============================================================================
# 5. Load Kernel Modules
# =============================================================================
log_info "Loading kernel modules..."

# Unload existing sfc driver if loaded
modprobe -r sfc 2>/dev/null || true

# Load OpenOnload modules
modprobe sfc_resource
modprobe sfc
modprobe onload

# Verify modules loaded
if lsmod | grep -q onload; then
    log_info "OpenOnload modules loaded successfully."
else
    log_error "Failed to load OpenOnload modules!"
    exit 1
fi

# =============================================================================
# 6. Configure Persistent Loading
# =============================================================================
log_info "Configuring persistent module loading..."

cat > /etc/modules-load.d/openonload.conf << 'EOF'
# Solarflare OpenOnload modules
sfc_resource
sfc
onload
EOF

# Create modprobe config
cat > /etc/modprobe.d/sfc.conf << 'EOF'
# Solarflare driver options for low latency
options sfc interrupt_mode=0
options sfc rx_ring=4096
options sfc tx_ring=4096
EOF

# =============================================================================
# 7. Configure Network Interface
# =============================================================================
log_info "Configuring Solarflare interface..."

# Find Solarflare interface
SF_IFACE=$(ip link | grep -B1 "link/ether" | grep -E "^[0-9]+:" | awk -F: '{print $2}' | tr -d ' ' | while read iface; do
    driver=$(readlink /sys/class/net/$iface/device/driver 2>/dev/null | xargs basename 2>/dev/null)
    if [[ "$driver" == "sfc" ]]; then
        echo $iface
        break
    fi
done)

if [[ -n "$SF_IFACE" ]]; then
    log_info "Found Solarflare interface: $SF_IFACE"

    # Optimize interface settings
    ethtool -G $SF_IFACE rx 4096 tx 4096 2>/dev/null || true
    ethtool -C $SF_IFACE rx-usecs 0 tx-usecs 0 2>/dev/null || true
    ethtool -K $SF_IFACE gro off lro off 2>/dev/null || true

    log_info "Interface optimized for low latency."
else
    log_warn "Could not identify Solarflare interface automatically."
fi

# =============================================================================
# 8. Create Onload Profile for Trading
# =============================================================================
log_info "Creating trading-optimized Onload profile..."

mkdir -p /etc/sysconfig
cat > /etc/sysconfig/openonload << 'EOF'
# OpenOnload configuration for low-latency trading

# Use spinning instead of interrupts (lower latency, higher CPU)
EF_POLL_USEC=100000

# Spin for up to 100ms waiting for data
EF_SPIN_USEC=100000

# Disable interrupt coalescing
EF_INT_DRIVEN=0

# Use huge pages if available
EF_USE_HUGE_PAGES=1

# TCP optimizations
EF_TCP_FASTSTART=0
EF_TCP_TIME_WAIT_ASSASSINATION=1

# Disable nagle
EF_TCP_NODELAY=1

# Bypass kernel for maximum speed
EF_UL_SELECT=1
EF_UL_POLL=1
EF_UL_EPOLL=1
EOF

# =============================================================================
# 9. Create Helper Scripts
# =============================================================================
log_info "Creating helper scripts..."

# Script to run application with Onload
cat > /usr/local/bin/onload-run << 'EOF'
#!/bin/bash
# Run an application with OpenOnload kernel bypass
# Usage: onload-run <command>

source /etc/sysconfig/openonload 2>/dev/null

exec onload "$@"
EOF
chmod +x /usr/local/bin/onload-run

# Script to check Onload status
cat > /usr/local/bin/onload-status << 'EOF'
#!/bin/bash
# Check OpenOnload status

echo "=== Kernel Modules ==="
lsmod | grep -E "sfc|onload" || echo "No modules loaded"

echo ""
echo "=== Solarflare Interfaces ==="
for iface in $(ls /sys/class/net/); do
    driver=$(readlink /sys/class/net/$iface/device/driver 2>/dev/null | xargs basename 2>/dev/null)
    if [[ "$driver" == "sfc" ]]; then
        echo "$iface: $driver"
        ethtool $iface 2>/dev/null | grep -E "Speed|Link"
    fi
done

echo ""
echo "=== Onload Stacks ==="
onload_stackdump 2>/dev/null || echo "No active stacks"

echo ""
echo "=== Onload Version ==="
onload --version 2>/dev/null || echo "Onload not in PATH"
EOF
chmod +x /usr/local/bin/onload-status

# =============================================================================
# 10. Update Argus Service to Use Onload
# =============================================================================
log_info "Updating Argus service for Onload..."

if [[ -f /etc/systemd/system/argus.service ]]; then
    # Backup original
    cp /etc/systemd/system/argus.service /etc/systemd/system/argus.service.backup

    # Update to use onload
    sed -i 's|ExecStart=/opt/argus/venv/bin/python|ExecStart=/usr/bin/onload /opt/argus/venv/bin/python|' /etc/systemd/system/argus.service

    # Add environment variables
    sed -i '/\[Service\]/a Environment=EF_POLL_USEC=100000\nEnvironment=EF_SPIN_USEC=100000\nEnvironment=EF_TCP_NODELAY=1' /etc/systemd/system/argus.service

    systemctl daemon-reload
    log_info "Argus service updated to use OpenOnload."
else
    log_warn "Argus service not found. Run setup-linux.sh first."
fi

# =============================================================================
# 11. Verify Installation
# =============================================================================
log_info "Verifying installation..."

echo ""
onload --version

echo ""
log_info "Running quick latency test..."
onload_latency --probe 2>/dev/null || log_warn "Latency test skipped (needs network config)"

# =============================================================================
# Cleanup
# =============================================================================
log_info "Cleaning up..."
rm -rf "$WORK_DIR"

# =============================================================================
# Summary
# =============================================================================
echo ""
log_info "=========================================="
log_info "OpenOnload Installation Complete!"
log_info "=========================================="
echo ""
echo "Usage:"
echo "  Run any application with kernel bypass:"
echo "    onload-run python my_script.py"
echo ""
echo "  Check status:"
echo "    onload-status"
echo ""
echo "  Argus will now automatically use OpenOnload."
echo "  Restart with: systemctl restart argus"
echo ""
echo "Expected latency improvement: 30-50% reduction"
echo ""
