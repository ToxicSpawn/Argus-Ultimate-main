#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Argus Linux Execution Island Apply Script
# -----------------------------------------------------------------------------
# Purpose:
# - Prepare deterministic execution profile on Linux hosts.
# - Write kernel/systemd/network tuning files.
# - Keep default mode dry-run (no host mutation unless --apply).
#
# Usage:
#   sudo ./ops/linux/apply_execution_island.sh --apply --iface enp3s0f0 \
#     --os-cpus 0-3 --exec-cpus 4-15 --cpu-isolation 4-15
# -----------------------------------------------------------------------------

set -euo pipefail

APPLY=0
IFACE="${IFACE:-eth0}"
OS_CPUS="${OS_CPUS:-0-3}"
EXEC_CPUS="${EXEC_CPUS:-4-15}"
CPU_ISOLATION="${CPU_ISOLATION:-4-15}"
ARGUS_USER="${ARGUS_USER:-argus}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --apply) APPLY=1; shift ;;
        --iface) IFACE="$2"; shift 2 ;;
        --os-cpus) OS_CPUS="$2"; shift 2 ;;
        --exec-cpus) EXEC_CPUS="$2"; shift 2 ;;
        --cpu-isolation) CPU_ISOLATION="$2"; shift 2 ;;
        --argus-user) ARGUS_USER="$2"; shift 2 ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ "${EUID}" -ne 0 ]]; then
    echo "[ERROR] Run as root (sudo)." >&2
    exit 1
fi

if [[ "${APPLY}" -eq 0 ]]; then
    echo "[INFO] Dry-run mode. No files will be written."
fi

SYSCTL_CONTENT=$(cat <<EOF
# Argus deterministic low-latency profile
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.core.netdev_max_backlog = 300000
net.core.somaxconn = 65535
net.ipv4.tcp_fastopen = 3
net.ipv4.tcp_slow_start_after_idle = 0
net.ipv4.tcp_low_latency = 1
vm.swappiness = 1
vm.dirty_ratio = 10
vm.dirty_background_ratio = 5
EOF
)

SERVICE_DROPIN_CONTENT=$(cat <<EOF
[Service]
CPUAffinity=${EXEC_CPUS}
Nice=-10
IOSchedulingClass=realtime
IOSchedulingPriority=0
LimitMEMLOCK=infinity
Environment=ARGUS_OS_CPUS=${OS_CPUS}
Environment=ARGUS_EXEC_CPUS=${EXEC_CPUS}
Environment=ARGUS_CPU_ISOLATION=${CPU_ISOLATION}
EOF
)

IRQ_SCRIPT_CONTENT=$(cat <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

IFACE="${1:-eth0}"
CPU_LIST="${2:-4-15}"

if ! [[ -e "/proc/interrupts" ]]; then
    exit 1
fi

# Expand CPU list token into an array (supports comma-separated ranges).
mapfile -t CPUS < <(
python3 - "${CPU_LIST}" <<'PY'
import sys
raw = sys.argv[1]
out = []
for part in raw.split(","):
    part = part.strip()
    if not part:
        continue
    if "-" in part:
        a, b = part.split("-", 1)
        out.extend(range(int(a), int(b) + 1))
    else:
        out.append(int(part))
for c in out:
    print(c)
PY
)

if [[ ${#CPUS[@]} -eq 0 ]]; then
    exit 1
fi

idx=0
while read -r irq; do
    cpu="${CPUS[$((idx % ${#CPUS[@]}))]}"
    echo "${cpu}" >"/proc/irq/${irq}/smp_affinity_list"
    idx=$((idx + 1))
done < <(grep -E "${IFACE}(|-TxRx|-(rx|tx)-[0-9]+)" /proc/interrupts | awk -F: '{print $1}' | tr -d ' ')
EOF
)

IRQ_SERVICE_CONTENT=$(cat <<EOF
[Unit]
Description=Pin NIC IRQs for Argus execution island
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/argus-pin-irqs.sh ${IFACE} ${EXEC_CPUS}
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
)

echo "[PLAN] iface=${IFACE} os_cpus=${OS_CPUS} exec_cpus=${EXEC_CPUS} isolation=${CPU_ISOLATION}"
echo "[PLAN] Will write:"
echo "  - /etc/sysctl.d/99-argus-lowlatency.conf"
echo "  - /etc/systemd/system/argus.service.d/override.conf"
echo "  - /usr/local/sbin/argus-pin-irqs.sh"
echo "  - /etc/systemd/system/argus-pin-irqs.service"
echo "[PLAN] Recommended GRUB kernel args:"
echo "  isolcpus=${CPU_ISOLATION} nohz_full=${CPU_ISOLATION} rcu_nocbs=${CPU_ISOLATION}"

if [[ "${APPLY}" -eq 0 ]]; then
    exit 0
fi

mkdir -p /etc/systemd/system/argus.service.d
mkdir -p /usr/local/sbin

printf "%s\n" "${SYSCTL_CONTENT}" >/etc/sysctl.d/99-argus-lowlatency.conf
sysctl -p /etc/sysctl.d/99-argus-lowlatency.conf >/dev/null

printf "%s\n" "${SERVICE_DROPIN_CONTENT}" >/etc/systemd/system/argus.service.d/override.conf

printf "%s\n" "${IRQ_SCRIPT_CONTENT}" >/usr/local/sbin/argus-pin-irqs.sh
chmod +x /usr/local/sbin/argus-pin-irqs.sh
printf "%s\n" "${IRQ_SERVICE_CONTENT}" >/etc/systemd/system/argus-pin-irqs.service

if systemctl list-unit-files | grep -q '^irqbalance\.service'; then
    systemctl stop irqbalance || true
    systemctl disable irqbalance || true
fi

# Force performance governor where available.
for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [[ -f "${gov}" ]] || continue
    echo "performance" >"${gov}" || true
done

systemctl daemon-reload
systemctl enable argus-pin-irqs.service || true
systemctl restart argus-pin-irqs.service || true
systemctl restart argus.service || true

echo "[DONE] Applied execution island profile."
echo "[NEXT] Persist CPU isolation in GRUB and reboot:"
echo "       sudo update-grub && sudo reboot"
