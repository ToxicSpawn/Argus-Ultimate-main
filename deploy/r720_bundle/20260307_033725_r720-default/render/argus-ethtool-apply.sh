#!/usr/bin/env bash
set -euo pipefail

for IFACE in "enp3s0f0" "enp3s0f1"; do
  ethtool -G "${IFACE}" rx 4096 tx 4096 || true
  ethtool -C "${IFACE}" rx-usecs 0 tx-usecs 0 || true
  ethtool -K "${IFACE}" gro off lro off gso off tso off || true
  ip link set "${IFACE}" mtu "9000" || true
done
