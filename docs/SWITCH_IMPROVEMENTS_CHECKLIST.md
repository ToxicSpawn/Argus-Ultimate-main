# Switch Improvements Checklist – Extreme X460-G2 (Dual MLAG)

**Your switches:** 2× Extreme Networks **X460-G2-48P-10GE4** (P/N 16704) — 48× 1G PoE+ + 4× 10GbE SFP+ each.

Best improvements for HFT / low-latency trading with this pair.

---

## 1. Hardware (physical)

| Improvement | Why |
|-------------|-----|
| **2× SFP+ cables for ISC** | Server→switch uses 2× cables; you need **2× 10Gb SFP+** for switch-to-switch (port 47↔47, 48↔48). Same type as Extreme 10305 3 m is ideal. |
| **Cable length** | Keep ISC and server uplinks short (1–3 m). |
| **PSU** | Each switch already has a 712W PSU. Optional: second PSU per chassis (e.g. 10951/10952) for HA. |
| **Cooling / airflow** | Ensure good airflow in the rack. |

---

## 2. Software / CLI tuning (both switches)

- `configure forwarding-mode cut-through`
- `enable jumbo-frame ports all`
- `configure qosprofile priority 7 buffer-pool percent 80`
- `configure qosprofile priority 7 schedule strict`
- Set server uplink (e.g. port 49) as edge; NTP on both switches.

---

## 3. Rack additions (optional)

- Dual PDUs, UPS, blanking panels, management switch (1G), NTP in rack.
- SNMP/sFlow + syslog collector; cable managers + labels; spare SFP+ DACs.

---

## 4. Switch modules (optional)

| Module | Part # | Benefit |
|--------|--------|---------|
| Second PSU | 10951 / 10952 | Power redundancy. |
| VIM-2x | 16711 | +2× 10G SFP+ per switch (6 total). |
| TM-CLK | 16715 | PTP/SyncE timing (±2 ns). |
| Spare DAC | — | Fast replacement for ISC or server link. |

---

## 5. Other servers for the 2 switches

- **Second trading server** → 2× 10G (needs VIM-2x for redundant links).
- **Monitoring** (Grafana, Prometheus) → 2× 1G.
- **Management / jump host** → 2× 1G.
- **Staging, NTP, backup** → 1G or 10G as needed.

---

## References

- Repo: `10gbe_network_config.json`, `10gbe_network_setup.py`, `10GBE_ADAPTIVE_README.md`
- Extreme: [X460-G2](https://www.extremenetworks.com/products/switches/extremexos-switches/x460-g2)
