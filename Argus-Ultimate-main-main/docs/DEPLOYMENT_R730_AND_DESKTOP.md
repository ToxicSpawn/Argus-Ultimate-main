# Deployment: R730 Trading Host + Desktop Dev

This guide ties together your hardware and the Argus bot: **Dell R730** as the 24/7 trading host, **desktop** (285K + RTX 5080) for dev/backtest/ML, and **2× Extreme X460-G2** switches.

---

## 1. Roles

| Machine | Role | Network |
|---------|------|---------|
| **Dell R730** | 24/7 trading host (paper then live) | Solarflare 10G → switch 1 & 2 (port 49 each) |
| **Desktop** (285K, RTX 5080) | Dev, backtest, ML, optional paper runs | 1G OK; optional 10G later |
| **2× X460-G2** | 10G fabric, MLAG | ISC: 2× SFP+ (47↔47, 48↔48) |

---

## 2. Order of operations

1. **Switches:** Cable ISC (2× SFP+), power, configure MLAG + jumbo + QoS (see [SWITCH_IMPROVEMENTS_CHECKLIST.md](SWITCH_IMPROVEMENTS_CHECKLIST.md)).
2. **R730:** Install OS (Ubuntu 22.04 LTS recommended), add RAM to 64GB if possible, install Solarflare + drivers.
3. **R730 network:** Cable Solarflare port 1 → switch 1 port 49, port 2 → switch 2 port 49; set MTU 9000, static IPs per `10gbe_network_config.json`.
4. **R730 bot:** Run [R730_SETUP.md](R730_SETUP.md) and `scripts/deploy_production_linux.sh` (or manual clone + pip + config).
5. **Desktop:** Clone repo, install deps, use for development and backtest; optionally enable GPU for ML.
6. **Monitoring:** Run Grafana + Prometheus (e.g. on R730 or desktop); point at bot and optionally switches (SNMP/sFlow).
7. **Go live** when paper results and risk settings are acceptable.

If the server is not built yet, use [R740_PREBUILD_PLAYBOOK.md](R740_PREBUILD_PLAYBOOK.md) to generate and validate deployment bundles in advance.

---

## 3. One source of truth

- **Repo:** `https://github.com/ToxicSpawn/Argus-Ultimate.git`
- **Config:** `unified_config.yaml` (and optional `unified_config.production.yaml` on R730 for overrides).
- **Secrets:** `.env` on each machine (never committed). On R730 use `/opt/argus/.env`.

---

## 4. Quick reference

| Task | Where | Command / doc |
|------|--------|----------------|
| R730 OS + 10G + bot | R730 | [R730_SETUP.md](R730_SETUP.md) |
| Deploy bot on R730 | R730 | `sudo -u argus /opt/argus/venv/bin/python main.py validate` then `systemctl start argus` |
| Dev / backtest | Desktop | `python main.py paper --capital 1000` or `python main.py validate` |
| Switch tuning | Both switches | [SWITCH_IMPROVEMENTS_CHECKLIST.md](SWITCH_IMPROVEMENTS_CHECKLIST.md) |
| Backups | R730 | `scripts/backup_config_and_logs.sh` (cron daily) |
| Monitoring | R730 or desktop | `docker-compose --profile monitoring up -d` or see [UNIFIED_RUNBOOK.md](../UNIFIED_RUNBOOK.md) |

---

## 5. Wiring (recap)

```
[Switch 1]  port 47 ←── SFP+ ──→ port 47 [Switch 2]
[Switch 1]  port 48 ←── SFP+ ──→ port 48 [Switch 2]
[Switch 1]  port 49 ←── SFP+ ──→ Solarflare port 1 [R730]
[Switch 2]  port 49 ←── SFP+ ──→ Solarflare port 2 [R730]
```

Desktop can plug into any 1G port on either switch for management/dev.

---

## 6. Monitoring

- **With Docker (on R730 or desktop):**  
  `docker-compose --profile monitoring up -d`  
  Then open Grafana at http://localhost:3000 (default admin/argus123) and Prometheus at http://localhost:9090.
- **Without Docker:** Install Prometheus and Grafana on the host; use `prometheus.yml` and `grafana/provisioning` from the repo.
- Optionally add SNMP/sFlow from the two X460-G2 switches to the same monitoring stack (see [SWITCH_IMPROVEMENTS_CHECKLIST.md](SWITCH_IMPROVEMENTS_CHECKLIST.md)).
