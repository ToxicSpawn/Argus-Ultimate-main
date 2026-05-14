# R730 Setup: Trading Host

Step-by-step setup for running Argus on a **Dell PowerEdge R730** (E5-2640 v4, 32GB+, H730, 16× 2.5") with **Solarflare 10G** into your two X460-G2 switches.

---

## 1. OS and base

- Install **Ubuntu Server 22.04 LTS** (or 24.04).
- Update: `sudo apt update && sudo apt upgrade -y`.
- Optional: add **64GB RAM** (or more) for headroom.

---

## 2. Solarflare NIC

- Install the **Solarflare SFN8522-PLUS** in a PCIe slot.
- Install drivers and OpenOnload (kernel bypass) per vendor docs, or use:
  ```bash
  sudo ./scripts/install-openonload.sh
  ```
- Confirm both ports are up and at 10G; set **MTU 9000** on the Solarflare interfaces to match `10gbe_network_config.json`.

---

## 3. 10G and network

- Cable: **Solarflare port 1 → Switch 1 port 49**, **Solarflare port 2 → Switch 2 port 49** (your 2× SFP+ server cables).
- Configure static IPs per your `10gbe_network_config.json` (e.g. 192.168.10.100/24 and 192.168.10.101/24), or use LACP if you bond the two ports.
- Ensure switches have MLAG + jumbo + QoS done (see [SWITCH_IMPROVEMENTS_CHECKLIST.md](SWITCH_IMPROVEMENTS_CHECKLIST.md)).

---

## 4. Linux tuning (optional but recommended)

Run the existing setup script (creates `argus` user, `/opt/argus`, kernel tuning, systemd service):

```bash
sudo ./scripts/setup-linux.sh
```

This will:

- Install Python 3.11, venv, and system deps.
- Apply kernel and CPU tuning for low latency.
- Create user `argus` and directories `/opt/argus/{data,logs,config}`.
- Install systemd service `argus.service` (paper mode by default).

After running it, copy the repo into `/opt/argus` and install dependencies (see step 5).

---

## 5. Deploy the bot

Option A – use the deploy script (from a clone on the R730 or from your workstation via rsync):

```bash
sudo ./scripts/deploy_production_linux.sh
```

Option B – manual:

```bash
# Clone (or rsync from your desktop)
sudo git clone https://github.com/ToxicSpawn/Argus-Ultimate.git /opt/argus/repo
sudo chown -R argus:argus /opt/argus/repo

# Symlink or copy so main.py and unified_config.yaml are in /opt/argus
sudo -u argus ln -s /opt/argus/repo/main.py /opt/argus/main.py
sudo -u argus ln -s /opt/argus/repo/unified_config.yaml /opt/argus/unified_config.yaml
# Or copy the whole repo to /opt/argus and run from there

sudo -u argus /opt/argus/venv/bin/pip install -r /opt/argus/repo/requirements.txt

# Config and env
sudo -u argus cp /opt/argus/repo/.env.example /opt/argus/.env
# Edit /opt/argus/.env with API keys (use nano or vim)
# Optional: copy unified_config.production.yaml to /opt/argus/ and use --config

# Validate
cd /opt/argus && sudo -u argus /opt/argus/venv/bin/python main.py validate
```

---

## 6. Systemd service

The script in step 4 creates `/etc/systemd/system/argus.service` running:

- `WorkingDirectory=/opt/argus`
- `ExecStart=.../venv/bin/python -O main.py paper --capital 1000`

To use a specific config file, edit the service:

```ini
ExecStart=/opt/argus/venv/bin/python -O main.py paper --capital 1000 --config /opt/argus/unified_config.yaml
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl start argus
sudo systemctl enable argus
sudo journalctl -u argus -f
```

---

## 7. Switching to live

- When ready for live: put real API keys in `/opt/argus/.env`.
- Change the service to `main.py live --capital 1000` (and add `--yes-live` only if you use non-interactive startup).
- Restart: `sudo systemctl restart argus`.

---

## 8. Backups

Run the backup script daily (cron):

```bash
0 2 * * * /opt/argus/repo/scripts/backup_config_and_logs.sh /opt/argus /opt/argus/backups
```

Adjust paths if your layout differs.

---

## 9. Monitoring

- Run Grafana + Prometheus on the R730 (or on your desktop) and point them at the bot.
- Optional: SNMP/sFlow from the two switches to the same monitoring stack.

See [UNIFIED_RUNBOOK.md](../UNIFIED_RUNBOOK.md) and `docker-compose --profile monitoring` if you use Docker.
