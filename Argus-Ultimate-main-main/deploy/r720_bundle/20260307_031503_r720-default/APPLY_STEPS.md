# R720 Apply Steps (Generated)

1. Review generated files under `render/`.
2. Install files on host:
   - `99-argus-lowlatency.conf` -> `/etc/sysctl.d/`
   - `argus.service.override.conf` -> `/etc/systemd/system/argus.service.d/override.conf`
   - `argus-pin-irqs.sh` -> `/usr/local/sbin/argus-pin-irqs.sh`
   - `argus-pin-irqs.service` -> `/etc/systemd/system/argus-pin-irqs.service`
   - `50-argus-10gbe.yaml` -> `/etc/netplan/50-argus-10gbe.yaml`
   - `chrony.argus.conf` -> `/etc/chrony/chrony.conf` (merge carefully)
   - `ptp4l.argus.conf` -> `/etc/linuxptp/ptp4l.conf`
   - `phc2sys.argus.service` -> `/etc/systemd/system/phc2sys.argus.service`
3. Apply kernel args from `grub_cmdline.txt` and run `update-grub`.
4. Restart networking/time services and run host verification:
   `python3 scripts/infra_verify_host.py --iface enp3s0f0 --strict`
5. Generate infra preflight report and keep live gated until PASS:
   `python3 scripts/infra_preflight.py --report reports/infra/verification_latest.json --output reports/infra/infra_preflight_latest.json --max-clock-offset-us 250`

Safety:
- Keep live trading gated and off by default.
- Prefer halting safely over uncertain recovery in ambiguous states.
