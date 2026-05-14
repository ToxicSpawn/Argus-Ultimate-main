# R740 Day-0 Acceptance

Use this on first boot after assembling the R740.

## 1) Capture host facts

Run on the R740 host:

```bash
python3 scripts/r740_capture_host_facts.py --output reports/infra/r740_host_facts_latest.json
```

## 2) Validate against acceptance spec

```bash
python3 scripts/r740_hardware_acceptance.py \
  --spec docs/hardware/R740_ACCEPTANCE_SPEC.yaml \
  --facts reports/infra/r740_host_facts_latest.json \
  --output reports/infra/r740_acceptance_latest.json
```

`PASS` means the host meets minimum expected CPU/memory/network/storage baseline.

## 3) NVMe U.2 drive health (use nvme-cli, not smartctl)

R740 primary storage is NVMe U.2 hot-swap. Prefer `nvme-cli` over `smartctl` for accurate health and endurance telemetry:

```bash
sudo nvme list
sudo nvme smart-log /dev/nvme0n1
sudo nvme smart-log /dev/nvme1n1
sudo nvme id-ctrl /dev/nvme0n1 | grep -E '(mn|fr|sn)'
```

Check for:
- `critical_warning` = 0x00
- `percentage_used` < 80
- `media_errors` = 0
- `num_err_log_entries` stable (not climbing rapidly)
- `temperature` within 20-70 C operating range

Set the I/O scheduler to `none` (NVMe devices handle queueing in hardware):

```bash
for d in /sys/block/nvme*n*/queue/scheduler; do echo none | sudo tee "$d"; done
```

## 4) Apply execution-island tuning

```bash
sudo ./ops/linux/apply_execution_island.sh --apply --iface ens1f0 --os-cpus 0-7 --exec-cpus 8-31 --cpu-isolation 8-31
```

Note the wider exec-island range — R740's Xeon Scalable parts typically expose 16-40 logical CPUs per socket.

## 5) Verify infra and preflight

```bash
python3 scripts/infra_verify_host.py --iface ens1f0 --strict --output reports/infra/verification_latest.json
python3 scripts/infra_preflight.py --report reports/infra/verification_latest.json --output reports/infra/infra_preflight_latest.json --max-clock-offset-us 250
```

Safety:
- Keep live trading gated and off by default.
- Prefer halting safely over uncertain recovery in ambiguous states.
