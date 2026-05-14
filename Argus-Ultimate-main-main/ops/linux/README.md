# Argus Linux Infra Hardening Pack

This pack implements a deterministic execution baseline for Linux trading hosts.

## 0) Generate host bundle before hardware is online

```bash
python3 scripts/r740_prepare_bundle.py --manifest docs/hardware/R740_PREBUILD_MANIFEST.yaml --output-root deploy/r740_bundle
python3 scripts/r740_bundle_check.py --bundle-root deploy/r740_bundle --output reports/infra/r740_bundle_check_latest.json
```

## 1) Apply execution-island profile (dry-run first)

```bash
sudo ./ops/linux/apply_execution_island.sh --iface enp3s0f0 --os-cpus 0-3 --exec-cpus 4-15 --cpu-isolation 4-15
```

Apply for real:

```bash
sudo ./ops/linux/apply_execution_island.sh --apply --iface enp3s0f0 --os-cpus 0-3 --exec-cpus 4-15 --cpu-isolation 4-15
```

## 2) Verify host determinism

```bash
python3 scripts/infra_verify_host.py --iface enp3s0f0 --output reports/infra/verification_latest.json
```

## 3) Fail-closed preflight gate

```bash
python3 scripts/infra_preflight.py --report reports/infra/verification_latest.json --max-clock-offset-us 250
```

## 4) Latency / jitter / fee-churn report

```bash
python3 scripts/infra_latency_report.py --db data/unified_trades.db --output-dir reports/infra
```

## 5) Audit replay bus export + replay

```bash
python3 scripts/export_audit_bus.py --db data/unified_trades.db --output logs/audit_bus_latest.jsonl --limit 5000
python3 scripts/replay_audit_bus.py --input logs/audit_bus_latest.jsonl --speed 0
```

## Notes

- Keep live trading gated and off by default.
- Prefer halting safely over uncertain auto-recovery.
- Internet exchange routing latency still dominates absolute round-trip time; this pack focuses on local jitter and operational safety.
