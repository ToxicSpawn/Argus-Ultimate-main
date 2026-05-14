# R740 Prebuild Playbook (Before Hardware Arrives)

This playbook lets you build and validate deployment artifacts now, so bring-up is fast when the R740 is on your desk.

## Goal

Produce a repeatable bundle containing:
- GRUB isolation flags (tuned for Xeon Scalable / Skylake-SP or Cascade Lake)
- sysctl low-latency profile
- systemd execution affinity override
- NIC IRQ pinning service/script
- netplan 10GbE/25GbE config
- chrony/PTP config stubs
- NVMe U.2 device preparation steps
- apply steps + integrity hashes

## Hardware Profile (R740 Baseline)

| Component | R740 Target |
|-----------|-------------|
| CPU | Intel Xeon Scalable (Gold/Silver, 1st/2nd gen) - e.g. Gold 6138/6248 |
| Memory | DDR4-2933 ECC RDIMM (24 DIMM slots, up to 3TB) |
| Primary Storage | NVMe U.2 (PCIe 3.0 x4 per drive, hot-swap via BOSS/front bays) |
| Secondary Storage | SAS/SATA SSDs on PERC H730P/H740P |
| Networking | 4x 1GbE onboard + rNDC 10GbE/25GbE (Intel X710/XXV710 preferred) |
| Power | Dual 750W/1100W/1600W Platinum PSUs (hot-plug) |
| Cooling | 6x hot-plug redundant fans (high-performance for GPU/low-latency profiles) |

R740 draws slightly more peak power than R720 but is materially more efficient per watt (2x) and delivers ~3x the per-core throughput at lower memory latency.

## 1) Edit manifest

Update:
- [docs/hardware/R740_PREBUILD_MANIFEST.yaml](docs/hardware/R740_PREBUILD_MANIFEST.yaml)

Set your planned:
- interfaces (`iface_primary`, `iface_secondary`) — typically `eno1np0` / `eno2np1` on Broadcom rNDC, `ens1f0` / `ens1f1` on Intel X710
- static IPs / gateway
- CPU split (`os_cpus`, `exec_cpus`, `cpu_isolation`) — with higher core counts, reserve `0-7` for OS and `8-N` for exec island
- NVMe U.2 device paths (`/dev/nvme0n1`, `/dev/nvme1n1`)

## 2) Generate bundle (Windows or Linux)

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\ops\windows\Build-R740Prep.ps1 -Manifest .\docs\hardware\R740_PREBUILD_MANIFEST.yaml -OutputRoot .\deploy\r740_bundle
```

Cross-platform via make target:

```powershell
powershell -ExecutionPolicy Bypass -File .\make.ps1 -Target r740_prep_bundle -R740Manifest .\docs\hardware\R740_PREBUILD_MANIFEST.yaml -R740OutputRoot .\deploy\r740_bundle
```

## 3) Validate generated bundle

```powershell
powershell -ExecutionPolicy Bypass -File .\ops\windows\Verify-R740Prep.ps1 -BundleRoot .\deploy\r740_bundle -Output .\reports\infra\r740_bundle_check_latest.json
```

Or:

```powershell
powershell -ExecutionPolicy Bypass -File .\make.ps1 -Target r740_bundle_check -R740OutputRoot .\deploy\r740_bundle
```

## 4) Run one-command prebuild suite

```powershell
powershell -ExecutionPolicy Bypass -File .\ops\windows\Run-R740PrebuildSuite.ps1 -Config .\unified_config.yaml -Manifest .\docs\hardware\R740_PREBUILD_MANIFEST.yaml -BundleRoot .\deploy\r740_bundle
```

This runs:
- strict config validation (`main.py validate`)
- bundle generation
- bundle integrity check (including hash verification)

Optional pre-hardware readiness lane:

```powershell
powershell -ExecutionPolicy Bypass -File .\make.ps1 -Target r740_prebuild_suite -Config .\unified_config.yaml -Profile prebuild -R740SuiteRunReadiness
```

`prebuild` is paper-only and keeps live gated/off by default.

## 5) Review before day-0

From latest bundle:
- `render/grub_cmdline.txt`
- `render/50-argus-10gbe.yaml`
- `render/argus.service.override.conf`
- `render/argus-pin-irqs.sh`
- `render/nvme-prep.sh` (NVMe U.2 namespace + scheduler prep)
- `APPLY_STEPS.md`

## 6) Day-0 on R740

1. Copy latest bundle to R740.
2. Apply configs to `/etc/...` paths listed in `APPLY_STEPS.md`.
3. Reboot (after GRUB update).
4. Run:
   - `python3 scripts/infra_verify_host.py --iface <iface> --strict`
   - `python3 scripts/infra_preflight.py ...`
   - `python3 scripts/r740_capture_host_facts.py --output reports/infra/r740_host_facts_latest.json`
   - `python3 scripts/r740_hardware_acceptance.py --spec docs/hardware/R740_ACCEPTANCE_SPEC.yaml --facts reports/infra/r740_host_facts_latest.json --output reports/infra/r740_acceptance_latest.json`
5. Keep live gated until infra preflight status is PASS.

Detailed day-0 sequence: `docs/hardware/R740_DAY0_ACCEPTANCE.md`.
