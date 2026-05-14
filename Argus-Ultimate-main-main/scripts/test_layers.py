#!/usr/bin/env python3
"""
Run tests for each layer independently to verify isolation.

Usage: py scripts/test_layers.py
"""
import subprocess
import sys


def run(name: str, cmd: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    passed = result.returncode == 0
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    return passed


def main():
    results = []

    results.append(run(
        "Layer 2: Core Modules",
        f"{sys.executable} -m pytest tests/test_structural_improvements.py tests/test_quant_principles.py -q",
    ))

    results.append(run(
        "Layer 3: Institutional Layer",
        f"{sys.executable} -m pytest tests/test_institutional_layer.py tests/test_assimilation.py -q",
    ))

    results.append(run(
        "Module Isolation",
        f"{sys.executable} -m pytest tests/test_module_isolation.py -q",
    ))

    results.append(run(
        "E2E Pipeline",
        f"{sys.executable} -m pytest tests/test_e2e_pipeline.py -q",
    ))

    results.append(run(
        "Advisory Gates",
        f"{sys.executable} -m pytest tests/test_argus_next_level.py -q",
    ))

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {sum(results)}/{len(results)} layers pass")
    print(f"{'='*60}")

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
