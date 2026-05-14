"""scripts/health_check.py

Post-deploy health checker for Argus Ultimate.
Can be run standalone or called from CI/CD.

Usage:
    python scripts/health_check.py [--host HOST] [--port PORT] [--timeout SECONDS]

Exits 0 on success, 1 on failure.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8080
DEFAULT_TIMEOUT = 10


def check_endpoint(url: str, timeout: int) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode()
            return {"status": resp.status, "body": json.loads(body)}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "error": str(e)}
    except Exception as e:
        return {"status": 0, "error": str(e)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Argus health check")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    checks = {
        "/health": "Health",
        "/status": "Status",
    }

    results: dict[str, Any] = {}
    failed = 0

    for path, label in checks.items():
        url = base + path
        result = check_endpoint(url, args.timeout)
        results[path] = result
        ok = result["status"] == 200
        if not ok:
            failed += 1
        if not args.json:
            mark = "✓" if ok else "✗"
            print(f"  [{mark}] {label:10s} {url:45s}  HTTP {result['status']}")

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print()
        if failed == 0:
            print("All checks passed.")
        else:
            print(f"{failed} check(s) failed.")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
