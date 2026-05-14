#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import socket
import sqlite3
import ssl
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _percentile(values: Iterable[float], q: float) -> float:
    xs = sorted(float(v) for v in values)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return float(xs[0])
    pos = max(0.0, min(1.0, float(q))) * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    w = pos - lo
    return float(xs[lo] * (1.0 - w) + xs[hi] * w)


def _safe_json(raw: Any) -> Dict[str, Any]:
    txt = str(raw or "").strip()
    if not txt:
        return {}
    try:
        out = json.loads(txt)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _stats(values: Iterable[float]) -> Dict[str, float]:
    xs = [float(v) for v in values]
    if not xs:
        return {
            "count": 0.0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "count": float(len(xs)),
        "mean": float(statistics.fmean(xs)),
        "min": float(min(xs)),
        "max": float(max(xs)),
        "p50": _percentile(xs, 0.50),
        "p90": _percentile(xs, 0.90),
        "p95": _percentile(xs, 0.95),
        "p99": _percentile(xs, 0.99),
    }


def run_ping_probe(host: str, count: int, timeout_ms: int) -> Dict[str, Any]:
    cmd = ["ping", str(host), "-n", str(max(1, int(count))), "-w", str(max(1, int(timeout_ms)))]
    result: Dict[str, Any] = {
        "host": str(host),
        "count": int(max(1, int(count))),
        "timeout_ms": int(max(1, int(timeout_ms))),
        "ok": False,
        "packet_loss_pct": 100.0,
        "rtt_ms": _stats([]),
        "raw_summary": "",
        "error": "",
    }
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as e:
        result["error"] = str(e)
        return result

    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    rtts: List[float] = []
    for m in re.finditer(r"time[=<]\s*(\d+)ms", out, flags=re.IGNORECASE):
        try:
            rtts.append(float(m.group(1)))
        except Exception:
            continue
    sent, recv, lost = 0, 0, 0
    m_sum = re.search(r"Packets:\s*Sent\s*=\s*(\d+),\s*Received\s*=\s*(\d+),\s*Lost\s*=\s*(\d+)", out, flags=re.IGNORECASE)
    if m_sum:
        sent = int(m_sum.group(1))
        recv = int(m_sum.group(2))
        lost = int(m_sum.group(3))
    else:
        sent = int(max(1, count))
        recv = int(len(rtts))
        lost = int(max(0, sent - recv))
    loss_pct = (float(lost) / float(max(1, sent))) * 100.0
    result["packet_loss_pct"] = float(loss_pct)
    result["rtt_ms"] = _stats(rtts)
    result["ok"] = bool(proc.returncode == 0 and loss_pct < 100.0)
    result["raw_summary"] = f"sent={sent} recv={recv} lost={lost}"
    if proc.returncode != 0 and not result["error"]:
        result["error"] = f"ping_return_code={proc.returncode}"
    return result


def _tls_http_probe(host: str, path: str, timeout_s: float) -> Dict[str, float]:
    t0 = time.perf_counter()
    infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    dns_ms = (time.perf_counter() - t0) * 1000.0
    addr = infos[0][4]

    t1 = time.perf_counter()
    sock = socket.create_connection(addr, timeout=timeout_s)
    tcp_ms = (time.perf_counter() - t1) * 1000.0
    try:
        ctx = ssl.create_default_context()
        t2 = time.perf_counter()
        with ctx.wrap_socket(sock, server_hostname=host) as ssock:
            _ = ssock.version()
        tls_ms = (time.perf_counter() - t2) * 1000.0
    finally:
        try:
            sock.close()
        except Exception:
            pass

    t3 = time.perf_counter()
    req = urllib.request.Request(f"https://{host}{path}", headers={"User-Agent": "argus-benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        _ = resp.read(256)
    http_ms = (time.perf_counter() - t3) * 1000.0
    return {
        "dns_ms": float(dns_ms),
        "tcp_ms": float(tcp_ms),
        "tls_ms": float(tls_ms),
        "http_ms": float(http_ms),
    }


def run_exchange_probe(host: str, path: str, attempts: int, timeout_s: float) -> Dict[str, Any]:
    dns_vals: List[float] = []
    tcp_vals: List[float] = []
    tls_vals: List[float] = []
    http_vals: List[float] = []
    errors: List[str] = []
    for _ in range(max(1, int(attempts))):
        try:
            sample = _tls_http_probe(host=host, path=path, timeout_s=float(timeout_s))
            dns_vals.append(float(sample["dns_ms"]))
            tcp_vals.append(float(sample["tcp_ms"]))
            tls_vals.append(float(sample["tls_ms"]))
            http_vals.append(float(sample["http_ms"]))
        except (socket.timeout, socket.gaierror, ConnectionError, urllib.error.URLError, TimeoutError) as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(str(e))
    return {
        "host": str(host),
        "path": str(path),
        "attempts": int(max(1, int(attempts))),
        "success_count": int(len(http_vals)),
        "error_count": int(len(errors)),
        "errors": list(errors[:5]),
        "dns_ms": _stats(dns_vals),
        "tcp_ms": _stats(tcp_vals),
        "tls_ms": _stats(tls_vals),
        "http_ms": _stats(http_vals),
        "ok": bool(len(http_vals) > 0),
    }


def run_speedtest_probe() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "available": False,
        "ok": False,
        "download_mbps": 0.0,
        "upload_mbps": 0.0,
        "latency_ms": 0.0,
        "error": "",
    }
    cmd = ["speedtest", "--format=json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=90)
    except FileNotFoundError:
        result["error"] = "speedtest_cli_not_found"
        return result
    except Exception as e:
        result["error"] = str(e)
        return result
    result["available"] = True
    if proc.returncode != 0:
        result["error"] = f"return_code={proc.returncode}"
        return result
    try:
        payload = json.loads(proc.stdout or "{}")
        down = float(payload.get("download", {}).get("bandwidth", 0.0) or 0.0) * 8.0 / 1_000_000.0
        up = float(payload.get("upload", {}).get("bandwidth", 0.0) or 0.0) * 8.0 / 1_000_000.0
        latency = float(payload.get("ping", {}).get("latency", 0.0) or 0.0)
        result["download_mbps"] = float(down)
        result["upload_mbps"] = float(up)
        result["latency_ms"] = float(latency)
        result["ok"] = bool(down > 0.0 and up >= 0.0)
    except Exception as e:
        result["error"] = str(e)
    return result


def collect_runtime_metrics(db_path: Path) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "db_exists": bool(db_path.exists()),
        "cycle_latency_ms": _stats([]),
        "decision_count": 0,
        "ws_reconnect_count": 0,
        "stale_book_count": 0,
        "recon_required_open": 0,
        "recon_halted_count": 0,
    }
    if not db_path.exists():
        return metrics
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    latencies: List[float] = []
    try:
        has_dec = cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='decision_snapshots'").fetchone() is not None
        if has_dec:
            for row in cur.execute("SELECT details_json FROM decision_snapshots"):
                metrics["decision_count"] += 1
                details = _safe_json(row["details_json"])
                for k in (
                    "cycle_latency_ms",
                    "latency_ms",
                    "decision_latency_ms",
                    "total_latency_ms",
                ):
                    v = details.get(k)
                    if isinstance(v, (int, float)):
                        latencies.append(float(v))
        has_sys_health = cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='system_health_snapshots'").fetchone() is not None
        if has_sys_health:
            for row in cur.execute("SELECT avg_latency_ms FROM system_health_snapshots"):
                try:
                    latencies.append(float(row["avg_latency_ms"] or 0.0))
                except Exception:
                    continue
        has_health = cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='health_events'").fetchone() is not None
        if has_health:
            for row in cur.execute("SELECT event_type, details_json FROM health_events"):
                et = str(row["event_type"] or "").lower()
                dj = str(row["details_json"] or "").lower()
                if "reconnect" in et or "reconnect" in dj:
                    metrics["ws_reconnect_count"] = int(metrics["ws_reconnect_count"]) + 1
                if "stale" in et or "stale" in dj:
                    metrics["stale_book_count"] = int(metrics["stale_book_count"]) + 1
        has_recon = cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='recon_recovery_state'").fetchone() is not None
        if has_recon:
            row = cur.execute(
                "SELECT "
                "SUM(CASE WHEN recovery_status IN ('pending','retrying') THEN 1 ELSE 0 END) AS open_count, "
                "SUM(CASE WHEN recovery_status = 'halted' THEN 1 ELSE 0 END) AS halted_count "
                "FROM recon_recovery_state"
            ).fetchone()
            if row is not None:
                metrics["recon_required_open"] = int(row["open_count"] or 0)
                metrics["recon_halted_count"] = int(row["halted_count"] or 0)
    finally:
        conn.close()
    metrics["cycle_latency_ms"] = _stats(latencies)
    return metrics


def evaluate_checks(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    ping = report.get("ping_probe", {}) or {}
    ping_p95 = float((ping.get("rtt_ms") or {}).get("p95", 0.0) or 0.0)
    ping_loss = float(ping.get("packet_loss_pct", 100.0) or 100.0)
    checks.append(
        {
            "name": "ping_jitter_loss",
            "pass": bool(ping.get("ok", False) and ping_loss <= 1.0 and ping_p95 <= 100.0),
            "value": {"loss_pct": ping_loss, "p95_ms": ping_p95},
            "target": {"loss_pct_max": 1.0, "p95_ms_max": 100.0},
        }
    )

    ex = report.get("exchange_probe", {}) or {}
    ex_http_p95 = float((ex.get("http_ms") or {}).get("p95", 0.0) or 0.0)
    checks.append(
        {
            "name": "exchange_http_rtt",
            "pass": bool(ex.get("ok", False) and ex_http_p95 <= 800.0),
            "value": {"p95_ms": ex_http_p95, "success_count": int(ex.get("success_count", 0) or 0)},
            "target": {"p95_ms_max": 800.0, "min_success_count": 1},
        }
    )

    runtime = report.get("runtime_metrics", {}) or {}
    cycle_p90 = float((runtime.get("cycle_latency_ms") or {}).get("p90", 0.0) or 0.0)
    checks.append(
        {
            "name": "cycle_latency",
            "pass": bool(cycle_p90 == 0.0 or cycle_p90 <= 3000.0),
            "value": {"p90_ms": cycle_p90},
            "target": {"p90_ms_max": 3000.0},
        }
    )
    checks.append(
        {
            "name": "recon_halted",
            "pass": bool(int(runtime.get("recon_halted_count", 0) or 0) == 0),
            "value": {"recon_halted_count": int(runtime.get("recon_halted_count", 0) or 0)},
            "target": {"recon_halted_count": 0},
        }
    )
    return checks


def build_report(
    *,
    db_path: Path,
    ping_host: str,
    ping_count: int,
    ping_timeout_ms: int,
    exchange_host: str,
    exchange_path: str,
    exchange_attempts: int,
    exchange_timeout_s: float,
    include_network_probes: bool,
    include_speedtest: bool,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "ping_probe": {},
        "exchange_probe": {},
        "speedtest_probe": {},
        "runtime_metrics": collect_runtime_metrics(db_path),
    }
    if include_network_probes:
        report["ping_probe"] = run_ping_probe(
            host=ping_host,
            count=ping_count,
            timeout_ms=ping_timeout_ms,
        )
        report["exchange_probe"] = run_exchange_probe(
            host=exchange_host,
            path=exchange_path,
            attempts=exchange_attempts,
            timeout_s=exchange_timeout_s,
        )
    else:
        report["ping_probe"] = {"ok": False, "skipped": True}
        report["exchange_probe"] = {"ok": False, "skipped": True}

    if include_speedtest:
        report["speedtest_probe"] = run_speedtest_probe()
    else:
        report["speedtest_probe"] = {"available": False, "skipped": True}

    checks = evaluate_checks(report)
    report["checks"] = checks
    report["overall_ready"] = bool(all(bool(c.get("pass", False)) for c in checks))
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="ARGUS network/runtime benchmark report")
    p.add_argument("--db", default="data/unified_trades.db")
    p.add_argument("--ping-host", default="1.1.1.1")
    p.add_argument("--ping-count", type=int, default=100)
    p.add_argument("--ping-timeout-ms", type=int, default=1000)
    p.add_argument("--exchange-host", default="api.kraken.com")
    p.add_argument("--exchange-path", default="/0/public/Time")
    p.add_argument("--exchange-attempts", type=int, default=8)
    p.add_argument("--exchange-timeout-seconds", type=float, default=5.0)
    p.add_argument("--output-dir", default="reports/benchmark")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip-network-probes", action="store_true")
    p.add_argument("--skip-speedtest", action="store_true")
    p.add_argument("--fail-on-checks", action="store_true")
    args = p.parse_args()

    ping_count = int(args.ping_count)
    exchange_attempts = int(args.exchange_attempts)
    if bool(args.quick):
        ping_count = min(ping_count, 10)
        exchange_attempts = min(exchange_attempts, 3)

    report = build_report(
        db_path=Path(str(args.db)),
        ping_host=str(args.ping_host),
        ping_count=ping_count,
        ping_timeout_ms=int(args.ping_timeout_ms),
        exchange_host=str(args.exchange_host),
        exchange_path=str(args.exchange_path),
        exchange_attempts=exchange_attempts,
        exchange_timeout_s=float(args.exchange_timeout_seconds),
        include_network_probes=not bool(args.skip_network_probes),
        include_speedtest=not bool(args.skip_speedtest),
    )

    out_dir = Path(str(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"network_benchmark_{stamp}.json"
    latest_path = out_dir / "network_benchmark_latest.json"
    payload = json.dumps(report, ensure_ascii=True, indent=2)
    out_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")
    print(str(out_path.resolve()))
    print(str(latest_path.resolve()))
    if bool(args.fail_on_checks) and not bool(report.get("overall_ready", False)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

