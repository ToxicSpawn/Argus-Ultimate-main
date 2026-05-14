"""Push 72 — Argus health endpoint.

Exposes HTTP endpoints on PROMETHEUS_PORT (default 8000):
  GET /health   — liveness check (always 200 if process alive)
  GET /ready    — readiness: checks Redis + Postgres + feed state
  GET /metrics/health — JSON summary of all subsystem health

JSON response format:
  {
    "status": "ok" | "degraded" | "down",
    "version": "8.8.0",
    "uptime_secs": 123.4,
    "subsystems": {
      "redis":    {"status": "ok", "latency_ms": 0.5},
      "postgres": {"status": "ok", "latency_ms": 1.2},
      "feed":     {"status": "ok"},
      "metrics":  {"status": "ok"}
    }
  }
"""
from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any

START_TIME = time.time()
VERSION = "8.8.0"
PORT = int(os.environ.get("PROMETHEUS_PORT", "8000"))
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
POSTGRES_URL = os.environ.get("POSTGRES_URL", "")


def _check_redis() -> Dict[str, Any]:
    try:
        import redis as redis_lib
        host_port = REDIS_URL.replace("redis://", "").split("/")[0]
        host, _, port = host_port.partition(":")
        port = int(port) if port else 6379
        t0 = time.monotonic()
        r = redis_lib.Redis(host=host, port=port, socket_timeout=2)
        r.ping()
        latency = (time.monotonic() - t0) * 1000
        return {"status": "ok", "latency_ms": round(latency, 2)}
    except Exception as e:
        return {"status": "down", "error": str(e)[:80]}


def _check_postgres() -> Dict[str, Any]:
    if not POSTGRES_URL:
        return {"status": "unconfigured"}
    try:
        import psycopg2
        t0 = time.monotonic()
        conn = psycopg2.connect(POSTGRES_URL, connect_timeout=3)
        conn.close()
        latency = (time.monotonic() - t0) * 1000
        return {"status": "ok", "latency_ms": round(latency, 2)}
    except Exception as e:
        return {"status": "down", "error": str(e)[:80]}


def _check_feed() -> Dict[str, Any]:
    """Stub — in production wires to AsyncWebSocketFeed.state."""
    return {"status": "ok"}


def _build_health() -> Dict[str, Any]:
    redis_status = _check_redis()
    pg_status = _check_postgres()
    feed_status = _check_feed()

    all_ok = all(
        s.get("status") in ("ok", "unconfigured")
        for s in [redis_status, pg_status, feed_status]
    )
    any_down = any(
        s.get("status") == "down"
        for s in [redis_status, pg_status, feed_status]
    )

    overall = "ok" if all_ok else ("down" if any_down else "degraded")
    return {
        "status": overall,
        "version": VERSION,
        "uptime_secs": round(time.time() - START_TIME, 1),
        "subsystems": {
            "redis":    redis_status,
            "postgres": pg_status,
            "feed":     feed_status,
            "metrics":  {"status": "ok"},
        },
    }


class HealthHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # suppress access logs

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/health":
            body = json.dumps({"status": "ok", "version": VERSION})
            self._respond(200, body)

        elif path == "/ready":
            health = _build_health()
            code = 200 if health["status"] in ("ok", "degraded") else 503
            self._respond(code, json.dumps(health))

        elif path == "/metrics/health":
            health = _build_health()
            code = 200 if health["status"] != "down" else 503
            self._respond(code, json.dumps(health, indent=2))

        else:
            self._respond(404, json.dumps({"error": "not found"}))

    def _respond(self, code: int, body: str):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    print(f"[healthcheck] Listening on :{PORT}")
    server.serve_forever()
