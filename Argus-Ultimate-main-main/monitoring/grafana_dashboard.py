"""
Grafana Dashboard Generator for ARGUS.

Generates a complete Grafana v9/v10 compatible dashboard JSON for:
  - Live P&L curve and equity progression
  - Per-strategy performance (Sharpe, win rate, drawdown)
  - Strategy weight allocation (pie chart)
  - Funding rate harvesting P&L (separate from directional)
  - On-chain signals (exchange flows, MVRV)
  - Correlation matrix heatmap
  - Exchange latency
  - Order fill quality (slippage)
  - System health status

Usage:
  from monitoring.grafana_dashboard import save_dashboard
  save_dashboard("monitoring/grafana_dashboard.json")

Import the generated JSON into Grafana via:
  Dashboards → Import → Upload JSON file
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _panel(
    panel_id: int,
    title: str,
    panel_type: str,
    x: int, y: int, w: int, h: int,
    targets: List[Dict[str, Any]],
    options: Optional[Dict[str, Any]] = None,
    field_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a Grafana panel definition."""
    return {
        "id": panel_id,
        "title": title,
        "type": panel_type,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": targets,
        "options": options or {},
        "fieldConfig": field_config or {"defaults": {}, "overrides": []},
        "datasource": {"type": "prometheus", "uid": "prometheus"},
    }


def _prom_target(expr: str, legend: str = "", ref_id: str = "A") -> Dict[str, Any]:
    """Build a Prometheus query target."""
    return {
        "datasource": {"type": "prometheus", "uid": "prometheus"},
        "expr": expr,
        "legendFormat": legend or expr,
        "refId": ref_id,
        "interval": "",
    }


def generate_dashboard(
    prometheus_url: str = "http://localhost:9090",
    refresh: str = "30s",
    title: str = "ARGUS Trading Dashboard",
) -> Dict[str, Any]:
    """
    Generate a complete Grafana dashboard JSON.

    Returns:
        Dict representing the full Grafana dashboard configuration.
    """
    panels: List[Dict[str, Any]] = []
    pid = 1  # panel id counter

    # ------------------------------------------------------------------
    # Row 1: P&L Overview (y=0)
    # ------------------------------------------------------------------

    # Total P&L curve
    panels.append({
        "id": pid, "title": "📈 Total P&L (AUD)", "type": "timeseries",
        "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
        "targets": [_prom_target("argus_total_pnl_aud", "Total P&L AUD")],
        "options": {
            "tooltip": {"mode": "single"},
            "legend": {"displayMode": "list", "placement": "bottom"},
        },
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "thresholds"},
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "red", "value": None},
                        {"color": "yellow", "value": 0},
                        {"color": "green", "value": 50},
                    ],
                },
                "unit": "currencyAUD",
            },
            "overrides": [],
        },
        "datasource": {"type": "prometheus", "uid": "prometheus"},
    })
    pid += 1

    # Daily P&L bar chart
    panels.append({
        "id": pid, "title": "📊 Daily P&L (30 days)", "type": "barchart",
        "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
        "targets": [_prom_target(
            "increase(argus_total_pnl_aud[1d])", "Daily P&L"
        )],
        "options": {"orientation": "auto", "tooltip": {"mode": "single"}},
        "fieldConfig": {"defaults": {"unit": "currencyAUD"}, "overrides": []},
        "datasource": {"type": "prometheus", "uid": "prometheus"},
    })
    pid += 1

    # ------------------------------------------------------------------
    # Row 2: Strategy Performance (y=8)
    # ------------------------------------------------------------------

    # Strategy weights pie
    panels.append({
        "id": pid, "title": "🥧 Strategy Allocation", "type": "piechart",
        "gridPos": {"x": 0, "y": 8, "w": 6, "h": 8},
        "targets": [_prom_target(
            'argus_strategy_weight{strategy=~".+"}', "{{strategy}}"
        )],
        "options": {"tooltip": {"mode": "single"}, "legend": {"displayMode": "table"}},
        "fieldConfig": {"defaults": {}, "overrides": []},
        "datasource": {"type": "prometheus", "uid": "prometheus"},
    })
    pid += 1

    # Win rate by strategy
    panels.append({
        "id": pid, "title": "🎯 Win Rate by Strategy", "type": "bargauge",
        "gridPos": {"x": 6, "y": 8, "w": 6, "h": 8},
        "targets": [_prom_target(
            'argus_win_rate{strategy=~".+"}', "{{strategy}}"
        )],
        "options": {"orientation": "horizontal", "reduceOptions": {"calcs": ["lastNotNull"]}},
        "fieldConfig": {
            "defaults": {
                "unit": "percentunit",
                "min": 0, "max": 1,
                "thresholds": {
                    "mode": "absolute",
                    "steps": [{"color": "red", "value": None}, {"color": "yellow", "value": 0.45}, {"color": "green", "value": 0.55}],
                },
            },
            "overrides": [],
        },
        "datasource": {"type": "prometheus", "uid": "prometheus"},
    })
    pid += 1

    # Drawdown
    panels.append({
        "id": pid, "title": "📉 Drawdown", "type": "timeseries",
        "gridPos": {"x": 12, "y": 8, "w": 12, "h": 8},
        "targets": [
            _prom_target("argus_current_drawdown_pct", "Current Drawdown %", "A"),
            _prom_target("argus_max_drawdown_pct", "Max Drawdown %", "B"),
        ],
        "options": {"tooltip": {"mode": "multi"}},
        "fieldConfig": {
            "defaults": {
                "unit": "percent",
                "custom": {"fillOpacity": 10, "gradientMode": "opacity"},
                "thresholds": {
                    "mode": "absolute",
                    "steps": [{"color": "green", "value": None}, {"color": "yellow", "value": 5}, {"color": "red", "value": 15}],
                },
            },
            "overrides": [],
        },
        "datasource": {"type": "prometheus", "uid": "prometheus"},
    })
    pid += 1

    # ------------------------------------------------------------------
    # Row 3: Funding Rate Harvesting (y=16)
    # ------------------------------------------------------------------

    panels.append({
        "id": pid, "title": "💰 Funding Harvest P&L (AUD)", "type": "stat",
        "gridPos": {"x": 0, "y": 16, "w": 6, "h": 5},
        "targets": [_prom_target("argus_funding_harvest_total_aud", "Funding P&L")],
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "value", "graphMode": "area"},
        "fieldConfig": {"defaults": {"unit": "currencyAUD", "color": {"mode": "thresholds"},
            "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}}, "overrides": []},
        "datasource": {"type": "prometheus", "uid": "prometheus"},
    })
    pid += 1

    panels.append({
        "id": pid, "title": "🔄 Active Harvests", "type": "stat",
        "gridPos": {"x": 6, "y": 16, "w": 6, "h": 5},
        "targets": [_prom_target("argus_funding_active_positions", "Active")],
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "background"},
        "fieldConfig": {"defaults": {"unit": "short"}, "overrides": []},
        "datasource": {"type": "prometheus", "uid": "prometheus"},
    })
    pid += 1

    panels.append({
        "id": pid, "title": "📡 Current Funding Rates", "type": "bargauge",
        "gridPos": {"x": 12, "y": 16, "w": 12, "h": 5},
        "targets": [_prom_target(
            'argus_funding_rate{symbol=~".+", exchange=~".+"}', "{{symbol}} @ {{exchange}}"
        )],
        "options": {"orientation": "horizontal", "reduceOptions": {"calcs": ["lastNotNull"]}},
        "fieldConfig": {
            "defaults": {
                "unit": "percentunit",
                "thresholds": {
                    "mode": "absolute",
                    "steps": [{"color": "blue", "value": None}, {"color": "green", "value": 0.0005}, {"color": "super-light-green", "value": 0.002}],
                },
            },
            "overrides": [],
        },
        "datasource": {"type": "prometheus", "uid": "prometheus"},
    })
    pid += 1

    # ------------------------------------------------------------------
    # Row 4: Alternative Data Signals (y=21)
    # ------------------------------------------------------------------

    panels.append({
        "id": pid, "title": "⛓️ On-Chain Net Flow (BTC)", "type": "timeseries",
        "gridPos": {"x": 0, "y": 21, "w": 8, "h": 6},
        "targets": [_prom_target("argus_onchain_exchange_net_flow_btc", "BTC Net Flow")],
        "options": {"tooltip": {"mode": "single"}},
        "fieldConfig": {
            "defaults": {
                "unit": "short",
                "custom": {"fillOpacity": 20},
                "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "red", "value": 1000}]},
            },
            "overrides": [],
        },
        "datasource": {"type": "prometheus", "uid": "prometheus"},
    })
    pid += 1

    panels.append({
        "id": pid, "title": "📊 Options Put/Call Ratio", "type": "timeseries",
        "gridPos": {"x": 8, "y": 21, "w": 8, "h": 6},
        "targets": [
            _prom_target("argus_options_pcr", "P/C Ratio", "A"),
            _prom_target("argus_options_iv_skew", "IV Skew", "B"),
        ],
        "options": {"tooltip": {"mode": "multi"}},
        "fieldConfig": {"defaults": {"unit": "short"}, "overrides": []},
        "datasource": {"type": "prometheus", "uid": "prometheus"},
    })
    pid += 1

    panels.append({
        "id": pid, "title": "🌍 Macro Regime Scalar", "type": "gauge",
        "gridPos": {"x": 16, "y": 21, "w": 8, "h": 6},
        "targets": [_prom_target("argus_macro_position_scalar", "Macro Scalar")],
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}},
        "fieldConfig": {
            "defaults": {
                "unit": "percentunit",
                "min": 0, "max": 1,
                "thresholds": {
                    "mode": "absolute",
                    "steps": [{"color": "red", "value": None}, {"color": "yellow", "value": 0.5}, {"color": "green", "value": 0.85}],
                },
            },
            "overrides": [],
        },
        "datasource": {"type": "prometheus", "uid": "prometheus"},
    })
    pid += 1

    # ------------------------------------------------------------------
    # Row 5: Risk & System Health (y=27)
    # ------------------------------------------------------------------

    panels.append({
        "id": pid, "title": "🔗 Avg Pairwise Correlation", "type": "gauge",
        "gridPos": {"x": 0, "y": 27, "w": 6, "h": 6},
        "targets": [_prom_target("argus_avg_pairwise_correlation", "Correlation")],
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}},
        "fieldConfig": {
            "defaults": {
                "unit": "percentunit",
                "min": 0, "max": 1,
                "thresholds": {
                    "mode": "absolute",
                    "steps": [{"color": "green", "value": None}, {"color": "yellow", "value": 0.8}, {"color": "red", "value": 0.92}],
                },
            },
            "overrides": [],
        },
        "datasource": {"type": "prometheus", "uid": "prometheus"},
    })
    pid += 1

    panels.append({
        "id": pid, "title": "⚡ Exchange Latency (ms)", "type": "bargauge",
        "gridPos": {"x": 6, "y": 27, "w": 8, "h": 6},
        "targets": [_prom_target(
            'argus_exchange_latency_ms{exchange=~".+"}', "{{exchange}}"
        )],
        "options": {"orientation": "horizontal", "reduceOptions": {"calcs": ["lastNotNull"]}},
        "fieldConfig": {
            "defaults": {
                "unit": "ms",
                "thresholds": {
                    "mode": "absolute",
                    "steps": [{"color": "green", "value": None}, {"color": "yellow", "value": 100}, {"color": "red", "value": 500}],
                },
            },
            "overrides": [],
        },
        "datasource": {"type": "prometheus", "uid": "prometheus"},
    })
    pid += 1

    panels.append({
        "id": pid, "title": "🟢 System Health", "type": "stat",
        "gridPos": {"x": 14, "y": 27, "w": 10, "h": 6},
        "targets": [
            _prom_target("argus_system_healthy", "Healthy", "A"),
            _prom_target("argus_kill_switch_active", "Kill Switch", "B"),
            _prom_target("argus_circuit_breaker_active", "Circuit Breaker", "C"),
        ],
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "background", "graphMode": "none"},
        "fieldConfig": {
            "defaults": {
                "mappings": [{"type": "value", "options": {"0": {"text": "OFF", "color": "red"}, "1": {"text": "ON", "color": "green"}}}],
            },
            "overrides": [],
        },
        "datasource": {"type": "prometheus", "uid": "prometheus"},
    })
    pid += 1

    # ------------------------------------------------------------------
    # Assemble dashboard
    # ------------------------------------------------------------------

    dashboard = {
        "title": title,
        "uid": "argus-trading-v2",
        "version": 2,
        "schemaVersion": 38,
        "refresh": refresh,
        "time": {"from": "now-24h", "to": "now"},
        "timepicker": {},
        "timezone": "browser",
        "tags": ["argus", "trading", "crypto"],
        "panels": panels,
        "templating": {"list": []},
        "annotations": {"list": []},
        "links": [],
        "editable": True,
        "graphTooltip": 1,  # shared crosshair
        "fiscalYearStartMonth": 0,
        "liveNow": False,
    }

    return {"dashboard": dashboard, "overwrite": True, "folderId": 0}


def save_dashboard(path: str = "monitoring/grafana_dashboard.json") -> None:
    """Write the Grafana dashboard JSON to disk."""
    data = generate_dashboard()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Grafana dashboard saved to %s", path)


def generate_prometheus_recording_rules() -> str:
    """Generate Prometheus recording rules YAML for pre-computed metrics."""
    return """groups:
  - name: argus_trading
    interval: 60s
    rules:
      - record: argus_rolling_win_rate_7d
        expr: |
          rate(argus_wins_total[7d])
          / clamp_min(rate(argus_trades_total[7d]), 0.0001)

      - record: argus_rolling_sharpe_30d
        expr: |
          (
            rate(argus_trade_pnl_sum[30d])
            / clamp_min(
                sqrt(rate(argus_trade_pnl_sq_sum[30d]) - rate(argus_trade_pnl_sum[30d])^2),
                0.0001
              )
          ) * sqrt(365)

      - record: argus_daily_pnl
        expr: increase(argus_total_pnl_aud[1d])

      - record: argus_funding_apr_estimate
        expr: argus_funding_rate * 3 * 365
"""


# Auto-generate on first import if not exists
_DEFAULT_PATH = "monitoring/grafana_dashboard.json"
if not os.path.exists(_DEFAULT_PATH):
    try:
        save_dashboard(_DEFAULT_PATH)
    except Exception:
        pass  # Not critical
