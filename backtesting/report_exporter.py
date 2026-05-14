"""Argus Ultimate — Backtest HTML Report Exporter.

Converts JSONL backtest output into a self-contained Plotly HTML report.
Designed as a drop-in finish hook for run_validation_backtest.py::

    from backtest.report_exporter import export_report
    export_report(results, output_path=Path("artifacts/reports/backtest_latest.html"))

Can also be run standalone::

    python -m backtest.report_exporter artifacts/backtest_results.jsonl
    python -m backtest.report_exporter artifacts/backtest_results.jsonl --output my_report.html

Closes the Jesse interactive charts gap.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _results_to_records(results: Any) -> list[dict]:
    """Normalise various result shapes into a list of trade dicts."""
    if isinstance(results, list):
        return results
    if isinstance(results, dict):
        for key in ("trades", "results", "records", "data"):
            if key in results and isinstance(results[key], list):
                return results[key]
        return [results]
    if hasattr(results, "trades"):
        return list(results.trades)
    if hasattr(results, "to_dict"):
        return [results.to_dict()]
    return []


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _equity_curve_fig(records: list[dict]):
    """Return a Plotly figure for the cumulative equity curve."""
    import plotly.graph_objects as go

    equity = []
    timestamps = []
    running = 0.0
    for r in records:
        pnl = float(r.get("pnl", r.get("profit", r.get("net_profit", 0))))
        running += pnl
        equity.append(running)
        ts = r.get("close_time", r.get("exit_time", r.get("timestamp", "")))
        timestamps.append(str(ts) if ts else str(len(equity)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=equity,
            mode="lines",
            name="Equity",
            line=dict(color="#00d4ff", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.08)",
        )
    )
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Trade",
        yaxis_title="Cumulative PnL",
        template="plotly_dark",
        hovermode="x unified",
    )
    return fig


def _drawdown_fig(records: list[dict]):
    """Return a Plotly figure for the drawdown series."""
    import plotly.graph_objects as go

    equity = []
    running = 0.0
    for r in records:
        pnl = float(r.get("pnl", r.get("profit", r.get("net_profit", 0))))
        running += pnl
        equity.append(running)

    drawdowns = []
    # Peak starts at 0.0 (representing starting equity before any trades)
    # so that initial losing trades correctly register as drawdowns.
    peak = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = (e - peak) / abs(peak) * 100 if peak != 0 else (e * 100 if e < 0 else 0.0)
        drawdowns.append(dd)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(drawdowns))),
            y=drawdowns,
            mode="lines",
            name="Drawdown %",
            line=dict(color="#ff4466", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255,68,102,0.12)",
        )
    )
    fig.update_layout(
        title="Drawdown (%)",
        xaxis_title="Trade",
        yaxis_title="Drawdown %",
        template="plotly_dark",
    )
    return fig


def _pnl_distribution_fig(records: list[dict]):
    """Return a histogram of per-trade PnL."""
    import plotly.graph_objects as go

    pnls = [
        float(r.get("pnl", r.get("profit", r.get("net_profit", 0))))
        for r in records
    ]
    fig = go.Figure(
        go.Histogram(
            x=pnls,
            nbinsx=50,
            marker_color="#9966ff",
            opacity=0.8,
            name="PnL distribution",
        )
    )
    fig.update_layout(
        title="Per-Trade PnL Distribution",
        xaxis_title="PnL",
        yaxis_title="Frequency",
        template="plotly_dark",
    )
    return fig


def _win_loss_fig(records: list[dict]):
    """Return a pie chart of win/loss ratio."""
    import plotly.graph_objects as go

    wins = sum(
        1 for r in records
        if float(r.get("pnl", r.get("profit", r.get("net_profit", 0)))) > 0
    )
    losses = len(records) - wins
    fig = go.Figure(
        go.Pie(
            labels=["Win", "Loss"],
            values=[wins, losses],
            marker_colors=["#00cc66", "#ff4466"],
            hole=0.4,
        )
    )
    fig.update_layout(title="Win / Loss Ratio", template="plotly_dark")
    return fig


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def _compute_summary(records: list[dict]) -> dict[str, Any]:
    if not records:
        return {}

    pnls = [
        float(r.get("pnl", r.get("profit", r.get("net_profit", 0))))
        for r in records
    ]
    total_pnl = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0.0

    # Running equity for drawdown — peak starts at 0.0 (pre-trade baseline)
    equity = []
    running = 0.0
    for p in pnls:
        running += p
        equity.append(running)

    peak = 0.0
    max_dd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = (e - peak) / abs(peak) * 100 if peak != 0 else (e * 100 if e < 0 else 0.0)
        if dd < max_dd:
            max_dd = dd

    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    profit_factor = abs(sum(wins) / sum(losses)) if losses else float("inf")

    # Sharpe (simplified, assumes 0 risk-free)
    try:
        import statistics
        mean_p = statistics.mean(pnls)
        std_p = statistics.stdev(pnls) if len(pnls) > 1 else 1.0
        sharpe = mean_p / std_p * (252 ** 0.5) if std_p != 0 else 0.0
    except Exception:
        sharpe = 0.0

    return {
        "total_trades": len(records),
        "total_pnl": round(total_pnl, 4),
        "win_rate_pct": round(win_rate, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "profit_factor": round(profit_factor, 3),
        "sharpe_ratio": round(sharpe, 3),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "largest_win": round(max(pnls), 4) if pnls else 0,
        "largest_loss": round(min(pnls), 4) if pnls else 0,
    }


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def _build_html(figures: list, summary: dict, title: str) -> str:
    """Assemble a single self-contained HTML string from Plotly figures."""
    try:
        from plotly.io import to_html
    except ImportError as exc:
        raise ImportError("plotly is required: pip install plotly") from exc

    chart_blocks = []
    for i, fig in enumerate(figures):
        div = to_html(
            fig,
            full_html=False,
            include_plotlyjs=("cdn" if i == 0 else False),
            div_id=f"chart_{i}",
        )
        chart_blocks.append(div)

    # Summary table rows
    rows = "".join(
        f"<tr><td>{k.replace('_', ' ').title()}</td><td><b>{v}</b></td></tr>"
        for k, v in summary.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    body {{ background: #0d0d0d; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; }}
    h1 {{ color: #00d4ff; border-bottom: 1px solid #333; padding-bottom: 8px; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 30px; }}
    .summary table {{ border-collapse: collapse; background: #1a1a2e; border-radius: 8px; overflow: hidden; }}
    .summary td {{ padding: 8px 16px; border-bottom: 1px solid #2a2a3e; }}
    .summary td:first-child {{ color: #aaa; font-size: 0.85em; }}
    .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 20px; }}
    .chart-wrap {{ background: #111; border-radius: 8px; padding: 10px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="summary"><table>{rows}</table></div>
  <div class="chart-grid">
    {''.join(f'<div class="chart-wrap">{b}</div>' for b in chart_blocks)}
  </div>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_report(
    results: Any,
    output_path: Path | None = None,
    title: str = "Argus Backtest Report",
) -> Path:
    """Convert backtest results to a self-contained HTML report.

    Args:
        results: Backtest output — list of trade dicts, a dict with a 'trades'
                 key, an object with a .trades attribute, or a Path/str to a
                 JSONL file.
        output_path: Where to write the HTML file.  Defaults to
                     artifacts/reports/backtest_<timestamp>.html
        title: Page title embedded in the HTML.

    Returns:
        Path to the written HTML file.
    """
    import time

    if output_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path("artifacts") / "reports" / f"backtest_{ts}.html"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Accept file path input
    if isinstance(results, (str, Path)) and Path(results).exists():
        records = _load_jsonl(Path(results))
    else:
        records = _results_to_records(results)

    summary = _compute_summary(records)

    figures = [
        _equity_curve_fig(records),
        _drawdown_fig(records),
        _pnl_distribution_fig(records),
        _win_loss_fig(records),
    ]

    html = _build_html(figures, summary, title)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Backtest HTML report -> %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Export Argus backtest JSONL results to self-contained Plotly HTML",
    )
    p.add_argument("input", type=Path, help="Path to JSONL backtest results file")
    p.add_argument("--output", type=Path, default=None, help="Output HTML path")
    p.add_argument("--title", type=str, default="Argus Backtest Report", help="Report title")
    args = p.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        out = export_report(args.input, output_path=args.output, title=args.title)
        print(f"Report written -> {out}")
        return 0
    except Exception as exc:
        logger.error("Export failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(_cli())
