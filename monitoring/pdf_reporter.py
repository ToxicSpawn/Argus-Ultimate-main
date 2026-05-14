"""
PDF/Text Performance Reporter — generates monthly/weekly performance reports.

Report sections:
  1. Executive Summary (P&L, Sharpe, max drawdown)
  2. Strategy Breakdown table
  3. Daily P&L chart (ASCII)
  4. Top 5 wins and losses
  5. Risk metrics snapshot
  6. Next period outlook

Uses reportlab if available, falls back to plain text .txt report.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as _rl_canvas
    _REPORTLAB = True
except ImportError:
    _REPORTLAB = False


@dataclass
class ReportConfig:
    period: str  # "daily" | "weekly" | "monthly"
    start_date: datetime
    end_date: datetime
    capital_usd: float = 1000.0
    output_path: Optional[str] = None


class PDFReporter:
    """Generates performance reports (PDF or plain text fallback)."""

    def __init__(
        self,
        trade_db: str = "data/paper_trades.db",
        output_dir: str = "reports/",
    ) -> None:
        self.trade_db = trade_db
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate(self, config: ReportConfig) -> str:
        """Generate report and return path to file."""
        start_ts = config.start_date.timestamp()
        end_ts = config.end_date.timestamp()
        stats = self._compute_stats(start_ts, end_ts)
        stats["capital_usd"] = config.capital_usd

        if config.output_path:
            path = config.output_path
        else:
            date_str = config.start_date.strftime("%Y%m%d")
            ext = "pdf" if _REPORTLAB else "txt"
            path = os.path.join(self.output_dir, f"argus_report_{config.period}_{date_str}.{ext}")

        if _REPORTLAB and path.endswith(".pdf"):
            return self._generate_pdf(config, stats)

        return self._generate_text(config, stats, path)

    # ------------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------------

    def _generate_text(self, config: ReportConfig, stats: Dict, path: str) -> str:
        lines = [
            "=" * 70,
            f"ARGUS PERFORMANCE REPORT — {config.period.upper()}",
            f"Period: {config.start_date.strftime('%Y-%m-%d')} to {config.end_date.strftime('%Y-%m-%d')}",
            "=" * 70,
            "",
            "EXECUTIVE SUMMARY",
            "-" * 40,
            f"  Total P&L:       ${stats.get('total_pnl', 0):+,.2f}",
            f"  Return:          {stats.get('return_pct', 0):+.2f}%",
            f"  Sharpe Ratio:    {stats.get('sharpe', 0):.2f}",
            f"  Max Drawdown:    {stats.get('max_dd_pct', 0)*100:.1f}%",
            f"  Win Rate:        {stats.get('win_rate', 0)*100:.1f}%",
            f"  Trade Count:     {stats.get('trade_count', 0)}",
            "",
        ]

        # Strategy table
        by_strat = stats.get("by_strategy", {})
        if by_strat:
            lines += [
                "STRATEGY BREAKDOWN",
                "-" * 40,
                f"  {'Strategy':<25} {'P&L':>10} {'Trades':>7} {'Win%':>6}",
                f"  {'-'*25} {'-'*10} {'-'*7} {'-'*6}",
            ]
            for strat, s in by_strat.items():
                lines.append(
                    f"  {strat:<25} ${s['pnl']:>9.2f} {s['trades']:>7} {s['win_rate']*100:>5.1f}%"
                )
            lines.append("")

        # ASCII daily P&L chart
        daily = stats.get("daily_pnl", [])
        if daily:
            values = [v for _, v in daily]
            chart = self._ascii_chart(values, width=50, height=8)
            lines += ["DAILY P&L CHART", "-" * 40, chart, ""]

        # Top wins/losses
        top_wins = stats.get("top_wins", [])
        top_losses = stats.get("top_losses", [])
        if top_wins:
            lines += ["TOP 5 WINS", "-" * 40]
            for t in top_wins[:5]:
                ts = datetime.fromtimestamp(t.get("timestamp", 0), tz=timezone.utc).strftime("%m-%d %H:%M")
                lines.append(f"  {ts}  {t.get('symbol','?'):<12} ${t.get('pnl',0):+.2f}")
            lines.append("")
        if top_losses:
            lines += ["TOP 5 LOSSES", "-" * 40]
            for t in top_losses[:5]:
                ts = datetime.fromtimestamp(t.get("timestamp", 0), tz=timezone.utc).strftime("%m-%d %H:%M")
                lines.append(f"  {ts}  {t.get('symbol','?'):<12} ${t.get('pnl',0):+.2f}")
            lines.append("")

        lines += ["=" * 70, "End of report", "=" * 70]

        content = "\n".join(lines)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("PDFReporter: report written to %s", path)
        return path

    # ------------------------------------------------------------------
    # PDF report (requires reportlab)
    # ------------------------------------------------------------------

    def _generate_pdf(self, config: ReportConfig, stats: Dict) -> str:
        date_str = config.start_date.strftime("%Y%m%d")
        path = os.path.join(self.output_dir, f"argus_report_{config.period}_{date_str}.pdf")

        c = _rl_canvas.Canvas(path, pagesize=A4)
        width, height = A4
        y = height - 50

        def draw_line(text: str, indent: int = 40, size: int = 11) -> None:
            nonlocal y
            c.setFont("Helvetica", size)
            c.drawString(indent, y, text)
            y -= size + 4
            if y < 60:
                c.showPage()
                y = height - 50

        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, y, f"ARGUS Performance Report — {config.period.capitalize()}")
        y -= 30
        draw_line(f"Period: {config.start_date.strftime('%Y-%m-%d')} → {config.end_date.strftime('%Y-%m-%d')}")
        y -= 10
        c.setFont("Helvetica-Bold", 13)
        c.drawString(40, y, "Executive Summary")
        y -= 20
        draw_line(f"Total P&L:    ${stats.get('total_pnl', 0):+,.2f}")
        draw_line(f"Return:       {stats.get('return_pct', 0):+.2f}%")
        draw_line(f"Sharpe:       {stats.get('sharpe', 0):.2f}")
        draw_line(f"Max Drawdown: {stats.get('max_dd_pct', 0)*100:.1f}%")
        draw_line(f"Win Rate:     {stats.get('win_rate', 0)*100:.1f}%")
        draw_line(f"Trades:       {stats.get('trade_count', 0)}")
        c.save()
        logger.info("PDFReporter: PDF written to %s", path)
        return path

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _compute_stats(self, start_ts: float, end_ts: float) -> Dict:
        """Load trades and compute report statistics."""
        try:
            conn = sqlite3.connect(self.trade_db)
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM trades WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
                (start_ts, end_ts),
            )
            trades = [dict(r) for r in cur.fetchall()]
            conn.close()
        except Exception:
            trades = []

        if not trades:
            return {"total_pnl": 0.0, "trade_count": 0, "win_rate": 0.0, "sharpe": 0.0, "max_dd_pct": 0.0, "return_pct": 0.0}

        pnls = [float(t.get("pnl", t.get("pnl_usd", 0))) for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        total = sum(pnls)

        # By strategy
        by_strat: Dict[str, Dict] = {}
        for t, p in zip(trades, pnls):
            s = t.get("strategy", "unknown")
            if s not in by_strat:
                by_strat[s] = {"pnl": 0.0, "trades": 0, "wins": 0}
            by_strat[s]["pnl"] += p
            by_strat[s]["trades"] += 1
            if p > 0:
                by_strat[s]["wins"] += 1
        for s in by_strat:
            n = by_strat[s]["trades"]
            by_strat[s]["win_rate"] = by_strat[s]["wins"] / n if n else 0.0

        # Daily P&L
        daily: Dict[str, float] = {}
        for t, p in zip(trades, pnls):
            day = datetime.fromtimestamp(float(t.get("timestamp", 0)), tz=timezone.utc).strftime("%Y-%m-%d")
            daily[day] = daily.get(day, 0.0) + p
        daily_series = sorted(daily.items())

        # Sharpe (annualised, rough)
        import statistics
        sharpe = 0.0
        if len(pnls) > 1:
            mean_p = statistics.mean(pnls)
            std_p = statistics.stdev(pnls)
            sharpe = (mean_p / std_p * (252 ** 0.5)) if std_p > 0 else 0.0

        # Max drawdown from equity curve
        equity = 1000.0
        peak = equity
        max_dd = 0.0
        for p in pnls:
            equity += p
            if equity > peak:
                peak = equity
            dd = (peak - equity) / max(peak, 1.0)
            max_dd = max(max_dd, dd)

        # Top wins/losses
        sorted_trades = sorted(zip(pnls, trades), key=lambda x: x[0])
        top_losses = [{"pnl": p, **t} for p, t in sorted_trades[:5]]
        top_wins = [{"pnl": p, **t} for p, t in sorted_trades[-5:][::-1]]

        return {
            "total_pnl": total,
            "return_pct": total / 1000.0 * 100,
            "trade_count": len(trades),
            "win_rate": wins / len(trades),
            "sharpe": sharpe,
            "max_dd_pct": max_dd,
            "by_strategy": by_strat,
            "daily_pnl": daily_series,
            "top_wins": top_wins,
            "top_losses": top_losses,
        }

    # ------------------------------------------------------------------
    # ASCII chart
    # ------------------------------------------------------------------

    def _ascii_chart(self, values: List[float], width: int = 60, height: int = 10) -> str:
        if not values:
            return "(no data)"
        mn, mx = min(values), max(values)
        rng = mx - mn or 1.0
        rows = []
        for row in range(height - 1, -1, -1):
            threshold = mn + rng * row / (height - 1)
            line = "│"
            for v in values[-width:]:
                line += "█" if v >= threshold else " "
            rows.append(line)
        # X-axis
        rows.append("└" + "─" * min(len(values), width))
        # Y-axis labels
        rows[0] = f"${mx:>7.1f} " + rows[0]
        rows[-1] = f"${mn:>7.1f} " + rows[-1]
        return "\n".join(rows)
