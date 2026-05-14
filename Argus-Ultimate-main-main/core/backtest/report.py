"""Push 69 — BacktestReport: full report generator.

Outputs:
  - JSON summary (metrics + MC stats + WF fold results)
  - CSV equity curve
  - HTML report with embedded tables
  - Console summary string
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from core.backtest.metrics import MetricsResult
from core.backtest.monte_carlo import MonteCarloResult


@dataclass
class BacktestReport:
    """Container for all backtest results and report generation."""

    strategy_name: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    metrics: MetricsResult
    equity_curve: np.ndarray
    monte_carlo: Optional[MonteCarloResult] = None
    walk_forward_sharpes: Optional[List[float]] = None
    extra_metadata: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "period": f"{self.start_date} to {self.end_date}",
            "metrics": self.metrics.to_dict(),
        }
        if self.monte_carlo is not None:
            mc = self.monte_carlo
            d["monte_carlo"] = {
                "n_paths": mc.n_paths,
                "ruin_probability": round(mc.ruin_probability, 4),
                "expected_final_equity": round(mc.expected_final_equity, 2),
                "ci_95_low": round(mc.ci_95_low, 2),
                "ci_95_high": round(mc.ci_95_high, 2),
                "median_sharpe": round(mc.median_sharpe, 4),
                "expected_return_pct": round(mc.expected_return * 100, 2),
            }
        if self.walk_forward_sharpes is not None:
            d["walk_forward"] = {
                "fold_sharpes": [round(s, 4) for s in self.walk_forward_sharpes],
                "mean_oos_sharpe": round(float(np.mean(self.walk_forward_sharpes)), 4),
                "min_oos_sharpe": round(float(np.min(self.walk_forward_sharpes)), 4),
            }
        if self.extra_metadata:
            d["metadata"] = self.extra_metadata
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())
        return p

    def save_equity_csv(self, path: str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = ["bar,equity"]
        for i, eq in enumerate(self.equity_curve):
            lines.append(f"{i},{eq:.4f}")
        p.write_text("\n".join(lines))
        return p

    def save_html(self, path: str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        m = self.metrics.to_dict()
        rows = "".join(
            f"<tr><td>{k}</td><td><b>{v}</b></td></tr>"
            for k, v in m.items() if v is not None
        )
        mc_section = ""
        if self.monte_carlo:
            mc = self.to_dict().get("monte_carlo", {})
            mc_rows = "".join(
                f"<tr><td>{k}</td><td><b>{v}</b></td></tr>"
                for k, v in mc.items()
            )
            mc_section = f"<h2>Monte Carlo ({self.monte_carlo.n_paths:,} paths)</h2><table border='1'>{mc_rows}</table>"

        wf_section = ""
        if self.walk_forward_sharpes:
            wf = self.to_dict().get("walk_forward", {})
            wf_rows = "".join(
                f"<tr><td>{k}</td><td><b>{v}</b></td></tr>"
                for k, v in wf.items()
            )
            wf_section = f"<h2>Walk-Forward OOS</h2><table border='1'>{wf_rows}</table>"

        html = f"""<!DOCTYPE html>
<html><head><title>Argus Backtest Report — {self.strategy_name}</title>
<style>body{{font-family:monospace;margin:2em}}table{{border-collapse:collapse;margin:1em 0}}
td,th{{padding:6px 12px;border:1px solid #ccc}}h1{{color:#2c3e50}}h2{{color:#2980b9}}</style>
</head><body>
<h1>Argus Backtest Report</h1>
<p><b>Strategy:</b> {self.strategy_name} | <b>Symbol:</b> {self.symbol} |
<b>Timeframe:</b> {self.timeframe} | <b>Period:</b> {self.start_date} &rarr; {self.end_date}</p>
<h2>Performance Metrics</h2>
<table border='1'><tr><th>Metric</th><th>Value</th></tr>{rows}</table>
{mc_section}
{wf_section}
</body></html>"""
        p.write_text(html)
        return p

    def print_summary(self) -> None:
        print(self.metrics.summary_str())
        if self.monte_carlo:
            mc = self.monte_carlo
            print(f"\nMonte Carlo ({mc.n_paths:,} paths):")
            print(f"  Ruin Probability:   {mc.ruin_probability:.2%}")
            print(f"  Median Sharpe:      {mc.median_sharpe:.4f}")
            print(f"  95% CI Final Eq:    [{mc.ci_95_low:.2f}, {mc.ci_95_high:.2f}]")
        if self.walk_forward_sharpes:
            print(f"\nWalk-Forward OOS Sharpes: {[round(s,3) for s in self.walk_forward_sharpes]}")
            print(f"  Mean OOS Sharpe:    {float(np.mean(self.walk_forward_sharpes)):.4f}")
