"""
StrategyDashboard
=================
Rich-terminal live dashboard that shows:
  • Strategy leaderboard (StrategyRanker)
  • Registry enable/disable status (StrategyRegistry)
  • Bandit posterior weights (BanditAllocator)
  • Last-cycle per-symbol performance

Call `dashboard.render()` periodically inside the trading loop.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box as rich_box
    _RICH = True
except ImportError:
    _RICH = False

if TYPE_CHECKING:
    from strategies.strategy_ranker import StrategyRanker
    from strategies.strategy_registry import StrategyRegistry
    from strategies.bandit_allocator import BanditAllocator

logger = logging.getLogger("argus.strategy_dashboard")

_console = Console() if _RICH else None


class StrategyDashboard:
    """
    Lightweight dashboard — no external dependencies beyond *rich*.

    Usage::

        dash = StrategyDashboard(ranker, registry, bandit)
        # every N cycles:
        dash.render(cycle=cycle, capital=self.capital)
    """

    def __init__(
        self,
        ranker: "StrategyRanker",
        registry: "StrategyRegistry",
        bandit: "BanditAllocator",
    ) -> None:
        self.ranker = ranker
        self.registry = registry
        self.bandit = bandit

    def render(
        self,
        cycle: int = 0,
        capital: float = 0.0,
        extra: Optional[Dict] = None,
    ) -> None:
        if not _RICH or _console is None:
            self._fallback_log()
            return

        active_names = self.registry.names(active_only=True)
        weights = self.bandit.weights(active_names) if active_names else {}
        ranker_snap = self.ranker.snapshot()

        # ── Leaderboard table ─────────────────────────────────────────── #
        table = Table(
            title=f"[bold magenta]Strategy Dashboard — Cycle {cycle} | Capital ${capital:,.2f}[/bold magenta]",
            box=rich_box.MINIMAL_DOUBLE_HEAD,
            show_lines=False,
        )
        table.add_column("Strategy", style="cyan", min_width=28)
        table.add_column("Score", justify="right")
        table.add_column("Sharpe", justify="right")
        table.add_column("Win%", justify="right")
        table.add_column("Trades", justify="right")
        table.add_column("PnL", justify="right")
        table.add_column("BanditW", justify="right")
        table.add_column("Active", justify="center")

        for row in ranker_snap[:15]:   # top-15 to avoid flooding the terminal
            name = row["name"]
            score_color = "green" if row["score"] > 0 else "red"
            pnl_color = "green" if row["total_pnl"] >= 0 else "red"
            bw = weights.get(name, 0.0)
            is_active = self.registry.is_enabled(name)
            active_str = "[green]ON[/green]" if is_active else "[red]OFF[/red]"
            table.add_row(
                name,
                f"[{score_color}]{row['score']:+.4f}[/{score_color}]",
                f"{row['sharpe']:+.3f}",
                f"{row['win_rate']:.1%}",
                str(row["trades"]),
                f"[{pnl_color}]${row['total_pnl']:+.2f}[/{pnl_color}]",
                f"{bw:.1%}",
                active_str,
            )

        _console.print(table)

        # ── Bandit posterior summary ──────────────────────────────────── #
        bandit_snap = self.bandit.snapshot()
        if bandit_snap:
            b_table = Table(
                title="[bold cyan]Bandit Posteriors[/bold cyan]",
                box=rich_box.SIMPLE,
                show_header=True,
            )
            b_table.add_column("Arm", style="cyan")
            b_table.add_column("α", justify="right")
            b_table.add_column("β", justify="right")
            b_table.add_column("Est Win%", justify="right")
            b_table.add_column("Pulls", justify="right")
            for snap in bandit_snap[:10]:
                wr_color = "green" if snap["est_win_rate"] >= 0.5 else "yellow"
                b_table.add_row(
                    snap["name"],
                    str(snap["alpha"]),
                    str(snap["beta"]),
                    f"[{wr_color}]{snap['est_win_rate']:.1%}[/{wr_color}]",
                    str(snap["pulls"]),
                )
            _console.print(b_table)

        if extra:
            for k, v in extra.items():
                _console.print(f"  [dim]{k}:[/dim] {v}")

    def _fallback_log(self) -> None:
        for row in self.ranker.snapshot()[:5]:
            logger.info(
                "Strategy %s: score=%.4f sharpe=%.3f win=%.1f%% trades=%d pnl=%.2f",
                row["name"], row["score"], row["sharpe"],
                row["win_rate"] * 100, row["trades"], row["total_pnl"],
            )
