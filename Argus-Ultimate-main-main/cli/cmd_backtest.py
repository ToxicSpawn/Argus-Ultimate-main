"""argus backtest command implementation — Push 63."""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import List, Optional


def run_backtest(
    strategy_name: str,
    data_path: Optional[Path],
    symbols: List[str],
    start_date: Optional[str],
    end_date: Optional[str],
    initial_equity: float,
    fee_bps: float,
    output_dir: Optional[Path],
    n_bars: int,
    config_path: Optional[Path],
) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()

    console.print(Panel(
        f"Strategy: [cyan]{strategy_name}[/cyan]  "
        f"Equity: [green]${initial_equity:,.0f}[/green]  "
        f"Fee: [yellow]{fee_bps}bps[/yellow]",
        title="Argus Backtest",
        border_style="magenta",
    ))

    # ------------------------------------------------------------------
    # Build components
    # ------------------------------------------------------------------
    from core.backtest.backtest_config import BacktestConfig
    from core.backtest.data_feed import DataFeed
    from core.backtest.backtest_engine import BacktestEngine
    from core.pnl.pnl_tracker import PnLTracker
    from core.execution.execution_engine import ExecutionEngine
    from core.strategy.strategy_registry import StrategyRegistry
    from core.strategy.strategy_runner import StrategyRunner
    from core.strategy.builtin.momentum_strategy import MomentumStrategy

    bt_config = BacktestConfig(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_equity=initial_equity,
        fee_bps=fee_bps,
    )

    # Data feed
    if data_path and Path(data_path).exists():
        console.print(f"[dim]Loading data from {data_path}...[/dim]")
        feed = DataFeed(path=data_path, symbol=symbols[0])
    else:
        console.print(f"[dim]Generating {n_bars} synthetic bars (seed=42)...[/dim]")
        feed = DataFeed.synthetic(n=n_bars, symbol=symbols[0], seed=42)

    # Strategy
    reg = StrategyRegistry()
    reg.register(MomentumStrategy)  # always register builtin
    runner = StrategyRunner(reg)

    pnl = PnLTracker(initial_equity=initial_equity)
    engine = ExecutionEngine(pnl_tracker=pnl, paper_trading=True)

    # Start strategy
    loop = asyncio.new_event_loop()
    loop.run_until_complete(runner.start("MomentumStrategy"))
    loop.close()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    t0 = time.time()
    bt = BacktestEngine(bt_config, runner, engine, pnl)
    result = bt.run(feed)
    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    table = Table(title="Backtest Results", border_style="magenta")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total Return", f"{result.total_return*100:.2f}%")
    table.add_row("Annualised Return", f"{result.annualised_return*100:.2f}%")
    table.add_row("Sharpe Ratio", f"{result.sharpe:.3f}")
    table.add_row("Sortino Ratio", f"{result.sortino:.3f}")
    table.add_row("Max Drawdown", f"{result.max_drawdown*100:.2f}%")
    table.add_row("Win Rate", f"{result.win_rate*100:.1f}%")
    table.add_row("Profit Factor", f"{result.profit_factor:.2f}")
    table.add_row("Calmar Ratio", f"{result.calmar:.3f}")
    table.add_row("Trades", str(result.n_trades))
    table.add_row("Initial Equity", f"${result.initial_equity:,.2f}")
    table.add_row("Final Equity", f"${result.final_equity:,.2f}")
    table.add_row("Bars Processed", str(bt.bar_count))
    table.add_row("Run Time", f"{elapsed:.2f}s")
    console.print(table)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result.to_csv(out / "equity_curve.csv")
        result.plot_equity_curve(out / "equity_curve.png")
        result.to_json(out / "backtest_result.json")
        console.print(f"[green]✓ Outputs saved to {out}[/green]")
