"""
Argus Trading System - CLI Dashboard
====================================

Real-time terminal dashboard for monitoring the trading system.

Features:
- Portfolio overview with P&L
- Active positions display
- Recent trades log
- Strategy performance
- System health metrics
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable, Awaitable

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Current portfolio state for display."""
    total_value_aud: float = 0.0
    available_balance: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl_today: float = 0.0
    total_positions: int = 0
    exposure_pct: float = 0.0


@dataclass
class PositionInfo:
    """Position information for display."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    pnl_pct: float
    strategy: str


@dataclass
class TradeInfo:
    """Trade information for display."""
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    pnl: float
    strategy: str


@dataclass
class StrategyStats:
    """Strategy performance statistics."""
    name: str
    trades_today: int
    win_rate: float
    avg_pnl: float
    total_pnl: float
    active_signals: int


@dataclass
class SystemHealth:
    """System health metrics."""
    status: str  # "healthy", "degraded", "error"
    uptime_seconds: float
    exchanges_connected: int
    websocket_latency_ms: float
    cpu_usage_pct: float
    memory_usage_pct: float
    last_signal_age_seconds: float


class Dashboard:
    """
    Rich terminal dashboard for Argus trading system.

    Displays real-time information about portfolio, positions,
    trades, and system health.
    """

    def __init__(
        self,
        refresh_rate: float = 1.0,
        console: Optional[Console] = None,
    ) -> None:
        self.refresh_rate = refresh_rate
        self.console = console or Console()
        self._running = False

        # State
        self._portfolio = PortfolioState()
        self._positions: List[PositionInfo] = []
        self._recent_trades: List[TradeInfo] = []
        self._strategy_stats: List[StrategyStats] = []
        self._health = SystemHealth(
            status="unknown",
            uptime_seconds=0,
            exchanges_connected=0,
            websocket_latency_ms=0,
            cpu_usage_pct=0,
            memory_usage_pct=0,
            last_signal_age_seconds=0,
        )
        self._log_messages: List[str] = []
        self._max_log_lines = 10
        self._start_time = datetime.now(timezone.utc)

        # Data providers (callbacks)
        self._portfolio_provider: Optional[Callable[[], Awaitable[PortfolioState]]] = None
        self._positions_provider: Optional[Callable[[], Awaitable[List[PositionInfo]]]] = None
        self._trades_provider: Optional[Callable[[], Awaitable[List[TradeInfo]]]] = None
        self._health_provider: Optional[Callable[[], Awaitable[SystemHealth]]] = None

    def set_portfolio_provider(
        self,
        provider: Callable[[], Awaitable[PortfolioState]],
    ) -> None:
        """Set callback to fetch portfolio state."""
        self._portfolio_provider = provider

    def set_positions_provider(
        self,
        provider: Callable[[], Awaitable[List[PositionInfo]]],
    ) -> None:
        """Set callback to fetch positions."""
        self._positions_provider = provider

    def set_trades_provider(
        self,
        provider: Callable[[], Awaitable[List[TradeInfo]]],
    ) -> None:
        """Set callback to fetch recent trades."""
        self._trades_provider = provider

    def set_health_provider(
        self,
        provider: Callable[[], Awaitable[SystemHealth]],
    ) -> None:
        """Set callback to fetch system health."""
        self._health_provider = provider

    def update_portfolio(self, portfolio: PortfolioState) -> None:
        """Update portfolio state."""
        self._portfolio = portfolio

    def update_positions(self, positions: List[PositionInfo]) -> None:
        """Update positions list."""
        self._positions = positions

    def add_trade(self, trade: TradeInfo) -> None:
        """Add a trade to the log."""
        self._recent_trades.insert(0, trade)
        self._recent_trades = self._recent_trades[:20]  # Keep last 20

    def update_health(self, health: SystemHealth) -> None:
        """Update system health."""
        self._health = health

    def log(self, message: str) -> None:
        """Add a log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log_messages.insert(0, f"[{timestamp}] {message}")
        self._log_messages = self._log_messages[:self._max_log_lines]

    async def start(self) -> None:
        """Start the dashboard."""
        self._running = True
        self._start_time = datetime.now(timezone.utc)

        with Live(
            self._generate_layout(),
            console=self.console,
            refresh_per_second=1 / self.refresh_rate,
            screen=True,
        ) as live:
            while self._running:
                # Fetch data from providers
                await self._fetch_data()

                # Update display
                live.update(self._generate_layout())

                await asyncio.sleep(self.refresh_rate)

    async def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False

    async def _fetch_data(self) -> None:
        """Fetch data from providers."""
        try:
            if self._portfolio_provider:
                self._portfolio = await self._portfolio_provider()
            if self._positions_provider:
                self._positions = await self._positions_provider()
            if self._trades_provider:
                self._recent_trades = await self._trades_provider()
            if self._health_provider:
                self._health = await self._health_provider()
        except Exception as e:
            logger.error("Failed to fetch dashboard data: %s", e)

    def _generate_layout(self) -> Layout:
        """Generate the dashboard layout."""
        layout = Layout()

        # Main structure
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )

        # Main area split
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1),
        )

        # Left side
        layout["left"].split_column(
            Layout(name="portfolio", size=8),
            Layout(name="positions", ratio=1),
            Layout(name="trades", size=12),
        )

        # Right side
        layout["right"].split_column(
            Layout(name="health", size=10),
            Layout(name="strategies", ratio=1),
            Layout(name="log", size=12),
        )

        # Populate panels
        layout["header"].update(self._make_header())
        layout["portfolio"].update(self._make_portfolio_panel())
        layout["positions"].update(self._make_positions_panel())
        layout["trades"].update(self._make_trades_panel())
        layout["health"].update(self._make_health_panel())
        layout["strategies"].update(self._make_strategies_panel())
        layout["log"].update(self._make_log_panel())
        layout["footer"].update(self._make_footer())

        return layout

    def _make_header(self) -> Panel:
        """Create header panel."""
        uptime = datetime.now(timezone.utc) - self._start_time
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        title = Text()
        title.append("⚡ ", style="yellow")
        title.append("ARGUS TRADING SYSTEM", style="bold cyan")
        title.append(" ⚡", style="yellow")

        status_color = {
            "healthy": "green",
            "degraded": "yellow",
            "error": "red",
        }.get(self._health.status, "white")

        status = Text()
        status.append(f"  Status: ", style="dim")
        status.append(f"● {self._health.status.upper()}", style=status_color)
        status.append(f"  |  Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}", style="dim")
        status.append(f"  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")

        return Panel(
            Group(title, status),
            box=box.DOUBLE,
            style="blue",
        )

    def _make_portfolio_panel(self) -> Panel:
        """Create portfolio overview panel."""
        p = self._portfolio

        table = Table(box=None, show_header=False, padding=(0, 2))
        table.add_column("Label", style="dim")
        table.add_column("Value", justify="right")
        table.add_column("Label2", style="dim")
        table.add_column("Value2", justify="right")

        pnl_color = "green" if p.unrealized_pnl >= 0 else "red"
        pnl_today_color = "green" if p.realized_pnl_today >= 0 else "red"

        table.add_row(
            "Total Value:",
            f"[bold]${p.total_value_aud:,.2f}[/bold]",
            "Unrealized P&L:",
            f"[{pnl_color}]${p.unrealized_pnl:+,.2f}[/{pnl_color}]",
        )
        table.add_row(
            "Available:",
            f"${p.available_balance:,.2f}",
            "Today's P&L:",
            f"[{pnl_today_color}]${p.realized_pnl_today:+,.2f}[/{pnl_today_color}]",
        )
        table.add_row(
            "Positions:",
            f"{p.total_positions}",
            "Exposure:",
            f"{p.exposure_pct:.1%}",
        )

        return Panel(
            table,
            title="[bold]💰 Portfolio[/bold]",
            border_style="green",
        )

    def _make_positions_panel(self) -> Panel:
        """Create positions panel."""
        table = Table(box=box.SIMPLE, padding=(0, 1))
        table.add_column("Symbol", style="cyan")
        table.add_column("Side")
        table.add_column("Qty", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("Strategy", style="dim")

        for pos in self._positions[:8]:  # Limit display
            side_color = "green" if pos.side.lower() == "long" else "red"
            pnl_color = "green" if pos.unrealized_pnl >= 0 else "red"

            table.add_row(
                pos.symbol,
                f"[{side_color}]{pos.side.upper()}[/{side_color}]",
                f"{pos.quantity:.6f}",
                f"${pos.entry_price:.2f}",
                f"${pos.current_price:.2f}",
                f"[{pnl_color}]${pos.unrealized_pnl:+.2f} ({pos.pnl_pct:+.1%})[/{pnl_color}]",
                pos.strategy,
            )

        if not self._positions:
            table.add_row("No open positions", "", "", "", "", "", "")

        return Panel(
            table,
            title="[bold]📊 Positions[/bold]",
            border_style="cyan",
        )

    def _make_trades_panel(self) -> Panel:
        """Create recent trades panel."""
        table = Table(box=box.SIMPLE, padding=(0, 1))
        table.add_column("Time", style="dim")
        table.add_column("Symbol", style="cyan")
        table.add_column("Side")
        table.add_column("Qty", justify="right")
        table.add_column("Price", justify="right")
        table.add_column("P&L", justify="right")

        for trade in self._recent_trades[:8]:
            side_color = "green" if trade.side.lower() == "buy" else "red"
            pnl_color = "green" if trade.pnl >= 0 else "red"

            table.add_row(
                trade.timestamp.strftime("%H:%M:%S"),
                trade.symbol,
                f"[{side_color}]{trade.side.upper()}[/{side_color}]",
                f"{trade.quantity:.6f}",
                f"${trade.price:.2f}",
                f"[{pnl_color}]${trade.pnl:+.2f}[/{pnl_color}]",
            )

        if not self._recent_trades:
            table.add_row("No trades yet", "", "", "", "", "")

        return Panel(
            table,
            title="[bold]📈 Recent Trades[/bold]",
            border_style="yellow",
        )

    def _make_health_panel(self) -> Panel:
        """Create system health panel."""
        h = self._health

        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")

        # Status indicator
        status_symbol = {"healthy": "✓", "degraded": "⚠", "error": "✗"}.get(h.status, "?")
        status_color = {"healthy": "green", "degraded": "yellow", "error": "red"}.get(h.status, "white")

        table.add_row("System Status:", f"[{status_color}]{status_symbol} {h.status.upper()}[/{status_color}]")
        table.add_row("Exchanges:", f"{h.exchanges_connected} connected")
        table.add_row("WS Latency:", f"{h.websocket_latency_ms:.0f}ms")
        table.add_row("CPU Usage:", f"{h.cpu_usage_pct:.1f}%")
        table.add_row("Memory:", f"{h.memory_usage_pct:.1f}%")
        table.add_row("Last Signal:", f"{h.last_signal_age_seconds:.0f}s ago")

        return Panel(
            table,
            title="[bold]🔧 System Health[/bold]",
            border_style="magenta",
        )

    def _make_strategies_panel(self) -> Panel:
        """Create strategies panel."""
        table = Table(box=box.SIMPLE, padding=(0, 1))
        table.add_column("Strategy", style="cyan")
        table.add_column("Trades", justify="right")
        table.add_column("Win%", justify="right")
        table.add_column("P&L", justify="right")

        for strat in self._strategy_stats[:6]:
            pnl_color = "green" if strat.total_pnl >= 0 else "red"
            win_color = "green" if strat.win_rate >= 0.5 else "yellow"

            table.add_row(
                strat.name,
                str(strat.trades_today),
                f"[{win_color}]{strat.win_rate:.0%}[/{win_color}]",
                f"[{pnl_color}]${strat.total_pnl:+.2f}[/{pnl_color}]",
            )

        if not self._strategy_stats:
            # Show default strategies
            for name in ["momentum", "mean_reversion", "breakout", "scalping"]:
                table.add_row(name, "0", "-", "$0.00")

        return Panel(
            table,
            title="[bold]🎯 Strategies[/bold]",
            border_style="blue",
        )

    def _make_log_panel(self) -> Panel:
        """Create log panel."""
        log_text = Text()

        for msg in self._log_messages[:self._max_log_lines]:
            if "ERROR" in msg.upper():
                log_text.append(msg + "\n", style="red")
            elif "WARN" in msg.upper():
                log_text.append(msg + "\n", style="yellow")
            elif "BUY" in msg.upper():
                log_text.append(msg + "\n", style="green")
            elif "SELL" in msg.upper():
                log_text.append(msg + "\n", style="red")
            else:
                log_text.append(msg + "\n", style="dim")

        if not self._log_messages:
            log_text.append("Waiting for activity...", style="dim italic")

        return Panel(
            log_text,
            title="[bold]📝 Log[/bold]",
            border_style="white",
        )

    def _make_footer(self) -> Panel:
        """Create footer panel."""
        text = Text()
        text.append("  [Q] Quit", style="dim")
        text.append("  |  ", style="dim")
        text.append("[P] Pause", style="dim")
        text.append("  |  ", style="dim")
        text.append("[R] Refresh", style="dim")
        text.append("  |  ", style="dim")
        text.append("[H] Help", style="dim")
        text.append("  |  ", style="dim")
        text.append("Argus v2.0", style="cyan dim")

        return Panel(text, box=box.ROUNDED, style="dim")


def create_dashboard(refresh_rate: float = 1.0) -> Dashboard:
    """Create a dashboard instance."""
    return Dashboard(refresh_rate=refresh_rate)


# Simple status display for non-interactive mode
def print_status(
    portfolio: PortfolioState,
    positions: List[PositionInfo],
    console: Optional[Console] = None,
) -> None:
    """Print a simple status update (non-interactive)."""
    console = console or Console()

    # Portfolio summary
    pnl_color = "green" if portfolio.unrealized_pnl >= 0 else "red"
    console.print(
        f"Portfolio: ${portfolio.total_value_aud:,.2f} | "
        f"P&L: [{pnl_color}]${portfolio.unrealized_pnl:+,.2f}[/{pnl_color}] | "
        f"Positions: {portfolio.total_positions}"
    )

    # Positions table
    if positions:
        table = Table(box=box.SIMPLE)
        table.add_column("Symbol")
        table.add_column("Side")
        table.add_column("P&L")

        for pos in positions[:5]:
            pnl_color = "green" if pos.unrealized_pnl >= 0 else "red"
            table.add_row(
                pos.symbol,
                pos.side,
                f"[{pnl_color}]${pos.unrealized_pnl:+.2f}[/{pnl_color}]",
            )

        console.print(table)
