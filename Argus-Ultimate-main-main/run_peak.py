#!/usr/bin/env python3
"""
Argus Trading System - PEAK PERFORMANCE MODE
=============================================

Maximum alpha generation with:
- Best performing trading pairs (high volume + volatility)
- Multi-factor Peak Alpha strategy
- Aggressive position sizing
- Fast 10-second cycles
- Automatic stop/take-profit management

Usage:
    python run_peak.py --capital 1000
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from typing import List, Optional, Dict

# Windows console fix
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Performance: uvloop on Linux
try:
    import uvloop
    uvloop.install()
    print("[OK] uvloop enabled - maximum performance!")
except ImportError:
    print("[--] uvloop not available (Windows)")

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("peak_trading.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("argus.peak")


# ============================================================================
# OPTIMAL TRADING PAIRS - Selected for maximum alpha potential
# ============================================================================
PEAK_TRADING_PAIRS = [
    # Tier 1: Highest liquidity + volatility (best for scalping/momentum)
    "BTC/AUD",      # King - highest volume, tightest spreads
    "ETH/AUD",      # Queen - second highest, great momentum

    # Tier 2: High volatility altcoins (best for breakouts/trends)
    "SOL/AUD",      # High beta, strong trends
    "AVAX/AUD",     # Volatile, good mean reversion
    "LINK/AUD",     # Consistent volatility

    # Tier 3: Meme/high-vol (best for scalping spikes)
    "DOGE/AUD",     # Retail favorite, volatile
    "XRP/AUD",      # News-driven moves

    # Tier 4: DeFi tokens (good for momentum)
    "AAVE/AUD",     # DeFi leader
    "UNI/AUD",      # High correlation to ETH

    # Tier 5: Layer 2s (emerging trends)
    "MATIC/AUD",    # Polygon - growing adoption
    "ARB/AUD",      # Arbitrum - if available
]


class PeakTradingBot:
    """
    Peak performance trading bot for maximum returns.
    """

    def __init__(
        self,
        capital: float = 1000.0,
        symbols: Optional[List[str]] = None,
        exchange: str = "kraken",
        aggressive: bool = True,
    ):
        self.initial_capital = capital
        self.capital = capital
        self.symbols = symbols or PEAK_TRADING_PAIRS
        self.exchange_name = exchange
        self.aggressive = aggressive

        # Components
        self.exchange = None
        self.data_store = None
        self.pipeline = None
        self.strategies = []

        # State
        self.running = False
        self.positions: Dict[str, dict] = {}
        self.trades: List[dict] = []
        self.signals_generated = 0
        self.trades_executed = 0
        self.start_time = None

        # P&L tracking
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_capital = capital
        self.min_capital = capital

    async def start(self):
        """Initialize and start peak trading."""
        self.running = True
        self.start_time = datetime.now(timezone.utc)

        console.print(Panel.fit(
            "[bold red]ARGUS PEAK PERFORMANCE MODE[/bold red]\n"
            f"Capital: [green]${self.capital:,.2f} AUD[/green]\n"
            f"Pairs: [yellow]{len(self.symbols)} assets[/yellow]\n"
            f"Mode: [red]{'AGGRESSIVE' if self.aggressive else 'BALANCED'}[/red]\n"
            f"Exchange: [blue]{self.exchange_name}[/blue]",
            title="PEAK ALPHA",
            border_style="red",
        ))

        try:
            await self._init_exchange()
            await self._filter_available_symbols()
            await self._init_data_store()
            await self._init_strategies()
            await self._init_pipeline()
            await self._trading_loop()

        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        except Exception as e:
            logger.exception("Fatal error: %s", e)
            raise
        finally:
            await self.stop()

    async def stop(self):
        """Clean shutdown."""
        self.running = False
        console.print("\n[yellow]Shutting down...[/yellow]")

        # Close all positions at market
        await self._close_all_positions("shutdown")

        if self.exchange:
            await self.exchange.close()

        if self.data_store:
            await self.data_store.close()

        self._print_summary()

    async def _init_exchange(self):
        """Initialize exchange connection."""
        from exchanges.centralized.kraken import KrakenClient

        console.print("  [dim]Connecting to Kraken...[/dim]")

        self.exchange = KrakenClient(
            api_key=os.environ.get("KRAKEN_API_KEY"),
            secret=os.environ.get("KRAKEN_API_SECRET"),
            dry_run=True,
        )

        connected = await self.exchange.connect()
        if connected:
            console.print("  [green]OK[/green] Kraken connected (paper mode)")
        else:
            raise ConnectionError("Failed to connect to Kraken")

    async def _filter_available_symbols(self):
        """Filter to only available symbols on exchange."""
        console.print("  [dim]Checking available pairs...[/dim]")

        available = []
        for symbol in self.symbols:
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                if ticker and ticker.get("last"):
                    available.append(symbol)
                    console.print(f"    [green]+[/green] {symbol} @ ${ticker['last']:,.2f}")
            except Exception:
                console.print(f"    [red]-[/red] {symbol} not available")

        self.symbols = available
        console.print(f"  [green]OK[/green] Trading {len(self.symbols)} pairs")

    async def _init_data_store(self):
        """Initialize data storage."""
        from data.store import DataStore

        console.print("  [dim]Initializing data store...[/dim]")

        self.data_store = DataStore()
        await self.data_store.connect()
        console.print("  [green]OK[/green] Data store ready")

    async def _init_strategies(self):
        """Initialize peak performance strategies."""
        from strategies.peak_alpha import PeakAlphaStrategy, PeakAlphaConfig
        from strategies.momentum import MomentumStrategy, MomentumConfig
        from strategies.mean_reversion import MeanReversionStrategy, MeanReversionConfig
        from strategies.scalping import ScalpingStrategy, ScalpingConfig

        console.print("  [dim]Loading PEAK strategies...[/dim]")

        if self.aggressive:
            # Aggressive mode: lower thresholds, more signals
            self.strategies = [
                # Peak Alpha - multi-factor (primary)
                PeakAlphaStrategy(PeakAlphaConfig(
                    name="peak_alpha",
                    min_confidence=0.50,
                    min_strength=0.35,
                    min_factor_agreement=2,  # Only need 2 factors
                    stop_loss_atr_mult=1.5,
                    take_profit_atr_mult=3.0,
                )),
                # Aggressive momentum
                MomentumStrategy(MomentumConfig(
                    name="momentum",
                    min_confidence=0.45,
                    min_strength=0.30,
                    rsi_oversold=35.0,
                    rsi_overbought=65.0,
                    require_macd_crossover=False,  # Don't require crossover
                )),
                # Tight mean reversion
                MeanReversionStrategy(MeanReversionConfig(
                    name="mean_reversion",
                    min_confidence=0.45,
                    min_strength=0.30,
                    zscore_entry_threshold=1.5,
                )),
                # Quick scalping
                ScalpingStrategy(ScalpingConfig(
                    name="scalping",
                    min_confidence=0.50,
                    min_strength=0.40,
                    take_profit_pct=0.004,  # 0.4% quick profit
                    stop_loss_pct=0.002,    # 0.2% tight stop
                    max_trades_per_hour=30,
                )),
            ]
        else:
            # Balanced mode: standard thresholds
            self.strategies = [
                PeakAlphaStrategy(PeakAlphaConfig(name="peak_alpha")),
                MomentumStrategy(MomentumConfig(name="momentum")),
                MeanReversionStrategy(MeanReversionConfig(name="mean_reversion")),
            ]

        for strat in self.strategies:
            console.print(f"    [green]+[/green] {strat.name}")

    async def _init_pipeline(self):
        """Initialize execution pipeline with aggressive settings."""
        from execution.pipeline import ExecutionPipeline, PipelineConfig
        from risk.position_sizing import PositionSizer, SizingConfig, SizingMethod

        console.print("  [dim]Setting up PEAK pipeline...[/dim]")

        if self.aggressive:
            # Aggressive pipeline config
            pipeline_config = PipelineConfig(
                min_confidence=0.45,
                min_strength=0.30,
                max_position_value_aud=self.capital * 0.30,  # 30% max per position
                max_portfolio_risk_pct=0.04,  # 4% risk per trade
                require_stop_loss=True,
                signal_ttl_seconds=30.0,  # Fresh signals only
            )

            # Aggressive position sizing
            sizing_config = SizingConfig(
                method=SizingMethod.DYNAMIC,
                fixed_risk_pct=0.025,          # 2.5% risk per trade
                target_risk_pct=0.03,          # 3% target risk
                max_position_pct=0.25,         # 25% max per position
                min_position_pct=0.03,         # 3% minimum
                kelly_fraction=0.35,           # 35% Kelly
                min_confidence_scale=0.65,     # Less scaling at low confidence
                high_vol_scale=0.6,            # Still trade in high vol
                regime_scaling=True,
            )
        else:
            pipeline_config = PipelineConfig(
                min_confidence=0.5,
                max_position_value_aud=self.capital * 0.20,
            )
            sizing_config = SizingConfig()

        self.pipeline = ExecutionPipeline(
            exchange=self.exchange,
            config=pipeline_config,
            position_sizer=PositionSizer(sizing_config),
        )

        console.print("  [green]OK[/green] PEAK pipeline ready")

    async def _trading_loop(self):
        """Main trading loop - FAST 10-second cycles."""
        import pandas as pd
        from core.types import MarketRegime, SignalAction

        console.print("\n[bold red]PEAK TRADING STARTED![/bold red] Press Ctrl+C to stop.\n")

        cycle = 0
        while self.running:
            cycle += 1

            try:
                for symbol in self.symbols:
                    # Fetch latest data
                    ohlcv_raw = await self.exchange.fetch_ohlcv(
                        symbol,
                        timeframe="1m",
                        limit=120,  # More data for indicators
                    )

                    if not ohlcv_raw or len(ohlcv_raw) < 60:
                        continue

                    df = pd.DataFrame(
                        ohlcv_raw,
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )

                    # Store data
                    await self.data_store.ohlcv.save_candles(symbol, "1m", ohlcv_raw)

                    current_price = df["close"].iloc[-1]

                    # Run all strategies
                    for strategy in self.strategies:
                        try:
                            signal = await strategy.generate_signal(
                                symbol=symbol,
                                ohlcv=df,
                                regime=MarketRegime.UNKNOWN,
                            )

                            if signal:
                                self.signals_generated += 1
                                logger.info(
                                    "SIGNAL: %s %s @ %.2f (conf=%.2f, str=%.2f) [%s]",
                                    signal.action.value,
                                    symbol,
                                    current_price,
                                    signal.confidence,
                                    signal.strength,
                                    strategy.name,
                                )

                                # Check if we already have a position
                                if symbol in self.positions:
                                    pos = self.positions[symbol]
                                    # If signal is opposite direction, close first
                                    if signal.action == SignalAction.SELL and pos["quantity"] > 0:
                                        await self._close_position(symbol, current_price, "reverse_signal")
                                        continue
                                elif signal.action == SignalAction.SELL:
                                    # Can't sell if no position (no shorting for now)
                                    continue

                                # Execute signal
                                result = await self.pipeline.execute_signal(
                                    signal=signal,
                                    capital=self.capital,
                                )

                                if result.success:
                                    self.trades_executed += 1
                                    self._record_trade(result)

                        except Exception as e:
                            logger.warning("Strategy %s error for %s: %s", strategy.name, symbol, e)

                # Check exits every cycle
                await self._check_exits()

                # Update unrealized P&L
                await self._update_unrealized_pnl()

                # Print status every 30 cycles (~5 min)
                if cycle % 30 == 0:
                    self._print_status(cycle)

                # Fast cycles for maximum opportunity capture
                await asyncio.sleep(10)  # 10 second cycles

            except Exception as e:
                logger.error("Trading cycle error: %s", e)
                await asyncio.sleep(5)

    async def _check_exits(self):
        """Check all positions for stop/take-profit."""
        for symbol in list(self.positions.keys()):
            pos = self.positions.get(symbol)
            if not pos or pos["quantity"] <= 0:
                continue

            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                current_price = ticker.get("last", 0)
                if not current_price:
                    continue

                # Update position price
                pos["current_price"] = current_price

                # Check stop loss
                if pos.get("stop_loss") and current_price <= pos["stop_loss"]:
                    logger.info("STOP LOSS: %s @ %.2f (entry: %.2f)", symbol, current_price, pos["avg_price"])
                    await self._close_position(symbol, current_price, "stop_loss")

                # Check take profit
                elif pos.get("take_profit") and current_price >= pos["take_profit"]:
                    logger.info("TAKE PROFIT: %s @ %.2f (entry: %.2f)", symbol, current_price, pos["avg_price"])
                    await self._close_position(symbol, current_price, "take_profit")

                # Trailing stop: move stop up if price moves up significantly
                elif current_price > pos["avg_price"] * 1.02:  # 2% profit
                    new_stop = current_price * 0.985  # Trail at 1.5%
                    if new_stop > pos.get("stop_loss", 0):
                        pos["stop_loss"] = new_stop
                        logger.debug("Trailing stop moved to %.2f for %s", new_stop, symbol)

            except Exception as e:
                logger.warning("Exit check error for %s: %s", symbol, e)

    async def _close_position(self, symbol: str, price: float, reason: str):
        """Close a position and record P&L."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        qty = pos["quantity"]
        sell_value = qty * price
        cost_basis = pos["cost_basis"]
        pnl = sell_value - cost_basis

        self.realized_pnl += pnl
        self.total_pnl = self.realized_pnl  # + unrealized

        if pnl >= 0:
            self.winning_trades += 1
            pnl_color = "green"
        else:
            self.losing_trades += 1
            pnl_color = "red"

        self.capital += sell_value
        self.max_capital = max(self.max_capital, self.capital)
        self.min_capital = min(self.min_capital, self.capital)

        logger.info(
            "CLOSED %s (%s): qty=%.6f, entry=%.2f, exit=%.2f, P&L=$%.2f",
            symbol, reason, qty, pos["avg_price"], price, pnl
        )
        console.print(f"  [{pnl_color}]CLOSED {symbol}: P&L ${pnl:+.2f} AUD ({reason})[/{pnl_color}]")

        del self.positions[symbol]

    async def _close_all_positions(self, reason: str):
        """Close all open positions."""
        for symbol in list(self.positions.keys()):
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                price = ticker.get("last", self.positions[symbol]["avg_price"])
                await self._close_position(symbol, price, reason)
            except Exception as e:
                logger.error("Failed to close %s: %s", symbol, e)

    async def _update_unrealized_pnl(self):
        """Update unrealized P&L for all positions."""
        unrealized = 0.0
        for symbol, pos in self.positions.items():
            if pos.get("current_price"):
                unrealized += (pos["current_price"] - pos["avg_price"]) * pos["quantity"]
        self.total_pnl = self.realized_pnl + unrealized

    def _record_trade(self, result):
        """Record an executed trade."""
        from core.types import SignalAction

        symbol = result.signal.symbol

        trade = {
            "timestamp": result.timestamp,
            "symbol": symbol,
            "action": result.signal.action.value,
            "quantity": result.filled_quantity,
            "price": result.filled_price,
            "cost": result.cost,
            "fee": result.fee,
            "strategy": result.signal.strategy_name,
        }
        self.trades.append(trade)

        if result.signal.action == SignalAction.BUY:
            if symbol not in self.positions:
                self.positions[symbol] = {
                    "quantity": 0,
                    "avg_price": 0,
                    "cost_basis": 0,
                    "entry_time": result.timestamp,
                }

            pos = self.positions[symbol]
            total_qty = pos["quantity"] + result.filled_quantity
            pos["cost_basis"] += result.cost
            pos["avg_price"] = pos["cost_basis"] / total_qty if total_qty > 0 else 0
            pos["quantity"] = total_qty
            pos["stop_loss"] = result.signal.stop_loss
            pos["take_profit"] = result.signal.take_profit
            pos["current_price"] = result.filled_price

            self.capital -= result.cost + result.fee

            console.print(
                f"  [cyan]OPENED {symbol}: {result.filled_quantity:.6f} @ ${result.filled_price:.2f} "
                f"(SL: ${result.signal.stop_loss:.2f}, TP: ${result.signal.take_profit:.2f})[/cyan]"
            )

    def _print_status(self, cycle: int):
        """Print current status."""
        elapsed = datetime.now(timezone.utc) - self.start_time
        hours = elapsed.total_seconds() / 3600

        pnl = self.total_pnl
        pnl_pct = (pnl / self.initial_capital) * 100
        pnl_color = "green" if pnl >= 0 else "red"

        win_rate = (self.winning_trades / max(1, self.winning_trades + self.losing_trades)) * 100

        console.print(
            f"\n[bold]Cycle {cycle}[/bold] | "
            f"Capital: [bold]${self.capital:,.2f}[/bold] | "
            f"P&L: [{pnl_color}]${pnl:+,.2f} ({pnl_pct:+.2f}%)[/{pnl_color}] | "
            f"Signals: {self.signals_generated} | "
            f"Trades: {self.trades_executed} | "
            f"Win Rate: {win_rate:.0f}% | "
            f"Positions: {len(self.positions)} | "
            f"Runtime: {hours:.1f}h"
        )

        # Show open positions
        if self.positions:
            for symbol, pos in self.positions.items():
                unrealized = (pos.get("current_price", pos["avg_price"]) - pos["avg_price"]) * pos["quantity"]
                u_color = "green" if unrealized >= 0 else "red"
                console.print(
                    f"    {symbol}: {pos['quantity']:.6f} @ ${pos['avg_price']:.2f} "
                    f"[{u_color}](${unrealized:+.2f})[/{u_color}]"
                )

    def _print_summary(self):
        """Print final trading summary."""
        pnl = self.total_pnl
        pnl_pct = (pnl / self.initial_capital) * 100
        pnl_color = "green" if pnl >= 0 else "red"

        total_trades = self.winning_trades + self.losing_trades
        win_rate = (self.winning_trades / max(1, total_trades)) * 100

        elapsed = datetime.now(timezone.utc) - self.start_time if self.start_time else None

        console.print("\n")
        console.print(Panel.fit(
            f"[bold]PEAK TRADING SUMMARY[/bold]\n\n"
            f"Initial Capital:  ${self.initial_capital:,.2f}\n"
            f"Final Capital:    ${self.capital:,.2f}\n"
            f"Total P&L:        [{pnl_color}]${pnl:+,.2f} ({pnl_pct:+.2f}%)[/{pnl_color}]\n"
            f"Max Capital:      ${self.max_capital:,.2f}\n"
            f"Min Capital:      ${self.min_capital:,.2f}\n\n"
            f"Total Signals:    {self.signals_generated}\n"
            f"Total Trades:     {self.trades_executed}\n"
            f"Winning Trades:   {self.winning_trades}\n"
            f"Losing Trades:    {self.losing_trades}\n"
            f"Win Rate:         {win_rate:.1f}%\n"
            f"Runtime:          {elapsed}",
            title="Results",
            border_style="red",
        ))


@click.command()
@click.option("--capital", "-c", default=1000.0, help="Starting capital in AUD")
@click.option("--aggressive/--balanced", default=True, help="Aggressive or balanced mode")
def main(capital: float, aggressive: bool):
    """Run Argus in PEAK PERFORMANCE mode."""

    def signal_handler(sig, frame):
        console.print("\n[yellow]Interrupt received, shutting down...[/yellow]")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    bot = PeakTradingBot(
        capital=capital,
        aggressive=aggressive,
    )

    asyncio.run(bot.start())


if __name__ == "__main__":
    main()
