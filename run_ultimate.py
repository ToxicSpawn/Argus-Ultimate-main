#!/usr/bin/env python3
"""
Argus Trading System - ULTIMATE MODE
=====================================

The absolute peak of algorithmic trading:
- 17+ strategies from multiple strategy libraries
- Tier ensemble strategies with consensus voting
- Regime-switching adaptive strategies
- Candlestick pattern recognition
- HFT grid trading
- Multi-factor Peak Alpha
- Aggressive position sizing with Kelly criterion
- 5-second ultra-fast cycles
- Trailing stops and automatic position management
- Live microstructure signals: OFI / VPIN / MicropriceDrift / DeepLOB
- Regime-aware quoting via RegimeScheduler
- Tick-to-trade latency telemetry
- Tick-level OFI/VPIN via KrakenWSClient <-> WSFeedAdapter <-> LiveSignalBus
- Cross-strategy signal consensus filter (min 2-of-N weighted agreement)
- AdaptiveRiskManager: regime + portfolio + correlation adaptive limits
- Correlation cap: max 3 simultaneous positions (prevents BTC concentration)
- MTFConfluenceFilter: 15m/1h/4h agreement required before entry
- RLExecutionAgent: PPO model gates order size (BUY_SMALL/BUY_LARGE/HOLD)
- StrategyRanker: online Sharpe/win-rate leaderboard per strategy
- StrategyRegistry: runtime enable/disable without restart
- BanditAllocator: Thompson-Sampling capital weighting across strategies
- StrategyDashboard: rich live dashboard (leaderboard + bandit posteriors)

Usage:
    python run_ultimate.py --capital 1000
"""

import asyncio
import logging
import math
import os
import signal
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    import uvloop
    uvloop.install()
    print("[OK] uvloop enabled - MAXIMUM PERFORMANCE!")
except ImportError:
    print("[--] uvloop not available (Windows)")

import click
from rich.console import Console
from rich.panel import Panel
from rich import box

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("ultimate_trading.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("argus.ultimate")


ULTIMATE_PAIRS = [
    "BTC/AUD", "ETH/AUD",
    "SOL/AUD", "XRP/AUD", "LINK/AUD",
    "DOGE/AUD", "ADA/AUD", "DOT/AUD",
    "ATOM/AUD", "LTC/AUD",
]

# Max simultaneous open positions — prevents BTC-correlated overexposure
MAX_CONCURRENT_POSITIONS = 3

# How often (in cycles) to render the strategy dashboard
DASHBOARD_EVERY_N_CYCLES = 120   # ~10 minutes at 5-s cycles


class UltimateTradingBot:
    """
    ULTIMATE trading bot — maximum alpha, adaptive risk control,
    MTF confluence filtering, RL-guided order sizing,
    bandit-allocated capital across strategy arms.
    """

    def __init__(
        self,
        capital: float = 1000.0,
        symbols: Optional[List[str]] = None,
    ):
        self.initial_capital = capital
        self.capital = capital
        self.symbols = symbols or ULTIMATE_PAIRS

        self.exchange = None
        self.data_store = None
        self.pipeline = None
        self.strategies = []
        self.consensus = None
        self.adaptive_risk = None
        self.mtf_filter = None          # MTFConfluenceFilter
        self.rl_agent = None            # RLExecutionAgent

        self.signal_bus = None
        self.ws_adapter = None
        self.regime_integration = None
        self.spread_schedule = None

        self.kraken_adapter = None
        self.ws_client = None
        self._ws_task: Optional[asyncio.Task] = None

        # ── NEW: ranker / registry / bandit / dashboard ──────────────────
        self.ranker = None              # StrategyRanker
        self.registry = None            # StrategyRegistry
        self.bandit = None              # BanditAllocator
        self.dashboard = None           # StrategyDashboard
        # ────────────────────────────────────────────────────────────────

        self.running = False
        self.positions: Dict[str, dict] = {}
        self.trades: List[dict] = []
        self.signals_generated = 0
        self.trades_executed = 0
        self.start_time = None

        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_capital = capital
        self.min_capital = capital
        self.best_trade = 0.0
        self.worst_trade = 0.0

        # Emergency stop state
        self._emergency_stopped = False
        self._peak_capital = capital
        self._daily_start_capital = capital

        # Per-symbol bars-since-last-trade counter for RL state
        self._bars_since_trade: Dict[str, int] = {}

        # Last MTF score per symbol (for status display)
        self._last_mtf_score: Dict[str, float] = {}

        # Last RL decision per symbol (for status display)
        self._last_rl_action: Dict[str, str] = {}

    async def start(self):
        """Initialize and start ULTIMATE trading."""
        self.running = True
        self.start_time = datetime.now(timezone.utc)

        console.print(Panel.fit(
            "[bold magenta]ARGUS ULTIMATE MODE[/bold magenta]\n"
            f"Capital: [green]${self.capital:,.2f} AUD[/green]\n"
            f"Pairs: [yellow]{len(self.symbols)} assets[/yellow]\n"
            f"Strategies: [cyan]17+ multi-factor + consensus + adaptive risk[/cyan]\n"
            f"Mode: [red]MAXIMUM ALPHA[/red]",
            title="ULTIMATE",
            border_style="magenta",
        ))

        try:
            await self._init_exchange()
            await self._filter_available_symbols()
            await self._init_microstructure_stack()
            await self._init_data_store()
            await self._init_all_strategies()
            await self._init_pipeline()
            await self._init_ws_feed()
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
        console.print("\n[yellow]Shutting down ULTIMATE mode...[/yellow]")

        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await asyncio.wait_for(self._ws_task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        if self.ws_client is not None:
            await self.ws_client.close()

        await self._close_all_positions("shutdown")

        if self.exchange:
            await self.exchange.close()

        if self.data_store:
            await self.data_store.close()

        if self.signal_bus is not None:
            try:
                await self.signal_bus.stop()
            except Exception as exc:
                logger.debug("signal_bus.stop() error: %s", exc)

        if self.rl_agent is not None:
            self.rl_agent.reset_episode()

        self._print_final_summary()

    async def _init_exchange(self):
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

    async def _init_microstructure_stack(self):
        console.print("  [dim]Booting microstructure + regime + latency stack...[/dim]")

        from alpha.microstructure import LiveSignalBus, WSFeedAdapter, RegimeIntegration
        from execution.session_spread_schedule import SessionSpreadSchedule

        clean_symbols = [s.split("/")[0] for s in self.symbols]
        self.spread_schedule = SessionSpreadSchedule(base_spread_bps=5.0)

        self.signal_bus = await LiveSignalBus.create(
            symbols=clean_symbols,
            ofi_window=500,
            ofi_alpha=0.94,
            vpin_bucket_volume=1.0,
            vpin_n_buckets=50,
            drift_window=20,
            drift_threshold=0.6,
            enable_deeplob=True,
        )
        deeplob_status = "yes" if getattr(self.signal_bus, "_deeplob", None) else "no"
        console.print(
            f"  [green]OK[/green] LiveSignalBus started "
            f"({len(clean_symbols)} symbols, DeepLOB={deeplob_status})"
        )

        self.ws_adapter = WSFeedAdapter(bus=self.signal_bus)
        console.print("  [green]OK[/green] WSFeedAdapter ready")

        self.regime_integration = RegimeIntegration.build(
            symbols=clean_symbols,
            spread_schedule=self.spread_schedule,
        )
        console.print("  [green]OK[/green] RegimeIntegration ready")

        def _on_vpin_spike(sym: str, vpin: float, threshold: float) -> None:
            logger.warning("VPIN SPIKE on %s (%.3f) — widening spread for 60s", sym, vpin)
            self.spread_schedule.set_override(multiplier=2.0, duration_s=60.0)

        try:
            self.signal_bus._vpin.register_spike_callback(_on_vpin_spike)
            console.print("  [green]OK[/green] VPIN spike -> spread override wired")
        except AttributeError:
            logger.debug("VPIN spike callback not supported on this bus version")

        console.print(
            "  [bold green]MICROSTRUCTURE STACK LIVE[/bold green] — "
            "OFI / VPIN / MicropriceDrift / DeepLOB / RegimeScheduler / LatencyTelemetry"
        )

    async def _init_data_store(self):
        from data.store import DataStore
        console.print("  [dim]Initializing data store...[/dim]")
        self.data_store = DataStore()
        await self.data_store.connect()
        console.print("  [green]OK[/green] Data store ready")

    async def _init_all_strategies(self):
        """Initialize ALL strategies + SignalConsensus + AdaptiveRiskManager
        + MTFConfluenceFilter + RLExecutionAgent + Ranker + Registry + Bandit + Dashboard."""
        console.print("  [dim]Loading ULTIMATE strategy arsenal...[/dim]")

        from strategies.peak_alpha import PeakAlphaStrategy, PeakAlphaConfig
        from strategies.momentum import MomentumStrategy, MomentumConfig
        from strategies.mean_reversion import MeanReversionStrategy, MeanReversionConfig
        from strategies.breakout import BreakoutStrategy
        from strategies.scalping import ScalpingStrategy, ScalpingConfig
        from strategies.advanced_adapter import create_adapted_strategies
        from strategies.signal_consensus import SignalConsensus
        from strategies.mtf_confluence import MTFConfluenceFilter
        from strategies.reinforcement_stub import RLExecutionAgent
        from risk.adaptive_risk_manager import AdaptiveRiskManager

        # ── NEW imports ──────────────────────────────────────────────────
        from strategies.strategy_ranker import StrategyRanker
        from strategies.strategy_registry import StrategyRegistry
        from strategies.bandit_allocator import BanditAllocator
        from monitoring.strategy_dashboard import StrategyDashboard
        # ────────────────────────────────────────────────────────────────

        self.strategies = []

        self.strategies.append(PeakAlphaStrategy(PeakAlphaConfig(
            name="peak_alpha",
            min_confidence=0.45,
            min_strength=0.35,
            min_factor_agreement=2,
        )))
        console.print("    [green]+[/green] peak_alpha (multi-factor)")

        self.strategies.append(MomentumStrategy(MomentumConfig(
            name="momentum",
            min_confidence=0.45,
            min_strength=0.30,
            rsi_oversold=32.0,
            rsi_overbought=68.0,
            require_macd_crossover=False,
        )))
        console.print("    [green]+[/green] momentum (aggressive)")

        self.strategies.append(MeanReversionStrategy(MeanReversionConfig(
            name="mean_reversion",
            min_confidence=0.45,
            min_strength=0.30,
            zscore_entry_threshold=1.5,
        )))
        console.print("    [green]+[/green] mean_reversion (tight)")

        self.strategies.append(BreakoutStrategy())
        console.print("    [green]+[/green] breakout")

        self.strategies.append(ScalpingStrategy(ScalpingConfig(
            name="scalping",
            min_confidence=0.55,
            min_strength=0.45,
            take_profit_pct=0.009,
            stop_loss_pct=0.0045,
            max_trades_per_hour=12,
        )))
        console.print("    [green]+[/green] scalping (fee-corrected: TP=0.9% SL=0.45%)")

        try:
            adapted = create_adapted_strategies()
            for strat in adapted:
                self.strategies.append(strat)
                console.print(f"    [cyan]+[/cyan] {strat.name} (library)")
        except Exception as e:
            logger.warning("Could not load adapted strategies: %s", e)

        # Signal consensus filter
        self.consensus = SignalConsensus(
            mode="weighted",
            min_agreement=0.60,
            min_strategies=2,
        )
        console.print(
            f"  [green]OK[/green] Loaded {len(self.strategies)} strategies + "
            "[bold cyan]SignalConsensus[/bold cyan] (weighted, 60%)")

        # Adaptive risk manager
        self.adaptive_risk = AdaptiveRiskManager(initial_capital=self.initial_capital)
        console.print(
            "  [green]OK[/green] [bold yellow]AdaptiveRiskManager[/bold yellow] ready — "
            "regime + portfolio + correlation + emergency-stop"
        )

        # MTF confluence filter
        self.mtf_filter = MTFConfluenceFilter(
            timeframes=["15m", "1h", "4h"],
            min_agreeing_tfs=2,
            min_confluence_score=0.60,
            fast_ema=9,
            slow_ema=21,
            trend_ema=50,
            require_higher_tf_agreement=True,
        )
        console.print(
            "  [green]OK[/green] [bold blue]MTFConfluenceFilter[/bold blue] ready — "
            "15m/1h/4h, min_tfs=2, score>=60%, 4h must not oppose"
        )

        # RL execution agent
        rl_model_path = "models/rl_execution_agent_ppo.zip"
        self.rl_agent = RLExecutionAgent(
            model_path=rl_model_path if os.path.exists(rl_model_path) else None,
            max_position_usd=self.initial_capital,
            slippage_budget_bps=20.0,
        )
        rl_status = "[MODEL]" if self.rl_agent.is_model_loaded else "[RULES]"
        rl_color = "green" if self.rl_agent.is_model_loaded else "yellow"
        console.print(
            f"  [green]OK[/green] [{rl_color}]RLExecutionAgent {rl_status}[/{rl_color}] — "
            "PPO order-sizing gate (BUY_SMALL=25% / BUY_LARGE=100% / HOLD=skip) | "
            f"rl_available={self.rl_agent.rl_available}"
        )

        # ── NEW: StrategyRanker ──────────────────────────────────────────
        self.ranker = StrategyRanker()
        console.print("  [green]OK[/green] [bold green]StrategyRanker[/bold green] ready — online Sharpe + win-rate leaderboard")

        # ── NEW: StrategyRegistry ────────────────────────────────────────
        self.registry = StrategyRegistry()
        self.registry.register_all(self.strategies, tags=["core"])
        console.print(
            f"  [green]OK[/green] [bold green]StrategyRegistry[/bold green] ready — "
            f"{len(self.strategies)} strategies registered"
        )

        # ── NEW: BanditAllocator ─────────────────────────────────────────
        self.bandit = BanditAllocator(
            decay_halflife_trades=100,
            min_weight=0.05,
            n_thompson_samples=3,
        )
        console.print("  [green]OK[/green] [bold green]BanditAllocator[/bold green] ready — Thompson-Sampling capital allocation")

        # ── NEW: StrategyDashboard ───────────────────────────────────────
        self.dashboard = StrategyDashboard(
            ranker=self.ranker,
            registry=self.registry,
            bandit=self.bandit,
        )
        console.print("  [green]OK[/green] [bold green]StrategyDashboard[/bold green] ready — renders every %d cycles", DASHBOARD_EVERY_N_CYCLES)

    async def _init_pipeline(self):
        from execution.pipeline import ExecutionPipeline, PipelineConfig
        from risk.position_sizing import PositionSizer, SizingConfig, SizingMethod
        from exchanges.centralized.kraken import build_kraken_adapter

        console.print("  [dim]Setting up ULTIMATE pipeline...[/dim]")

        pipeline_config = PipelineConfig(
            min_confidence=0.45,
            min_strength=0.30,
            max_position_value_aud=self.capital * 0.35,
            max_portfolio_risk_pct=0.05,
            require_stop_loss=True,
            signal_ttl_seconds=20.0,
        )

        sizing_config = SizingConfig(
            method=SizingMethod.DYNAMIC,
            fixed_risk_pct=0.03,
            target_risk_pct=0.04,
            max_position_pct=0.30,
            min_position_pct=0.03,
            kelly_fraction=0.40,
            min_confidence_scale=0.70,
            high_vol_scale=0.65,
            regime_scaling=True,
        )

        self.kraken_adapter = build_kraken_adapter(
            exchange_client=self.exchange.client,
            dry_run=True,
            ws_adapter=self.ws_adapter,
        )
        console.print("  [green]OK[/green] CcxtKrakenAdapter built with WSFeedAdapter")

        self.pipeline = ExecutionPipeline(
            exchange=self.exchange,
            venue_adapter=self.kraken_adapter,
            config=pipeline_config,
            position_sizer=PositionSizer(sizing_config),
        )
        console.print("  [green]OK[/green] ULTIMATE pipeline ready")

    async def _init_ws_feed(self):
        from exchanges.centralized.kraken_ws import KrakenWSClient

        console.print("  [dim]Starting Kraken WS feed...[/dim]")
        self.ws_client = KrakenWSClient(
            symbols=self.symbols,
            kraken_adapter=self.kraken_adapter,
            ws_adapter=self.ws_adapter,
            reconnect=True,
        )
        self._ws_task = asyncio.create_task(
            self.ws_client.run(),
            name="kraken_ws_feed",
        )
        await asyncio.sleep(1.5)
        console.print(
            "  [bold green]WS FEED LIVE[/bold green] — "
            f"trade + book subscribed for {len(self.symbols)} symbols -> OFI/VPIN tick-frequency"
        )

    async def _trading_loop(self):
        """ULTIMATE trading loop — 5-second cycles.

        Flow per cycle per symbol:
          1. Fetch 1m OHLCV + update regime
          2. Emergency stop check
          3. Regime gate
          4. Toxicity gate
          5. Collect signals from registry-active strategies (bandit-weighted)
          6. Consensus filter
          7. MTFConfluenceFilter (15m/1h/4h agreement)
          8. Correlation cap (max 3 open positions)
          9. AdaptiveRiskManager check_trade_risk
         10. RLExecutionAgent order-size gate
         11. Execute
         12. Record P&L to StrategyRanker + BanditAllocator
         13. Render StrategyDashboard every N cycles
        """
        import pandas as pd
        from core.types import MarketRegime, SignalAction

        console.print("\n[bold magenta]ULTIMATE TRADING STARTED![/bold magenta] Press Ctrl+C to stop.\n")

        cycle = 0
        _last_daily_reset = datetime.now(timezone.utc).date()

        while self.running:
            cycle += 1

            # ── Daily loss reset ─────────────────────────────────────────
            today = datetime.now(timezone.utc).date()
            if today != _last_daily_reset:
                self._daily_start_capital = self.capital
                _last_daily_reset = today

            try:
                # ── Get bandit-weighted active strategy list ──────────────
                active_names = self.registry.names(active_only=True)
                bandit_weights = (
                    self.bandit.weights(active_names)
                    if (self.bandit is not None and active_names)
                    else {n: 1.0 / max(1, len(active_names)) for n in active_names}
                )
                # Resolve to actual strategy objects in bandit-weight order
                active_strategies = [
                    s for s in self.strategies
                    if getattr(s, "name", s.__class__.__name__) in active_names
                ]

                for symbol in self.symbols:
                    base_sym = symbol.split("/")[0].upper()

                    # ── 1. Fetch 1m OHLCV ────────────────────────────────
                    ohlcv_raw = await self.exchange.fetch_ohlcv(
                        symbol, timeframe="1m", limit=150,
                    )
                    if not ohlcv_raw or len(ohlcv_raw) < 80:
                        continue

                    df = pd.DataFrame(
                        ohlcv_raw,
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    await self.data_store.ohlcv.save_candles(symbol, "1m", ohlcv_raw)
                    current_price = float(df["close"].iloc[-1])

                    # Feed regime
                    if self.regime_integration is not None:
                        try:
                            self.regime_integration.update(base_sym, df)
                        except Exception as exc:
                            logger.debug("RegimeIntegration.update error %s: %s", base_sym, exc)

                    live_regime = MarketRegime.UNKNOWN
                    if self.regime_integration is not None:
                        try:
                            detected = self.regime_integration.get_regime(base_sym)
                            live_regime = MarketRegime[detected.value.upper()]
                        except Exception:
                            pass

                    # Feed AdaptiveRiskManager (BTC as market proxy)
                    if self.adaptive_risk is not None and base_sym == "BTC":
                        rsi_val = 50.0
                        try:
                            delta = df["close"].diff()
                            gain = delta.where(delta > 0, 0).ewm(span=14).mean()
                            loss = (-delta).where(delta < 0, 0).ewm(span=14).mean()
                            rs = gain / loss.replace(0, float('inf'))
                            rsi_val = float((100 - 100 / (1 + rs)).iloc[-1])
                        except Exception:
                            pass
                        self.adaptive_risk.update_market_conditions({
                            "price": current_price,
                            "volume": float(df["volume"].iloc[-1]),
                            "indicators": {
                                "rsi": rsi_val,
                                "macd_signal": 0.0,
                                "bollinger_width": 0.03,
                            },
                            "sentiment": {},
                        })

                    # ── 2. Emergency stop ─────────────────────────────────
                    if self.adaptive_risk is not None and not self._emergency_stopped:
                        current_drawdown = (
                            (self._peak_capital - self.capital) / self._peak_capital
                            if self._peak_capital > 0 else 0.0
                        )
                        daily_loss = (
                            (self._daily_start_capital - self.capital) / self._daily_start_capital
                            if self._daily_start_capital > 0 else 0.0
                        )
                        stop, reason = self.adaptive_risk.emergency_stop_check(
                            current_drawdown, daily_loss
                        )
                        if stop:
                            console.print(
                                f"\n[bold red]EMERGENCY STOP: {reason}[/bold red]\n"
                                "[red]All new entries blocked. Closing positions...[/red]"
                            )
                            logger.critical("EMERGENCY STOP triggered: %s", reason)
                            self._emergency_stopped = True
                            await self._close_all_positions("emergency_stop")
                            self.running = False
                            return

                    if self._emergency_stopped:
                        continue

                    # ── 3. Regime gate ────────────────────────────────────
                    if self.regime_integration is not None:
                        if not self.regime_integration.should_quote(base_sym):
                            logger.debug(
                                "RegimeGate: halting %s (%s)",
                                base_sym,
                                self.regime_integration.get_regime(base_sym).value,
                            )
                            continue

                    # Live microstructure signal
                    live_sig = (
                        self.signal_bus.get(base_sym)
                        if self.signal_bus is not None
                        else None
                    )
                    if live_sig is not None and live_sig.vpin_alert:
                        if self.spread_schedule is not None:
                            self.spread_schedule.set_override(multiplier=1.8, duration_s=30.0)

                    # ── 4. Toxicity gate ──────────────────────────────────
                    if live_sig is not None and live_sig.composite_toxicity > 0.80:
                        logger.debug(
                            "ToxicityGate: skipping %s (toxicity=%.3f)",
                            base_sym, live_sig.composite_toxicity,
                        )
                        continue

                    # Tick bars-since-trade counter
                    self._bars_since_trade[base_sym] = (
                        self._bars_since_trade.get(base_sym, 0) + 1
                    )

                    # ── 5. Collect signals (registry-filtered, bandit-aware) ─
                    raw_signals = []
                    for strategy in active_strategies:
                        strat_name = getattr(strategy, "name", strategy.__class__.__name__)
                        try:
                            sig = await strategy.generate_signal(
                                symbol=symbol,
                                ohlcv=df,
                                regime=live_regime,
                            )
                            if sig is not None:
                                self.signals_generated += 1
                                # Scale confidence by bandit weight so
                                # high-performing strategies get a boost
                                bw = bandit_weights.get(strat_name, 1.0 / max(1, len(active_names)))
                                boosted_conf = min(1.0, float(sig.confidence) * (0.5 + 0.5 * bw * len(active_names)))
                                raw_signals.append({
                                    "symbol": sig.symbol,
                                    "action": sig.action.value,
                                    "confidence": boosted_conf,
                                    "strength": getattr(sig, "strength", 0.5),
                                    "stop_loss": sig.stop_loss,
                                    "take_profit": sig.take_profit,
                                    "entry_price": getattr(sig, "entry_price", None),
                                    "strategy_name": sig.strategy_name,
                                    "regime": live_regime,
                                    "timestamp": sig.timestamp,
                                    "_signal_obj": sig,
                                })
                        except Exception as e:
                            logger.debug("Strategy %s error: %s", strategy.name, e)

                    if not raw_signals:
                        continue

                    # ── 6. Consensus filter ───────────────────────────────
                    approved = (
                        self.consensus.filter_signals(raw_signals)
                        if self.consensus is not None
                        else raw_signals
                    )
                    if not approved:
                        continue

                    best = max(approved, key=lambda s: float(s.get("confidence", 0.0)))
                    signal = best.get("_signal_obj")
                    if signal is None:
                        continue

                    try:
                        object.__setattr__(signal, "confidence", float(best["confidence"]))
                    except (TypeError, AttributeError):
                        pass

                    action_str = str(best.get("action", "")).upper()

                    # ── SELL path ─────────────────────────────────────────
                    if action_str == "SELL":
                        if symbol in self.positions:
                            pos = self.positions[symbol]
                            entry_time = pos.get("entry_time")
                            if entry_time:
                                hold_time = (datetime.now(timezone.utc) - entry_time).total_seconds()
                                if hold_time < 30:
                                    continue
                            if signal.confidence < 0.65:
                                continue
                            logger.info(
                                "CONSENSUS SELL: %s @ %.2f (conf=%.2f, agree=%s) -> CLOSING",
                                symbol, current_price, signal.confidence,
                                best.get("consensus_agreement", "n/a"),
                            )
                            await self._close_position(symbol, current_price, "consensus_signal")
                        continue

                    # ── BUY path ──────────────────────────────────────────
                    if symbol in self.positions:
                        continue

                    # ── 7. MTF Confluence filter ──────────────────────────
                    if self.mtf_filter is not None:
                        market_data_mtf = {
                            "15m": {"close": df["close"].tolist()},
                        }
                        for tf, limit in (("1h", 100), ("4h", 60)):
                            try:
                                tf_raw = await self.exchange.fetch_ohlcv(
                                    symbol, timeframe=tf, limit=limit
                                )
                                if tf_raw and len(tf_raw) >= 51:
                                    tf_df = pd.DataFrame(
                                        tf_raw,
                                        columns=["timestamp", "open", "high",
                                                 "low", "close", "volume"],
                                    )
                                    market_data_mtf[tf] = {"close": tf_df["close"].tolist()}
                            except Exception as exc:
                                logger.debug("MTF fetch %s %s: %s", symbol, tf, exc)

                        mtf_approved, mtf_score, mtf_reason = self.mtf_filter.check(
                            symbol=symbol,
                            signal_direction="buy",
                            market_data=market_data_mtf,
                        )
                        self._last_mtf_score[symbol] = mtf_score

                        if not mtf_approved:
                            logger.debug(
                                "MTFGate: blocked %s — %s (score=%.2f)",
                                symbol, mtf_reason, mtf_score,
                            )
                            continue

                    # ── 8. Correlation cap ────────────────────────────────
                    if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
                        continue

                    # ── 9. AdaptiveRiskManager check ──────────────────────
                    arm_approved = True
                    arm_reason = "OK"
                    if self.adaptive_risk is not None:
                        portfolio_view = {}
                        for sym, pos in self.positions.items():
                            w = pos.get("cost_basis", 0) / max(1.0, self.capital)
                            portfolio_view[sym] = {
                                "current_value": pos.get("cost_basis", 0),
                                "exposure": pos.get("cost_basis", 0),
                                "weight": w,
                                "current_price": pos.get("current_price", 0),
                                "sector": "crypto",
                                "asset_class": "digital",
                            }
                        self.adaptive_risk.update_portfolio(portfolio_view)
                        self.adaptive_risk.current_capital = self.capital

                        stop_price = signal.stop_loss or (current_price * 0.97)
                        risk_per_unit = abs(current_price - stop_price)
                        est_qty = (self.capital * 0.03) / risk_per_unit if risk_per_unit > 0 else 0.001

                        arm_approved, arm_reason, _ = self.adaptive_risk.check_trade_risk(
                            symbol=symbol,
                            position_size=est_qty,
                            entry_price=current_price,
                            stop_price=stop_price,
                            portfolio_value=self.capital,
                        )

                    if not arm_approved:
                        logger.debug("AdaptiveRisk BLOCKED %s: %s", symbol, arm_reason)
                        continue

                    # ── 10. RL execution agent ────────────────────────────
                    rl_size_factor = 1.0
                    if self.rl_agent is not None:
                        from strategies.reinforcement_stub import (
                            RLState, BUY_SMALL, BUY_LARGE, HOLD,
                        )

                        try:
                            tr = pd.concat([
                                df["high"] - df["low"],
                                (df["high"] - df["close"].shift(1)).abs(),
                                (df["low"] - df["close"].shift(1)).abs(),
                            ], axis=1).max(axis=1)
                            atr_val = float(tr.ewm(span=14).mean().iloc[-1])
                            volatility_1h = min(1.0, atr_val / max(1e-9, current_price))
                        except Exception:
                            volatility_1h = 0.3

                        spread_bps = 5.0
                        if self.spread_schedule is not None:
                            try:
                                spread_bps = self.spread_schedule.current_spread_bps()
                            except Exception:
                                pass

                        ob_imbalance = 0.0
                        if live_sig is not None:
                            ob_imbalance = max(-1.0, min(1.0, live_sig.ofi_zscore / 3.0))

                        now_utc = datetime.now(timezone.utc)
                        frac = (now_utc.hour * 3600 + now_utc.minute * 60 + now_utc.second) / 86400.0
                        tod_sin = math.sin(2 * math.pi * frac)
                        tod_cos = math.cos(2 * math.pi * frac)

                        rl_state = RLState(
                            position_usd=0.0,
                            unrealised_pnl=0.0,
                            volatility_1h=volatility_1h,
                            spread_bps=spread_bps,
                            ob_imbalance=ob_imbalance,
                            time_of_day_sin=tod_sin,
                            time_of_day_cos=tod_cos,
                            slippage_budget_remaining=max(
                                0.0,
                                self.rl_agent._slippage_budget - self.rl_agent._episode_slippage,
                            ),
                            bars_since_last_trade=self._bars_since_trade.get(base_sym, 0),
                        )

                        rl_decision = self.rl_agent.decide(rl_state)
                        self._last_rl_action[symbol] = (
                            f"{rl_decision.action_name}(sf={rl_decision.size_factor:.2f})"
                        )

                        if rl_decision.action == HOLD:
                            continue

                        if rl_decision.action in (BUY_SMALL, BUY_LARGE):
                            rl_size_factor = rl_decision.size_factor
                        else:
                            continue

                    # ── 11. Execute ───────────────────────────────────────
                    effective_capital = self.capital * rl_size_factor

                    logger.info(
                        "CONSENSUS BUY: %s @ %.2f (conf=%.2f, agree=%s, n=%s) "
                        "[%s] risk=%s mtf=%.2f rl=%s ofi=%.3f vpin=%.3f",
                        symbol, current_price,
                        signal.confidence,
                        best.get("consensus_agreement", "n/a"),
                        best.get("consensus_signals_count", 1),
                        best.get("strategy_name", "?"),
                        self.adaptive_risk.risk_level.value if self.adaptive_risk else "n/a",
                        self._last_mtf_score.get(symbol, 0.0),
                        self._last_rl_action.get(symbol, "n/a"),
                        live_sig.ofi_zscore if live_sig else 0.0,
                        live_sig.vpin if live_sig else 0.0,
                    )

                    result = await self.pipeline.execute_signal(
                        signal=signal,
                        capital=effective_capital,
                    )

                    if result.success:
                        self.trades_executed += 1
                        self._record_trade(result)
                        self._peak_capital = max(self._peak_capital, self.capital)
                        self._bars_since_trade[base_sym] = 0

                        if self.rl_agent is not None and result.fee and result.cost:
                            estimated_slippage_bps = (result.fee / result.cost) * 10000
                            self.rl_agent.record_slippage(estimated_slippage_bps)

                        if self.ws_adapter is not None:
                            try:
                                self.ws_adapter.complete_journey(base_sym)
                            except Exception:
                                pass

                await self._check_exits()
                await self._update_unrealized_pnl()
                self._peak_capital = max(self._peak_capital, self.capital)

                if cycle % 60 == 0:
                    self._print_status(cycle)

                # ── 13. StrategyDashboard render ──────────────────────────
                if (
                    self.dashboard is not None
                    and cycle % DASHBOARD_EVERY_N_CYCLES == 0
                ):
                    self.dashboard.render(
                        cycle=cycle,
                        capital=self.capital,
                        extra={
                            "bandit_top": self.bandit.best_strategy(active_names) if self.bandit else "n/a",
                        },
                    )

                await asyncio.sleep(5)

            except Exception as e:
                logger.error("Trading cycle error: %s", e)
                await asyncio.sleep(3)

    async def _check_exits(self):
        for symbol in list(self.positions.keys()):
            pos = self.positions.get(symbol)
            if not pos or pos["quantity"] <= 0:
                continue
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                current_price = ticker.get("last", 0)
                if not current_price:
                    continue
                pos["current_price"] = current_price
                entry = pos["avg_price"]
                profit_pct = (current_price - entry) / entry

                if current_price > pos.get("high_water", entry):
                    pos["high_water"] = current_price

                if pos.get("stop_loss") and current_price <= pos["stop_loss"]:
                    logger.info("STOP LOSS: %s @ %.2f", symbol, current_price)
                    await self._close_position(symbol, current_price, "stop_loss")
                elif pos.get("take_profit") and current_price >= pos["take_profit"]:
                    logger.info("TAKE PROFIT: %s @ %.2f", symbol, current_price)
                    await self._close_position(symbol, current_price, "take_profit")
                elif profit_pct > 0.01:
                    high_water = pos.get("high_water", current_price)
                    trail_stop = high_water * 0.995
                    if trail_stop > pos.get("stop_loss", 0):
                        pos["stop_loss"] = trail_stop
                    if current_price < trail_stop:
                        logger.info("TRAILING STOP: %s @ %.2f", symbol, current_price)
                        await self._close_position(symbol, current_price, "trailing_stop")
            except Exception as e:
                logger.debug("Exit check error %s: %s", symbol, e)

    async def _close_position(self, symbol: str, price: float, reason: str):
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        qty = pos["quantity"]
        sell_value = qty * price
        cost_basis = pos["cost_basis"]
        pnl = sell_value - cost_basis

        self.realized_pnl += pnl
        self.total_pnl = self.realized_pnl

        if pnl >= 0:
            self.winning_trades += 1
            pnl_color = "green"
        else:
            self.losing_trades += 1
            pnl_color = "red"

        if pnl > self.best_trade:
            self.best_trade = pnl
        if pnl < self.worst_trade:
            self.worst_trade = pnl

        self.capital += sell_value
        self.max_capital = max(self.max_capital, self.capital)
        self.min_capital = min(self.min_capital, self.capital)

        logger.info("CLOSED %s (%s): qty=%.6f entry=%.2f exit=%.2f P&L=$%.2f",
                    symbol, reason, qty, pos["avg_price"], price, pnl)
        console.print(f"  [{pnl_color}]CLOSED {symbol}: P&L ${pnl:+.2f} ({reason})[/{pnl_color}]")

        # ── 12. Feed ranker + bandit with trade outcome ───────────────────
        strategy_name = pos.get("strategy", "unknown")
        regime_str = getattr(pos.get("regime", None), "value", "unknown")
        if self.ranker is not None:
            self.ranker.record_trade(strategy_name, pnl, regime=regime_str)
        if self.bandit is not None:
            self.bandit.update(strategy_name, pnl)

        # Reset RL bars counter on close
        base_sym = symbol.split("/")[0].upper()
        self._bars_since_trade[base_sym] = 0

        del self.positions[symbol]

    async def _close_all_positions(self, reason: str):
        for symbol in list(self.positions.keys()):
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                price = ticker.get("last", self.positions[symbol]["avg_price"])
                await self._close_position(symbol, price, reason)
            except Exception as e:
                logger.error("Failed to close %s: %s", symbol, e)

    async def _update_unrealized_pnl(self):
        unrealized = 0.0
        for symbol, pos in self.positions.items():
            if pos.get("current_price"):
                unrealized += (pos["current_price"] - pos["avg_price"]) * pos["quantity"]
        self.total_pnl = self.realized_pnl + unrealized

    def _record_trade(self, result):
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

        if result.signal.action.value.upper() == "BUY":
            if symbol not in self.positions:
                self.positions[symbol] = {
                    "quantity": 0,
                    "avg_price": 0,
                    "cost_basis": 0,
                    "entry_time": result.timestamp,
                    "high_water": result.filled_price,
                    "strategy": result.signal.strategy_name,   # NEW: track originating strategy
                    "regime": None,                             # filled below if available
                }
            pos = self.positions[symbol]
            total_qty = pos["quantity"] + result.filled_quantity
            pos["cost_basis"] += result.cost
            pos["avg_price"] = pos["cost_basis"] / total_qty if total_qty > 0 else 0
            pos["quantity"] = total_qty
            pos["stop_loss"] = result.signal.stop_loss
            pos["take_profit"] = result.signal.take_profit
            pos["current_price"] = result.filled_price
            pos["high_water"] = max(pos.get("high_water", 0), result.filled_price)
            self.capital -= result.cost + result.fee
            console.print(
                f"  [cyan]OPENED {symbol}: {result.filled_quantity:.6f} @ "
                f"${result.filled_price:.2f} [{result.signal.strategy_name}][/cyan]"
            )

    def _print_status(self, cycle: int):
        elapsed = datetime.now(timezone.utc) - self.start_time
        hours = elapsed.total_seconds() / 3600
        pnl = self.total_pnl
        pnl_pct = (pnl / self.initial_capital) * 100
        pnl_color = "green" if pnl >= 0 else "red"
        total_closed = self.winning_trades + self.losing_trades
        win_rate = (self.winning_trades / max(1, total_closed)) * 100

        console.print(
            f"\n[bold]Cycle {cycle}[/bold] | "
            f"Capital: [bold]${self.capital:,.2f}[/bold] | "
            f"P&L: [{pnl_color}]${pnl:+,.2f} ({pnl_pct:+.2f}%)[/{pnl_color}] | "
            f"Signals: {self.signals_generated} | "
            f"Trades: {self.trades_executed} | "
            f"Win: {win_rate:.0f}% | "
            f"Positions: {len(self.positions)}/{MAX_CONCURRENT_POSITIONS} | "
            f"Runtime: {hours:.1f}h"
        )

        if self._ws_task is not None:
            ws_status = "running" if not self._ws_task.done() else "[red]STOPPED[/red]"
            console.print(f"  [dim]WS feed:[/dim] {ws_status}")

        # Adaptive risk status
        if self.adaptive_risk is not None:
            try:
                risk_color = {
                    "ultra_conservative": "blue",
                    "conservative": "cyan",
                    "moderate": "green",
                    "aggressive": "yellow",
                    "high_risk": "red",
                }.get(self.adaptive_risk.risk_level.value, "white")
                current_dd = (
                    (self._peak_capital - self.capital) / self._peak_capital * 100
                    if self._peak_capital > 0 else 0.0
                )
                daily_loss = (
                    (self._daily_start_capital - self.capital) / self._daily_start_capital * 100
                    if self._daily_start_capital > 0 else 0.0
                )
                console.print(
                    f"  [dim]AdaptiveRisk:[/dim] "
                    f"level=[{risk_color}]{self.adaptive_risk.risk_level.value}[/{risk_color}] | "
                    f"drawdown={current_dd:.1f}% (limit=15%) | "
                    f"daily_loss={daily_loss:.1f}% (limit=2.5%)"
                )
            except Exception:
                pass

        # MTF confluence scores
        if self.mtf_filter is not None and self._last_mtf_score:
            mtf_parts = []
            for sym, score in self._last_mtf_score.items():
                color = "green" if score >= 0.6 else "yellow" if score >= 0.33 else "red"
                mtf_parts.append(f"{sym.split('/')[0]}=[{color}]{score:.2f}[/{color}]")
            console.print(f"  [dim]MTF scores:[/dim] {' '.join(mtf_parts)}")

        # RL agent status
        if self.rl_agent is not None:
            rl_model_str = "[green]MODEL[/green]" if self.rl_agent.is_model_loaded else "[yellow]RULES[/yellow]"
            rl_slippage = self.rl_agent._episode_slippage
            rl_budget = self.rl_agent._slippage_budget
            console.print(
                f"  [dim]RLAgent:[/dim] {rl_model_str} | "
                f"slippage={rl_slippage:.1f}/{rl_budget:.0f}bps"
            )
            if self._last_rl_action:
                rl_action_str = " ".join(
                    f"{s.split('/')[0]}={a}" for s, a in list(self._last_rl_action.items())[:5]
                )
                console.print(f"  [dim]RL actions:[/dim] {rl_action_str}")

        # Consensus filter stats
        if self.consensus is not None:
            try:
                cs = self.consensus.get_consensus_stats()
                filter_color = "yellow" if cs["filter_rate"] > 0.8 else "green"
                console.print(
                    f"  [dim]Consensus:[/dim] "
                    f"in={cs['signals_in']} out={cs['signals_out']} "
                    f"filter=[{filter_color}]{cs['filter_rate']:.0%}[/{filter_color}] "
                    f"reached={cs['consensus_reached']} missed={cs['consensus_missed']}"
                )
            except Exception:
                pass

        # Microstructure snapshot
        if self.signal_bus is not None:
            try:
                health = self.signal_bus.health()
                for sym, h in health.items():
                    toxicity_color = "red" if h.get("vpin_alert") else "green"
                    console.print(
                        f"  [dim]{sym}[/dim] "
                        f"ofi_z=[cyan]{h.get('ofi_zscore', 0):+.3f}[/cyan] "
                        f"vpin=[{toxicity_color}]{h.get('vpin', 0):.3f}[/{toxicity_color}] "
                        f"drift=[yellow]{h.get('drift_signal', 'n/a')}[/yellow] "
                        f"toxicity={h.get('composite_toxicity', 0):.3f}"
                    )
            except Exception:
                pass

        # Regime breakdown
        if self.regime_integration is not None:
            try:
                regimes = self.regime_integration.current_regimes()
                regime_str = " | ".join(f"{s}={r}" for s, r in regimes.items())
                console.print(f"  [dim]Regimes:[/dim] {regime_str}")
                stats = self.regime_integration.session_stats()
                console.print(
                    f"  [dim]Regime session:[/dim] "
                    f"quoting={stats.get('time_quoting_pct', 0):.0%} "
                    f"halted={stats.get('time_halted_pct', 0):.0%}"
                )
            except Exception:
                pass

        # Latency telemetry
        if self.ws_adapter is not None:
            try:
                lat = self.ws_adapter.latency_stats()
                t2o = lat.get("TICK_TO_ORDER", {})
                p99 = t2o.get("p99_us", 0.0)
                lat_color = "red" if p99 > 5000 else "green"
                console.print(
                    f"  [dim]Latency:[/dim] "
                    f"t2o_p99=[{lat_color}]{p99:.0f}µs[/{lat_color}] "
                    f"completed={lat.get('completed_journeys', 0)}"
                )
            except Exception:
                pass

        # ── NEW: Bandit top strategy ──────────────────────────────────────
        if self.bandit is not None and self.registry is not None:
            active_names = self.registry.names(active_only=True)
            if active_names:
                top = self.bandit.best_strategy(active_names)
                bw = self.bandit.weights(active_names)
                top_weight = bw.get(top, 0.0) if top else 0.0
                console.print(
                    f"  [dim]Bandit:[/dim] top_arm=[bold cyan]{top}[/bold cyan] "
                    f"(w={top_weight:.1%})"
                )

        # Open positions
        if self.positions:
            for sym, pos in self.positions.items():
                unrealized = (
                    (pos.get("current_price", pos["avg_price"]) - pos["avg_price"])
                    * pos["quantity"]
                )
                u_color = "green" if unrealized >= 0 else "red"
                console.print(
                    f"    {sym}: {pos['quantity']:.6f} @ ${pos['avg_price']:.2f} "
                    f"[{u_color}](${unrealized:+.2f})[/{u_color}] "
                    f"[dim]{pos.get('strategy', '?')}[/dim]"
                )

    def _print_final_summary(self):
        pnl = self.total_pnl
        pnl_pct = (pnl / self.initial_capital) * 100
        pnl_color = "green" if pnl >= 0 else "red"
        total_trades = self.winning_trades + self.losing_trades
        win_rate = (self.winning_trades / max(1, total_trades)) * 100
        elapsed = datetime.now(timezone.utc) - self.start_time if self.start_time else None
        hours = elapsed.total_seconds() / 3600 if elapsed else 0
        hourly_return = (pnl / max(0.01, hours)) if hours > 0 else 0
        projected_daily = hourly_return * 24
        projected_monthly = projected_daily * 30

        consensus_summary = ""
        if self.consensus is not None:
            try:
                cs = self.consensus.get_consensus_stats()
                consensus_summary = (
                    f"\nConsensus: {cs['signals_in']} in → {cs['signals_out']} out "
                    f"({cs['filter_rate']:.0%} filtered)"
                )
            except Exception:
                pass

        mtf_summary = ""
        if self.mtf_filter is not None and self._last_mtf_score:
            avg_score = sum(self._last_mtf_score.values()) / len(self._last_mtf_score)
            mtf_summary = f"\nMTF avg confluence score: {avg_score:.2f}"

        rl_summary = ""
        if self.rl_agent is not None:
            rl_summary = (
                f"\nRL Agent: {'MODEL' if self.rl_agent.is_model_loaded else 'RULES'} | "
                f"total slippage={self.rl_agent._episode_slippage:.1f}bps"
            )

        # ── NEW: Ranker final leaderboard ─────────────────────────────────
        ranker_summary = ""
        if self.ranker is not None:
            snap = self.ranker.snapshot()[:5]
            if snap:
                lines = ["\nTop Strategies (final):"] + [
                    f"  #{i+1} {r['name']}: score={r['score']:+.4f} "
                    f"sharpe={r['sharpe']:+.3f} win={r['win_rate']:.1%} "
                    f"pnl=${r['total_pnl']:+.2f}"
                    for i, r in enumerate(snap)
                ]
                ranker_summary = "\n".join(lines)

        emergency_str = "\n[red]EMERGENCY STOP triggered[/red]" if self._emergency_stopped else ""
        arm_level = self.adaptive_risk.risk_level.value if self.adaptive_risk else "n/a"

        # Final dashboard render
        if self.dashboard is not None:
            active_names = self.registry.names(active_only=True) if self.registry else []
            self.dashboard.render(
                cycle=-1,
                capital=self.capital,
                extra={"note": "FINAL SESSION SUMMARY"},
            )

        console.print("\n")
        console.print(Panel.fit(
            f"[bold magenta]ULTIMATE TRADING SUMMARY[/bold magenta]\n\n"
            f"Initial Capital:   ${self.initial_capital:,.2f}\n"
            f"Final Capital:     ${self.capital:,.2f}\n"
            f"Total P&L:         [{pnl_color}]${pnl:+,.2f} ({pnl_pct:+.2f}%)[/{pnl_color}]\n"
            f"Max Capital:       ${self.max_capital:,.2f}\n"
            f"Min Capital:       ${self.min_capital:,.2f}\n\n"
            f"Total Signals:     {self.signals_generated}\n"
            f"Total Trades:      {self.trades_executed}\n"
            f"Winning:           {self.winning_trades}\n"
            f"Losing:            {self.losing_trades}\n"
            f"Win Rate:          {win_rate:.1f}%\n"
            f"Best Trade:        ${self.best_trade:+.2f}\n"
            f"Worst Trade:       ${self.worst_trade:+.2f}\n"
            f"Adaptive Risk:     {arm_level}\n"
            f"{consensus_summary}{mtf_summary}{rl_summary}{ranker_summary}{emergency_str}\n\n"
            f"Runtime:           {hours:.2f} hours\n"
            f"Hourly Return:     ${hourly_return:+.2f}/hr\n"
            f"Projected Daily:   ${projected_daily:+.2f}\n"
            f"Projected Monthly: ${projected_monthly:+.2f}",
            title="ULTIMATE RESULTS",
            border_style="magenta",
        ))


@click.command()
@click.option("--capital", "-c", default=1000.0, help="Starting capital in AUD")
def main(capital: float):
    def signal_handler(sig, frame):
        console.print("\n[yellow]Interrupt received, shutting down...[/yellow]")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    bot = UltimateTradingBot(capital=capital)
    asyncio.run(bot.start())


if __name__ == "__main__":
    main()
