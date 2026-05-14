"""Push 91 — ArgusSystem v8.27.0: full integration.

Wires all layers from Pushes 87–90 into ArgusSystem:

  StrategyRegistry
    └─ Strategies (Momentum, MeanReversion, ML)
         └─ AsyncSignalBus
              └─ ExecutionEngine
                   ├─ OrderManager
                   └─ ExchangeAdapter (Paper or Binance)

  RiskManager         ──► gates every order
  MarginWatcher       ──► async margin poll loop
  BanditRouter  [P87] ──► regime-aware capital allocation
  TradeLedger   [P88] ──► realised P&L persistence
  FillTracker   [P88] ──► slippage tracking
  LedgerFillObserver [P89] ──► bridges fills → Ledger + Bandit
  LiveRegimeDetector [P90] ──► auto-updates market_regime on every tick
  PrometheusRegistry  ──► updated on every tick
  FastAPI app         ──► /health /status /metrics /positions /orders

Usage:
    system = ArgusSystem.from_config(config_dict)
    await system.start()
    for tick in ticks:
        await system.tick(tick.symbol, tick.price, high=tick.high, low=tick.low)
    await system.stop()

Version: v8.27.0
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.strategy.strategy_registry import StrategyRegistry, get_registry
from core.strategy.signal_bus import AsyncSignalBus
from core.strategy.base_strategy import StrategyConfig
from core.execution.order_manager import OrderManager
from core.execution.exchange_adapter import PaperAdapter, AbstractExchangeAdapter
from core.execution.execution_engine import ExecutionEngine
from core.risk.risk_manager import RiskManager, RiskConfig
from core.risk.position_sizer import PositionSizer, SizerConfig, SizingMethod
from core.risk.margin_watcher import MarginWatcher, MarginConfig
from core.risk.risk_event import RiskEventBus
from core.api.prometheus import PrometheusRegistry
from core.api.app import AppContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SystemConfig
# ---------------------------------------------------------------------------

@dataclass
class SystemConfig:
    # ── Strategies ──────────────────────────────────────────────────────────
    strategies:              List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "momentum",       "strategy_id": "mom_BTCUSDT", "symbol": "BTCUSDT"},
        {"name": "mean_reversion", "strategy_id": "mr_BTCUSDT",  "symbol": "BTCUSDT"},
    ])

    # ── Exchange ─────────────────────────────────────────────────────────────
    paper_mode:              bool  = True
    initial_balance:         float = 100_000.0
    binance_api_key:         str   = ""
    binance_api_secret:      str   = ""
    binance_testnet:         bool  = True

    # ── Risk ─────────────────────────────────────────────────────────────────
    risk:                    Dict[str, Any] = field(default_factory=dict)

    # ── Execution ────────────────────────────────────────────────────────────
    max_open_orders:         int   = 10
    max_position_usd:        float = 50_000.0
    fee_rate:                float = 0.001
    signal_cooldown:         float = 5.0
    initial_equity:          float = 10_000.0

    # ── API ──────────────────────────────────────────────────────────────────
    api_host:                str   = "0.0.0.0"
    api_port:                int   = 8080
    log_level:               str   = "INFO"

    # ── Push 87: BanditRouter ────────────────────────────────────────────────
    bandit_enabled:          bool  = True
    bandit_max_concentration: float = 0.40
    bandit_sharpe_kill:      float = -0.5
    market_regime:           str   = "UNKNOWN"
    strategy_categories:     Dict[str, str] = field(default_factory=dict)

    # ── Push 88: TradeLedger ─────────────────────────────────────────────────
    ledger_db_path:          str   = "data/ledger.db"

    # ── Push 90: RegimeDetector ──────────────────────────────────────────────
    regime_warmup_ticks:     int   = 50
    regime_hysteresis_ticks: int   = 5
    regime_high_vol_thresh:  float = 2.5
    regime_trend_thresh:     float = 0.003


# ---------------------------------------------------------------------------
# ArgusSystem
# ---------------------------------------------------------------------------

class ArgusSystem:
    """Top-level orchestrator for Argus Ultimate v8.27.0.

    All Pushes 87–90 are wired here:
    - BanditRouter allocates capital per-strategy based on regime + performance.
    - LedgerFillObserver feeds confirmed fills → TradeLedger + FillTracker → Bandit.
    - LiveRegimeDetector updates market_regime on every tick automatically.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config   = config or SystemConfig()
        self._built   = False
        self._running = False
        self._start_time: Optional[float] = None

        # Core sub-systems
        self.registry:        StrategyRegistry
        self.bus:             AsyncSignalBus
        self.order_manager:   OrderManager
        self.adapter:         AbstractExchangeAdapter
        self.engine:          ExecutionEngine
        self.risk_manager:    RiskManager
        self.margin_watcher:  MarginWatcher
        self.risk_event_bus:  RiskEventBus
        self.prom:            PrometheusRegistry
        self.app_context:     AppContext
        self.strategies:      List[Any] = []

        # Push 87–90 components (None until _build)
        self.bandit_router    = None
        self.trade_ledger     = None
        self.fill_tracker     = None
        self.fill_observer    = None
        self.regime_detector  = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        if self._built:
            return
        cfg = self.config

        # ── Core infrastructure ──────────────────────────────────────────
        self.registry = get_registry()
        self.bus      = AsyncSignalBus()

        if cfg.paper_mode:
            self.adapter = PaperAdapter(
                initial_balance=cfg.initial_balance,
                fee_rate=cfg.fee_rate,
            )
        else:
            from core.execution.exchange_adapter import BinanceAdapter
            self.adapter = BinanceAdapter(
                api_key=cfg.binance_api_key,
                api_secret=cfg.binance_api_secret,
                testnet=cfg.binance_testnet,
            )

        self.order_manager = OrderManager(
            max_open_orders=cfg.max_open_orders,
            max_position_usd=cfg.max_position_usd,
            fee_rate=cfg.fee_rate,
        )

        risk_cfg = RiskConfig(
            initial_equity=cfg.initial_equity,
            **{k: v for k, v in cfg.risk.items()
               if k in RiskConfig.__dataclass_fields__}
        )
        self.risk_event_bus = RiskEventBus()
        self.risk_manager   = RiskManager(
            config=risk_cfg,
            event_bus=self.risk_event_bus,
        )
        self.margin_watcher = MarginWatcher(
            config=MarginConfig(),
            order_manager=self.order_manager,
            adapter=self.adapter,
            risk_manager=self.risk_manager,
            event_bus=self.risk_event_bus,
        )

        self.engine = ExecutionEngine(
            order_manager=self.order_manager,
            adapter=self.adapter,
            signal_bus=self.bus,
            signal_cooldown_secs=cfg.signal_cooldown,
            initial_equity=cfg.initial_equity,
        )

        # ── Strategies ───────────────────────────────────────────────────
        self.strategies = []
        for s_cfg in cfg.strategies:
            s_cfg_copy  = dict(s_cfg)
            name        = s_cfg_copy.pop("name")
            params      = s_cfg_copy.pop("params", {})
            strat_config = StrategyConfig(
                params=params,
                initial_equity=cfg.initial_equity,
                **s_cfg_copy,
            )
            try:
                strat = self.registry.instantiate(name, strat_config)
                self.strategies.append(strat)
            except Exception as exc:
                logger.warning("[ArgusSystem] strategy '%s' skipped: %s", name, exc)

        # ── Push 87: BanditRouter ────────────────────────────────────────
        if cfg.bandit_enabled:
            try:
                from core.bandit_router import BanditRouter, BanditConfig
                strategy_ids = [
                    s.get("strategy_id", s.get("name", f"s{i}"))
                    for i, s in enumerate(cfg.strategies)
                ]
                bandit_cfg = BanditConfig(
                    strategy_ids=strategy_ids,
                    max_concentration=cfg.bandit_max_concentration,
                    sharpe_kill_threshold=cfg.bandit_sharpe_kill,
                    strategy_categories=cfg.strategy_categories or {},
                )
                self.bandit_router = BanditRouter(config=bandit_cfg)
                logger.info("[ArgusSystem] BanditRouter wired (%d strategies)",
                            len(strategy_ids))
            except ImportError as exc:
                logger.warning("[ArgusSystem] BanditRouter unavailable: %s", exc)

        # ── Push 88: TradeLedger + FillTracker ───────────────────────────
        try:
            from core.trade_ledger import TradeLedger
            from core.fill_tracker import FillTracker
            self.trade_ledger = TradeLedger(db_path=cfg.ledger_db_path)
            self.fill_tracker = FillTracker()
            logger.info("[ArgusSystem] TradeLedger + FillTracker wired")
        except ImportError as exc:
            logger.warning("[ArgusSystem] Ledger components unavailable: %s", exc)

        # ── Push 89: LedgerFillObserver ──────────────────────────────────
        if self.trade_ledger and self.fill_tracker and self.bandit_router:
            try:
                from core.ledger_fill_observer import LedgerFillObserver
                self.fill_observer = LedgerFillObserver(
                    ledger=self.trade_ledger,
                    fill_tracker=self.fill_tracker,
                    bandit_router=self.bandit_router,
                )
                logger.info("[ArgusSystem] LedgerFillObserver wired")
            except ImportError as exc:
                logger.warning("[ArgusSystem] LedgerFillObserver unavailable: %s", exc)

        # ── Push 90: LiveRegimeDetector ──────────────────────────────────
        try:
            from core.regime_detector import LiveRegimeDetector, RegimeDetectorConfig
            det_cfg = RegimeDetectorConfig(
                warmup_ticks=cfg.regime_warmup_ticks,
                hysteresis_ticks=cfg.regime_hysteresis_ticks,
                high_vol_threshold=cfg.regime_high_vol_thresh,
                trend_threshold=cfg.regime_trend_thresh,
            )
            self.regime_detector = LiveRegimeDetector(config=det_cfg)
            self.regime_detector.attach_system_config(self.config)
            logger.info("[ArgusSystem] LiveRegimeDetector wired")
        except ImportError as exc:
            logger.warning("[ArgusSystem] LiveRegimeDetector unavailable: %s", exc)

        # ── Prometheus + App ─────────────────────────────────────────────
        self.prom = PrometheusRegistry()
        self.app_context = AppContext(
            engine=self.engine,
            order_manager=self.order_manager,
            risk_manager=self.risk_manager,
            signal_bus=self.bus,
            adapter=self.adapter,
            registry=self.prom,
        )

        self._built = True
        logger.info("[ArgusSystem] build complete — v8.27.0")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._build()
        await self.engine.start()
        await self.margin_watcher.start()
        for strat in self.strategies:
            strat.start()
        self._running    = True
        self._start_time = time.time()
        logger.info("[ArgusSystem] started")

    async def stop(self) -> None:
        self._running = False
        for strat in self.strategies:
            strat.stop()
        await self.margin_watcher.stop()
        await self.engine.stop()
        logger.info("[ArgusSystem] stopped")

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    async def tick(
        self,
        symbol:    str,
        price:     float,
        volume:    float = 0.0,
        high:      Optional[float] = None,
        low:       Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Feed a price tick to all sub-systems for the given symbol.

        New in v8.27.0:
        - RegimeDetector is updated first (requires high/low for ATR).
        - BanditRouter allocations are consulted before publishing signals.
        - Confirmed fills are routed through LedgerFillObserver.
        """
        if not self._running:
            return

        _high = high  if high  is not None else price * 1.001
        _low  = low   if low   is not None else price * 0.999

        # ── 1. Regime detector ───────────────────────────────────────────
        if self.regime_detector is not None:
            self.regime_detector.update(price=price, high=_high, low=_low)

        # ── 2. Paper adapter price injection ────────────────────────────
        if hasattr(self.adapter, "set_price"):
            self.adapter.set_price(symbol, price)

        # ── 3. Risk unrealised P&L ────────────────────────────────────────
        pos = self.order_manager.get_position(symbol)
        if pos:
            pos.update_unrealised(price)
            self.risk_manager.update_open_notional(symbol, pos.notional)

        # ── 4. Strategy tick + Bandit-gated signal publishing ────────────
        regime = self.config.market_regime

        allocs: Dict[str, float] = {}
        if self.bandit_router is not None:
            try:
                allocs = self.bandit_router.allocations(regime)
            except Exception as exc:
                logger.debug("[ArgusSystem] bandit alloc error: %s", exc)

        for strat in self.strategies:
            if strat.symbol != symbol:
                continue

            signal = strat.tick(price, volume, timestamp)
            if signal is None:
                continue

            sid   = getattr(strat, "strategy_id", getattr(strat, "name", "unknown"))
            alloc = allocs.get(sid, 1.0 / max(len(self.strategies), 1))

            if alloc <= 0:
                logger.debug("[ArgusSystem] strategy %s suspended (alloc=0)", sid)
                continue

            notional = alloc * self.config.initial_equity * signal.strength

            allowed, reason = self.risk_manager.check_order_allowed(
                symbol, notional
            )
            if not allowed:
                logger.debug("[ArgusSystem] risk blocked %s: %s", sid, reason)
                continue

            await self.bus.publish(signal)

        # ── 5. Route confirmed fills through LedgerFillObserver ──────────
        await self._drain_fills(symbol, price)

        # ── 6. Prometheus ────────────────────────────────────────────────
        self.prom.update_from_engine(self.engine.stats)
        self.prom.update_from_risk(self.risk_manager.stats)
        self.prom.update_from_om(self.order_manager.stats)

    # ------------------------------------------------------------------
    # Fill drain  (P89 LedgerFillObserver)
    # ------------------------------------------------------------------

    async def _drain_fills(self, symbol: str, price: float) -> None:
        """Pull any newly confirmed fills from the engine and route them."""
        if self.fill_observer is None:
            return
        try:
            recent_fills = self.engine.pop_confirmed_fills()
        except AttributeError:
            return

        for fill in recent_fills:
            try:
                sid = getattr(fill, "strategy_id",
                              getattr(fill, "strategy", "unknown"))
                self.fill_observer.record_fill_outcome(
                    strategy_id=sid,
                    fill=fill,
                    current_price=price,
                )
            except Exception as exc:
                logger.debug("[ArgusSystem] fill observer error: %s", exc)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        if not self._built:
            return {}

        base = {
            "version":     "8.27.0",
            "codename":    "FullIntegration",
            "running":     self._running,
            "uptime_secs": round(time.time() - self._start_time, 1)
                           if self._start_time else 0,
            "strategies":  [str(s) for s in self.strategies],
            "engine":      self.engine.stats,
            "risk":        self.risk_manager.stats,
            "execution":   self.order_manager.stats,
            "bus":         self.bus.stats,
            "market_regime": self.config.market_regime,
        }

        if self.bandit_router is not None:
            try:
                base["bandit"] = self.bandit_router.summary()
            except Exception:
                pass

        if self.trade_ledger is not None:
            try:
                base["ledger_pnl"] = self.trade_ledger.realised_pnl()
            except Exception:
                pass

        if self.regime_detector is not None:
            snap = self.regime_detector.snapshot()
            base["regime_detector"] = {
                "regime":      snap.regime,
                "vol_ratio":   snap.vol_ratio,
                "trend_score": snap.trend_score,
                "confidence":  snap.confidence,
                "tick_count":  snap.tick_count,
            }

        return base

    # ------------------------------------------------------------------
    # App + Factory
    # ------------------------------------------------------------------

    def get_app(self):
        self._build()
        from core.api.app import create_app
        return create_app(self.app_context)

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "ArgusSystem":
        cfg = SystemConfig(**{
            k: v for k, v in config_dict.items()
            if k in SystemConfig.__dataclass_fields__
        })
        return cls(cfg)

    @classmethod
    def paper(cls, symbol: str = "BTCUSDT", equity: float = 10_000.0) -> "ArgusSystem":
        """Convenience factory: single-symbol paper trading system."""
        cfg = SystemConfig(
            paper_mode=True,
            initial_equity=equity,
            initial_balance=equity * 10,
            strategies=[
                {"name": "momentum",       "strategy_id": f"mom_{symbol}",  "symbol": symbol},
                {"name": "mean_reversion", "strategy_id": f"mr_{symbol}",   "symbol": symbol},
            ],
            strategy_categories={
                f"mom_{symbol}": "momentum",
                f"mr_{symbol}":  "mean_reversion",
            },
        )
        return cls(cfg)
