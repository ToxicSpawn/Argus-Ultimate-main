#!/usr/bin/env python3
"""
core/system_init.py
===================
SystemInitMixin — extracted from unified_trading_system.py.

Contains every _initialize_* method and live-safe helper that previously
lived on UnifiedSystemArchitecture, plus the main ``initialize()`` coroutine
and the full __init__ body refactored as ``_init_instance()`` so subclasses
can call it from their own __init__.

Usage
-----
UnifiedSystemArchitecture (in unified_trading_system.py) already inherits this
mixin; nothing else needs to change.  New code that wants the init logic
without the full monolith can subclass SystemInitMixin directly.

All method bodies are verbatim copies from unified_trading_system.py;
zero functional changes.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.system_state import SystemState
from core.omega_sqlite_store import OmegaSQLiteStore
from core.paper_exchange import _PaperCCXTWrapper

logger = logging.getLogger(__name__)


class SystemInitMixin:
    """
    Mixin that provides all _initialize_* methods and the main initialize()
    coroutine for UnifiedSystemArchitecture.

    Expects ``self.config`` to be a UnifiedConfig instance and ``self.run_id``
    to be set before ``initialize()`` is called.  Both are set by
    ``_init_instance()`` below.
    """

    # ------------------------------------------------------------------
    # Instance bootstrap (replaces the body of __init__)
    # ------------------------------------------------------------------

    def _init_instance(self, config: Any) -> None:
        """Bootstrap all instance attributes.  Call from __init__ of the concrete class."""
        self.config = config

        # Ω audit IDs
        try:
            self.run_id = uuid.uuid4().hex[:8]
        except Exception:
            self.run_id = "unknown"
        self._trace_id = None
        self.node_role = str(getattr(self.config, "node_role", "single-node") or "single-node").strip().lower()
        self.command_bus = None
        self.execution_mesh = None

        # Ω spine persistence
        omega_db = None
        try:
            omega_db = getattr(getattr(self.config, "trade_ledger", None), "db_path", None)
        except Exception:
            omega_db = None
        if not omega_db:
            omega_db = "data/unified_trades.db"
        self.omega_store = OmegaSQLiteStore(str(omega_db))
        self.state = SystemState.INITIALIZING
        self.start_time = datetime.now()

        # Core components
        self.ai_brain = None
        self.execution_engine = None
        self.argus_strategies = None
        self.monitoring = None
        self.hft_engine = None
        self.hft_infrastructure = None
        self.language_orchestrator = None

        # Thread-safe state
        import threading as _threading
        self._state_lock = _threading.Lock()
        self.portfolio_value_aud = config.starting_capital_aud
        self.cash_balance_aud = config.starting_capital_aud
        self.positions: Dict[str, Dict] = {}
        self.trade_history: deque = deque(maxlen=10000)

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl_aud = 0.0
        self.realized_pnl_aud = 0.0
        self.unrealized_pnl_aud = 0.0
        self.total_fees_aud = 0.0
        self.daily_pnl_aud = 0.0
        self.max_drawdown_aud = 0.0
        self.peak_equity_aud = config.starting_capital_aud
        self.mark_price_method = "position.current_price"
        self._ledger_sanity_violations = 0

        # Risk tracking
        self.consecutive_losses = 0
        self.error_count = 0
        self.total_operations = 0

        # Exchange connections
        self.exchanges: Dict[str, Any] = {}
        self.market_data_service = None
        self.live_market_data = None
        self.self_improver = None
        self._self_improver_task = None
        self.continuous_scanner = None
        self._continuous_scanner_task = None
        self.adaptive_risk_controller = None
        self.universe_selector = None
        self.strategy_allocator = None
        self._strategy_state_store = None
        self.strategy_evaluation_engine = None
        self.self_optimizing_meta_engine = None
        self.champion_challenger_engine = None
        self._last_regime_consensus: Dict = {}
        self.target_engine = None
        self._latest_targets: List[Any] = []
        self.liquidity_risk_engine = None
        self._latest_liquidity_state: Dict[str, Any] = {}
        self.market_microstructure_engine = None
        self._latest_microstructure_state: Dict[str, Any] = {}
        self._latest_strategy_weights: Dict[str, float] = {}
        self.system_health_metrics = None
        self.feature_store = None
        self.regime_classifier = None
        self._latest_regime_label = ""

        self.quant_fund_risk_engine = None

        # Partial TP / trailing stop
        self._position_high_water: Dict[str, float] = {}
        self._position_low_water: Dict[str, float] = {}
        self._partial_tp_taken: Dict[str, bool] = {}

        # Salvaged production modules
        self.rate_limiter = None
        self.data_sanitizer = None
        self.position_tracker = None
        self.audit_chain = None
        self.self_healer = None
        self.health_monitor = None
        self.unified_risk_manager = None
        self.component_registry = None
        self.signal_service = None
        self.api_server = None
        self.perf_feeder = None
        self.regime_alerter = None
        self.model_manager = None
        self.checkpoint_manager = None

        # Execution pipeline state
        self._process_lock = None
        self._pending_orders: Dict[str, Dict[str, Any]] = {}
        self._reconcile_every_n_cycles: int = int(getattr(config, "reconcile_every_n_cycles", 10) or 10)
        self._order_timeout_seconds: float = float(getattr(config, "order_timeout_seconds", 60.0) or 60.0)
        self._paper_slippage_bps: float = float(getattr(config, "paper_slippage_bps", 5.0) or 5.0)
        self._total_fee_savings_usd: float = 0.0
        self._limit_order_fill_timeout: float = 10.0
        self._limit_price_offset_bps: float = 2.0
        self._volatility_cache: Dict[str, float] = {}
        self._vwap_threshold_usd: float = 100.0

        # Price history / pyramid / partial exit / OCO tracking
        self._price_history: Dict[str, List[float]] = {}
        self._pyramid_count: Dict[str, int] = {}
        self._partial_exit_done: Dict[str, bool] = {}
        self._oco_orders: Dict[str, Dict[str, Any]] = {}

        # Quantum Monte Carlo risk
        self._equity_history: List[float] = []
        self._after_risk_update_hook = None

        # Emergency shutdown state
        self._last_cycle_total_ms: Optional[float] = None
        self._last_cycle_stage_timing_ms: Dict[str, float] = {}
        self._last_target_pipeline_stage_ms: Dict[str, float] = {}
        self._completed_cycles: int = 0
        self._last_event_loop_delay_ms: float = 0.0
        self._live_safe_locked_symbols: List[str] = []
        self._last_price_by_symbol: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        self._exchange_unreachable: bool = False
        self._last_spread_bps: Optional[float] = None

        # System health metrics
        try:
            from metrics.system_health_metrics import SystemHealthMetricsCollector
            self.system_health_metrics = SystemHealthMetricsCollector(
                enabled=bool(getattr(self.config, "system_health_metrics_enabled", True)),
                snapshot_interval_cycles=int(
                    getattr(self.config, "system_health_metrics_snapshot_interval_cycles", 10) or 10
                ),
            )
        except Exception as _health_e:
            logger.warning("System health metrics module unavailable: %s", _health_e)
            self.system_health_metrics = None

        try:
            from core.version import __version__ as _ver
        except ImportError:
            _ver = "3.0.0"
        logger.info("=" * 70)
        logger.info("UNIFIED TRADING SYSTEM - INITIALIZING (Argus v%s)", _ver)
        logger.info("=" * 70)
        logger.info("Config version: %s", getattr(config, "config_version", 1))
        logger.info(f"Starting Capital: ${config.starting_capital_aud:,.2f} AUD")
        logger.info(f"Primary Exchange: {config.primary_exchange}")
        logger.info(f"Secondary Exchange: {config.secondary_exchange}")
        logger.info("Node role: %s", self.node_role)
        logger.info("Multi-Language Support: 23+ languages enabled")
        self._runtime_module_registry: Dict[str, Dict[str, Any]] = {}
        self._apply_live_safe_scope_controls()

    # ------------------------------------------------------------------
    # Main async initializer
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize all system components (phases 1–6)."""
        try:
            logger.info("Phase 1: Initializing unified architecture...")

            try:
                self.omega_store.init_schema()
                logger.info("✅ Ω spine DB initialized (%s)", getattr(self.omega_store, "db_path", ""))
            except Exception as e:
                logger.warning("Ω spine DB init failed (continuing without): %s", e)

            try:
                from utils.fx_rate import get_aud_usd_rate
                live_rate = get_aud_usd_rate(fallback=float(getattr(self.config, "aud_to_usd", 0.65) or 0.65))
                self.config.aud_to_usd = live_rate
                logger.info("AUD/USD rate set to %.5f (live)", live_rate)
            except Exception as exc:
                logger.warning("Could not fetch live AUD/USD rate, using config default: %s", exc)

            if getattr(self.config, "multi_language_enabled", True):
                await self._initialize_multi_language_system()
            else:
                logger.info("Multi-language system disabled by config")

            await self._initialize_exchanges()
            await self._initialize_quant_fund_upgrades()
            await self._initialize_argus_strategies()
            await self._initialize_ai_brain()

            if getattr(self.config, "use_quantum_monte_carlo_risk", False) and getattr(self, "_after_risk_update_hook", None) is None:
                self._after_risk_update_hook = self._quantum_monte_carlo_risk_hook
                logger.info("Quantum Monte Carlo risk hook registered (VaR/CVaR in cycle)")

            await self._initialize_command_bus()
            await self._initialize_execution_engine()
            await self._initialize_execution_mesh()
            await self._initialize_capital_optimizer()
            await self._initialize_hft_engine()
            await self._initialize_hft_infrastructure()

            try:
                from risk.unified_risk_manager import UnifiedRiskManager
                self.unified_risk_manager = UnifiedRiskManager(
                    initial_capital=float(getattr(self.config, "starting_capital_aud", 1000.0) or 1000.0),
                    max_daily_loss=float(getattr(self.config, "max_daily_loss_pct", 0.02) or 0.02),
                    max_total_exposure=float(getattr(self.config, "max_total_exposure_pct", 0.8) or 0.8),
                    max_leverage=float(getattr(self.config, "max_leverage", 3.0) or 3.0),
                    max_consecutive_losses=int(getattr(self.config, "max_consecutive_losses", 5) or 5),
                )
            except Exception as e:
                logger.warning("UnifiedRiskManager unavailable: %s", e)
                self.unified_risk_manager = None

            try:
                from adaptive.adaptive_risk_controller import AdaptiveRiskController
                self.adaptive_risk_controller = AdaptiveRiskController(config=self.config)
                setattr(self.config, "adaptive_risk_controller", self.adaptive_risk_controller)
            except Exception as e:
                logger.warning("AdaptiveRiskController unavailable: %s", e)
                self.adaptive_risk_controller = None

            try:
                from adaptive.universe_selector import AdaptiveUniverseSelector
                self.universe_selector = AdaptiveUniverseSelector(
                    persist_path=str(getattr(self.config, "adaptive_universe_state_path", "data/adaptive_universe.json") or "data/adaptive_universe.json"),
                    max_active=int(getattr(self.config, "adaptive_universe_max_active", 5) or 5),
                    min_hold_cycles=int(getattr(self.config, "adaptive_universe_min_hold_cycles", 20) or 20),
                )
                self.universe_selector.load()
            except Exception as e:
                logger.warning("AdaptiveUniverseSelector unavailable: %s", e)
                self.universe_selector = None

            await self._initialize_monitoring()
            await self._initialize_production_modules()

            if getattr(self.config, "evolution_load_evolved", False):
                try:
                    from evolution.apply_evolved_strategies import apply_from_file
                    path = str(getattr(self.config, "evolution_params_path", "data/evolved_params.json") or "data/evolved_params.json")
                    n = apply_from_file(self.config, path, key="best_params")
                    if n > 0:
                        logger.info("Loaded %d evolved params from %s into config", n, path)
                except Exception as e:
                    logger.debug("Evolved params load (optional): %s", e)

            await self._initialize_strategy_allocator()
            self._initialize_strategy_evaluation_engine()
            self._initialize_self_optimizing_meta_engine()
            self._initialize_champion_challenger_engine()
            self._log_runtime_module_registry()

            self.state = SystemState.RUNNING
            logger.info("✅ Unified system architecture initialized successfully")
            if self.language_orchestrator:
                logger.info("✅ Multi-language orchestrator: %d languages", len(self.language_orchestrator.languages))
            else:
                logger.info("✅ Multi-language orchestrator: disabled")

        except Exception as e:
            logger.error(f"Failed to initialize unified system: {e}", exc_info=True)
            self.state = SystemState.EMERGENCY_STOP
            raise

    # ------------------------------------------------------------------
    # Live-safe scope controls
    # ------------------------------------------------------------------

    def _is_live_safe_runtime(self) -> bool:
        mode = str(getattr(self.config, "run_mode", "paper") or "paper").strip().lower()
        profile = str(getattr(self.config, "config_profile", "") or "").strip().lower()
        if profile in {"live_safe", "live-safe", "livesafe"}:
            return True
        return mode == "live" and bool(getattr(self.config, "institutional_mode", False))

    def _apply_live_safe_scope_controls(self) -> None:
        if not self._is_live_safe_runtime():
            return
        disabled = []
        for attr in (
            "ai_enabled", "strategy_library_enabled", "quantum_features_enabled",
            "quant_fund_upgrades_enabled", "evolution_continuous_enabled",
            "self_improvement_enabled", "self_optimizing_meta_enabled",
            "champion_challenger_enabled", "multi_language_enabled",
            "continuous_scan_enabled", "hft_enabled",
            "use_advanced_hft_infrastructure", "dynamic_universe_enabled",
            "adaptive_universe_enabled",
        ):
            if bool(getattr(self.config, attr, False)):
                setattr(self.config, attr, False)
                disabled.append(attr)
        if bool(getattr(self.config, "adaptive_universe_modes", [])):
            setattr(self.config, "adaptive_universe_modes", [])
            disabled.append("adaptive_universe_modes")
        if not bool(getattr(self.config, "live_safe_disable_pinnacle_ai_brain", False)):
            setattr(self.config, "live_safe_disable_pinnacle_ai_brain", True)
            disabled.append("live_safe_disable_pinnacle_ai_brain")
        locked_pairs = [
            str(s).strip()
            for s in list(getattr(self.config, "trading_pairs", []) or [])
            if str(s).strip()
        ]
        self._live_safe_locked_symbols = list(dict.fromkeys(locked_pairs))
        if self._live_safe_locked_symbols:
            setattr(self.config, "trading_pairs", list(self._live_safe_locked_symbols))
            logger.info("LIVE_SAFE symbol lock enabled: %s", ", ".join(self._live_safe_locked_symbols))
        if disabled:
            logger.warning("LIVE_SAFE scope control: disabled research/advisory modules: %s", ", ".join(disabled))

    def _enforce_live_safe_symbol_lock(self) -> None:
        if not self._is_live_safe_runtime():
            return
        locked = list(self._live_safe_locked_symbols or [])
        if not locked:
            return
        current = [str(s).strip() for s in list(getattr(self.config, "trading_pairs", []) or []) if str(s).strip()]
        if current != locked:
            setattr(self.config, "trading_pairs", list(locked))
            logger.warning("LIVE_SAFE symbol lock restored trading_pairs to profile set: %s", ", ".join(locked))

    def _filter_live_safe_signals(self, signals: List[Any], *, stage: str) -> List[Any]:
        rows = list(signals or [])
        if not self._is_live_safe_runtime():
            return rows
        locked = {str(s).strip() for s in list(self._live_safe_locked_symbols or []) if str(s).strip()}
        if not locked:
            return rows
        kept: List[Any] = []
        dropped = 0
        for sig in rows:
            sym = str(self._signal_get(sig, "symbol", "") or "").strip()
            if not sym or sym in locked:
                kept.append(sig)
            else:
                dropped += 1
        if dropped > 0:
            logger.warning(
                "LIVE_SAFE symbol lock filtered %s signal(s) at %s; allowed symbols: %s",
                int(dropped), str(stage), ", ".join(sorted(locked)),
            )
        return kept

    # ------------------------------------------------------------------
    # Runtime module registry
    # ------------------------------------------------------------------

    def _build_runtime_module_registry(self) -> Dict[str, Dict[str, Any]]:
        entries: Dict[str, Dict[str, Any]] = {}

        def _put(name: str, *, category: str, enabled: bool, ready: bool, critical: bool = False) -> None:
            if not enabled:
                status = "disabled"
            elif ready:
                status = "active"
            else:
                status = "failed-safe" if critical else "degraded"
            entries[name] = {"category": str(category), "enabled": bool(enabled), "status": str(status)}

        _put("exchange_connectivity", category="critical", enabled=True, ready=bool(self.exchanges), critical=True)
        _put("execution_engine", category="critical", enabled=True, ready=bool(self.execution_engine), critical=True)
        _put("hard_risk_gate", category="critical", enabled=True, ready=bool(getattr(self.execution_engine, "risk_manager", None)), critical=True)
        _put("order_intents_store", category="critical", enabled=True, ready=bool(getattr(self.execution_engine, "state_store", None)), critical=True)
        _put("reconciliation", category="critical", enabled=True, ready=bool(self.execution_engine), critical=True)
        _put("liquidity_risk_engine", category="optional", enabled=bool(getattr(self.config, "liquidity_risk_enabled", True)), ready=bool(self.liquidity_risk_engine))
        _put("market_microstructure_engine", category="optional", enabled=bool(getattr(self.config, "market_microstructure_enabled", True)), ready=bool(self.market_microstructure_engine))
        _put("system_health_metrics", category="optional", enabled=bool(getattr(self.config, "system_health_metrics_enabled", True)), ready=bool(self.system_health_metrics))
        _put("recon_recovery_engine", category="optional", enabled=bool(getattr(self.config, "recon_recovery_enabled", True)), ready=bool(getattr(self.execution_engine, "recon_recovery_engine", None)))
        _put("strategy_evaluation_engine", category="advisory", enabled=bool(getattr(self.config, "strategy_evaluation_enabled", True)), ready=bool(self.strategy_evaluation_engine))
        _put("self_optimizing_meta_engine", category="advisory", enabled=bool(getattr(self.config, "self_optimizing_meta_enabled", True)), ready=bool(self.self_optimizing_meta_engine))
        _put("champion_challenger", category="advisory", enabled=bool(getattr(self.config, "champion_challenger_enabled", True)), ready=bool(self.champion_challenger_engine))
        _put("strategy_library", category="research_only", enabled=bool(getattr(self.config, "strategy_library_enabled", True)), ready=bool(getattr(self.config, "strategy_library_enabled", True)))
        _put("quantum_features", category="research_only", enabled=bool(getattr(self.config, "quantum_features_enabled", True)), ready=bool(getattr(self.config, "quantum_features_enabled", True)))
        _put("quant_fund_upgrades", category="research_only", enabled=bool(getattr(self.config, "quant_fund_upgrades_enabled", True)), ready=bool(getattr(self.config, "quant_fund_upgrades_enabled", True)))
        _put("self_improvement", category="research_only", enabled=bool(getattr(self.config, "self_improvement_enabled", True)), ready=bool(getattr(self.config, "self_improvement_enabled", True)))

        self._runtime_module_registry = entries
        return entries

    def _log_runtime_module_registry(self) -> None:
        profile = str(getattr(self.config, "config_profile", "") or "").strip() or "default"
        source = str(getattr(self.config, "config_source", "") or "").strip()
        registry = self._build_runtime_module_registry()
        logger.info("Runtime profile: %s", profile)
        if source:
            logger.info("Runtime config source: %s", source)
        counts = {"active": 0, "disabled": 0, "degraded": 0, "failed-safe": 0}
        for item in registry.values():
            status = str(item.get("status", "") or "")
            if status in counts:
                counts[status] += 1
        logger.info("Runtime module status summary: active=%d disabled=%d degraded=%d failed-safe=%d",
                    counts["active"], counts["disabled"], counts["degraded"], counts["failed-safe"])
        for name in sorted(registry.keys()):
            entry = registry[name]
            logger.info("module %-28s category=%-13s status=%-10s enabled=%s",
                        name, str(entry.get("category", "")), str(entry.get("status", "")),
                        str(bool(entry.get("enabled", False))).lower())

    # ------------------------------------------------------------------
    # Phase initializers
    # ------------------------------------------------------------------

    async def _initialize_multi_language_system(self) -> None:
        logger.info("Initializing multi-language system (23+ languages)...")
        try:
            from unified_language_orchestrator import get_orchestrator
            self.language_orchestrator = get_orchestrator(
                self.config.__dict__ if hasattr(self.config, "__dict__") else {}
            )
            logger.info("✅ Multi-language system initialized")
            status = self.language_orchestrator.get_status()
            logger.info("   Languages active: %s/%s", status["languages_active"], status["languages_registered"])
            if getattr(self.config, "multi_language_warm_on_start", False) and hasattr(self.language_orchestrator, "warm_all"):
                asyncio.create_task(self._warm_multilang_once())
        except Exception as e:
            logger.warning(f"Multi-language initialization warning: {e}")

    async def _warm_multilang_once(self) -> None:
        try:
            if self.language_orchestrator and hasattr(self.language_orchestrator, "warm_all"):
                result = await self.language_orchestrator.warm_all()
                ok = sum(1 for v in result.values() if v)
                logger.debug("Multi-language warm: %s/%s endpoints responded", ok, len(result))
        except Exception as e:
            logger.debug("Multi-language warm: %s", e)

    async def _initialize_exchanges(self) -> None:
        logger.info("Initializing exchange connections...")
        primary = str(getattr(self.config, "primary_exchange", "kraken") or "kraken")
        use_ccxt = getattr(self.config, "use_ccxt", True)
        try:
            if use_ccxt:
                from data.ccxt_data_provider import get_ccxt_async_exchange
                primary_ex = get_ccxt_async_exchange(primary)
                run_mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
                if run_mode != "live":
                    primary_ex = _PaperCCXTWrapper(primary_ex, primary)
                self.exchanges[primary] = primary_ex
                logger.info("✅ Primary exchange via CCXT: %s%s", primary, " (paper/dry-run)" if run_mode != "live" else "")
            else:
                from exchanges.centralized.kraken import KrakenClient
                kraken_api_key = os.getenv("KRAKEN_API_KEY", "")
                kraken_secret = os.getenv("KRAKEN_SECRET_KEY", "")
                run_mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
                want_live = run_mode == "live"
                self.exchanges["kraken"] = KrakenClient(
                    api_key=kraken_api_key or None,
                    secret=kraken_secret or None,
                    dry_run=(not want_live) or (not bool(kraken_api_key and kraken_secret)),
                )
                if kraken_api_key and kraken_secret:
                    logger.info("✅ Kraken exchange connected (credentials present, dry_run=%s)", self.exchanges["kraken"].dry_run)
                else:
                    logger.info("✅ Kraken exchange connected (public-only / paper trading)")

            if not use_ccxt:
                from exchanges.centralized.coinbase_advanced import CoinbaseAdvancedClient
                coinbase_api_key = os.getenv("COINBASE_ADVANCED_API_KEY", "") or os.getenv("COINBASE_CDP_API_KEY", "")
                coinbase_secret_pem = os.getenv("COINBASE_ADVANCED_API_SECRET", "") or os.getenv("COINBASE_CDP_API_SECRET", "")
                if coinbase_api_key and coinbase_secret_pem:
                    self.exchanges["coinbase_advanced"] = CoinbaseAdvancedClient(
                        api_key=coinbase_api_key,
                        api_secret_pem=coinbase_secret_pem,
                        dry_run=(str(getattr(self.config, "run_mode", "paper") or "paper").lower() != "live"),
                    )
                    logger.info("✅ Coinbase Advanced Trade exchange connected (dry_run=%s)",
                                getattr(self.exchanges["coinbase_advanced"], "dry_run", True))
                elif os.getenv("COINBASE_PRO_API_KEY") or os.getenv("COINBASE_PRO_SECRET_KEY"):
                    logger.warning("Detected legacy COINBASE_PRO_* env vars. Use COINBASE_ADVANCED_* (PEM).")
        except Exception as e:
            logger.warning("Exchange initialization warning: %s", e)
        finally:
            try:
                from services.market_data_service import MarketDataService
                self.market_data_service = MarketDataService(
                    exchanges=self.exchanges,
                    primary=str(getattr(self.config, "primary_exchange", "kraken")),
                    secondary=str(getattr(self.config, "secondary_exchange", "coinbase_advanced")),
                    ohlcv_ttl_s=float(getattr(self.config, "market_data_ohlcv_cache_seconds", 30.0) or 30.0),
                    ohlcv_poll_interval_s=float(getattr(self.config, "market_data_ohlcv_poll_interval_seconds", 30.0) or 30.0),
                    ohlcv_retry_attempts=int(getattr(self.config, "market_data_ohlcv_retry_attempts", 2) or 2),
                    persist_tick_store=bool(getattr(self.config, "persist_tick_store", True)),
                )
            except Exception as e:
                logger.warning(f"Market data service initialization warning: {e}")
                self.market_data_service = None

    async def _initialize_quant_fund_upgrades(self) -> None:
        try:
            if not bool(getattr(self.config, "quant_fund_upgrades_enabled", True)):
                return
            mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
            modes = list(getattr(self.config, "quant_fund_upgrades_modes", ["paper", "backtest"]) or ["paper", "backtest"])
            if mode not in [str(m).lower() for m in modes]:
                return
            if not bool(getattr(self.config, "quant_fund_risk_engine_enabled", True)):
                return
            from quant_fund_upgrades.multi_factor_risk_engine import MultiFactorRiskEngine
            self.quant_fund_risk_engine = MultiFactorRiskEngine()
            logger.info("Quant-fund risk engine enabled (%s mode)", mode)
        except Exception as e:
            logger.warning("Quant-fund upgrades unavailable: %s", e)
            self.quant_fund_risk_engine = None

    async def _initialize_argus_strategies(self) -> None:
        logger.info("Initializing ARGUS Ultimate strategies...")
        self.argus_strategies = {
            "quantum_emotion_arbitrage": True,
            "fractal_volatility_harvest": True,
            "adaptive_regime_scalping": True,
            "multi_asset_sentiment_sync": True,
        }
        logger.info("✅ ARGUS strategies initialized")

    async def _initialize_ai_brain(self) -> None:
        enabled = bool(getattr(self.config, "ai_enabled", True))
        if not enabled:
            logger.info("Phase 2: AI brain disabled by config")
            self.ai_brain = None
            return
        force_fallback = bool(getattr(self.config, "live_safe_disable_pinnacle_ai_brain", False))
        logger.info("Phase 2: Initializing AI brain (Pinnacle optional module)...")
        if not force_fallback:
            try:
                from unified_ai_brain import PinnacleAIBrain
                self.ai_brain = PinnacleAIBrain(self.config, market_data_service=self.market_data_service)
                await self.ai_brain.initialize()
                logger.info("✅ Pinnacle AI brain initialized")
                return
            except Exception as e:
                logger.warning("Pinnacle AI brain unavailable: %s; using fallback (strategy engine only)", e)
        else:
            logger.info("LIVE_SAFE runtime: Pinnacle AI brain disabled; using fallback AI brain")
        try:
            from unified_ai_brain import FallbackAIBrain
            self.ai_brain = FallbackAIBrain(self.config, market_data_service=self.market_data_service)
            await self.ai_brain.initialize()
            logger.info("✅ Fallback AI brain initialized (optional modules)")
        except Exception as e2:
            logger.warning("Fallback AI brain unavailable: %s", e2)
            self.ai_brain = None

    async def _quantum_monte_carlo_risk_hook(
        self, *, prev_equity: float = 0.0, new_equity: float = 0.0,
        peak_equity: float = 0.0, max_drawdown: float = 0.0,
    ) -> None:
        if not getattr(self.config, "use_quantum_monte_carlo_risk", False):
            return
        try:
            if len(self._equity_history) < 10:
                return
            arr = np.array(self._equity_history, dtype=float)
            returns = np.diff(arr) / np.maximum(arr[:-1], 1e-9)
            returns_list = returns.tolist()
            use_cloud = getattr(self.config, "use_cloud_quantum", False)
            risk = None
            if use_cloud:
                try:
                    from quantum_unified_stubs import cloud_quantum_run
                    cloud_out = cloud_quantum_run({"returns": returns_list}, provider="simulator", backend=None)
                    res = cloud_out.get("result") if isinstance(cloud_out, dict) else None
                    if isinstance(res, dict) and res.get("var") is not None and res.get("cvar") is not None:
                        risk = {"var": res["var"], "cvar": res["cvar"], "from_classical": False}
                except Exception:
                    pass
            if risk is None:
                from quantum_unified_stubs import quantum_monte_carlo_risk
                risk = quantum_monte_carlo_risk(returns_list, n_samples=min(5000, len(returns_list) * 50), confidence=0.95)
            var_95 = risk.get("var", 0.0)
            cvar_95 = risk.get("cvar", 0.0)
            from_classical = risk.get("from_classical", True)
            es_bps = risk.get("expected_shortfall_bps", (-cvar_95 * 1e4) if cvar_95 < 0 else 0.0)
            n_used = risk.get("n_samples_used", len(returns_list))
            setattr(self, "_last_quantum_var_cvar", {
                "var": var_95, "cvar": cvar_95, "var_95": var_95, "cvar_95": cvar_95,
                "expected_shortfall_bps": es_bps, "n_samples_used": n_used,
            })
            logger.info("Quantum Monte Carlo risk: VaR_95=%.4f CVaR_95=%.4f ES_bps=%.1f n=%s (%s)",
                        var_95, cvar_95, float(es_bps), n_used, "classical" if from_classical else "quantum")
            if getattr(self.config, "use_quantum_var_circuit_breaker", False) and peak_equity > 0:
                cooldown = int(getattr(self.config, "quantum_circuit_breaker_cooldown_seconds", 0) or 0)
                mult = float(getattr(self.config, "quantum_circuit_breaker_threshold_multiplier", 1.0) or 1.0)
                last_trip = getattr(self, "_quantum_circuit_breaker_trip_time", None)
                if cooldown and last_trip is not None and (time.time() - last_trip) < cooldown:
                    pass
                else:
                    threshold = abs(cvar_95) * mult
                    current_dd = (peak_equity - new_equity) / peak_equity
                    if cvar_95 < 0 and current_dd > threshold:
                        logger.critical("Quantum VaR circuit breaker: drawdown %.2f%% > threshold %.2f%% (|CVaR_95|*%.2f)",
                                        current_dd * 100, threshold * 100, mult)
                        self.state = SystemState.EMERGENCY_STOP
                        setattr(self, "_quantum_circuit_breaker_trip_time", time.time())
                        setattr(self, "_quantum_circuit_breaker_trips", int(getattr(self, "_quantum_circuit_breaker_trips", 0)) + 1)
        except Exception as e:
            logger.debug("Quantum Monte Carlo risk hook: %s", e)

    async def _initialize_command_bus(self) -> None:
        if not bool(getattr(self.config, "command_bus_enabled", False)):
            logger.info("Command bus disabled (single-process execution path)")
            return
        try:
            from execution.command_bus import LocalInstructionBus
            db_path = str(getattr(self.config, "command_bus_db_path", "data/command_bus.db") or "data/command_bus.db")
            queue = str(getattr(self.config, "command_bus_queue", "default") or "default")
            self.command_bus = LocalInstructionBus(db_path=db_path, queue=queue)
            logger.info("✅ Command bus initialized (db=%s queue=%s)", db_path, queue)
        except Exception as e:
            self.command_bus = None
            logger.error("Command bus initialization failed: %s", e)

    def _command_bus_secret(self) -> str:
        env_name = str(getattr(self.config, "command_bus_hmac_key_env", "ARGUS_COMMAND_HMAC_KEY") or "ARGUS_COMMAND_HMAC_KEY")
        return str(os.getenv(env_name, "") or "")

    def _signal_to_instruction_payload(self, signal: Any, *, cycle_id: int, correlation_id: str, trace_id: str) -> Dict[str, Any]:
        symbol = str(getattr(signal, "symbol", "") or (signal.get("symbol") if isinstance(signal, dict) else "") or "")
        action = str(
            getattr(signal, "action", "") or getattr(signal, "side", "")
            or (signal.get("action") if isinstance(signal, dict) else "")
            or (signal.get("side") if isinstance(signal, dict) else "") or ""
        ).upper()
        qty_raw = getattr(signal, "quantity", None) if not isinstance(signal, dict) else signal.get("quantity")
        px_raw = getattr(signal, "entry_price", None) if not isinstance(signal, dict) else signal.get("entry_price")
        conf_raw = getattr(signal, "confidence", None) if not isinstance(signal, dict) else signal.get("confidence")
        strategy = str(
            getattr(signal, "strategy", "") or getattr(signal, "source_strategy", "")
            or (signal.get("strategy") if isinstance(signal, dict) else "")
            or (signal.get("source_strategy") if isinstance(signal, dict) else "") or "unknown"
        )
        now_ts = float(time.time())
        ttl = max(0.5, float(getattr(self.config, "command_bus_instruction_ttl_seconds", 5.0) or 5.0))
        return {
            "run_id": str(self.run_id), "trace_id": str(trace_id or ""),
            "correlation_id": str(correlation_id or ""), "cycle_id": int(cycle_id),
            "created_ts": now_ts, "expires_ts": now_ts + ttl,
            "symbol": symbol, "action": action,
            "quantity": float(qty_raw or 0.0), "entry_price": float(px_raw or 0.0),
            "confidence": float(conf_raw or 0.0), "strategy": strategy,
            "reason": "strategy_node_instruction",
            "max_notional_aud": float(getattr(self.config, "command_bus_max_notional_aud", 0.0) or 0.0),
        }

    def _publish_signals_to_command_bus(self, signals: List[Any], *, cycle_id: int, correlation_id: str) -> Dict[str, Any]:
        from execution.command_bus import deterministic_instruction_id, sign_instruction_payload
        if self.command_bus is None:
            return {"published": 0, "rejected": int(len(signals or [])), "reason": "command_bus_unavailable"}
        require_sig = bool(getattr(self.config, "command_bus_require_signature", True))
        secret = self._command_bus_secret()
        if require_sig and not secret:
            logger.error("Command bus publish blocked: signature required but HMAC secret env is missing")
            return {"published": 0, "rejected": int(len(signals or [])), "reason": "missing_hmac_secret"}
        published = 0
        rejected = 0
        for idx, sig in enumerate(list(signals or [])):
            try:
                trace_id = str(getattr(sig, "trace_id", "") or f"{self.run_id}_{cycle_id}_{idx}")
                payload = self._signal_to_instruction_payload(sig, cycle_id=cycle_id, correlation_id=correlation_id, trace_id=trace_id)
                iid = deterministic_instruction_id(payload)
                signature = sign_instruction_payload(payload, secret) if (require_sig and secret) else ""
                self.command_bus.publish(payload=payload, signature=signature, instruction_id=iid,
                                         producer_role="strategy-node", consumer_role="execution-node")
                published += 1
            except Exception as e:
                rejected += 1
                logger.debug("Command bus publish error: %s", e)
        return {"published": int(published), "rejected": int(rejected)}

    def _consume_signals_from_command_bus(self) -> Tuple[List[Any], Dict[str, Any]]:
        from execution.command_bus import instruction_to_signal, validate_instruction_payload, verify_instruction_payload
        if self.command_bus is None:
            return [], {"claimed": 0, "accepted": 0, "rejected": 0, "reason": "command_bus_unavailable"}
        max_batch = max(1, int(getattr(self.config, "command_bus_max_batch", 64) or 64))
        claimed = self.command_bus.claim_pending(limit=max_batch)
        accepted_signals: List[Any] = []
        rejected = 0
        require_sig = bool(getattr(self.config, "command_bus_require_signature", True))
        secret = self._command_bus_secret()
        max_notional_aud = float(getattr(self.config, "command_bus_max_notional_aud", 0.0) or 0.0)
        aud_to_usd = float(getattr(self.config, "aud_to_usd", 0.65) or 0.65)
        for row in claimed:
            iid = str(row.get("instruction_id", "") or "")
            payload = dict(row.get("payload") or {})
            signature = str(row.get("signature", "") or "")
            if require_sig:
                if not secret:
                    self.command_bus.mark_rejected(iid, "missing_hmac_secret")
                    rejected += 1
                    continue
                if not verify_instruction_payload(payload, signature, secret):
                    self.command_bus.mark_rejected(iid, "invalid_signature")
                    rejected += 1
                    continue
            valid = validate_instruction_payload(payload, max_notional_aud=max_notional_aud, aud_to_usd=aud_to_usd)
            if not valid.ok:
                self.command_bus.mark_rejected(iid, valid.reason)
                rejected += 1
                continue
            try:
                state_store = getattr(self.execution_engine, "state_store", None) if self.execution_engine else None
                if state_store is not None and hasattr(state_store, "seen_or_mark"):
                    if state_store.seen_or_mark(f"bus:{iid}"):
                        self.command_bus.mark_rejected(iid, "duplicate_instruction")
                        rejected += 1
                        continue
            except Exception:
                pass
            sig, reason = instruction_to_signal(payload)
            if sig is None:
                self.command_bus.mark_rejected(iid, reason)
                rejected += 1
                continue
            self.command_bus.mark_consumed(iid)
            accepted_signals.append(sig)
        return accepted_signals, {"claimed": int(len(claimed)), "accepted": int(len(accepted_signals)), "rejected": int(rejected)}

    async def _initialize_execution_mesh(self) -> None:
        if not bool(getattr(self.config, "execution_mesh_enabled", False)):
            logger.info("Execution mesh disabled (single execution lane)")
            return
        try:
            from execution.execution_mesh import ExecutionMeshCoordinator
            self.execution_mesh = ExecutionMeshCoordinator(
                max_lanes=int(getattr(self.config, "execution_mesh_max_lanes", 8) or 8),
                max_queue_per_lane=int(getattr(self.config, "execution_mesh_max_queue_per_lane", 128) or 128),
                batch_size=int(getattr(self.config, "execution_mesh_batch_size", 8) or 8),
                parallel_lanes=bool(getattr(self.config, "execution_mesh_parallel_lanes", True)),
                halt_on_lane_error=bool(getattr(self.config, "execution_mesh_halt_on_lane_error", True)),
                allowed_symbols=list(getattr(self.config, "execution_mesh_symbols", []) or []),
            )
            logger.info("✅ Execution mesh initialized (max_lanes=%s queue_per_lane=%s batch_size=%s parallel=%s)",
                        getattr(self.config, "execution_mesh_max_lanes", 8),
                        getattr(self.config, "execution_mesh_max_queue_per_lane", 128),
                        getattr(self.config, "execution_mesh_batch_size", 8),
                        getattr(self.config, "execution_mesh_parallel_lanes", True))
        except Exception as e:
            self.execution_mesh = None
            logger.error("Execution mesh initialization failed: %s", e)

    async def _initialize_execution_engine(self) -> None:
        logger.info("Phase 3: Initializing Kraken DCA execution engine...")
        from unified_execution_engine import KrakenDCAExecutionEngine
        self.execution_engine = KrakenDCAExecutionEngine(self.config, self.exchanges)
        await self.execution_engine.initialize()
        # [WIRING-1] Attach OmegaSQLiteStore as the execution engine state store
        # so order-intent deduplication and bus replay protection share one DB.
        try:
            if hasattr(self.execution_engine, "attach_state_store"):
                self.execution_engine.attach_state_store(self.omega_store)
                logger.info("✅ OmegaSQLiteStore attached to execution engine (state_store wired)")
            else:
                logger.debug("execution_engine.attach_state_store not found; skipping state store wiring")
        except Exception as _attach_e:
            logger.warning("attach_state_store failed (non-fatal): %s", _attach_e)
        logger.info("✅ Kraken DCA execution engine initialized")

    async def _initialize_capital_optimizer(self) -> None:
        logger.info("Phase 4: Initializing capital optimizer...")
        # [WIRING-3] Log paper_trading_peak_mode state at startup so it is
        # always visible in logs regardless of which launcher was used.
        peak_mode = bool(getattr(self.config, "paper_trading_peak_mode", False))
        run_mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
        logger.info(
            "[advisory] paper_trading_peak_mode=%s (run_mode=%s)",
            peak_mode, run_mode,
        )
        from unified_capital_optimizer import CapitalOptimizer1K
        self.capital_optimizer = CapitalOptimizer1K(self.config)
        await self.capital_optimizer.initialize()
        # [WIRING-4] Forward AI brain Black-Litterman views into the capital
        # optimizer so compute_budgets() can use them for allocation.
        try:
            bl_views: Optional[Dict[str, float]] = None
            if self.ai_brain is not None:
                bl_views = (
                    getattr(self.ai_brain, "bl_views", None)
                    or getattr(self.ai_brain, "black_litterman_views", None)
                    or (self.ai_brain.get_bl_views() if callable(getattr(self.ai_brain, "get_bl_views", None)) else None)
                )
            if bl_views and hasattr(self.capital_optimizer, "set_bl_views"):
                self.capital_optimizer.set_bl_views(bl_views)
                logger.info("✅ bl_views forwarded to capital optimizer (%d views)", len(bl_views))
            elif bl_views:
                # Fallback: store directly so compute_budgets() can pick it up
                self.capital_optimizer.bl_views = bl_views
                logger.info("✅ bl_views stored on capital optimizer directly (%d views)", len(bl_views))
            else:
                logger.debug("No bl_views available from AI brain at init time; skipping BL pass-through")
        except Exception as _bl_e:
            logger.warning("bl_views pass-through failed (non-fatal): %s", _bl_e)
        logger.info("✅ Capital optimizer initialized")

    async def _initialize_strategy_allocator(self) -> None:
        if not getattr(self.config, "strategy_allocator_enabled", True):
            return
        try:
            from adaptive.strategy_allocator import StrategyAllocator
            mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
            modes = list(getattr(self.config, "strategy_allocator_modes", ["paper", "backtest"]) or ["paper", "backtest"])
            if mode not in modes:
                return
            self.strategy_allocator = StrategyAllocator(
                enabled=True,
                persist_path=str(getattr(self.config, "strategy_allocator_persist_path", "data/strategy_allocator_stats.json") or "data/strategy_allocator_stats.json"),
                timeframe=str(getattr(self.config, "strategy_allocator_timeframe", "") or ""),
                min_trades_before_bias=int(getattr(self.config, "strategy_allocator_min_trades_before_bias", 5) or 5),
                exploration_c=float(getattr(self.config, "strategy_allocator_exploration_c", 1.2) or 1.2),
                ema_alpha=float(getattr(self.config, "strategy_allocator_ema_alpha", 0.15) or 0.15),
            )
            self.strategy_allocator.load()
            logger.info("✅ Strategy allocator initialized (PnL-based ranking)")
        except Exception as e:
            logger.debug("Strategy allocator unavailable: %s", e)
            self.strategy_allocator = None

    def _initialize_strategy_evaluation_engine(self) -> None:
        if not bool(getattr(self.config, "strategy_evaluation_enabled", True)):
            self.strategy_evaluation_engine = None
            return
        try:
            from evaluation.strategy_evaluation_engine import StrategyEvaluationEngine
            db_path = str(
                getattr(self.config, "strategy_evaluation_db_path", "")
                or getattr(getattr(self, "omega_store", None), "db_path", "")
                or "data/strategy_metrics.db"
            )
            self.strategy_evaluation_engine = StrategyEvaluationEngine(
                db_path=db_path,
                enabled=bool(getattr(self.config, "strategy_evaluation_enabled", True)),
                persist_interval_cycles=int(getattr(self.config, "strategy_evaluation_persist_interval_cycles", 10) or 10),
                min_trades_for_ranking=int(getattr(self.config, "strategy_evaluation_min_trades_for_ranking", 5) or 5),
                use_regime_scoped_metrics=bool(getattr(self.config, "strategy_evaluation_use_regime_scoped_metrics", True)),
                sharpe_like_min_trades=int(getattr(self.config, "strategy_evaluation_sharpe_like_min_trades", 5) or 5),
                max_metrics_history_points=int(getattr(self.config, "strategy_evaluation_max_metrics_history_points", 500) or 500),
            )
            logger.info("✅ Strategy evaluation engine initialized (%s)", db_path)
        except Exception as e:
            self.strategy_evaluation_engine = None
            logger.warning("Strategy evaluation engine unavailable: %s", e)

    def _initialize_self_optimizing_meta_engine(self) -> None:
        if not bool(getattr(self.config, "self_optimizing_meta_enabled", True)):
            self.self_optimizing_meta_engine = None
            return
        try:
            from adaptive.self_optimizing_meta_engine import SelfOptimizingMetaEngine
            db_path = str(getattr(self.config, "self_optimizing_meta_db_path", "") or "data/meta_weights.db")
            self.self_optimizing_meta_engine = SelfOptimizingMetaEngine(
                db_path=db_path,
                enabled=bool(getattr(self.config, "self_optimizing_meta_enabled", True)),
                advisory_only=bool(getattr(self.config, "self_optimizing_meta_advisory_only", False)),
                update_interval_cycles=int(getattr(self.config, "self_optimizing_meta_update_interval_cycles", 10) or 10),
                min_trades_for_reweighting=int(getattr(self.config, "self_optimizing_meta_min_trades_for_reweighting", 5) or 5),
                meta_alpha=float(getattr(self.config, "self_optimizing_meta_alpha", 0.2) or 0.2),
                max_weight_change_per_update=float(getattr(self.config, "self_optimizing_meta_max_weight_change_per_update", 0.10) or 0.10),
                min_weight_per_strategy=float(getattr(self.config, "self_optimizing_meta_min_weight_per_strategy", 0.05) or 0.05),
                max_weight_per_strategy=float(getattr(self.config, "self_optimizing_meta_max_weight_per_strategy", 0.45) or 0.45),
                baseline_weight_mode=str(getattr(self.config, "self_optimizing_meta_baseline_weight_mode", "equal") or "equal"),
                score_weights=dict(getattr(self.config, "self_optimizing_meta_score_weights", {}) or {}),
                regime_multipliers=dict(getattr(self.config, "self_optimizing_meta_regime_multipliers", {}) or {}),
            )
            logger.info("✅ Self-optimizing meta engine initialized (%s)", db_path)
        except Exception as e:
            self.self_optimizing_meta_engine = None
            logger.warning("Self-optimizing meta engine unavailable: %s", e)

    def _handle_strategy_eval_error(self, error: Exception, *, context: str) -> None:
        logger.warning("Strategy evaluation error (%s): %s", context, error)
        if bool(getattr(self.config, "strategy_evaluation_halt_on_error", False)):
            self.state = SystemState.EMERGENCY_STOP
            logger.critical("Strategy evaluation configured fail-closed; entering EMERGENCY_STOP")

    def _initialize_champion_challenger_engine(self) -> None:
        if not bool(getattr(self.config, "champion_challenger_enabled", True)):
            self.champion_challenger_engine = None
            return
        try:
            from evaluation.champion_challenger_engine import ChampionChallengerEngine
            db_path = str(getattr(self.config, "champion_challenger_db_path", "") or "data/champion_challenger.db")
            artifacts_dir = str(getattr(self.config, "champion_challenger_artifacts_dir", "") or "deploy/promotions")
            weights = dict(
                getattr(self.config, "champion_challenger_promotion_weights", None)
                or {"net_pnl": 1.0, "expectancy": 1.0, "profit_factor": 0.75,
                    "sharpe_like": 1.0, "drawdown_penalty": 1.25, "fee_penalty": 0.5}
            )
            self.champion_challenger_engine = ChampionChallengerEngine(
                db_path=db_path, artifacts_dir=artifacts_dir,
                enabled=bool(getattr(self.config, "champion_challenger_enabled", True)),
                advisory_only=bool(getattr(self.config, "champion_challenger_advisory_only", True)),
                min_trades_for_promotion=int(getattr(self.config, "champion_challenger_min_trades_for_promotion", 10) or 10),
                max_drawdown_pct_for_promotion=float(getattr(self.config, "champion_challenger_max_drawdown_pct_for_promotion", 0.12) or 0.12),
                require_expectancy_improvement=bool(getattr(self.config, "champion_challenger_require_expectancy_improvement", True)),
                require_profit_factor_improvement=bool(getattr(self.config, "champion_challenger_require_profit_factor_improvement", False)),
                require_sharpe_like_improvement=bool(getattr(self.config, "champion_challenger_require_sharpe_like_improvement", True)),
                promotion_weights=weights,
                persist_interval_cycles=int(getattr(self.config, "champion_challenger_persist_interval_cycles", 10) or 10),
            )
            active_champion = self.champion_challenger_engine.get_active_champion()
            if active_champion is None:
                strategy_set = list(getattr(self.config, "strategies_enabled", []) or []) or ["unknown"]
                config_payload = json.dumps(self.config.__dict__, sort_keys=True, ensure_ascii=True, default=str)
                config_hash = hashlib.sha256(config_payload.encode("utf-8")).hexdigest()
                bundle_hint = "deploy/bundles" if Path("deploy/bundles").exists() else ""
                champion_profile = self.champion_challenger_engine.register_champion(
                    profile_id=f"champion_{str(getattr(self, 'run_id', 'unknown'))}",
                    source_bundle_path=str(bundle_hint),
                    config_hash=str(config_hash),
                    strategy_set=[str(s) for s in strategy_set if str(s).strip()],
                    version_label=f"runtime_{str(getattr(self, 'run_id', 'unknown'))}",
                    status="active",
                )
                logger.info("✅ Champion/challenger engine initialized and champion bootstrapped (%s)", champion_profile.profile_id)
            else:
                logger.info("✅ Champion/challenger engine initialized (active champion=%s)", active_champion.profile_id)
        except Exception as e:
            self.champion_challenger_engine = None
            logger.warning("Champion/challenger engine unavailable: %s", e)

    def _attach_champion_challenger_context(self, signals: List[Any], regime_label: str) -> None:
        cc_engine = getattr(self, "champion_challenger_engine", None)
        if cc_engine is None:
            return
        active = cc_engine.get_active_champion()
        if active is None:
            return
        best = cc_engine.best_challengers_by_promotion_score(limit=1)
        best_decision = best[0] if best else None
        for sig in list(signals or []):
            try:
                if isinstance(sig, dict):
                    sig["champion_profile_id"] = str(active.profile_id)
                    if best_decision is not None:
                        sig["challenger_profile_id"] = str(best_decision.challenger_id)
                        sig["promotion_decision"] = str(best_decision.decision)
                        sig["promotion_score"] = float(best_decision.promotion_score)
                        sig["promotion_reasons"] = list(best_decision.reasons or [])
                        if regime_label:
                            sig["promotion_regime_label"] = str(regime_label)
                else:
                    setattr(sig, "champion_profile_id", str(active.profile_id))
                    if best_decision is not None:
                        setattr(sig, "challenger_profile_id", str(best_decision.challenger_id))
                        setattr(sig, "promotion_decision", str(best_decision.decision))
                        setattr(sig, "promotion_score", float(best_decision.promotion_score))
                        setattr(sig, "promotion_reasons", list(best_decision.reasons or []))
                        if regime_label:
                            setattr(sig, "promotion_regime_label", str(regime_label))
            except Exception as e:
                logger.debug("Champion/challenger context attach failed: %s", e)
                break

    def _attach_strategy_evaluation_context(self, signals: List[Any], regime_label: str) -> None:
        engine = getattr(self, "strategy_evaluation_engine", None)
        if engine is None:
            return
        rankable = 0
        try:
            rankable = int(engine.rankable_strategy_count())
        except Exception:
            rankable = 0
        for sig in list(signals or []):
            try:
                strategy = str(
                    self._signal_get(sig, "source_strategy", None)
                    or self._signal_get(sig, "strategy", None) or "unknown"
                )
                symbol = str(self._signal_get(sig, "symbol", None) or "")
                reg = str(self._signal_get(sig, "regime_label", None) or regime_label or "")
                ctx = engine.get_decision_context(strategy_name=strategy, symbol=symbol or None, regime_label=reg or None)
                if not ctx:
                    continue
                if isinstance(sig, dict):
                    sig.update(ctx)
                else:
                    for k, v in ctx.items():
                        setattr(sig, k, v)
            except Exception as e:
                self._handle_strategy_eval_error(e, context="attach_context")
                break
        if rankable > 0:
            logger.info("strategy ranking context available for %s strategies", rankable)

    async def _initialize_hft_engine(self) -> None:
        if not bool(getattr(self.config, "hft_enabled", True)):
            logger.info("Phase 4.5: HFT Scalping Engine disabled by config")
            self.hft_engine = None
            return
        logger.info("Phase 4.5: Initializing HFT Scalping Engine...")
        try:
            from hft_engine.hft_scalping_engine import HFTScalpingEngine
            self.hft_engine = HFTScalpingEngine(self.config, exchanges=self.exchanges)
            logger.info("✅ HFT Scalping Engine initialized")
        except Exception as e:
            logger.warning(f"HFT Engine initialization failed: {e}")
            self.hft_engine = None

    async def _initialize_hft_infrastructure(self) -> None:
        if not bool(getattr(self.config, "hft_enabled", True)) or not bool(getattr(self.config, "use_advanced_hft_infrastructure", False)):
            logger.info("Phase 4.6: Advanced HFT Infrastructure disabled by config")
            self.hft_infrastructure = None
            return
        logger.info("Phase 4.6: Initializing Advanced HFT Infrastructure...")
        try:
            from hft_engine.advanced_realtime_hft_infrastructure import get_hft_infrastructure
            cfg = self.config.__dict__ if hasattr(self.config, "__dict__") else {}
            self.hft_infrastructure = get_hft_infrastructure(config=cfg, hft_engine=self.hft_engine)
            asyncio.create_task(self.hft_infrastructure.run_event_loop())
            logger.info("✅ Advanced HFT Infrastructure initialized (Kernel Bypass Sim Active)")
        except Exception as e:
            logger.warning(f"Advanced HFT Infrastructure initialization failed: {e}")
            self.hft_infrastructure = None

    async def _initialize_monitoring(self) -> None:
        logger.info("Phase 5: Initializing monitoring infrastructure...")
        try:
            from core.health import ArgusHealthMonitor
            self.monitoring = ArgusHealthMonitor()
            logger.info("✅ Monitoring system initialized (ArgusHealthMonitor)")
        except Exception as e:
            logger.warning("Monitoring system unavailable: %s", e)
            self.monitoring = None

    async def _initialize_production_modules(self) -> None:
        logger.info("Phase 6: Initializing production modules...")
        count = 0
        try:
            from core.rate_limiter import RateLimiterManager
            self.rate_limiter = RateLimiterManager()
            count += 1
        except Exception as e:
            logger.debug("RateLimiter unavailable: %s", e)
        try:
            from core.data_sanitizer import DataSanitizer
            self.data_sanitizer = DataSanitizer()
            count += 1
        except Exception as e:
            logger.debug("DataSanitizer unavailable: %s", e)
        logger.info("Phase 6: %d production module(s) loaded", count)
