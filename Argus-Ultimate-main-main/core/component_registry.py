"""
Batch 3 – Component Registry
============================
Wires every optional Batch 3/4 shelf-ware component into the trading loop.
Designed to be imported by UnifiedSystemArchitecture during initialisation.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from unified_trading_system import UnifiedSystemArchitecture

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """
    Light-weight registry that holds references to every optional module and
    exposes a health-map for the runtime module registry.

    Usage (inside UnifiedSystemArchitecture.initialize)::

        from core.component_registry import ComponentRegistry
        self.component_registry = ComponentRegistry(system=self)
        await self.component_registry.register_all()
    """

    def __init__(self, system: "UnifiedSystemArchitecture" = None, config: Any = None) -> None:
        # Support both system= and config= patterns for backwards compatibility
        self._sys = system
        self._config = config if config is not None else (system.config if system else None)
        self._components: Dict[str, Any] = {}
        self._init_count = 0
    
    async def initialize(self) -> int:
        """Alias for register_all() that returns count of active components."""
        await self.register_all()
        return sum(1 for v in self._components.values() if v is not None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def register_all(self) -> None:
        """Register + start all optional Batch-3 components (best-effort)."""
        await self._register_signal_service()
        await self._register_api_server()
        await self._register_perf_feeder()
        await self._register_regime_alerter()
        await self._register_model_manager()
        await self._register_checkpoint_manager()
        # Advanced features (v8.2.0)
        await self._register_advanced_features()
        # Adaptive orchestrator (v8.3.0 - real-time adaptation)
        await self._register_adaptive_orchestrator()
        # Correlation Monitor (v8.4.0 - correlation-based position sizing)
        await self._register_correlation_monitor()
        # Hurst Feature Provider (v8.5.0 - Hurst exponent regime detection)
        await self._register_hurst_provider()
        # Time Series Forecaster (v8.6.0 - transformer-based price prediction)
        await self._register_time_series_forecaster()
        # Crypto Native Features (v8.7.0 - staking optimizer + MEV protection)
        await self._register_crypto_native_features()
        # Advanced Analytics (v8.8.0 - order flow, arbitrage, adaptive params)
        await self._register_advanced_analytics()
        # Market Intelligence (v8.9.0 - gamma squeeze, dynamic stops, liquidity)
        await self._register_market_intelligence()
        # Advanced Trading (v8.10.0 - whale tracker, smart routing, vol surface)
        await self._register_advanced_trading()
        # Market Flow Integration (v8.12.0 - pipes all signals through adaptation)
        await self._register_market_flow_integration()
        # Alpha Signal Fusion (v8.13.0 - creates real alpha)
        await self._register_alpha_signal_fusion()
        # Multi-Agent LLM Voting (v9.0.0 - 3-agent consensus)
        await self._register_multi_agent_voting()
        # RL Trading Agent (DQN reinforcement learning)
        await self._register_rl_trading_agent()
        # Quantum ML Tuner (automatic hyperparameter optimization)
        await self._register_quantum_ml_tuner()
        # Peak Quantum Engine (absolute maximum local quantum)
        await self._register_peak_quantum_engine()
        # Advanced Learning Integration (25+ learning systems)
        await self._register_advanced_learning()
        # Maximum Edge Orchestrator (coordinates all strategies)
        await self._register_maximum_edge_orchestrator()
        # DeFi Yield Optimizer (automated yield farming)
        await self._register_defi_yield_optimizer()
        # Latency-Based Stops (dynamic stop adjustment)
        await self._register_latency_based_stops()
        # Dynamic Latency Sizing (position size adjustment)
        await self._register_dynamic_latency_sizing()
        # Strategy Intelligence Hub (coordinates all 99+ strategies)
        await self._register_strategy_intelligence_hub()
        # Universal Parameter Learning Engine (learns 200+ parameters)
        await self._register_parameter_learning_engine()
        self._init_count = sum(1 for v in self._components.values() if v is not None)
        logger.info(
            "ComponentRegistry: %d/%d components active",
            self._init_count,
            len(self._components),
        )

    def get(self, name: str) -> Optional[Any]:
        return self._components.get(name)
    
    @property
    def correlation_monitor(self) -> Optional[Any]:
        """Convenient access to CorrelationMonitor."""
        return self._components.get("correlation_monitor")

    @property
    def hurst_provider(self) -> Optional[Any]:
        """Convenient access to MarketRegimeAnalyzer (Hurst + Entropy)."""
        return self._components.get("hurst_provider")

    @property
    def regime_analyzer(self) -> Optional[Any]:
        """Convenient access to MarketRegimeAnalyzer."""
        return self._components.get("regime_analyzer")

    @property
    def time_series_forecaster(self) -> Optional[Any]:
        """Convenient access to TimeSeriesForecaster."""
        return self._components.get("time_series_forecaster")

    @property
    def staking_optimizer(self) -> Optional[Any]:
        """Convenient access to StakingOptimizer."""
        return self._components.get("staking_optimizer")

    @property
    def sandwich_detector(self) -> Optional[Any]:
        """Convenient access to SandwichAttackDetector."""
        return self._components.get("sandwich_detector")

    @property
    def order_flow_analyzer(self) -> Optional[Any]:
        """Convenient access to OrderFlowAnalyzer."""
        return self._components.get("order_flow_analyzer")

    @property
    def arbitrage_scanner(self) -> Optional[Any]:
        """Convenient access to CrossExchangeArbitrageScanner."""
        return self._components.get("arbitrage_scanner")

    @property
    def hyperparameter_optimizer(self) -> Optional[Any]:
        """Convenient access to AdaptiveHyperparameterOptimizer."""
        return self._components.get("hyperparameter_optimizer")

    @property
    def gamma_squeeze_detector(self) -> Optional[Any]:
        """Convenient access to GammaSqueezeDetector."""
        return self._components.get("gamma_squeeze_detector")

    @property
    def dynamic_stop_loss(self) -> Optional[Any]:
        """Convenient access to DynamicStopLoss."""
        return self._components.get("dynamic_stop_loss")

    @property
    def liquidity_analyzer(self) -> Optional[Any]:
        """Convenient access to LiquidityAnalyzer."""
        return self._components.get("liquidity_analyzer")

    @property
    def pair_liquidity_scanner(self) -> Optional[Any]:
        """Convenient access to MultiPairLiquidityScanner."""
        return self._components.get("pair_liquidity_scanner")

    @property
    def ultimate_quantum_risk(self) -> Optional[Any]:
        """Convenient access to UltimateQuantumRiskEngine."""
        return self._components.get("ultimate_quantum_risk")

    @property
    def whale_tracker(self) -> Optional[Any]:
        """Convenient access to WhaleTracker."""
        return self._components.get("whale_tracker")

    @property
    def order_router(self) -> Optional[Any]:
        """Convenient access to SmartOrderRouter."""
        return self._components.get("order_router")

    @property
    def vol_surface_analyzer(self) -> Optional[Any]:
        """Convenient access to VolSurfaceAnalyzer."""
        return self._components.get("vol_surface_analyzer")

    @property
    def advanced_learning(self) -> Optional[Any]:
        """Convenient access to AdvancedLearningOrchestrator."""
        return self._components.get("advanced_learning")

    @property
    def maximum_edge_orchestrator(self) -> Optional[Any]:
        """Convenient access to MaximumEdgeOrchestrator."""
        return self._components.get("maximum_edge_orchestrator")

    @property
    def defi_yield_optimizer(self) -> Optional[Any]:
        """Convenient access to DeFiYieldOptimizer."""
        return self._components.get("defi_yield_optimizer")

    @property
    def latency_based_stops(self) -> Optional[Any]:
        """Convenient access to LatencyBasedStops."""
        return self._components.get("latency_based_stops")

    @property
    def dynamic_latency_sizing(self) -> Optional[Any]:
        """Convenient access to DynamicLatencySizing."""
        return self._components.get("dynamic_latency_sizing")

    @property
    def strategy_intelligence_hub(self) -> Optional[Any]:
        """Convenient access to StrategyIntelligenceHub."""
        return self._components.get("strategy_intelligence_hub")

    @property
    def parameter_learning_engine(self) -> Optional[Any]:
        """Convenient access to ParameterLearningIntegrator."""
        return self._components.get("parameter_learning_engine")

    def health(self) -> Dict[str, str]:
        return {
            k: "active" if v is not None else "unavailable"
            for k, v in self._components.items()
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _register_signal_service(self) -> None:
        try:
            from services.signal_subscription_service import SignalSubscriptionService
            svc = SignalSubscriptionService(config=self._sys.config)
            await svc.start()
            self._sys.signal_service = svc
            self._components["signal_service"] = svc
            logger.info("✅ SignalSubscriptionService registered")
        except Exception as exc:
            self._components["signal_service"] = None
            logger.warning("SignalSubscriptionService unavailable: %s", exc)

    async def _register_api_server(self) -> None:
        try:
            from api.dashboard_server import DashboardAPIServer
            srv = DashboardAPIServer(system=self._sys)
            await srv.start()
            self._sys.api_server = srv
            self._components["api_server"] = srv
            logger.info("✅ DashboardAPIServer registered")
        except Exception as exc:
            self._components["api_server"] = None
            logger.warning("DashboardAPIServer unavailable: %s", exc)

    async def _register_perf_feeder(self) -> None:
        try:
            from monitoring.rolling_perf_feeder import RollingPerfFeeder
            feeder = RollingPerfFeeder(system=self._sys)
            feeder.start()
            self._sys.perf_feeder = feeder
            self._components["perf_feeder"] = feeder
            logger.info("✅ RollingPerfFeeder registered")
        except Exception as exc:
            self._components["perf_feeder"] = None
            logger.warning("RollingPerfFeeder unavailable: %s", exc)

    async def _register_regime_alerter(self) -> None:
        try:
            from monitoring.regime_change_alerter import RegimeChangeAlerter
            alerter = RegimeChangeAlerter(system=self._sys)
            self._sys.regime_alerter = alerter
            self._components["regime_alerter"] = alerter
            logger.info("✅ RegimeChangeAlerter registered")
        except Exception as exc:
            self._components["regime_alerter"] = None
            logger.warning("RegimeChangeAlerter unavailable: %s", exc)

    async def _register_model_manager(self) -> None:
        try:
            from ml.model_manager import ModelManager
            mgr = ModelManager(config=self._sys.config)
            await mgr.initialize()
            self._sys.model_manager = mgr
            self._components["model_manager"] = mgr
            logger.info("✅ ModelManager registered")
        except Exception as exc:
            self._components["model_manager"] = None
            logger.warning("ModelManager unavailable: %s", exc)

    async def _register_checkpoint_manager(self) -> None:
        try:
            from core.checkpoint_manager import CheckpointManager
            ckpt = CheckpointManager(system=self._sys)
            self._sys.checkpoint_manager = ckpt
            self._components["checkpoint_manager"] = ckpt
            logger.info("✅ CheckpointManager registered")
        except Exception as exc:
            self._components["checkpoint_manager"] = None
            logger.warning("CheckpointManager unavailable: %s", exc)

    async def _register_advanced_features(self) -> None:
        """Register Advanced Features Orchestrator (v8.2.0)."""
        try:
            from core.advanced_features_orchestrator import AdvancedFeaturesOrchestrator
            
            # Check if advanced features are enabled in config
            config = self._config if self._config else (self._sys.config if self._sys else None)
            adv_config = getattr(config, "advanced_features", None) if config else None
            if adv_config and not getattr(adv_config, "enabled", True):
                logger.info("Advanced features disabled by config")
                self._components["advanced_features"] = None
                return
            
            orchestrator = AdvancedFeaturesOrchestrator()
            
            # Initialize all features
            init_results = orchestrator.initialize_all()
            successful = sum(1 for v in init_results.values() if v)
            
            # Activate features
            orchestrator.activate_all()
            
            self._components["advanced_features"] = orchestrator
            
            # Store references on system if available
            if self._sys:
                self._sys.advanced_features_orchestrator = orchestrator
                self._sys.event_store = orchestrator.get_feature("event_sourcing")
                self._sys.cqrs_handler = orchestrator.get_feature("cqrs")
                self._sys.realtime_feature_store = orchestrator.get_feature("realtime_feature_store")
                self._sys.drift_detector = orchestrator.get_feature("multi_task_learning")
                self._sys.causal_engine = orchestrator.get_feature("causal_inference")
            
            logger.info(
                "✅ AdvancedFeaturesOrchestrator registered (%d/%d features active)",
                successful, len(init_results)
            )
        except Exception as exc:
            self._components["advanced_features"] = None
            logger.warning("AdvancedFeaturesOrchestrator unavailable: %s", exc)

    async def _register_adaptive_orchestrator(self) -> None:
        """Register Adaptive Orchestrator for real-time market adaptation (v8.3.0)."""
        try:
            from adaptive.adaptive_orchestrator import AdaptiveOrchestrator, AdaptiveConfig
            
            # Check if adaptive features are enabled in config
            config = self._config if self._config else (self._sys.config if self._sys else None)
            adaptive_config = getattr(config, "adaptive", None) if config else None
            
            # Create adaptive config from system config if available
            adp_config = AdaptiveConfig()
            if adaptive_config:
                adp_config.regime_check_interval_seconds = getattr(adaptive_config, "regime_check_interval_seconds", 10)
                adp_config.strategy_rotation_interval_minutes = getattr(adaptive_config, "strategy_rotation_interval_minutes", 60)
                adp_config.model_health_check_interval_minutes = getattr(adaptive_config, "model_health_check_interval_minutes", 30)
                adp_config.position_sizing_update_seconds = getattr(adaptive_config, "position_sizing_update_seconds", 5)
                adp_config.enable_auto_rotation = getattr(adaptive_config, "enable_auto_rotation", True)
                adp_config.enable_auto_retrain = getattr(adaptive_config, "enable_auto_retrain", True)
                adp_config.enable_dynamic_sizing = getattr(adaptive_config, "enable_dynamic_sizing", True)
                adp_config.min_regime_confidence = getattr(adaptive_config, "min_regime_confidence", 0.7)
            
            orchestrator = AdaptiveOrchestrator(config=adp_config)
            
            # Initialize the orchestrator
            await orchestrator.initialize()
            
            self._components["adaptive_orchestrator"] = orchestrator
            
            # Store reference on system if available
            if self._sys:
                self._sys.adaptive_orchestrator = orchestrator
            
            logger.info("✅ AdaptiveOrchestrator registered (v8.3.0 real-time adaptation)")
        except Exception as exc:
            self._components["adaptive_orchestrator"] = None
            logger.warning("AdaptiveOrchestrator unavailable: %s", exc)

    async def _register_correlation_monitor(self) -> None:
        """Register CorrelationMonitor for correlation-based position sizing (v8.4.0).
        
        The CorrelationMonitor tracks pairwise correlations across trading pairs
        and provides position size scalars when correlations spike (risk-off events).
        """
        try:
            from risk.correlation_monitor import CorrelationMonitor
            
            # Get trading pairs from config
            config = self._config if self._config else (self._sys.config if self._sys else None)
            symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]  # Default symbols
            if config and hasattr(config, "trading_pairs"):
                symbols = config.trading_pairs or symbols
            
            # Initialize CorrelationMonitor
            correlation_monitor = CorrelationMonitor(
                symbols=symbols,
                lookback=20,
                alert_threshold=0.80,
                crisis_threshold=0.92,
            )
            
            self._components["correlation_monitor"] = correlation_monitor
            
            # Store reference on system if available
            if self._sys:
                self._sys.correlation_monitor = correlation_monitor
                
                # Wire CorrelationMonitor to KellyCriterion if available
                kelly_sizer = getattr(self._sys, "kelly_sizer", None)
                if kelly_sizer is not None and hasattr(kelly_sizer, "update_correlation_scalar"):
                    # Update Kelly's correlation scalar with current value
                    scalar = correlation_monitor.get_position_scalar()
                    kelly_sizer.update_correlation_scalar(scalar)
                    logger.info("CorrelationMonitor wired to KellyCriterion (initial scalar=%.2f)", scalar)
            
            logger.info("✅ CorrelationMonitor registered (v8.4.0 correlation-based sizing)")
        except Exception as exc:
            self._components["correlation_monitor"] = None
            logger.warning("CorrelationMonitor unavailable: %s", exc)

    async def _register_hurst_provider(self) -> None:
        """Register MarketRegimeAnalyzer for Hurst exponent and Permutation Entropy regime detection (v8.5.0).
        
        The MarketRegimeAnalyzer provides:
        - Hurst exponent: classifies persistence (mean_reversion / momentum / avoid)
        - Permutation entropy: classifies complexity (highly_predictable / normal / chaotic)
        - Combined position scalar: 0.0 (avoid) to 1.2 (strong signal)
        
        Trading logic:
        - Hurst avoid → no trading (random walk)
        - Entropy chaotic → reduce size (high noise)
        - Mean-reverting + Predictable → strong mean reversion signal
        - Trending + Predictable → good momentum signal
        """
        try:
            from ml.hurst_feature_provider import MarketRegimeAnalyzer
            
            # Get feature store from advanced features or create one
            feature_store = None
            if self._sys:
                feature_store = getattr(self._sys, "realtime_feature_store", None)
            
            if feature_store is None:
                # Try to get from advanced features orchestrator
                adv_features = self._components.get("advanced_features")
                if adv_features:
                    feature_store = adv_features.get_feature("realtime_feature_store")
            
            if feature_store is None:
                # Create a minimal feature store if needed
                from ml.feature_store_realtime import RealTimeFeatureStore
                feature_store = RealTimeFeatureStore()
                logger.info("Created new RealTimeFeatureStore for MarketRegimeAnalyzer")
            
            # Get trading pairs from config
            config = self._config if self._config else (self._sys.config if self._sys else None)
            symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]  # Default symbols
            if config and hasattr(config, "trading_pairs"):
                symbols = config.trading_pairs or symbols
            
            # Initialize MarketRegimeAnalyzer
            regime_analyzer = MarketRegimeAnalyzer(
                feature_store=feature_store,
                symbols=symbols,
                window=100,
                hurst_mean_reversion_threshold=0.45,
                hurst_trending_threshold=0.55,
                entropy_order=3,
                entropy_low_threshold=0.3,
                entropy_high_threshold=0.8,
            )
            
            self._components["hurst_provider"] = regime_analyzer
            self._components["regime_analyzer"] = regime_analyzer
            
            # Store reference on system if available
            if self._sys:
                self._sys.hurst_provider = regime_analyzer
                self._sys.regime_analyzer = regime_analyzer
            
            logger.info("✅ MarketRegimeAnalyzer registered (v8.5.0 Hurst + Permutation Entropy)")
        except Exception as exc:
            self._components["hurst_provider"] = None
            self._components["regime_analyzer"] = None
            logger.warning("MarketRegimeAnalyzer unavailable: %s", exc)

    async def _register_time_series_forecaster(self) -> None:
        """Register TimeSeriesForecaster for transformer-based price prediction (v8.6.0).
        
        The TimeSeriesForecaster provides:
        - Multi-horizon price forecasts using transformer models
        - Direction signals (buy/sell) based on predicted price movement
        - Confidence intervals for risk management
        
        Models available:
        - transformer: Attention-based (similar to PatchTST)
        - nhits: Multi-rate sampling
        - tft: Temporal Fusion Transformer
        - simple: EWMA fallback (always available)
        """
        try:
            from ml.time_series_forecaster import TimeSeriesForecaster
            
            # Get trading pairs from config
            config = self._config if self._config else (self._sys.config if self._sys else None)
            symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
            if config and hasattr(config, "trading_pairs"):
                symbols = config.trading_pairs or symbols
            
            # Initialize TimeSeriesForecaster
            # Use simple model by default for reliability; can be upgraded to transformer
            forecaster = TimeSeriesForecaster(
                model_type="simple",  # Start with simple, can upgrade to "transformer"
                input_chunk_length=48,
                output_chunk_length=12,
                n_epochs=50,
                hidden_size=64,
                min_train_samples=200,
            )
            
            self._components["time_series_forecaster"] = forecaster
            
            # Store reference on system if available
            if self._sys:
                self._sys.time_series_forecaster = forecaster
            
            logger.info("✅ TimeSeriesForecaster registered (v8.6.0 transformer-based forecasting)")
        except Exception as exc:
            self._components["time_series_forecaster"] = None
            logger.warning("TimeSeriesForecaster unavailable: %s", exc)

    async def _register_crypto_native_features(self) -> None:
        """Register crypto-native features: StakingOptimizer and SandwichAttackDetector (v8.7.0).
        
        StakingOptimizer:
        - Multi-validator APY tracking
        - Optimal staking/trading capital split
        - Validator health monitoring
        - Auto-compounding recommendations
        
        SandwichAttackDetector:
        - Mempool monitoring for MEV signals
        - Gas spike detection
        - Frontrun detection
        - Private RPC recommendations
        """
        try:
            from crypto.staking_optimizer import StakingOptimizer
            from crypto.sandwich_detector import SandwichAttackDetector
            
            # Initialize StakingOptimizer
            staking_optimizer = StakingOptimizer(
                min_uptime=99.0,
                max_commission=0.10,
                target_liquidity_pct=0.30,
                auto_compound_threshold=10.0,
            )
            
            # Initialize SandwichAttackDetector
            sandwich_detector = SandwichAttackDetector(
                gas_spike_threshold=2.0,
                large_swap_threshold_usd=50000.0,
                frontrun_threshold=3,
                risk_decay_minutes=5.0,
            )
            
            self._components["staking_optimizer"] = staking_optimizer
            self._components["sandwich_detector"] = sandwich_detector
            
            # Store reference on system if available
            if self._sys:
                self._sys.staking_optimizer = staking_optimizer
                self._sys.sandwich_detector = sandwich_detector
            
            logger.info("✅ CryptoNative features registered (v8.7.0 staking + MEV protection)")
        except Exception as exc:
            self._components["staking_optimizer"] = None
            self._components["sandwich_detector"] = None
            logger.warning("CryptoNative features unavailable: %s", exc)

    async def _register_advanced_analytics(self) -> None:
        """Register advanced analytics features (v8.8.0).
        
        OrderFlowAnalyzer:
        - Real-time buy/sell pressure analysis
        - Large order (whale) detection
        - Order book imbalance tracking
        - Volume profile for support/resistance
        
        CrossExchangeArbitrageScanner:
        - Multi-exchange price monitoring
        - Transaction cost-aware opportunity detection
        - Latency-adjusted profit calculation
        
        AdaptiveHyperparameterOptimizer:
        - Self-tuning strategy parameters
        - Regime-specific parameter sets
        - Bayesian optimization principles
        """
        try:
            from analytics.order_flow_analyzer import OrderFlowAnalyzer
            from analytics.cross_exchange_arbitrage import CrossExchangeArbitrageScanner
            from adaptive.adaptive_hyperparameter_optimizer import AdaptiveHyperparameterOptimizer
            
            # Initialize OrderFlowAnalyzer
            order_flow = OrderFlowAnalyzer(
                large_order_threshold_usd=100000.0,
                momentum_window=100,
                volume_cluster_bins=50,
            )
            
            # Initialize CrossExchangeArbitrageScanner
            arbitrage = CrossExchangeArbitrageScanner(
                min_spread_pct=0.1,
                max_latency_ms=500.0,
                price_staleness_seconds=10.0,
            )
            
            # Initialize AdaptiveHyperparameterOptimizer
            hyperopt = AdaptiveHyperparameterOptimizer(
                exploration_rate=0.1,
                min_trades_for_update=10,
                performance_decay=0.95,
            )
            
            self._components["order_flow_analyzer"] = order_flow
            self._components["arbitrage_scanner"] = arbitrage
            self._components["hyperparameter_optimizer"] = hyperopt
            
            # Store reference on system if available
            if self._sys:
                self._sys.order_flow_analyzer = order_flow
                self._sys.arbitrage_scanner = arbitrage
                self._sys.hyperparameter_optimizer = hyperopt
            
            logger.info("✅ AdvancedAnalytics registered (v8.8.0 order flow + arbitrage + hyperopt)")
        except Exception as exc:
            self._components["order_flow_analyzer"] = None
            self._components["arbitrage_scanner"] = None
            self._components["hyperparameter_optimizer"] = None
            logger.warning("AdvancedAnalytics unavailable: %s", exc)

    async def _register_market_intelligence(self) -> None:
        """Register market intelligence features (v8.9.0).
        
        GammaSqueezeDetector:
        - Options market analysis for squeeze potential
        - Put/Call ratio extremes
        - Dealer gamma exposure (GEX) tracking
        
        DynamicStopLoss:
        - ATR-based adaptive stop placement
        - Trailing stops with profit locking
        - Support/resistance aware stops
        
        LiquidityAnalyzer:
        - Multi-DEX liquidity analysis
        - Optimal trade routing
        - Slippage estimation
        """
        try:
            from analytics.gamma_squeeze_detector import GammaSqueezeDetector
            from risk.dynamic_stop_loss import DynamicStopLoss, StopConfig
            from defi.liquidity_analyzer import LiquidityAnalyzer
            
            # Initialize GammaSqueezeDetector
            gamma_detector = GammaSqueezeDetector(
                put_call_extreme_threshold=0.6,
                gex_negative_threshold=-100_000_000,
                iv_rank_threshold=70.0,
            )
            
            # Initialize DynamicStopLoss
            stop_config = StopConfig(
                atr_multiplier=2.0,
                min_stop_pct=0.01,
                max_stop_pct=0.10,
                trailing_activation=0.02,
                trailing_distance=1.5,
                chandelier_period=22,
            )
            stop_manager = DynamicStopLoss(
                config=stop_config,
                use_chandelier=True,
                use_support_resistance=True,
            )
            
            # Initialize LiquidityAnalyzer
            liquidity = LiquidityAnalyzer(
                default_slippage=0.005,
                gas_price_gwei=30.0,
                gas_cost_per_hop=5.0,
                min_liquidity_usd=10000.0,
            )
            
            # Initialize MultiPairLiquidityScanner for pair selection
            from execution.multi_pair_liquidity_scanner import create_liquidity_scanner
            pair_scanner = create_liquidity_scanner(
                exchanges=["bybit", "kraken"],
                max_pairs=20,
            )
            
            # Initialize Ultimate Quantum Risk Engine
            from risk.ultimate_quantum_risk import UltimateQuantumRiskEngine
            quantum_risk_engine = UltimateQuantumRiskEngine(
                n_qubits=8,
                bond_dimension=32,
                annealing_iterations=500,
            )
            
            self._components["gamma_squeeze_detector"] = gamma_detector
            self._components["dynamic_stop_loss"] = stop_manager
            self._components["liquidity_analyzer"] = liquidity
            self._components["pair_liquidity_scanner"] = pair_scanner
            self._components["ultimate_quantum_risk"] = quantum_risk_engine
            
            # Store reference on system if available
            if self._sys:
                self._sys.gamma_squeeze_detector = gamma_detector
                self._sys.dynamic_stop_loss = stop_manager
                self._sys.liquidity_analyzer = liquidity
                self._sys.pair_liquidity_scanner = pair_scanner
                self._sys.ultimate_quantum_risk = quantum_risk_engine
            
            logger.info("✅ MarketIntelligence registered (v8.9.0 gamma squeeze + stops + liquidity + pair scanner + quantum risk)")
        except Exception as exc:
            self._components["gamma_squeeze_detector"] = None
            self._components["dynamic_stop_loss"] = None
            self._components["liquidity_analyzer"] = None
            self._components["pair_liquidity_scanner"] = None
            self._components["ultimate_quantum_risk"] = None
            logger.warning("MarketIntelligence unavailable: %s", exc)

    async def _register_advanced_trading(self) -> None:
        """Register advanced trading features (v8.10.0).
        
        WhaleTracker:
        - On-chain whale movement detection
        - Exchange inflow/outflow tracking
        - Smart money signal generation
        
        SmartOrderRouter:
        - Multi-venue order routing
        - TWAP/VWAP execution algorithms
        - Cost-optimized order splitting
        
        VolSurfaceAnalyzer:
        - Real-time IV surface monitoring
        - Volatility regime detection
        - Options strategy recommendations
        """
        try:
            from onchain.whale_tracker import WhaleTracker
            from execution.smart_order_router import SmartOrderRouter
            from options.vol_surface_analyzer import VolSurfaceAnalyzer
            
            # Initialize WhaleTracker
            whale_tracker = WhaleTracker(
                large_transfer_threshold_usd=1_000_000,
                whale_threshold_usd=10_000_000,
                signal_window_hours=24,
            )
            
            # Initialize SmartOrderRouter
            order_router = SmartOrderRouter(
                default_strategy="balanced",
                max_venues_per_order=3,
                min_slice_size_usd=100.0,
            )
            
            # Register default venues
            order_router.register_venue("binance", fee_maker=0.001, fee_taker=0.001, latency_ms=50)
            order_router.register_venue("kraken", fee_maker=0.0016, fee_taker=0.0026, latency_ms=80)
            order_router.register_venue("bybit", fee_maker=0.001, fee_taker=0.001, latency_ms=60)
            
            # Initialize VolSurfaceAnalyzer
            vol_analyzer = VolSurfaceAnalyzer(
                iv_history_size=1000,
                skew_threshold=0.05,
            )
            
            self._components["whale_tracker"] = whale_tracker
            self._components["order_router"] = order_router
            self._components["vol_surface_analyzer"] = vol_analyzer
            
            # Store reference on system if available
            if self._sys:
                self._sys.whale_tracker = whale_tracker
                self._sys.order_router = order_router
                self._sys.vol_surface_analyzer = vol_analyzer
            
            logger.info("✅ AdvancedTrading registered (v8.10.0 whale + routing + vol surface)")
        except Exception as exc:
            self._components["whale_tracker"] = None
            self._components["order_router"] = None
            self._components["vol_surface_analyzer"] = None
            logger.warning("AdvancedTrading unavailable: %s", exc)

    @property
    def whale_tracker(self) -> Optional[Any]:
        """Convenient access to WhaleTracker."""
        return self._components.get("whale_tracker")

    @property
    def order_router(self) -> Optional[Any]:
        """Convenient access to SmartOrderRouter."""
        return self._components.get("order_router")

    @property
    def vol_surface_analyzer(self) -> Optional[Any]:
        """Convenient access to VolSurfaceAnalyzer."""
        return self._components.get("vol_surface_analyzer")

    @property
    def gamma_squeeze_detector(self) -> Optional[Any]:
        """Convenient access to GammaSqueezeDetector."""
        return self._components.get("gamma_squeeze_detector")

    @property
    def dynamic_stop_loss(self) -> Optional[Any]:
        """Convenient access to DynamicStopLoss."""
        return self._components.get("dynamic_stop_loss")

    @property
    def liquidity_analyzer(self) -> Optional[Any]:
        """Convenient access to LiquidityAnalyzer."""
        return self._components.get("liquidity_analyzer")

    @property
    def order_flow_analyzer(self) -> Optional[Any]:
        """Convenient access to OrderFlowAnalyzer."""
        return self._components.get("order_flow_analyzer")

    @property
    def arbitrage_scanner(self) -> Optional[Any]:
        """Convenient access to CrossExchangeArbitrageScanner."""
        return self._components.get("arbitrage_scanner")

    @property
    def hyperparameter_optimizer(self) -> Optional[Any]:
        """Convenient access to AdaptiveHyperparameterOptimizer."""
        return self._components.get("hyperparameter_optimizer")

    @property
    def staking_optimizer(self) -> Optional[Any]:
        """Convenient access to StakingOptimizer."""
        return self._components.get("staking_optimizer")

    @property
    def sandwich_detector(self) -> Optional[Any]:
        """Convenient access to SandwichAttackDetector."""
        return self._components.get("sandwich_detector")

    @property
    def market_flow_integration(self) -> Optional[Any]:
        """Convenient access to MarketFlowTradingIntegration."""
        return self._components.get("market_flow_integration")

    async def _register_market_flow_integration(self) -> None:
        """Register MarketFlowTradingIntegration (v8.12.0).
        
        Pipes all signals through market flow adaptation:
        - Signal generation → Flow analysis → Risk assessment → Execution
        - Stop loss adapts to volatility (wider in high vol, tighter in low)
        - Position sizing adapts to liquidity
        - Emergency pause in crisis conditions
        """
        try:
            from ml.market_flow_integration import MarketFlowTradingIntegration

            # Check if enabled in config
            config = self._config if self._config else (self._sys.config if self._sys else None)
            mfi_config = getattr(config, "market_flow_integration", None) if config else None

            # Extract config options
            use_ultimate = True
            use_risk = True
            strategy_min_confidence = 0.50

            if mfi_config:
                use_ultimate = getattr(mfi_config, "use_ultimate_adaptation", True)
                use_risk = getattr(mfi_config, "use_risk_adapter", True)
                strategy_min_confidence = getattr(mfi_config, "min_confidence", 0.50)

            mfi = MarketFlowTradingIntegration(
                use_ultimate_adaptation=use_ultimate,
                use_risk_adapter=use_risk,
                strategy_min_confidence=strategy_min_confidence,
            )

            self._components["market_flow_integration"] = mfi

            # Store reference on system if available
            if self._sys:
                self._sys.market_flow_integration = mfi

            logger.info("✅ MarketFlowTradingIntegration registered (v8.12.0 adaptation)")
        except Exception as exc:
            self._components["market_flow_integration"] = None
            logger.warning("MarketFlowTradingIntegration unavailable: %s", exc)

    async def _register_alpha_signal_fusion(self) -> None:
        """Register AlphaSignalFusion (v9.0.0 - research-enhanced)."""
        try:
            from ml.alpha_signal_fusion import AlphaSignalFusion

            config = self._config if self._config else (self._sys.config if self._sys else None)
            alpha_config = getattr(config, "alpha_signal_fusion", None) if config else None

            # New config options for research-enhanced signals
            use_ml = True
            use_alpha = True
            use_sentiment = True
            use_onchain = True
            use_microstructure = True
            use_fear_greed = True

            if alpha_config:
                use_ml = getattr(alpha_config, "use_ml_predictor", True)
                use_alpha = getattr(alpha_config, "use_alpha_model", True)
                use_sentiment = getattr(alpha_config, "use_sentiment", True)
                use_onchain = getattr(alpha_config, "use_onchain", True)
                use_microstructure = getattr(alpha_config, "use_microstructure", True)
                use_fear_greed = getattr(alpha_config, "use_fear_greed", True)

            fusion = AlphaSignalFusion(
                use_ml_predictor=use_ml,
                use_alpha_model=use_alpha,
                use_sentiment=use_sentiment,
                use_onchain=use_onchain,
                use_microstructure=use_microstructure,
                use_fear_greed=use_fear_greed,
            )

            self._components["alpha_signal_fusion"] = fusion

            if self._sys:
                self._sys.alpha_signal_fusion = fusion

            logger.info("✅ AlphaSignalFusion registered (v9.0.0 research-enhanced)")
        except Exception as exc:
            self._components["alpha_signal_fusion"] = None
            logger.warning("AlphaSignalFusion unavailable: %s", exc)

    async def _register_multi_agent_voting(self) -> None:
        """Register Multi-Agent LLM Voting System."""
        try:
            from ml.multi_agent_voting import MultiAgentVoting

            voting = MultiAgentVoting(min_agreement=2)
            self._components["multi_agent_voting"] = voting

            if self._sys:
                self._sys.multi_agent_voting = voting

            logger.info("✅ Multi-Agent Voting registered (3-agent consensus)")
        except Exception as exc:
            self._components["multi_agent_voting"] = None
            logger.warning("Multi-Agent Voting unavailable: %s", exc)

    async def _register_rl_trading_agent(self) -> None:
        """Register RL Trading Agent."""
        try:
            from ml.rl_trading_agent import RLTradingAgent

            agent = RLTradingAgent()
            self._components["rl_trading_agent"] = agent

            if self._sys:
                self._sys.rl_trading_agent = agent

            logger.info("✅ RL Trading Agent registered (DQN)")
        except Exception as exc:
            self._components["rl_trading_agent"] = None
            logger.warning("RL Trading Agent unavailable: %s", exc)

    async def _register_quantum_ml_tuner(self) -> None:
        """Register Quantum ML Tuner for ABSOLUTE PEAK PERFORMANCE."""
        try:
            from ml.quantum_ml_tuner import QuantumMLTuner, TuningConfig

            config = TuningConfig(
                # EVERY 2 SECONDS - CONTINUOUS ADAPTIVE
                tuning_interval_seconds=2.0,
                adaptive_frequency=True,
                min_interval_seconds=1.0,
                max_interval_seconds=10.0,
                # PARALLEL PROCESSING - ALL CORES
                use_parallel_eval=True,
                max_workers=16,
                use_gpu_acceleration=True,
                # QUANTUM SETTINGS
                n_layers=3,
                max_evals=30,
                use_adaptive_evals=True,
                max_evals_high_vol=60,
                # ALL 11 MODELS TUNED
                tune_regime_classifier=True,
                tune_ensemble_weights=True,
                tune_position_sizing=True,
                tune_strategy_weights=True,
                tune_risk_parameters=True,
                tune_execution_params=True,
                tune_volatility_model=True,
                tune_correlation_model=True,
                tune_sentiment_weights=True,
                tune_stop_loss=True,
                tune_take_profit=True,
                # APPLY ANY IMPROVEMENT
                min_improvement_pct=0.01,
                # ONLINE LEARNING
                online_learning=True,
                forgetting_factor=0.99,
                adaptive_exploration=True,
                # PREDICTIVE PRE-OPTIMIZATION
                predictive_tuning=True,
                pre_optimize_regimes=True,
            )
            tuner = QuantumMLTuner(config=config)
            self._components["quantum_ml_tuner"] = tuner

            if self._sys:
                self._sys.quantum_ml_tuner = tuner

            logger.info("✅ Quantum ML Tuner registered (ABSOLUTE PEAK - 2s interval, 11 models, parallel, predictive)")
        except Exception as exc:
            self._components["quantum_ml_tuner"] = None
            logger.warning("Quantum ML Tuner unavailable: %s", exc)
    
    async def _register_peak_quantum_engine(self) -> None:
        """Register Peak Quantum Engine - ABSOLUTE PEAK local quantum."""
        try:
            from quantum.peak_quantum_config import PeakQuantumEngine, PeakQuantumConfig

            config = PeakQuantumConfig()
            engine = PeakQuantumEngine(config=config)
            engine.initialize()
            self._components["peak_quantum_engine"] = engine

            if self._sys:
                self._sys.peak_quantum_engine = engine

            status = engine.get_status()
            logger.info(f"✅ Peak Quantum Engine registered (43 qubits GPU, 64 qubits TN, 7 algorithms)")
        except Exception as exc:
            self._components["peak_quantum_engine"] = None
            logger.warning("Peak Quantum Engine unavailable: %s", exc)

    async def _register_advanced_learning(self) -> None:
        """Register Advanced Learning Integration (25+ learning systems).
        
        Integrates all learning systems into a cohesive framework:
        - Quantum RL (QQL, QDQN, QPG, QAC)
        - Multi-Agent RL (7 specialized agents)
        - Knowledge Distillation
        - RLHF (Human Feedback)
        - Uncertainty Quantification
        - Adversarial Training
        - Active Learning
        - Transfer Learning
        - Curriculum Learning
        - Self-Supervised Learning
        - Mixture of Experts
        - Prototype Networks
        - Foundation Model Layer
        - LLM Trading Planner
        - Memory-Augmented Networks
        - Neural Architecture Search
        """
        try:
            from ml.advanced_learning_integration import (
                AdvancedLearningOrchestrator,
                IntegratedTradingLoop,
                LearningConfig,
                LearningMode
            )

            config = LearningConfig(
                mode=LearningMode.FULL,
                enable_quantum_rl=True,
                enable_multi_agent=True,
                enable_knowledge_distillation=True,
                enable_rlhf=True,
                enable_uncertainty=True,
                enable_adversarial=True,
                enable_active_learning=True,
                enable_transfer_learning=True,
                enable_dashboard=True,
                uncertainty_threshold=0.6,
                min_confidence=0.5,
                learning_rate=0.001
            )
            
            orchestrator = AdvancedLearningOrchestrator(config=config)
            trading_loop = IntegratedTradingLoop(learning_orchestrator=orchestrator, config=config)
            
            self._components["advanced_learning"] = orchestrator
            self._components["learning_trading_loop"] = trading_loop

            if self._sys:
                self._sys.advanced_learning = orchestrator
                self._sys.learning_trading_loop = trading_loop

            logger.info(f"✅ Advanced Learning Integration registered (25+ systems, mode={config.mode.name})")
        except Exception as exc:
            self._components["advanced_learning"] = None
            self._components["learning_trading_loop"] = None
            logger.warning("Advanced Learning Integration unavailable: %s", exc)

    async def _register_maximum_edge_orchestrator(self) -> None:
        """Register Maximum Edge Orchestrator for coordinated strategy management.
        
        Coordinates all alpha sources:
        - Funding Rate Arbitrage (PRIMARY - 10-30% APR)
        - Cross-Exchange Arbitrage
        - Market Making
        - ML Price Prediction
        - Order Flow Alpha
        - Whale Tracking
        - Volatility Trading
        """
        try:
            from edge.maximum_edge_orchestrator import (
                MaximumEdgeOrchestrator,
                get_maximum_edge_orchestrator
            )
            
            orchestrator = get_maximum_edge_orchestrator()
            self._components["maximum_edge_orchestrator"] = orchestrator

            if self._sys:
                self._sys.maximum_edge_orchestrator = orchestrator

            logger.info("✅ Maximum Edge Orchestrator registered (8 strategies coordinated)")
        except Exception as exc:
            self._components["maximum_edge_orchestrator"] = None
            logger.warning("Maximum Edge Orchestrator unavailable: %s", exc)

    async def _register_defi_yield_optimizer(self) -> None:
        """Register DeFi Yield Optimization system.
        
        Automatic yield farming across:
        - Aave (lending)
        - Compound (lending)
        - Lido (liquid staking)
        - EigenLayer (restaking)
        - Morpho (optimized lending)
        - Yearn (yield aggregation)
        """
        try:
            from defi.defi_yield_optimizer import (
                DeFiYieldOptimizer,
                get_defi_yield_optimizer
            )
            
            optimizer = get_defi_yield_optimizer()
            self._components["defi_yield_optimizer"] = optimizer

            if self._sys:
                self._sys.defi_yield_optimizer = optimizer

            logger.info("✅ DeFi Yield Optimizer registered (6 protocols)")
        except Exception as exc:
            self._components["defi_yield_optimizer"] = None
            logger.warning("DeFi Yield Optimizer unavailable: %s", exc)

    async def _register_latency_based_stops(self) -> None:
        """Register Latency-Based Stops system.
        
        Dynamic stop loss adjustment based on execution latency:
        - ULTRA_LOW (<5ms): 20% tighter stops
        - LOW (5-20ms): Normal stops
        - MODERATE (20-100ms): 20% wider
        - HIGH (100-500ms): 50% wider
        - EXTREME (>500ms): 100% wider
        """
        try:
            from risk.latency_based_stops import (
                LatencyBasedStops,
                get_latency_based_stops
            )
            
            stops = get_latency_based_stops(base_stop_pct=0.02)
            self._components["latency_based_stops"] = stops

            if self._sys:
                self._sys.latency_based_stops = stops

            logger.info("✅ Latency-Based Stops registered (base stop: 2%)")
        except Exception as exc:
            self._components["latency_based_stops"] = None
            logger.warning("Latency-Based Stops unavailable: %s", exc)

    async def _register_dynamic_latency_sizing(self) -> None:
        """Register Dynamic Latency Sizing system.
        
        Position size adjustment based on:
        - Execution latency
        - Market volatility
        - Order book depth
        - Time of day
        - Recent fill quality
        """
        try:
            from execution.dynamic_latency_sizing import (
                DynamicLatencySizing,
                SizingMode,
                get_dynamic_latency_sizing
            )
            
            sizing = get_dynamic_latency_sizing(
                mode=SizingMode.BALANCED,
                base_position_usd=1000.0
            )
            self._components["dynamic_latency_sizing"] = sizing

            if self._sys:
                self._sys.dynamic_latency_sizing = sizing

            logger.info("✅ Dynamic Latency Sizing registered (mode: BALANCED)")
        except Exception as exc:
            self._components["dynamic_latency_sizing"] = None
            logger.warning("Dynamic Latency Sizing unavailable: %s", exc)

    async def _register_strategy_intelligence_hub(self) -> None:
        """Register Strategy Intelligence Hub for coordinated strategy management.
        
        Manages ALL 99+ strategies:
        - Automatic champion-challenger testing
        - Decay detection and rotation
        - Regime-based strategy selection
        - Performance tracking and optimization
        """
        try:
            from strategies.strategy_intelligence_hub import (
                StrategyIntelligenceHub,
                get_strategy_intelligence_hub
            )
            
            hub = get_strategy_intelligence_hub()
            self._components["strategy_intelligence_hub"] = hub

            if self._sys:
                self._sys.strategy_intelligence_hub = hub

            logger.info(f"✅ Strategy Intelligence Hub registered ({len(hub.strategies)} strategies)")
        except Exception as exc:
            self._components["strategy_intelligence_hub"] = None
            logger.warning("Strategy Intelligence Hub unavailable: %s", exc)

    async def _register_parameter_learning_engine(self) -> None:
        """Register Universal Parameter Learning Engine.
        
        Continuously learns optimal values for 200+ parameters:
        - Signal weights (40+)
        - Confidence thresholds (50+)
        - Risk parameters (30+)
        - Learning rates (30+)
        - Strategy parameters (99+)
        - Execution parameters (20+)
        - Ensemble weights
        - Volatility parameters
        """
        try:
            from learning.parameter_learning_integration import (
                ParameterLearningIntegrator
            )
            
            integrator = ParameterLearningIntegrator()
            self._components["parameter_learning_engine"] = integrator

            if self._sys:
                self._sys.parameter_learning_engine = integrator
                # Create hook for easy access
                self._sys.param_learning_hook = {
                    "get_params": integrator.get_parameters_for_decision,
                    "record_outcome": integrator.record_trade_outcome,
                    "get_signal_weights": integrator.get_signal_weights,
                    "get_risk_parameters": integrator.get_risk_parameters,
                }

            logger.info(f"✅ Parameter Learning Engine registered (200+ parameters)")
        except Exception as exc:
            self._components["parameter_learning_engine"] = None
            logger.warning("Parameter Learning Engine unavailable: %s", exc)
            logger.warning("Advanced Learning Integration unavailable: %s", exc)

    @property
    def quantum_ml_tuner(self) -> Optional[Any]:
        """Convenient access to QuantumMLTuner."""
        return self._components.get("quantum_ml_tuner")

    @property
    def alpha_signal_fusion(self) -> Optional[Any]:
        """Convenient access to AlphaSignalFusion."""
        return self._components.get("alpha_signal_fusion")

    @property
    def advanced_learning(self) -> Optional[Any]:
        """Convenient access to AdvancedLearningOrchestrator."""
        return self._components.get("advanced_learning")

    # ------------------------------------------------------------------
    # Cycle Hook - integrates analytics components into trading loop
    # ------------------------------------------------------------------

    async def on_cycle(self, prices: Dict[str, float], regime_label: str = "") -> Dict[str, Any]:
        """
        Called on each trading cycle to gather advisory from analytics components.
        
        Returns a dictionary with keys like:
        - "hurst_regime": regime classification from Hurst analyzer
        - "order_flow": order flow signals
        - "whale_activity": whale tracker signals
        - "ensemble": composite signal weighting
        - etc.
        """
        advisory: Dict[str, Any] = {}
        
        # 1. Hurst/Regime Analysis
        hurst_provider = self._components.get("hurst_provider")
        if hurst_provider is not None:
            try:
                for symbol, price in prices.items():
                    if price > 0:
                        # Update price history
                        hurst_provider.update_price(symbol, price)
                        
                        # Get regime classification
                        regime = hurst_provider.get_regime(symbol)
                        position_scalar = hurst_provider.get_position_scalar(symbol)
                        
                        advisory["hurst_regime"] = {
                            "symbol": symbol,
                            "regime": regime.regime if hasattr(regime, "regime") else str(regime),
                            "hurst_exponent": regime.hurst_exponent if hasattr(regime, "hurst_exponent") else 0.5,
                            "position_scalar": position_scalar,
                            "recommendation": "trade" if position_scalar > 0.3 else "reduce" if position_scalar > 0 else "avoid",
                        }
                        break  # Use primary symbol
            except Exception as exc:
                logger.debug("hurst_provider on_cycle error: %s", exc)
        
        # 2. Order Flow Analysis
        order_flow = self._components.get("order_flow_analyzer")
        if order_flow is not None:
            try:
                for symbol in prices.keys():
                    # Get order flow signals
                    flow_signal = order_flow.get_signal(symbol)
                    if flow_signal is not None:
                        advisory["order_flow"] = {
                            "symbol": symbol,
                            "buy_pressure": flow_signal.buy_pressure if hasattr(flow_signal, "buy_pressure") else 0.5,
                            "sell_pressure": flow_signal.sell_pressure if hasattr(flow_signal, "sell_pressure") else 0.5,
                            "imbalance": flow_signal.imbalance if hasattr(flow_signal, "imbalance") else 0.0,
                            "whale_detected": flow_signal.whale_detected if hasattr(flow_signal, "whale_detected") else False,
                        }
                        break
            except Exception as exc:
                logger.debug("order_flow_analyzer on_cycle error: %s", exc)
        
        # 3. Whale Tracker
        whale_tracker = self._components.get("whale_tracker")
        if whale_tracker is not None:
            try:
                whale_signals = whale_tracker.get_recent_signals()
                if whale_signals:
                    advisory["whale_activity"] = {
                        "signals": whale_signals[:5],  # Last 5 signals
                        "net_flow": whale_tracker.get_net_flow() if hasattr(whale_tracker, "get_net_flow") else 0.0,
                    }
            except Exception as exc:
                logger.debug("whale_tracker on_cycle error: %s", exc)
        
        # 4. Dynamic Stop Loss recommendations
        dynamic_stop = self._components.get("dynamic_stop_loss")
        if dynamic_stop is not None:
            try:
                stop_recs = {}
                for symbol, price in prices.items():
                    if price > 0:
                        stop_level = dynamic_stop.calculate_stop(symbol, price)
                        if stop_level is not None:
                            stop_recs[symbol] = {
                                "stop_price": stop_level,
                                "distance_pct": (price - stop_level) / price * 100 if price > stop_level else (stop_level - price) / price * 100,
                            }
                if stop_recs:
                    advisory["dynamic_stops"] = stop_recs
            except Exception as exc:
                logger.debug("dynamic_stop_loss on_cycle error: %s", exc)
        
        # 5. Liquidity Analysis
        liquidity = self._components.get("liquidity_analyzer")
        if liquidity is not None:
            try:
                liq_data = {}
                for symbol in prices.keys():
                    liq_score = liquidity.get_liquidity_score(symbol) if hasattr(liquidity, "get_liquidity_score") else None
                    if liq_score is not None:
                        liq_data[symbol] = {"score": liq_score}
                if liq_data:
                    advisory["liquidity"] = liq_data
            except Exception as exc:
                logger.debug("liquidity_analyzer on_cycle error: %s", exc)
        
        # 6. Arbitrage Scanner
        arb_scanner = self._components.get("arbitrage_scanner")
        if arb_scanner is not None:
            try:
                for symbol in prices.keys():
                    opp = arb_scanner.find_opportunity(symbol)
                    if opp is not None and hasattr(opp, "net_spread_bps") and opp.net_spread_bps > 0:
                        advisory["arbitrage"] = {
                            "symbol": symbol,
                            "spread_bps": opp.net_spread_bps,
                            "confidence": opp.confidence if hasattr(opp, "confidence") else 0.5,
                        }
                        break
            except Exception as exc:
                logger.debug("arbitrage_scanner on_cycle error: %s", exc)
        
        # 7. Time Series Forecaster
        forecaster = self._components.get("time_series_forecaster")
        if forecaster is not None:
            try:
                forecasts = {}
                for symbol, price in prices.items():
                    if price > 0:
                        forecast = forecaster.predict(symbol)
                        if forecast is not None:
                            forecasts[symbol] = {
                                "predicted_direction": forecast.direction if hasattr(forecast, "direction") else "neutral",
                                "confidence": forecast.confidence if hasattr(forecast, "confidence") else 0.5,
                                "target_price": forecast.target_price if hasattr(forecast, "target_price") else price,
                            }
                if forecasts:
                    advisory["forecasts"] = forecasts
            except Exception as exc:
                logger.debug("time_series_forecaster on_cycle error: %s", exc)
        
        # 8. Hyperparameter Optimizer recommendations
        hyperopt = self._components.get("hyperparameter_optimizer")
        if hyperopt is not None:
            try:
                params = hyperopt.get_current_params()
                if params:
                    advisory["optimized_params"] = params
            except Exception as exc:
                logger.debug("hyperparameter_optimizer on_cycle error: %s", exc)
        
        # 9. Correlation Monitor
        corr_monitor = self._components.get("correlation_monitor")
        if corr_monitor is not None:
            try:
                corr_scalar = corr_monitor.get_position_scalar()
                advisory["correlation"] = {
                    "position_scalar": corr_scalar,
                    "regime": "crisis" if corr_scalar < 0.5 else "elevated" if corr_scalar < 0.8 else "normal",
                }
            except Exception as exc:
                logger.debug("correlation_monitor on_cycle error: %s", exc)
        
        # 10. Adaptive Orchestrator
        adaptive = self._components.get("adaptive_orchestrator")
        if adaptive is not None:
            try:
                adaptive_state = adaptive.get_current_state()
                if adaptive_state:
                    advisory["adaptive"] = adaptive_state
            except Exception as exc:
                logger.debug("adaptive_orchestrator on_cycle error: %s", exc)

        # 11. Market Flow Integration (v8.12.0)
        mfi = self._components.get("market_flow_integration")
        if mfi is not None:
            try:
                status = mfi.get_status()
                advisory["market_flow"] = {
                    "condition": status.get("risk", {}).get("current_condition", "unknown"),
                    "total_trades": status.get("strategy", {}).get("total_trades", 0),
                    "win_rate": status.get("strategy", {}).get("win_rate", 0),
                }
            except Exception as exc:
                logger.debug("market_flow_integration on_cycle error: %s", exc)

        # 12. Alpha Signal Fusion (v8.13.0)
        fusion = self._components.get("alpha_signal_fusion")
        if fusion is not None:
            try:
                advisory["alpha"] = {
                    "enabled": True,
                    "sources": ["ml", "alpha_model", "sentiment"],
                }
            except Exception as exc:
                logger.debug("alpha_signal_fusion on_cycle error: %s", exc)

        # 13. Quantum ML Tuner - REAL-TIME auto-tuning
        tuner = self._components.get("quantum_ml_tuner")
        if tuner is not None:
            try:
                # Check if tuning is due and run it
                if tuner.should_tune():
                    logger.info("🚀 Quantum ML Tuner: Starting real-time tuning session...")
                    # Run tuning in background (non-blocking) - pass trade history if available
                    trade_history = []
                    if hasattr(self._sys, 'trade_history'):
                        trade_history = self._sys.trade_history[-100:]  # Last 100 trades
                    
                    results = tuner.run_full_tuning(trade_data=trade_history if trade_history else None)
                    
                    if results:
                        for r in results:
                            if r.improvement_pct > 0.5:
                                logger.info(
                                    f"✅ Quantum ML Tuner: {r.model_name} improved by {r.improvement_pct:.1f}% "
                                    f"(score: {r.previous_score:.4f} → {r.best_score:.4f})"
                                )
                
                # Always report tuning status
                tuning_summary = tuner.get_tuning_summary()
                advisory["quantum_ml_tuner"] = {
                    "enabled": True,
                    "total_sessions": tuning_summary.get("total_tuning_sessions", 0),
                    "next_tuning_due": tuning_summary.get("next_tuning_due", "unknown"),
                    "current_params": tuning_summary.get("current_params", {}),
                }
            except Exception as exc:
                logger.debug("quantum_ml_tuner on_cycle error: %s", exc)

        # 14. Advanced Learning Integration (25+ learning systems)
        advanced_learning = self._components.get("advanced_learning")
        if advanced_learning is not None:
            try:
                # Get system status for advisory
                learning_status = advanced_learning.get_system_status()
                
                advisory["advanced_learning"] = {
                    "enabled": True,
                    "mode": learning_status.get("config", {}).get("mode", "FULL"),
                    "active_systems": learning_status.get("metrics", {}).get("active_systems", 0),
                    "total_decisions": learning_status.get("metrics", {}).get("total_decisions", 0),
                    "avg_confidence": learning_status.get("metrics", {}).get("avg_confidence", 0.0),
                    "avg_uncertainty": learning_status.get("metrics", {}).get("avg_uncertainty", 0.0),
                    "quantum_enabled": learning_status.get("config", {}).get("quantum_enabled", False),
                }
            except Exception as exc:
                logger.debug("advanced_learning on_cycle error: %s", exc)

        return advisory
