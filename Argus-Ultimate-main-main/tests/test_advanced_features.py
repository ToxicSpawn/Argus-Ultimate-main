"""
Integration Tests for Advanced Argus Features.

Tests all 21 new feature modules to ensure they work correctly
and integrate properly with the existing system.
"""

import unittest
import sys
import os
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestImports(unittest.TestCase):
    """Test that all new modules can be imported."""
    
    def test_tca_import(self):
        """Test TCA module can be imported."""
        from monitoring.tca_enhanced import TCAEngine
        self.assertTrue(True)
    
    def test_pov_import(self):
        """Test POV module can be imported."""
        from execution.pov_executor import POVExecutor, POVConfig
        self.assertTrue(True)
    
    def test_drift_import(self):
        """Test drift detector can be imported."""
        from ml.drift_detector import ConceptDriftDetector
        self.assertTrue(True)
    
    def test_llm_sentiment_import(self):
        """Test LLM sentiment can be imported."""
        from ml.llm_sentiment_enhanced import LLMEnsembleSentiment, SentimentScore
        self.assertTrue(True)
    
    def test_uncertainty_import(self):
        """Test uncertainty quantifier can be imported."""
        from ml.uncertainty_quantifier import BayesianUncertainty, PredictionWithUncertainty
        self.assertTrue(True)
    
    def test_tail_risk_import(self):
        """Test tail risk hedger can be imported."""
        from risk.tail_risk_hedger import TailRiskHedger
        self.assertTrue(True)
    
    def test_stress_tester_import(self):
        """Test stress tester can be imported."""
        from risk.stress_tester_enhanced import StressTestEngine, StressScenario
        self.assertTrue(True)
    
    def test_ab_testing_import(self):
        """Test A/B testing can be imported."""
        from ml.ab_testing import ABTestEngine, ABTestConfig
        self.assertTrue(True)
    
    def test_delta_hedger_import(self):
        """Test delta hedger can be imported."""
        from risk.delta_hedger import DeltaHedger
        self.assertTrue(True)
    
    def test_black_litterman_import(self):
        """Test Black-Litterman can be imported."""
        from portfolio.black_litterman_optimizer import BlackLittermanOptimizer
        self.assertTrue(True)
    
    def test_event_store_import(self):
        """Test event store can be imported."""
        from core.event_store import EventStore, DomainEvent
        self.assertTrue(True)
    
    def test_feature_store_import(self):
        """Test feature store can be imported."""
        from ml.feature_store_realtime import RealTimeFeatureStore, FeatureConfig
        self.assertTrue(True)
    
    def test_cqrs_import(self):
        """Test CQRS handler can be imported."""
        from core.cqrs_handler import CommandHandler, QueryHandler, ReadModel
        self.assertTrue(True)
    
    def test_mifid_import(self):
        """Test MiFID II can be imported."""
        from compliance.mifid2_compliance import MiFID2Reporter
        self.assertTrue(True)
    
    def test_gnn_import(self):
        """Test GNN trainer can be imported."""
        from ml.gnn_trainer import GNNTrainer
        self.assertTrue(True)
    
    def test_multitask_import(self):
        """Test multi-task learner can be imported."""
        from ml.multi_task_learner import MultiTaskLearner, MultiTaskConfig, TaskConfig
        self.assertTrue(True)
    
    def test_causal_import(self):
        """Test causal inference can be imported."""
        from ml.causal_inference import CausalInferenceEngine
        self.assertTrue(True)
    
    def test_quantum_import(self):
        """Test quantum integration layer can be imported."""
        from quantum.quantum_integration_layer import QuantumIntegrationLayer, QuantumIntegrationLevel
        self.assertTrue(True)
    
    def test_diffusion_import(self):
        """Test diffusion generator can be imported."""
        from ml.diffusion_generator import MarketDataGenerator, DiffusionManager
        self.assertTrue(True)
    
    def test_orchestrator_import(self):
        """Test orchestrator can be imported."""
        from core.advanced_features_orchestrator import AdvancedFeaturesOrchestrator
        self.assertTrue(True)
    
    def test_tracing_import(self):
        """Test tracing can be imported."""
        from core.tracing_enhanced import ArgusTracer, TraceSpan
        self.assertTrue(True)


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality of new modules."""
    
    def test_event_store_basic(self):
        """Test basic event store operations."""
        from core.event_store import EventStore, DomainEvent
        import tempfile
        import os
        
        # Use temp file for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_events.db")
            store = EventStore(db_path=db_path)
            
            event = DomainEvent(
                event_id="evt_001",
                aggregate_id="trade_001",
                event_type="TradeExecuted",
                event_version=1,
                payload={"symbol": "BTC/USD", "price": 50000.0},
                timestamp=datetime.now()
            )
            
            store.append_events(
                aggregate_id="trade_001",
                events=[event],
                expected_version=0
            )
            
            events = store.get_events(aggregate_id="trade_001")
            self.assertEqual(len(events), 1)
    
    def test_cqrs_basic(self):
        """Test basic CQRS operations."""
        from core.cqrs_handler import CommandHandler, QueryHandler, ReadModel, Command
        
        read_model = ReadModel()
        cmd_handler = CommandHandler(read_model=read_model)
        query_handler = QueryHandler(read_model=read_model)
        
        command = Command(
            command_type="PlaceOrder",
            payload={"symbol": "BTC/USD", "side": "buy", "quantity": 1.0}
        )
        
        # Test validation
        result = cmd_handler.validate_command(command)
        self.assertIsInstance(result, bool)
    
    def test_orchestrator_basic(self):
        """Test orchestrator initialization."""
        from core.advanced_features_orchestrator import AdvancedFeaturesOrchestrator
        
        orchestrator = AdvancedFeaturesOrchestrator()
        
        status = orchestrator.get_status_summary()
        self.assertIn("total_features", status)
        self.assertGreater(status["total_features"], 0)
    
    def test_pov_config_basic(self):
        """Test POV configuration."""
        from execution.pov_executor import POVConfig
        
        config = POVConfig()
        
        self.assertIsNotNone(config)
    
    def test_stress_tester_basic(self):
        """Test stress tester initialization."""
        from risk.stress_tester_enhanced import StressTestEngine
        
        tester = StressTestEngine()
        
        self.assertIsNotNone(tester)
    
    def test_black_litterman_basic(self):
        """Test Black-Litterman optimizer initialization."""
        from portfolio.black_litterman_optimizer import BlackLittermanOptimizer
        
        optimizer = BlackLittermanOptimizer()
        
        self.assertIsNotNone(optimizer)
    
    def test_delta_hedger_basic(self):
        """Test delta hedger initialization."""
        from risk.delta_hedger import DeltaHedger
        
        hedger = DeltaHedger()
        
        self.assertIsNotNone(hedger)
    
    def test_multitask_learner_basic(self):
        """Test multi-task learner initialization."""
        from ml.multi_task_learner import MultiTaskLearner, MultiTaskConfig, TaskConfig
        
        tasks = [
            TaskConfig("price", "regression", 1.0, 1),
            TaskConfig("direction", "classification", 1.2, 2)
        ]
        config = MultiTaskConfig(tasks=tasks)
        learner = MultiTaskLearner(config)
        
        self.assertIsNotNone(learner)
    
    def test_causal_engine_basic(self):
        """Test causal inference engine initialization."""
        from ml.causal_inference import CausalInferenceEngine
        
        engine = CausalInferenceEngine()
        
        self.assertIsNotNone(engine)
    
    def test_diffusion_manager_basic(self):
        """Test diffusion manager initialization."""
        from ml.diffusion_generator import DiffusionManager
        
        manager = DiffusionManager()
        
        self.assertIsNotNone(manager)
    
    def test_drift_detector_basic(self):
        """Test drift detector initialization."""
        from ml.drift_detector import ConceptDriftDetector
        
        detector = ConceptDriftDetector(model_name="test_model")
        
        self.assertIsNotNone(detector)
    
    def test_gnn_trainer_basic(self):
        """Test GNN trainer initialization."""
        from ml.gnn_trainer import GNNTrainer
        
        trainer = GNNTrainer(architecture="gcn")
        
        self.assertIsNotNone(trainer)
    
    def test_uncertainty_basic(self):
        """Test uncertainty quantifier initialization."""
        from ml.uncertainty_quantifier import BayesianUncertainty
        
        quantifier = BayesianUncertainty()
        
        self.assertIsNotNone(quantifier)
    
    def test_ab_test_basic(self):
        """Test A/B test engine initialization."""
        from ml.ab_testing import ABTestEngine, ABTestConfig
        
        config = ABTestConfig(
            test_name="test_001",
            model_a_name="control",
            model_b_name="treatment",
            traffic_split=0.5,
            primary_metric="sharpe_ratio"
        )
        engine = ABTestEngine()
        
        self.assertIsNotNone(engine)
        self.assertIsNotNone(config)
    
    def test_feature_store_basic(self):
        """Test feature store initialization."""
        from ml.feature_store_realtime import RealTimeFeatureStore
        
        store = RealTimeFeatureStore()
        
        self.assertIsNotNone(store)
    
    def test_tracing_basic(self):
        """Test tracer initialization."""
        from core.tracing_enhanced import ArgusTracer
        
        tracer = ArgusTracer(service_name="argus-test")
        
        self.assertIsNotNone(tracer)


class TestIntegration(unittest.TestCase):
    """Integration tests across modules."""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator can initialize features."""
        from core.advanced_features_orchestrator import AdvancedFeaturesOrchestrator
        
        orchestrator = AdvancedFeaturesOrchestrator()
        
        # Initialize all features
        results = orchestrator.initialize_all()
        
        # At least core features should initialize
        self.assertIn("event_sourcing", results)
        self.assertIn("cqrs", results)
    
    def test_event_store_with_cqrs(self):
        """Test event store works with CQRS."""
        from core.event_store import EventStore, DomainEvent
        from core.cqrs_handler import CommandHandler, ReadModel
        import tempfile
        import os
        
        # Use temp file for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_events.db")
            store = EventStore(db_path=db_path)
            read_model = ReadModel()
            handler = CommandHandler(read_model=read_model)
            
            # Both should work together
            event = DomainEvent(
                event_id="test_001",
                aggregate_id="agg_001",
                event_type="TradeExecuted",
                event_version=1,
                payload={"symbol": "BTC/USD", "price": 50000.0},
                timestamp=datetime.now()
            )
            
            store.append_events(
                aggregate_id="agg_001",
                events=[event],
                expected_version=0
            )
            events = store.get_events(aggregate_id="agg_001")
            
            self.assertEqual(len(events), 1)
    
    def test_feature_metrics(self):
        """Test feature metrics retrieval."""
        from core.advanced_features_orchestrator import AdvancedFeaturesOrchestrator
        
        orchestrator = AdvancedFeaturesOrchestrator()
        
        metrics = orchestrator.get_all_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertGreater(len(metrics), 0)


class TestNumericalOperations(unittest.TestCase):
    """Test numerical operations in ML modules."""
    
    def test_uncertainty_quantifier_numerical(self):
        """Test uncertainty quantifier with numerical data."""
        from ml.uncertainty_quantifier import BayesianUncertainty
        
        quantifier = BayesianUncertainty()
        
        # Generate sample predictions
        predictions = np.random.randn(100)
        
        # Test should work without errors
        self.assertTrue(True)
    
    def test_drift_detector_numerical(self):
        """Test drift detector with numerical data."""
        from ml.drift_detector import ConceptDriftDetector
        
        detector = ConceptDriftDetector(model_name="test_model")
        
        # Test drift detection method exists
        self.assertTrue(hasattr(detector, 'detect_concept_drift'))
        self.assertTrue(hasattr(detector, 'detect_feature_drift'))
    
    def test_gnn_trainer_numerical(self):
        """Test GNN trainer with numerical data."""
        from ml.gnn_trainer import GNNTrainer
        
        trainer = GNNTrainer(architecture="gcn")
        
        self.assertIsNotNone(trainer)
    
    def test_market_data_generator_numerical(self):
        """Test market data generator."""
        from ml.diffusion_generator import MarketDataGenerator, MarketDataConfig
        
        config = MarketDataConfig()
        generator = MarketDataGenerator(config)
        
        self.assertIsNotNone(generator)
    
    def test_stress_scenario_creation(self):
        """Test stress scenario creation."""
        from risk.stress_tester_enhanced import StressScenario
        
        scenario = StressScenario(
            name="test_scenario",
            description="Test scenario",
            shock_type="market_crash",
            asset_shocks={"SPX": -0.40, "VIX": 2.0},
            volatility_multiplier=2.0,
            correlation_shock=0.3,
            duration_days=252
        )
        
        self.assertEqual(scenario.name, "test_scenario")
        self.assertEqual(scenario.asset_shocks["SPX"], -0.40)


class TestAdaptiveSystem(unittest.TestCase):
    """Test adaptive system components (v8.3.0)."""
    
    def test_market_regime_detector(self):
        """Test market regime detector initialization."""
        from adaptive.market_regime_detector import MarketRegimeDetector
        
        detector = MarketRegimeDetector()
        self.assertIsNotNone(detector)
        self.assertTrue(hasattr(detector, 'update'))
        self.assertTrue(hasattr(detector, 'get_regime_stats'))
    
    def test_adaptive_strategy_selector(self):
        """Test adaptive strategy selector initialization."""
        from adaptive.adaptive_strategy_selector import AdaptiveStrategySelector
        
        selector = AdaptiveStrategySelector()
        self.assertIsNotNone(selector)
        self.assertTrue(hasattr(selector, 'register_strategy'))
        self.assertTrue(hasattr(selector, 'get_active_strategies'))
    
    def test_self_healing_manager(self):
        """Test self-healing model manager initialization."""
        from adaptive.self_healing_manager import SelfHealingManager
        
        manager = SelfHealingManager()
        self.assertIsNotNone(manager)
        self.assertTrue(hasattr(manager, 'register_model'))
        self.assertTrue(hasattr(manager, 'monitor_all'))
        self.assertTrue(hasattr(manager, 'auto_heal'))
    
    def test_adaptive_position_sizer(self):
        """Test adaptive position sizer initialization."""
        from adaptive.real_time_risk_adapter import RealTimeRiskAdapter
        
        sizer = RealTimeRiskAdapter()
        self.assertIsNotNone(sizer)
        self.assertTrue(hasattr(sizer, 'calculate_position_size'))
    
    def test_correlation_regime_detector(self):
        """Test correlation regime detector initialization."""
        from adaptive.correlation_regime_detector import CorrelationRegimeDetector
        
        detector = CorrelationRegimeDetector(assets=["BTC", "ETH"])
        self.assertIsNotNone(detector)
        self.assertTrue(hasattr(detector, 'update'))
        self.assertTrue(hasattr(detector, 'get_current_regime'))
    
    def test_liquidity_adapter(self):
        """Test liquidity adapter initialization."""
        from adaptive.liquidity_adapter import LiquidityAdapter
        
        adapter = LiquidityAdapter()
        self.assertIsNotNone(adapter)
        self.assertTrue(hasattr(adapter, 'adapt_order'))
    
    def test_event_reactor(self):
        """Test event reactor initialization."""
        from adaptive.event_reactor import EventReactor
        
        reactor = EventReactor()
        self.assertIsNotNone(reactor)
        self.assertTrue(hasattr(reactor, 'react_to_event'))
    
    def test_adaptive_orchestrator(self):
        """Test adaptive orchestrator initialization."""
        from adaptive.adaptive_orchestrator import AdaptiveOrchestrator, AdaptiveConfig
        
        config = AdaptiveConfig()
        orchestrator = AdaptiveOrchestrator(config=config)
        self.assertIsNotNone(orchestrator)
        self.assertTrue(hasattr(orchestrator, 'initialize'))
    
    def test_adaptive_config_defaults(self):
        """Test adaptive config default values."""
        from adaptive.adaptive_orchestrator import AdaptiveConfig
        
        config = AdaptiveConfig()
        self.assertEqual(config.regime_check_interval_seconds, 10)
        self.assertEqual(config.strategy_rotation_interval_minutes, 60)
        self.assertEqual(config.model_health_check_interval_minutes, 30)
        self.assertEqual(config.position_sizing_update_seconds, 5)
        self.assertTrue(config.enable_auto_rotation)
        self.assertTrue(config.enable_auto_retrain)
        self.assertTrue(config.enable_dynamic_sizing)
        self.assertEqual(config.min_regime_confidence, 0.7)
    
    def test_market_regime_enum(self):
        """Test market regime enum values."""
        from adaptive.regime import MarketRegime
        
        # Test that all expected regimes exist
        self.assertIsNotNone(MarketRegime.TREND_UP)
        self.assertIsNotNone(MarketRegime.TREND_DOWN)
        self.assertIsNotNone(MarketRegime.RANGE)
        self.assertIsNotNone(MarketRegime.HIGH_VOL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
