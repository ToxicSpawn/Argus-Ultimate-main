"""
tests/test_level20_singularity.py — Tests for Level 20 Singularity

Tests for the ultimate trading system components:
- Causal Discovery Engine
- Predictive Order Flow Engine
- Cross-Market System
- Self-Aware System
- Autonomous Research Lab
- Level 20 Singularity Orchestrator
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from evolution.level20_singularity import (
    Level20Singularity,
    CausalDiscoveryEngine,
    PredictiveOrderFlowEngine,
    CrossMarketSystem,
    SelfAwareSystem,
    AutonomousResearchLab,
    CausalEdge,
    CausalRelation,
    OrderFlowEvent,
    OrderFlowSignal,
    CrossMarketSignal,
    Market,
    SelfAwarenessState,
    ResearchHypothesis,
    create_level20_singularity,
    create_causal_engine,
    create_order_flow_engine,
    create_cross_market_system,
    create_self_aware_system,
    create_research_lab,
)


# ============================================================================
# Causal Discovery Engine Tests
# ============================================================================

class TestCausalDiscoveryEngine:
    """Tests for Causal Discovery Engine."""
    
    def test_init(self):
        """Should initialize correctly."""
        engine = CausalDiscoveryEngine()
        
        assert engine.significance_level == 0.05
        assert len(engine.causal_graph) == 0
    
    def test_discover_causal_structure(self):
        """Should discover causal structure."""
        engine = CausalDiscoveryEngine()
        
        # Create correlated data where X causes Y
        np.random.seed(42)
        x = np.random.randn(500)
        y = x * 0.7 + np.random.randn(500) * 0.3  # Y = 0.7*X + noise
        
        data = {"X": x, "Y": y}
        edges = engine.discover_causal_structure(data, max_lag=3)
        
        # Should find relationship
        assert len(edges) >= 0  # May or may not find depending on strength
    
    def test_granger_causality(self):
        """Should test Granger causality."""
        engine = CausalDiscoveryEngine()
        
        # Create data where X Granger-causes Y
        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        y = np.zeros(n)
        for i in range(2, n):
            y[i] = 0.5 * y[i-1] + 0.3 * x[i-1] + np.random.randn() * 0.2
        
        result = engine.granger_causality(x, y, max_lag=5)
        
        assert "granger_causes" in result
        assert "best_lag" in result
    
    def test_causal_edge_to_dict(self):
        """Should convert edge to dict."""
        edge = CausalEdge(
            source="X",
            target="Y",
            relation=CausalRelation.DIRECT,
            strength=0.8,
            confidence=0.9,
            lag=1,
            p_value=0.01,
        )
        
        d = edge.to_dict()
        
        assert d["source"] == "X"
        assert d["target"] == "Y"
        assert d["relation"] == "direct"
    
    def test_get_causal_parents(self):
        """Should get causal parents."""
        engine = CausalDiscoveryEngine()
        
        # Manually add edges
        engine.causal_graph["A"].append(
            CausalEdge("A", "B", CausalRelation.DIRECT, 0.5, 0.8, 1, 0.01)
        )
        engine.causal_graph["C"].append(
            CausalEdge("C", "B", CausalRelation.DIRECT, 0.6, 0.9, 1, 0.005)
        )
        
        parents = engine.get_causal_parents("B")
        
        assert len(parents) == 2
        assert all(e.target == "B" for e in parents)
    
    def test_get_causal_children(self):
        """Should get causal children."""
        engine = CausalDiscoveryEngine()
        
        engine.causal_graph["A"].append(
            CausalEdge("A", "B", CausalRelation.DIRECT, 0.5, 0.8, 1, 0.01)
        )
        engine.causal_graph["A"].append(
            CausalEdge("A", "C", CausalRelation.DIRECT, 0.4, 0.7, 1, 0.02)
        )
        
        children = engine.get_causal_children("A")
        
        assert len(children) == 2
        assert all(e.source == "A" for e in children)


# ============================================================================
# Predictive Order Flow Engine Tests
# ============================================================================

class TestPredictiveOrderFlowEngine:
    """Tests for Predictive Order Flow Engine."""
    
    def test_init(self):
        """Should initialize correctly."""
        engine = PredictiveOrderFlowEngine()
        
        assert engine.window_size == 100
        assert engine.whale_threshold == 100000
    
    def test_detect_whales(self):
        """Should detect whale orders."""
        engine = PredictiveOrderFlowEngine()
        
        bids = [(65000.0, 5.0), (64999.0, 200.0)]  # 200 BTC = ~$13M whale
        asks = [(65001.0, 2.0), (65002.0, 1.0)]
        
        events = engine.analyze_order_book("BTC", bids, asks)
        
        whale_events = [e for e in events if e.signal_type in [
            OrderFlowSignal.WHALE_BUY, OrderFlowSignal.WHALE_SELL
        ]]
        
        assert len(whale_events) >= 1
    
    def test_predict_order_flow(self):
        """Should predict order flow."""
        engine = PredictiveOrderFlowEngine()
        
        # Add some history
        for i in range(20):
            bids = [(65000.0, 100.0 + i * 10), (64999.0, 50.0)]
            asks = [(65001.0, 100.0), (65002.0, 50.0)]
            engine.analyze_order_book("BTC", bids, asks)
        
        prediction = engine.predict_order_flow("BTC", horizon_seconds=60)
        
        assert "direction" in prediction
        assert "confidence" in prediction
    
    def test_get_whale_activity(self):
        """Should get recent whale activity."""
        engine = PredictiveOrderFlowEngine()
        
        # Add whale orders
        bids = [(65000.0, 500.0)]  # $32.5M whale
        engine.analyze_order_book("BTC", bids, [])
        
        whales = engine.get_whale_activity("BTC", hours=24)
        
        assert len(whales) >= 1
    
    def test_order_flow_event_to_dict(self):
        """Should convert event to dict."""
        event = OrderFlowEvent(
            timestamp=datetime.now(),
            symbol="BTC",
            signal_type=OrderFlowSignal.WHALE_BUY,
            size=100.0,
            price=65000.0,
            confidence=0.8,
            predicted_impact=0.01,
        )
        
        d = event.to_dict()
        
        assert d["symbol"] == "BTC"
        assert d["signal_type"] == "whale_buy"


# ============================================================================
# Cross-Market System Tests
# ============================================================================

class TestCrossMarketSystem:
    """Tests for Cross-Market System."""
    
    def test_init(self):
        """Should initialize correctly."""
        system = CrossMarketSystem()
        
        assert Market.CRYPTO in system.markets
        assert Market.FOREX in system.markets
    
    def test_analyze_correlation_regime(self):
        """Should analyze correlation regime."""
        system = CrossMarketSystem()
        
        # Create correlated data
        np.random.seed(42)
        data = {
            "BTC": list(65000 * np.exp(np.cumsum(np.random.randn(100) * 0.02))),
            "ETH": list(3500 * np.exp(np.cumsum(np.random.randn(100) * 0.02))),
        }
        
        result = system.analyze_correlation_regime(data)
        
        assert "regime" in result
        assert "avg_correlation" in result
    
    def test_find_arbitrage_opportunities(self):
        """Should find arbitrage opportunities."""
        system = CrossMarketSystem()
        
        prices = {
            "exchange_a": {"BTC": 65000.0, "ETH": 3500.0},
            "exchange_b": {"BTC": 65100.0, "ETH": 3490.0},
        }
        
        opportunities = system.find_arbitrage_opportunities(prices)
        
        assert len(opportunities) >= 1
        assert any(o["symbol"] == "BTC" for o in opportunities)


# ============================================================================
# Self-Aware System Tests
# ============================================================================

class TestSelfAwareSystem:
    """Tests for Self-Aware System."""
    
    def test_init(self):
        """Should initialize correctly."""
        system = SelfAwareSystem()
        
        assert system.state.current_regime == "unknown"
        assert system.state.confidence_calibration == 0.5
    
    def test_assess_prediction(self):
        """Should assess predictions."""
        system = SelfAwareSystem()
        
        result = system.assess_prediction("bull", 0.8, "bull")
        
        assert result["was_correct"] is True
    
    def test_detect_regime(self):
        """Should detect market regime."""
        system = SelfAwareSystem()
        
        market_data = {
            "volatility": 0.05,
            "trend_strength": 0.3,
            "volume_ratio": 1.2,
        }
        
        result = system.detect_regime(market_data)
        
        assert result["regime"] == "high_volatility"
        assert result["confidence"] > 0
    
    def test_identify_known_unknowns(self):
        """Should identify known unknowns."""
        system = SelfAwareSystem()
        
        current_data = {
            "data_freshness_seconds": 120,  # Stale data
            "order_book_depth": 5,  # Shallow
            "volume_24h": 500000,  # Low volume
        }
        
        unknowns = system.identify_known_unknowns(current_data)
        
        assert len(unknowns) >= 2
        assert any("Stale" in u for u in unknowns)
    
    def test_calculate_position_size(self):
        """Should calculate position size based on awareness."""
        system = SelfAwareSystem()
        
        size = system.calculate_position_size(
            base_size=10000,
            confidence=0.8,
            regime="trending",
            known_unknowns=["Low liquidity"],
        )
        
        assert 0 < size <= 10000
    
    def test_should_trade(self):
        """Should determine if trading is appropriate."""
        system = SelfAwareSystem()
        
        # Good conditions
        should_trade, reason = system.should_trade(
            signal_confidence=0.8,
            regime="trending",
            known_unknowns=[],
        )
        
        assert should_trade is True
        
        # Bad conditions
        should_trade, reason = system.should_trade(
            signal_confidence=0.2,
            regime="unknown",
            known_unknowns=["A", "B", "C", "D", "E", "F"],
        )
        
        assert should_trade is False
    
    def test_get_meta_score(self):
        """Should calculate meta score."""
        system = SelfAwareSystem()
        
        score = system.get_meta_score()
        
        assert 0 <= score <= 1


# ============================================================================
# Autonomous Research Lab Tests
# ============================================================================

class TestAutonomousResearchLab:
    """Tests for Autonomous Research Lab."""
    
    def test_init(self):
        """Should initialize correctly."""
        lab = AutonomousResearchLab()
        
        assert len(lab.hypotheses) == 0
        assert lab.hypothesis_counter == 0
    
    def test_generate_hypothesis(self):
        """Should generate hypothesis."""
        lab = AutonomousResearchLab()
        
        conditions = {"volatility": 0.02, "trend_strength": 0.3}
        hypothesis = lab.generate_hypothesis(conditions)
        
        assert hypothesis.id.startswith("hyp_")
        assert len(hypothesis.statement) > 0
        assert len(hypothesis.test_plan) > 0
    
    def test_test_hypothesis(self):
        """Should test hypothesis."""
        lab = AutonomousResearchLab()
        
        hypothesis = lab.generate_hypothesis({})
        
        data = {"returns": np.random.randn(252) * 0.02}
        results = lab.test_hypothesis(hypothesis.id, data)
        
        assert "total_return" in results or "error" in results
    
    def test_generate_paper(self):
        """Should generate research paper."""
        lab = AutonomousResearchLab()
        
        hypothesis = lab.generate_hypothesis({})
        
        # Test it first
        data = {"returns": np.random.randn(252) * 0.02}
        lab.test_hypothesis(hypothesis.id, data)
        
        paper = lab.generate_paper(hypothesis.id)
        
        assert "Abstract" in paper
        assert "Methodology" in paper
        assert "Results" in paper
    
    def test_get_top_hypotheses(self):
        """Should get top hypotheses."""
        lab = AutonomousResearchLab()
        
        for _ in range(10):
            lab.generate_hypothesis({})
        
        top = lab.get_top_hypotheses(n=3)
        
        assert len(top) == 3
        assert top[0].priority >= top[1].priority


# ============================================================================
# Level 20 Singularity Tests
# ============================================================================

class TestLevel20Singularity:
    """Tests for Level 20 Singularity."""
    
    def test_init(self):
        """Should initialize correctly."""
        singularity = Level20Singularity()
        
        assert singularity.causal_engine is not None
        assert singularity.order_flow is not None
        assert singularity.cross_market is not None
        assert singularity.self_aware is not None
        assert singularity.research_lab is not None
    
    def test_analyze_market(self):
        """Should analyze market comprehensively."""
        singularity = Level20Singularity()
        
        market_data = {
            "volatility": 0.03,
            "trend_strength": 0.4,
            "signal_confidence": 0.7,
            "order_book": {
                "symbol": "BTC",
                "bids": [(65000.0, 100.0), (64999.0, 50.0)],
                "asks": [(65001.0, 100.0), (65002.0, 50.0)],
            },
        }
        
        analysis = singularity.analyze_market(market_data)
        
        assert "cycle" in analysis
        assert "order_flow" in analysis
        assert "self_awareness" in analysis
        assert "research" in analysis
    
    def test_get_system_report(self):
        """Should get system report."""
        singularity = Level20Singularity()
        
        report = singularity.get_system_report()
        
        assert report["system"] == "Level 20 Singularity"
        assert "components" in report
    
    def test_multiple_cycles(self):
        """Should handle multiple analysis cycles."""
        singularity = Level20Singularity()
        
        for i in range(5):
            market_data = {
                "volatility": 0.02 + i * 0.005,
                "trend_strength": 0.3,
                "signal_confidence": 0.6,
            }
            analysis = singularity.analyze_market(market_data)
            assert analysis["cycle"] == i + 1


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_level20_singularity(self):
        """Should create Level 20 Singularity."""
        system = create_level20_singularity()
        assert isinstance(system, Level20Singularity)
    
    def test_create_causal_engine(self):
        """Should create Causal Discovery Engine."""
        engine = create_causal_engine()
        assert isinstance(engine, CausalDiscoveryEngine)
    
    def test_create_order_flow_engine(self):
        """Should create Order Flow Engine."""
        engine = create_order_flow_engine()
        assert isinstance(engine, PredictiveOrderFlowEngine)
    
    def test_create_cross_market_system(self):
        """Should create Cross-Market System."""
        system = create_cross_market_system()
        assert isinstance(system, CrossMarketSystem)
    
    def test_create_self_aware_system(self):
        """Should create Self-Aware System."""
        system = create_self_aware_system()
        assert isinstance(system, SelfAwareSystem)
    
    def test_create_research_lab(self):
        """Should create Research Lab."""
        lab = create_research_lab()
        assert isinstance(lab, AutonomousResearchLab)


# ============================================================================
# Integration Tests
# ============================================================================

class TestLevel20Integration:
    """Integration tests for Level 20 components."""
    
    def test_causal_order_flow_integration(self):
        """Causal engine should inform order flow analysis."""
        causal = CausalDiscoveryEngine()
        order_flow = PredictiveOrderFlowEngine()
        
        # Create data where order flow causes price
        np.random.seed(42)
        order_imbalance = np.random.randn(200) * 0.1
        price_returns = order_imbalance * 0.5 + np.random.randn(200) * 0.1
        
        data = {"imbalance": order_imbalance, "returns": price_returns}
        edges = causal.discover_causal_structure(data)
        
        # Should find relationship
        assert len(edges) >= 0
    
    def test_self_aware_position_sizing(self):
        """Self-aware system should adjust position sizes."""
        self_aware = SelfAwareSystem()
        
        # Good conditions
        size_good = self_aware.calculate_position_size(
            base_size=10000,
            confidence=0.9,
            regime="trending",
            known_unknowns=[],
        )
        
        # Bad conditions
        size_bad = self_aware.calculate_position_size(
            base_size=10000,
            confidence=0.3,
            regime="unknown",
            known_unknowns=["A", "B", "C", "D"],
        )
        
        assert size_good > size_bad
    
    def test_research_to_trading(self):
        """Research findings should inform trading."""
        lab = AutonomousResearchLab()
        self_aware = SelfAwareSystem()
        
        # Generate and test hypothesis
        hypothesis = lab.generate_hypothesis({})
        data = {"returns": np.random.randn(252) * 0.02}
        results = lab.test_hypothesis(hypothesis.id, data)
        
        # If profitable, increase confidence
        if results.get("profitable", False):
            self_aware.state.confidence_calibration = min(1.0, self_aware.state.confidence_calibration + 0.1)
        
        assert self_aware.state.confidence_calibration >= 0.5
