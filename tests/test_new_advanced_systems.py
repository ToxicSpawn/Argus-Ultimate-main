"""
tests/test_new_advanced_systems.py — Tests for New Advanced AI Systems

Tests for:
- Deep RL Trading Agent
- Real-Time NLP Pipeline
- Multi-Agent Trading System
- On-Chain Intelligence
- Self-Improving Code Generator
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock

# Deep RL Agent
from ml.deep_rl_trading_agent import (
    PPOTradingAgent,
    SACTradingAgent,
    TradingEnvironment,
    RLConfig,
    ReplayBuffer,
    Transition,
    create_ppo_agent,
    create_sac_agent,
)

# NLP Pipeline
from ml.realtime_nlp_pipeline import (
    RealTimeNLPPipeline,
    SentimentAnalyzer,
    EntityExtractor,
    EventDetector,
    TopicClassifier,
    SentimentLevel,
    EventType,
    TopicCategory,
    create_nlp_pipeline,
)

# Multi-Agent System
from agents.multi_agent_trading_system import (
    MultiAgentTradingSystem,
    MarketState,
    SignalType,
    MomentumAgent,
    MeanReversionAgent,
    RiskManagerAgent,
    create_multi_agent_system,
)

# On-Chain Intelligence
from analytics.onchain_intelligence import (
    OnChainIntelligence,
    FlowDirection,
    WhaleAction,
    DeFiProtocol,
    create_onchain_intelligence,
)

# Code Generator
from self_improvement.code_generator import (
    StrategyGenerator,
    GeneratedStrategy,
    StrategyStatus,
    CodeQualityChecker,
    create_strategy_generator,
)


# ============================================================================
# Deep RL Agent Tests
# ============================================================================

class TestPPOTradingAgent:
    """Tests for PPO Trading Agent."""
    
    def test_init(self):
        """Should initialize with correct dimensions."""
        agent = PPOTradingAgent(observation_dim=20, action_dim=3)
        
        assert agent.observation_dim == 20
        assert agent.action_dim == 3
        assert agent.actor is not None
        assert agent.critic is not None
    
    def test_get_action(self):
        """Should return valid action."""
        agent = PPOTradingAgent(observation_dim=20, action_dim=3)
        obs = np.random.randn(20)
        
        action, log_prob, value = agent.get_action(obs)
        
        assert len(action) == 3
        assert -1 <= action[0] <= 1
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
    
    def test_predict(self):
        """Should return deterministic prediction."""
        agent = PPOTradingAgent(observation_dim=20, action_dim=3)
        obs = np.random.randn(20)
        
        action = agent.predict(obs, deterministic=True)
        
        assert len(action) == 3
    
    def test_compute_gae(self):
        """Should compute GAE correctly."""
        agent = PPOTradingAgent(observation_dim=20, action_dim=3)
        
        rewards = [1.0, -0.5, 0.3, 0.8]
        values = [0.5, 0.3, 0.4, 0.6]
        dones = [0, 0, 0, 0]
        
        advantages, returns = agent.compute_gae(rewards, values, dones, 0.5)
        
        assert len(advantages) == 4
        assert len(returns) == 4


class TestSACTradingAgent:
    """Tests for SAC Trading Agent."""
    
    def test_init(self):
        """Should initialize with twin Q-networks."""
        agent = SACTradingAgent(observation_dim=20, action_dim=3)
        
        assert agent.q1 is not None
        assert agent.q2 is not None
        assert agent.q1_target is not None
        assert agent.q2_target is not None
    
    def test_get_action(self):
        """Should return valid action."""
        agent = SACTradingAgent(observation_dim=20, action_dim=3)
        obs = np.random.randn(20)
        
        action = agent.get_action(obs)
        
        assert len(action) == 3
        assert all(-1 <= a <= 1 for a in action)
    
    def test_store_transition(self):
        """Should store transition in replay buffer."""
        agent = SACTradingAgent(observation_dim=20, action_dim=3)
        
        transition = Transition(
            observation=np.random.randn(20),
            action=np.random.randn(3),
            reward=1.0,
            next_observation=np.random.randn(20),
            done=False,
        )
        
        agent.store_transition(transition)
        
        assert len(agent.replay_buffer) == 1


class TestTradingEnvironment:
    """Tests for Trading Environment."""
    
    def test_init(self):
        """Should initialize correctly."""
        prices = np.random.randn(100) * 10 + 100
        env = TradingEnvironment(prices)
        
        assert env.observation_dim == 13
        assert env.action_dim == 1
    
    def test_reset(self):
        """Should reset to initial state."""
        prices = np.random.randn(100) * 10 + 100
        env = TradingEnvironment(prices)
        
        obs = env.reset()
        
        assert len(obs) == 13
        assert env.capital == env.initial_capital
    
    def test_step(self):
        """Should execute step correctly."""
        prices = np.random.randn(100) * 10 + 100
        env = TradingEnvironment(prices)
        env.reset()
        
        action = np.array([0.5])
        obs, reward, done, info = env.step(action)
        
        assert len(obs) == 13
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "capital" in info


# ============================================================================
# NLP Pipeline Tests
# ============================================================================

class TestSentimentAnalyzer:
    """Tests for Sentiment Analyzer."""
    
    def test_positive_sentiment(self):
        """Should detect positive sentiment."""
        analyzer = SentimentAnalyzer()
        
        text = "Apple reports record profits and strong growth"
        score, level, confidence = analyzer.analyze(text)
        
        assert score > 0
        assert level in (SentimentLevel.POSITIVE, SentimentLevel.VERY_POSITIVE)
    
    def test_negative_sentiment(self):
        """Should detect negative sentiment."""
        analyzer = SentimentAnalyzer()
        
        text = "Company reports losses and declining revenue"
        score, level, confidence = analyzer.analyze(text)
        
        assert score < 0
        assert level in (SentimentLevel.NEGATIVE, SentimentLevel.VERY_NEGATIVE)


class TestEntityExtractor:
    """Tests for Entity Extractor."""
    
    def test_extract_ticker(self):
        """Should extract ticker symbols."""
        extractor = EntityExtractor()
        
        text = "I bought $AAPL and MSFT today"
        entities = extractor.extract(text)
        
        tickers = [e.ticker for e in entities if e.ticker]
        assert "AAPL" in tickers
        assert "MSFT" in tickers


class TestEventDetector:
    """Tests for Event Detector."""
    
    def test_earnings_event(self):
        """Should detect earnings event."""
        detector = EventDetector()
        
        text = "Company beats Q4 earnings estimates"
        event_type, confidence = detector.detect(text)
        
        assert event_type == EventType.EARNINGS
        assert confidence > 0.5


class TestRealTimeNLPPipeline:
    """Tests for Real-Time NLP Pipeline."""
    
    def test_process_article(self):
        """Should process article and return signal."""
        pipeline = RealTimeNLPPipeline()
        
        result = pipeline.process("Apple reports record Q4 earnings, beats estimates")
        
        assert result.sentiment_score != 0 or result.event_type != EventType.UNKNOWN
        assert result.processing_time_ms < 100
    
    def test_aggregate_signals(self):
        """Should aggregate signals over time window."""
        pipeline = RealTimeNLPPipeline()
        
        pipeline.process("Apple reports strong earnings")
        pipeline.process("Microsoft announces layoffs")
        pipeline.process("Tesla stock surges on delivery numbers")
        
        aggregated = pipeline.aggregate_signals()
        
        assert aggregated.n_articles == 3
    
    def test_get_ticker_sentiment(self):
        """Should get sentiment for specific ticker."""
        pipeline = RealTimeNLPPipeline()
        
        pipeline.process("Apple reports record profits")
        pipeline.process("$AAPL stock surges")
        
        sentiment, count = pipeline.get_ticker_sentiment("AAPL")
        
        assert count >= 1


# ============================================================================
# Multi-Agent System Tests
# ============================================================================

class TestMultiAgentTradingSystem:
    """Tests for Multi-Agent Trading System."""
    
    def test_init(self):
        """Should initialize with default agents."""
        system = MultiAgentTradingSystem()
        
        assert len(system.agents) >= 7
    
    def test_decide(self):
        """Should make consensus decision."""
        system = MultiAgentTradingSystem()
        
        market_state = MarketState(
            timestamp=datetime.utcnow(),
            prices=np.random.randn(50, 4) * 10 + 100,
            volume=np.random.randn(50) * 1000 + 10000,
            returns=np.random.randn(50) * 0.02,
            volatility=0.02,
            trend_strength=0.5,
            regime="trending",
        )
        
        decision = system.decide(market_state)
        
        assert decision.signal_type in SignalType
        assert 0 <= decision.confidence <= 1
        assert len(decision.agent_signals) >= 3


# ============================================================================
# On-Chain Intelligence Tests
# ============================================================================

class TestOnChainIntelligence:
    """Tests for On-Chain Intelligence."""
    
    def test_init(self):
        """Should initialize correctly."""
        intel = OnChainIntelligence()
        
        assert intel.whale_threshold_usd == 100000.0
    
    def test_analyze_whale_activity(self):
        """Should analyze whale activity."""
        intel = OnChainIntelligence()
        
        signal = intel.analyze_whale_activity("ETH")
        
        assert signal.token == "ETH"
        assert -1 <= signal.whale_accumulation_score <= 1
    
    def test_get_exchange_flows(self):
        """Should get exchange flows."""
        intel = OnChainIntelligence()
        
        flows = intel.get_exchange_flows("BTC")
        
        assert "net_flow_usd" in flows
        assert "signal" in flows
    
    def test_get_aggregated_signal(self):
        """Should get aggregated signal."""
        intel = OnChainIntelligence()
        
        signal = intel.get_aggregated_signal("ETH")
        
        assert -1 <= signal["overall_signal"] <= 1
        assert "confidence" in signal


# ============================================================================
# Code Generator Tests
# ============================================================================

class TestCodeQualityChecker:
    """Tests for Code Quality Checker."""
    
    def test_good_code(self):
        """Should pass quality check for good code."""
        checker = CodeQualityChecker()
        
        code = '''
class TestStrategy:
    """Test strategy."""
    
    def generate_signal(self, prices):
        """Generate signal."""
        signal = "hold"
        return signal, 0.5
'''
        
        score, issues = checker.check(code)
        
        assert score >= 70
    
    def test_bad_code(self):
        """Should fail quality check for bad code."""
        checker = CodeQualityChecker()
        
        code = '''
def test():
    pass
    TODO: implement
    eval("dangerous")
'''
        
        score, issues = checker.check(code)
        
        assert score < 70
        assert len(issues) > 0


class TestStrategyGenerator:
    """Tests for Strategy Generator."""
    
    def test_init(self):
        """Should initialize correctly."""
        generator = StrategyGenerator()
        
        assert generator.population_size == 20
    
    def test_generate_strategy(self):
        """Should generate a strategy."""
        generator = StrategyGenerator()
        
        strategy = generator.generate_strategy(
            "Mean reversion with RSI and Bollinger Bands",
            strategy_type="mean_reversion",
        )
        
        assert strategy.status == StrategyStatus.VALIDATED
        assert "generate_signal" in strategy.code
    
    def test_optimize_strategy(self):
        """Should optimize strategy parameters."""
        generator = StrategyGenerator()
        
        strategy = generator.generate_strategy("Test strategy")
        original_gen = strategy.generation
        optimized = generator.optimize_strategy(strategy, generations=3)
        
        assert optimized.generation >= original_gen
        assert optimized.fitness_score > 0
    
    def test_ab_test(self):
        """Should A/B test two strategies."""
        generator = StrategyGenerator()
        
        strategy_a = generator.generate_strategy("Strategy A")
        strategy_b = generator.generate_strategy("Strategy B")
        
        result = generator.ab_test(strategy_a, strategy_b)
        
        assert result.winner in ("a", "b", "tie")
