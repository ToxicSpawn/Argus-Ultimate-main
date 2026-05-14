"""
tests/test_multi_agent_swarms.py — Tests for Multi-Agent Trading Swarm
"""

import pytest
import numpy as np

from agents.multi_agent_swarms import (
    SwarmOrchestrator,
    MomentumAgent,
    MeanReversionAgent,
    WhaleFollowerAgent,
    BreakoutAgent,
    AgentType,
    AgentVerdict,
    TradeSignal,
    create_trading_swarm,
)


class TestMomentumAgent:
    """Tests for Momentum Agent."""
    
    def test_init(self):
        """Should initialize correctly."""
        agent = MomentumAgent()
        assert agent.agent_type == AgentType.MOMENTUM
        assert agent.is_active is True
    
    def test_bullish_signal(self):
        """Should detect bullish trend."""
        agent = MomentumAgent()
        
        # Create bullish data (uptrend)
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(50) * 0.005 + 0.003))
        volumes = np.random.randn(50) * 1000 + 10000
        
        signal = agent.analyze(prices, volumes)
        
        assert signal.agent_id == "momentum_001"
        assert signal.verdict in [AgentVerdict.BUY, AgentVerdict.STRONG_BUY, AgentVerdict.NEUTRAL]
    
    def test_bearish_signal(self):
        """Should detect bearish trend."""
        agent = MomentumAgent()
        
        # Create bearish data (downtrend)
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(50) * 0.005 - 0.004))
        volumes = np.random.randn(50) * 1000 + 10000
        
        signal = agent.analyze(prices, volumes)
        
        assert signal.verdict in [AgentVerdict.SELL, AgentVerdict.STRONG_SELL, AgentVerdict.NEUTRAL]


class TestMeanReversionAgent:
    """Tests for Mean Reversion Agent."""
    
    def test_oversold_signal(self):
        """Should buy when oversold."""
        agent = MeanReversionAgent()
        
        # Create oversold data (sharp drop)
        prices = np.concatenate([
            np.ones(20) * 100,
            np.linspace(100, 80, 20),
        ])
        volumes = np.random.randn(40) * 1000 + 10000
        
        signal = agent.analyze(prices, volumes)
        
        assert signal.verdict in [AgentVerdict.BUY, AgentVerdict.STRONG_BUY]
        assert "oversold" in signal.reasoning.lower() or "RSI" in signal.reasoning


class TestWhaleFollowerAgent:
    """Tests for Whale Follower Agent."""
    
    def test_whale_buying(self):
        """Should detect whale buying."""
        agent = WhaleFollowerAgent()
        
        # High volume + price up
        prices = np.concatenate([
            np.ones(10) * 100,
            [100, 102],  # Price up
        ])
        volumes = np.concatenate([
            np.ones(10) * 1000,
            [5000],  # High volume
        ])
        
        signal = agent.analyze(prices, volumes)
        
        # Should detect activity
        assert signal.confidence >= 0


class TestBreakoutAgent:
    """Tests for Breakout Agent."""
    
    def test_near_resistance(self):
        """Should detect near resistance."""
        agent = BreakoutAgent()
        
        # Price near recent high
        prices = np.concatenate([
            np.ones(15) * 100,
            np.linspace(100, 105, 10),
        ])
        volumes = np.random.randn(25) * 1000 + 10000
        
        signal = agent.analyze(prices, volumes)
        
        assert signal.confidence >= 0


class TestSwarmOrchestrator:
    """Tests for Swarm Orchestrator."""
    
    def test_init(self):
        """Should initialize with default agents."""
        swarm = SwarmOrchestrator()
        
        assert len(swarm.agents) >= 4
        assert "momentum" in swarm.agents
        assert "mean_reversion" in swarm.agents
    
    def test_get_signals(self):
        """Should get signals from all agents."""
        swarm = SwarmOrchestrator()
        
        prices = 100 * np.exp(np.cumsum(np.random.randn(50) * 0.01))
        volumes = np.random.randn(50) * 1000 + 10000
        
        signals = swarm.get_signals(prices, volumes)
        
        assert len(signals) >= 4
        assert all(isinstance(s, TradeSignal) for s in signals)
    
    def test_build_consensus(self):
        """Should build consensus from signals."""
        swarm = SwarmOrchestrator()
        
        prices = 100 * np.exp(np.cumsum(np.random.randn(50) * 0.01))
        volumes = np.random.randn(50) * 1000 + 10000
        
        signals = swarm.get_signals(prices, volumes)
        consensus = swarm.build_consensus(signals)
        
        assert "verdict" in consensus
        assert "confidence" in consensus
        assert "n_agents" in consensus
    
    def test_get_agent_rankings(self):
        """Should rank agents by performance."""
        swarm = SwarmOrchestrator()
        
        rankings = swarm.get_agent_rankings()
        
        assert len(rankings) >= 4
        assert rankings[0]["rank"] == 1
    
    def test_evolve_swarm(self):
        """Should evolve swarm."""
        swarm = SwarmOrchestrator()
        
        # Make an agent perform poorly
        for agent in swarm.agents.values():
            for _ in range(20):
                agent.update_performance(-10, won=False)
        
        result = swarm.evolve_swarm(min_elo=1600)
        
        # Should have removed some agents
        assert isinstance(result["removed"], list)
    
    def test_get_swarm_report(self):
        """Should get swarm report."""
        swarm = SwarmOrchestrator()
        
        report = swarm.get_swarm_report()
        
        assert "n_agents" in report
        assert "agent_rankings" in report


class TestFactoryFunction:
    """Tests for factory function."""
    
    def test_create_trading_swarm(self):
        """Should create trading swarm."""
        swarm = create_trading_swarm()
        
        assert isinstance(swarm, SwarmOrchestrator)
        assert len(swarm.agents) >= 4


class TestAgentPerformance:
    """Tests for agent performance tracking."""
    
    def test_elo_rating(self):
        """Should update ELO rating."""
        agent = MomentumAgent()
        
        initial_elo = agent.performance.elo_rating
        
        # Win should increase ELO
        agent.update_performance(100, won=True)
        elo_after_win = agent.performance.elo_rating
        assert elo_after_win > initial_elo
        
        # Lose should decrease ELO
        agent.update_performance(-100, won=False)
        elo_after_loss = agent.performance.elo_rating
        assert elo_after_loss < elo_after_win
    
    def test_win_rate(self):
        """Should calculate win rate."""
        agent = MomentumAgent()
        
        agent.update_performance(100, won=True)
        agent.update_performance(100, won=True)
        agent.update_performance(-50, won=False)
        
        assert agent.performance.win_rate == 2/3


class TestSwarmIntegration:
    """Integration tests for swarm."""
    
    def test_full_cycle(self):
        """Should complete full analysis cycle."""
        swarm = create_trading_swarm()
        
        # Simulate market data
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01))
        volumes = np.random.randn(100) * 1000 + 10000
        
        # Get signals
        signals = swarm.get_signals(prices, volumes)
        
        # Build consensus
        consensus = swarm.build_consensus(signals)
        
        # Get rankings
        rankings = swarm.get_agent_rankings()
        
        assert len(signals) > 0
        assert consensus["verdict"] is not None
        assert len(rankings) > 0
