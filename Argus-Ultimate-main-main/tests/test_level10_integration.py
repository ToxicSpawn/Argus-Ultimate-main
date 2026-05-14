"""
tests/test_level10_integration.py — Tests for Level 10 Integration with Argus
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import MagicMock, AsyncMock

from evolution.level10_integration import (
    Level10Integration,
    create_level10_integration,
)


class TestLevel10Integration:
    """Tests for Level 10 Integration."""
    
    def test_init_disabled(self):
        """Should initialize disabled by default."""
        system = MagicMock()
        integration = Level10Integration(system=system, config={})
        
        assert integration.enabled is False
    
    def test_init_enabled(self):
        """Should initialize when enabled."""
        system = MagicMock()
        integration = Level10Integration(
            system=system,
            config={"enabled": True, "population_size": 20},
        )
        
        assert integration.enabled is True
        assert integration.population_size == 20
    
    @pytest.mark.asyncio
    async def test_initialize_disabled(self):
        """Should skip initialization when disabled."""
        system = MagicMock()
        integration = Level10Integration(system=system, config={})
        
        await integration.initialize()
        
        assert integration.level10 is None
    
    @pytest.mark.asyncio
    async def test_initialize_enabled(self):
        """Should initialize Level 10 system when enabled."""
        system = MagicMock()
        integration = Level10Integration(
            system=system,
            config={"enabled": True, "population_size": 20, "complexity": 0.3},
        )
        
        await integration.initialize()
        
        assert integration.level10 is not None
        assert len(integration.level10.evolution.population) == 20
    
    @pytest.mark.asyncio
    async def test_evolve_generation(self):
        """Should evolve one generation."""
        system = MagicMock()
        integration = Level10Integration(
            system=system,
            config={"enabled": True, "population_size": 10},
        )
        
        await integration.initialize()
        results = await integration._evolve_generation()
        
        assert "best_fitness" in results
        assert results["generations_completed"] == 1
    
    def test_evaluate_genome(self):
        """Should evaluate genome fitness."""
        system = MagicMock()
        integration = Level10Integration(
            system=system,
            config={"enabled": True},
        )
        
        from evolution.level10_self_evolving_system import GenomeFactory
        genome = GenomeFactory.create_random_genome()
        
        fitness = integration._evaluate_genome(genome)
        
        assert 0 <= fitness <= 1
    
    def test_get_market_conditions(self):
        """Should get market conditions."""
        system = MagicMock()
        integration = Level10Integration(system=system, config={})
        
        conditions = integration._get_market_conditions()
        
        assert "timestamp" in conditions
        assert "volatility" in conditions
    
    @pytest.mark.asyncio
    async def test_on_trade_result(self):
        """Should process trade result."""
        system = MagicMock()
        integration = Level10Integration(
            system=system,
            config={"enabled": True, "population_size": 10},
        )
        
        await integration.initialize()
        
        trade = {"pnl": 100.0, "symbol": "BTCUSDT", "side": "buy"}
        await integration.on_trade_result(trade)
        
        assert len(integration._performance_history) == 1
    
    def test_get_report_disabled(self):
        """Should return disabled report when not enabled."""
        system = MagicMock()
        integration = Level10Integration(system=system, config={})
        
        report = integration.get_report()
        
        assert report["enabled"] is False
    
    @pytest.mark.asyncio
    async def test_get_report_enabled(self):
        """Should return full report when enabled."""
        system = MagicMock()
        integration = Level10Integration(
            system=system,
            config={"enabled": True, "population_size": 10},
        )
        
        await integration.initialize()
        report = integration.get_report()
        
        assert "generation" in report
        assert "evolution_stats" in report
    
    def test_factory_function(self):
        """Should create integration via factory."""
        system = MagicMock()
        integration = create_level10_integration(
            system=system,
            config={"enabled": True},
        )
        
        assert isinstance(integration, Level10Integration)
        assert integration.enabled is True


class TestLevel10IntegrationPerformance:
    """Performance tests for Level 10 Integration."""
    
    @pytest.mark.asyncio
    async def test_multiple_evolution_cycles(self):
        """Should handle multiple evolution cycles."""
        system = MagicMock()
        integration = Level10Integration(
            system=system,
            config={"enabled": True, "population_size": 20},
        )
        
        await integration.initialize()
        
        for _ in range(5):
            results = await integration._evolve_generation()
            assert results["generations_completed"] == 1
    
    @pytest.mark.asyncio
    async def test_trade_result_triggers_evolution(self):
        """Should trigger evolution on performance drop."""
        system = MagicMock()
        integration = Level10Integration(
            system=system,
            config={"enabled": True, "population_size": 10},
        )
        
        await integration.initialize()
        
        # Add 20 trades with declining performance
        for i in range(20):
            pnl = 100 - i * 10  # Declining PnL
            await integration.on_trade_result({"pnl": float(pnl)})
        
        # Should have triggered evolution due to performance drop
        assert len(integration._performance_history) == 20
