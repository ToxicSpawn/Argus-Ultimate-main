"""
tests/test_mexc_connector.py — Tests for MEXC Exchange Connector
"""

import pytest
from unittest.mock import MagicMock, patch

from connectors.mexc_connector import (
    MEXCConnector,
    MEXCFeeStructure,
    create_mexc_connector,
)


class TestMEXCFeeStructure:
    """Tests for MEXC fee structure."""
    
    def test_default_fees(self):
        """Should have 0% maker, 0.01% taker fees."""
        fees = MEXCFeeStructure()
        
        assert fees.maker_fee == 0.0
        assert fees.taker_fee == 0.0001
    
    def test_calculate_limit_cost(self):
        """Limit orders should have 0% fee."""
        fees = MEXCFeeStructure()
        
        cost = fees.calculate_cost(1000.0, "limit")
        
        assert cost == 0.0
    
    def test_calculate_market_cost(self):
        """Market orders should have 0.01% fee."""
        fees = MEXCFeeStructure()
        
        cost = fees.calculate_cost(1000.0, "market")
        
        assert cost == 0.1  # 0.01% of 1000 = 0.1


class TestMEXCConnector:
    """Tests for MEXC Connector."""
    
    def test_init(self):
        """Should initialize correctly."""
        connector = MEXCConnector(api_key="test", api_secret="test")
        
        assert connector.api_key == "test"
        assert connector.fees.maker_fee == 0.0
    
    def test_fee_summary(self):
        """Should return fee summary."""
        connector = MEXCConnector()
        
        summary = connector.get_fee_summary()
        
        assert summary["exchange"] == "MEXC"
        assert "0%" in summary["maker_fee"]
    
    def test_estimate_monthly_fees(self):
        """Should estimate monthly fees."""
        connector = MEXCConnector()
        
        estimate = connector.estimate_monthly_fees(
            monthly_volume=10000,
            limit_order_pct=0.8,
        )
        
        assert estimate["mexc_total_fees"] < estimate["kraken_estimated_fees"]
        assert estimate["monthly_savings"] > 0
    
    def test_fee_comparison(self):
        """MEXC should be cheaper than Kraken."""
        connector = MEXCConnector()
        
        # $10K monthly volume
        estimate = connector.estimate_monthly_fees(10000, limit_order_pct=1.0)
        
        # MEXC: 0% fees = $0
        # Kraken: ~0.2% = $20
        assert estimate["mexc_total_fees"] == 0.0
        assert estimate["monthly_savings"] == 20.0


class TestMEXCFactory:
    """Tests for MEXC factory function."""
    
    def test_create_connector(self):
        """Should create connector via factory."""
        connector = create_mexc_connector(api_key="test", api_secret="test")
        
        assert isinstance(connector, MEXCConnector)
        assert connector.api_key == "test"


class TestMEXCFeeOptimization:
    """Tests for fee optimization."""
    
    def test_limit_order_savings(self):
        """Should calculate savings from limit orders."""
        connector = MEXCConnector()
        
        # $1000 order
        limit_cost = connector.fees.calculate_cost(1000, "limit")
        market_cost = connector.fees.calculate_cost(1000, "market")
        
        assert limit_cost == 0.0
        assert market_cost == 0.1
        assert market_cost > limit_cost
    
    def test_annual_savings(self):
        """Should calculate annual savings."""
        connector = MEXCConnector()
        
        estimate = connector.estimate_monthly_fees(50000, limit_order_pct=0.9)
        
        # Annual savings should be significant
        assert estimate["annual_savings"] > 1000
