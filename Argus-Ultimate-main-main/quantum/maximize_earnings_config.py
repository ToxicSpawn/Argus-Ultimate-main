
from typing import Any
'''
Maximum Earnings Configuration
Pre-configured settings to maximize bot profitability
'''


# Maximum Earnings Configuration
MAX_EARNINGS_CONFIG: dict[str, Any] = {
    # Quantum settings - Ultimate mode for auto-tuning
    "quantum": {
        "performance_level": "ultimate",  # Auto-tuning + self-learning
        "qaoa_layers": 5,  # Maximum optimization layers
        "vqe_qubits": 6,  # More qubits = better risk analysis
        "vqc_layers": 3,  # More layers = better patterns
        "qnn_layers": 3,  # More layers = better predictions
        "risk_aversion": 0.5,  # Aggressive (lower = more risk)
        "auto_tune": True,  # Auto-optimize performance
        "use_error_mitigation": True,  # Better accuracy
    },
    # Risk settings - Aggressive for maximum returns
    "risk": {
        "risk_per_trade": 0.03,  # 3% per trade (higher risk)
        "max_drawdown": 0.15,  # 15% max drawdown tolerance
        "position_size": 0.95,  # Use 95% of capital
        "stop_loss": 0.02,  # 2% stop loss
        "take_profit": 0.05,  # 5% take profit
        "trailing_stop": True,  # Trailing stop for profits
        "max_position_per_symbol": 0.3,  # Max 30% per symbol
        "daily_loss_limit": 0.05,  # 5% daily loss limit
    },
    # Trading settings
    "trading": {
        "min_confidence": 0.65,  # Higher confidence threshold
        "compound_returns": True,  # Reinvest profits
        "reinvest_threshold": 0.05,  # Reinvest at 5% profit
        "compound_frequency": "daily",  # Daily compounding
        "use_multiple_timeframes": True,  # Multi-timeframe analysis
        "trend_timeframe": "4h",  # Long-term trend
        "entry_timeframe": "15m",  # Entry timing
        "exit_timeframe": "5m",  # Exit timing
        "ensemble_voting": True,  # Require multiple strategy agreement
        "min_strategies_agree": 2,  # At least 2 strategies must agree
    },
    # Market selection - Best markets only
    "markets": {
        "min_daily_volume": 100000000,  # $100M minimum daily volume
        "min_volatility": 0.02,  # 2% minimum daily volatility
        "max_volatility": 0.10,  # 10% maximum (too risky)
        "prefer_trending": True,  # Focus on trending markets
        "avoid_choppy": True,  # Avoid choppy/ranging markets
        "max_symbols": 10,  # Top 10 assets
        "symbols": [  # Recommended symbols
            "BTC/USDT",  # High volume, trending
            "ETH/USDT",  # High volume, trending
            "SOL/USDT",  # High volatility, trending
            "BNB/USDT",  # Exchange token, good liquidity
            "ADA/USDT",  # Good volume
            "XRP/USDT",  # High volume
            "DOGE/USDT",  # High volatility
            "MATIC/USDT",  # Good trends
            "AVAX/USDT",  # High volatility
            "LINK/USDT",  # Good trends
        ],
    },
    # Execution optimization
    "execution": {
        "use_smart_routing": True,  # Smart order routing
        "slippage_limit": 0.001,  # 0.1% max slippage
        "use_twap": True,  # TWAP for large orders
        "use_iceberg": True,  # Iceberg for hiding intentions
        "use_pegged": True,  # Pegged orders for better fills
        "min_order_size": 10.0,  # $10 minimum order
        "max_order_size": 10000.0,  # $10k maximum order
    },
    # Strategy ensemble
    "strategies": {
        "use_quantum": True,  # Quantum strategy
        "use_hunter": True,  # Momentum strategy
        "use_farmer": True,  # Grid strategy
        "use_shadow": True,  # Mean reversion
        "use_ml": True,  # ML strategy
        "strategy_weights": {  # Weight each strategy
            "quantum": 0.4,  # Quantum gets highest weight
            "hunter": 0.2,
            "farmer": 0.2,
            "shadow": 0.1,
            "ml": 0.1,
        },
    },
    # Position sizing
    "position_sizing": {
        "method": "kelly",  # Kelly Criterion
        "base_size": 0.1,  # 10% base position
        "max_size": 0.3,  # 30% maximum
        "confidence_multiplier": True,  # Scale by confidence
        "volatility_adjustment": True,  # Adjust for volatility
        "kelly_fraction": 0.5,  # Half-Kelly for safety
    },
    # Performance monitoring
    "monitoring": {
        "track_metrics": True,  # Track all metrics
        "alert_on_drawdown": 0.10,  # Alert at 10% drawdown
        "alert_on_daily_loss": 0.03,  # Alert at 3% daily loss
        "optimize_weekly": True,  # Weekly optimization
        "rebalance_daily": True,  # Daily rebalancing
    },
}


# Aggressive Growth Configuration
AGGRESSIVE_GROWTH_CONFIG: dict[str, Any] = {
    **MAX_EARNINGS_CONFIG,
    "risk": {
        **MAX_EARNINGS_CONFIG["risk"],
        "risk_per_trade": 0.05,  # 5% per trade
        "max_drawdown": 0.20,  # 20% max drawdown
        "position_size": 0.98,  # Use 98% of capital
    },
    "trading": {
        **MAX_EARNINGS_CONFIG["trading"],
        "min_confidence": 0.70,  # Very high confidence
    },
    "quantum": {
        **MAX_EARNINGS_CONFIG["quantum"],
        "risk_aversion": 0.3,  # Very aggressive
    },
}


# Balanced Growth Configuration
BALANCED_GROWTH_CONFIG: dict[str, Any] = {
    **MAX_EARNINGS_CONFIG,
    "risk": {
        **MAX_EARNINGS_CONFIG["risk"],
        "risk_per_trade": 0.03,  # 3% per trade
        "max_drawdown": 0.12,  # 12% max drawdown
        "position_size": 0.90,  # Use 90% of capital
    },
    "trading": {
        **MAX_EARNINGS_CONFIG["trading"],
        "min_confidence": 0.60,  # Moderate confidence
    },
    "quantum": {
        **MAX_EARNINGS_CONFIG["quantum"],
        "risk_aversion": 0.7,  # Balanced
    },
}


# Conservative Growth Configuration
CONSERVATIVE_GROWTH_CONFIG: dict[str, Any] = {
    **MAX_EARNINGS_CONFIG,
    "risk": {
        **MAX_EARNINGS_CONFIG["risk"],
        "risk_per_trade": 0.02,  # 2% per trade
        "max_drawdown": 0.08,  # 8% max drawdown
        "position_size": 0.80,  # Use 80% of capital
    },
    "trading": {
        **MAX_EARNINGS_CONFIG["trading"],
        "min_confidence": 0.70,  # High confidence
    },
    "quantum": {
        **MAX_EARNINGS_CONFIG["quantum"],
        "risk_aversion": 1.5,  # Conservative
    },
}


def get_max_earnings_config(risk_level: str = "balanced") -> dict[str, Any]:
    '''
    Get maximum earnings configuration

    Args:
        risk_level: 'aggressive', 'balanced', or 'conservative'

    Returns:
        Configuration dictionary
    '''
    configs = {
        "aggressive": AGGRESSIVE_GROWTH_CONFIG,
        "balanced": BALANCED_GROWTH_CONFIG,
        "conservative": CONSERVATIVE_GROWTH_CONFIG,
    }

    return (configs.get(risk_level.lower(), BALANCED_GROWTH_CONFIG)
)