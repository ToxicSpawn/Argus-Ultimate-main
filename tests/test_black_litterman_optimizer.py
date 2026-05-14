from __future__ import annotations

import numpy as np
import pytest

from portfolio.black_litterman_optimizer import (
    BlackLittermanConfig,
    BlackLittermanOptimizer,
    MarketView,
)


def test_market_view_validates_confidence() -> None:
    with pytest.raises(ValueError):
        MarketView(symbol="BTC/USD", expected_return=0.05, confidence=1.2)


def test_calculate_equilibrium_and_implied_returns() -> None:
    config = BlackLittermanConfig(
        risk_aversion=2.0,
        tau=0.05,
        market_cap_weights={"BTC/USD": 0.6, "ETH/USD": 0.4},
        risk_free_rate=0.01,
    )
    optimizer = BlackLittermanOptimizer(symbols=["BTC/USD", "ETH/USD"], config=config)
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]], dtype=float)

    implied = optimizer.calculate_implied_returns(config.market_cap_weights, covariance)
    equilibrium = optimizer.calculate_equilibrium_returns(covariance, config.market_cap_weights)

    assert implied.shape == (2,)
    assert equilibrium.shape == (2,)
    assert np.allclose(equilibrium, implied + config.risk_free_rate)


def test_posterior_returns_shift_toward_views() -> None:
    config = BlackLittermanConfig(
        market_cap_weights={"BTC/USD": 0.5, "ETH/USD": 0.5},
        risk_free_rate=0.0,
    )
    optimizer = BlackLittermanOptimizer(symbols=["BTC/USD", "ETH/USD"], config=config)
    covariance = np.array([[0.05, 0.01], [0.01, 0.06]], dtype=float)
    prior = optimizer.calculate_equilibrium_returns(covariance, config.market_cap_weights)

    views = [
        MarketView(symbol="BTC/USD", expected_return=0.12, confidence=0.9, view_type="absolute")
    ]
    posterior = optimizer.calculate_posterior_returns(prior, views, config.tau)

    assert posterior.shape == prior.shape
    assert posterior[0] > prior[0]


def test_optimize_portfolio_respects_weight_bounds() -> None:
    config = BlackLittermanConfig(
        risk_aversion=2.5,
        market_cap_weights={"BTC/USD": 0.5, "ETH/USD": 0.3, "SOL/USD": 0.2},
    )
    optimizer = BlackLittermanOptimizer(
        symbols=["BTC/USD", "ETH/USD", "SOL/USD"],
        config=config,
        min_weight=0.05,
        max_weight=0.70,
    )
    covariance = np.array(
        [[0.05, 0.01, 0.015], [0.01, 0.04, 0.012], [0.015, 0.012, 0.07]],
        dtype=float,
    )
    returns = np.array([0.10, 0.08, 0.12], dtype=float)

    weights = optimizer.optimize_portfolio(returns, covariance, config.risk_aversion)

    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert all(0.05 - 1e-6 <= weight <= 0.70 + 1e-6 for weight in weights.values())


def test_end_to_end_market_data_and_ml_signals() -> None:
    config = BlackLittermanConfig(
        market_cap_weights={"BTC/USD": 0.5, "ETH/USD": 0.3, "SOL/USD": 0.2},
        tau=0.05,
    )
    optimizer = BlackLittermanOptimizer(
        symbols=["BTC/USD", "ETH/USD", "SOL/USD"],
        config=config,
        min_weight=0.0,
        max_weight=0.80,
    )

    rng = np.random.default_rng(42)
    market_data = {
        "BTC/USD": rng.normal(0.0010, 0.020, 80),
        "ETH/USD": rng.normal(0.0008, 0.022, 80),
        "SOL/USD": rng.normal(0.0014, 0.030, 80),
    }
    ml_signals = {
        "BTC/USD": {"predicted_return": 0.09, "confidence": 0.7},
        "ETH/USD": {"predicted_return": 0.06, "confidence": 0.6},
        "SOL/USD": {"predicted_return": 0.11, "confidence": 0.8},
    }

    weights = optimizer.optimize_from_market_data(market_data, ml_signals=ml_signals, blend_weight=0.25)

    assert set(weights.keys()) == {"BTC/USD", "ETH/USD", "SOL/USD"}
    assert abs(sum(weights.values()) - 1.0) < 1e-6
