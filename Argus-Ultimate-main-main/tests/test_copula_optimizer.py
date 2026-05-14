from __future__ import annotations

import numpy as np

from portfolio.copula_optimizer import CopulaBlackLitterman, CopulaFitter, CopulaType, MacroeconomicViews, PortfolioOptimizer


def test_copula_fitter_student_t_exposes_tail_dependence() -> None:
    rng = np.random.default_rng(7)
    returns = rng.standard_t(df=6, size=(240, 3)) * np.array([0.015, 0.02, 0.018])

    fitter = CopulaFitter()
    correlation = fitter.fit_student_t_copula(returns)
    tail = fitter.get_tail_dependence()

    assert correlation.shape == (3, 3)
    assert tail["lower"].shape == (3, 3)
    assert tail["upper"].shape == (3, 3)
    assert np.all(tail["lower"] >= 0.0)


def test_copula_black_litterman_optimizes_long_only_weights() -> None:
    rng = np.random.default_rng(21)
    returns = rng.normal(loc=[0.0012, 0.0009, 0.0014], scale=[0.02, 0.018, 0.025], size=(180, 3))

    optimizer = CopulaBlackLitterman({"BTC": 0.5, "ETH": 0.3, "SOL": 0.2}, risk_aversion=2.5, tau=0.05)
    optimizer._set_returns(returns, symbols=["BTC", "ETH", "SOL"])
    optimizer.set_copula(CopulaType.STUDENT_T)
    optimizer.set_views([
        {"assets": ["BTC"], "coeffs": [1.0], "return": 0.12, "confidence": 0.8},
        {"assets": ["SOL", "ETH"], "coeffs": [1.0, -1.0], "return": 0.04, "confidence": 0.65},
    ], [0.8, 0.65])

    weights = optimizer.optimize()
    frontier = optimizer.get_efficient_frontier(5)
    cvar = optimizer.compute_cvar()

    assert set(weights.keys()) == {"BTC", "ETH", "SOL"}
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert all(weight >= 0.0 for weight in weights.values())
    assert len(frontier) == 5
    assert cvar >= 0.0


def test_macro_views_and_portfolio_optimizer_work_end_to_end() -> None:
    rng = np.random.default_rng(99)
    returns = {
        "BTC": rng.normal(0.0014, 0.025, 220),
        "ETH": rng.normal(0.0010, 0.022, 220),
        "GLD": rng.normal(0.0005, 0.012, 220),
    }

    macro = MacroeconomicViews()
    macro.add_cycle_view("expansion", {"BTC": 0.10, "ETH": 0.08, "GLD": -0.02})
    macro.add_inflation_view({"GLD": 0.03})
    macro.add_rate_view({"BTC": 0.01})

    optimizer = PortfolioOptimizer({"BTC": 0.45, "ETH": 0.35, "GLD": 0.20})
    weights = optimizer.optimize_with_copula(returns, macro.generate_bl_views(), CopulaType.VINE)
    backtest = optimizer.backtest(20, 200)
    metrics = optimizer.compute_risk_metrics()

    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert backtest["equity_curve"].size == 180
    assert {"var_95", "cvar_95", "max_drawdown", "sharpe"}.issubset(metrics.keys())
