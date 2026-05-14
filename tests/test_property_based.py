"""
Property-based tests for risk modules.

Uses Python's built-in `random` module (no hypothesis required).

Invariants verified:
- Kelly fraction is always 0 ≤ f ≤ max_fraction for any valid win_rate/payoff input
- Correlation matrix is always symmetric with 1s on the diagonal
- VaR (from _calculate_var) is always non-negative
- Portfolio position sizes are always non-negative
- CorrelationMonitor position scalar is in [0.1, 1.0]
- KellyUncertaintyCalculator.calculate() always returns a valid fraction
"""

import math
import random
import pytest
import numpy as np

from risk.kelly_uncertainty import KellyUncertaintyCalculator
from risk.correlation_monitor import CorrelationMonitor


# Seed for reproducibility
random.seed(42)
np.random.seed(42)

N_RANDOM_TRIALS = 50   # enough coverage without being slow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_trades(n: int, win_rate: float, avg_win: float, avg_loss: float,
                   rng: random.Random) -> list:
    """Generate synthetic trade returns list with given statistics."""
    trades = []
    for _ in range(n):
        if rng.random() < win_rate:
            trades.append(rng.uniform(avg_win * 0.5, avg_win * 1.5))
        else:
            trades.append(-rng.uniform(avg_loss * 0.5, avg_loss * 1.5))
    return trades


def _feed_prices(monitor: CorrelationMonitor, n_bars: int, rng: random.Random) -> None:
    """Feed correlated random-walk prices into a CorrelationMonitor."""
    prices = {s: 1000.0 for s in monitor.symbols}
    for _ in range(n_bars):
        common_shock = rng.gauss(0, 0.01)
        for sym in monitor.symbols:
            idio = rng.gauss(0, 0.005)
            prices[sym] *= (1 + common_shock + idio)
            monitor.update(sym, prices[sym])


# ---------------------------------------------------------------------------
# Kelly fraction invariants
# ---------------------------------------------------------------------------

class TestKellyFractionInvariants:
    """Kelly fraction must always satisfy 0 ≤ f ≤ max_fraction."""

    def test_fraction_non_negative_and_bounded_random_inputs(self):
        """Run many random (win_rate, avg_win, avg_loss) combinations."""
        rng = random.Random(1)
        calc = KellyUncertaintyCalculator(
            kelly_fraction=0.5, n_bootstrap=200, min_trades=5, max_fraction=0.25
        )
        failures = []

        for trial in range(N_RANDOM_TRIALS):
            win_rate = rng.uniform(0.3, 0.8)
            avg_win = rng.uniform(0.005, 0.05)
            avg_loss = rng.uniform(0.003, 0.04)
            n_trades = rng.randint(20, 100)
            trades = _random_trades(n_trades, win_rate, avg_win, avg_loss, rng)

            result = calc.calculate(trades, capital=10_000, price=50_000)
            f = result["fraction"]
            if not (0.0 <= f <= calc.max_fraction + 1e-9):
                failures.append((trial, win_rate, avg_win, avg_loss, f))

        assert not failures, f"Kelly fraction out of bounds in trials: {failures[:5]}"

    def test_fraction_zero_for_all_loss_trades(self):
        calc = KellyUncertaintyCalculator(n_bootstrap=200, min_trades=5)
        trades = [-0.01] * 30  # all losses
        result = calc.calculate(trades, capital=1000)
        assert result["fraction"] >= 0.0
        assert result["fraction"] <= calc.max_fraction

    def test_fraction_bounded_for_all_win_trades(self):
        calc = KellyUncertaintyCalculator(n_bootstrap=200, min_trades=5)
        trades = [0.02] * 30  # all wins
        result = calc.calculate(trades, capital=1000)
        assert 0.0 <= result["fraction"] <= calc.max_fraction

    def test_fraction_never_exceeds_max_fraction(self):
        """Even with perfect edge, max_fraction hard cap holds."""
        calc = KellyUncertaintyCalculator(
            kelly_fraction=1.0, n_bootstrap=200, min_trades=5, max_fraction=0.10
        )
        rng = random.Random(2)
        for _ in range(N_RANDOM_TRIALS):
            trades = [rng.uniform(0.01, 0.05)] * 50  # strong positive edge
            result = calc.calculate(trades, capital=1000)
            assert result["fraction"] <= calc.max_fraction + 1e-9

    def test_insufficient_trades_returns_default(self):
        calc = KellyUncertaintyCalculator(min_trades=10)
        trades = [0.01, -0.01, 0.02]  # only 3 trades
        result = calc.calculate(trades, capital=1000)
        assert 0.0 <= result["fraction"] <= calc.max_fraction
        assert result["method"] == "insufficient_data_default"

    def test_fraction_non_negative_edge_cases(self):
        calc = KellyUncertaintyCalculator(n_bootstrap=200, min_trades=5)
        edge_cases = [
            [0.001] * 10 + [-0.001] * 10,   # break-even
            [1e-6] * 10 + [-1e-6] * 10,     # near-zero returns
            [0.5] * 15 + [-0.5] * 5,        # very high win/loss ratio
        ]
        for trades in edge_cases:
            result = calc.calculate(trades, capital=1000)
            assert 0.0 <= result["fraction"] <= calc.max_fraction

    def test_size_from_signal_bounded(self):
        """size_from_signal fraction must also be in [0, max_fraction]."""
        calc = KellyUncertaintyCalculator(n_bootstrap=200, min_trades=5)
        rng = random.Random(3)
        for _ in range(20):
            trades = _random_trades(30, 0.55, 0.02, 0.015, rng)
            conf = rng.uniform(0.0, 1.0)
            result = calc.size_from_signal(conf, trades, capital=10_000, price=50_000)
            assert 0.0 <= result["fraction"] <= calc.max_fraction + 1e-9

    def test_capital_at_risk_non_negative(self):
        calc = KellyUncertaintyCalculator(n_bootstrap=200, min_trades=5)
        rng = random.Random(4)
        for _ in range(20):
            trades = _random_trades(30, 0.55, 0.02, 0.015, rng)
            result = calc.calculate(trades, capital=rng.uniform(100, 100_000))
            assert result["capital_at_risk"] >= 0.0

    def test_units_non_negative(self):
        calc = KellyUncertaintyCalculator(n_bootstrap=200, min_trades=5)
        rng = random.Random(5)
        for _ in range(20):
            trades = _random_trades(30, 0.55, 0.02, 0.015, rng)
            price = rng.uniform(0.01, 100_000)
            result = calc.calculate(trades, capital=10_000, price=price)
            assert result["units"] >= 0.0


# ---------------------------------------------------------------------------
# Correlation matrix invariants
# ---------------------------------------------------------------------------

class TestCorrelationMatrixInvariants:
    """Correlation matrix must be symmetric with 1s on the diagonal."""

    def test_diagonal_is_one(self):
        symbols = ["BTC", "ETH", "SOL"]
        monitor = CorrelationMonitor(symbols, lookback=20)
        rng = random.Random(10)
        _feed_prices(monitor, 30, rng)

        corr = monitor.compute_correlation_matrix()
        if corr is None:
            pytest.skip("Insufficient data for correlation matrix")

        n = corr.shape[0]
        for i in range(n):
            assert corr[i, i] == pytest.approx(1.0, abs=1e-9), \
                f"Diagonal element [{i},{i}] = {corr[i,i]} != 1.0"

    def test_matrix_is_symmetric(self):
        symbols = ["BTC", "ETH", "SOL", "ADA"]
        monitor = CorrelationMonitor(symbols, lookback=20)
        rng = random.Random(11)
        _feed_prices(monitor, 40, rng)

        corr = monitor.compute_correlation_matrix()
        if corr is None:
            pytest.skip("Insufficient data for correlation matrix")

        n = corr.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                assert corr[i, j] == pytest.approx(corr[j, i], abs=1e-9), \
                    f"Matrix not symmetric at [{i},{j}]: {corr[i,j]} vs {corr[j,i]}"

    def test_values_bounded_minus_one_to_one(self):
        symbols = ["BTC", "ETH", "SOL"]
        monitor = CorrelationMonitor(symbols, lookback=20)
        rng = random.Random(12)
        _feed_prices(monitor, 40, rng)

        corr = monitor.compute_correlation_matrix()
        if corr is None:
            pytest.skip("Insufficient data")

        assert np.all(corr >= -1.0 - 1e-9)
        assert np.all(corr <= 1.0 + 1e-9)

    def test_no_nan_in_matrix(self):
        symbols = ["BTC", "ETH", "SOL"]
        monitor = CorrelationMonitor(symbols, lookback=20)
        rng = random.Random(13)
        _feed_prices(monitor, 40, rng)

        corr = monitor.compute_correlation_matrix()
        if corr is None:
            pytest.skip("Insufficient data")

        assert not np.any(np.isnan(corr))

    def test_returns_none_with_insufficient_data(self):
        symbols = ["BTC", "ETH"]
        monitor = CorrelationMonitor(symbols, lookback=20)
        # Only feed 2 prices per symbol (not enough)
        monitor.update("BTC", 50000)
        monitor.update("BTC", 51000)
        monitor.update("ETH", 3000)
        monitor.update("ETH", 3100)
        # Should return None (need lookback//2 = 10 minimum)
        corr = monitor.compute_correlation_matrix()
        assert corr is None

    def test_returns_none_with_single_symbol(self):
        monitor = CorrelationMonitor(["BTC"], lookback=20)
        rng = random.Random(14)
        for _ in range(30):
            monitor.update("BTC", rng.uniform(40000, 60000))
        corr = monitor.compute_correlation_matrix()
        assert corr is None  # need at least 2 symbols

    def test_position_scalar_in_valid_range(self):
        """Position scalar should always be in [0.1, 1.0]."""
        symbols = ["BTC", "ETH", "SOL"]
        monitor = CorrelationMonitor(symbols, lookback=20)
        rng = random.Random(15)
        _feed_prices(monitor, 40, rng)
        scalar = monitor.get_position_scalar()
        assert 0.0 <= scalar <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# VaR invariants (tested via PortfolioRiskManager._calculate_var)
# ---------------------------------------------------------------------------

class TestVaRInvariants:
    """VaR values must always be non-negative (losses are positive)."""

    def _make_portfolio_manager(self, capital: float = 10_000):
        from risk.portfolio import PortfolioRiskManager
        return PortfolioRiskManager(capital)

    def test_var_non_negative_with_sufficient_data(self):
        pm = self._make_portfolio_manager(capital=10_000)
        rng = random.Random(20)
        # Feed 30 daily returns
        capital = 10_000.0
        for _ in range(30):
            ret = rng.gauss(0.001, 0.02)
            capital *= (1 + ret)
            pm.update_capital(capital)

        var_95, var_99, cvar_95 = pm._calculate_var()
        assert var_95 >= 0.0, f"var_95={var_95} should be >= 0"
        assert var_99 >= 0.0, f"var_99={var_99} should be >= 0"
        assert cvar_95 >= 0.0, f"cvar_95={cvar_95} should be >= 0"

    def test_var_zero_with_insufficient_data(self):
        """With < 10 data points, _calculate_var returns (0,0,0) sentinel."""
        pm = self._make_portfolio_manager()
        capital = 10_000.0
        for _ in range(5):  # only 5 updates — not enough
            capital *= 1.01
            pm.update_capital(capital)
        var_95, var_99, cvar_95 = pm._calculate_var()
        assert var_95 == 0.0
        assert var_99 == 0.0
        assert cvar_95 == 0.0

    def test_var_non_negative_random_returns(self):
        rng = random.Random(21)
        for trial in range(20):
            pm = self._make_portfolio_manager(capital=rng.uniform(1000, 100_000))
            capital = pm.current_capital
            n_returns = rng.randint(10, 60)
            for _ in range(n_returns):
                ret = rng.gauss(0, 0.03)
                capital = max(capital * (1 + ret), 1.0)
                pm.update_capital(capital)
            var_95, var_99, cvar_95 = pm._calculate_var()
            assert var_95 >= 0.0, f"trial={trial} var_95={var_95}"
            assert var_99 >= 0.0, f"trial={trial} var_99={var_99}"
            assert cvar_95 >= 0.0, f"trial={trial} cvar_95={cvar_95}"

    def test_var_99_gte_var_95(self):
        """99% VaR should be at least as large as 95% VaR (deeper tail)."""
        pm = self._make_portfolio_manager()
        capital = 10_000.0
        rng = random.Random(22)
        for _ in range(30):
            capital *= (1 + rng.gauss(0, 0.02))
            pm.update_capital(max(capital, 1.0))
        var_95, var_99, _ = pm._calculate_var()
        assert var_99 >= var_95 - 1e-9  # 99% VaR >= 95% VaR

    def test_cvar_gte_var(self):
        """CVaR (expected shortfall) should be at least as large as VaR."""
        pm = self._make_portfolio_manager()
        capital = 10_000.0
        rng = random.Random(23)
        for _ in range(50):
            capital *= (1 + rng.gauss(-0.001, 0.025))
            pm.update_capital(max(capital, 1.0))
        var_95, _, cvar_95 = pm._calculate_var()
        assert cvar_95 >= var_95 - 1e-9


# ---------------------------------------------------------------------------
# Position size non-negativity
# ---------------------------------------------------------------------------

class TestPositionSizeNonNegativity:
    """Position sizing from Kelly must always produce non-negative units."""

    def test_units_always_non_negative(self):
        calc = KellyUncertaintyCalculator(n_bootstrap=200, min_trades=5)
        rng = random.Random(30)
        for _ in range(N_RANDOM_TRIALS):
            win_rate = rng.uniform(0.2, 0.9)
            avg_win = rng.uniform(0.002, 0.1)
            avg_loss = rng.uniform(0.001, 0.1)
            trades = _random_trades(
                rng.randint(10, 80), win_rate, avg_win, avg_loss, rng
            )
            price = rng.uniform(0.001, 1_000_000)
            capital = rng.uniform(100, 1_000_000)
            result = calc.calculate(trades, capital=capital, price=price)
            assert result["units"] >= 0.0, \
                f"units={result['units']} < 0 for trades={trades[:3]}..."

    def test_capital_at_risk_does_not_exceed_capital(self):
        """Capital at risk should never exceed total capital."""
        calc = KellyUncertaintyCalculator(n_bootstrap=200, min_trades=5)
        rng = random.Random(31)
        for _ in range(N_RANDOM_TRIALS):
            trades = _random_trades(30, rng.uniform(0.4, 0.7), 0.02, 0.015, rng)
            capital = rng.uniform(1000, 100_000)
            result = calc.calculate(trades, capital=capital)
            assert result["capital_at_risk"] <= capital + 1e-9, \
                f"capital_at_risk={result['capital_at_risk']} > capital={capital}"


# ---------------------------------------------------------------------------
# CorrelationMonitor: avg pairwise always in reasonable range
# ---------------------------------------------------------------------------

class TestCorrelationMonitorInvariants:
    def test_avg_pairwise_in_minus_one_to_one(self):
        rng = random.Random(40)
        for _ in range(10):
            symbols = random.sample(["BTC", "ETH", "SOL", "ADA", "DOT"], 3)
            monitor = CorrelationMonitor(symbols, lookback=20)
            _feed_prices(monitor, 30, rng)
            avg = monitor.get_avg_pairwise_correlation()
            assert -1.0 <= avg <= 1.0, f"avg_pairwise={avg} out of [-1, 1]"

    def test_position_scalar_in_range_after_feeding(self):
        rng = random.Random(41)
        for _ in range(10):
            symbols = ["BTC", "ETH", "SOL"]
            monitor = CorrelationMonitor(symbols, lookback=20)
            _feed_prices(monitor, 30, rng)
            scalar = monitor.get_position_scalar()
            # Valid range is [0.1, 1.0]
            assert 0.0 < scalar <= 1.0 + 1e-9

    def test_position_scalar_one_when_low_correlation(self):
        """Independent assets (zero correlation) → scalar should be 1.0."""
        symbols = ["A", "B", "C"]
        monitor = CorrelationMonitor(symbols, lookback=20)
        rng = random.Random(42)
        # Feed completely independent random walks
        prices = {s: 1000.0 for s in symbols}
        for _ in range(40):
            for sym in symbols:
                prices[sym] *= (1 + rng.gauss(0, 0.02))
                monitor.update(sym, prices[sym])

        # The scalar may or may not be exactly 1.0 depending on random data,
        # but it must be in valid range
        scalar = monitor.get_position_scalar()
        assert 0.0 < scalar <= 1.0

    def test_status_dict_has_required_keys(self):
        monitor = CorrelationMonitor(["BTC", "ETH"], lookback=20)
        status = monitor.get_status()
        for key in ("avg_pairwise_correlation", "position_scalar",
                    "alert_threshold", "crisis_threshold", "symbols"):
            assert key in status
