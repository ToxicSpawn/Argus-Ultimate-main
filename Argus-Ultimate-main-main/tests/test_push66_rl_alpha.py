"""Push 66 — Tests: RL agent, Kelly sizer, CVaR halt,
per-strategy risk, correlation monitor, OBI signal,
adverse selection gate, funding arb strategy. 26 tests.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Bar:
    def __init__(self, o=100.0, h=102.0, l=98.0, c=101.0, sym="BTCUSDT"):
        self.open = o; self.high = h; self.low = l
        self.close = c; self.symbol = sym


def _make_feed(n=50):
    rng = np.random.default_rng(42)
    bars = []
    price = 50_000.0
    for _ in range(n):
        ret = rng.normal(0, 0.005)
        o = price
        c = price * (1 + ret)
        bars.append(_Bar(o=o, h=max(o,c)*1.001, l=min(o,c)*0.999, c=c))
        price = c
    return bars


# ---------------------------------------------------------------------------
# RLConfig (2)
# ---------------------------------------------------------------------------

class TestRLConfig:
    def test_default_algorithm_ppo(self):
        from core.rl.rl_config import RLConfig
        assert RLConfig().algorithm == "PPO"

    def test_invalid_algorithm_raises(self):
        from core.rl.rl_config import RLConfig
        with pytest.raises(AssertionError):
            RLConfig(algorithm="DQN")


# ---------------------------------------------------------------------------
# FeatureBuilder (3)
# ---------------------------------------------------------------------------

class TestFeatureBuilder:
    def test_build_returns_7dim(self):
        from core.rl.rl_feature_builder import FeatureBuilder
        fb = FeatureBuilder()
        bar = _Bar()
        fb.update(bar)
        obs = fb.build(bar, inventory=0.0, pnl_norm=0.0)
        assert obs.shape == (7,)

    def test_dtype_float32(self):
        from core.rl.rl_feature_builder import FeatureBuilder
        fb = FeatureBuilder()
        bar = _Bar()
        fb.update(bar)
        assert fb.build(bar, 0.0, 0.0).dtype == np.float32

    def test_rsi_converges_after_warmup(self):
        from core.rl.rl_feature_builder import FeatureBuilder
        fb = FeatureBuilder()
        for _ in range(20):
            fb.update(_Bar(c=100.0 + np.random.randn()))
        rsi = fb._rsi()
        assert 0.0 <= rsi <= 100.0


# ---------------------------------------------------------------------------
# ArgusRLEnv (5)
# ---------------------------------------------------------------------------

class TestArgusRLEnv:
    def test_reset_returns_obs_and_dict(self):
        from core.rl.rl_env import ArgusRLEnv
        env = ArgusRLEnv(_make_feed())
        obs, info = env.reset()
        assert obs.shape == (7,)
        assert isinstance(info, dict)

    def test_step_returns_5tuple(self):
        from core.rl.rl_env import ArgusRLEnv
        env = ArgusRLEnv(_make_feed())
        env.reset()
        obs, reward, terminated, truncated, info = env.step(np.array([0.5]))
        assert obs.shape == (7,)
        assert isinstance(reward, float)

    def test_equity_changes_after_step(self):
        from core.rl.rl_env import ArgusRLEnv
        env = ArgusRLEnv(_make_feed(100))
        env.reset()
        for _ in range(10):
            env.step(np.array([0.8]))
        assert env.current_equity != 10_000.0

    def test_terminates_at_end_of_feed(self):
        from core.rl.rl_env import ArgusRLEnv
        feed = _make_feed(5)
        env = ArgusRLEnv(feed)
        env.reset()
        terminated = False
        for _ in range(10):
            _, _, terminated, _, _ = env.step(np.array([0.0]))
            if terminated:
                break
        assert terminated

    def test_total_return_property(self):
        from core.rl.rl_env import ArgusRLEnv
        env = ArgusRLEnv(_make_feed())
        env.reset()
        assert isinstance(env.total_return, float)


# ---------------------------------------------------------------------------
# KellySizer (4)
# ---------------------------------------------------------------------------

class TestKellySizer:
    def test_positive_edge_gives_position(self):
        from core.risk.kelly_sizer import KellySizer
        k = KellySizer()
        result = k.size(10_000.0, win_rate=0.6, avg_win=0.02, avg_loss=0.01)
        assert result.position_usd > 0

    def test_negative_edge_gives_zero(self):
        from core.risk.kelly_sizer import KellySizer
        k = KellySizer()
        result = k.size(10_000.0, win_rate=0.3, avg_win=0.01, avg_loss=0.02)
        assert result.position_usd == 0.0

    def test_quarter_kelly_safety(self):
        from core.risk.kelly_sizer import KellySizer
        k = KellySizer(safety_multiplier=0.25)
        result = k.size(10_000.0, win_rate=0.6, avg_win=0.03, avg_loss=0.01)
        assert result.safe_fraction <= result.kelly_fraction

    def test_size_from_trades(self):
        from core.risk.kelly_sizer import KellySizer
        rng = np.random.default_rng(0)
        trades = list(rng.normal(0.005, 0.02, 50))
        result = KellySizer().size_from_trades(10_000.0, trades)
        assert result.position_usd >= 0


# ---------------------------------------------------------------------------
# CVaRHalt (3)
# ---------------------------------------------------------------------------

class TestCVaRHalt:
    def test_no_halt_with_few_samples(self):
        from core.risk.cvar_halt import CVaRHalt
        halt = CVaRHalt()
        for _ in range(5):
            halt.record_return(-0.1)
        snap = halt.evaluate()
        assert not snap.halted

    def test_halts_on_large_losses(self):
        from core.risk.cvar_halt import CVaRHalt
        halt = CVaRHalt(max_cvar_95=0.02, min_samples=5)
        for _ in range(25):
            halt.record_return(-0.15)
        snap = halt.evaluate()
        assert snap.halted

    def test_reset_clears_halt(self):
        from core.risk.cvar_halt import CVaRHalt
        halt = CVaRHalt(max_cvar_95=0.02, min_samples=5)
        for _ in range(25):
            halt.record_return(-0.15)
        halt.evaluate()
        halt.reset()
        assert not halt.halted


# ---------------------------------------------------------------------------
# PerStrategyRisk (2)
# ---------------------------------------------------------------------------

class TestPerStrategyRisk:
    def test_active_when_within_limits(self):
        from core.risk.per_strategy_risk import PerStrategyRisk
        psr = PerStrategyRisk()
        active = psr.update("MomentumStrategy", 10_000.0, 50.0)
        assert active

    def test_halts_on_drawdown(self):
        from core.risk.per_strategy_risk import PerStrategyRisk
        from core.risk.per_strategy_risk import StrategyRiskBudget
        psr = PerStrategyRisk()
        budget = StrategyRiskBudget("M", max_drawdown_pct=2.0)
        budget.peak_equity = 10_000.0
        psr._budgets["M"] = budget
        active = psr.update("M", 9_700.0, -300.0)  # 3% drawdown
        assert not active


# ---------------------------------------------------------------------------
# CorrelationMonitor (2)
# ---------------------------------------------------------------------------

class TestCorrelationMonitor:
    def test_no_halt_with_one_symbol(self):
        from core.risk.correlation_monitor import CorrelationMonitor
        mon = CorrelationMonitor(min_samples=5)
        for i in range(10):
            mon.record_return("BTCUSDT", float(i) * 0.001)
        snap = mon.evaluate(["BTCUSDT"])
        assert not snap.halt_new_positions

    def test_detects_high_correlation(self):
        from core.risk.correlation_monitor import CorrelationMonitor
        mon = CorrelationMonitor(halt_threshold=0.8, min_samples=5)
        rets = list(np.linspace(0.01, 0.05, 30))
        for r in rets:
            mon.record_return("BTCUSDT", r)
            mon.record_return("ETHUSDT", r * 1.01)  # near-perfect correlation
        snap = mon.evaluate(["BTCUSDT", "ETHUSDT"])
        assert snap.halt_new_positions


# ---------------------------------------------------------------------------
# OBICalculator (3)
# ---------------------------------------------------------------------------

class TestOBICalculator:
    def test_balanced_book_gives_zero_obi(self):
        from core.signals.obi_signal import OBICalculator
        calc = OBICalculator()
        bids = [(99.0, 10.0), (98.0, 10.0)]
        asks = [(101.0, 10.0), (102.0, 10.0)]
        sig = calc.compute(bids, asks)
        assert abs(sig.obi) < 1e-9

    def test_bid_heavy_gives_buy_signal(self):
        from core.signals.obi_signal import OBICalculator
        calc = OBICalculator(neutral_band=0.05)
        bids = [(99.0, 100.0), (98.0, 50.0)]
        asks = [(101.0, 10.0), (102.0, 5.0)]
        sig = calc.compute(bids, asks)
        assert sig.direction == "BUY"
        assert sig.obi > 0

    def test_fair_price_close_to_mid(self):
        from core.signals.obi_signal import OBICalculator
        calc = OBICalculator()
        bids = [(99.0, 10.0)]
        asks = [(101.0, 10.0)]
        sig = calc.compute(bids, asks)
        assert abs(sig.fair_price - 100.0) < 1.0


# ---------------------------------------------------------------------------
# AdverseSelectionGate (1)
# ---------------------------------------------------------------------------

class TestAdverseSelectionGate:
    def test_heuristic_fallback_works(self):
        from core.signals.adverse_selection_gate import AdverseSelectionGate
        gate = AdverseSelectionGate(threshold=0.65)
        result = gate.evaluate(
            obi=0.3, spread_norm=0.001,
            trade_flow_imbalance=0.2, depth_ratio=1.1,
            recent_vol=0.005,
        )
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.fill_recommended, bool)


# ---------------------------------------------------------------------------
# FundingArbStrategy (1)
# ---------------------------------------------------------------------------

class TestFundingArbStrategy:
    def test_enters_on_high_funding(self):
        from core.strategy.builtin.funding_arb_strategy import FundingArbStrategy
        strat = FundingArbStrategy(min_apy=0.10)
        # 0.001 per 8h * 3/day * 365 = 109.5% APY
        sig = strat.on_funding_tick("BTCUSDT", 0.001, 50_000.0, 50_010.0)
        assert sig.action == "ENTER"

    def test_holds_after_entry(self):
        from core.strategy.builtin.funding_arb_strategy import FundingArbStrategy
        strat = FundingArbStrategy(min_apy=0.10)
        strat.on_funding_tick("BTCUSDT", 0.001, 50_000.0, 50_010.0)
        sig = strat.on_funding_tick("BTCUSDT", 0.001, 50_000.0, 50_010.0)
        assert sig.action == "HOLD"
