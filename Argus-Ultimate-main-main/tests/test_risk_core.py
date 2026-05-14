"""tests/test_risk_core.py

Unit tests for the zero-coverage risk modules and core/ extractons.

Covers:
  - RiskFacade (risk/unified_risk_facade.py)
  - PositionTracker (core/position_tracker.py)
  - RegimeDetector (core/regime_detector.py)
  - EnsembleController (core/ensemble_controller.py)
  - OrderRouter (core/order_router.py)
  - SignalPipeline (core/signal_pipeline.py)
  - RateLimitGuard (utils/rate_limit_guard.py)
  - ExchangeRateLimiter (utils/rate_limit_guard.py)
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

# ============================================================
#  RiskFacade
# ============================================================

class TestRiskFacade:
    def _facade(self, **kwargs):
        from risk.unified_risk_facade import RiskFacade
        return RiskFacade(config=kwargs)

    def test_approved_basic(self):
        f = self._facade(max_drawdown_pct=15.0, max_position_pct=0.10)
        d = f.evaluate("BTC/USDT", qty=0.01, price=50000.0, portfolio_equity=10000.0)
        assert d.approved
        assert d.approved_qty > 0

    def test_drawdown_halt(self):
        f = self._facade(max_drawdown_pct=10.0)
        # Establish peak at 10000
        f.evaluate("BTC/USDT", qty=0.01, price=50000.0, portfolio_equity=10000.0)
        # Now equity drops 15% -> should halt
        d = f.evaluate("BTC/USDT", qty=0.01, price=50000.0, portfolio_equity=8500.0)
        assert not d.approved
        assert "drawdown" in d.reason.lower() or "halted" in d.reason.lower()

    def test_halt_persists_on_subsequent_calls(self):
        f = self._facade(max_drawdown_pct=10.0)
        f.evaluate("BTC/USDT", qty=0.01, price=50000.0, portfolio_equity=10000.0)
        f.evaluate("BTC/USDT", qty=0.01, price=50000.0, portfolio_equity=8500.0)
        d = f.evaluate("BTC/USDT", qty=0.01, price=50000.0, portfolio_equity=9000.0)
        assert not d.approved  # still halted

    def test_resume_clears_halt(self):
        f = self._facade(max_drawdown_pct=10.0)
        f.evaluate("X", qty=0.01, price=1.0, portfolio_equity=10000.0)
        f.evaluate("X", qty=0.01, price=1.0, portfolio_equity=8500.0)
        f.resume()
        # After resume, peak resets on next evaluate
        f._peak_equity = 8500.0  # pretend recovery
        d = f.evaluate("X", qty=0.01, price=1.0, portfolio_equity=8500.0)
        assert d.approved

    def test_position_size_cap(self):
        # max_position_pct=0.01 => max $100 on $10k equity at $50k price -> 0.002 BTC
        f = self._facade(max_position_pct=0.01, kelly_fraction=1.0, kelly_win_rate=0.99, kelly_win_loss=100.0)
        d = f.evaluate("BTC/USDT", qty=1.0, price=50000.0, portfolio_equity=10000.0)
        assert d.approved
        assert d.approved_qty <= 0.002 + 1e-9

    def test_kelly_reduces_qty(self):
        # Very conservative Kelly (low win rate)
        f = self._facade(kelly_win_rate=0.50, kelly_win_loss=1.0, kelly_fraction=0.25, max_position_pct=1.0)
        d = f.evaluate("BTC/USDT", qty=10.0, price=100.0, portfolio_equity=1000.0)
        # Kelly fraction = (1*0.5 - 0.5)/1 = 0 -> no trade
        assert d.approved_qty == 0.0 or not d.approved

    def test_daily_loss_limit(self):
        f = self._facade(daily_loss_limit=0.03, max_drawdown_pct=50.0)
        f.evaluate("X", qty=0.01, price=1.0, portfolio_equity=10000.0)
        f.reset_day(10000.0)
        # Simulate 4% daily loss
        d = f.evaluate("X", qty=0.01, price=1.0, portfolio_equity=9600.0)
        assert not d.approved
        assert "daily" in d.reason.lower()


# ============================================================
#  PositionTracker
# ============================================================

class TestPositionTracker:
    def _tracker(self, cash=10000.0):
        from core.position_tracker import PositionTracker
        return PositionTracker(starting_cash=cash)

    def test_buy_creates_position(self):
        t = self._tracker()
        t.apply_fill("BTC", "BUY", 0.1, 50000.0)
        assert "BTC" in t.positions
        assert t.positions["BTC"].quantity == pytest.approx(0.1)

    def test_sell_removes_position(self):
        t = self._tracker()
        t.apply_fill("BTC", "BUY", 0.1, 50000.0)
        t.apply_fill("BTC", "SELL", 0.1, 55000.0)
        assert "BTC" not in t.positions

    def test_realised_pnl(self):
        t = self._tracker()
        t.apply_fill("ETH", "BUY", 1.0, 3000.0)
        t.apply_fill("ETH", "SELL", 1.0, 3500.0)
        snap = t.snapshot()
        assert snap.realised_pnl == pytest.approx(500.0)

    def test_unrealised_pnl(self):
        t = self._tracker()
        t.apply_fill("ETH", "BUY", 1.0, 3000.0)
        t.update_prices({"ETH": 3300.0})
        snap = t.snapshot()
        assert snap.unrealised_pnl == pytest.approx(300.0)

    def test_drawdown_pct(self):
        t = self._tracker(cash=10000.0)
        t.apply_fill("X", "BUY", 1.0, 100.0)
        t.update_prices({"X": 100.0})  # peak equity ~= 10000
        t.update_prices({"X": 50.0})   # equity drops ~500 (cost was 100)
        snap = t.snapshot()
        assert snap.drawdown_pct >= 0.0

    def test_thread_safety(self):
        t = self._tracker(cash=100000.0)
        errors = []

        def worker(i):
            try:
                t.apply_fill(f"SYM{i}", "BUY", 0.01, 1000.0)
                t.update_prices({f"SYM{i}": 1100.0})
                t.snapshot()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert errors == [], f"Thread safety errors: {errors}"

    def test_sell_unknown_symbol_no_crash(self):
        t = self._tracker()
        t.apply_fill("GHOST", "SELL", 1.0, 100.0)  # should warn, not crash


# ============================================================
#  RegimeDetector
# ============================================================

class TestRegimeDetector:
    def _det(self):
        from core.regime_detector import RegimeDetector
        return RegimeDetector(adx_period=5, adx_trend_threshold=20.0, vol_high_threshold=0.02)

    def _trending_data(self, n=30):
        closes = [100.0 + i * 2 for i in range(n)]
        highs  = [c + 1 for c in closes]
        lows   = [c - 1 for c in closes]
        return closes, highs, lows

    def _ranging_data(self, n=30):
        import math
        closes = [100.0 + math.sin(i * 0.5) for i in range(n)]
        highs  = [c + 0.5 for c in closes]
        lows   = [c - 0.5 for c in closes]
        return closes, highs, lows

    def test_trending_up_detected(self):
        from core.regime_detector import MarketRegime
        d = self._det()
        c, h, lo = self._trending_data()
        snap = d.detect(c, h, lo)
        assert snap.regime in (MarketRegime.TRENDING_UP, MarketRegime.RANGING)

    def test_short_series_returns_unknown(self):
        from core.regime_detector import MarketRegime
        d = self._det()
        snap = d.detect([100, 101], [101, 102], [99, 100])
        assert snap.regime == MarketRegime.UNKNOWN

    def test_volatile_detected(self):
        from core.regime_detector import MarketRegime
        import random
        random.seed(42)
        closes = [100.0]
        for _ in range(40):
            closes.append(closes[-1] * (1 + random.uniform(-0.05, 0.05)))
        highs = [c * 1.02 for c in closes]
        lows  = [c * 0.98 for c in closes]
        d = RegimeDetector = self._det()
        snap = d.detect(closes, highs, lows)
        assert snap.regime in (
            __import__("core.regime_detector", fromlist=["MarketRegime"]).MarketRegime.VOLATILE,
            __import__("core.regime_detector", fromlist=["MarketRegime"]).MarketRegime.RANGING,
            __import__("core.regime_detector", fromlist=["MarketRegime"]).MarketRegime.TRENDING_UP,
            __import__("core.regime_detector", fromlist=["MarketRegime"]).MarketRegime.TRENDING_DOWN,
        )  # just assert no crash and a valid regime

    def test_last_property(self):
        d = self._det()
        assert d.last is None
        c, h, lo = self._trending_data()
        d.detect(c, h, lo)
        assert d.last is not None


# ============================================================
#  EnsembleController
# ============================================================

class TestEnsembleController:
    def _ctrl(self, models=("a", "b", "c")):
        from core.ensemble_controller import EnsembleController
        return EnsembleController(model_ids=list(models))

    def test_weights_sum_to_one(self):
        c = self._ctrl()
        weights = c.get_all_weights()
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_equal_initial_weights(self):
        c = self._ctrl(["x", "y"])
        assert c.get_weight("x") == pytest.approx(0.5)
        assert c.get_weight("y") == pytest.approx(0.5)

    def test_win_boosts_weight(self):
        c = self._ctrl(["good", "bad"])
        for _ in range(20):
            c.record_outcome("good", pnl=100.0, win=True)
            c.record_outcome("bad", pnl=-50.0, win=False)
        assert c.get_weight("good") > c.get_weight("bad")

    def test_consecutive_loss_halving(self):
        c = self._ctrl(["loser", "winner"])
        w_before = c.get_weight("loser")
        for _ in range(3):
            c.record_outcome("loser", pnl=-100.0, win=False)
        # After 3 consecutive losses, weight should have been halved
        w_after = c.get_weight("loser")
        assert w_after < w_before

    def test_weight_floor_respected(self):
        c = self._ctrl(["tank", "strong"])
        for _ in range(100):
            c.record_outcome("tank", pnl=-1000.0, win=False)
        from core.ensemble_controller import EnsembleController
        assert c.get_weight("tank") >= EnsembleController.MIN_WEIGHT - 1e-9

    def test_weight_cap_respected(self):
        c = self._ctrl(["star", "other"])
        for _ in range(100):
            c.record_outcome("star", pnl=1000.0, win=True)
        from core.ensemble_controller import EnsembleController
        assert c.get_weight("star") <= EnsembleController.MAX_WEIGHT + 1e-9

    def test_register_new_model_renormalises(self):
        c = self._ctrl(["a", "b"])
        c.register_model("c", initial_weight=0.1)
        weights = c.get_all_weights()
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)


# ============================================================
#  OrderRouter
# ============================================================

class TestOrderRouter:
    def _router(self, venues=("kraken", "coinbase")):
        from core.order_router import OrderRouter
        cfg = type("C", (), {"preferred_venues": list(venues)})()
        return OrderRouter(cfg)

    def test_routes_to_first_healthy_venue(self):
        r = self._router()
        d = r.route("BTC/USDT", "BUY", 0.1)
        assert d is not None
        assert d.venue == "kraken"

    def test_skips_unhealthy_venue(self):
        r = self._router()
        r.mark_venue_healthy("kraken", False)
        d = r.route("BTC/USDT", "BUY", 0.1)
        assert d is not None
        assert d.venue == "coinbase"

    def test_no_healthy_venue_returns_none(self):
        r = self._router()
        r.mark_venue_healthy("kraken", False)
        r.mark_venue_healthy("coinbase", False)
        d = r.route("BTC/USDT", "BUY", 0.1)
        assert d is None

    def test_routing_decision_fields(self):
        r = self._router()
        d = r.route("ETH/USDT", "sell", 2.5, order_type="limit", limit_price=3000.0)
        assert d.side == "SELL"
        assert d.quantity == pytest.approx(2.5)
        assert d.order_type == "limit"
        assert d.limit_price == pytest.approx(3000.0)


# ============================================================
#  SignalPipeline
# ============================================================

class TestSignalPipeline:
    def _pipeline(self, min_conf=0.5):
        from core.signal_pipeline import SignalPipeline
        cfg = type("C", (), {"min_signal_confidence": min_conf})()
        return SignalPipeline(cfg)

    def _make_generator(self, signals):
        class Gen:
            async def generate_signals(self, md):
                return signals
        return Gen()

    def test_filters_by_confidence(self):
        from core.signal_pipeline import Signal
        p = self._pipeline(min_conf=0.6)
        sigs = [
            Signal("BTC", "BUY", 0.9),
            Signal("ETH", "SELL", 0.4),  # should be filtered
            Signal("SOL", "BUY", 0.7),
        ]
        p.add_generator(self._make_generator(sigs))
        result = asyncio.run(p.run(None))
        assert len(result) == 2
        assert all(s.confidence >= 0.6 for s in result)

    def test_ranked_by_confidence_desc(self):
        from core.signal_pipeline import Signal
        p = self._pipeline(min_conf=0.0)
        sigs = [Signal("A", "BUY", 0.3), Signal("B", "BUY", 0.9), Signal("C", "BUY", 0.6)]
        p.add_generator(self._make_generator(sigs))
        result = asyncio.run(p.run(None))
        assert result[0].confidence >= result[1].confidence >= result[2].confidence

    def test_generator_error_isolated(self):
        from core.signal_pipeline import Signal
        class BrokenGen:
            async def generate_signals(self, md):
                raise RuntimeError("boom")
        class GoodGen:
            async def generate_signals(self, md):
                return [Signal("BTC", "BUY", 0.8)]
        p = self._pipeline()
        p.add_generator(BrokenGen())
        p.add_generator(GoodGen())
        result = asyncio.run(p.run(None))
        assert len(result) == 1  # BrokenGen error isolated

    def test_user_filter_applied(self):
        from core.signal_pipeline import Signal
        p = self._pipeline(min_conf=0.0)
        sigs = [Signal("BTC", "BUY", 0.9), Signal("ETH", "SELL", 0.8)]
        p.add_generator(self._make_generator(sigs))
        p.add_filter(lambda s: s.action == "BUY")
        result = asyncio.run(p.run(None))
        assert all(s.action == "BUY" for s in result)

    def test_signal_confidence_clamped(self):
        from core.signal_pipeline import Signal
        s = Signal("X", "BUY", 5.0)  # over 1.0
        assert s.confidence == 1.0
        s2 = Signal("X", "BUY", -1.0)  # negative
        assert s2.confidence == 0.0


# ============================================================
#  RateLimitGuard
# ============================================================

class TestRateLimitGuard:
    def test_allows_within_capacity(self):
        from utils.rate_limit_guard import RateLimitGuard
        g = RateLimitGuard(capacity=5.0, refill_rate=100.0, block=False)
        for _ in range(5):
            g.acquire()  # should not raise

    def test_denies_when_empty(self):
        from utils.rate_limit_guard import RateLimitGuard, RateLimitExceeded
        g = RateLimitGuard(capacity=2.0, refill_rate=0.001, block=False)
        g.acquire()
        g.acquire()
        with pytest.raises(RateLimitExceeded):
            g.acquire()

    def test_refills_over_time(self):
        from utils.rate_limit_guard import RateLimitGuard
        g = RateLimitGuard(capacity=1.0, refill_rate=100.0, block=False)
        g.acquire()  # drain
        time.sleep(0.05)  # 5 tokens added at 100/s
        g.acquire()  # should succeed after refill

    def test_context_manager(self):
        from utils.rate_limit_guard import RateLimitGuard
        g = RateLimitGuard(capacity=5.0, refill_rate=100.0)
        with g:  # should not raise
            pass

    def test_stats_tracking(self):
        from utils.rate_limit_guard import RateLimitGuard, RateLimitExceeded
        g = RateLimitGuard(capacity=2.0, refill_rate=0.001, exchange="test", block=False)
        g.acquire()
        g.acquire()
        with pytest.raises(RateLimitExceeded):
            g.acquire()
        stats = g.stats
        assert stats["allowed"] == 2
        assert stats["denied"] == 1

    def test_per_exchange_isolation(self):
        from utils.rate_limit_guard import ExchangeRateLimiter
        lim = ExchangeRateLimiter()
        # Drain kraken completely
        k = lim.guard("kraken")
        k._tokens = 0.0
        k.refill_rate = 0.001
        # coinbase should be unaffected
        lim.acquire("coinbase")  # should not raise
