#!/usr/bin/env python3
"""
Tests for smart execution and capital efficiency modules (Batch — Smart Execution).

Covers:
- SmartOrderRouterV2 (execution/smart_order_router_v2.py)
- TimeOfDayOptimizer  (execution/time_of_day_optimizer.py)
- CrossMarginOptimizer (risk/cross_margin_optimizer.py)
- EOFYHarvester       (compliance/eofy_harvester.py)
- DeadMansSwitch      (execution/dead_mans_switch.py)
- TradeAttributionV2  (monitoring/trade_attribution_v2.py)
- DeadTradeDetector   (ops/dead_trade_detector.py)

60+ tests total.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure repo root is on sys.path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ============================================================================
# SmartOrderRouterV2
# ============================================================================

from execution.smart_order_router_v2 import (
    SmartOrderRouterV2,
    VenueBook,
    VenueOrder,
    VenueRecommendation,
)


def _make_books() -> dict:
    """Helper: create venue books for three exchanges."""
    return {
        "kraken": VenueBook(
            best_bid=60000, best_ask=60010,
            bid_depth_usd=50000, ask_depth_usd=50000,
        ),
        "bybit": VenueBook(
            best_bid=59995, best_ask=60005,
            bid_depth_usd=80000, ask_depth_usd=80000,
        ),
        "coinbase": VenueBook(
            best_bid=60005, best_ask=60020,
            bid_depth_usd=30000, ask_depth_usd=30000,
        ),
    }


class TestSmartOrderRouterV2Init(unittest.TestCase):
    """SmartOrderRouterV2 — initialisation and database."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.db = os.path.join(self._tmp, "router.db")
        self.router = SmartOrderRouterV2(db_path=self.db)

    def test_init_creates_db(self):
        self.assertTrue(os.path.exists(self.db))

    def test_venue_recommendation_dataclass(self):
        rec = VenueRecommendation("kraken", 60010, 2.0, 26.0, 28.0, 0.9)
        self.assertEqual(rec.venue, "kraken")
        self.assertEqual(rec.total_cost_bps, 28.0)

    def test_venue_book_mid_price_auto(self):
        book = VenueBook(best_bid=100, best_ask=102)
        self.assertAlmostEqual(book.mid_price, 101.0)


class TestSmartOrderRouterV2BestVenue(unittest.TestCase):
    """SmartOrderRouterV2 — get_best_venue."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.router = SmartOrderRouterV2(db_path=os.path.join(self._tmp, "r.db"))

    def test_best_venue_returns_recommendation(self):
        rec = self.router.get_best_venue("BTC/USD", "buy", 500, _make_books())
        self.assertIsInstance(rec, VenueRecommendation)
        self.assertIn(rec.venue, ("kraken", "bybit", "coinbase"))

    def test_best_venue_prefers_lower_cost(self):
        rec = self.router.get_best_venue("BTC/USD", "buy", 500, _make_books())
        # Bybit has lowest fees (6 bps taker) and tightest spread
        self.assertEqual(rec.venue, "bybit")

    def test_best_venue_sell_side(self):
        rec = self.router.get_best_venue("BTC/USD", "sell", 500, _make_books())
        self.assertIsInstance(rec, VenueRecommendation)
        self.assertGreater(rec.confidence, 0)

    def test_best_venue_no_routable(self):
        books = {
            "tiny": VenueBook(best_bid=100, best_ask=101, bid_depth_usd=1, ask_depth_usd=1),
        }
        rec = self.router.get_best_venue("BTC/USD", "buy", 500, books)
        self.assertEqual(rec.venue, "none")
        self.assertEqual(rec.confidence, 0.0)

    def test_best_venue_empty_books(self):
        rec = self.router.get_best_venue("BTC/USD", "buy", 500, {})
        self.assertEqual(rec.venue, "none")

    def test_confidence_penalized_by_staleness(self):
        books = {
            "stale": VenueBook(
                best_bid=60000, best_ask=60010,
                bid_depth_usd=50000, ask_depth_usd=50000,
                last_trade_ts=time.time() - 300,  # 5 minutes stale
            ),
        }
        rec = self.router.get_best_venue("BTC/USD", "buy", 500, books)
        self.assertLess(rec.confidence, 1.0)

    def test_large_order_higher_slippage(self):
        books = _make_books()
        small = self.router.get_best_venue("BTC/USD", "buy", 100, books)
        large = self.router.get_best_venue("BTC/USD", "buy", 40000, books)
        self.assertGreater(large.expected_slippage_bps, small.expected_slippage_bps)


class TestSmartOrderRouterV2Split(unittest.TestCase):
    """SmartOrderRouterV2 — split_across_venues."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.router = SmartOrderRouterV2(db_path=os.path.join(self._tmp, "r.db"))

    def test_split_returns_list(self):
        orders = self.router.split_across_venues("BTC/USD", "buy", 2000, _make_books())
        self.assertIsInstance(orders, list)
        self.assertTrue(all(isinstance(o, VenueOrder) for o in orders))

    def test_split_total_covers_size(self):
        orders = self.router.split_across_venues("BTC/USD", "buy", 2000, _make_books())
        total = sum(o.size_usd for o in orders)
        self.assertGreaterEqual(total, 1999)  # Allow rounding

    def test_split_prefers_cheaper_venues(self):
        orders = self.router.split_across_venues("BTC/USD", "buy", 2000, _make_books())
        if len(orders) >= 2:
            # Cheapest venue should get more allocation
            venue_sizes = {o.venue: o.size_usd for o in orders}
            self.assertGreaterEqual(
                venue_sizes.get("bybit", 0),
                venue_sizes.get("coinbase", 0),
            )

    def test_split_empty_books(self):
        orders = self.router.split_across_venues("BTC/USD", "buy", 2000, {})
        self.assertEqual(orders, [])

    def test_split_single_venue(self):
        books = {
            "kraken": VenueBook(best_bid=60000, best_ask=60010,
                                bid_depth_usd=50000, ask_depth_usd=50000),
        }
        orders = self.router.split_across_venues("BTC/USD", "buy", 500, books)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].venue, "kraken")


class TestSmartOrderRouterV2Learning(unittest.TestCase):
    """SmartOrderRouterV2 — record_execution and bias learning."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.router = SmartOrderRouterV2(db_path=os.path.join(self._tmp, "r.db"))

    def test_record_execution(self):
        self.router.record_execution("kraken", "BTC/USD", 10.0, 12.0)
        stats = self.router.get_venue_stats("kraken", "BTC/USD")
        self.assertEqual(stats["count"], 1)

    def test_venue_stats_empty(self):
        stats = self.router.get_venue_stats("kraken", "BTC/USD")
        self.assertEqual(stats["count"], 0)

    def test_bias_accumulates(self):
        for _ in range(15):
            self.router.record_execution("kraken", "BTC/USD", 10.0, 15.0)
        stats = self.router.get_venue_stats("kraken", "BTC/USD")
        self.assertAlmostEqual(stats["avg_bias_bps"], 5.0, places=1)


# ============================================================================
# TimeOfDayOptimizer
# ============================================================================

from execution.time_of_day_optimizer import TimeOfDayOptimizer, HourStats


class TestTimeOfDayOptimizerInit(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.opt = TimeOfDayOptimizer(
            db_path=os.path.join(self._tmp, "tod.db"),
            min_observations=5,  # lower for testing
        )

    def test_init_creates_db(self):
        self.assertTrue(os.path.exists(os.path.join(self._tmp, "tod.db")))

    def test_record_execution(self):
        self.opt.record_execution("BTC/USD", 14, 2.5, 8.0, 120.0)
        stats = self.opt.get_all_hour_stats("BTC/USD")
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0].hour, 14)


class TestTimeOfDayOptimizerQueries(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.opt = TimeOfDayOptimizer(
            db_path=os.path.join(self._tmp, "tod.db"),
            min_observations=5,
        )
        # Populate: hour 10 is good, hour 3 is bad
        for _ in range(10):
            self.opt.record_execution("BTC/USD", 10, 1.5, 5.0, 80.0)
            self.opt.record_execution("BTC/USD", 3, 8.0, 20.0, 500.0)
            self.opt.record_execution("BTC/USD", 14, 3.0, 10.0, 150.0)

    def test_optimal_hours(self):
        best = self.opt.get_optimal_hours("BTC/USD", top_n=2)
        self.assertGreater(len(best), 0)
        self.assertEqual(best[0][0], 10)  # hour 10 is cheapest

    def test_worst_hours(self):
        worst = self.opt.get_worst_hours("BTC/USD", top_n=2)
        self.assertGreater(len(worst), 0)
        self.assertEqual(worst[0][0], 3)  # hour 3 is worst

    def test_should_delay_bad_hour(self):
        delay, minutes = self.opt.should_delay("BTC/USD", 3)
        self.assertTrue(delay)
        self.assertGreater(minutes, 0)

    def test_should_not_delay_good_hour(self):
        delay, minutes = self.opt.should_delay("BTC/USD", 10)
        self.assertFalse(delay)
        self.assertEqual(minutes, 0)

    def test_insufficient_data_returns_empty(self):
        best = self.opt.get_optimal_hours("ETH/USD")
        self.assertEqual(best, [])

    def test_should_delay_no_data(self):
        delay, minutes = self.opt.should_delay("ETH/USD", 10)
        self.assertFalse(delay)

    def test_all_hour_stats(self):
        stats = self.opt.get_all_hour_stats("BTC/USD")
        self.assertEqual(len(stats), 3)
        hours = {s.hour for s in stats}
        self.assertEqual(hours, {3, 10, 14})


# ============================================================================
# CrossMarginOptimizer
# ============================================================================

from risk.cross_margin_optimizer import (
    CrossMarginOptimizer,
    NettingOpportunity,
    TransferRecommendation,
)


class TestCrossMarginOptimizerNetting(unittest.TestCase):

    def setUp(self):
        self.opt = CrossMarginOptimizer(margin_rate=0.10)

    def test_add_position(self):
        self.opt.add_position("kraken", "BTC/USD", "long", 0.05, 500)
        self.assertEqual(len(self.opt._positions), 1)

    def test_netting_opposite_venues(self):
        self.opt.add_position("kraken", "BTC/USD", "long", 0.05, 500)
        self.opt.add_position("bybit", "BTC/USD", "short", 0.03, 300)
        opps = self.opt.get_netting_opportunities()
        self.assertEqual(len(opps), 1)
        self.assertAlmostEqual(opps[0].size_to_net, 0.03, places=4)
        self.assertGreater(opps[0].margin_saved_usd, 0)

    def test_no_netting_same_side(self):
        self.opt.add_position("kraken", "BTC/USD", "long", 0.05, 500)
        self.opt.add_position("bybit", "BTC/USD", "long", 0.03, 300)
        opps = self.opt.get_netting_opportunities()
        self.assertEqual(len(opps), 0)

    def test_no_netting_same_venue(self):
        self.opt.add_position("kraken", "BTC/USD", "long", 0.05, 500)
        self.opt.add_position("kraken", "BTC/USD", "short", 0.03, 300)
        opps = self.opt.get_netting_opportunities()
        self.assertEqual(len(opps), 0)

    def test_netting_multiple_symbols(self):
        self.opt.add_position("kraken", "BTC/USD", "long", 0.05, 500)
        self.opt.add_position("bybit", "BTC/USD", "short", 0.05, 500)
        self.opt.add_position("kraken", "ETH/USD", "long", 1.0, 200)
        self.opt.add_position("bybit", "ETH/USD", "short", 0.5, 100)
        opps = self.opt.get_netting_opportunities()
        self.assertEqual(len(opps), 2)

    def test_clear_positions(self):
        self.opt.add_position("kraken", "BTC/USD", "long", 0.05, 500)
        self.opt.clear_positions()
        self.assertEqual(len(self.opt._positions), 0)


class TestCrossMarginOptimizerEfficiency(unittest.TestCase):

    def setUp(self):
        self.opt = CrossMarginOptimizer(margin_rate=0.10)

    def test_efficiency_empty(self):
        eff = self.opt.get_total_margin_efficiency()
        self.assertEqual(eff, 1.0)

    def test_efficiency_perfect(self):
        self.opt.add_position("kraken", "BTC/USD", "long", 0.05, 500)
        eff = self.opt.get_total_margin_efficiency()
        self.assertEqual(eff, 1.0)

    def test_efficiency_with_netting(self):
        self.opt.add_position("kraken", "BTC/USD", "long", 0.05, 500)
        self.opt.add_position("bybit", "BTC/USD", "short", 0.05, 500)
        eff = self.opt.get_total_margin_efficiency()
        # Should be < 1.0 since positions fully offset
        self.assertLess(eff, 0.1)

    def test_suggest_transfers_needs_multiple_venues(self):
        self.opt.add_position("kraken", "BTC/USD", "long", 0.05, 500)
        transfers = self.opt.suggest_transfers()
        self.assertEqual(transfers, [])

    def test_suggest_transfers_imbalanced(self):
        self.opt.add_position("kraken", "BTC/USD", "long", 0.05, 5000)
        self.opt.add_position("bybit", "ETH/USD", "long", 1.0, 100)
        transfers = self.opt.suggest_transfers()
        # Might suggest rebalancing
        self.assertIsInstance(transfers, list)


# ============================================================================
# EOFYHarvester
# ============================================================================

from compliance.eofy_harvester import EOFYHarvester, HarvestCandidate, EOFYPlan


class TestEOFYHarvesterScan(unittest.TestCase):

    def setUp(self):
        self.harvester = EOFYHarvester(marginal_rate=0.325)

    def test_scan_finds_losses(self):
        positions = {
            "BTC/AUD": {"quantity": 0.1, "entry_price_aud": 100000},
            "ETH/AUD": {"quantity": 2.0, "entry_price_aud": 5000},
        }
        prices = {"BTC/AUD": 90000, "ETH/AUD": 5500}
        candidates = self.harvester.scan_unrealized_losses(positions, prices)
        self.assertEqual(len(candidates), 1)  # Only BTC has a loss
        self.assertEqual(candidates[0].symbol, "BTC/AUD")
        self.assertLess(candidates[0].unrealized_loss_aud, 0)

    def test_tax_saving_calculation(self):
        positions = {"BTC/AUD": {"quantity": 1.0, "entry_price_aud": 100000}}
        prices = {"BTC/AUD": 90000}
        candidates = self.harvester.scan_unrealized_losses(positions, prices)
        # Loss = $10,000 → saving = $3,250 at 32.5%
        self.assertAlmostEqual(candidates[0].tax_saving_aud, 3250.0, places=0)

    def test_wash_sale_flagging(self):
        self.harvester.add_recent_sale("BTC/AUD")
        positions = {"BTC/AUD": {"quantity": 0.1, "entry_price_aud": 100000}}
        prices = {"BTC/AUD": 90000}
        candidates = self.harvester.scan_unrealized_losses(positions, prices)
        self.assertTrue(candidates[0].wash_sale_risk)

    def test_no_wash_sale_clean(self):
        positions = {"BTC/AUD": {"quantity": 0.1, "entry_price_aud": 100000}}
        prices = {"BTC/AUD": 90000}
        candidates = self.harvester.scan_unrealized_losses(positions, prices)
        self.assertFalse(candidates[0].wash_sale_risk)

    def test_skips_profitable_positions(self):
        positions = {"BTC/AUD": {"quantity": 0.1, "entry_price_aud": 90000}}
        prices = {"BTC/AUD": 100000}
        candidates = self.harvester.scan_unrealized_losses(positions, prices)
        self.assertEqual(len(candidates), 0)

    def test_empty_positions(self):
        candidates = self.harvester.scan_unrealized_losses({}, {})
        self.assertEqual(len(candidates), 0)


class TestEOFYHarvesterPlan(unittest.TestCase):

    def setUp(self):
        self.harvester = EOFYHarvester()

    def test_plan_with_candidates(self):
        positions = {
            "BTC/AUD": {"quantity": 0.1, "entry_price_aud": 100000},
            "SOL/AUD": {"quantity": 10, "entry_price_aud": 250},
        }
        prices = {"BTC/AUD": 90000, "SOL/AUD": 200}
        self.harvester.scan_unrealized_losses(positions, prices)
        plan = self.harvester.get_eofy_strategy()
        self.assertIsInstance(plan, EOFYPlan)
        self.assertGreater(plan.total_potential_saving, 0)
        self.assertGreater(len(plan.recommended_actions), 0)

    def test_plan_empty(self):
        plan = self.harvester.get_eofy_strategy()
        self.assertEqual(plan.total_potential_saving, 0)
        self.assertIn("No unrealised losses", plan.recommended_actions[0])

    def test_plan_deadline_date(self):
        plan = self.harvester.get_eofy_strategy("2026-06-30")
        self.assertEqual(plan.deadline_date, date(2026, 6, 30))


# ============================================================================
# DeadMansSwitch
# ============================================================================

from execution.dead_mans_switch import DeadMansSwitch


class TestDeadMansSwitchBasic(unittest.TestCase):

    def test_init_default(self):
        switch = DeadMansSwitch()
        self.assertEqual(switch.timeout_minutes, 30)
        self.assertFalse(switch.is_triggered)

    def test_heartbeat_keeps_alive(self):
        switch = DeadMansSwitch(timeout_minutes=1)
        switch.heartbeat()
        self.assertTrue(switch.check())

    def test_check_returns_true_when_fresh(self):
        switch = DeadMansSwitch(timeout_minutes=60)
        self.assertTrue(switch.check())

    def test_trigger_after_timeout(self):
        switch = DeadMansSwitch(timeout_minutes=1)
        # Simulate timeout by backdating heartbeat
        switch._last_heartbeat = time.time() - 120
        alive = switch.check()
        self.assertFalse(alive)
        self.assertTrue(switch.is_triggered)

    def test_trigger_count(self):
        switch = DeadMansSwitch(timeout_minutes=1)
        switch._last_heartbeat = time.time() - 120
        switch.check()
        self.assertEqual(switch.trigger_count, 1)

    def test_recovery_after_heartbeat(self):
        switch = DeadMansSwitch(timeout_minutes=1)
        switch._last_heartbeat = time.time() - 120
        switch.check()
        self.assertTrue(switch.is_triggered)
        switch.heartbeat()
        self.assertFalse(switch.is_triggered)

    def test_get_last_heartbeat(self):
        switch = DeadMansSwitch()
        switch.heartbeat()
        hb = switch.get_last_heartbeat()
        self.assertIsInstance(hb, datetime)

    def test_get_silence_duration(self):
        switch = DeadMansSwitch()
        switch.heartbeat()
        dur = switch.get_silence_duration()
        self.assertIsInstance(dur, timedelta)
        self.assertLess(dur.total_seconds(), 2)


class TestDeadMansSwitchCallbacks(unittest.TestCase):

    def test_close_callback_on_trigger(self):
        closed = []
        switch = DeadMansSwitch(
            timeout_minutes=1,
            close_positions_on_timeout=True,
            close_callback=lambda: closed.append(True),
        )
        switch._last_heartbeat = time.time() - 120
        switch.check()
        self.assertEqual(len(closed), 1)

    def test_alert_callback_on_trigger(self):
        alerts = []
        switch = DeadMansSwitch(
            timeout_minutes=1,
            alert_callback=lambda msg: alerts.append(msg),
        )
        switch._last_heartbeat = time.time() - 120
        switch.check()
        self.assertEqual(len(alerts), 1)
        self.assertIn("DEAD MAN'S SWITCH", alerts[0])

    def test_no_close_when_disabled(self):
        closed = []
        switch = DeadMansSwitch(
            timeout_minutes=1,
            close_positions_on_timeout=False,
            close_callback=lambda: closed.append(True),
        )
        switch._last_heartbeat = time.time() - 120
        switch.check()
        self.assertEqual(len(closed), 0)


# ============================================================================
# TradeAttributionV2
# ============================================================================

from monitoring.trade_attribution_v2 import TradeAttributionV2, PnLDecomposition


class TestTradeAttributionV2Record(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.attr = TradeAttributionV2(db_path=os.path.join(self._tmp, "attr.db"))

    def test_record_trade(self):
        self.attr.record_trade({
            "symbol": "BTC/USD", "side": "buy", "quantity": 0.01,
            "entry_price": 60000, "exit_price": 61000,
            "strategy": "momentum",
        })
        decomp = self.attr.decompose_pnl(lookback_days=1)
        self.assertEqual(decomp.trade_count, 1)

    def test_gross_pnl_buy(self):
        self.attr.record_trade({
            "symbol": "BTC/USD", "side": "buy", "quantity": 0.1,
            "entry_price": 60000, "exit_price": 61000,
        })
        decomp = self.attr.decompose_pnl(1)
        self.assertAlmostEqual(decomp.gross_pnl, 100.0, places=0)

    def test_gross_pnl_sell(self):
        self.attr.record_trade({
            "symbol": "BTC/USD", "side": "sell", "quantity": 0.1,
            "entry_price": 61000, "exit_price": 60000,
        })
        decomp = self.attr.decompose_pnl(1)
        self.assertAlmostEqual(decomp.gross_pnl, 100.0, places=0)


class TestTradeAttributionV2Decompose(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.attr = TradeAttributionV2(db_path=os.path.join(self._tmp, "attr.db"))
        # Record several trades
        for i in range(5):
            self.attr.record_trade({
                "symbol": "BTC/USD", "side": "buy", "quantity": 0.01,
                "entry_price": 60000, "exit_price": 60000 + (i + 1) * 100,
                "strategy": "momentum",
                "market_return_pct": 0.5,
                "slippage_bps": 3.0,
                "fees_usd": 1.5,
                "funding_usd": -0.1,
                "fx_impact_usd": 0.0,
            })

    def test_decomposition_fields(self):
        decomp = self.attr.decompose_pnl(1)
        self.assertIsInstance(decomp, PnLDecomposition)
        self.assertEqual(decomp.trade_count, 5)
        self.assertGreater(decomp.gross_pnl, 0)

    def test_net_pnl_less_than_gross(self):
        decomp = self.attr.decompose_pnl(1)
        self.assertLess(decomp.net_pnl, decomp.gross_pnl)

    def test_slippage_cost_positive(self):
        decomp = self.attr.decompose_pnl(1)
        self.assertGreater(decomp.slippage_cost, 0)

    def test_alpha_by_strategy(self):
        alpha = self.attr.get_alpha_by_strategy(1)
        self.assertIn("momentum", alpha)

    def test_cost_breakdown(self):
        costs = self.attr.get_cost_breakdown(1)
        self.assertIn("slippage", costs)
        self.assertIn("fees", costs)
        self.assertIn("total", costs)
        self.assertGreater(costs["total"], 0)

    def test_improvement_suggestions(self):
        suggestions = self.attr.get_improvement_suggestions(1)
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)

    def test_empty_period(self):
        attr2 = TradeAttributionV2(db_path=os.path.join(self._tmp, "empty.db"))
        decomp = attr2.decompose_pnl(1)
        self.assertEqual(decomp.trade_count, 0)
        self.assertEqual(decomp.gross_pnl, 0.0)


# ============================================================================
# DeadTradeDetector
# ============================================================================

from ops.dead_trade_detector import DeadTradeDetector, DeadTradeAlert


class TestDeadTradeDetectorBasic(unittest.TestCase):

    def test_init(self):
        det = DeadTradeDetector()
        self.assertIsNotNone(det)

    def test_record_trade(self):
        det = DeadTradeDetector()
        det.record_trade(time.time())
        self.assertEqual(len(det._trade_timestamps), 1)

    def test_no_alert_when_fresh(self):
        det = DeadTradeDetector(alert_after_hours=4)
        det.record_trade(time.time())
        alert = det.check()
        self.assertIsNone(alert)

    def test_no_alert_no_trades(self):
        det = DeadTradeDetector()
        alert = det.check()
        self.assertIsNone(alert)  # No trades ever = can't alert


class TestDeadTradeDetectorAlerts(unittest.TestCase):

    def test_alert_after_silence(self):
        det = DeadTradeDetector(alert_after_hours=1)
        det.record_trade(time.time() - 7200)  # 2 hours ago
        alert = det.check()
        self.assertIsInstance(alert, DeadTradeAlert)
        self.assertGreater(alert.hours_since_last_trade, 1.0)
        self.assertGreater(len(alert.possible_reasons), 0)

    def test_alert_severity(self):
        det = DeadTradeDetector(alert_after_hours=1)
        det.record_trade(time.time() - 3600 * 3)  # 3 hours ago
        alert = det.check()
        self.assertEqual(alert.severity, "critical")

    def test_alert_with_state_callback(self):
        def state_cb():
            return {
                "exchange_status": "down",
                "circuit_breaker_active": True,
                "signals_suppressed": False,
                "process_healthy": True,
            }
        det = DeadTradeDetector(alert_after_hours=1, system_state_callback=state_cb)
        det.record_trade(time.time() - 7200)
        alert = det.check()
        self.assertIn("exchange down", alert.possible_reasons)
        self.assertIn("circuit breaker active", alert.possible_reasons)

    def test_alert_cooldown(self):
        det = DeadTradeDetector(alert_after_hours=1)
        det.record_trade(time.time() - 7200)
        alert1 = det.check()
        alert2 = det.check()
        self.assertIsNotNone(alert1)
        self.assertIsNone(alert2)  # Cooldown prevents second alert


class TestDeadTradeDetectorStats(unittest.TestCase):

    def test_trade_frequency(self):
        det = DeadTradeDetector()
        now = time.time()
        for i in range(10):
            det.record_trade(now - i * 600)  # every 10 min over ~1.5 hours
        freq = det.get_trade_frequency(hours=2)
        self.assertGreater(freq, 0)

    def test_last_trade_time(self):
        det = DeadTradeDetector()
        det.record_trade(time.time())
        lt = det.get_last_trade_time()
        self.assertIsInstance(lt, datetime)

    def test_hours_since_last_trade(self):
        det = DeadTradeDetector()
        det.record_trade(time.time() - 3600)
        hours = det.get_hours_since_last_trade()
        self.assertAlmostEqual(hours, 1.0, places=0)

    def test_no_trades_returns_none(self):
        det = DeadTradeDetector()
        self.assertIsNone(det.get_last_trade_time())
        self.assertIsNone(det.get_hours_since_last_trade())


if __name__ == "__main__":
    unittest.main()
