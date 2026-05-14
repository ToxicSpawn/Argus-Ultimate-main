from __future__ import annotations

import math
import os
import sqlite3
import tempfile
import unittest

from evaluation.strategy_evaluation_engine import StrategyEvaluationEngine


class TestStrategyEvaluationEngine(unittest.TestCase):
    def _make_db_path(self) -> str:
        fd, path = tempfile.mkstemp(prefix="argus_strategy_metrics_", suffix=".db")
        os.close(fd)
        return path

    def test_metric_updates_fee_aware_expectancy_profit_factor(self) -> None:
        db_path = self._make_db_path()
        engine = StrategyEvaluationEngine(
            db_path=db_path,
            min_trades_for_ranking=1,
            sharpe_like_min_trades=1,
            use_regime_scoped_metrics=True,
            max_metrics_history_points=200,
        )

        engine.record_open(strategy_name="momentum", symbol="BTC/USD", quantity=1.0, ts=100.0, regime_label="trend")
        hold_1 = engine.consume_hold_time_seconds(strategy_name="momentum", symbol="BTC/USD", quantity=1.0, ts=130.0)
        engine.record_trade_close(
            strategy_name="momentum",
            symbol="BTC/USD",
            gross_pnl_aud=25.0,
            net_pnl_aud=20.0,
            fees_aud=5.0,
            expected_net_edge_bps=12.0,
            realized_slippage_bps=3.0,
            hold_time_seconds=hold_1,
            regime_label="trend",
            ts=130.0,
        )

        engine.record_open(strategy_name="momentum", symbol="BTC/USD", quantity=1.0, ts=200.0, regime_label="trend")
        hold_2 = engine.consume_hold_time_seconds(strategy_name="momentum", symbol="BTC/USD", quantity=1.0, ts=260.0)
        engine.record_trade_close(
            strategy_name="momentum",
            symbol="BTC/USD",
            gross_pnl_aud=-8.0,
            net_pnl_aud=-10.0,
            fees_aud=2.0,
            expected_net_edge_bps=4.0,
            realized_slippage_bps=5.0,
            hold_time_seconds=hold_2,
            regime_label="trend",
            ts=260.0,
        )

        self.assertAlmostEqual(hold_1, 30.0, places=6)
        self.assertAlmostEqual(hold_2, 60.0, places=6)

        m = engine.get_metrics(strategy_name="momentum", symbol="BTC/USD")
        self.assertIsNotNone(m)
        assert m is not None
        self.assertEqual(m.trades_count, 2)
        self.assertEqual(m.wins_count, 1)
        self.assertEqual(m.losses_count, 1)
        self.assertAlmostEqual(m.win_rate, 0.5, places=6)
        self.assertAlmostEqual(m.net_pnl_aud, 10.0, places=6)
        self.assertAlmostEqual(m.total_fees_aud, 7.0, places=6)
        self.assertAlmostEqual(m.avg_net_pnl_per_trade, 5.0, places=6)
        self.assertAlmostEqual(m.expectancy, 5.0, places=6)
        self.assertAlmostEqual(m.profit_factor, 25.0 / 8.0, places=6)
        self.assertAlmostEqual(m.avg_hold_time_seconds, 45.0, places=6)
        self.assertAlmostEqual(m.avg_expected_net_edge_bps, 8.0, places=6)
        self.assertAlmostEqual(m.avg_realized_slippage_bps, 4.0, places=6)
        self.assertTrue(m.enabled_for_ranking)
        self.assertGreaterEqual(m.max_drawdown_pct, 49.9)

        m_reg = engine.get_metrics(strategy_name="momentum", symbol="BTC/USD", regime_label="trend")
        self.assertIsNotNone(m_reg)
        assert m_reg is not None
        self.assertEqual(m_reg.regime_label, "trend")
        self.assertEqual(m_reg.trades_count, 2)

    def test_per_symbol_and_global_metrics_tracking(self) -> None:
        db_path = self._make_db_path()
        engine = StrategyEvaluationEngine(db_path=db_path, min_trades_for_ranking=1, sharpe_like_min_trades=1)

        engine.record_trade_close(
            strategy_name="mean_reversion",
            symbol="BTC/USD",
            gross_pnl_aud=6.0,
            net_pnl_aud=5.0,
            fees_aud=1.0,
            expected_net_edge_bps=7.0,
            realized_slippage_bps=2.0,
            hold_time_seconds=20.0,
            regime_label="range",
            ts=1.0,
        )
        engine.record_trade_close(
            strategy_name="mean_reversion",
            symbol="ETH/USD",
            gross_pnl_aud=-2.0,
            net_pnl_aud=-3.0,
            fees_aud=1.0,
            expected_net_edge_bps=3.0,
            realized_slippage_bps=4.0,
            hold_time_seconds=15.0,
            regime_label="range",
            ts=2.0,
        )

        g = engine.get_metrics(strategy_name="mean_reversion", symbol=None)
        b = engine.get_metrics(strategy_name="mean_reversion", symbol="BTC/USD")
        e = engine.get_metrics(strategy_name="mean_reversion", symbol="ETH/USD")
        self.assertIsNotNone(g)
        self.assertIsNotNone(b)
        self.assertIsNotNone(e)
        assert g is not None and b is not None and e is not None
        self.assertEqual(g.trades_count, 2)
        self.assertEqual(b.trades_count, 1)
        self.assertEqual(e.trades_count, 1)
        self.assertAlmostEqual(g.net_pnl_aud, 2.0, places=6)

        rg = engine.get_metrics(strategy_name="mean_reversion", symbol=None, regime_label="range")
        self.assertIsNotNone(rg)
        assert rg is not None
        self.assertEqual(rg.regime_label, "range")
        self.assertEqual(rg.trades_count, 2)

    def test_persistence_and_reload(self) -> None:
        db_path = self._make_db_path()
        engine = StrategyEvaluationEngine(db_path=db_path, min_trades_for_ranking=1, sharpe_like_min_trades=1)
        engine.record_trade_close(
            strategy_name="breakout",
            symbol="BTC/USD",
            gross_pnl_aud=3.0,
            net_pnl_aud=2.0,
            fees_aud=1.0,
            expected_net_edge_bps=5.0,
            realized_slippage_bps=2.0,
            hold_time_seconds=10.0,
            regime_label="trend",
            ts=10.0,
        )
        engine.persist_to_db()

        rows = 0
        with sqlite3.connect(db_path) as con:
            rows = int(con.execute("SELECT COUNT(*) FROM strategy_metrics").fetchone()[0])
        self.assertGreaterEqual(rows, 2)  # global + symbol scopes

        reloaded = StrategyEvaluationEngine(db_path=db_path, min_trades_for_ranking=1, sharpe_like_min_trades=1)
        m = reloaded.get_metrics(strategy_name="breakout", symbol="BTC/USD")
        self.assertIsNotNone(m)
        assert m is not None
        self.assertEqual(m.trades_count, 1)
        self.assertAlmostEqual(m.net_pnl_aud, 2.0, places=6)

    def test_rank_helpers_require_min_trades(self) -> None:
        db_path = self._make_db_path()
        engine = StrategyEvaluationEngine(
            db_path=db_path,
            min_trades_for_ranking=2,
            sharpe_like_min_trades=2,
        )

        engine.record_trade_close(
            strategy_name="alpha_a",
            symbol="BTC/USD",
            gross_pnl_aud=12.0,
            net_pnl_aud=10.0,
            fees_aud=2.0,
            expected_net_edge_bps=10.0,
            realized_slippage_bps=2.0,
            hold_time_seconds=8.0,
            ts=1.0,
        )
        engine.record_trade_close(
            strategy_name="alpha_b",
            symbol="ETH/USD",
            gross_pnl_aud=2.0,
            net_pnl_aud=1.0,
            fees_aud=1.0,
            expected_net_edge_bps=4.0,
            realized_slippage_bps=3.0,
            hold_time_seconds=8.0,
            ts=1.0,
        )
        engine.record_trade_close(
            strategy_name="alpha_b",
            symbol="ETH/USD",
            gross_pnl_aud=2.0,
            net_pnl_aud=1.0,
            fees_aud=1.0,
            expected_net_edge_bps=4.0,
            realized_slippage_bps=3.0,
            hold_time_seconds=8.0,
            ts=2.0,
        )

        top_initial = engine.top_by_net_pnl(limit=5)
        self.assertEqual([m.strategy_name for m in top_initial], ["alpha_b"])

        engine.record_trade_close(
            strategy_name="alpha_a",
            symbol="BTC/USD",
            gross_pnl_aud=-1.0,
            net_pnl_aud=-2.0,
            fees_aud=1.0,
            expected_net_edge_bps=1.0,
            realized_slippage_bps=4.0,
            hold_time_seconds=8.0,
            ts=3.0,
        )
        top_after = engine.top_by_net_pnl(limit=5)
        self.assertEqual(top_after[0].strategy_name, "alpha_a")
        self.assertTrue(math.isfinite(top_after[0].expectancy))


if __name__ == "__main__":
    unittest.main()
