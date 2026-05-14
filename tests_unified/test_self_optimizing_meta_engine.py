from __future__ import annotations

import os
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from adaptive.self_optimizing_meta_engine import SelfOptimizingMetaEngine
from execution.target_portfolio_engine import TargetPortfolioEngine
from evaluation.strategy_evaluation_engine import StrategyMetrics
from unified_trading_system import OmegaSQLiteStore, SystemState, UnifiedConfig, UnifiedSystemArchitecture


class _Signal(SimpleNamespace):
    pass


class _FakeStrategyEval:
    def __init__(self, metrics_by_strategy):
        self.metrics_by_strategy = dict(metrics_by_strategy)

    def get_metrics(self, *, strategy_name: str, symbol=None, regime_label=None):
        _ = symbol
        key = (str(strategy_name), str(regime_label or ""))
        if key in self.metrics_by_strategy:
            return self.metrics_by_strategy[key]
        return self.metrics_by_strategy.get((str(strategy_name), ""))


def _metric(
    *,
    strategy_name: str,
    trades_count: int,
    net_pnl_aud: float,
    expectancy: float,
    profit_factor: float,
    sharpe_like_score: float,
    max_drawdown_pct: float,
    total_fees_aud: float,
    avg_realized_slippage_bps: float,
    regime_label: str | None = None,
) -> StrategyMetrics:
    wins = max(0, int(round(trades_count * 0.6)))
    losses = max(0, trades_count - wins)
    gross = net_pnl_aud + total_fees_aud
    return StrategyMetrics(
        strategy_name=strategy_name,
        symbol="__ALL__",
        trades_count=trades_count,
        wins_count=wins,
        losses_count=losses,
        win_rate=float(wins / max(1, trades_count)),
        gross_pnl_aud=float(gross),
        net_pnl_aud=float(net_pnl_aud),
        total_fees_aud=float(total_fees_aud),
        avg_net_pnl_per_trade=float(net_pnl_aud / max(1, trades_count)),
        avg_expected_net_edge_bps=8.0,
        avg_realized_slippage_bps=float(avg_realized_slippage_bps),
        avg_hold_time_seconds=20.0,
        max_drawdown_pct=float(max_drawdown_pct),
        profit_factor=float(profit_factor),
        expectancy=float(expectancy),
        sharpe_like_score=float(sharpe_like_score),
        last_updated_ts=1.0,
        regime_label=regime_label,
        enabled_for_ranking=True,
    )


class TestSelfOptimizingMetaEngine(unittest.TestCase):
    def _make_engine(self, **kwargs) -> SelfOptimizingMetaEngine:
        base = tempfile.mkdtemp(prefix="argus_meta_")
        db = os.path.join(base, "meta.db")
        return SelfOptimizingMetaEngine(db_path=db, **kwargs)

    def test_score_and_bounds_normalization(self) -> None:
        engine = self._make_engine(
            min_trades_for_reweighting=5,
            min_weight_per_strategy=0.05,
            max_weight_per_strategy=0.45,
            meta_alpha=1.0,
            max_weight_change_per_update=1.0,
        )
        fake = _FakeStrategyEval(
            {
                ("momentum", ""): _metric(
                    strategy_name="momentum",
                    trades_count=20,
                    net_pnl_aud=120.0,
                    expectancy=5.0,
                    profit_factor=2.0,
                    sharpe_like_score=1.5,
                    max_drawdown_pct=6.0,
                    total_fees_aud=8.0,
                    avg_realized_slippage_bps=2.0,
                ),
                ("mean_reversion", ""): _metric(
                    strategy_name="mean_reversion",
                    trades_count=20,
                    net_pnl_aud=40.0,
                    expectancy=2.0,
                    profit_factor=1.2,
                    sharpe_like_score=0.5,
                    max_drawdown_pct=12.0,
                    total_fees_aud=10.0,
                    avg_realized_slippage_bps=6.0,
                ),
            }
        )
        weights, _reasons, _src = engine.compute_weights(
            strategy_names=["momentum", "mean_reversion"],
            strategy_evaluation_engine=fake,
            regime_label="trend:mid_vol",
        )
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertGreater(weights["momentum"], weights["mean_reversion"])
        self.assertLessEqual(weights["momentum"], 1.0 + 1e-9)

    def test_fallback_baseline_for_insufficient_trades(self) -> None:
        engine = self._make_engine(min_trades_for_reweighting=10, meta_alpha=1.0, max_weight_change_per_update=1.0)
        fake = _FakeStrategyEval(
            {
                ("momentum", ""): _metric(
                    strategy_name="momentum",
                    trades_count=3,
                    net_pnl_aud=10.0,
                    expectancy=3.0,
                    profit_factor=1.1,
                    sharpe_like_score=0.5,
                    max_drawdown_pct=5.0,
                    total_fees_aud=1.0,
                    avg_realized_slippage_bps=2.0,
                ),
                ("breakout", ""): _metric(
                    strategy_name="breakout",
                    trades_count=3,
                    net_pnl_aud=9.0,
                    expectancy=2.9,
                    profit_factor=1.1,
                    sharpe_like_score=0.4,
                    max_drawdown_pct=5.0,
                    total_fees_aud=1.0,
                    avg_realized_slippage_bps=2.1,
                ),
            }
        )
        weights, reasons, _src = engine.compute_weights(
            strategy_names=["momentum", "breakout"],
            strategy_evaluation_engine=fake,
            regime_label="range:low_vol",
        )
        self.assertAlmostEqual(weights["momentum"], 0.5, places=6)
        self.assertAlmostEqual(weights["breakout"], 0.5, places=6)
        self.assertTrue(any("insufficient_trades" in r for r in reasons["momentum"]))

    def test_regime_multiplier_boosts_strategy(self) -> None:
        engine = self._make_engine(
            meta_alpha=1.0,
            max_weight_change_per_update=1.0,
            regime_multipliers={"trend": {"momentum": 1.5}},
        )
        fake = _FakeStrategyEval(
            {
                ("momentum", "trend:mid_vol"): _metric(
                    strategy_name="momentum",
                    trades_count=20,
                    net_pnl_aud=80.0,
                    expectancy=3.0,
                    profit_factor=1.5,
                    sharpe_like_score=1.0,
                    max_drawdown_pct=8.0,
                    total_fees_aud=4.0,
                    avg_realized_slippage_bps=3.0,
                    regime_label="trend:mid_vol",
                ),
                ("mean_reversion", "trend:mid_vol"): _metric(
                    strategy_name="mean_reversion",
                    trades_count=20,
                    net_pnl_aud=80.0,
                    expectancy=3.0,
                    profit_factor=1.5,
                    sharpe_like_score=1.0,
                    max_drawdown_pct=8.0,
                    total_fees_aud=4.0,
                    avg_realized_slippage_bps=3.0,
                    regime_label="trend:mid_vol",
                ),
            }
        )
        weights, _reasons, _src = engine.compute_weights(
            strategy_names=["momentum", "mean_reversion"],
            strategy_evaluation_engine=fake,
            regime_label="trend:mid_vol",
        )
        self.assertGreater(weights["momentum"], weights["mean_reversion"])

    def test_ema_smoothing_and_max_weight_change(self) -> None:
        engine = self._make_engine(meta_alpha=0.2, max_weight_change_per_update=0.10)
        fake = _FakeStrategyEval(
            {
                ("momentum", ""): _metric(
                    strategy_name="momentum",
                    trades_count=20,
                    net_pnl_aud=120.0,
                    expectancy=5.0,
                    profit_factor=2.0,
                    sharpe_like_score=1.5,
                    max_drawdown_pct=6.0,
                    total_fees_aud=8.0,
                    avg_realized_slippage_bps=2.0,
                ),
                ("mean_reversion", ""): _metric(
                    strategy_name="mean_reversion",
                    trades_count=20,
                    net_pnl_aud=20.0,
                    expectancy=1.0,
                    profit_factor=1.0,
                    sharpe_like_score=0.2,
                    max_drawdown_pct=20.0,
                    total_fees_aud=12.0,
                    avg_realized_slippage_bps=8.0,
                ),
            }
        )
        weights = engine.maybe_update(
            cycle_id=1,
            strategy_names=["momentum", "mean_reversion"],
            strategy_evaluation_engine=fake,
            regime_label="trend:mid_vol",
            run_id="run1",
            trace_id="trace1",
        )
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        # With alpha=0.2 and max_delta=0.10, momentum cannot jump too aggressively in one update.
        self.assertLessEqual(abs(weights["momentum"] - 0.5), 0.12)

    def test_persistence_reload_and_apply_to_candidates(self) -> None:
        base = tempfile.mkdtemp(prefix="argus_meta_reload_")
        db = os.path.join(base, "meta.db")
        engine = SelfOptimizingMetaEngine(db_path=db, meta_alpha=1.0, max_weight_change_per_update=1.0)
        fake = _FakeStrategyEval(
            {
                ("momentum", ""): _metric(
                    strategy_name="momentum",
                    trades_count=20,
                    net_pnl_aud=120.0,
                    expectancy=5.0,
                    profit_factor=2.0,
                    sharpe_like_score=1.5,
                    max_drawdown_pct=6.0,
                    total_fees_aud=8.0,
                    avg_realized_slippage_bps=2.0,
                ),
                ("breakout", ""): _metric(
                    strategy_name="breakout",
                    trades_count=20,
                    net_pnl_aud=30.0,
                    expectancy=1.5,
                    profit_factor=1.1,
                    sharpe_like_score=0.3,
                    max_drawdown_pct=15.0,
                    total_fees_aud=12.0,
                    avg_realized_slippage_bps=6.0,
                ),
            }
        )
        weights = engine.maybe_update(
            cycle_id=1,
            strategy_names=["momentum", "breakout"],
            strategy_evaluation_engine=fake,
            regime_label="trend:mid_vol",
            run_id="run2",
            trace_id="trace2",
        )
        self.assertTrue(weights)

        with sqlite3.connect(db) as con:
            weight_rows = int(con.execute("SELECT COUNT(*) FROM strategy_weights").fetchone()[0])
            snapshot_rows = int(con.execute("SELECT COUNT(*) FROM meta_weight_snapshots").fetchone()[0])
        self.assertGreaterEqual(weight_rows, 2)
        self.assertGreaterEqual(snapshot_rows, 1)

        reloaded = SelfOptimizingMetaEngine(db_path=db)
        self.assertTrue(reloaded.current_strategy_weights())
        states = getattr(reloaded, "_states", {})
        self.assertTrue(states)
        sample_state = next(iter(states.values()))
        self.assertTrue(hasattr(sample_state, "drawdown_pct"))
        self.assertTrue(hasattr(sample_state, "slippage_bps"))

        sigs = [
            _Signal(symbol="BTC/USD", source_strategy="momentum", confidence=0.8, priority_score=0.8),
            _Signal(symbol="ETH/USD", source_strategy="breakout", confidence=0.8, priority_score=0.8),
        ]
        out = reloaded.apply_to_candidates(sigs)
        self.assertEqual(len(out), 2)
        self.assertTrue(hasattr(out[0], "strategy_weight"))
        self.assertTrue(hasattr(out[0], "meta_priority_adjustment"))
        self.assertTrue(hasattr(out[0], "weighting_reason"))


class _FakeMonitoring:
    async def update_metrics(self, _metrics):
        return None


class _FakeRiskManager:
    def update_capital(self, *_args, **_kwargs):
        return None

    def set_total_exposure(self, *_args, **_kwargs):
        return None

    def check_circuit_breaker(self):
        return False


class _FakeCapitalOptimizer:
    def update_capital(self, *_args, **_kwargs):
        return None

    async def optimize_signals(self, signals):
        return list(signals)


class _FakeExecutionEngine:
    def __init__(self):
        self.captured_signals = None
        self.trade_ledger = SimpleNamespace(record_event=lambda **_kwargs: None)

    async def execute_signals(self, signals, correlation_id=None):
        _ = correlation_id
        self.captured_signals = list(signals)
        return []


class _FakeAIBrain:
    def __init__(self, signals):
        self._signals = list(signals)

    async def generate_trading_signals(self):
        return list(self._signals)


class TestMetaIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_persists_weights_and_snapshot_fields(self) -> None:
        cfg = UnifiedConfig()
        cfg.run_mode = "paper"
        cfg.continuous_scan_enabled = False
        cfg.self_improvement_enabled = False
        cfg.ai_enabled = True
        cfg.quant_fund_upgrades_enabled = False
        cfg.targets_enabled = True
        cfg.target_convergence_alpha = 1.0
        cfg.target_cluster_cap_pct = 1.0
        cfg.feature_store_enabled = True
        cfg.feature_store_window = 10
        cfg.regime_classifier_enabled = True
        cfg.regime_gating_enabled = False
        cfg.reconciliation_interval_cycles = 0
        cfg.paper_trading_peak_mode = False
        cfg.max_trades_per_day = 0
        cfg.edge_cost_gate_enabled = False
        cfg.max_consecutive_losses = 1000
        cfg.multi_language_enabled = False
        cfg.self_optimizing_meta_enabled = True
        cfg.self_optimizing_meta_update_interval_cycles = 1
        cfg.self_optimizing_meta_min_trades_for_reweighting = 1
        cfg.self_optimizing_meta_alpha = 1.0
        cfg.self_optimizing_meta_max_weight_change_per_update = 1.0

        td = Path(tempfile.mkdtemp())
        cfg.self_optimizing_meta_db_path = str(td / "meta_weights.db")

        sys = UnifiedSystemArchitecture(cfg)
        sys.state = SystemState.RUNNING
        sys.monitoring = _FakeMonitoring()
        sys.unified_risk_manager = _FakeRiskManager()
        sys.capital_optimizer = _FakeCapitalOptimizer()
        sys.execution_engine = _FakeExecutionEngine()
        sys.ai_brain = _FakeAIBrain(
            [
                _Signal(
                    symbol="BTC/USD",
                    action="BUY",
                    confidence=0.9,
                    quantity=0.05,
                    entry_price=100.0,
                    strategy="momentum",
                    source_strategy="momentum",
                    spread_bps=8.0,
                    top_of_book_bid_size=5.0,
                    top_of_book_ask_size=5.0,
                )
            ]
        )
        sys.target_engine = TargetPortfolioEngine(cfg)

        omega_db = td / "omega_meta.db"
        sys.omega_store = OmegaSQLiteStore(str(omega_db))
        sys.omega_store.init_schema()

        await sys.run_trading_loop(cycle_seconds=0.0, max_cycles=1)

        with sqlite3.connect(str(cfg.self_optimizing_meta_db_path)) as con:
            w_rows = int(con.execute("SELECT COUNT(*) FROM strategy_weights").fetchone()[0])
            s_rows = int(con.execute("SELECT COUNT(*) FROM meta_weight_snapshots").fetchone()[0])
        self.assertGreaterEqual(w_rows, 1)
        self.assertGreaterEqual(s_rows, 1)

        conn = sqlite3.connect(str(omega_db))
        cur = conn.cursor()
        cur.execute(
            """
            SELECT details_json
            FROM decision_snapshots
            WHERE reason_code IN ('PRE_EXEC', 'PRE_PUBLISH')
            ORDER BY id DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        conn.close()
        self.assertIsNotNone(row)
        details = json.loads(str(row[0] or "{}"))
        self.assertIn("strategy_weight", details)
        self.assertIn("meta_priority_adjustment", details)
        self.assertIn("weighting_reason", details)


if __name__ == "__main__":
    unittest.main()
