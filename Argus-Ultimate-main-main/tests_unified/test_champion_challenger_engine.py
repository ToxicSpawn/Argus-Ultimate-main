from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from evaluation.champion_challenger_engine import ChampionChallengerEngine
from evaluation.strategy_evaluation_engine import StrategyMetrics


class _FakeStrategyEvalEngine:
    def __init__(self, metrics_by_key):
        self.metrics_by_key = dict(metrics_by_key)

    def get_metrics(self, *, strategy_name: str, symbol=None, regime_label=None):
        key = (str(strategy_name), str(regime_label or ""))
        if key in self.metrics_by_key:
            return self.metrics_by_key[key]
        return self.metrics_by_key.get((str(strategy_name), ""))


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
    regime_label: str | None = None,
) -> StrategyMetrics:
    wins = max(0, int(round(trades_count * 0.6)))
    losses = max(0, trades_count - wins)
    return StrategyMetrics(
        strategy_name=strategy_name,
        symbol="__ALL__",
        trades_count=int(trades_count),
        wins_count=int(wins),
        losses_count=int(losses),
        win_rate=float(wins / max(1, trades_count)),
        gross_pnl_aud=float(net_pnl_aud + total_fees_aud),
        net_pnl_aud=float(net_pnl_aud),
        total_fees_aud=float(total_fees_aud),
        avg_net_pnl_per_trade=float(net_pnl_aud / max(1, trades_count)),
        avg_expected_net_edge_bps=5.0,
        avg_realized_slippage_bps=2.0,
        avg_hold_time_seconds=12.0,
        max_drawdown_pct=float(max_drawdown_pct),
        profit_factor=float(profit_factor),
        expectancy=float(expectancy),
        sharpe_like_score=float(sharpe_like_score),
        last_updated_ts=1.0,
        regime_label=regime_label,
        enabled_for_ranking=True,
    )


class TestChampionChallengerEngine(unittest.TestCase):
    def _make_paths(self) -> tuple[str, str]:
        base = tempfile.mkdtemp(prefix="argus_cc_")
        db_path = os.path.join(base, "champion_challenger.db")
        artifacts_dir = os.path.join(base, "promotions")
        return db_path, artifacts_dir

    def test_registration_persistence_and_reload(self) -> None:
        db_path, artifacts_dir = self._make_paths()
        engine = ChampionChallengerEngine(db_path=db_path, artifacts_dir=artifacts_dir)
        champion = engine.register_champion(
            profile_id="champion_a",
            source_bundle_path="deploy/bundles/c1",
            config_hash="hash1",
            strategy_set=["momentum"],
            version_label="v1",
            status="active",
        )
        challenger = engine.register_challenger(
            profile_id="challenger_a",
            source_bundle_path="deploy/bundles/ch1",
            parent_champion_id=champion.profile_id,
            strategy_set=["mean_reversion"],
            version_label="v1c",
            status="candidate",
        )

        self.assertEqual(engine.get_active_champion().profile_id, "champion_a")
        self.assertEqual(engine.list_challengers()[0].profile_id, "challenger_a")

        reloaded = ChampionChallengerEngine(db_path=db_path, artifacts_dir=artifacts_dir)
        self.assertEqual(reloaded.get_active_champion().profile_id, "champion_a")
        self.assertEqual(reloaded.list_challengers()[0].profile_id, challenger.profile_id)

    def test_hold_due_to_insufficient_trades(self) -> None:
        db_path, artifacts_dir = self._make_paths()
        engine = ChampionChallengerEngine(
            db_path=db_path,
            artifacts_dir=artifacts_dir,
            min_trades_for_promotion=10,
        )
        champion = engine.register_champion(
            profile_id="champion_b",
            source_bundle_path="",
            config_hash="hash2",
            strategy_set=["momentum"],
            version_label="v1",
            status="active",
        )
        engine.register_challenger(
            profile_id="challenger_b",
            source_bundle_path="",
            parent_champion_id=champion.profile_id,
            strategy_set=["challenger_low_trades"],
            version_label="v2",
            status="candidate",
        )

        fake = _FakeStrategyEvalEngine(
            {
                ("momentum", ""): _metric(
                    strategy_name="momentum",
                    trades_count=20,
                    net_pnl_aud=80.0,
                    expectancy=4.0,
                    profit_factor=1.6,
                    sharpe_like_score=1.2,
                    max_drawdown_pct=8.0,
                    total_fees_aud=10.0,
                ),
                ("challenger_low_trades", ""): _metric(
                    strategy_name="challenger_low_trades",
                    trades_count=3,
                    net_pnl_aud=40.0,
                    expectancy=13.0,
                    profit_factor=2.0,
                    sharpe_like_score=2.1,
                    max_drawdown_pct=2.0,
                    total_fees_aud=2.0,
                ),
            }
        )

        decision = engine.evaluate_challenger(
            challenger_id="challenger_b",
            strategy_evaluation_engine=fake,
        )
        self.assertEqual(decision.decision, "hold")
        self.assertTrue(any("insufficient_trades" in r for r in decision.reasons))

    def test_reject_due_to_drawdown(self) -> None:
        db_path, artifacts_dir = self._make_paths()
        engine = ChampionChallengerEngine(
            db_path=db_path,
            artifacts_dir=artifacts_dir,
            min_trades_for_promotion=5,
            max_drawdown_pct_for_promotion=0.12,
        )
        champion = engine.register_champion(
            profile_id="champion_c",
            source_bundle_path="",
            config_hash="hash3",
            strategy_set=["baseline"],
            version_label="v1",
            status="active",
        )
        engine.register_challenger(
            profile_id="challenger_c",
            source_bundle_path="",
            parent_champion_id=champion.profile_id,
            strategy_set=["risky_alpha"],
            version_label="v2",
            status="candidate",
        )

        fake = _FakeStrategyEvalEngine(
            {
                ("baseline", ""): _metric(
                    strategy_name="baseline",
                    trades_count=20,
                    net_pnl_aud=50.0,
                    expectancy=2.0,
                    profit_factor=1.4,
                    sharpe_like_score=0.8,
                    max_drawdown_pct=5.0,
                    total_fees_aud=8.0,
                ),
                ("risky_alpha", ""): _metric(
                    strategy_name="risky_alpha",
                    trades_count=20,
                    net_pnl_aud=120.0,
                    expectancy=6.0,
                    profit_factor=2.2,
                    sharpe_like_score=1.8,
                    max_drawdown_pct=35.0,
                    total_fees_aud=11.0,
                ),
            }
        )

        decision = engine.evaluate_challenger(
            challenger_id="challenger_c",
            strategy_evaluation_engine=fake,
        )
        self.assertEqual(decision.decision, "reject")
        self.assertTrue(any("drawdown_too_high" in r for r in decision.reasons))

    def test_promote_and_artifact_generation(self) -> None:
        db_path, artifacts_dir = self._make_paths()
        engine = ChampionChallengerEngine(
            db_path=db_path,
            artifacts_dir=artifacts_dir,
            min_trades_for_promotion=5,
            max_drawdown_pct_for_promotion=0.20,
            advisory_only=True,
        )
        champion = engine.register_champion(
            profile_id="champion_d",
            source_bundle_path="deploy/bundles/base",
            config_hash="hash4",
            strategy_set=["champion_strategy"],
            version_label="v1",
            status="active",
        )
        engine.register_challenger(
            profile_id="challenger_d",
            source_bundle_path="deploy/bundles/challenger",
            parent_champion_id=champion.profile_id,
            strategy_set=["challenger_strategy"],
            version_label="v2",
            status="candidate",
        )

        fake = _FakeStrategyEvalEngine(
            {
                ("champion_strategy", "trend"): _metric(
                    strategy_name="champion_strategy",
                    trades_count=30,
                    net_pnl_aud=90.0,
                    expectancy=3.0,
                    profit_factor=1.5,
                    sharpe_like_score=0.9,
                    max_drawdown_pct=9.0,
                    total_fees_aud=12.0,
                    regime_label="trend",
                ),
                ("challenger_strategy", "trend"): _metric(
                    strategy_name="challenger_strategy",
                    trades_count=30,
                    net_pnl_aud=160.0,
                    expectancy=5.0,
                    profit_factor=2.0,
                    sharpe_like_score=1.5,
                    max_drawdown_pct=8.0,
                    total_fees_aud=13.0,
                    regime_label="trend",
                ),
            }
        )

        decision = engine.evaluate_challenger(
            challenger_id="challenger_d",
            strategy_evaluation_engine=fake,
            regime_label="trend",
        )
        self.assertEqual(decision.decision, "promote")
        self.assertTrue(any("advisory_only_no_auto_promotion" in r for r in decision.reasons))
        self.assertGreater(decision.promotion_score, 0.0)
        artifact_path = decision.metrics_summary.get("artifact_path", "")
        self.assertTrue(artifact_path)
        self.assertTrue(os.path.exists(os.path.join(artifact_path, "promotion_decision.json")))
        self.assertTrue(os.path.exists(os.path.join(artifact_path, "promotion_note.md")))

        with sqlite3.connect(db_path) as con:
            count = int(con.execute("SELECT COUNT(*) FROM promotion_decisions").fetchone()[0])
        self.assertGreaterEqual(count, 1)

        best = engine.best_challengers_by_promotion_score(limit=1)
        self.assertEqual(best[0].challenger_id, "challenger_d")


if __name__ == "__main__":
    unittest.main()
