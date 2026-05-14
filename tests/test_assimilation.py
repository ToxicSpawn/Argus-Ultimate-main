"""Tests for the assimilation layer — bridging existing ARGUS to sealed runtime."""
import unittest


class TestAdvisoryBridge(unittest.TestCase):

    def test_classify_target_key(self):
        from argus_live.assimilation.advisory_bridge import classify_advisory_key
        self.assertEqual(classify_advisory_key("ensemble"), "target")
        self.assertEqual(classify_advisory_key("pretrained_alpha"), "target")

    def test_classify_risk_key(self):
        from argus_live.assimilation.advisory_bridge import classify_advisory_key
        self.assertEqual(classify_advisory_key("system_status"), "risk")
        self.assertEqual(classify_advisory_key("risk_score"), "risk")

    def test_classify_execution_hint(self):
        from argus_live.assimilation.advisory_bridge import classify_advisory_key
        self.assertEqual(classify_advisory_key("toxicity"), "execution_hint")

    def test_classify_informational(self):
        from argus_live.assimilation.advisory_bridge import classify_advisory_key
        self.assertEqual(classify_advisory_key("vol_forecasts"), "informational")
        self.assertEqual(classify_advisory_key("unknown_key"), "informational")

    def test_process_empty(self):
        from argus_live.assimilation.advisory_bridge import AdvisoryBridge
        bridge = AdvisoryBridge()
        self.assertEqual(bridge.process({}), [])

    def test_process_dict_advisory(self):
        from argus_live.assimilation.advisory_bridge import AdvisoryBridge
        bridge = AdvisoryBridge()
        proposals = bridge.process({
            "ensemble": {"composite": 0.5, "confidence": 0.8, "direction": "bullish"},
        })
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].source, "ensemble")
        self.assertEqual(proposals[0].category, "target")
        self.assertEqual(proposals[0].direction, "buy")

    def test_process_scalar_advisory(self):
        from argus_live.assimilation.advisory_bridge import AdvisoryBridge
        bridge = AdvisoryBridge()
        proposals = bridge.process({"risk_score": 0.85})
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].category, "risk")

    def test_blocked_keys_filtered(self):
        from argus_live.assimilation.advisory_bridge import AdvisoryBridge
        bridge = AdvisoryBridge(blocked_keys={"ensemble"})
        proposals = bridge.process({"ensemble": {"composite": 0.5}})
        self.assertEqual(len(proposals), 0)

    def test_system_status_critical(self):
        from argus_live.assimilation.advisory_bridge import AdvisoryBridge
        bridge = AdvisoryBridge()
        proposals = bridge.process({"system_status": {"status": "CRITICAL"}})
        self.assertEqual(proposals[0].direction, "reduce")
        self.assertAlmostEqual(proposals[0].strength, 1.0)


class TestExecutionBridge(unittest.TestCase):

    def test_paper_fill(self):
        from argus_live.assimilation.execution_bridge import ExecutionBridge, ExecutionRequest
        bridge = ExecutionBridge(mode="paper")
        req = ExecutionRequest(
            symbol="BTC/USD", side="buy", quantity=0.001, price=50000.0,
            order_type="market", strategy="momentum", confidence=0.8,
        )
        result = bridge.submit(req, portfolio_value=100000.0)  # large portfolio so $50 is within limits
        self.assertEqual(result.status, "filled")
        self.assertGreater(result.fill_price, 0)
        self.assertGreater(result.commission, 0)

    def test_constitution_blocks_oversized(self):
        from argus_live.assimilation.execution_bridge import ExecutionBridge, ExecutionRequest
        bridge = ExecutionBridge(mode="paper", max_single_exposure_pct=0.08)
        req = ExecutionRequest(
            symbol="BTC/USD", side="buy", quantity=1.0, price=50000.0,
            order_type="market", strategy="momentum", confidence=0.8,
        )
        result = bridge.submit(req, portfolio_value=1000.0)
        self.assertEqual(result.status, "rejected")
        self.assertIn("single_symbol_exposure", result.reason)

    def test_shadow_mode(self):
        from argus_live.assimilation.execution_bridge import ExecutionBridge, ExecutionRequest
        bridge = ExecutionBridge(mode="shadow")
        req = ExecutionRequest(
            symbol="ETH/USD", side="buy", quantity=0.1, price=2000.0,
            order_type="limit", strategy="mean_reversion", confidence=0.7,
        )
        result = bridge.submit(req, portfolio_value=10000.0)
        self.assertEqual(result.status, "shadow")

    def test_stats(self):
        from argus_live.assimilation.execution_bridge import ExecutionBridge, ExecutionRequest
        bridge = ExecutionBridge(mode="paper")
        req = ExecutionRequest(
            symbol="BTC/USD", side="buy", quantity=0.001, price=50000.0,
            order_type="market", strategy="test", confidence=0.5,
        )
        bridge.submit(req, portfolio_value=100000.0)
        stats = bridge.get_stats()
        self.assertEqual(stats["executed"], 1)


class TestStrategyBridge(unittest.TestCase):

    def test_convert_buy_signal(self):
        from argus_live.assimilation.strategy_bridge import StrategyBridge
        from types import SimpleNamespace
        bridge = StrategyBridge()
        signal = SimpleNamespace(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, strategy="momentum", reasoning="test",
        )
        proposal = bridge.convert_signal(signal)
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.direction, "long")
        self.assertEqual(proposal.strategy, "momentum")

    def test_promote_approved(self):
        from argus_live.assimilation.strategy_bridge import StrategyBridge, TargetProposal
        bridge = StrategyBridge(approved_strategies=["momentum"])
        proposal = TargetProposal(
            strategy="momentum", symbol="BTC/USD", target_weight_pct=0.10,
            direction="long", confidence=0.8, time_horizon_hours=1.0,
            execution_hint="limit", reasoning="test",
        )
        decision = bridge.promote(proposal)
        self.assertTrue(decision.approved)

    def test_promote_unapproved_rejected(self):
        from argus_live.assimilation.strategy_bridge import StrategyBridge, TargetProposal
        bridge = StrategyBridge(approved_strategies=["momentum"])
        proposal = TargetProposal(
            strategy="flash_loan_arb", symbol="BTC/USD", target_weight_pct=0.10,
            direction="long", confidence=0.9, time_horizon_hours=0.1,
            execution_hint="market", reasoning="arb",
        )
        decision = bridge.promote(proposal)
        self.assertFalse(decision.approved)
        self.assertIn("not in approved", decision.reason)

    def test_promote_disabled_rejected(self):
        from argus_live.assimilation.strategy_bridge import StrategyBridge, TargetProposal
        bridge = StrategyBridge(
            approved_strategies=["momentum"],
            disabled_strategies={"momentum"},
        )
        proposal = TargetProposal(
            strategy="momentum", symbol="BTC/USD", target_weight_pct=0.10,
            direction="long", confidence=0.8, time_horizon_hours=1.0,
            execution_hint="limit", reasoning="test",
        )
        decision = bridge.promote(proposal)
        self.assertFalse(decision.approved)
        self.assertIn("disabled", decision.reason)

    def test_promote_low_confidence_rejected(self):
        from argus_live.assimilation.strategy_bridge import StrategyBridge, TargetProposal
        bridge = StrategyBridge(min_confidence=0.50)
        proposal = TargetProposal(
            strategy="mean_reversion", symbol="ETH/USD", target_weight_pct=0.05,
            direction="long", confidence=0.30, time_horizon_hours=2.0,
            execution_hint="limit", reasoning="weak signal",
        )
        decision = bridge.promote(proposal)
        self.assertFalse(decision.approved)
        self.assertIn("confidence", decision.reason)

    def test_family_cap_applied(self):
        from argus_live.assimilation.strategy_bridge import StrategyBridge, TargetProposal
        bridge = StrategyBridge(family_caps={"momentum": 0.05})
        proposal = TargetProposal(
            strategy="momentum", symbol="BTC/USD", target_weight_pct=0.15,
            direction="long", confidence=0.9, time_horizon_hours=1.0,
            execution_hint="limit", reasoning="test",
        )
        decision = bridge.promote(proposal)
        self.assertTrue(decision.approved)
        self.assertAlmostEqual(decision.adjusted_weight_pct, 0.05)


if __name__ == "__main__":
    unittest.main()
