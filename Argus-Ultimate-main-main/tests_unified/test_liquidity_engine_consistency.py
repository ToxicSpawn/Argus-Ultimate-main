from __future__ import annotations

import unittest
from types import SimpleNamespace

from risk.liquidity_risk_engine import LiquidityRiskEngine


class _Signal(SimpleNamespace):
    pass


def _cfg(**overrides):
    cfg = SimpleNamespace(
        liquidity_risk_depth_fraction_limit=0.04,
        liquidity_risk_thin_spread_threshold_bps=6.0,
        liquidity_risk_danger_spread_threshold_bps=12.0,
        liquidity_risk_min_depth_threshold=0.5,
        liquidity_risk_slippage_threshold_bps=10.0,
        liquidity_risk_min_liquidity_score=0.2,
        liquidity_risk_score_weights={"depth": 1.0, "spread": 1.0, "fill_ratio": 0.75},
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestLiquidityEngineConsistency(unittest.TestCase):
    def test_normal_liquidity_has_positive_safe_size(self) -> None:
        engine = LiquidityRiskEngine(_cfg())
        state = engine.evaluate_state(
            symbol="BTC/USD",
            spread_bps=2.0,
            bid_size=8.0,
            ask_size=8.0,
            depth_estimate=16.0,
            slippage_estimate_bps=2.0,
            maker_fill_ratio=0.7,
        )
        self.assertEqual(state.liquidity_state, "normal")
        self.assertGreater(state.max_safe_trade_size, 0.0)

    def test_zero_safe_size_never_reports_normal_state(self) -> None:
        engine = LiquidityRiskEngine(_cfg(liquidity_risk_min_liquidity_score=0.95))
        state = engine.evaluate_state(
            symbol="ETH/USD",
            spread_bps=2.0,
            bid_size=0.0,
            ask_size=0.0,
            depth_estimate=0.0,
            slippage_estimate_bps=3.0,
            maker_fill_ratio=0.0,
        )
        self.assertEqual(state.max_safe_trade_size, 0.0)
        self.assertIn(state.liquidity_state, {"thin", "danger"})

    def test_danger_liquidity_detected_on_wide_spread(self) -> None:
        engine = LiquidityRiskEngine(_cfg())
        state = engine.evaluate_state(
            symbol="SOL/USD",
            spread_bps=25.0,
            bid_size=1.0,
            ask_size=1.0,
            depth_estimate=2.0,
            slippage_estimate_bps=15.0,
            maker_fill_ratio=0.2,
        )
        self.assertEqual(state.liquidity_state, "danger")

    def test_target_clamp_and_thin_suppression_reason(self) -> None:
        engine = LiquidityRiskEngine(_cfg())
        target = _Signal(
            symbol="ETH/USD",
            target_exposure_pct=0.20,
            current_exposure_pct=0.00,
            delta_exposure_pct=0.20,
            priority_score=1.0,
            expected_net_edge_bps=10.0,
            regime_label="range:mid_vol",
            reasons=[],
            price=100.0,
            reference_price=100.0,
            current_qty=0.0,
            target_qty=2.0,
            delta_qty=2.0,
        )
        adjusted, states, clamp_count = engine.adjust_targets(
            targets=[target],
            symbol_market_state={
                "ETH/USD": {
                    "spread_bps": 7.0,
                    "top_of_book_bid_size": 0.0,
                    "top_of_book_ask_size": 0.0,
                    "orderbook_depth_estimate": 0.0,
                    "price": 100.0,
                }
            },
            execution_telemetry={"ETH/USD": {"maker_fill_ratio": 0.1, "slippage_p90": 20.0}},
            equity_aud=1000.0,
            aud_to_usd=1.0,
        )
        self.assertGreaterEqual(clamp_count, 1)
        self.assertIn(states["ETH/USD"].liquidity_state, {"thin", "danger"})
        row = adjusted[0]
        self.assertTrue(bool(getattr(row, "liquidity_clamp_flag", False)))
        self.assertAlmostEqual(float(getattr(row, "delta_qty", 0.0)), 0.0, places=9)
        self.assertTrue(
            any(
                str(r).startswith("suppressed:liquidity_thin")
                or str(r).startswith("suppressed:liquidity_danger")
                for r in list(getattr(row, "reasons", []) or [])
            )
        )


if __name__ == "__main__":
    unittest.main()
