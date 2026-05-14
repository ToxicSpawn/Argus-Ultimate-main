from __future__ import annotations

import unittest

from unified_trading_system import UnifiedConfig, UnifiedSystemArchitecture


class TestPR05LedgerHardening(unittest.IsolatedAsyncioTestCase):
    async def test_open_mark_close_fee_aware_pnl_lifecycle(self) -> None:
        cfg = UnifiedConfig(starting_capital_aud=1000.0, aud_to_usd=1.0)
        system = UnifiedSystemArchitecture(cfg)

        system._record_trade(
            {
                "order_id": "buy_1",
                "symbol": "BTC/USD",
                "side": "BUY",
                "quantity": 1.0,
                "price": 100.0,
                "commission": 1.0,
            }
        )
        self.assertAlmostEqual(system.cash_balance_aud, 899.0, places=6)
        self.assertAlmostEqual(float(system.positions["BTC/USD"]["quantity"]), 1.0, places=6)
        # Entry fee is included in cost basis.
        self.assertAlmostEqual(float(system.positions["BTC/USD"]["avg_price"]), 101.0, places=6)
        self.assertAlmostEqual(system.realized_pnl_aud, 0.0, places=6)
        self.assertAlmostEqual(system.total_fees_aud, 1.0, places=6)

        system.positions["BTC/USD"]["current_price"] = 110.0
        await system._update_portfolio_value()
        self.assertAlmostEqual(system.unrealized_pnl_aud, 9.0, places=6)
        self.assertAlmostEqual(system.portfolio_value_aud, 1009.0, places=6)

        system._record_trade(
            {
                "order_id": "sell_1",
                "symbol": "BTC/USD",
                "side": "SELL",
                "quantity": 1.0,
                "price": 110.0,
                "commission": 1.0,
            }
        )
        await system._update_portfolio_value()

        # Net realized pnl accounts for both entry and exit fees.
        self.assertAlmostEqual(system.realized_pnl_aud, 8.0, places=6)
        self.assertAlmostEqual(system.total_pnl_aud, 8.0, places=6)
        self.assertAlmostEqual(system.daily_pnl_aud, 8.0, places=6)
        self.assertAlmostEqual(system.total_fees_aud, 2.0, places=6)
        self.assertAlmostEqual(system.cash_balance_aud, 1008.0, places=6)
        self.assertAlmostEqual(float(system.positions["BTC/USD"]["quantity"]), 0.0, places=6)
        self.assertAlmostEqual(system.unrealized_pnl_aud, 0.0, places=6)
        self.assertAlmostEqual(system.portfolio_value_aud, 1008.0, places=6)
        self.assertEqual(int(system._ledger_sanity_violations), 0)

        status = system.get_status()
        self.assertAlmostEqual(float(status["realized_pnl_aud"]), 8.0, places=6)
        self.assertAlmostEqual(float(status["unrealized_pnl_aud"]), 0.0, places=6)
        self.assertAlmostEqual(float(status["total_fees_aud"]), 2.0, places=6)
        self.assertEqual(str(status["mark_price_method"]), "position.current_price")

    async def test_impossible_sell_when_flat_is_rejected_safely(self) -> None:
        cfg = UnifiedConfig(starting_capital_aud=1000.0, aud_to_usd=1.0)
        system = UnifiedSystemArchitecture(cfg)
        before_cash = float(system.cash_balance_aud)

        system._record_trade(
            {
                "order_id": "sell_flat",
                "symbol": "ETH/USD",
                "side": "SELL",
                "quantity": 0.5,
                "price": 2000.0,
                "commission": 0.0,
            }
        )

        self.assertAlmostEqual(system.cash_balance_aud, before_cash, places=6)
        self.assertGreaterEqual(int(system._ledger_sanity_violations), 1)

