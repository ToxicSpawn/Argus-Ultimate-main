from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Signal:
    symbol: str
    action: str
    confidence: float
    strength: float
    entry_price: float
    stop_loss: float | None = None
    take_profit: float | None = None


def test_capital_optimizer_produces_optimized_signal() -> None:
    import asyncio

    from unified_trading_system import UnifiedConfig
    from unified_capital_optimizer import CapitalOptimizer1K

    cfg = UnifiedConfig()
    cfg.starting_capital_aud = 1000.0
    cfg.min_position_size_aud = 5.0
    cfg.max_position_size_aud = 40.0
    cfg.max_position_pct = 0.04
    cfg.max_total_exposure_pct = 0.25
    cfg.slippage_pct = 0.002
    cfg.kraken_taker_fee = 0.0026
    cfg.coinbase_taker_fee = 0.008

    opt = CapitalOptimizer1K(cfg)

    async def _run():
        await opt.initialize()
        out = await opt.optimize_signals(
            [
                _Signal(
                    symbol="BTC/USD",
                    action="BUY",
                    confidence=0.9,
                    strength=0.8,
                    entry_price=50000.0,
                )
            ]
        )
        assert len(out) == 1
        s = out[0]
        assert s.symbol == "BTC/USD"
        assert s.action == "BUY"
        assert s.position_value_aud >= cfg.min_position_size_aud
        assert s.position_value_aud <= cfg.max_position_size_aud
        assert s.quantity > 0

    asyncio.run(_run())

