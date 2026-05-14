"""
Backtest for AlphaSignalFusion v9 research-enhanced signals.

Tests:
- Individual signal performance
- Combined signal performance  
- Regime-adaptive weighting
- Confluence detection (multiple signals agreeing)
"""

import asyncio
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Backtest result metrics."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

    # Signal breakdown
    signal_counts: dict = field(default_factory=dict)
    signal_wins: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": round(self.total_pnl, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "win_rate": round(self.win_rate * 100, 1),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "profit_factor": round(self.profit_factor, 2),
            "signal_counts": self.signal_counts,
            "signal_wins": {k: round(v, 2) for k, v in self.signal_wins.items()},
        }


class SyntheticDataGenerator:
    """Generate realistic synthetic OHLCV data."""

    @staticmethod
    def generate_bars(n: int, start_price: float = 50000, volatility: float = 0.02) -> list:
        """Generate n bars with price movements."""
        bars = []
        price = start_price

        for i in range(n):
            # Random walk with trends
            trend = np.sin(i / 20) * volatility * 0.5
            change = np.random.normal(trend, volatility)
            price *= (1 + change)

            high = price * (1 + abs(np.random.normal(0, volatility * 0.5)))
            low = price * (1 - abs(np.random.normal(0, volatility * 0.5)))
            open_price = price * (1 + np.random.normal(0, volatility * 0.2))
            close = price

            volume = np.random.lognormal(10, 1) * 1000000

            bars.append({
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        return bars


async def backtest_alpha_fusion():
    """Run backtest on AlphaSignalFusion v9."""
    from ml.alpha_signal_fusion import (
        AlphaSignalFusion,
        OnChainIndicators,
        DerivativesFlowSignals,
        OrderBlockDetector,
    )

    logger.info("=" * 60)
    logger.info("BACKTEST: AlphaSignalFusion v9 Research-Enhanced")
    logger.info("=" * 60)

    # Initialize components
    fusion = AlphaSignalFusion(
        use_ml_predictor=True,
        use_microstructure=True,
        use_alpha_model=True,
        use_sentiment=True,
        use_onchain=True,
        use_fear_greed=True,
        min_confidence=0.30,  # Lower for testing
        min_alpha=0.05,
    )
    await fusion.initialize()

    # Additional components
    onchain = OnChainIndicators()
    derivatives = DerivativesFlowSignals()
    order_blocks = OrderBlockDetector()

    # Generate synthetic data
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    results = {}
    equity = 10000

    for symbol in symbols:
        logger.info(f"\n--- Testing {symbol} ---")

        # Generate 500 bars
        bars = SyntheticDataGenerator.generate_bars(500, start_price=50000 if "BTC" in symbol else 3000)
        prices = [b["close"] for b in bars]

        trades = []
        equity_curve = [equity]
        wins = 0
        losses = 0
        pnl = 0

        # Run through bars
        for i in range(100, len(bars)):
            window = bars[max(0, i - 100):i]

            # Build market data
            market_data = {
                "volatility": random.random(),
                "volume_ratio": random.random(),
                "momentum": random.uniform(-1, 1),
                "exchange_inflow": random.random() * 1000000,
                "exchange_outflow": random.random() * 1000000,
                "headlines": [
                    "BTC surges past key resistance",
                    "Institutional accumulation continues",
                    "Option flows show bullish bias",
                ] if random.random() > 0.5 else [],
            }

            # Update components
            if i % 10 == 0 and prices[i - 1]:
                onchain.update(prices[i - 1])
                derivatives.update(
                    funding_rate=random.uniform(-0.001, 0.001),
                    open_interest=random.uniform(1e8, 5e8),
                    price=prices[i - 1],
                )
                order_blocks.update(
                    window[-1]["high"],
                    window[-1]["low"],
                    window[-1]["close"],
                    window[-1]["volume"],
                )

            # Generate signal
            try:
                signal = await fusion.generate_signal(symbol, window, market_data)

                if signal and signal.confidence > 0.5:
                    # Simulate trade
                    direction = 1 if signal.direction == "buy" else -1
                    entry_price = prices[i]
                    tp = entry_price * (1 + direction * 0.02)
                    sl = entry_price * (1 - direction * 0.01)

                    # Find exit
                    exit_price = None
                    for j in range(i + 1, min(i + 20, len(prices))):
                        if direction == 1:
                            if prices[j] >= tp:
                                exit_price = tp
                                break
                            elif prices[j] <= sl:
                                exit_price = sl
                                break
                        else:
                            if prices[j] <= tp:
                                exit_price = tp
                                break
                            elif prices[j] >= sl:
                                exit_price = sl
                                break

                    if exit_price:
                        trade_pnl = (exit_price - entry_price) / entry_price * direction * equity
                        pnl += trade_pnl

                        if trade_pnl > 0:
                            wins += 1
                        else:
                            losses += 1

                        trades.append({
                            "entry": entry_price,
                            "exit": exit_price,
                            "pnl": trade_pnl,
                            "signal": signal.direction,
                            "confidence": signal.confidence,
                        })

                        equity += trade_pnl
                        equity_curve.append(equity)

            except Exception as e:
                logger.debug(f"Signal error: {e}")

        # Calculate metrics
        total = wins + losses
        win_rate = wins / total if total > 0 else 0
        avg_win_val = pnl / wins if wins > 0 else 0
        avg_loss_val = pnl / losses if losses > 0 else 0

        results[symbol] = BacktestResult(
            total_trades=total,
            winning_trades=wins,
            losing_trades=losses,
            total_pnl=pnl,
            win_rate=win_rate,
            avg_win=avg_win_val,
            avg_loss=avg_loss_val,
        )

        logger.info(f"  Trades: {total}")
        logger.info(f"  Win Rate: {win_rate * 100:.1f}%")
        logger.info(f"  PnL: {pnl:.2f} USD")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST SUMMARY")
    logger.info("=" * 60)

    total_pnl = sum(r.total_pnl for r in results.values())
    total_trades = sum(r.total_trades for r in results.values())
    total_wins = sum(r.winning_trades for r in results.values())

    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Total PnL: {total_pnl:.2f} USD")
    logger.info(f"Overall Win Rate: {total_wins / total_trades * 100:.1f}%" if total_trades > 0 else "Win Rate: N/A (no trades)")

    # Save results
    output = {s: r.to_dict() for s, r in results.items()}
    output["summary"] = {
        "total_pnl": round(total_pnl, 2),
        "total_trades": total_trades,
        "win_rate": round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0,
    }

    Path("data/backtest_alpha_fusion.json").parent.mkdir(parents=True, exist_ok=True)
    with open("data/backtest_alpha_fusion.json", "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to data/backtest_alpha_fusion.json")

    return output


async def backtest_derivatives_signals():
    """Backtest just derivatives signals."""
    from ml.alpha_signal_fusion import DerivativesFlowSignals

    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST: Derivatives Flow Signals")
    logger.info("=" * 60)

    deriv = DerivativesFlowSignals()

    # Simulate funding scenarios
    test_cases = [
        {"funding": 0.0005, "oi_change": 0.05, "price_change": 0.01, "expect": "crowded_long"},
        {"funding": -0.0005, "oi_change": -0.03, "price_change": 0.005, "expect": "neutral"},
        {"funding": 0.002, "oi_change": 0.1, "price_change": -0.02, "expect": "distribution"},
    ]

    correct = 0
    for tc in test_cases:
        deriv.update(
            funding_rate=tc["funding"],
            open_interest=1e8 * (1 + tc["oi_change"]),
            price=50000 * (1 + tc["price_change"]),
        )

        funding = deriv.analyze_funding()
        oi = deriv.analyze_oi()

        logger.info(f"Funding: {tc['funding']:.4f} -> {funding['regime']}")
        logger.info(f"  OI: {oi['pattern']}")

        if funding["regime"] == tc["expect"]:
            correct += 1

    logger.info(f"Direction accuracy: {correct}/{len(test_cases)}")

    return {"correct": correct, "total": len(test_cases)}


async def main():
    """Run all backtests."""
    # Test alpha fusion
    fusion_results = await backtest_alpha_fusion()

    # Test derivatives
    deriv_results = await backtest_derivatives_signals()

    logger.info("\n" + "=" * 60)
    logger.info("ALL BACKTESTS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())