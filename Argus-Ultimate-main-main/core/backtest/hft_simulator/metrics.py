"""HFT execution and PnL metrics."""
# pyright: reportMissingImports=false

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np


@dataclass(slots=True)
class HFTMetrics:
    fill_rate: float = 0.0
    partial_fill_rate: float = 0.0
    maker_ratio: float = 0.0
    average_latency_us: float = 0.0
    average_queue_ahead: float = 0.0
    average_slippage_bps: float = 0.0
    adverse_selection_bps: float = 0.0
    realized_pnl: float = 0.0
    max_drawdown: float = 0.0
    inventory_turnover: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "fill_rate": self.fill_rate,
            "partial_fill_rate": self.partial_fill_rate,
            "maker_ratio": self.maker_ratio,
            "average_latency_us": self.average_latency_us,
            "average_queue_ahead": self.average_queue_ahead,
            "average_slippage_bps": self.average_slippage_bps,
            "adverse_selection_bps": self.adverse_selection_bps,
            "realized_pnl": self.realized_pnl,
            "max_drawdown": self.max_drawdown,
            "inventory_turnover": self.inventory_turnover,
        }


def compute_hft_metrics(fill_records: Sequence[dict[str, float | str | bool]], equity_curve: Sequence[float]) -> HFTMetrics:
    if not fill_records:
        return HFTMetrics(max_drawdown=_max_drawdown(equity_curve), realized_pnl=_pnl(equity_curve))
    requested = np.asarray([float(record.get("requested_quantity", 0.0)) for record in fill_records], dtype=float)
    filled = np.asarray([float(record.get("filled_quantity", 0.0)) for record in fill_records], dtype=float)
    latencies = np.asarray([float(record.get("latency_us", 0.0)) for record in fill_records], dtype=float)
    slippage = np.asarray([float(record.get("slippage_bps", 0.0)) for record in fill_records], dtype=float)
    queue_ahead = np.asarray([float(record.get("queue_ahead", 0.0)) for record in fill_records], dtype=float)
    makers = np.asarray([1.0 if bool(record.get("is_maker", False)) else 0.0 for record in fill_records], dtype=float)
    directions = np.asarray([1.0 if record.get("side") == "buy" else -1.0 for record in fill_records], dtype=float)
    prices = np.asarray([float(record.get("average_price", 0.0)) for record in fill_records], dtype=float)
    post_mid = np.asarray([float(record.get("post_trade_mid", record.get("average_price", 0.0))) for record in fill_records], dtype=float)
    fill_fraction = np.divide(filled, np.maximum(requested, 1e-12))
    partial_rate = float(np.mean((filled > 0) & (filled + 1e-12 < requested)))
    adverse = np.zeros(len(fill_records), dtype=float)
    valid = (prices > 0) & (post_mid > 0)
    adverse[valid] = directions[valid] * (post_mid[valid] - prices[valid]) / prices[valid] * 10_000.0
    inventory_turnover = float(np.sum(np.abs(filled), dtype=float) / max(np.max(np.abs(np.cumsum(directions * filled, dtype=float))), 1.0))
    return HFTMetrics(
        fill_rate=float(np.mean(np.clip(fill_fraction, 0.0, 1.0))),
        partial_fill_rate=partial_rate,
        maker_ratio=float(np.mean(makers)),
        average_latency_us=float(np.mean(latencies)),
        average_queue_ahead=float(np.mean(queue_ahead)),
        average_slippage_bps=float(np.mean(slippage)),
        adverse_selection_bps=float(np.mean(adverse)),
        realized_pnl=_pnl(equity_curve),
        max_drawdown=_max_drawdown(equity_curve),
        inventory_turnover=inventory_turnover,
    )


def _pnl(equity_curve: Sequence[float]) -> float:
    if len(equity_curve) < 2:
        return 0.0
    return float(equity_curve[-1] - equity_curve[0])


def _max_drawdown(equity_curve: Sequence[float]) -> float:
    if not equity_curve:
        return 0.0
    equity = np.asarray(equity_curve, dtype=float)
    peaks = np.maximum.accumulate(equity)
    drawdowns = (peaks - equity) / np.maximum(peaks, 1e-12)
    return float(np.max(drawdowns)) if drawdowns.size else 0.0
