"""BacktestConfig dataclass — Push 59."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""

    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    start_date: Optional[Union[str, date]] = None
    end_date: Optional[Union[str, date]] = None
    initial_equity: float = 10_000.0
    fee_bps: float = 2.0
    slippage_bps: float = 0.5
    spread_bps: float = 1.0
    data_path: Optional[Union[str, Path]] = None
    resample_freq: Optional[str] = None   # e.g. "1h", "4h", "1d"
    warmup_bars: int = 0
    max_position_pct: float = 0.95
    risk_per_trade_pct: float = 1.0

    def to_dict(self) -> dict:
        return {
            "symbols": self.symbols,
            "start_date": str(self.start_date) if self.start_date else None,
            "end_date": str(self.end_date) if self.end_date else None,
            "initial_equity": self.initial_equity,
            "fee_bps": self.fee_bps,
            "slippage_bps": self.slippage_bps,
            "spread_bps": self.spread_bps,
            "data_path": str(self.data_path) if self.data_path else None,
            "resample_freq": self.resample_freq,
            "warmup_bars": self.warmup_bars,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BacktestConfig":
        return cls(
            symbols=d.get("symbols", ["BTCUSDT"]),
            start_date=d.get("start_date"),
            end_date=d.get("end_date"),
            initial_equity=float(d.get("initial_equity", 10_000)),
            fee_bps=float(d.get("fee_bps", 2.0)),
            slippage_bps=float(d.get("slippage_bps", 0.5)),
            spread_bps=float(d.get("spread_bps", 1.0)),
            data_path=d.get("data_path"),
            resample_freq=d.get("resample_freq"),
            warmup_bars=int(d.get("warmup_bars", 0)),
        )
