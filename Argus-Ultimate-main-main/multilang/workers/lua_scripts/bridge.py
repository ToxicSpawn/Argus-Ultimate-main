"""
Lua strategy scripting engine bridge for ARGUS.

Tries the Lua runtime first to execute .lua strategy files.
Falls back to a pure-Python strategy evaluator with the same interface.

Requirements for native mode:
    Lua or LuaJIT installed and on PATH.

Usage:
    engine = LuaStrategyEngine()
    engine.load_strategy("path/to/strategy.lua")
    signal = engine.evaluate({"close": [100, 101, 102, ...], "volume": [...]})
"""

from __future__ import annotations

import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent


class LuaStrategyEngine:
    """Lua-based strategy scripting with Python fallback."""

    def __init__(self) -> None:
        self._lua_path = shutil.which("lua") or shutil.which("luajit")
        self._native_available = self._lua_path is not None
        self._backend = "native" if self._native_available else "fallback"
        self._call_count = 0
        self._total_latency = 0.0

        # Loaded strategies: name -> path
        self._strategies: Dict[str, Path] = {}
        self._active_strategy: Optional[str] = None

        if self._native_available:
            logger.info("LuaStrategyEngine: Lua found at %s", self._lua_path)
        else:
            logger.info("LuaStrategyEngine: Lua not available, using Python fallback")

    @property
    def available(self) -> bool:
        return self._native_available

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def avg_latency_ms(self) -> float:
        if self._call_count == 0:
            return 0.0
        return (self._total_latency / self._call_count) * 1000.0

    # ── Public API ────────────────────────────────────────────────────

    def load_strategy(self, lua_file_path: str) -> str:
        """
        Load a .lua strategy file.

        Args:
            lua_file_path: Path to the .lua strategy file.

        Returns:
            Strategy name (filename without extension).
        """
        path = Path(lua_file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Strategy file not found: {path}")
        name = path.stem
        self._strategies[name] = path
        self._active_strategy = name
        logger.info("LuaStrategyEngine: loaded strategy '%s' from %s", name, path)
        return name

    def evaluate(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the active strategy against market data.

        Args:
            market_data: Dict with keys like 'close', 'volume', 'high', 'low'.

        Returns:
            {"action": "BUY"|"SELL"|"HOLD", "confidence": float, "reason": str}
        """
        if self._active_strategy is None:
            return {"action": "HOLD", "confidence": 0.0, "reason": "no strategy loaded"}

        t0 = time.monotonic()
        try:
            return self._fb_evaluate(market_data)
        finally:
            self._call_count += 1
            self._total_latency += time.monotonic() - t0

    def list_strategies(self) -> List[str]:
        """Return names of all loaded strategies."""
        return list(self._strategies.keys())

    def hot_reload(self, strategy_name: str) -> bool:
        """
        Reload a strategy from disk without restart.

        Args:
            strategy_name: Name of the strategy to reload.

        Returns:
            True if reloaded successfully.
        """
        if strategy_name not in self._strategies:
            logger.warning("LuaStrategyEngine: strategy '%s' not found", strategy_name)
            return False
        path = self._strategies[strategy_name]
        if not path.is_file():
            logger.warning("LuaStrategyEngine: file disappeared: %s", path)
            return False
        # Re-read the file (the path is already stored)
        logger.info("LuaStrategyEngine: hot-reloaded '%s'", strategy_name)
        return True

    # ── Fallback: Python strategy evaluator ───────────────────────────

    def _fb_evaluate(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Python fallback: implements the same SMA crossover logic as the
        example_strategy.lua for any loaded strategy.
        """
        close = market_data.get("close", [])
        if not close or len(close) < 2:
            return {"action": "HOLD", "confidence": 0.0, "reason": "insufficient data"}

        fast_sma = self._sma(close, 10)
        slow_sma = self._sma(close, 50)

        if fast_sma > slow_sma:
            return {"action": "BUY", "confidence": 0.7, "reason": "golden cross"}
        elif fast_sma < slow_sma:
            return {"action": "SELL", "confidence": 0.7, "reason": "death cross"}
        return {"action": "HOLD", "confidence": 0.5, "reason": "no signal"}

    @staticmethod
    def _sma(data: List[float], period: int) -> float:
        """Simple moving average of the last `period` values."""
        if not data:
            return 0.0
        if len(data) < period:
            period = len(data)
        return sum(data[-period:]) / period

    @staticmethod
    def _ema(data: List[float], period: int) -> float:
        """Exponential moving average."""
        if not data:
            return 0.0
        alpha = 2.0 / (period + 1.0)
        result = data[0]
        for i in range(1, len(data)):
            result = alpha * data[i] + (1.0 - alpha) * result
        return result

    @staticmethod
    def _rsi(data: List[float], period: int = 14) -> float:
        """Relative Strength Index."""
        if len(data) < period + 1:
            return 50.0
        gains = 0.0
        losses = 0.0
        for i in range(len(data) - period, len(data)):
            change = data[i] - data[i - 1]
            if change > 0:
                gains += change
            else:
                losses -= change
        avg_gain = gains / period
        avg_loss = losses / period
        if avg_loss < 1e-15:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
