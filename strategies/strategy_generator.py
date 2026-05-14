#!/usr/bin/env python3
"""
Self-Generating Strategies via LLM — Tier 3 Self-Improvement Module.

Generates new strategy ideas either via an LLM client or rule-based templates,
backtests them against OHLCV data, and promotes winners to a persistent store.

Usage (standalone)::

    gen = StrategyGenerator()
    idea = gen.generate_strategy_idea({"regime": "bull", "volatility": "high"})
    result = gen.backtest_idea(idea, ohlcv_data)
    if result.passed:
        gen.promote_strategy(idea, result)
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StrategyIdea:
    """A generated strategy concept ready for backtesting."""

    name: str
    description: str
    entry_logic: str
    exit_logic: str
    regime_preference: str
    risk_params: Dict[str, Any]
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class BacktestResult:
    """Results from backtesting a StrategyIdea against historical data."""

    idea_name: str
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    trade_count: int
    passed: bool


# ---------------------------------------------------------------------------
# Rule-based templates
# ---------------------------------------------------------------------------

_RULE_TEMPLATES: List[Dict[str, Any]] = [
    {
        "name": "rsi_oversold_bounce",
        "description": "Buy when RSI drops below 30 (oversold), sell when RSI exceeds 70 (overbought).",
        "entry_logic": "RSI(14) < 30",
        "exit_logic": "RSI(14) > 70",
        "regime_preference": "ranging",
        "risk_params": {"stop_loss_pct": 2.0, "take_profit_pct": 6.0, "position_size_pct": 10.0},
        "confidence": 0.65,
    },
    {
        "name": "macd_crossover",
        "description": "Enter long on MACD line crossing above signal line; exit on cross below.",
        "entry_logic": "MACD_line crosses above Signal_line",
        "exit_logic": "MACD_line crosses below Signal_line",
        "regime_preference": "trending",
        "risk_params": {"stop_loss_pct": 1.5, "take_profit_pct": 5.0, "position_size_pct": 12.0},
        "confidence": 0.60,
    },
    {
        "name": "bollinger_mean_reversion",
        "description": "Buy when price touches lower Bollinger Band; sell at upper band.",
        "entry_logic": "close <= BB_lower(20, 2)",
        "exit_logic": "close >= BB_upper(20, 2)",
        "regime_preference": "ranging",
        "risk_params": {"stop_loss_pct": 2.5, "take_profit_pct": 4.0, "position_size_pct": 8.0},
        "confidence": 0.55,
    },
    {
        "name": "volume_breakout",
        "description": "Enter when volume exceeds 2x 20-period average and price breaks above recent high.",
        "entry_logic": "volume > 2 * SMA(volume, 20) AND close > highest(close, 20)",
        "exit_logic": "close < SMA(close, 10) OR volume < SMA(volume, 20)",
        "regime_preference": "breakout",
        "risk_params": {"stop_loss_pct": 3.0, "take_profit_pct": 8.0, "position_size_pct": 10.0},
        "confidence": 0.58,
    },
]


# ---------------------------------------------------------------------------
# Strategy Generator
# ---------------------------------------------------------------------------


class StrategyGenerator:
    """
    Generates, backtests, and promotes trading strategy ideas.

    When an LLM client is available the generator delegates idea creation to
    the model.  Otherwise it falls back to a library of rule-based indicator
    templates that combine RSI, MACD, Bollinger Bands, and volume signals.

    Promoted strategies are persisted to ``data/generated_strategies.json``
    relative to the project root so they survive restarts.
    """

    # Pass thresholds for backtest results
    MIN_SHARPE: float = 0.5
    MAX_DRAWDOWN_PCT: float = 20.0
    MIN_TRADE_COUNT: int = 10

    def __init__(
        self,
        data_dir: Optional[str] = None,
        *,
        min_sharpe: float = 0.5,
        max_drawdown_pct: float = 20.0,
        min_trade_count: int = 10,
    ) -> None:
        if data_dir is None:
            data_dir = str(Path(__file__).resolve().parent.parent / "data")
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._promoted_path = self._data_dir / "generated_strategies.json"
        self.MIN_SHARPE = min_sharpe
        self.MAX_DRAWDOWN_PCT = max_drawdown_pct
        self.MIN_TRADE_COUNT = min_trade_count
        log.info(
            "StrategyGenerator initialised — data_dir=%s  pass_criteria=(sharpe>%.2f, dd<%.1f%%, trades>%d)",
            self._data_dir,
            self.MIN_SHARPE,
            self.MAX_DRAWDOWN_PCT,
            self.MIN_TRADE_COUNT,
        )

    # ------------------------------------------------------------------
    # Idea generation
    # ------------------------------------------------------------------

    def generate_strategy_idea(
        self,
        market_conditions: Dict[str, Any],
        llm_client: Any = None,
    ) -> StrategyIdea:
        """
        Generate a new strategy idea.

        Parameters
        ----------
        market_conditions:
            Dict describing the current market state (e.g. regime, volatility,
            trend direction).  Passed to the LLM prompt or used to bias
            template selection.
        llm_client:
            Optional object with a ``generate(prompt: str) -> str`` method.
            When *None* the generator falls back to rule-based templates.

        Returns
        -------
        StrategyIdea
            A fully populated idea ready for backtesting.
        """
        if llm_client is not None:
            return self._generate_via_llm(market_conditions, llm_client)
        return self._generate_rule_based(market_conditions)

    def _generate_via_llm(
        self,
        market_conditions: Dict[str, Any],
        llm_client: Any,
    ) -> StrategyIdea:
        """Ask an LLM to propose a strategy idea in structured JSON."""
        prompt = (
            "You are a quantitative trading strategy designer.  Given the current market "
            "conditions, propose a novel crypto trading strategy.\n\n"
            f"Market conditions: {json.dumps(market_conditions)}\n\n"
            "Respond ONLY with a JSON object containing these fields:\n"
            "  name (str), description (str), entry_logic (str), exit_logic (str),\n"
            "  regime_preference (str), risk_params (dict with stop_loss_pct, take_profit_pct, "
            "position_size_pct), confidence (float 0-1).\n"
        )
        try:
            raw = llm_client.generate(prompt)
            # Strip markdown fences if present
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[: text.rfind("```")]
            data = json.loads(text)
            idea = StrategyIdea(
                name=str(data.get("name", "llm_strategy")),
                description=str(data.get("description", "")),
                entry_logic=str(data.get("entry_logic", "")),
                exit_logic=str(data.get("exit_logic", "")),
                regime_preference=str(data.get("regime_preference", "any")),
                risk_params=data.get("risk_params", {}),
                confidence=float(data.get("confidence", 0.5)),
            )
            log.info("LLM generated strategy idea: %s (confidence=%.2f)", idea.name, idea.confidence)
            return idea
        except Exception:
            log.warning("LLM strategy generation failed, falling back to rule-based templates", exc_info=True)
            return self._generate_rule_based(market_conditions)

    def _generate_rule_based(self, market_conditions: Dict[str, Any]) -> StrategyIdea:
        """Select and adapt a rule-based template based on market conditions."""
        regime = str(market_conditions.get("regime", "")).lower()

        # Prefer templates that match the current regime
        matching = [t for t in _RULE_TEMPLATES if t["regime_preference"] == regime]
        if not matching:
            matching = list(_RULE_TEMPLATES)

        template = random.choice(matching)  # noqa: S311 — non-security use

        # Adapt risk params by volatility
        risk = dict(template["risk_params"])
        vol = str(market_conditions.get("volatility", "")).lower()
        if vol == "high":
            risk["stop_loss_pct"] = risk.get("stop_loss_pct", 2.0) * 1.5
            risk["position_size_pct"] = risk.get("position_size_pct", 10.0) * 0.7
        elif vol == "low":
            risk["stop_loss_pct"] = risk.get("stop_loss_pct", 2.0) * 0.75
            risk["position_size_pct"] = risk.get("position_size_pct", 10.0) * 1.2

        idea = StrategyIdea(
            name=template["name"],
            description=template["description"],
            entry_logic=template["entry_logic"],
            exit_logic=template["exit_logic"],
            regime_preference=template["regime_preference"],
            risk_params=risk,
            confidence=template["confidence"],
        )
        log.info("Rule-based strategy idea selected: %s (regime=%s)", idea.name, regime or "any")
        return idea

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------

    def backtest_idea(self, idea: StrategyIdea, ohlcv_data: List[Dict[str, Any]]) -> BacktestResult:
        """
        Run a simplified backtest of *idea* against *ohlcv_data*.

        Parameters
        ----------
        idea:
            The strategy idea to evaluate.
        ohlcv_data:
            List of dicts with at least ``open``, ``high``, ``low``, ``close``,
            ``volume`` keys.  Must be in chronological order.

        Returns
        -------
        BacktestResult
            Performance metrics plus a ``passed`` flag.

        Notes
        -----
        This is a *screening* backtest, not a full simulation.  It applies a
        simplified signal model derived from the entry/exit logic strings to
        produce approximate metrics suitable for strategy promotion decisions.
        """
        if len(ohlcv_data) < 30:
            log.warning("Insufficient OHLCV data for backtest (%d bars)", len(ohlcv_data))
            return BacktestResult(
                idea_name=idea.name,
                total_return_pct=0.0,
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                win_rate=0.0,
                trade_count=0,
                passed=False,
            )

        trades = self._simulate_trades(idea, ohlcv_data)
        trade_count = len(trades)

        if trade_count == 0:
            return BacktestResult(
                idea_name=idea.name,
                total_return_pct=0.0,
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                win_rate=0.0,
                trade_count=0,
                passed=False,
            )

        # Compute metrics
        returns = [t["return_pct"] for t in trades]
        total_return = sum(returns)
        wins = sum(1 for r in returns if r > 0)
        win_rate = wins / trade_count if trade_count > 0 else 0.0

        # Sharpe ratio (annualised assuming ~365 daily trades per year)
        import statistics

        mean_ret = statistics.mean(returns) if returns else 0.0
        std_ret = statistics.stdev(returns) if len(returns) > 1 else 1e-9
        sharpe = (mean_ret / std_ret) * (365**0.5) if std_ret > 1e-9 else 0.0

        # Max drawdown from cumulative equity curve
        equity = 100.0
        peak = equity
        max_dd = 0.0
        for r in returns:
            equity *= 1 + r / 100.0
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100.0
            if dd > max_dd:
                max_dd = dd

        passed = (
            sharpe > self.MIN_SHARPE
            and max_dd < self.MAX_DRAWDOWN_PCT
            and trade_count >= self.MIN_TRADE_COUNT
        )

        result = BacktestResult(
            idea_name=idea.name,
            total_return_pct=round(total_return, 4),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown_pct=round(max_dd, 4),
            win_rate=round(win_rate, 4),
            trade_count=trade_count,
            passed=passed,
        )
        log.info(
            "Backtest %s — return=%.2f%% sharpe=%.2f dd=%.2f%% trades=%d passed=%s",
            idea.name,
            result.total_return_pct,
            result.sharpe_ratio,
            result.max_drawdown_pct,
            result.trade_count,
            result.passed,
        )
        return result

    def _simulate_trades(
        self,
        idea: StrategyIdea,
        ohlcv_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Produce a list of simulated trades from the idea's entry/exit logic.

        Uses a simplified signal model that maps indicator keywords in the
        entry/exit logic to lightweight calculations on the OHLCV data.
        """
        closes = [float(bar["close"]) for bar in ohlcv_data]
        volumes = [float(bar.get("volume", 0)) for bar in ohlcv_data]
        n = len(closes)
        entry_logic = idea.entry_logic.lower()
        exit_logic = idea.exit_logic.lower()

        # Pre-compute indicators we might need
        rsi_values = self._compute_rsi(closes, 14)
        sma_20 = self._compute_sma(closes, 20)
        sma_vol_20 = self._compute_sma(volumes, 20)
        bb_upper, bb_lower = self._compute_bollinger(closes, 20, 2.0)

        # MACD (12, 26, 9)
        ema_12 = self._compute_ema(closes, 12)
        ema_26 = self._compute_ema(closes, 26)
        macd_line = [e12 - e26 for e12, e26 in zip(ema_12, ema_26)]
        macd_signal = self._compute_ema(macd_line, 9)

        stop_loss_pct = idea.risk_params.get("stop_loss_pct", 2.0)
        take_profit_pct = idea.risk_params.get("take_profit_pct", 6.0)

        trades: List[Dict[str, Any]] = []
        in_position = False
        entry_price = 0.0

        for i in range(26, n):  # start after longest indicator warm-up
            if not in_position:
                # Check entry conditions
                enter = False
                if "rsi" in entry_logic and rsi_values[i] is not None:
                    if rsi_values[i] < 30:
                        enter = True
                if "macd" in entry_logic:
                    if i > 0 and macd_line[i] > macd_signal[i] and macd_line[i - 1] <= macd_signal[i - 1]:
                        enter = True
                if "bb_lower" in entry_logic and bb_lower[i] is not None:
                    if closes[i] <= bb_lower[i]:
                        enter = True
                if "volume" in entry_logic and sma_vol_20[i] is not None:
                    if volumes[i] > 2 * sma_vol_20[i] and closes[i] > max(closes[max(0, i - 20) : i]):
                        enter = True

                if enter:
                    in_position = True
                    entry_price = closes[i]
            else:
                # Check exit conditions
                ret_pct = (closes[i] - entry_price) / entry_price * 100.0
                exit_trade = False

                # Stop loss / take profit
                if ret_pct <= -stop_loss_pct or ret_pct >= take_profit_pct:
                    exit_trade = True

                # Signal-based exit
                if "rsi" in exit_logic and rsi_values[i] is not None:
                    if rsi_values[i] > 70:
                        exit_trade = True
                if "macd" in exit_logic:
                    if i > 0 and macd_line[i] < macd_signal[i] and macd_line[i - 1] >= macd_signal[i - 1]:
                        exit_trade = True
                if "bb_upper" in exit_logic and bb_upper[i] is not None:
                    if closes[i] >= bb_upper[i]:
                        exit_trade = True
                if "sma" in exit_logic and sma_20[i] is not None:
                    if closes[i] < sma_20[i]:
                        exit_trade = True

                if exit_trade:
                    trades.append({"return_pct": ret_pct})
                    in_position = False

        return trades

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rsi(closes: List[float], period: int) -> List[Optional[float]]:
        """Compute RSI series.  Returns None for indices before warm-up."""
        result: List[Optional[float]] = [None] * len(closes)
        if len(closes) < period + 1:
            return result
        gains = []
        losses = []
        for i in range(1, period + 1):
            delta = closes[i] - closes[i - 1]
            gains.append(max(delta, 0))
            losses.append(max(-delta, 0))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            result[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[period] = 100 - 100 / (1 + rs)
        for i in range(period + 1, len(closes)):
            delta = closes[i] - closes[i - 1]
            avg_gain = (avg_gain * (period - 1) + max(delta, 0)) / period
            avg_loss = (avg_loss * (period - 1) + max(-delta, 0)) / period
            if avg_loss == 0:
                result[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                result[i] = 100 - 100 / (1 + rs)
        return result

    @staticmethod
    def _compute_sma(values: List[float], period: int) -> List[Optional[float]]:
        """Simple moving average.  Returns None before warm-up."""
        result: List[Optional[float]] = [None] * len(values)
        if len(values) < period:
            return result
        window_sum = sum(values[:period])
        result[period - 1] = window_sum / period
        for i in range(period, len(values)):
            window_sum += values[i] - values[i - period]
            result[i] = window_sum / period
        return result

    @staticmethod
    def _compute_ema(values: List[float], period: int) -> List[float]:
        """Exponential moving average over *values*."""
        if not values:
            return []
        k = 2 / (period + 1)
        ema = [values[0]]
        for i in range(1, len(values)):
            ema.append(values[i] * k + ema[-1] * (1 - k))
        return ema

    @staticmethod
    def _compute_bollinger(
        closes: List[float],
        period: int,
        num_std: float,
    ) -> tuple:
        """Return (upper, lower) Bollinger Band series."""
        n = len(closes)
        upper: List[Optional[float]] = [None] * n
        lower: List[Optional[float]] = [None] * n
        if n < period:
            return upper, lower
        for i in range(period - 1, n):
            window = closes[i - period + 1 : i + 1]
            mean = sum(window) / period
            var = sum((x - mean) ** 2 for x in window) / period
            std = var**0.5
            upper[i] = mean + num_std * std
            lower[i] = mean - num_std * std
        return upper, lower

    # ------------------------------------------------------------------
    # Promotion / persistence
    # ------------------------------------------------------------------

    def promote_strategy(self, idea: StrategyIdea, backtest_result: BacktestResult) -> None:
        """
        Save a successful strategy idea and its backtest result to the
        promoted strategies JSON file.

        Parameters
        ----------
        idea:
            The strategy idea that passed backtesting.
        backtest_result:
            The backtest metrics for this idea.
        """
        promoted = self._load_promoted()
        entry = {
            "idea": asdict(idea),
            "backtest": asdict(backtest_result),
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }
        promoted.append(entry)
        self._save_promoted(promoted)
        log.info("Promoted strategy '%s' — sharpe=%.2f, return=%.2f%%", idea.name, backtest_result.sharpe_ratio, backtest_result.total_return_pct)

    def get_promoted_strategies(self) -> List[StrategyIdea]:
        """
        Load and return all previously promoted strategy ideas.

        Returns
        -------
        list[StrategyIdea]
            Promoted ideas in chronological order.
        """
        promoted = self._load_promoted()
        ideas: List[StrategyIdea] = []
        for entry in promoted:
            idea_data = entry.get("idea", {})
            try:
                ideas.append(StrategyIdea(**idea_data))
            except Exception:
                log.warning("Skipping malformed promoted strategy entry: %s", idea_data.get("name", "?"))
        return ideas

    def _load_promoted(self) -> List[Dict[str, Any]]:
        """Load promoted strategies from JSON file."""
        if not self._promoted_path.exists():
            return []
        try:
            with open(self._promoted_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            log.warning("Failed to load promoted strategies from %s", self._promoted_path, exc_info=True)
            return []

    def _save_promoted(self, data: List[Dict[str, Any]]) -> None:
        """Persist promoted strategies to JSON file."""
        try:
            with open(self._promoted_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception:
            log.error("Failed to save promoted strategies to %s", self._promoted_path, exc_info=True)
