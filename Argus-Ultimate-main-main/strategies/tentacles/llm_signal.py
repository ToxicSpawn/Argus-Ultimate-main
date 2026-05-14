"""
llm_signal.py — OctoBot-inspired ChatGPT / LLM prediction overlay.

Sends a structured market context prompt to an OpenAI-compatible
endpoint and parses a directional signal [-1, 1] from the response.
Falls back to 0.0 (neutral) on API error or parse failure.

OctoBot reference
-----------------
OctoBot Cloud uses ChatGPT-assisted DCA signal overlays to augment
TA evaluators with natural-language market reasoning.

Usage
-----
    tentacle = LLMSignal(config={
        "api_key": "sk-...",
        "model": "gpt-4o",
        "symbol": "BTC/USDT",
    })
    result = await tentacle.async_evaluate(candles)

Prompt design
-------------
The prompt includes: symbol, current price, 24h change, RSI, EMA trend,
volume ratio, and Fear & Greed index. The model is asked to reply with
a JSON object: {"signal": <float -1 to 1>, "reasoning": <str>}.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional

import numpy as np

try:
    import aiohttp
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False

from .base_tentacle import (
    BaseTentacle, EvalResult, TentacleType, register_tentacle,
    candles_close, candles_volume, ema, rsi,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL       = "gpt-4o"
OPENAI_CHAT_URL     = "https://api.openai.com/v1/chat/completions"
DEFAULT_TIMEOUT_SEC = 20
CACHE_TTL_SEC       = 300   # re-query at most every 5 minutes

SYSTEM_PROMPT = (
    "You are a professional quantitative crypto trader. "
    "Analyse the provided market context and output ONLY valid JSON in this exact format: "
    '{"signal": <float between -1.0 and 1.0>, "reasoning": "<one sentence>"}. '
    "Signal 1.0 = strong buy, -1.0 = strong sell, 0.0 = neutral. "
    "Be concise and data-driven. Do not include any text outside the JSON object."
)


@register_tentacle
class LLMSignal(BaseTentacle):
    """
    LLM-based signal overlay tentacle.

    Builds a market context prompt from candle-derived features and
    queries an OpenAI-compatible API for a directional signal.

    Config keys
    -----------
    api_key          : str   OpenAI API key (fallback: OPENAI_API_KEY env var)
    api_base_url     : str   API base URL (default: OpenAI)
    model            : str   model name (default: gpt-4o)
    symbol           : str   trading pair label for the prompt
    fear_greed       : int   current Fear & Greed value (0-100); injected externally
    timeout_sec      : int   HTTP timeout (default: 20)
    cache_ttl_sec    : int   seconds to cache last response (default: 300)
    """

    name = "LLMSignal"
    tentacle_type = TentacleType.SOCIAL_EVALUATOR
    version = "1.0.0"
    weight = 0.6   # lower weight — LLM is advisory, not primary

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._api_key     = self.config.get("api_key") or os.getenv("OPENAI_API_KEY", "")
        self._api_url     = self.config.get("api_base_url", OPENAI_CHAT_URL)
        self._model       = self.config.get("model", DEFAULT_MODEL)
        self._symbol      = self.config.get("symbol", "BTC/USD")
        self._timeout     = int(self.config.get("timeout_sec", DEFAULT_TIMEOUT_SEC))
        self._cache_ttl   = int(self.config.get("cache_ttl_sec", CACHE_TTL_SEC))

        self._cached_result: Optional[EvalResult] = None
        self._cached_at: float = 0.0

    # ------------------------------------------------------------------
    # Sync evaluate (returns cached or neutral; use async_evaluate for live)
    # ------------------------------------------------------------------

    def evaluate(self, candles: np.ndarray, **kwargs: Any) -> EvalResult:
        """
        Synchronous wrapper. Returns cached result if within TTL,
        otherwise returns neutral (0.0). Call async_evaluate() for
        a live LLM query from an async context.
        """
        if self._cached_result and (time.time() - self._cached_at) < self._cache_ttl:
            return self._cached_result
        # Kick off async query in background without blocking
        context = self._build_context(candles, kwargs.get("fear_greed"))
        logger.info("[LLMSignal] Cache miss — returning neutral (use async_evaluate for live query)")
        return EvalResult(
            tentacle_name=self.name,
            signal=0.0,
            confidence=0.0,
            metadata={"context": context, "note": "use async_evaluate for live signal"},
        )

    async def async_evaluate(self, candles: np.ndarray, **kwargs: Any) -> EvalResult:
        """Async live LLM query with TTL cache."""
        if self._cached_result and (time.time() - self._cached_at) < self._cache_ttl:
            logger.debug("[LLMSignal] Returning cached result (age=%.0fs)",
                         time.time() - self._cached_at)
            return self._cached_result

        fear_greed = kwargs.get("fear_greed") or self.config.get("fear_greed")
        context    = self._build_context(candles, fear_greed)
        result     = await self._query_llm(context)
        self._cached_result = result
        self._cached_at     = time.time()
        return result

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_context(self, candles: np.ndarray, fear_greed: Optional[int]) -> str:
        close = candles_close(candles)
        vol   = candles_volume(candles)
        current_price = float(close[-1])

        change_24h = 0.0
        if len(close) >= 24:
            change_24h = (close[-1] - close[-24]) / close[-24] * 100

        rsi_val  = float(rsi(close, 14)[-1]) if len(close) >= 14 else 50.0
        ema_fast = float(ema(close, 10)[-1]) if len(close) >= 10 else current_price
        ema_slow = float(ema(close, 30)[-1]) if len(close) >= 30 else current_price
        ema_trend = "uptrend" if ema_fast > ema_slow else "downtrend"

        vol_ratio = 1.0
        if len(vol) >= 20:
            avg_vol = float(np.mean(vol[-20:-1]))
            if avg_vol > 0:
                vol_ratio = float(vol[-1]) / avg_vol

        fg_str = f"{fear_greed} (" + (
            "Extreme Fear" if fear_greed and fear_greed <= 20 else
            "Fear" if fear_greed and fear_greed <= 40 else
            "Neutral" if fear_greed and fear_greed <= 60 else
            "Greed" if fear_greed and fear_greed <= 80 else
            "Extreme Greed" if fear_greed else "N/A"
        ) + ")" if fear_greed is not None else "N/A"

        return (
            f"Symbol: {self._symbol}\n"
            f"Current Price: {current_price:.4f}\n"
            f"24h Change: {change_24h:+.2f}%\n"
            f"RSI(14): {rsi_val:.1f}\n"
            f"EMA Trend (10/30): {ema_trend}\n"
            f"Volume Ratio vs 20-bar avg: {vol_ratio:.2f}x\n"
            f"Fear & Greed Index: {fg_str}\n"
            f"Candles available: {len(candles)}\n"
        )

    # ------------------------------------------------------------------
    # LLM query
    # ------------------------------------------------------------------

    async def _query_llm(self, context: str) -> EvalResult:
        if not _AIOHTTP_AVAILABLE:
            logger.warning("[LLMSignal] aiohttp not installed — returning neutral")
            return self._neutral("aiohttp_missing")

        if not self._api_key:
            logger.warning("[LLMSignal] No API key configured — returning neutral")
            return self._neutral("no_api_key")

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": context},
            ],
            "temperature": 0.2,
            "max_tokens": 120,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        timeout = aiohttp.ClientTimeout(total=self._timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self._api_url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            content = data["choices"][0]["message"]["content"].strip()
            parsed  = json.loads(content)
            signal  = max(-1.0, min(1.0, float(parsed["signal"])))
            reasoning = parsed.get("reasoning", "")

            logger.info("[LLMSignal] signal=%.3f | %s", signal, reasoning)
            return EvalResult(
                tentacle_name=self.name,
                signal=signal,
                confidence=0.7,
                metadata={"reasoning": reasoning, "model": self._model},
            )

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("[LLMSignal] Parse error: %s", exc)
            return self._neutral(f"parse_error: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.error("[LLMSignal] API error: %s", exc)
            return self._neutral(f"api_error: {exc}")

    def _neutral(self, reason: str) -> EvalResult:
        return EvalResult(
            tentacle_name=self.name,
            signal=0.0,
            confidence=0.0,
            metadata={"reason": reason},
        )

    def reset(self) -> None:
        super().reset()
        self._cached_result = None
        self._cached_at = 0.0
