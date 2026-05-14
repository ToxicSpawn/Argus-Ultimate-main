"""
LLM Signal — generates trading signals using a local/remote LLM.

Queries an LLM (Ollama local or OpenAI) with market context and asks for
a directional signal. The LLM provides qualitative reasoning that complements
quantitative signals.

Context provided to LLM:
  - Current regime
  - Recent price action (10-day summary)
  - Key support/resistance levels
  - Funding rate
  - News headlines (optional)

LLM output parsed for: BULLISH/BEARISH/NEUTRAL + confidence + reasoning.
Falls back to NEUTRAL if LLM unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIRECTION_BULLISH = "BULLISH"
DIRECTION_BEARISH = "BEARISH"
DIRECTION_NEUTRAL = "NEUTRAL"

_VALID_DIRECTIONS = {DIRECTION_BULLISH, DIRECTION_BEARISH, DIRECTION_NEUTRAL}

PROVIDER_OLLAMA = "ollama"
PROVIDER_OPENAI = "openai"

_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_OPENAI_URL = "https://api.openai.com/v1"

# Confidence keyword map — first match wins
_CONFIDENCE_KEYWORDS: List[tuple] = [
    ("very high confidence", 0.9),
    ("high confidence", 0.8),
    ("confident", 0.75),
    ("moderate confidence", 0.6),
    ("moderate", 0.55),
    ("low confidence", 0.35),
    ("uncertain", 0.25),
    ("very uncertain", 0.15),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LLMSignal:
    """Parsed output from the LLM signal generator."""

    direction: str  # BULLISH | BEARISH | NEUTRAL
    confidence: float  # [0, 1]
    reasoning: str
    model_used: str
    latency_ms: float
    timestamp: float = field(default_factory=time.time)

    @property
    def as_numeric(self) -> float:
        """Convert direction to numeric signal value ∈ {-1, 0, 1}."""
        if self.direction == DIRECTION_BULLISH:
            return 1.0
        elif self.direction == DIRECTION_BEARISH:
            return -1.0
        return 0.0


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class LLMSignalGenerator:
    """
    Generates trading directional signals by querying an LLM.

    Parameters
    ----------
    provider : str
        "ollama" for local Ollama server, "openai" for OpenAI API.
    model : str
        Model identifier (e.g. "mistral", "gpt-4o-mini").
    base_url : str | None
        Override base URL. Defaults to provider-specific default.
    timeout : float
        HTTP request timeout in seconds (applied at socket level).
    max_tokens : int
        Maximum tokens for LLM response.
    inference_timeout : float
        Overall async timeout in seconds for the entire generate_signal()
        call.  For large models (70B+) inference can take 5-60 s; if
        Ollama does not return within this window the call is cancelled
        and a neutral signal (0.0) is returned immediately.  Defaults to
        10 s.  Set to 0 to disable.
    """

    def __init__(
        self,
        provider: str = PROVIDER_OLLAMA,
        model: str = "mistral",
        base_url: Optional[str] = None,
        timeout: float = 10.0,
        max_tokens: int = 200,
        inference_timeout: float = 10.0,
    ) -> None:
        self.provider = provider
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.inference_timeout = inference_timeout

        # Instance-level caches: symbol -> (timestamp, result_dict)
        self._analysis_cache: Dict[str, tuple] = {}
        self._sentiment_cache: Dict[str, tuple] = {}

        if base_url:
            self.base_url = base_url.rstrip("/")
        elif provider == PROVIDER_OPENAI:
            self.base_url = _DEFAULT_OPENAI_URL
        else:
            self.base_url = _DEFAULT_OLLAMA_URL

    _CACHE_TTL = 300.0  # 5 minutes

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_market_analysis(
        self,
        symbol: str,
        ohlcv_data: Sequence[float],
        regime: str = "UNKNOWN",
        indicators: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a structured market analysis using the LLM.

        Synchronous wrapper with caching. Won't call LLM more than once per
        5 minutes per symbol.

        Parameters
        ----------
        symbol : str
            Trading pair (e.g. "BTC/USD").
        ohlcv_data : Sequence[float]
            Recent close prices (newest last).
        regime : str
            Current market regime label.
        indicators : dict | None
            Optional dict of indicator values {rsi, macd, macd_signal, etc.}.

        Returns
        -------
        dict
            {direction, confidence, reasoning, model_used}
        """
        now = time.time()
        cached = self._analysis_cache.get(symbol)
        if cached is not None and (now - cached[0]) < self._CACHE_TTL:
            return cached[1]

        prompt = self._build_analysis_prompt(symbol, ohlcv_data, regime, indicators or {})

        try:
            if self.provider == PROVIDER_OPENAI:
                raw_text = self._query_openai(prompt)
            else:
                raw_text = self._query_ollama(prompt)
        except Exception as exc:
            logger.warning("LLMSignalGenerator: market analysis query error — %s", exc)
            raw_text = ""

        parsed = self._parse_response(raw_text)
        result: Dict[str, Any] = {
            "direction": parsed.direction,
            "confidence": parsed.confidence,
            "reasoning": parsed.reasoning,
            "model_used": f"{self.provider}/{self.model}",
        }
        self._analysis_cache[symbol] = (now, result)
        return result

    def generate_news_sentiment(
        self,
        headlines: List[str],
    ) -> Dict[str, Any]:
        """
        Classify overall crypto sentiment from recent headlines.

        Synchronous with caching (cache key = joined headline fingerprint).

        Parameters
        ----------
        headlines : list[str]
            Recent crypto news headlines.

        Returns
        -------
        dict
            {sentiment: BULLISH|BEARISH|NEUTRAL, confidence: float, key_factors: list}
        """
        if not headlines:
            return {
                "sentiment": DIRECTION_NEUTRAL,
                "confidence": 0.0,
                "key_factors": [],
            }

        cache_key = "|".join(sorted(headlines[:5]))
        now = time.time()
        cached = self._sentiment_cache.get(cache_key)
        if cached is not None and (now - cached[0]) < self._CACHE_TTL:
            return cached[1]

        prompt = self._build_sentiment_prompt(headlines)

        try:
            if self.provider == PROVIDER_OPENAI:
                raw_text = self._query_openai(prompt)
            else:
                raw_text = self._query_ollama(prompt)
        except Exception as exc:
            logger.warning("LLMSignalGenerator: news sentiment query error — %s", exc)
            raw_text = ""

        parsed = self._parse_response(raw_text)

        # Extract key factors from reasoning
        key_factors: List[str] = []
        if parsed.reasoning and parsed.reasoning != "No response from LLM.":
            # Split reasoning on sentence boundaries or commas
            parts = parsed.reasoning.replace(". ", "|").replace(", ", "|").split("|")
            key_factors = [p.strip() for p in parts if len(p.strip()) > 5][:3]

        result: Dict[str, Any] = {
            "sentiment": parsed.direction,
            "confidence": parsed.confidence,
            "key_factors": key_factors,
        }
        self._sentiment_cache[cache_key] = (now, result)
        return result

    async def generate_signal(
        self,
        symbol: str,
        regime: str,
        price_data: Sequence[float],
        funding_rate: float = 0.0,
        headlines: Optional[List[str]] = None,
    ) -> LLMSignal:
        """
        Generate a trading signal for the given symbol and market context.

        Parameters
        ----------
        symbol : str
            Trading pair (e.g. "BTC/USD").
        regime : str
            Current market regime label from regime detector.
        price_data : Sequence[float]
            Recent close prices (up to 10 most recent used).
        funding_rate : float
            Current perpetual funding rate.
        headlines : list[str] | None
            Optional news headlines to include in context.

        Returns
        -------
        LLMSignal
            Always returns a signal; falls back to NEUTRAL on any error.
        """
        t_start = time.monotonic()
        prompt = self._build_prompt(symbol, regime, price_data, funding_rate, headlines)

        try:
            loop = asyncio.get_running_loop()
            if self.provider == PROVIDER_OPENAI:
                coro = loop.run_in_executor(None, self._query_openai, prompt)
            else:
                coro = loop.run_in_executor(None, self._query_ollama, prompt)

            if self.inference_timeout > 0:
                raw_text = await asyncio.wait_for(coro, timeout=self.inference_timeout)
            else:
                raw_text = await coro
        except asyncio.TimeoutError:
            logger.warning(
                "LLMSignalGenerator: inference timeout (%.1fs) querying %s — returning neutral.",
                self.inference_timeout,
                self.provider,
            )
            raw_text = ""
        except Exception as exc:
            logger.warning("LLMSignalGenerator: query error — %s.", exc)
            raw_text = ""

        latency_ms = (time.monotonic() - t_start) * 1000.0
        signal = self._parse_response(raw_text)
        signal.latency_ms = latency_ms
        signal.model_used = f"{self.provider}/{self.model}"
        return signal

    async def generate_signal_with_reflection(
        self,
        symbol: str,
        regime: str,
        price_data: Sequence[float],
        funding_rate: float = 0.0,
        headlines: Optional[List[str]] = None,
        *,
        trade_rag: Optional[Any] = None,
        decision_journal: Optional[Any] = None,
        llm_client: Optional[Any] = None,
    ) -> "LLMSignal":
        """
        Phase W6 — generate a signal and run a 1-step reflection cycle.

        Falls back gracefully to the original signal if the reflection
        module is unavailable or the LLM client cannot be resolved.

        Parameters
        ----------
        symbol, regime, price_data, funding_rate, headlines :
            Same as ``generate_signal``.
        trade_rag : Any, optional
            RAG system for similar-setup retrieval.
        decision_journal : Any, optional
            Decision journal for recent accuracy lookups.
        llm_client : Any, optional
            Explicit LLM client for the reflection call. If ``None``, a
            default singleton is pulled from ``core/llm/client.py``.
        """
        # Step 1: generate the base signal
        signal = await self.generate_signal(
            symbol=symbol,
            regime=regime,
            price_data=price_data,
            funding_rate=funding_rate,
            headlines=headlines,
        )

        # Step 2: run reflection (best-effort)
        try:
            from core.reasoning import ReflectionLoop
            if llm_client is None:
                try:
                    from core.llm import get_llm_client
                    llm_client = get_llm_client()
                except Exception:
                    return signal

            loop = ReflectionLoop(
                llm_client=llm_client,
                trade_rag=trade_rag,
                decision_journal=decision_journal,
            )
            context = {
                "symbol": symbol,
                "regime": regime,
                "price": float(price_data[-1]) if price_data else 0.0,
                "volatility": 0.0,  # caller can override via headlines metadata
            }
            result = loop.reflect(
                signal_direction=signal.direction,
                signal_confidence=signal.confidence,
                signal_reasoning=signal.reasoning,
                context=context,
            )

            if result.action == "reject":
                signal.direction = DIRECTION_NEUTRAL
                signal.confidence = 0.0
                signal.reasoning = f"[Reflection REJECTED] {result.critique}"
            elif result.action == "revise" and result.changed:
                signal.direction = result.revised_direction
                signal.confidence = result.revised_confidence
                signal.reasoning = f"[Reflection REVISED] {result.critique}"
        except Exception as exc:
            logger.debug("LLM reflection skipped: %s", exc)

        return signal

    def is_available(self) -> bool:
        """
        Check whether the configured LLM provider is reachable.

        Returns
        -------
        bool
            True if a simple connectivity check succeeds.
        """
        try:
            if self.provider == PROVIDER_OLLAMA:
                url = f"{self.base_url}/api/tags"
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=3.0):
                    pass
                return True
            elif self.provider == PROVIDER_OPENAI:
                api_key = os.environ.get("OPENAI_API_KEY", "")
                return bool(api_key)
            return False
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        symbol: str,
        regime: str,
        price_data: Sequence[float],
        funding_rate: float,
        headlines: Optional[List[str]],
    ) -> str:
        prices = list(price_data)[-10:]  # Use at most 10 most recent
        if len(prices) >= 2:
            price_change_pct = (prices[-1] / prices[-2] - 1.0) * 100.0
            price_range_pct = (max(prices) / min(prices) - 1.0) * 100.0 if min(prices) > 0 else 0.0
            current_price = prices[-1]
            high = max(prices)
            low = min(prices)
        else:
            current_price = prices[0] if prices else 0.0
            price_change_pct = 0.0
            price_range_pct = 0.0
            high = current_price
            low = current_price

        lines = [
            f"You are a quantitative crypto trading analyst. Provide a brief directional signal for {symbol}.",
            "",
            "## Market Context",
            f"- Symbol: {symbol}",
            f"- Current Regime: {regime}",
            f"- Current Price: {current_price:.4f}",
            f"- 10-Period High: {high:.4f}",
            f"- 10-Period Low: {low:.4f}",
            f"- Last Period Change: {price_change_pct:+.2f}%",
            f"- 10-Period Range: {price_range_pct:.2f}%",
            f"- Funding Rate: {funding_rate*100:.4f}%",
        ]

        if headlines:
            lines.append("")
            lines.append("## Recent Headlines")
            for h in headlines[:5]:
                lines.append(f"- {h}")

        lines += [
            "",
            "## Instructions",
            "Based on the above context, provide a directional signal.",
            "Your response MUST include exactly one of: BULLISH, BEARISH, or NEUTRAL",
            "followed by your confidence level (high confidence / moderate confidence / low confidence / uncertain)",
            "and a brief one-sentence reasoning.",
            "",
            "Example format:",
            "Signal: BULLISH",
            "Confidence: moderate confidence",
            "Reasoning: Price is trending above key support with positive funding momentum.",
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Enhanced prompt construction
    # ------------------------------------------------------------------

    def _build_analysis_prompt(
        self,
        symbol: str,
        ohlcv_data: Sequence[float],
        regime: str,
        indicators: Dict[str, float],
    ) -> str:
        """Construct a detailed market analysis prompt with indicators."""
        prices = list(ohlcv_data)[-20:]
        if len(prices) >= 2:
            current_price = prices[-1]
            price_change_pct = (prices[-1] / prices[-2] - 1.0) * 100.0
            price_range_pct = (max(prices) / min(prices) - 1.0) * 100.0 if min(prices) > 0 else 0.0
            high = max(prices)
            low = min(prices)
        else:
            current_price = prices[0] if prices else 0.0
            price_change_pct = 0.0
            price_range_pct = 0.0
            high = current_price
            low = current_price

        lines = [
            f"You are a quantitative crypto trading analyst. Provide a detailed directional assessment for {symbol}.",
            "",
            "## Market Context",
            f"- Symbol: {symbol}",
            f"- Current Regime: {regime}",
            f"- Current Price: {current_price:.4f}",
            f"- 20-Period High: {high:.4f}",
            f"- 20-Period Low: {low:.4f}",
            f"- Last Period Change: {price_change_pct:+.2f}%",
            f"- Period Range: {price_range_pct:.2f}%",
        ]

        # Add indicators if provided
        if indicators:
            lines.append("")
            lines.append("## Technical Indicators")
            for name, value in indicators.items():
                lines.append(f"- {name.upper()}: {value:.4f}")

        lines += [
            "",
            "## Instructions",
            "Assess the market for this asset and provide:",
            "1. Direction: BULLISH, BEARISH, or NEUTRAL",
            "2. Confidence: a phrase like 'high confidence', 'moderate confidence', 'low confidence', or 'uncertain'",
            "3. Reasoning: a brief 1-2 sentence explanation of your analysis",
            "",
            "Format your response exactly as:",
            "Signal: BULLISH",
            "Confidence: moderate confidence",
            "Reasoning: Price is consolidating near support with improving momentum indicators.",
        ]
        return "\n".join(lines)

    def _build_sentiment_prompt(self, headlines: List[str]) -> str:
        """Construct a news sentiment classification prompt."""
        lines = [
            "You are a crypto market sentiment analyst. Classify the overall sentiment "
            "from these recent headlines.",
            "",
            "## Headlines",
        ]
        for h in headlines[:10]:
            lines.append(f"- {h}")

        lines += [
            "",
            "## Instructions",
            "Based on these headlines, determine the overall crypto market sentiment.",
            "Your response MUST include exactly one of: BULLISH, BEARISH, or NEUTRAL",
            "followed by your confidence level and key factors driving your assessment.",
            "",
            "Format:",
            "Signal: BULLISH",
            "Confidence: high confidence",
            "Reasoning: Multiple headlines indicate institutional adoption and price support.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, text: str) -> LLMSignal:
        """
        Extract direction, confidence, and reasoning from raw LLM text.
        Always returns a valid LLMSignal; defaults to NEUTRAL on parse failure.
        """
        if not text or not text.strip():
            return LLMSignal(
                direction=DIRECTION_NEUTRAL,
                confidence=0.0,
                reasoning="No response from LLM.",
                model_used=f"{self.provider}/{self.model}",
                latency_ms=0.0,
            )

        text_upper = text.upper()

        # Direction detection — first keyword found wins
        direction = DIRECTION_NEUTRAL
        for candidate in (DIRECTION_BULLISH, DIRECTION_BEARISH, DIRECTION_NEUTRAL):
            if candidate in text_upper:
                direction = candidate
                break

        # Confidence detection
        text_lower = text.lower()
        confidence = 0.5  # Default moderate
        for keyword, conf_val in _CONFIDENCE_KEYWORDS:
            if keyword in text_lower:
                confidence = conf_val
                break

        # Reasoning: attempt to extract sentence after "Reasoning:" label
        reasoning = ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("reasoning:"):
                reasoning = stripped[len("reasoning:"):].strip()
                break
        if not reasoning:
            # Fallback: use first non-empty line that contains direction keyword
            for line in text.splitlines():
                if line.strip():
                    reasoning = line.strip()[:200]
                    break

        return LLMSignal(
            direction=direction,
            confidence=confidence,
            reasoning=reasoning or "No reasoning provided.",
            model_used=f"{self.provider}/{self.model}",
            latency_ms=0.0,
        )

    # ------------------------------------------------------------------
    # HTTP query methods (synchronous — called via executor)
    # ------------------------------------------------------------------

    def _query_ollama(self, prompt: str) -> str:
        """POST to Ollama's /api/generate endpoint."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": 0.3,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Use inference_timeout (not socket timeout) — Ollama cold starts
        # can take 10-30s to load the model into VRAM before responding.
        _socket_timeout = max(self.timeout, self.inference_timeout, 60.0)
        with urllib.request.urlopen(req, timeout=_socket_timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return str(body.get("response", ""))

    def _query_openai(self, prompt: str) -> str:
        """POST to OpenAI's chat completions endpoint."""
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("LLMSignalGenerator: OPENAI_API_KEY not set.")
            return ""

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a quantitative crypto trading analyst. "
                        "Be concise and always include BULLISH, BEARISH, or NEUTRAL in your response."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.3,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return str(body["choices"][0]["message"]["content"])
