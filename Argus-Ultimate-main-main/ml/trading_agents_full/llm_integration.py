from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


@dataclass(slots=True)
class LLMTokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass(slots=True)
class LLMResponse:
    content: str
    reasoning: str
    score: float
    confidence: float
    provider: str
    model: str
    used_llm: bool
    latency_ms: float
    token_usage: LLMTokenUsage = field(default_factory=LLMTokenUsage)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProviderConfig:
    provider: str = "heuristic"
    model: str = "heuristic"
    base_url: str = ""
    api_key: str = ""
    timeout_seconds: float = 8.0
    max_tokens: int = 300
    temperature: float = 0.2
    enabled: bool = False


@dataclass(slots=True)
class TokenCostTracker:
    prompt_cost_per_1k: float = 0.0
    completion_cost_per_1k: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0

    def record(self, prompt_tokens: int, completion_tokens: int) -> LLMTokenUsage:
        self.total_prompt_tokens += max(int(prompt_tokens), 0)
        self.total_completion_tokens += max(int(completion_tokens), 0)
        prompt_cost = (max(int(prompt_tokens), 0) / 1000.0) * self.prompt_cost_per_1k
        completion_cost = (max(int(completion_tokens), 0) / 1000.0) * self.completion_cost_per_1k
        total_cost = prompt_cost + completion_cost
        self.total_cost_usd += total_cost
        return LLMTokenUsage(
            prompt_tokens=max(int(prompt_tokens), 0),
            completion_tokens=max(int(completion_tokens), 0),
            total_tokens=max(int(prompt_tokens), 0) + max(int(completion_tokens), 0),
            estimated_cost_usd=total_cost,
        )


class LLMProvider:
    PROMPT_TEMPLATES: Dict[str, str] = {
        "fundamental_analyst": "You are a fundamental analyst. Review earnings, growth, valuation, leverage, and cash flow. Return concise trading judgment.",
        "sentiment_analyst": "You are a sentiment analyst. Review news, social positioning, and fear-greed context. Return concise trading judgment.",
        "technical_analyst": "You are a technical analyst. Review indicators, trend, and breakout structure. Return concise trading judgment.",
        "news_analyst": "You are a news analyst. Review breaking events, regulatory developments, and likely market impact. Return concise trading judgment.",
        "bull_researcher": "You are a bullish researcher. Build the strongest upside case and name catalysts.",
        "bear_researcher": "You are a bearish researcher. Build the strongest downside case and name risks.",
        "risk_manager": "You are a portfolio risk manager. Review exposure, VaR, and drawdown. Return conservative trading judgment.",
    }

    def __init__(self, config: Optional[ProviderConfig] = None, cost_tracker: Optional[TokenCostTracker] = None) -> None:
        self.config = config or ProviderConfig(
            provider=os.getenv("ARGUS_TRADING_AGENTS_PROVIDER", "heuristic"),
            model=os.getenv("ARGUS_TRADING_AGENTS_MODEL", "heuristic"),
            base_url=os.getenv("ARGUS_TRADING_AGENTS_BASE_URL", ""),
            api_key=os.getenv("ARGUS_TRADING_AGENTS_API_KEY", ""),
            enabled=os.getenv("ARGUS_TRADING_AGENTS_ENABLED", "false").lower() in {"1", "true", "yes"},
        )
        self.cost_tracker = cost_tracker or TokenCostTracker()

    def build_prompt(self, role: str, context: Any, heuristic: Any) -> str:
        template = self.PROMPT_TEMPLATES.get(role, "You are a trading analyst. Return a directional judgment.")
        return (
            f"{template}\n"
            f"Symbol: {getattr(context, 'symbol', 'UNKNOWN')}\n"
            f"Regime: {getattr(context, 'regime', 'unknown')}\n"
            f"Heuristic action: {getattr(heuristic, 'action', 'hold')}\n"
            f"Heuristic score: {getattr(heuristic, 'score', 0.0):.2f}\n"
            f"Evidence: {json.dumps(getattr(heuristic, 'evidence', {}), sort_keys=True)}\n"
            f"Reasoning: {json.dumps(getattr(heuristic, 'reasoning', []), ensure_ascii=True)}\n"
            "Respond as JSON with keys: score [-1,1], confidence [0,1], reasoning."
        )

    def generate(self, prompt: str, *, role: str, context: Any, heuristic: Any) -> LLMResponse:
        start = time.perf_counter()
        if not self.config.enabled or self.config.provider == "heuristic":
            return self._heuristic_response("provider_disabled", start, heuristic)
        try:
            if self.config.provider == "openai":
                response = self._call_openai(prompt)
            elif self.config.provider == "anthropic":
                response = self._call_anthropic(prompt)
            elif self.config.provider == "local":
                response = self._call_local_model(prompt)
            else:
                return self._heuristic_response("unsupported_provider", start, heuristic)
            response.latency_ms = (time.perf_counter() - start) * 1000.0
            return response
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLMProvider generate failed for %s: %s", role, exc)
            return self._heuristic_response(str(exc), start, heuristic)

    def _call_openai(self, prompt: str) -> LLMResponse:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        request = urllib.request.Request(
            url=(self.config.base_url or "https://api.openai.com/v1") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        return self._http_json_to_response(request, provider="openai")

    def _call_anthropic(self, prompt: str) -> LLMResponse:
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        request = urllib.request.Request(
            url=(self.config.base_url or "https://api.anthropic.com/v1") + "/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        return self._http_json_to_response(request, provider="anthropic")

    def _call_local_model(self, prompt: str) -> LLMResponse:
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.config.temperature, "num_predict": self.config.max_tokens},
        }
        request = urllib.request.Request(
            url=(self.config.base_url or "http://localhost:11434/api") + "/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._http_json_to_response(request, provider="local")

    def _http_json_to_response(self, request: urllib.request.Request, provider: str) -> LLMResponse:
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as handle:  # noqa: S310
                payload = json.loads(handle.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{provider}_unreachable") from exc
        if provider == "openai":
            content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = payload.get("usage", {})
        elif provider == "anthropic":
            content_chunks = payload.get("content", [])
            content = "".join(str(item.get("text", "")) for item in content_chunks if isinstance(item, dict))
            usage = payload.get("usage", {})
        else:
            content = str(payload.get("response", ""))
            usage = payload.get("usage", {})
        parsed = self._parse_response(content, provider=provider)
        prompt_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", len(content) // 4 or 1)))
        completion_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", len(content) // 5 or 1)))
        parsed.token_usage = self.cost_tracker.record(prompt_tokens, completion_tokens)
        parsed.provider = provider
        parsed.model = self.config.model
        return parsed

    def _parse_response(self, raw: str, provider: Optional[str] = None) -> LLMResponse:
        raw = str(raw or "").strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {
                "score": 0.0,
                "confidence": 0.35 if raw else 0.0,
                "reasoning": raw or "No LLM response.",
            }
        score = _clamp(float(payload.get("score", 0.0)), -1.0, 1.0)
        confidence = _clamp(float(payload.get("confidence", 0.0)), 0.0, 1.0)
        reasoning = str(payload.get("reasoning", raw) or raw or "No LLM response.")
        return LLMResponse(
            content=raw,
            reasoning=reasoning,
            score=score,
            confidence=confidence,
            provider=provider or self.config.provider,
            model=self.config.model,
            used_llm=True,
            latency_ms=0.0,
        )

    def _heuristic_response(self, fallback_reason: str, start: float, heuristic: Any) -> LLMResponse:
        score = float(getattr(heuristic, "score", 0.0))
        confidence = max(float(getattr(heuristic, "confidence", 0.0)) - 0.05, 0.0)
        return LLMResponse(
            content="",
            reasoning="Heuristic fallback used because LLM was unavailable.",
            score=score,
            confidence=confidence,
            provider="heuristic",
            model="heuristic",
            used_llm=False,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            metadata={"fallback_reason": fallback_reason},
        )
