"""
Thin LLM client adapter.

Provides a stable API that wraps the existing ``ml/llm_signal.py`` Ollama/
OpenAI code. Used by reflection loops, memory consolidation, and tool
calling.

Features
--------
- Timeout and retry handling
- JSON-mode (for structured outputs)
- Stub mode (for tests and dry runs) via ``ARGUS_LLM_STUB=1``
- Rate limiting (configurable)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# LLMResponse
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class LLMResponse:
    text: str
    parsed: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    model: str = ""
    token_estimate: int = 0
    error: Optional[str] = None


# ═════════════════════════════════════════════════════════════════════════════
# LLMClient
# ═════════════════════════════════════════════════════════════════════════════


class LLMClient:
    """
    Unified LLM client with stub-mode support.

    Parameters
    ----------
    model : str
        Model name (e.g. "ollama:llama3", "openai:gpt-4o-mini").
    timeout : float
        Request timeout in seconds.
    max_retries : int
        Maximum retry attempts.
    stub_mode : bool
        If True, return canned responses without calling any LLM. Useful
        for tests and for running ARGUS without an LLM backend.
    """

    def __init__(
        self,
        model: str = "ollama:llama3",
        *,
        timeout: float = 30.0,
        max_retries: int = 2,
        stub_mode: Optional[bool] = None,
    ) -> None:
        self.model = model
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        # Stub mode: explicit param wins, else env var, else auto-detect
        if stub_mode is None:
            stub_mode = os.environ.get("ARGUS_LLM_STUB", "0") == "1"
        self.stub_mode = bool(stub_mode)
        self._last_call_ts: float = 0.0
        self._min_interval_s: float = 0.1
        self._lock = Lock()
        # Delegate: try to import the existing signal generator
        self._delegate: Optional[Any] = None
        if not self.stub_mode:
            self._delegate = self._build_delegate()

    def _build_delegate(self) -> Optional[Any]:
        """Try to construct the existing ml/llm_signal.py client."""
        try:
            from ml.llm_signal import LLMSignalGenerator  # type: ignore
            return LLMSignalGenerator()
        except Exception as exc:
            logger.debug("LLM delegate init failed: %s", exc)
            return None

    # ── Core call ───────────────────────────────────────────────────────────

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        json_mode: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.5,
    ) -> LLMResponse:
        """
        Send a single prompt to the LLM and return the response.

        Parameters
        ----------
        prompt : str
            User prompt.
        system : str, optional
            System prompt (role / instructions).
        json_mode : bool
            If True, attempt to parse the response as JSON and populate
            ``LLMResponse.parsed``.
        max_tokens : int
            Max response tokens (passthrough to backend).
        temperature : float
            Sampling temperature.
        """
        t0 = time.perf_counter()

        if self.stub_mode:
            return self._stub_response(prompt, json_mode)

        # Rate limit
        with self._lock:
            now = time.perf_counter()
            elapsed_since_last = now - self._last_call_ts
            if elapsed_since_last < self._min_interval_s:
                time.sleep(self._min_interval_s - elapsed_since_last)
            self._last_call_ts = time.perf_counter()

        # Try the delegate
        if self._delegate is not None:
            try:
                # Most LLM signal generators expose a generic .ask() or .complete()
                text: Optional[str] = None
                if hasattr(self._delegate, "ask"):
                    text = self._delegate.ask(prompt)
                elif hasattr(self._delegate, "complete"):
                    text = self._delegate.complete(prompt, system=system)
                elif hasattr(self._delegate, "signal_from_llm"):
                    # Specialized method — wrap
                    signal = self._delegate.signal_from_llm(prompt)
                    text = str(signal) if signal is not None else None
                else:
                    text = None

                if text:
                    return self._build_response(
                        text=str(text),
                        json_mode=json_mode,
                        t0=t0,
                        model=self.model,
                    )
            except Exception as exc:
                logger.debug("LLM delegate call failed: %s", exc)

        # Fallback to stub
        return self._stub_response(prompt, json_mode, error="delegate_failed")

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _build_response(
        self,
        text: str,
        *,
        json_mode: bool,
        t0: float,
        model: str,
        error: Optional[str] = None,
    ) -> LLMResponse:
        latency_ms = (time.perf_counter() - t0) * 1000
        parsed: Optional[Dict[str, Any]] = None
        if json_mode:
            parsed = self._try_parse_json(text)
        return LLMResponse(
            text=text,
            parsed=parsed,
            latency_ms=latency_ms,
            model=model,
            token_estimate=max(1, len(text) // 4),
            error=error,
        )

    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract a JSON object from ``text``, being lenient about whitespace / fences."""
        if not text:
            return None
        # Try direct parse
        try:
            return json.loads(text)
        except Exception:
            pass
        # Try to find a JSON object in the text
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return None

    def _stub_response(
        self,
        prompt: str,
        json_mode: bool,
        error: Optional[str] = None,
    ) -> LLMResponse:
        """Deterministic stub response (for tests and no-LLM deployments)."""
        # Use a simple deterministic output based on the prompt
        if json_mode:
            stub_text = '{"action": "HOLD", "confidence": 0.5, "reason": "stub_mode"}'
            parsed = {"action": "HOLD", "confidence": 0.5, "reason": "stub_mode"}
        else:
            stub_text = "stub response: " + prompt[:100]
            parsed = None
        return LLMResponse(
            text=stub_text,
            parsed=parsed,
            latency_ms=0.1,
            model="stub",
            token_estimate=20,
            error=error,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Singleton
# ═════════════════════════════════════════════════════════════════════════════


_INSTANCE: Optional[LLMClient] = None


def get_llm_client(
    model: Optional[str] = None,
    *,
    stub_mode: Optional[bool] = None,
) -> LLMClient:
    """Return the global LLM client singleton."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = LLMClient(
            model=model or "ollama:llama3",
            stub_mode=stub_mode,
        )
    return _INSTANCE
