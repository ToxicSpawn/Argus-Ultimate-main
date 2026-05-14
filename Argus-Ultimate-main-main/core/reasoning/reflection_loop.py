"""
Reflection loop: one-step LLM self-critique for trading signals.

Pipeline
--------
1. Accept an initial signal (direction, confidence, reasoning).
2. Gather evidence:
   - Top-k similar historical setups (from RAG)
   - Recent decision-journal accuracy for this symbol/regime
3. Ask the LLM: "Given this evidence, is the signal correct? If not, what's
   the revised signal?"
4. Parse the response and return either the original, a revised version,
   or a rejection.

Feature-flagged behind ``use_llm_reflection`` in config — OFF by default.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Data structures
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class ReflectionResult:
    """Output of one reflection cycle."""

    original_direction: str  # BULLISH | BEARISH | NEUTRAL
    original_confidence: float
    revised_direction: str
    revised_confidence: float
    action: str  # "confirm" | "revise" | "reject"
    critique: str  # LLM's reasoning for its decision
    evidence_used: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    @property
    def changed(self) -> bool:
        return (
            self.revised_direction != self.original_direction
            or abs(self.revised_confidence - self.original_confidence) > 0.05
        )


# ═════════════════════════════════════════════════════════════════════════════
# ReflectionLoop
# ═════════════════════════════════════════════════════════════════════════════


class ReflectionLoop:
    """
    One-step LLM self-critique over a proposed trading signal.

    Parameters
    ----------
    llm_client : Any
        Instance of ``core/llm/client.py::LLMClient`` (or anything exposing
        ``complete(prompt, system, json_mode) → LLMResponse``).
    trade_rag : Any, optional
        Instance of ``core/rag/trade_rag.py::TradeRAG`` for historical
        similar-setup retrieval. If ``None``, reflection uses only
        decision journal evidence.
    decision_journal : Any, optional
        Decision journal instance for recent accuracy lookups.
    """

    _SYSTEM_PROMPT = (
        "You are an expert trading analyst reviewing a proposed signal. "
        "Given the evidence, respond with JSON: "
        '{"action": "confirm"|"revise"|"reject", '
        '"direction": "BULLISH"|"BEARISH"|"NEUTRAL", '
        '"confidence": 0.0-1.0, '
        '"critique": "1-2 sentence explanation"}'
    )

    def __init__(
        self,
        llm_client: Any,
        trade_rag: Optional[Any] = None,
        decision_journal: Optional[Any] = None,
    ) -> None:
        if llm_client is None or not hasattr(llm_client, "complete"):
            raise ValueError("llm_client must implement complete()")
        self.llm = llm_client
        self.rag = trade_rag
        self.journal = decision_journal

    # ── Main entry ───────────────────────────────────────────────────────────

    def reflect(
        self,
        signal_direction: str,
        signal_confidence: float,
        signal_reasoning: str,
        context: Dict[str, Any],
    ) -> ReflectionResult:
        """
        Run one reflection cycle on a signal.

        Parameters
        ----------
        signal_direction : str
            Original direction from signal generator.
        signal_confidence : float
            Original confidence in [0, 1].
        signal_reasoning : str
            Original reasoning from signal generator.
        context : Dict[str, Any]
            Must contain: ``symbol``, ``regime``, ``price``, ``volatility``.
            Optional: ``cycle_context``, ``recent_pnl``, ``recent_win_rate``.

        Returns
        -------
        ReflectionResult
        """
        import time

        t0 = time.perf_counter()

        evidence = self._gather_evidence(context)
        prompt = self._build_critique_prompt(
            signal_direction=signal_direction,
            signal_confidence=signal_confidence,
            signal_reasoning=signal_reasoning,
            context=context,
            evidence=evidence,
        )

        try:
            response = self.llm.complete(
                prompt=prompt,
                system=self._SYSTEM_PROMPT,
                json_mode=True,
                max_tokens=300,
            )
        except Exception as exc:
            logger.warning("ReflectionLoop: LLM complete failed — %s", exc)
            return ReflectionResult(
                original_direction=signal_direction,
                original_confidence=signal_confidence,
                revised_direction=signal_direction,
                revised_confidence=signal_confidence,
                action="confirm",
                critique=f"Reflection skipped: {exc}",
                evidence_used=evidence,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        parsed = self._parse_response(response, signal_direction, signal_confidence)
        parsed.evidence_used = evidence
        parsed.latency_ms = (time.perf_counter() - t0) * 1000.0
        return parsed

    # ── Evidence gathering ───────────────────────────────────────────────────

    def _gather_evidence(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect evidence from RAG + decision journal."""
        evidence: Dict[str, Any] = {
            "rag_similar_setups": None,
            "recent_accuracy": None,
        }

        # RAG: similar setups
        if self.rag is not None:
            try:
                stats = self.rag.win_rate_of_similar(
                    context, k=5, min_similarity=0.3,
                )
                evidence["rag_similar_setups"] = {
                    "n_similar": int(stats.get("n_similar", 0)),
                    "win_rate": float(stats.get("win_rate", 0.5)),
                    "avg_pnl": float(stats.get("avg_pnl", 0.0)),
                    "confidence": float(stats.get("confidence", 0.0)),
                }
            except Exception as exc:
                logger.debug("RAG retrieval failed: %s", exc)

        # Decision journal recent accuracy
        if self.journal is not None:
            try:
                symbol = str(context.get("symbol", ""))
                regime = str(context.get("regime", ""))
                if hasattr(self.journal, "recent_accuracy"):
                    acc = self.journal.recent_accuracy(
                        symbol=symbol, regime=regime, window=20,
                    )
                    evidence["recent_accuracy"] = acc
            except Exception as exc:
                logger.debug("Decision journal lookup failed: %s", exc)

        return evidence

    # ── Prompt construction ──────────────────────────────────────────────────

    def _build_critique_prompt(
        self,
        signal_direction: str,
        signal_confidence: float,
        signal_reasoning: str,
        context: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> str:
        lines = [
            "=== PROPOSED SIGNAL ===",
            f"Direction: {signal_direction}",
            f"Confidence: {signal_confidence:.2f}",
            f"Reasoning: {signal_reasoning}",
            "",
            "=== CURRENT CONTEXT ===",
            f"Symbol: {context.get('symbol', 'unknown')}",
            f"Regime: {context.get('regime', 'unknown')}",
            f"Price: {context.get('price', 0.0)}",
            f"Volatility: {context.get('volatility', 0.0)}",
        ]

        # RAG evidence
        rag = evidence.get("rag_similar_setups")
        if rag and rag["n_similar"] > 0:
            lines += [
                "",
                "=== HISTORICAL SIMILAR SETUPS ===",
                f"Found {rag['n_similar']} similar past trades",
                f"Historical win rate: {rag['win_rate']:.2%}",
                f"Average P&L: {rag['avg_pnl']:.4f}",
            ]

        # Decision journal
        acc = evidence.get("recent_accuracy")
        if acc is not None:
            lines += [
                "",
                "=== RECENT ACCURACY ===",
                f"Last 20 signals accuracy: {acc:.2%}",
            ]

        lines += [
            "",
            "=== CRITIQUE TASK ===",
            "Given the evidence above, critique the proposed signal.",
            "If the evidence contradicts the signal (e.g. similar setups lost money),",
            "revise the direction or lower the confidence. If no strong contradiction,",
            "confirm the original. If the evidence strongly opposes the signal, reject it.",
            "Respond with the JSON format specified in the system prompt.",
        ]
        return "\n".join(lines)

    # ── Response parsing ─────────────────────────────────────────────────────

    def _parse_response(
        self,
        response: Any,
        original_direction: str,
        original_confidence: float,
    ) -> ReflectionResult:
        """Parse LLM response (dict or text) into a ReflectionResult."""
        action = "confirm"
        direction = original_direction
        confidence = original_confidence
        critique = ""

        parsed = getattr(response, "parsed", None)
        text = getattr(response, "text", "") or ""

        if isinstance(parsed, dict):
            action = str(parsed.get("action", "confirm")).lower()
            direction = str(parsed.get("direction", original_direction)).upper()
            try:
                confidence = float(parsed.get("confidence", original_confidence))
            except (TypeError, ValueError):
                confidence = original_confidence
            critique = str(parsed.get("critique", ""))
        elif text:
            # Best-effort regex fallback
            m = re.search(r'"action"\s*:\s*"(\w+)"', text)
            if m:
                action = m.group(1).lower()
            m = re.search(r'"direction"\s*:\s*"(\w+)"', text)
            if m:
                direction = m.group(1).upper()
            m = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
            if m:
                try:
                    confidence = float(m.group(1))
                except ValueError:
                    pass
            m = re.search(r'"critique"\s*:\s*"([^"]+)"', text)
            if m:
                critique = m.group(1)

        # Normalize
        if action not in ("confirm", "revise", "reject"):
            action = "confirm"
        if direction not in ("BULLISH", "BEARISH", "NEUTRAL"):
            direction = original_direction
        confidence = max(0.0, min(1.0, confidence))

        # If rejected, force NEUTRAL with confidence 0
        if action == "reject":
            direction = "NEUTRAL"
            confidence = 0.0

        return ReflectionResult(
            original_direction=original_direction,
            original_confidence=original_confidence,
            revised_direction=direction,
            revised_confidence=confidence,
            action=action,
            critique=critique or "No critique provided",
        )


# ═════════════════════════════════════════════════════════════════════════════
# Stateless convenience
# ═════════════════════════════════════════════════════════════════════════════


def reflect_signal(
    llm_client: Any,
    signal_direction: str,
    signal_confidence: float,
    signal_reasoning: str,
    context: Dict[str, Any],
    *,
    trade_rag: Optional[Any] = None,
    decision_journal: Optional[Any] = None,
) -> ReflectionResult:
    """One-shot reflection helper."""
    loop = ReflectionLoop(
        llm_client=llm_client,
        trade_rag=trade_rag,
        decision_journal=decision_journal,
    )
    return loop.reflect(
        signal_direction=signal_direction,
        signal_confidence=signal_confidence,
        signal_reasoning=signal_reasoning,
        context=context,
    )
