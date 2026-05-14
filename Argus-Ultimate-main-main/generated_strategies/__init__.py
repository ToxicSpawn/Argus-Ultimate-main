"""
Generated Strategies — sandboxed directory for ARGUS auto-generated trading code.

This package holds Python files written by the code_evolution_engine.
NEVER edit files here manually — they are auto-managed.

Subdirectories:
  candidates/     — newly generated, awaiting review
  active/         — passed review, currently in use
  graveyard/      — failed or retired generations (kept for analysis)

Each generated file follows a strict template:
  - Single class inheriting from BaseGeneratedStrategy
  - No filesystem or network access
  - Whitelisted imports only
  - Deterministic (no time.time() or random without seed)
  - Validated by code_review_gate before execution
"""
from __future__ import annotations

__all__ = ["BaseGeneratedStrategy"]


class BaseGeneratedStrategy:
    """
    Base class all generated strategies must inherit from.
    Provides the interface that ARGUS expects.
    """

    name: str = "unnamed"
    version: int = 1
    generation_id: str = ""
    target_regime: str = "ANY"
    description: str = ""

    def evaluate(self, market_state: dict) -> dict:
        """
        Evaluate the strategy on current market state.

        Args:
            market_state: dict with keys like price, volume, rsi, regime, etc.

        Returns:
            dict with keys:
                action: "BUY" / "SELL" / "HOLD"
                confidence: 0.0 to 1.0
                reasoning: human-readable explanation
        """
        return {"action": "HOLD", "confidence": 0.0, "reasoning": "base class"}
