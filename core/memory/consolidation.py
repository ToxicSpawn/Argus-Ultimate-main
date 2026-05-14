"""
Memory consolidator — episodic → semantic compression worker.

Runs periodically (default every 1000 cycles). Reads recent episodic
entries, groups them by event_type + symbol, and emits semantic facts
via either:
  1. A rule-based extractor (default) — pattern matches event type +
     regime + outcome.
  2. An LLM-based extractor (optional) — asks the LLM to summarize a
     batch of episodes as facts.

Writes to ``core/memory/semantic_memory.py`` (new DB). Does NOT touch
``cross_session_memory.sqlite``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .episodic_memory import EpisodicEntry, EpisodicMemory
from .semantic_memory import SemanticMemory

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# MemoryConsolidator
# ═════════════════════════════════════════════════════════════════════════════


class MemoryConsolidator:
    """
    Consolidate episodic entries into semantic facts.

    Parameters
    ----------
    episodic : EpisodicMemory
    semantic : SemanticMemory
    llm_client : Any, optional
        Instance of ``core/llm/client.py::LLMClient`` for LLM-based
        consolidation. If ``None``, uses rule-based only.
    consolidation_interval_cycles : int, default 1000
        Cycles between consolidation runs.
    """

    def __init__(
        self,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        llm_client: Optional[Any] = None,
        consolidation_interval_cycles: int = 1000,
    ) -> None:
        self.episodic = episodic
        self.semantic = semantic
        self.llm = llm_client
        self.consolidation_interval_cycles = int(consolidation_interval_cycles)
        self._last_consolidation_cycle = 0
        self._last_consolidation_count = 0

    # ── Main entry ───────────────────────────────────────────────────────────

    def should_run(self, current_cycle: int) -> bool:
        """Check if it's time to run consolidation."""
        return (current_cycle - self._last_consolidation_cycle) >= self.consolidation_interval_cycles

    def run_if_due(self, current_cycle: int, batch_size: int = 500) -> Dict[str, Any]:
        """
        Run consolidation if the interval has elapsed.

        Returns a dict with ``ran``, ``n_episodes``, ``n_facts``, ``cycle``.
        """
        if not self.should_run(current_cycle):
            return {"ran": False, "cycle": current_cycle}

        result = self.run(batch_size=batch_size)
        self._last_consolidation_cycle = current_cycle
        self._last_consolidation_count = result.get("n_facts", 0)
        result["cycle"] = current_cycle
        result["ran"] = True
        return result

    def run(self, batch_size: int = 500) -> Dict[str, Any]:
        """
        Single consolidation pass.

        Loads the most recent ``batch_size`` episodic entries, extracts
        facts via rule-based + optional LLM, and writes them to semantic
        memory.

        Returns a summary dict.
        """
        episodes = self.episodic.get_recent(n=batch_size)
        if not episodes:
            return {"n_episodes": 0, "n_facts": 0, "method": "skipped"}

        facts: List[Tuple[str, str, str, float, List[int]]] = []

        # Rule-based extraction: group by (symbol, regime) and compute stats
        facts.extend(self._rule_based_facts(episodes))

        # LLM-based extraction (optional)
        if self.llm is not None:
            try:
                llm_facts = self._llm_based_facts(episodes[:50])  # cap prompts
                facts.extend(llm_facts)
            except Exception as exc:
                logger.debug("LLM-based consolidation skipped: %s", exc)

        # Write facts to semantic memory
        written = 0
        for subject, predicate, obj, confidence, source_ids in facts:
            try:
                self.semantic.add_fact(
                    subject=subject,
                    predicate=predicate,
                    obj=obj,
                    confidence=confidence,
                    source_episode_ids=source_ids,
                )
                written += 1
            except Exception as exc:
                logger.debug("add_fact failed: %s", exc)

        # Apply decay to existing facts (part of the "forgetting" pass)
        self.semantic.decay_all(factor=0.98)

        return {
            "n_episodes": len(episodes),
            "n_facts": written,
            "method": "rule_based" if self.llm is None else "rule_based+llm",
        }

    # ── Rule-based extraction ────────────────────────────────────────────────

    def _rule_based_facts(
        self,
        episodes: List[EpisodicEntry],
    ) -> List[Tuple[str, str, str, float, List[int]]]:
        """Extract facts by grouping episodes and computing simple statistics."""
        facts: List[Tuple[str, str, str, float, List[int]]] = []

        # Group by (symbol, regime) → pnl stats
        groups: Dict[Tuple[str, str], List[EpisodicEntry]] = defaultdict(list)
        for ep in episodes:
            symbol = str(ep.metadata.get("symbol", ""))
            regime = str(ep.metadata.get("regime", ""))
            if symbol and regime:
                groups[(symbol, regime)].append(ep)

        for (symbol, regime), entries in groups.items():
            if len(entries) < 3:
                continue  # need at least 3 samples per regime

            pnls = [float(e.metadata.get("pnl", 0.0)) for e in entries]
            total_pnl = sum(pnls)
            n_win = sum(1 for p in pnls if p > 0)
            win_rate = n_win / len(pnls)
            source_ids = [e.id for e in entries]

            # Fact 1: win rate in this regime
            if win_rate >= 0.6:
                facts.append((
                    symbol,
                    "performs_well_in_regime",
                    regime,
                    min(1.0, win_rate),
                    source_ids,
                ))
            elif win_rate <= 0.4:
                facts.append((
                    symbol,
                    "struggles_in_regime",
                    regime,
                    min(1.0, 1.0 - win_rate),
                    source_ids,
                ))

            # Fact 2: profitability
            if total_pnl > 0:
                facts.append((
                    symbol,
                    "net_profitable_in_regime",
                    regime,
                    0.7,
                    source_ids,
                ))

        return facts

    # ── LLM-based extraction ─────────────────────────────────────────────────

    def _llm_based_facts(
        self,
        episodes: List[EpisodicEntry],
    ) -> List[Tuple[str, str, str, float, List[int]]]:
        """Ask the LLM to summarize episodes into semantic facts."""
        if not episodes:
            return []

        # Build a compact prompt
        ep_descriptions: List[str] = []
        for ep in episodes[:30]:  # safety cap
            symbol = ep.metadata.get("symbol", "")
            regime = ep.metadata.get("regime", "")
            pnl = ep.metadata.get("pnl", 0)
            ep_descriptions.append(
                f"- {ep.event_type} {symbol} regime={regime} pnl={pnl}: {ep.content[:120]}"
            )
        ep_text = "\n".join(ep_descriptions)

        system_prompt = (
            "You are a trading knowledge extractor. Given a list of recent "
            "trading events, output a JSON array of semantic facts, each with "
            '"subject", "predicate", "object", "confidence" (0-1). '
            "Focus on robust cross-cutting patterns, not individual trades."
        )
        user_prompt = f"Recent trading events:\n{ep_text}\n\nExtract up to 5 robust semantic facts as JSON."

        try:
            response = self.llm.complete(
                prompt=user_prompt,
                system=system_prompt,
                json_mode=True,
                max_tokens=500,
            )
        except Exception as exc:
            logger.debug("LLM consolidation failed: %s", exc)
            return []

        parsed = getattr(response, "parsed", None)
        if not isinstance(parsed, list):
            # Try wrapping: some LLM outputs are {"facts": [...]}
            if isinstance(parsed, dict) and "facts" in parsed:
                parsed = parsed["facts"]
            else:
                return []

        out: List[Tuple[str, str, str, float, List[int]]] = []
        all_ids = [e.id for e in episodes]
        for fact in parsed[:10]:
            try:
                subj = str(fact.get("subject", ""))
                pred = str(fact.get("predicate", ""))
                obj = str(fact.get("object", ""))
                conf = float(fact.get("confidence", 0.5))
                if subj and pred and obj:
                    out.append((subj, pred, obj, conf, all_ids))
            except (TypeError, ValueError):
                continue
        return out

    def snapshot(self) -> Dict[str, Any]:
        return {
            "last_consolidation_cycle": self._last_consolidation_cycle,
            "last_consolidation_count": self._last_consolidation_count,
            "consolidation_interval_cycles": self.consolidation_interval_cycles,
            "n_episodes": self.episodic.count(),
            "n_facts": self.semantic.count(),
        }
