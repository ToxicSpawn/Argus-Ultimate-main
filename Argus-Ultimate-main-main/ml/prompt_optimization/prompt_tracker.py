"""
Prompt performance tracking for trading agents.

Tracks:
- Prompt versions and their market performance
- Win rate, Sharpe ratio, max drawdown per prompt
- Regime-specific performance
- Prompt mutation history
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PromptPerformance:
    """Performance metrics for a prompt version."""
    prompt_id: str
    version: int
    prompt_hash: str
    
    # Trading performance
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    
    # Risk metrics
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    
    # Regime-specific performance
    regime_performance: Dict[str, float] = field(default_factory=dict)
    
    # Timing
    first_used: Optional[datetime] = None
    last_used: Optional[datetime] = None
    
    @property
    def win_rate(self) -> float:
        if self.trades == 0:
            return 0.0
        return self.wins / self.trades
    
    @property
    def avg_return_pct(self) -> float:
        if self.trades == 0:
            return 0.0
        return self.total_return_pct / self.trades
    
    @property
    def sample_size_sufficient(self) -> bool:
        """Check if we have enough trades for statistical significance."""
        return self.trades >= 30
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "version": self.version,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "avg_return_pct": self.avg_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "regime_performance": self.regime_performance,
        }


@dataclass
class PromptRecord:
    """Complete record of a prompt version."""
    prompt_id: str
    version: int
    content: str
    parent_id: Optional[str] = None  # For tracking mutations
    mutation_type: Optional[str] = None  # "rewrite", "tweak", "regime_adapt"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    performance: Optional[PromptPerformance] = None
    
    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


class PromptTracker:
    """Track prompt versions and their market performance.
    
    Maintains a history of prompts and their outcomes,
    enabling data-driven prompt optimization.
    """
    
    def __init__(self, max_prompts: int = 1000) -> None:
        self._max_prompts = max_prompts
        self._prompts: Dict[str, PromptRecord] = {}
        self._prompt_history: List[str] = []  # Ordered by creation
        self._performance_history: List[PromptPerformance] = []
    
    def register_prompt(
        self,
        content: str,
        parent_id: Optional[str] = None,
        mutation_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register a new prompt version. Returns prompt_id."""
        # Generate prompt ID from content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        prompt_id = f"prompt_{content_hash}"
        
        # Check if already registered
        if prompt_id in self._prompts:
            return prompt_id
        
        # Get version number
        version = 1
        if parent_id and parent_id in self._prompts:
            version = self._prompts[parent_id].version + 1
        
        record = PromptRecord(
            prompt_id=prompt_id,
            version=version,
            content=content,
            parent_id=parent_id,
            mutation_type=mutation_type,
            metadata=metadata or {},
        )
        
        self._prompts[prompt_id] = record
        self._prompt_history.append(prompt_id)
        
        # Trim if needed
        if len(self._prompts) > self._max_prompts:
            oldest_id = self._prompt_history.pop(0)
            del self._prompts[oldest_id]
        
        return prompt_id
    
    def record_trade(
        self,
        prompt_id: str,
        pnl: float,
        return_pct: float,
        regime: str = "unknown",
        is_win: bool = True,
    ) -> None:
        """Record a trade outcome for a prompt."""
        if prompt_id not in self._prompts:
            logger.warning("Unknown prompt_id: %s", prompt_id)
            return
        
        record = self._prompts[prompt_id]
        
        if record.performance is None:
            record.performance = PromptPerformance(
                prompt_id=prompt_id,
                version=record.version,
                prompt_hash=record.prompt_hash,
                first_used=datetime.now(),
            )
        
        perf = record.performance
        perf.trades += 1
        perf.total_pnl += pnl
        perf.total_return_pct += return_pct
        perf.last_used = datetime.now()
        
        if is_win:
            perf.wins += 1
        else:
            perf.losses += 1
        
        # Update regime performance
        if regime not in perf.regime_performance:
            perf.regime_performance[regime] = 0.0
        perf.regime_performance[regime] += return_pct
    
    def get_prompt(self, prompt_id: str) -> Optional[PromptRecord]:
        """Get a prompt record by ID."""
        return self._prompts.get(prompt_id)
    
    def get_best_prompts(
        self,
        n: int = 10,
        min_trades: int = 10,
        metric: str = "sharpe_ratio",
    ) -> List[PromptRecord]:
        """Get top N prompts by specified metric."""
        eligible = [
            p for p in self._prompts.values()
            if p.performance and p.performance.trades >= min_trades
        ]
        
        sorted_prompts = sorted(
            eligible,
            key=lambda p: getattr(p.performance, metric, 0.0),
            reverse=True,
        )
        
        return sorted_prompts[:n]
    
    def get_worst_prompts(
        self,
        n: int = 10,
        min_trades: int = 10,
    ) -> List[PromptRecord]:
        """Get worst performing prompts (for anti-patterns)."""
        return self.get_best_prompts(n, min_trades, "sharpe_ratio")[::-1]
    
    def get_regime_best_prompts(
        self,
        regime: str,
        n: int = 5,
    ) -> List[PromptRecord]:
        """Get best prompts for a specific market regime."""
        eligible = [
            p for p in self._prompts.values()
            if p.performance
            and regime in p.performance.regime_performance
            and p.performance.trades >= 10
        ]
        
        def get_regime_pnl(p: PromptRecord) -> float:
            if p.performance is None:
                return 0.0
            return p.performance.regime_performance.get(regime, 0.0)
        
        sorted_prompts = sorted(
            eligible,
            key=get_regime_pnl,
            reverse=True,
        )
        
        return sorted_prompts[:n]
    
    def get_mutation_lineage(self, prompt_id: str) -> List[PromptRecord]:
        """Get the full mutation lineage of a prompt."""
        lineage = []
        current_id = prompt_id
        
        while current_id and current_id in self._prompts:
            record = self._prompts[current_id]
            lineage.append(record)
            current_id = record.parent_id
        
        return lineage[::-1]  # Return in chronological order
    
    def compute_improvement_rate(self, prompt_id: str) -> float:
        """Compute improvement rate from parent to child prompt."""
        if prompt_id not in self._prompts:
            return 0.0
        
        record = self._prompts[prompt_id]
        if not record.parent_id or record.parent_id not in self._prompts:
            return 0.0
        
        parent = self._prompts[record.parent_id]
        if not parent.performance or not record.performance:
            return 0.0
        if parent.performance.trades < 10 or record.performance.trades < 10:
            return 0.0
        
        parent_sharpe = parent.performance.sharpe_ratio
        child_sharpe = record.performance.sharpe_ratio
        
        if parent_sharpe == 0:
            return 0.0
        
        return (child_sharpe - parent_sharpe) / abs(parent_sharpe)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall tracking statistics."""
        total_prompts = len(self._prompts)
        prompts_with_trades = sum(
            1 for p in self._prompts.values()
            if p.performance and p.performance.trades > 0
        )
        total_trades = sum(
            p.performance.trades for p in self._prompts.values()
            if p.performance
        )
        surviving_prompts = sum(
            1 for p in self._prompts.values()
            if p.performance and p.performance.sample_size_sufficient
            and p.performance.sharpe_ratio > 0
        )
        
        return {
            "total_prompts": total_prompts,
            "prompts_with_trades": prompts_with_trades,
            "total_trades": total_trades,
            "surviving_prompts": surviving_prompts,
            "survival_rate": surviving_prompts / max(total_prompts, 1),
        }
