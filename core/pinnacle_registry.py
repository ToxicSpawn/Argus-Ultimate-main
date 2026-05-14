"""
PINNACLE MODULES - Integration Registry

This module provides a central registry for all pinnacle-tier modules.
When modules are ready, they are registered here for system-wide access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModuleStatus:
    """Status of a pinnacle module."""
    name: str
    category: str
    status: str  # "pending", "loading", "ready", "error"
    file_path: str
    loaded_at: Optional[datetime] = None
    error: Optional[str] = None


class PinnacleRegistry:
    """
    Central registry for all pinnacle-tier modules.
    
    Provides lazy loading and system-wide access to:
    - Infrastructure: Ultra-low latency, execution
    - Intelligence: Meta-learning, RL, foundation models, causal
    - Alpha: Microstructure, on-chain, alt-data
    - Risk: Advanced risk engine, real-time VaR
    - Evolution: Self-evolution, strategy genesis
    """
    
    # Module definitions
    MODULES = {
        # Infrastructure
        "ultra_low_latency": {
            "category": "infrastructure",
            "file_path": "core/ultra_low_latency_v2.py",
            "class_name": "UltraLowLatencyCore",
        },
        "institutional_execution": {
            "category": "infrastructure",
            "file_path": "execution/institutional_execution.py",
            "class_name": "InstitutionalExecutionSuite",
        },
        
        # Intelligence
        "meta_learning": {
            "category": "intelligence",
            "file_path": "ml/meta_learning.py",
            "class_name": "MetaLearningSystem",
        },
        "multi_agent_rl": {
            "category": "intelligence",
            "file_path": "ml/multi_agent_rl.py",
            "class_name": "MultiAgentRLSystem",
        },
        "foundation_model": {
            "category": "intelligence",
            "file_path": "ml/foundation_model.py",
            "class_name": "FoundationModelInterface",
        },
        "causal_intelligence": {
            "category": "intelligence",
            "file_path": "ml/causal_intelligence.py",
            "class_name": "CausalIntelligenceEngine",
        },
        
        # Alpha
        "market_microstructure": {
            "category": "alpha",
            "file_path": "analytics/market_microstructure.py",
            "class_name": "MarketMicrostructureEngine",
        },
        "onchain_alpha": {
            "category": "alpha",
            "file_path": "analytics/onchain_alpha.py",
            "class_name": "OnChainAlphaEngine",
        },
        "alt_data_fusion": {
            "category": "alpha",
            "file_path": "analytics/alt_data_fusion.py",
            "class_name": "AltDataFusionEngine",
        },
        
        # Risk
        "advanced_risk": {
            "category": "risk",
            "file_path": "risk/advanced_risk_engine.py",
            "class_name": "AdvancedRiskEngine",
        },
        "realtime_var": {
            "category": "risk",
            "file_path": "risk/realtime_var_aggregator.py",
            "class_name": "RealtimeVARAggregator",
        },
        
        # Evolution
        "self_evolution": {
            "category": "evolution",
            "file_path": "core/self_evolution.py",
            "class_name": "SelfEvolutionPipeline",
        },
        "strategy_genesis": {
            "category": "evolution",
            "file_path": "strategies/strategy_genesis.py",
            "class_name": "StrategyGenesisSystem",
        },
    }
    
    def __init__(self) -> None:
        self._modules: Dict[str, ModuleStatus] = {}
        self._instances: Dict[str, Any] = {}
        self._initialize_registry()
    
    def _initialize_registry(self) -> None:
        """Initialize module status tracking."""
        for name, info in self.MODULES.items():
            self._modules[name] = ModuleStatus(
                name=name,
                category=info["category"],
                status="pending",
                file_path=info["file_path"],
            )
    
    def get_status(self) -> Dict[str, str]:
        """Get status of all modules."""
        return {name: mod.status for name, mod in self._modules.items()}
    
    def get_ready_modules(self) -> List[str]:
        """Get list of ready module names."""
        return [name for name, mod in self._modules.items() if mod.status == "ready"]
    
    def get_modules_by_category(self, category: str) -> List[str]:
        """Get module names by category."""
        return [
            name for name, mod in self._modules.items()
            if mod.category == category
        ]
    
    def mark_ready(self, name: str) -> None:
        """Mark a module as ready."""
        if name in self._modules:
            self._modules[name].status = "ready"
            self._modules[name].loaded_at = datetime.now()
            logger.info("Pinnacle module ready: %s", name)
    
    def mark_error(self, name: str, error: str) -> None:
        """Mark a module as errored."""
        if name in self._modules:
            self._modules[name].status = "error"
            self._modules[name].error = error
            logger.error("Pinnacle module error: %s - %s", name, error)
    
    def get_module(self, name: str) -> Optional[Any]:
        """Get a module instance (lazy loading)."""
        if name in self._instances:
            return self._instances[name]
        
        if name not in self._modules:
            return None
        
        mod = self._modules[name]
        if mod.status != "ready":
            return None
        
        # Lazy import
        try:
            info = self.MODULES[name]
            module_path = info["file_path"].replace("/", ".").replace(".py", "")
            class_name = info["class_name"]
            
            # Dynamic import
            import importlib
            module = importlib.import_module(f"{module_path}")
            cls = getattr(module, class_name)
            instance = cls()
            self._instances[name] = instance
            return instance
        except Exception as e:
            logger.error("Failed to load module %s: %s", name, e)
            return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of pinnacle system status."""
        ready = self.get_ready_modules()
        total = len(self._modules)
        
        categories = {}
        for name, mod in self._modules.items():
            if mod.category not in categories:
                categories[mod.category] = {"ready": 0, "total": 0}
            categories[mod.category]["total"] += 1
            if mod.status == "ready":
                categories[mod.category]["ready"] += 1
        
        return {
            "total_modules": total,
            "ready_modules": len(ready),
            "categories": categories,
            "readiness_pct": (len(ready) / total * 100) if total > 0 else 0,
        }


# Global registry instance
_registry: Optional[PinnacleRegistry] = None


def get_pinnacle_registry() -> PinnacleRegistry:
    """Get the global pinnacle registry."""
    global _registry
    if _registry is None:
        _registry = PinnacleRegistry()
    return _registry
