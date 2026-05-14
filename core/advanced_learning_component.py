# pyright: reportMissingImports=false
"""
Integration with Argus Component Registry.

This module registers all advanced learning systems with Argus's component registry
for proper lifecycle management and initialization.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AdvancedLearningComponent:
    """
    Component wrapper for advanced learning systems.
    
    Integrates with Argus's component registry for lifecycle management.
    """

    COMPONENT_NAME = "advanced_learning"
    COMPONENT_VERSION = "1.0.0"
    COMPONENT_DESCRIPTION = "Advanced Learning Systems Integration"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the advanced learning component."""
        self.config = config or {}
        self.orchestrator = None
        self.trading_loop = None
        self.is_initialized = False
        
        logger.info(f"Initializing {self.COMPONENT_NAME} v{self.COMPONENT_VERSION}")

    def initialize(self) -> bool:
        """Initialize all advanced learning systems."""
        try:
            from ml.advanced_learning_integration import (
                AdvancedLearningOrchestrator, IntegratedTradingLoop, LearningConfig, LearningMode
            )

            # Create configuration from config
            mode_str = self.config.get("mode", "full").lower()
            mode_map = {
                "full": LearningMode.FULL,
                "lightweight": LearningMode.LIGHTWEIGHT,
                "quantum": LearningMode.QUANTUM,
                "classical": LearningMode.CLASSICAL,
                "training": LearningMode.TRAINING
            }
            mode = mode_map.get(mode_str, LearningMode.FULL)

            learning_config = LearningConfig(
                mode=mode,
                enable_quantum_rl=self.config.get("enable_quantum_rl", True),
                enable_multi_agent=self.config.get("enable_multi_agent", True),
                enable_knowledge_distillation=self.config.get("enable_knowledge_distillation", True),
                enable_rlhf=self.config.get("enable_rlhf", True),
                enable_uncertainty=self.config.get("enable_uncertainty", True),
                enable_adversarial=self.config.get("enable_adversarial", True),
                enable_active_learning=self.config.get("enable_active_learning", True),
                enable_transfer_learning=self.config.get("enable_transfer_learning", True),
                enable_dashboard=self.config.get("enable_dashboard", True),
                uncertainty_threshold=self.config.get("uncertainty_threshold", 0.6),
                min_confidence=self.config.get("min_confidence", 0.5)
            )

            # Initialize orchestrator
            self.orchestrator = AdvancedLearningOrchestrator(learning_config)
            
            # Initialize trading loop
            self.trading_loop = IntegratedTradingLoop(self.orchestrator)

            self.is_initialized = True
            logger.info(f"{self.COMPONENT_NAME} initialized successfully")
            
            return True

        except Exception as e:
            logger.error(f"Failed to initialize {self.COMPONENT_NAME}: {e}")
            return False

    def process_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data through advanced learning systems."""
        if not self.is_initialized or not self.trading_loop:
            logger.warning("Advanced learning not initialized, returning default decision")
            return {
                "action": 0,
                "action_name": "hold",
                "confidence": 0.5,
                "uncertainty": 0.5,
                "position_size": 1.0,
                "reasoning": "System not initialized"
            }

        return self.trading_loop.process_market_data(market_data)

    def record_trade_outcome(self, 
                            market_data: Dict[str, Any],
                            decision: Dict[str, Any],
                            reward: float,
                            human_rating: Optional[float] = None) -> None:
        """Record trade outcome for learning."""
        if self.trading_loop:
            self.trading_loop.record_trade_outcome(market_data, decision, reward, human_rating)

    def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        if not self.is_initialized:
            return {"status": "not_initialized"}

        return self.orchestrator.get_system_status() if self.orchestrator else {"status": "error"}

    def get_performance(self) -> Dict[str, Any]:
        """Get performance summary."""
        if self.trading_loop:
            return self.trading_loop.get_performance_summary()
        return {"status": "not_available"}

    def shutdown(self) -> bool:
        """Shutdown the component."""
        logger.info(f"Shutting down {self.COMPONENT_NAME}")
        self.is_initialized = False
        return True


def register_component(component_registry: Any, config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Register the advanced learning component with the component registry.
    
    Args:
        component_registry: The Argus component registry
        config: Optional configuration for the component
    
    Returns:
        True if registration was successful
    """
    try:
        component = AdvancedLearningComponent(config)
        
        # Register with component registry
        if hasattr(component_registry, 'register'):
            component_registry.register(
                name=AdvancedLearningComponent.COMPONENT_NAME,
                component=component,
                version=AdvancedLearningComponent.COMPONENT_VERSION,
                description=AdvancedLearningComponent.COMPONENT_DESCRIPTION
            )
            logger.info(f"Registered {AdvancedLearningComponent.COMPONENT_NAME} with component registry")
            return True
        else:
            logger.warning("Component registry does not support registration")
            return False
            
    except Exception as e:
        logger.error(f"Failed to register component: {e}")
        return False


# Convenience function for quick initialization
def create_advanced_learning_system(config: Optional[Dict[str, Any]] = None) -> Optional[AdvancedLearningComponent]:
    """Create and initialize an advanced learning system."""
    component = AdvancedLearningComponent(config)
    if component.initialize():
        return component
    return None


__all__ = [
    "AdvancedLearningComponent",
    "register_component",
    "create_advanced_learning_system"
]