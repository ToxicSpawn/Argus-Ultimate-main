"""
Real-Time Learning Orchestrator - Core System

This module coordinates all adaptive components in the Argus trading system.
It handles:
- Component registration and lifecycle
- Safety validation of parameter changes
- Audit trail of all adaptations
- State management and persistence
- Emergency rollback procedures
"""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Type, Union
import json
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LearningComponent(ABC):
    """Base class for all adaptive components"""

    name: str
    version: str = "1.0"
    enabled: bool = True
    last_updated: Optional[datetime] = None
    update_frequency: int = 1  # How often to update (in trade cycles)

    @abstractmethod
    def learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from new data and return updated parameters"""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get current parameters"""
        pass

    @abstractmethod
    def rollback(self) -> None:
        """Revert to last known good state"""
        pass

    @abstractmethod
    def validate(self, new_params: Dict[str, Any]) -> bool:
        """Validate proposed parameter changes"""
        pass

    def should_update(self, cycle_count: int) -> bool:
        """Determine if component should update on this cycle"""
        return self.enabled and (cycle_count % self.update_frequency == 0)


@dataclass
class LearningAudit:
    """Tracks all learning events and parameter changes"""

    max_history: int = 1000
    history: List[Dict] = field(default_factory=list)
    rollbacks: List[Dict] = field(default_factory=list)

    def log_change(self, component: str, old_params: Dict, new_params: Dict) -> None:
        """Log a parameter change"""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": component,
            "old_params": old_params,
            "new_params": new_params,
            "changes": {k: new_params[k] for k in new_params if k in old_params and new_params[k] != old_params[k]}
        }
        self.history.append(entry)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def log_rollback(self, component: str, timestamp: str, reason: str) -> None:
        """Log a rollback event"""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": component,
            "rolled_back_to": timestamp,
            "reason": reason
        }
        self.rollbacks.append(entry)
        if len(self.rollbacks) > self.max_history:
            self.rollbacks.pop(0)

    def get_recent_changes(self, hours: int = 1) -> List[Dict]:
        """Get recent changes within time window"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [e for e in self.history
                if datetime.fromisoformat(e["timestamp"]) >= cutoff]

    def get_recent_rollbacks(self, hours: int = 1) -> List[Dict]:
        """Get recent rollbacks within time window"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [e for e in self.rollbacks
                if datetime.fromisoformat(e["timestamp"]) >= cutoff]


class SafetyValidator:
    """Validates all parameter changes before application"""

    def __init__(self):
        # Default safety rules (can be overridden per component)
        self.global_rules = {
            "max_parameter_change": 0.25,  # 25% max change at once
            "max_adaptations_per_hour": 10,
            "min_confidence_threshold": 0.7
        }

        # Component-specific rules
        self.component_rules = {
            "sizing": {
                "max_leverage_change": 0.5,
                "max_position_pct_change": 0.1
            },
            "execution": {
                "max_participation_change": 0.1,
                "min_latency_buffer": 2  # ms
            },
            "risk": {
                "min_var_confidence": 0.90,
                "max_drawdown_increase": 0.02  # 2%
            }
        }

    def validate_change(self, component_name: str, old_params: Dict, new_params: Dict) -> bool:
        """Validate a proposed parameter change"""
        # Check global rules
        if not self._check_global_rules(component_name, old_params, new_params):
            return False

        # Check component-specific rules
        if not self._check_component_rules(component_name, old_params, new_params):
            return False

        return True

    def _check_global_rules(self, component_name: str, old_params: Dict, new_params: Dict) -> bool:
        """Check global safety rules"""
        # Check adaptation rate (per hour)
        recent_changes = len([e for e in self.orchestrator.audit.get_recent_changes(hours=1)
                             if e["component"] == component_name])
        if recent_changes >= self.global_rules["max_adaptations_per_hour"]:
            logger.warning(f"Adaptation rate limit reached for {component_name}")
            return False

        # Check parameter change magnitude
        for param in new_params:
            if param in old_params:
                old_val = old_params[param]
                new_val = new_params[param]

                # Skip non-numeric parameters
                if not isinstance(old_val, (int, float)) or not isinstance(new_val, (int, float)):
                    continue

                # Calculate percentage change
                if old_val != 0:
                    change_pct = abs((new_val - old_val) / old_val)
                    if change_pct > self.global_rules["max_parameter_change"]:
                        logger.warning(
                            f"Parameter {param} change too large in {component_name}: "
                            f"{change_pct:.1%} > {self.global_rules['max_parameter_change']:.1%}"
                        )
                        return False
        return True

    def _check_component_rules(self, component_name: str, old_params: Dict, new_params: Dict) -> bool:
        """Check component-specific safety rules"""
        if component_name not in self.component_rules:
            return True  # No specific rules for this component

        rules = self.component_rules[component_name]

        # Check sizing-specific rules
        if component_name == "sizing":
            if "max_position_pct" in new_params and "max_position_pct" in old_params:
                change = new_params["max_position_pct"] - old_params["max_position_pct"]
                if abs(change) > rules["max_position_pct_change"]:
                    logger.warning(
                        f"Position size change too large in {component_name}: "
                        f"{change:.1%} > {rules['max_position_pct_change']:.1%}"
                    )
                    return False

            if "leverage" in new_params and "leverage" in old_params:
                change = new_params["leverage"] - old_params["leverage"]
                if abs(change) > rules["max_leverage_change"]:
                    logger.warning(
                        f"Leverage change too large in {component_name}: "
                        f"{change:.1f}x > {rules['max_leverage_change']:.1f}x"
                    )
                    return False

        # Check execution-specific rules
        elif component_name == "execution":
            if "participation_rate" in new_params and "participation_rate" in old_params:
                change = new_params["participation_rate"] - old_params["participation_rate"]
                if abs(change) > rules["max_participation_change"]:
                    logger.warning(
                        f"Participation rate change too large in {component_name}: "
                        f"{change:.1%} > {rules['max_participation_change']:.1%}"
                    )
                    return False

            if "latency_buffer" in new_params and new_params["latency_buffer"] < rules["min_latency_buffer"]:
                logger.warning(
                    f"Latency buffer too small in {component_name}: "
                    f"{new_params['latency_buffer']}ms < {rules['min_latency_buffer']}ms"
                )
                return False

        # Check risk-specific rules
        elif component_name == "risk":
            if "var_confidence" in new_params and new_params["var_confidence"] < rules["min_var_confidence"]:
                logger.warning(
                    f"VaR confidence too low in {component_name}: "
                    f"{new_params['var_confidence']:.0%} < {rules['min_var_confidence']:.0%}"
                )
                return False

        return True


class RealTimeLearningOrchestrator:
    """Core system that coordinates all real-time learning components"""

    def __init__(self):
        self.components: Dict[str, LearningComponent] = {}
        self.audit = LearningAudit()
        self.safety = SafetyValidator()
        self.safety.orchestrator = self  # Circular reference for audit access
        self.cycle_count = 0
        self.paused_components: Dict[str, datetime] = {}

        # Performance tracking
        self.performance_history = []
        self.max_history = 1000

    def register_component(self, component: LearningComponent) -> None:
        """Register a new learning component"""
        if component.name in self.components:
            logger.warning(f"Component {component.name} already registered - replacing")
        self.components[component.name] = component
        logger.info(f"Registered component: {component.name} v{component.version}")

    def on_market_data(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Process market data through all components"""
        self.cycle_count += 1
        results = {}

        # Determine which components should update this cycle
        components_to_update = [
            name for name, component in self.components.items()
            if component.should_update(self.cycle_count)
        ]

        logger.debug(f"Cycle {self.cycle_count}: Updating components {components_to_update}")

        for name in components_to_update:
            if name in self.paused_components:
                continue

            component = self.components[name]
            try:
                old_params = component.get_params()
                new_params = component.learn(data)

                # Validate changes
                if not self.safety.validate_change(name, old_params, new_params):
                    component.rollback()
                    self.audit.log_rollback(
                        name,
                        datetime.now(timezone.utc).isoformat(),
                        "Safety validation failed"
                    )
                    continue

                # Log successful change
                self.audit.log_change(name, old_params, new_params)
                results[name] = {
                    "old_params": old_params,
                    "new_params": new_params,
                    "status": "success"
                }

            except Exception as e:
                logger.error(f"Error in component {name}: {str(e)}")
                component.rollback()
                self.audit.log_rollback(
                    name,
                    datetime.now(timezone.utc).isoformat(),
                    f"Exception: {str(e)}"
                )
                results[name] = {
                    "status": "error",
                    "error": str(e)
                }

        return results

    def on_trade(self, trade: Dict[str, Any]) -> None:
        """Process trade execution data"""
        trade_results = {}

        for name, component in self.components.items():
            if hasattr(component, 'learn_from_trade'):
                try:
                    old_params = component.get_params()
                    component.learn_from_trade(trade)
                    new_params = component.get_params()

                    if old_params != new_params:
                        self.audit.log_change(name, old_params, new_params)
                        trade_results[name] = {
                            "old_params": old_params,
                            "new_params": new_params,
                            "status": "success"
                        }
                except Exception as e:
                    logger.error(f"Error in trade learning for {name}: {str(e)}")
                    component.rollback()
                    self.audit.log_rollback(
                        name,
                        datetime.now(timezone.utc).isoformat(),
                        f"Trade learning exception: {str(e)}"
                    )
                    trade_results[name] = {
                        "status": "error",
                        "error": str(e)
                    }

        return trade_results

    def pause_learning(self, component_name: Optional[str] = None) -> None:
        """Pause learning for a specific component or all components"""
        if component_name:
            if component_name in self.components:
                self.paused_components[component_name] = datetime.now(timezone.utc)
                logger.info(f"Paused learning for {component_name}")
        else:
            self.paused_components = {
                name: datetime.now(timezone.utc)
                for name in self.components.keys()
            }
            logger.info("Paused learning for all components")

    def resume_learning(self, component_name: Optional[str] = None) -> None:
        """Resume learning for a specific component or all components"""
        if component_name:
            if component_name in self.paused_components:
                del self.paused_components[component_name]
                logger.info(f"Resumed learning for {component_name}")
        else:
            self.paused_components = {}
            logger.info("Resumed learning for all components")

    def rollback_component(self, component_name: str, timestamp: Optional[str] = None) -> None:
        """Force rollback of a component"""
        if component_name not in self.components:
            logger.warning(f"Component {component_name} not found")
            return

        self.components[component_name].rollback()
        reason = f"Manual rollback{' to ' + timestamp if timestamp else ''}"
        self.audit.log_rollback(
            component_name,
            datetime.now(timezone.utc).isoformat(),
            reason
        )
        logger.info(f"Rolled back {component_name}: {reason}")

    def rollback_all(self) -> None:
        """Rollback all components to last known good state"""
        for name in self.components.keys():
            self.rollback_component(name)
        logger.warning("Rolled back all components")

    def emergency_shutdown(self) -> None:
        """Immediately disable all learning"""
        self.rollback_all()
        self.pause_learning()
        logger.critical("EMERGENCY SHUTDOWN: All learning paused and rolled back")

    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "cycle_count": self.cycle_count,
            "components": {
                name: {
                    "version": comp.version,
                    "enabled": comp.enabled,
                    "last_updated": comp.last_updated,
                    "paused": name in self.paused_components,
                    "params": comp.get_params()
                }
                for name, comp in self.components.items()
            },
            "recent_changes": self.audit.get_recent_changes(hours=1),
            "recent_rollbacks": self.audit.get_recent_rollbacks(hours=1),
            "paused_components": list(self.paused_components.keys())
        }

    def save_state(self, path: Union[str, Path]) -> None:
        """Save current state to file"""
        state = {
            "cycle_count": self.cycle_count,
            "components": {
                name: {
                    "params": comp.get_params(),
                    "version": comp.version,
                    "last_updated": comp.last_updated
                }
                for name, comp in self.components.items()
            },
            "audit_history": self.audit.history,
            "audit_rollbacks": self.audit.rollbacks,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved state to {path}")

    def load_state(self, path: Union[str, Path]) -> None:
        """Load state from file"""
        path = Path(path)
        if not path.exists():
            logger.warning(f"State file not found: {path}")
            return

        with open(path, 'r') as f:
            state = json.load(f)

        for name, comp_data in state["components"].items():
            if name in self.components:
                if hasattr(self.components[name], '_restore_state'):
                    self.components[name]._restore_state(comp_data)
                else:
                    # Fallback for components without _restore_state
                    for k, v in comp_data["params"].items():
                        if hasattr(self.components[name], 'params') and k in self.components[name].params:
                            self.components[name].params[k] = v

        # Restore audit history
        self.audit.history = state.get("audit_history", [])
        self.audit.rollbacks = state.get("audit_rollbacks", [])
        self.cycle_count = state.get("cycle_count", 0)

        logger.info(f"Loaded state from {path}")