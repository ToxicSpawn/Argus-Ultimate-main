"""
Quantum Improvements Integration
Integrates all enhanced quantum features into bot when EnhancedQuantumSystem is available.
Safe when enhanced_quantum_system is a placeholder: returns None and skips integration.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def integrate_enhanced_quantum(config: dict[str, Any]) -> Optional[Any]:
    """
    Integrate enhanced quantum system into bot when available.

    Args:
        config: Bot configuration

    Returns:
        Enhanced quantum system instance, or None if module is placeholder/unavailable.
    """
    try:
        from .enhanced_quantum_system import EnhancedQuantumSystem
    except Exception as e:
        logger.debug("Enhanced quantum system not available (placeholder or deps): %s", e)
        return None

    try:
        quantum_config = config.get("quantum", {}) or {}
        enhanced_quantum = EnhancedQuantumSystem(
            {
                **quantum_config,
                "use_gpu": quantum_config.get("use_gpu", True),
                "max_qubits": quantum_config.get("max_qubits", 20),
                "enable_all_features": True,
                "use_real_hardware": False,
            }
        )
        logger.info("✅ Enhanced Quantum System integrated")
        return enhanced_quantum
    except RuntimeError as e:
        if "placeholder" in str(e).lower() or "stub" in str(e).lower():
            logger.debug("Enhanced quantum system is placeholder: %s", e)
            return None
        raise
    except Exception as e:
        logger.debug("Enhanced quantum system integration skipped: %s", e)
        return None