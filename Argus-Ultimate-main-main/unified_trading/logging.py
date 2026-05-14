"""
Logging Module
==============

Audit logging for trading operations.
Refactored from unified_trading_system.py.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class AuditLog:
    """Audit log entry."""
    action: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user: str = "system"


class AuditLogger:
    """
    Handles audit logging for trading operations.
    """
    
    def __init__(self):
        self._logs: list = []
        logger.info("AuditLogger initialized")
    
    async def log_order(self, order: Any, action: str):
        """Log order action."""
        log = AuditLog(
            action=f"order_{action}",
            details={
                "order_id": getattr(order, 'id', None),
                "symbol": getattr(order, 'symbol', None),
                "side": getattr(order, 'side', None),
                "quantity": str(getattr(order, 'quantity', None))
            }
        )
        self._logs.append(log)
        logger.info(f"Order {action}: {getattr(order, 'id', None)}")
    
    async def log_risk_violation(self, signal: Any, reason: str):
        """Log risk violation."""
        log = AuditLog(
            action="risk_violation",
            details={
                "symbol": getattr(signal, 'symbol', None),
                "reason": reason
            }
        )
        self._logs.append(log)
        logger.warning(f"Risk violation: {reason}")
