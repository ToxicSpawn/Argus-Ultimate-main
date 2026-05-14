"""
domain — canonical typed contracts for the Argus trading system.

Import from here to avoid coupling to internal module layout:

    from domain import Signal, Order, Fill, BotState
"""
from __future__ import annotations

from domain.signal import Signal
from domain.order import Order
from domain.fill import Fill
from domain.state import BotState

__all__ = ["Signal", "Order", "Fill", "BotState"]
