from __future__ import annotations

import enum


class SystemState(enum.Enum):
    HEALTHY = "HEALTHY"
    DEGRADED_DATA = "DEGRADED_DATA"
    DEGRADED_EXECUTION = "DEGRADED_EXECUTION"
    DEGRADED_RECON = "DEGRADED_RECON"
    FROZEN = "FROZEN"


_current_state: SystemState = SystemState.HEALTHY


def set_state(state: SystemState) -> None:
    global _current_state
    _current_state = state


def get_state() -> SystemState:
    return _current_state


def assert_can_trade() -> None:
    if _current_state == SystemState.DEGRADED_RECON:
        raise RuntimeError(
            f"Trading blocked — system in {_current_state.value} state"
        )
    if _current_state == SystemState.FROZEN:
        raise RuntimeError(
            f"Trading blocked — system in {_current_state.value} state"
        )
