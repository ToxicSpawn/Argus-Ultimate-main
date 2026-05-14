"""Push 95 — Pydantic response models for all API endpoints."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel:  # type: ignore
        pass
    def Field(*a, **kw):  # type: ignore
        return None


# ---------------------------------------------------------------------------
# Push 79 models (unchanged)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Push 79 models (unchanged)
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status:      str
    version:     str
    uptime_secs: float
    codename:    str = ""


class StatusResponse(BaseModel):
    engine:    Dict[str, Any] = {}
    risk:      Dict[str, Any] = {}
    execution: Dict[str, Any] = {}
    bus:       Dict[str, Any] = {}


class PositionModel(BaseModel):
    symbol:         str
    side:           str
    qty:            float
    avg_entry:      float
    realised_pnl:   float
    unrealised_pnl: float
    notional:       float


class OrderModel(BaseModel):
    order_id:    str
    symbol:      str
    side:        str
    type:        str
    qty:         float
    price:       float
    status:      str
    filled_qty:  float
    avg_price:   float
    strategy_id: Optional[str] = None


class SignalModel(BaseModel):
    symbol:      str
    side:        str
    strength:    float
    strategy_id: Optional[str] = None
    order_type:  Optional[str] = None
    timestamp:   Optional[float] = None


class KillSwitchRequest(BaseModel):
    action: str
    reason: Optional[str] = None


class KillSwitchResponse(BaseModel):
    success:            bool
    kill_switch_active: bool
    message:            str = ""


class BacktestRequest(BaseModel):
    strategy: str = "momentum"
    n_bars:   int = 300
    mc_sims:  int = 200


class ErrorResponse(BaseModel):
    error:  str
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Push 93 models
# ---------------------------------------------------------------------------

class RegimeResponse(BaseModel):
    regime:       str
    vol_ratio:    float
    trend_score:  float
    bb_pos:       float
    autocorr:     float
    confidence:   float
    regime_wired: bool


class SizerSummaryResponse(BaseModel):
    scalars:       Dict[str, float]
    active_regime: Optional[str]
    sizer_wired:   bool


class BanditAllocationResponse(BaseModel):
    allocations:  Dict[str, float]
    regime:       Optional[str]
    bandit_wired: bool


# ---------------------------------------------------------------------------
# Push 94 models
# ---------------------------------------------------------------------------

class RegimeTransitionModel(BaseModel):
    seq:           int
    timestamp:     float
    iso:           str
    from_regime:   Optional[str]
    to_regime:     str
    duration_secs: Optional[float]
    context:       Dict[str, Any]


class RegimeHistoryResponse(BaseModel):
    transitions:   List[RegimeTransitionModel]
    count:         int
    buffer_maxlen: int
    history_wired: bool


class RegimeStatsResponse(BaseModel):
    total_transitions:     int
    unique_regimes:        List[str]
    regime_counts:         Dict[str, int]
    avg_duration_secs:     Optional[float]
    min_duration_secs:     Optional[float]
    max_duration_secs:     Optional[float]
    current_regime:        Optional[str]
    current_since:         Optional[float]
    current_duration_secs: Optional[float]
    history_wired:         bool


# ---------------------------------------------------------------------------
# Push 95 models — Alert Rules
# ---------------------------------------------------------------------------

class AlertRule(BaseModel):
    """Single configurable alert threshold."""
    name:        str                   # e.g. "regime_dwell_min_secs"
    value:       float                 # threshold value
    enabled:     bool    = True
    description: str     = ""


class AlertRulesResponse(BaseModel):
    rules:        List[AlertRule]
    count:        int
    alerts_wired: bool


class AlertRuleUpdateRequest(BaseModel):
    name:    str
    value:   float
    enabled: bool = True
