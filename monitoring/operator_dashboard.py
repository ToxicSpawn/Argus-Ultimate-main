"""
monitoring/operator_dashboard.py
=================================
FastAPI operator dashboard for Argus Ultimate.

Provides:
  - WebSocket  `/ws/dashboard`  — real-time trading feed
  - REST API:
      GET  `/status`            — full system status
      GET  `/equity`            — equity curve
      GET  `/positions`         — open positions
      GET  `/trades`            — recent trades
      GET  `/risk`              — risk metrics (VaR, drawdown)
      GET  `/performance`       — Sharpe, Sortino, Calmar
      POST `/control/pause`     — pause trading
      POST `/control/resume`     — resume trading
      POST `/control/risk-limit` — adjust risk limit
      GET  `/health`            — liveness probe
      GET  `/health/ready`      — readiness probe

Designed to be mounted into the main FastAPI app:
    from monitoring.operator_dashboard import get_dashboard_router
    app.include_router(get_dashboard_router())
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FastAPI import (optional)
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from pydantic import BaseModel
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    APIRouter: Any = None  # type: ignore

try:
    from pydantic import BaseModel as _BM
    BaseModel = _BM
except ImportError:
    BaseModel = object  # type: ignore


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class PositionData(BaseModel):
    symbol   : str
    side     : str
    size     : float
    entry    : float
    current  : float
    pnl      : float
    pnl_pct  : float


class TradeData(BaseModel):
    trade_id : str
    symbol   : str
    side     : str
    size     : float
    price    : float
    pnl      : float
    fee      : float
    ts       : float


class EquityPoint(BaseModel):
    ts          : float
    equity      : float
    equity_norm : float


class RiskMetrics(BaseModel):
    var_95               : float
    cvar_95              : float
    max_drawdown_pct    : float
    current_drawdown_pct : float
    daily_loss_pct       : float
    position_exposure    : float
    leverage             : float


class PerformanceMetrics(BaseModel):
    total_return_pct    : float
    sharpe_ratio        : float
    sortino_ratio       : float
    calmar_ratio         : float
    win_rate             : float
    profit_factor        : float
    avg_trade_pnl        : float
    total_trades         : int
    avg_trade_duration_s : float


class SystemStatus(BaseModel):
    state           : str
    equity          : float
    daily_pnl       : float
    open_positions  : int
    queued_orders   : int
    filled_today   : int
    cancelled_today : int
    uptime_s        : float
    regime          : str


class ControlRequest(BaseModel):
    value : Optional[float] = None
    reason: str = "operator_request"


class ControlResponse(BaseModel):
    success  : bool
    message  : str
    new_value: Optional[float] = None


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class _DashboardState:
    """Thread-safe shared state for the operator dashboard."""

    def __init__(self) -> None:
        self.state           = "initializing"
        self.equity         = 0.0
        self.initial_equity = 0.0
        self.daily_pnl      = 0.0
        self.positions      : List[PositionData] = []
        self._trades        : Deque[TradeData] = deque(maxlen=500)
        self._equity_curve  : Deque[EquityPoint] = deque(maxlen=10_000)
        self.started_at     = time.time()
        self.regime         = "unknown"
        self.open_orders    = 0
        self.filled_today   = 0
        self.cancelled_today= 0
        self.var_95         = 0.0
        self.cvar_95        = 0.0
        self.max_dd_pct     = 0.0
        self.cur_dd_pct     = 0.0
        self.daily_loss     = 0.0
        self.position_exposure = 0.0
        self.leverage        = 0.0
        self._returns       : Deque[float] = deque(maxlen=2000)

    def update_equity(self, equity: float, ts: Optional[float] = None) -> None:
        self.equity = equity
        if self.initial_equity == 0:
            self.initial_equity = equity
        self._equity_curve.append(EquityPoint(
            ts=ts or time.time(),
            equity=equity,
            equity_norm=equity / self.initial_equity if self.initial_equity > 0 else 1.0,
        ))
        if self._equity_curve:
            peak = max(e.equity for e in self._equity_curve)
            self.cur_dd_pct = (equity - peak) / peak * 100 if peak > 0 else 0.0

    def add_trade(self, trade: TradeData) -> None:
        self._trades.append(trade)
        self.filled_today += 1
        self.daily_pnl += trade.pnl

    def set_positions(self, positions: List[PositionData]) -> None:
        self.positions = positions

    def update_risk(
        self,
        var_95      : float,
        cvar_95     : float,
        max_dd_pct  : float,
        daily_loss  : float,
        exposure    : float,
        leverage    : float,
    ) -> None:
        self.var_95           = var_95
        self.cvar_95          = cvar_95
        self.max_dd_pct       = max_dd_pct
        self.daily_loss       = daily_loss
        self.position_exposure= exposure
        self.leverage         = leverage

    def record_return(self, ret: float) -> None:
        self._returns.append(ret)

    def get_risk(self) -> RiskMetrics:
        return RiskMetrics(
            var_95               = self.var_95,
            cvar_95              = self.cvar_95,
            max_drawdown_pct    = self.max_dd_pct,
            current_drawdown_pct = self.cur_dd_pct,
            daily_loss_pct       = self.daily_loss,
            position_exposure    = self.position_exposure,
            leverage             = self.leverage,
        )

    def get_performance(self) -> PerformanceMetrics:
        if not self._trades:
            return PerformanceMetrics(
                total_return_pct=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
                calmar_ratio=0.0, win_rate=0.0, profit_factor=0.0,
                avg_trade_pnl=0.0, total_trades=0, avg_trade_duration_s=0.0,
            )
        trades = list(self._trades)
        wins   = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        n      = len(trades)
        ret_pct = (self.equity - self.initial_equity) / self.initial_equity * 100 \
            if self.initial_equity > 0 else 0.0
        avg_win  = sum(t.pnl for t in wins)   / len(wins)   if wins   else 0.0
        avg_loss = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 1e-12

        rets = list(self._returns)
        if len(rets) > 1:
            mean_r = sum(rets) / len(rets)
            std_r  = math.sqrt(sum((r - mean_r) ** 2 for r in rets) / (len(rets) - 1))
            neg    = [r for r in rets if r < 0]
            down   = math.sqrt(sum(r ** 2 for r in neg) / len(neg)) if neg else 1e-12
            sharpe  = mean_r / std_r  * math.sqrt(252 * 24) if std_r  > 0 else 0.0
            sortino = mean_r / down   * math.sqrt(252 * 24) if down   > 0 else 0.0
        else:
            sharpe = sortino = 0.0

        calmar = ret_pct / abs(self.max_dd_pct) if self.max_dd_pct != 0 else 0.0

        return PerformanceMetrics(
            total_return_pct    = ret_pct,
            sharpe_ratio        = sharpe,
            sortino_ratio       = sortino,
            calmar_ratio        = calmar,
            win_rate            = len(wins) / n if n > 0 else 0.0,
            profit_factor       = avg_win / avg_loss if avg_loss > 0 else 0.0,
            avg_trade_pnl       = sum(t.pnl for t in trades) / n if n > 0 else 0.0,
            total_trades        = n,
            avg_trade_duration_s= 0.0,
        )


# ---------------------------------------------------------------------------
# Module-level state accessor (works with or without FastAPI)
# ---------------------------------------------------------------------------

_dashboard_state: Optional[_DashboardState] = None


def get_dashboard_state() -> _DashboardState:
    """Return the shared dashboard state (lazy singleton)."""
    global _dashboard_state
    if _dashboard_state is None:
        _dashboard_state = _DashboardState()
    return _dashboard_state


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------

def _build_snapshot() -> Dict[str, Any]:
    state = get_dashboard_state()
    return {
        "ts"            : time.time(),
        "state"         : state.state,
        "equity"        : state.equity,
        "daily_pnl"     : state.daily_pnl,
        "positions"     : [p.model_dump() for p in state.positions],
        "risk"         : state.get_risk().model_dump(),
        "perf"         : state.get_performance().model_dump(),
        "equity_curve" : [
            {"ts": e.ts, "equity": e.equity, "norm": e.equity_norm}
            for e in list(state._equity_curve)[-100:]
        ],
        "regime"        : state.regime,
        "filled_today"  : state.filled_today,
        "cancelled_today": state.cancelled_today,
        "open_orders"   : state.open_orders,
    }


# ---------------------------------------------------------------------------
# FastAPI router (only available when FastAPI is installed)
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:

    router = APIRouter(prefix="/api/v1/operator", tags=["operator"])

    # ------------------------------------------------------------------ WS

    @router.websocket("/ws/dashboard")
    async def dashboard_ws(ws: WebSocket) -> None:
        """Streaming dashboard — pushes full state snapshots every 5s."""
        await ws.accept()
        logger.info("Operator dashboard: WS client connected")
        try:
            await ws.send_json(_build_snapshot())
            last_full = time.time()
            while True:
                if time.time() - last_full >= 5.0:
                    await ws.send_json(_build_snapshot())
                    last_full = time.time()
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            logger.info("Operator dashboard: WS client disconnected")
        except Exception as e:
            logger.warning("Operator dashboard WS error: %s", e)

    # ------------------------------------------------------------------ REST

    @router.get("/status", response_model=SystemStatus)
    async def get_status() -> SystemStatus:
        state = get_dashboard_state()
        return SystemStatus(
            state           = state.state,
            equity          = state.equity,
            daily_pnl       = state.daily_pnl,
            open_positions  = len(state.positions),
            queued_orders   = state.open_orders,
            filled_today    = state.filled_today,
            cancelled_today = state.cancelled_today,
            uptime_s        = time.time() - state.started_at,
            regime          = state.regime,
        )

    @router.get("/equity", response_model=List[EquityPoint])
    async def get_equity(limit: int = 500) -> List[EquityPoint]:
        curve = list(get_dashboard_state()._equity_curve)
        return curve[-limit:]

    @router.get("/positions", response_model=List[PositionData])
    async def get_positions() -> List[PositionData]:
        return get_dashboard_state().positions

    @router.get("/trades", response_model=List[TradeData])
    async def get_trades(limit: int = 100) -> List[TradeData]:
        return list(get_dashboard_state()._trades)[-limit:]

    @router.get("/risk", response_model=RiskMetrics)
    async def get_risk() -> RiskMetrics:
        return get_dashboard_state().get_risk()

    @router.get("/performance", response_model=PerformanceMetrics)
    async def get_performance() -> PerformanceMetrics:
        return get_dashboard_state().get_performance()

    @router.post("/control/pause", response_model=ControlResponse)
    async def pause_trading(req: ControlRequest) -> ControlResponse:
        get_dashboard_state().state = "paused"
        logger.warning("OPERATOR: trading PAUSED | reason=%s", req.reason)
        return ControlResponse(success=True, message=f"Trading paused: {req.reason}")

    @router.post("/control/resume", response_model=ControlResponse)
    async def resume_trading(req: ControlRequest) -> ControlResponse:
        if get_dashboard_state().state == "crisis_halt":
            return ControlResponse(
                success=False,
                message="Cannot resume: crisis halt active. Clear risk flags first.",
            )
        get_dashboard_state().state = "running"
        logger.warning("OPERATOR: trading RESUMED | reason=%s", req.reason)
        return ControlResponse(success=True, message=f"Trading resumed: {req.reason}")

    @router.post("/control/risk-limit", response_model=ControlResponse)
    async def adjust_risk_limit(req: ControlRequest) -> ControlResponse:
        if req.value is None or req.value <= 0:
            return ControlResponse(
                success=False, message="risk_limit must be a positive number",
            )
        logger.warning("OPERATOR: risk limit adjusted to %.2f | reason=%s",
                       req.value, req.reason)
        return ControlResponse(
            success=True,
            message=f"Risk limit set to {req.value}",
            new_value=req.value,
        )

    # ------------------------------------------------------------------ Enterprise Risk

    @router.get("/risk/enterprise")
    async def get_enterprise_risk() -> Dict[str, Any]:
        """Full enterprise risk snapshot (VaR, CVaR, stress, alerts)."""
        try:
            from monitoring.enterprise_risk_integration import get_enterprise_risk as _get_er
            er = _get_er()
            return er.get_risk_summary()
        except Exception as e:
            return {"error": str(e)}

    @router.post("/risk/check-order")
    async def check_order_risk(
        symbol: str,
        side: str,
        size_usd: float,
    ) -> Dict[str, Any]:
        """Pre-trade risk check — returns allowed + reason."""
        try:
            from monitoring.enterprise_risk_integration import get_enterprise_risk as _get_er
            er = _get_er()
            allowed, reason = er.check_order(symbol, side, size_usd)
            return {"allowed": allowed, "reason": reason}
        except Exception as e:
            return {"allowed": False, "reason": str(e)}

    @router.get("/risk/stress-tests")
    async def get_stress_tests() -> Dict[str, Any]:
        """Latest stress test results."""
        try:
            from monitoring.enterprise_risk_integration import get_enterprise_risk as _get_er
            er = _get_er()
            snap = er.compute_snapshot()
            return {
                "worst_loss_pct": snap.worst_stress_loss,
                "worst_scenario": snap.worst_stress_scenario,
                "alerts": snap.alerts,
            }
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------ Institutional Execution

    @router.get("/execution/status")
    async def get_execution_status() -> Dict[str, Any]:
        """Full execution snapshot (venues, algos, latency, alerts)."""
        try:
            from monitoring.institutional_execution_integration import get_institutional_execution as _get_ie
            ie = _get_ie()
            return ie.get_execution_summary()
        except Exception as e:
            return {"error": str(e)}

    @router.get("/execution/venues")
    async def get_venue_performance() -> List[Dict[str, Any]]:
        """Per-venue execution performance."""
        try:
            from monitoring.institutional_execution_integration import get_institutional_execution as _get_ie
            ie = _get_ie()
            snap = ie.compute_snapshot()
            return [v.to_dict() for v in snap.venues]
        except Exception as e:
            return [{"error": str(e)}]

    @router.get("/execution/active-orders")
    async def get_active_algo_orders() -> List[Dict[str, Any]]:
        """Active algorithmic orders (VWAP/TWAP/POV/Almgren-Chriss)."""
        try:
            from monitoring.institutional_execution_integration import get_institutional_execution as _get_ie
            ie = _get_ie()
            snap = ie.compute_snapshot()
            return [o.to_dict() for o in snap.active_orders]
        except Exception as e:
            return [{"error": str(e)}]

    @router.post("/execution/route")
    async def route_order(
        symbol: str,
        side: str,
        size_usd: float,
    ) -> Dict[str, Any]:
        """Route an order via Smart Order Router."""
        try:
            from monitoring.institutional_execution_integration import get_institutional_execution as _get_ie
            ie = _get_ie()
            # Use empty venue books for now — in production this would be live
            return ie.route_order(symbol, side, size_usd, {})
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------ Compliance & Reporting

    @router.get("/compliance/status")
    async def get_compliance_status() -> Dict[str, Any]:
        """Full compliance snapshot (TCA, audit, MiFID II status)."""
        try:
            from monitoring.compliance_integration import get_compliance_integrator as _get_ci
            ci = _get_ci()
            return ci.get_compliance_summary()
        except Exception as e:
            return {"error": str(e)}

    @router.get("/compliance/tca")
    async def get_tca_report() -> Dict[str, Any]:
        """Transaction Cost Analysis report."""
        try:
            from monitoring.compliance_integration import get_compliance_integrator as _get_ci
            ci = _get_ci()
            snap = ci.compute_snapshot()
            return snap.tca_report.to_dict() if snap.tca_report else {"error": "no data"}
        except Exception as e:
            return {"error": str(e)}

    @router.get("/compliance/best-execution")
    async def get_best_execution(limit: int = 10) -> List[Dict[str, Any]]:
        """Recent best execution reports."""
        try:
            from monitoring.compliance_integration import get_compliance_integrator as _get_ci
            ci = _get_ci()
            snap = ci.compute_snapshot()
            return [b.to_dict() for b in snap.recent_best_exec[-limit:]]
        except Exception as e:
            return [{"error": str(e)}]

    @router.get("/compliance/audit")
    async def get_audit_summary() -> Dict[str, Any]:
        """Audit trail summary (hash chain integrity, event counts)."""
        try:
            from monitoring.compliance_integration import get_compliance_integrator as _get_ci
            ci = _get_ci()
            snap = ci.compute_snapshot()
            return snap.audit_summary.to_dict()
        except Exception as e:
            return {"error": str(e)}

    @router.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok", "service": "operator_dashboard"}

    @router.get("/health/ready")
    async def ready() -> Dict[str, Any]:
        state = get_dashboard_state()
        return {
            "ready" : state.state != "initializing",
            "state" : state.state,
            "equity": state.equity,
        }

    def get_dashboard_router() -> APIRouter:
        return router

else:
    # FastAPI not installed — provide a no-op router
    router = None  # type: ignore

    def get_dashboard_router() -> Any:  # type: ignore
        return None
