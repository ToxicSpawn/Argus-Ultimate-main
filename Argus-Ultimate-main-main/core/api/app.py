"""Push 95 — FastAPI application for Argus dashboard.

Routes:
  GET  /health               liveness + version
  GET  /status               engine + risk + execution + bus stats
  GET  /metrics              Prometheus text exposition
  GET  /positions            open positions (all or ?symbol=X)
  GET  /orders               open orders   (all or ?symbol=X)
  GET  /signals              last N signals (?limit=50)
  POST /kill-switch          activate or reset kill switch
  GET  /backtest             run synthetic backtest, return summary
  -- Push 93 --
  GET  /regime               live regime snapshot
  POST /regime/scalars       hot-update a sizer scalar
  GET  /sizer                all 5 regime scalars + active
  GET  /bandit               per-strategy allocations
  WS   /ws/regime            push-only regime transition stream
  -- Push 94 --
  GET  /regime/history       last N regime transitions from ring-buffer
  GET  /regime/stats         aggregate stats over the ring-buffer
  -- Push 95 --
  GET  /alert-rules          list all configured alert thresholds
  POST /alert-rules          create or update an alert rule
  DELETE /alert-rules/{name} remove an alert rule by name
  WS   /ws/prices            real-time price feed
  WS   /ws/signals           real-time signal feed
  WS   /ws/risk              real-time risk event feed

All components are injected via AppContext dataclass
so the app is fully testable without real exchanges.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import PlainTextResponse
    from pydantic import BaseModel
    _FASTAPI = True
except ImportError:
    _FASTAPI = False
    class BaseModel:  # type: ignore
        pass

from core.api.prometheus import PrometheusRegistry
from core.api.ws_feed import (
    ConnectionManager,
    signal_to_ws_payload,
    risk_event_to_ws_payload,
    price_tick_to_ws_payload,
)
from core.api.models import (
    HealthResponse, StatusResponse, PositionModel, OrderModel,
    SignalModel, KillSwitchRequest, KillSwitchResponse,
    BacktestRequest,
    RegimeResponse, SizerSummaryResponse, BanditAllocationResponse,
    RegimeTransitionModel, RegimeHistoryResponse, RegimeStatsResponse,
    AlertRule, AlertRulesResponse, AlertRuleUpdateRequest,
)

# ---------------------------------------------------------------------------
# Default alert rules shipped with the bot
# ---------------------------------------------------------------------------

_DEFAULT_ALERT_RULES: List[Dict[str, Any]] = [
    {"name": "regime_dwell_min_secs",  "value": 60.0,   "enabled": True,  "description": "Alert if any regime lasts less than this many seconds"},
    {"name": "vol_spike_ratio",         "value": 3.0,    "enabled": True,  "description": "Alert when vol_ratio exceeds this threshold"},
    {"name": "max_transitions_per_hour","value": 20.0,   "enabled": True,  "description": "Alert if regime flips more than N times per hour"},
    {"name": "confidence_floor",        "value": 0.35,   "enabled": True,  "description": "Alert when detector confidence drops below this value"},
    {"name": "drawdown_pct",            "value": 5.0,    "enabled": True,  "description": "Alert when unrealised drawdown exceeds this % of equity"},
    {"name": "kill_switch_auto_pct",    "value": 10.0,   "enabled": False, "description": "Auto-trigger kill switch at this drawdown % (disabled by default)"},
]


@dataclass
class AppContext:
    """Dependency container injected into all route handlers."""
    engine:           Any = None   # ExecutionEngine
    order_manager:    Any = None   # OrderManager
    risk_manager:     Any = None   # RiskManager
    signal_bus:       Any = None   # AsyncSignalBus
    adapter:          Any = None   # ExchangeAdapter
    regime_detector:  Any = None   # RegimeDetector  (Push 93)
    regime_sizer:     Any = None   # RegimeAwareSizer (Push 93)
    bandit_router:    Any = None   # BanditRouter     (Push 93)
    regime_history:   Any = None   # RegimeHistoryBuffer (Push 94)
    alert_config:     Any = None   # dict[str, AlertRule] (Push 95)
    registry:     PrometheusRegistry = field(default_factory=PrometheusRegistry)
    ws_manager:   ConnectionManager  = field(default_factory=ConnectionManager)
    start_time:   float = field(default_factory=time.time)


def _default_alert_config() -> Dict[str, AlertRule]:
    return {
        r["name"]: AlertRule(**r)
        for r in _DEFAULT_ALERT_RULES
    }


_ctx: AppContext = AppContext()


def create_app(context: Optional[AppContext] = None) -> Any:
    """Factory — creates and returns the FastAPI app."""
    if not _FASTAPI:
        raise ImportError("fastapi required: pip install fastapi uvicorn")

    global _ctx
    if context:
        _ctx = context

    # Seed default alert rules if caller didn't provide any
    if _ctx.alert_config is None:
        _ctx.alert_config = _default_alert_config()

    app = FastAPI(
        title="Argus Ultimate",
        version="8.31.0",
        description="Algorithmic trading bot dashboard API",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            version="8.31.0",
            uptime_secs=round(time.time() - _ctx.start_time, 1),
            codename="AlertRules",
        )

    # ------------------------------------------------------------------
    # GET /status
    # ------------------------------------------------------------------

    @app.get("/status", response_model=StatusResponse)
    async def status():
        engine_stats = _ctx.engine.stats       if _ctx.engine        else {}
        risk_stats   = _ctx.risk_manager.stats if _ctx.risk_manager  else {}
        om_stats     = _ctx.order_manager.stats if _ctx.order_manager else {}
        bus_stats    = _ctx.signal_bus.stats   if _ctx.signal_bus    else {}
        _ctx.registry.update_from_engine(engine_stats)
        _ctx.registry.update_from_risk(risk_stats)
        _ctx.registry.update_from_om(om_stats)
        return StatusResponse(
            engine=engine_stats,
            risk=risk_stats,
            execution=om_stats,
            bus=bus_stats,
        )

    # ------------------------------------------------------------------
    # GET /metrics
    # ------------------------------------------------------------------

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        return _ctx.registry.text_exposition()

    # ------------------------------------------------------------------
    # GET /positions
    # ------------------------------------------------------------------

    @app.get("/positions", response_model=List[PositionModel])
    async def positions(symbol: Optional[str] = Query(None)):
        if _ctx.order_manager is None:
            return []
        stats  = _ctx.order_manager.stats
        result = []
        for sym, pos in stats.get("positions", {}).items():
            if symbol and sym != symbol:
                continue
            result.append(PositionModel(
                symbol=sym,
                side=pos.get("side", "FLAT"),
                qty=pos.get("qty", 0.0),
                avg_entry=pos.get("avg_entry", 0.0),
                realised_pnl=pos.get("realised_pnl", 0.0),
                unrealised_pnl=pos.get("unrealised_pnl", 0.0),
                notional=pos.get("notional", 0.0),
            ))
        return result

    # ------------------------------------------------------------------
    # GET /orders
    # ------------------------------------------------------------------

    @app.get("/orders", response_model=List[OrderModel])
    async def orders(symbol: Optional[str] = Query(None)):
        if _ctx.order_manager is None:
            return []
        open_orders = _ctx.order_manager.get_open_orders(symbol=symbol)
        return [
            OrderModel(
                order_id=o.order_id,
                symbol=o.symbol,
                side=o.side.value,
                type=o.order_type.value,
                qty=o.qty,
                price=o.price,
                status=o.status.value,
                filled_qty=o.filled_qty,
                avg_price=o.avg_fill_price,
                strategy_id=o.strategy_id,
            )
            for o in open_orders
        ]

    # ------------------------------------------------------------------
    # GET /signals
    # ------------------------------------------------------------------

    @app.get("/signals", response_model=List[SignalModel])
    async def signals(limit: int = Query(50, ge=1, le=500)):
        if _ctx.signal_bus is None:
            return []
        history = _ctx.signal_bus.history[-limit:]
        return [
            SignalModel(
                symbol=s.symbol,
                side=s.side.value,
                strength=s.strength,
                strategy_id=s.strategy_id,
                order_type=s.order_type,
                timestamp=s.timestamp,
            )
            for s in history
        ]

    # ------------------------------------------------------------------
    # POST /kill-switch
    # ------------------------------------------------------------------

    @app.post("/kill-switch", response_model=KillSwitchResponse)
    async def kill_switch(req: KillSwitchRequest):
        if _ctx.risk_manager is None:
            raise HTTPException(status_code=503, detail="Risk manager not initialised")
        if req.action == "activate":
            _ctx.risk_manager.activate_kill_switch(req.reason or "API request")
            return KillSwitchResponse(
                success=True, kill_switch_active=True,
                message="Kill switch activated",
            )
        elif req.action == "reset":
            _ctx.risk_manager.reset_kill_switch()
            return KillSwitchResponse(
                success=True, kill_switch_active=False,
                message="Kill switch reset",
            )
        else:
            raise HTTPException(status_code=400, detail="action must be 'activate' or 'reset'")

    # ------------------------------------------------------------------
    # GET /backtest
    # ------------------------------------------------------------------

    @app.get("/backtest")
    async def backtest(
        strategy: str = Query("momentum"),
        n_bars:   int = Query(300, ge=60,  le=2000),
        mc_sims:  int = Query(200, ge=50,  le=2000),
    ):
        from core.backtest.backtest_runner import BacktestRunner, BacktestConfig
        cfg     = BacktestConfig(
            strategy_name=strategy,
            mc_n_simulations=mc_sims,
            wf_n_splits=3,
            output_dir="/tmp/argus_api_reports",
        )
        runner  = BacktestRunner(cfg)
        prices  = BacktestRunner.generate_synthetic_prices(n=n_bars, seed=42)
        summary = runner.run(prices=prices)
        return {
            "strategy":    summary["strategy"],
            "n_bars":      summary["n_bars"],
            "metrics":     summary["metrics"],
            "walk_forward": summary.get("walk_forward"),
            "monte_carlo":  summary.get("monte_carlo"),
        }

    # ------------------------------------------------------------------
    # Push 93 — Regime / Sizer / Bandit
    # ------------------------------------------------------------------

    @app.get("/regime", response_model=RegimeResponse)
    async def regime():
        if _ctx.regime_detector is None:
            return RegimeResponse(
                regime="UNKNOWN", vol_ratio=0.0, trend_score=0.0,
                bb_pos=0.0, autocorr=0.0, confidence=0.0, regime_wired=False,
            )
        snap = _ctx.regime_detector.snapshot()
        return RegimeResponse(
            regime=snap.get("regime", "UNKNOWN"),
            vol_ratio=snap.get("vol_ratio", 0.0),
            trend_score=snap.get("trend_score", 0.0),
            bb_pos=snap.get("bb_pos", 0.0),
            autocorr=snap.get("autocorr", 0.0),
            confidence=snap.get("confidence", 0.0),
            regime_wired=True,
        )

    @app.post("/regime/scalars")
    async def update_scalar(regime: str, scalar: str, value: float):
        if _ctx.regime_sizer is None:
            return {"sizer_wired": False}
        _ctx.regime_sizer.set_scalar(regime, scalar, value)
        return {"sizer_wired": True, "regime": regime, "scalar": scalar, "value": value}

    @app.get("/sizer", response_model=SizerSummaryResponse)
    async def sizer():
        if _ctx.regime_sizer is None:
            return SizerSummaryResponse(scalars={}, active_regime=None, sizer_wired=False)
        return SizerSummaryResponse(
            scalars=_ctx.regime_sizer.scalars,
            active_regime=_ctx.regime_sizer.active_regime,
            sizer_wired=True,
        )

    @app.get("/bandit", response_model=BanditAllocationResponse)
    async def bandit(regime: Optional[str] = Query(None)):
        if _ctx.bandit_router is None:
            return BanditAllocationResponse(allocations={}, regime=None, bandit_wired=False)
        allocs = _ctx.bandit_router.allocations(regime=regime)
        return BanditAllocationResponse(
            allocations=allocs,
            regime=regime or _ctx.bandit_router.current_regime,
            bandit_wired=True,
        )

    # ------------------------------------------------------------------
    # Push 94 — Regime history
    # ------------------------------------------------------------------

    @app.get("/regime/history", response_model=RegimeHistoryResponse)
    async def regime_history(
        limit: int             = Query(50,  ge=1, le=500),
        since: Optional[float] = Query(None, description="Unix epoch — return only transitions at or after this timestamp"),
    ):
        if _ctx.regime_history is None:
            return RegimeHistoryResponse(
                transitions=[], count=0, buffer_maxlen=0, history_wired=False,
            )
        if since is not None:
            items = _ctx.regime_history.since(since)
        else:
            items = _ctx.regime_history.last_n(limit)
        transitions = [RegimeTransitionModel(**t.to_dict()) for t in items]
        return RegimeHistoryResponse(
            transitions=transitions,
            count=len(transitions),
            buffer_maxlen=_ctx.regime_history.maxlen,
            history_wired=True,
        )

    @app.get("/regime/stats", response_model=RegimeStatsResponse)
    async def regime_stats():
        if _ctx.regime_history is None:
            return RegimeStatsResponse(
                total_transitions=0, unique_regimes=[], regime_counts={},
                avg_duration_secs=None, min_duration_secs=None, max_duration_secs=None,
                current_regime=None, current_since=None, current_duration_secs=None,
                history_wired=False,
            )
        s = _ctx.regime_history.stats()
        return RegimeStatsResponse(**s.to_dict(), history_wired=True)

    # ------------------------------------------------------------------
    # Push 95 — Alert Rules
    # ------------------------------------------------------------------

    @app.get("/alert-rules", response_model=AlertRulesResponse)
    async def get_alert_rules():
        """Return all configured alert threshold rules."""
        cfg = _ctx.alert_config or {}
        rules = list(cfg.values()) if isinstance(cfg, dict) else []
        return AlertRulesResponse(
            rules=rules,
            count=len(rules),
            alerts_wired=True,
        )

    @app.post("/alert-rules", response_model=AlertRule)
    async def upsert_alert_rule(req: AlertRuleUpdateRequest):
        """Create or update an alert rule by name."""
        if _ctx.alert_config is None:
            _ctx.alert_config = {}
        rule = AlertRule(
            name=req.name,
            value=req.value,
            enabled=req.enabled,
        )
        _ctx.alert_config[req.name] = rule
        return rule

    @app.delete("/alert-rules/{name}")
    async def delete_alert_rule(name: str):
        """Remove an alert rule by name. No-op if not found."""
        cfg = _ctx.alert_config or {}
        existed = name in cfg
        cfg.pop(name, None)
        return {"name": name, "deleted": existed}

    # ------------------------------------------------------------------
    # WebSocket endpoints
    # ------------------------------------------------------------------

    @app.websocket("/ws/regime")
    async def ws_regime(websocket: WebSocket):
        """Push-only stream — fires only on regime transitions.

        Each message includes the full RegimeTransition payload
        (seq, from_regime, duration_secs, context).
        """
        await _ctx.ws_manager.connect("regime", websocket)
        try:
            last_seq = 0
            while True:
                if _ctx.regime_history is not None:
                    latest = _ctx.regime_history.latest
                    if latest is not None and latest.seq != last_seq:
                        last_seq = latest.seq
                        await websocket.send_json(latest.to_dict())
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            await _ctx.ws_manager.disconnect("regime", websocket)

    @app.websocket("/ws/prices")
    async def ws_prices(websocket: WebSocket):
        await _ctx.ws_manager.connect("prices", websocket)
        try:
            while True:
                data = await websocket.receive_text()
                await websocket.send_text(f'{{"subscribed": "{data}"}}')
        except WebSocketDisconnect:
            await _ctx.ws_manager.disconnect("prices", websocket)

    @app.websocket("/ws/signals")
    async def ws_signals(websocket: WebSocket):
        await _ctx.ws_manager.connect("signals", websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await _ctx.ws_manager.disconnect("signals", websocket)

    @app.websocket("/ws/risk")
    async def ws_risk(websocket: WebSocket):
        await _ctx.ws_manager.connect("risk", websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await _ctx.ws_manager.disconnect("risk", websocket)

    return app
