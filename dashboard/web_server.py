# pyright: reportMissingImports=false
"""
Institutional Dashboard Web Server
===================================
FastAPI server for real-time trading dashboard.

ENDPOINTS:
- GET  /api/status          - System status
- GET  /api/metrics         - Current metrics
- GET  /api/positions       - Open positions
- GET  /api/alerts          - Alert history
- GET  /api/learning        - Learning status
- POST /api/control/stop    - Emergency stop
- POST /api/control/resume  - Resume trading
- POST /api/control/pause   - Pause trading
- POST /api/control/close   - Close position
- POST /api/control/strategy - Enable/disable strategy
- WS   /ws/live             - Real-time updates via WebSocket

Usage:
    py dashboard/web_server.py
    # Dashboard at http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# FastAPI detection
try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    logger.warning("FastAPI not available. Install with: pip install fastapi uvicorn")


# ============================================================================
# Request/Response Models
# ============================================================================

class ControlRequest(BaseModel):
    """Control action request."""
    reason: str = "Manual control"
    symbol: Optional[str] = None
    strategy: Optional[str] = None
    value: Optional[float] = None


class ControlResponse(BaseModel):
    """Control action response."""
    status: str
    timestamp: str
    message: Optional[str] = None


# ============================================================================
# Dashboard HTML (embedded)
# ============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Argus Trading Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0a0a; color: #e0e0e0; }
        .header { background: #1a1a2e; padding: 15px 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #16213e; }
        .header h1 { color: #00ff88; font-size: 24px; }
        .system-status { display: flex; gap: 10px; align-items: center; }
        .status-badge { padding: 5px 15px; border-radius: 20px; font-weight: bold; }
        .status-running { background: #00ff88; color: #000; }
        .status-paused { background: #ffaa00; color: #000; }
        .status-emergency { background: #ff0044; color: #fff; animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .controls { display: flex; gap: 10px; }
        .control-btn { padding: 8px 16px; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; }
        .btn-emergency { background: #ff0044; color: white; }
        .btn-pause { background: #ffaa00; color: black; }
        .btn-resume { background: #00ff88; color: black; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; padding: 20px; }
        .card { background: #1a1a2e; border-radius: 10px; padding: 20px; border: 1px solid #16213e; }
        .card h3 { color: #00ff88; margin-bottom: 15px; font-size: 14px; text-transform: uppercase; }
        .metric { display: flex; justify-content: space-between; margin-bottom: 10px; }
        .metric-label { color: #888; }
        .metric-value { font-weight: bold; font-size: 18px; }
        .positive { color: #00ff88; }
        .negative { color: #ff0044; }
        .neutral { color: #888; }
        .alert { padding: 10px; margin-bottom: 8px; border-radius: 5px; font-size: 14px; }
        .alert-critical { background: rgba(255,0,68,0.2); border-left: 3px solid #ff0044; }
        .alert-warning { background: rgba(255,170,0,0.2); border-left: 3px solid #ffaa00; }
        .alert-info { background: rgba(0,255,136,0.1); border-left: 3px solid #00ff88; }
        .position-table { width: 100%; border-collapse: collapse; }
        .position-table th, .position-table td { padding: 10px; text-align: left; border-bottom: 1px solid #333; }
        .position-table th { color: #888; font-size: 12px; text-transform: uppercase; }
        .learning-bar { height: 8px; background: #333; border-radius: 4px; overflow: hidden; margin-top: 5px; }
        .learning-fill { height: 100%; background: linear-gradient(90deg, #00ff88, #00aaff); transition: width 0.5s; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔷 ARGUS Trading Dashboard</h1>
        <div class="system-status">
            <span id="status-badge" class="status-badge status-running">RUNNING</span>
            <span id="uptime" class="neutral">Uptime: 0h 0m</span>
        </div>
        <div class="controls">
            <button class="control-btn btn-pause" onclick="pauseTrading()">⏸ Pause</button>
            <button class="control-btn btn-resume" onclick="resumeTrading()">▶ Resume</button>
            <button class="control-btn btn-emergency" onclick="emergencyStop()">🛑 EMERGENCY STOP</button>
        </div>
    </div>
    
    <div class="grid">
        <!-- PnL Card -->
        <div class="card">
            <h3>💰 Performance</h3>
            <div class="metric"><span class="metric-label">Total P&L</span><span id="total-pnl" class="metric-value neutral">$0.00</span></div>
            <div class="metric"><span class="metric-label">Daily P&L</span><span id="daily-pnl" class="metric-value neutral">$0.00</span></div>
            <div class="metric"><span class="metric-label">Win Rate</span><span id="win-rate" class="metric-value">0%</span></div>
            <div class="metric"><span class="metric-label">Sharpe Ratio</span><span id="sharpe" class="metric-value">0.00</span></div>
        </div>
        
        <!-- Risk Card -->
        <div class="card">
            <h3>⚠️ Risk</h3>
            <div class="metric"><span class="metric-label">Current Drawdown</span><span id="drawdown" class="metric-value">0.00%</span></div>
            <div class="metric"><span class="metric-label">Max Drawdown</span><span id="max-drawdown" class="metric-value">0.00%</span></div>
            <div class="metric"><span class="metric-label">Open Positions</span><span id="positions" class="metric-value">0</span></div>
            <div class="metric"><span class="metric-label">Total Exposure</span><span id="exposure" class="metric-value">$0</span></div>
        </div>
        
        <!-- Learning Card -->
        <div class="card">
            <h3>🧠 Learning</h3>
            <div class="metric"><span class="metric-label">Learning Cycles</span><span id="learning-cycles" class="metric-value">0</span></div>
            <div class="metric"><span class="metric-label">Parameters Updated</span><span id="params-updated" class="metric-value">0</span></div>
            <div class="metric"><span class="metric-label">Quantum Signals</span><span id="quantum-signals" class="metric-value">0</span></div>
            <div class="metric"><span class="metric-label">Latency</span><span id="latency" class="metric-value">0ms</span></div>
        </div>
        
        <!-- Trading Card -->
        <div class="card">
            <h3>📊 Trading</h3>
            <div class="metric"><span class="metric-label">Total Trades</span><span id="total-trades" class="metric-value">0</span></div>
            <div class="metric"><span class="metric-label">Active Strategies</span><span id="strategies" class="metric-value">0</span></div>
            <div class="metric"><span class="metric-label">Signals/Second</span><span id="signals-sec" class="metric-value">0</span></div>
            <div class="metric"><span class="metric-label">Decisions/Second</span><span id="decisions-sec" class="metric-value">0</span></div>
        </div>
    </div>
    
    <!-- Positions Table -->
    <div class="card" style="margin: 20px;">
        <h3>📋 Open Positions</h3>
        <table class="position-table">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Size</th>
                    <th>Entry</th>
                    <th>Current</th>
                    <th>P&L</th>
                    <th>Action</th>
                </tr>
            </thead>
            <tbody id="positions-body">
                <tr><td colspan="7" class="neutral">No open positions</td></tr>
            </tbody>
        </table>
    </div>
    
    <!-- Alerts -->
    <div class="card" style="margin: 20px;">
        <h3>🔔 Recent Alerts</h3>
        <div id="alerts-container">
            <p class="neutral">No alerts</p>
        </div>
    </div>
    
    <script>
        const ws = new WebSocket('ws://' + window.location.host + '/ws/live');
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };
        
        ws.onclose = function() {
            document.getElementById('status-badge').textContent = 'DISCONNECTED';
            document.getElementById('status-badge').className = 'status-badge status-emergency';
        };
        
        function updateDashboard(data) {
            if (data.system) {
                updateSystemStatus(data.system);
            }
            if (data.metrics) {
                updateMetrics(data.metrics);
            }
            if (data.positions) {
                updatePositions(data.positions);
            }
            if (data.alerts) {
                updateAlerts(data.alerts);
            }
        }
        
        function updateSystemStatus(system) {
            const badge = document.getElementById('status-badge');
            badge.textContent = system.state.toUpperCase();
            badge.className = 'status-badge status-' + system.state;
            
            const hours = Math.floor(system.uptime_seconds / 3600);
            const mins = Math.floor((system.uptime_seconds % 3600) / 60);
            document.getElementById('uptime').textContent = 'Uptime: ' + hours + 'h ' + mins + 'm';
        }
        
        function updateMetrics(metrics) {
            document.getElementById('total-pnl').textContent = formatCurrency(metrics.total_pnl);
            document.getElementById('total-pnl').className = 'metric-value ' + (metrics.total_pnl >= 0 ? 'positive' : 'negative');
            
            document.getElementById('daily-pnl').textContent = formatCurrency(metrics.daily_pnl);
            document.getElementById('daily-pnl').className = 'metric-value ' + (metrics.daily_pnl >= 0 ? 'positive' : 'negative');
            
            document.getElementById('win-rate').textContent = (metrics.win_rate * 100).toFixed(1) + '%';
            document.getElementById('sharpe').textContent = metrics.sharpe_ratio.toFixed(2);
            document.getElementById('drawdown').textContent = (metrics.current_drawdown * 100).toFixed(2) + '%';
            document.getElementById('max-drawdown').textContent = (metrics.max_drawdown * 100).toFixed(2) + '%';
            document.getElementById('positions').textContent = metrics.open_positions;
            document.getElementById('exposure').textContent = formatCurrency(metrics.total_exposure);
            
            document.getElementById('learning-cycles').textContent = metrics.learning_cycles;
            document.getElementById('params-updated').textContent = metrics.parameters_updated;
            document.getElementById('quantum-signals').textContent = metrics.quantum_signals;
            document.getElementById('latency').textContent = metrics.avg_latency_ms.toFixed(2) + 'ms';
            
            document.getElementById('total-trades').textContent = metrics.total_trades;
        }
        
        function updatePositions(positions) {
            const tbody = document.getElementById('positions-body');
            const entries = Object.entries(positions);
            
            if (entries.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" class="neutral">No open positions</td></tr>';
                return;
            }
            
            tbody.innerHTML = entries.map(([symbol, pos]) => `
                <tr>
                    <td>${symbol}</td>
                    <td class="${pos.side === 'long' ? 'positive' : 'negative'}">${pos.side}</td>
                    <td>${pos.size.toFixed(4)}</td>
                    <td>$${pos.entry_price.toFixed(2)}</td>
                    <td>$${pos.current_price.toFixed(2)}</td>
                    <td class="${pos.unrealized_pnl >= 0 ? 'positive' : 'negative'}">${formatCurrency(pos.unrealized_pnl)}</td>
                    <td><button onclick="closePosition('${symbol}')" style="background:#ff0044;color:white;border:none;padding:5px 10px;border-radius:3px;cursor:pointer;">Close</button></td>
                </tr>
            `).join('');
        }
        
        function updateAlerts(alerts) {
            const container = document.getElementById('alerts-container');
            if (alerts.length === 0) {
                container.innerHTML = '<p class="neutral">No alerts</p>';
                return;
            }
            
            container.innerHTML = alerts.slice().reverse().slice(0, 10).map(alert => `
                <div class="alert alert-${alert.level}">
                    <strong>${alert.level.toUpperCase()}</strong> ${alert.message}
                    <span class="neutral" style="float:right;font-size:12px;">${new Date(alert.timestamp).toLocaleTimeString()}</span>
                </div>
            `).join('');
        }
        
        function formatCurrency(value) {
            const prefix = value >= 0 ? '$' : '-$';
            return prefix + Math.abs(value).toFixed(2);
        }
        
        function emergencyStop() {
            if (confirm('Are you sure you want to EMERGENCY STOP? This will close all positions.')) {
                fetch('/api/control/stop', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({reason: 'Manual emergency stop'}) });
            }
        }
        
        function pauseTrading() {
            fetch('/api/control/pause', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({reason: 'Manual pause'}) });
        }
        
        function resumeTrading() {
            fetch('/api/control/resume', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({reason: 'Manual resume'}) });
        }
        
        function closePosition(symbol) {
            if (confirm('Close position ' + symbol + '?')) {
                fetch('/api/control/close', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({symbol: symbol, reason: 'Manual close'}) });
            }
        }
    </script>
</body>
</html>
"""


# ============================================================================
# FastAPI Application
# ============================================================================

def create_app(dashboard=None) -> Any:
    """Create FastAPI application."""
    if not _HAS_FASTAPI:
        raise ImportError("FastAPI required. Install: pip install fastapi uvicorn")
    
    from dashboard.institutional_dashboard import get_dashboard, TradingMetrics, AlertLevel, Position
    
    app = FastAPI(title="Argus Trading Dashboard", version="1.0.0")
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Get dashboard instance
    dash = dashboard or get_dashboard()
    
    # WebSocket connections
    connections: List[WebSocket] = []
    
    @app.get("/", response_class=HTMLResponse)
    async def dashboard_page():
        """Serve dashboard HTML."""
        return DASHBOARD_HTML
    
    @app.get("/api/status")
    async def get_status():
        """Get system status."""
        return dash.get_dashboard_state()
    
    @app.get("/api/metrics")
    async def get_metrics():
        """Get current metrics."""
        state = dash.get_dashboard_state()
        return state.get("metrics", {})
    
    @app.get("/api/positions")
    async def get_positions():
        """Get open positions."""
        state = dash.get_dashboard_state()
        return state.get("positions", {})
    
    @app.get("/api/alerts")
    async def get_alerts(unacknowledged: bool = False):
        """Get alerts."""
        return dash.get_alerts(unacknowledged_only=unacknowledged)
    
    @app.get("/api/learning")
    async def get_learning_status():
        """Get learning system status."""
        return dash._get_learning_status()
    
    @app.get("/api/control")
    async def get_control_state():
        """Get control state."""
        return dash.get_control_state()
    
    @app.post("/api/control/stop", response_model=ControlResponse)
    async def emergency_stop(request: ControlRequest):
        """Emergency stop."""
        result = dash.emergency_stop(reason=request.reason)
        await broadcast({"type": "control", "action": "emergency_stop", "data": result})
        return ControlResponse(
            status="emergency_stop_activated",
            timestamp=datetime.now().isoformat(),
            message=f"Emergency stop: {request.reason}"
        )
    
    @app.post("/api/control/resume", response_model=ControlResponse)
    async def resume_trading(request: ControlRequest):
        """Resume trading."""
        result = dash.resume_trading(reason=request.reason)
        await broadcast({"type": "control", "action": "resume", "data": result})
        return ControlResponse(
            status="resumed",
            timestamp=datetime.now().isoformat(),
            message=f"Trading resumed: {request.reason}"
        )
    
    @app.post("/api/control/pause", response_model=ControlResponse)
    async def pause_trading(request: ControlRequest):
        """Pause trading."""
        result = dash.pause_trading(reason=request.reason)
        await broadcast({"type": "control", "action": "pause", "data": result})
        return ControlResponse(
            status="paused",
            timestamp=datetime.now().isoformat(),
            message=f"Trading paused: {request.reason}"
        )
    
    @app.post("/api/control/close", response_model=ControlResponse)
    async def close_position(request: ControlRequest):
        """Close position."""
        if not request.symbol:
            raise HTTPException(status_code=400, detail="Symbol required")
        result = dash.close_position(symbol=request.symbol, reason=request.reason)
        await broadcast({"type": "control", "action": "close_position", "data": result})
        return ControlResponse(
            status="closing",
            timestamp=datetime.now().isoformat(),
            message=f"Closing {request.symbol}"
        )
    
    @app.post("/api/control/close_all", response_model=ControlResponse)
    async def close_all_positions(request: ControlRequest):
        """Close all positions."""
        result = dash.close_all_positions(reason=request.reason)
        await broadcast({"type": "control", "action": "close_all", "data": result})
        return ControlResponse(
            status="closing_all",
            timestamp=datetime.now().isoformat(),
            message=f"Closing all positions: {request.reason}"
        )
    
    @app.post("/api/control/strategy", response_model=ControlResponse)
    async def control_strategy(request: ControlRequest):
        """Enable/disable strategy."""
        if not request.strategy:
            raise HTTPException(status_code=400, detail="Strategy name required")
        result = dash.disable_strategy(request.strategy, reason=request.reason)
        await broadcast({"type": "control", "action": "strategy", "data": result})
        return ControlResponse(
            status="updated",
            timestamp=datetime.now().isoformat(),
            message=f"Strategy {request.strategy} updated"
        )
    
    @app.websocket("/ws/live")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket for real-time updates."""
        await websocket.accept()
        connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(connections)}")
        
        try:
            while True:
                # Send periodic updates
                state = dash.get_dashboard_state()
                await websocket.send_json({"type": "state", "data": state})
                await asyncio.sleep(1.0)  # 1 second updates
        except WebSocketDisconnect:
            connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Total connections: {len(connections)}")
    
    async def broadcast(message: Dict[str, Any]):
        """Broadcast message to all WebSocket connections."""
        disconnected = []
        for conn in connections:
            try:
                await conn.send_json(message)
            except:
                disconnected.append(conn)
        for conn in disconnected:
            connections.remove(conn)
    
    return app


# ============================================================================
# Standalone Server
# ============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the dashboard server."""
    import uvicorn
    
    app = create_app()
    logger.info(f"Starting dashboard server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
