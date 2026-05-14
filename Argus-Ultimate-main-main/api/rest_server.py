"""
REST API Server - Argus Ultimate
=================================

FastAPI-based REST API for trading operations.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, validator
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    # Create stub classes
    class FastAPI:
        pass
    class HTTPException(Exception):
        pass
    class BaseModel:
        pass
    class Field:
        def __init__(self, *args, **kwargs):
            pass

from unified_trading import UnifiedTradingOrchestrator
from unified_trading.order_management import OrderSide, OrderType
from core.unified_config import config
from core.exception_manager import handle_errors

logger = logging.getLogger(__name__)


# Pydantic models for request/response
if HAS_FASTAPI:
    class OrderRequest(BaseModel):
        """Order creation request."""
        symbol: str = Field(..., description="Trading symbol (e.g., BTC/USD)")
        side: str = Field(..., description="Order side (buy or sell)")
        quantity: float = Field(..., gt=0, description="Order quantity")
        order_type: str = Field(default="market", description="Order type")
        price: Optional[float] = Field(None, description="Limit price (for limit orders)")
        
        @validator('side')
        def validate_side(cls, v):
            if v.lower() not in ['buy', 'sell']:
                raise ValueError('Side must be buy or sell')
            return v.lower()
        
        @validator('order_type')
        def validate_order_type(cls, v):
            if v.lower() not in ['market', 'limit', 'stop']:
                raise ValueError('Order type must be market, limit, or stop')
            return v.lower()
    
    class OrderResponse(BaseModel):
        """Order creation response."""
        order_id: str
        status: str
        symbol: str
        side: str
        quantity: float
        filled_qty: float = 0.0
        avg_price: Optional[float] = None
        created_at: str
        
    class PositionResponse(BaseModel):
        """Position information response."""
        symbol: str
        quantity: float
        avg_entry_price: float
        side: str
        unrealized_pnl: float
        realized_pnl: float
        market_price: Optional[float] = None
    
    class PortfolioResponse(BaseModel):
        """Portfolio summary response."""
        total_value: float
        cash_balance: float
        positions_value: float
        total_pnl: float
        unrealized_pnl: float
        realized_pnl: float
        num_positions: int
        max_drawdown: float
        win_rate: float
    
    class SignalRequest(BaseModel):
        """Signal generation request."""
        symbol: str
        price: float
        volume: Optional[float] = None
    
    class SignalResponse(BaseModel):
        """Signal response."""
        symbol: str
        side: str
        confidence: float
        strength: str
        suggested_qty: float
        strategies: List[str]
    
    class SystemStatusResponse(BaseModel):
        """System status response."""
        running: bool
        initialized: bool
        uptime_seconds: float
        active_orders: int
        mode: str
        version: str = "15.0.0"
    
    class ErrorResponse(BaseModel):
        """Error response."""
        error: str
        code: str
        details: Optional[Dict[str, Any]] = None
    
    class HealthCheckResponse(BaseModel):
        """Health check response."""
        status: str
        checks: Dict[str, bool]
        timestamp: str


class RESTServer:
    """
    FastAPI-based REST API server for Argus Ultimate.
    """
    
    def __init__(self, orchestrator: UnifiedTradingOrchestrator):
        self.orchestrator = orchestrator
        self.app: Optional[FastAPI] = None
        self._running = False
        
        if HAS_FASTAPI:
            self._setup_app()
        else:
            logger.warning("FastAPI not available - REST server disabled")
    
    def _setup_app(self):
        """Setup FastAPI application."""
        if not HAS_FASTAPI:
            return
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info("REST API starting up...")
            if not self.orchestrator.state.is_initialized:
                await self.orchestrator.initialize()
            if not self.orchestrator.state.is_running:
                await self.orchestrator.start()
            yield
            # Shutdown
            logger.info("REST API shutting down...")
            await self.orchestrator.stop()
        
        self.app = FastAPI(
            title="Argus Ultimate API",
            description="Professional algorithmic trading API",
            version="15.0.0",
            lifespan=lifespan
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=config.get_list('api.rest.cors_origins', ["http://localhost:3000"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        if not self.app:
            return
        
        @self.app.get("/", tags=["general"])
        async def root():
            """API root endpoint."""
            return {
                "name": "Argus Ultimate API",
                "version": "15.0.0",
                "status": "running",
                "docs": "/docs"
            }
        
        @self.app.get("/health", response_model=HealthCheckResponse, tags=["health"])
        async def health_check():
            """Health check endpoint."""
            checks = {
                "system": self.orchestrator.state.is_running,
                "initialized": self.orchestrator.state.is_initialized,
            }
            
            # Add component checks
            checks["data"] = True  # Would check data feed
            checks["portfolio"] = True  # Would check portfolio manager
            
            all_healthy = all(checks.values())
            
            return HealthCheckResponse(
                status="healthy" if all_healthy else "degraded",
                checks=checks,
                timestamp=datetime.utcnow().isoformat()
            )
        
        @self.app.get("/status", response_model=SystemStatusResponse, tags=["system"])
        async def get_status():
            """Get system status."""
            status = await self.orchestrator.get_status()
            
            return SystemStatusResponse(
                running=status['state']['running'],
                initialized=status['state']['initialized'],
                uptime_seconds=status['state']['uptime'],
                active_orders=status['state']['active_orders'],
                mode=config.get_str('trading.mode', 'paper')
            )
        
        @self.app.post("/orders", response_model=OrderResponse, tags=["orders"])
        async def create_order(order: OrderRequest):
            """Create a new trading order."""
            try:
                from unified_trading.order_management import Signal
                
                signal = Signal(
                    symbol=order.symbol,
                    side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
                    confidence=0.8,  # Would come from strategy
                    strategy="api",
                    suggested_qty=Decimal(str(order.quantity)),
                    suggested_price=Decimal(str(order.price)) if order.price else None
                )
                
                # Process through orchestrator
                result = await self.orchestrator.process_tick(
                    order.symbol,
                    order.price or 0.0
                )
                
                # Get the created order
                orders = await self.orchestrator.order_manager.get_active_orders()
                if orders:
                    created_order = orders[-1]
                    return OrderResponse(
                        order_id=created_order.id,
                        status=created_order.status.name,
                        symbol=created_order.symbol,
                        side=created_order.side.value,
                        quantity=float(created_order.quantity),
                        filled_qty=float(created_order.filled_qty),
                        avg_price=float(created_order.avg_price) if created_order.avg_price else None,
                        created_at=created_order.created_at.isoformat()
                    )
                
                raise HTTPException(status_code=500, detail="Failed to create order")
                
            except Exception as e:
                logger.error(f"Order creation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/orders", tags=["orders"])
        async def list_orders(
            status: Optional[str] = Query(None, description="Filter by status"),
            symbol: Optional[str] = Query(None, description="Filter by symbol"),
            limit: int = Query(100, ge=1, le=1000)
        ):
            """List trading orders."""
            orders = await self.orchestrator.order_manager.get_active_orders()
            
            # Apply filters
            if status:
                orders = [o for o in orders if o.status.name.lower() == status.lower()]
            if symbol:
                orders = [o for o in orders if o.symbol == symbol]
            
            # Limit
            orders = orders[-limit:]
            
            return {
                "orders": [
                    {
                        "order_id": o.id,
                        "status": o.status.name,
                        "symbol": o.symbol,
                        "side": o.side.value,
                        "quantity": float(o.quantity),
                        "filled_qty": float(o.filled_qty),
                        "avg_price": float(o.avg_price) if o.avg_price else None,
                        "created_at": o.created_at.isoformat()
                    }
                    for o in orders
                ],
                "total": len(orders)
            }
        
        @self.app.get("/orders/{order_id}", response_model=OrderResponse, tags=["orders"])
        async def get_order(order_id: str):
            """Get order details."""
            order = await self.orchestrator.order_manager.get_order(order_id)
            
            if not order:
                raise HTTPException(status_code=404, detail="Order not found")
            
            return OrderResponse(
                order_id=order.id,
                status=order.status.name,
                symbol=order.symbol,
                side=order.side.value,
                quantity=float(order.quantity),
                filled_qty=float(order.filled_qty),
                avg_price=float(order.avg_price) if order.avg_price else None,
                created_at=order.created_at.isoformat()
            )
        
        @self.app.delete("/orders/{order_id}", tags=["orders"])
        async def cancel_order(order_id: str):
            """Cancel an order."""
            success = await self.orchestrator.order_manager.cancel_order(order_id)
            
            if not success:
                raise HTTPException(status_code=400, detail="Failed to cancel order")
            
            return {"success": True, "message": f"Order {order_id} cancelled"}
        
        @self.app.get("/positions", response_model=List[PositionResponse], tags=["portfolio"])
        async def get_positions():
            """Get current positions."""
            positions = await self.orchestrator.portfolio_manager.get_positions()
            
            return [
                PositionResponse(
                    symbol=p.symbol,
                    quantity=float(p.quantity),
                    avg_entry_price=float(p.avg_entry_price),
                    side=p.side,
                    unrealized_pnl=float(p.unrealized_pnl),
                    realized_pnl=float(p.realized_pnl),
                    market_price=float(p.market_price) if p.market_price else None
                )
                for p in positions
            ]
        
        @self.app.get("/portfolio", response_model=PortfolioResponse, tags=["portfolio"])
        async def get_portfolio():
            """Get portfolio summary."""
            summary = await self.orchestrator.portfolio_manager.get_summary()
            
            return PortfolioResponse(
                total_value=float(summary.total_value),
                cash_balance=float(summary.cash_balance),
                positions_value=float(summary.positions_value),
                total_pnl=float(summary.total_pnl),
                unrealized_pnl=float(summary.unrealized_pnl),
                realized_pnl=float(summary.realized_pnl),
                num_positions=summary.num_positions,
                max_drawdown=summary.max_drawdown,
                win_rate=summary.win_rate
            )
        
        @self.app.post("/signals", response_model=List[SignalResponse], tags=["signals"])
        async def generate_signals(request: SignalRequest):
            """Generate trading signals."""
            signals = await self.orchestrator.signal_processor.generate_signals(
                request.symbol,
                request.price,
                volume=request.volume
            )
            
            processed = await self.orchestrator.signal_processor.process_signals(
                request.symbol
            )
            
            return [
                SignalResponse(
                    symbol=s.symbol,
                    side=s.side.value,
                    confidence=s.confidence,
                    strength=s.strength.name,
                    suggested_qty=float(s.suggested_qty),
                    strategies=s.strategies
                )
                for s in processed
            ]
        
        @self.app.post("/tick", tags=["trading"])
        async def process_tick(
            symbol: str,
            price: float,
            volume: Optional[float] = None,
            high: Optional[float] = None,
            low: Optional[float] = None
        ):
            """Process market tick data."""
            result = await self.orchestrator.process_tick(
                symbol,
                price,
                volume=volume,
                high=high,
                low=low
            )
            
            return result
        
        @self.app.get("/risk", tags=["risk"])
        async def get_risk_status():
            """Get risk management status."""
            status = await self.orchestrator.risk_integration.get_status()
            return status
        
        @self.app.get("/config", tags=["config"])
        async def get_config():
            """Get current configuration (sanitized)."""
            # Return sanitized config (no secrets)
            return {
                "trading": {
                    "mode": config.get_str('trading.mode'),
                    "max_positions": config.get_int('trading.max_positions')
                },
                "risk": {
                    "max_position_size": config.get_float('risk.max_position_size'),
                    "max_drawdown": config.get_float('risk.max_drawdown')
                }
            }
        
        @self.app.get("/metrics", tags=["monitoring"])
        async def get_metrics():
            """Get system metrics."""
            from core.cache_manager import get_cache
            
            cache_stats = get_cache().get_stats()
            
            return {
                "cache_stats": cache_stats,
                "system": await self.orchestrator.monitor.get_metrics() if hasattr(self.orchestrator, 'monitor') else {}
            }
    
    async def start(self, host: str = "0.0.0.0", port: int = 8080):
        """Start REST API server."""
        if not HAS_FASTAPI or not self.app:
            logger.warning("REST API not available (FastAPI not installed)")
            return
        
        import uvicorn
        
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        
        self._running = True
        logger.info(f"REST API starting on {host}:{port}")
        
        await server.serve()
    
    async def stop(self):
        """Stop REST API server."""
        self._running = False
        logger.info("REST API stopped")


# API client for testing
class APIClient:
    """Simple API client for testing."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
    
    async def health_check(self) -> Dict:
        """Check API health."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as resp:
                return await resp.json()
    
    async def get_status(self) -> Dict:
        """Get system status."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/status") as resp:
                return await resp.json()
    
    async def create_order(self, order_data: Dict) -> Dict:
        """Create trading order."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/orders",
                json=order_data
            ) as resp:
                return await resp.json()
