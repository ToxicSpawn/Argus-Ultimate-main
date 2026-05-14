"""
Hyperliquid L1 Exchange Connector
==================================
Features:
  - REST + WebSocket API
  - Maker rebate: -0.01% (exchange PAYS you per filled maker order)
  - Post-only order enforcement (never pays taker fees)
  - Wallet-based signing via eth_account
  - Full order lifecycle: place / cancel / status / positions
  - Graceful WS reconnect with exponential back-off
  - Rate-limit aware (1200 req/min)

Docs: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HL_REST_URL = "https://api.hyperliquid.xyz"
HL_WS_URL = "wss://api.hyperliquid.xyz/ws"
MAKER_REBATE_PCT = Decimal("-0.0001")  # -0.01% — exchange pays maker
TAKER_FEE_PCT = Decimal("0.00035")    # 0.035% taker (avoid at all costs)
RATE_LIMIT_RPS = 20  # conservative: 1200/min ÷ 60


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class HLOrder:
    symbol: str
    side: str           # "B" buy | "A" sell (Hyperliquid convention)
    size: Decimal
    price: Decimal
    order_id: Optional[str] = None
    status: str = "pending"
    filled_size: Decimal = Decimal("0")
    avg_fill_price: Decimal = Decimal("0")
    rebate_earned: Decimal = Decimal("0")
    timestamp: float = field(default_factory=time.time)


@dataclass
class HLPosition:
    symbol: str
    side: str
    size: Decimal
    entry_price: Decimal
    unrealised_pnl: Decimal
    margin_used: Decimal
    leverage: int


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class HyperliquidClient:
    """
    Async Hyperliquid exchange client.

    Usage::

        client = HyperliquidClient(wallet_address="0x...", private_key="0x...")
        await client.connect()
        order = await client.place_maker_order("BTC", "B", size=0.001, price=60000)
        await client.disconnect()
    """

    def __init__(
        self,
        wallet_address: str,
        private_key: str,
        testnet: bool = False,
        maker_only: bool = True,
    ) -> None:
        self.wallet_address = wallet_address.lower()
        self.private_key = private_key
        self.testnet = testnet
        self.maker_only = maker_only

        self._rest_url = (
            "https://api.hyperliquid-testnet.xyz" if testnet else HL_REST_URL
        )
        self._ws_url = (
            "wss://api.hyperliquid-testnet.xyz/ws" if testnet else HL_WS_URL
        )

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[Any] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._subscriptions: Dict[str, List[Callable]] = {}
        self._rate_limiter = asyncio.Semaphore(RATE_LIMIT_RPS)

        # Stats
        self.total_rebates_earned: Decimal = Decimal("0")
        self.total_fees_paid: Decimal = Decimal("0")
        self.orders_placed: int = 0
        self.orders_filled: int = 0

        # Try importing eth_account for signing
        try:
            from eth_account import Account
            from eth_account.messages import encode_defunct
            self._account = Account.from_key(private_key)
            self._encode_defunct = encode_defunct
            self._signing_available = True
        except ImportError:
            logger.warning(
                "eth_account not installed. Order signing unavailable. "
                "Install with: pip install eth-account"
            )
            self._signing_available = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        """Open aiohttp session and start WS listener."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={"Content-Type": "application/json"},
        )
        logger.info(
            "HyperliquidClient connected | testnet=%s maker_only=%s",
            self.testnet, self.maker_only,
        )

    async def disconnect(self) -> None:
        """Clean up session and WS."""
        if self._ws_task:
            self._ws_task.cancel()
        if self._session:
            await self._session.close()
        logger.info("HyperliquidClient disconnected")

    # ------------------------------------------------------------------
    # REST helpers
    # ------------------------------------------------------------------
    async def _post(self, endpoint: str, payload: dict) -> dict:
        async with self._rate_limiter:
            assert self._session is not None
            async with self._session.post(
                f"{self._rest_url}{endpoint}", json=payload
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def _post_info(self, payload: dict) -> dict:
        return await self._post("/info", payload)

    async def _post_exchange(self, action: dict) -> dict:
        """Sign and submit an exchange action."""
        if not self._signing_available:
            raise RuntimeError(
                "eth_account required for order placement. "
                "pip install eth-account"
            )
        nonce = int(time.time() * 1000)
        payload = {
            "action": action,
            "nonce": nonce,
            "signature": self._sign(action, nonce),
        }
        return await self._post("/exchange", payload)

    def _sign(self, action: dict, nonce: int) -> dict:
        """Sign action payload with wallet private key."""
        msg = json.dumps({"action": action, "nonce": nonce}, separators=(",", ":"))
        msg_hash = self._encode_defunct(text=msg)
        signed = self._account.sign_message(msg_hash)
        return {
            "r": hex(signed.r),
            "s": hex(signed.s),
            "v": signed.v,
        }

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    async def get_orderbook(self, symbol: str, depth: int = 20) -> dict:
        """Fetch L2 orderbook snapshot."""
        data = await self._post_info({
            "type": "l2Book",
            "coin": symbol,
            "nSigFigs": 5,
        })
        levels = data.get("levels", [[], []])
        return {
            "symbol": symbol,
            "bids": [(Decimal(p), Decimal(s)) for p, s in levels[0][:depth]],
            "asks": [(Decimal(p), Decimal(s)) for p, s in levels[1][:depth]],
            "timestamp": time.time(),
        }

    async def get_mid_price(self, symbol: str) -> Decimal:
        """Return mid price from L2 book."""
        book = await self.get_orderbook(symbol, depth=1)
        if book["bids"] and book["asks"]:
            return (book["bids"][0][0] + book["asks"][0][0]) / 2
        raise ValueError(f"Empty orderbook for {symbol}")

    async def get_positions(self) -> List[HLPosition]:
        """Fetch all open perpetual positions."""
        data = await self._post_info({
            "type": "clearinghouseState",
            "user": self.wallet_address,
        })
        positions = []
        for pos in data.get("assetPositions", []):
            p = pos.get("position", {})
            size = Decimal(str(p.get("szi", "0")))
            if size == 0:
                continue
            positions.append(HLPosition(
                symbol=p.get("coin", ""),
                side="long" if size > 0 else "short",
                size=abs(size),
                entry_price=Decimal(str(p.get("entryPx", "0"))),
                unrealised_pnl=Decimal(str(p.get("unrealizedPnl", "0"))),
                margin_used=Decimal(str(p.get("marginUsed", "0"))),
                leverage=int(p.get("leverage", {}).get("value", 1)),
            ))
        return positions

    async def get_account_balance(self) -> Decimal:
        """Return USDC account equity."""
        data = await self._post_info({
            "type": "clearinghouseState",
            "user": self.wallet_address,
        })
        return Decimal(str(
            data.get("crossMarginSummary", {}).get("accountValue", "0")
        ))

    async def get_open_orders(self) -> List[dict]:
        """Return all open orders for the wallet."""
        return await self._post_info({
            "type": "openOrders",
            "user": self.wallet_address,
        })

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------
    async def place_maker_order(
        self,
        symbol: str,
        side: str,  # "buy" | "sell"
        size: float,
        price: float,
        reduce_only: bool = False,
    ) -> HLOrder:
        """
        Place a post-only limit order (maker rebate guaranteed).
        If maker_only=True and order would immediately fill (cross spread),
        the order is rejected rather than paying taker fees.
        """
        # Validate we won't cross the spread
        if self.maker_only:
            book = await self.get_orderbook(symbol, depth=1)
            hl_side = "B" if side.lower() == "buy" else "A"
            price_dec = Decimal(str(price))
            if hl_side == "B" and book["asks"] and price_dec >= book["asks"][0][0]:
                raise ValueError(
                    f"Maker-only: buy price {price} would cross ask "
                    f"{book['asks'][0][0]} — order rejected to avoid taker fee"
                )
            if hl_side == "A" and book["bids"] and price_dec <= book["bids"][0][0]:
                raise ValueError(
                    f"Maker-only: sell price {price} would cross bid "
                    f"{book['bids'][0][0]} — order rejected to avoid taker fee"
                )

        hl_side = "B" if side.lower() == "buy" else "A"
        action = {
            "type": "order",
            "orders": [{
                "a": symbol,          # asset
                "b": hl_side == "B",  # isBuy
                "p": str(round(price, 6)),
                "s": str(round(size, 6)),
                "r": reduce_only,
                "t": {
                    "limit": {
                        "tif": "Alo"  # Add-Liquidity-Only = post-only maker
                    }
                },
            }],
            "grouping": "na",
        }

        resp = await self._post_exchange(action)
        status = resp.get("response", {}).get("data", {}).get("statuses", [{}])[0]
        order_id = str(status.get("resting", {}).get("oid", ""))

        order = HLOrder(
            symbol=symbol,
            side=hl_side,
            size=Decimal(str(size)),
            price=Decimal(str(price)),
            order_id=order_id,
            status="open" if order_id else "rejected",
        )
        self.orders_placed += 1
        logger.info(
            "Placed maker order | %s %s %.6f @ %.2f | id=%s",
            symbol, side, size, price, order_id,
        )
        return order

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order by ID."""
        action = {
            "type": "cancel",
            "cancels": [{"a": symbol, "o": int(order_id)}],
        }
        resp = await self._post_exchange(action)
        statuses = resp.get("response", {}).get("data", {}).get("statuses", [])
        success = statuses and statuses[0] == "success"
        logger.info("Cancel order %s | success=%s", order_id, success)
        return success

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders, optionally filtered by symbol."""
        open_orders = await self.get_open_orders()
        cancelled = 0
        for order in open_orders:
            sym = order.get("coin", "")
            oid = str(order.get("oid", ""))
            if symbol and sym != symbol:
                continue
            if await self.cancel_order(sym, oid):
                cancelled += 1
        return cancelled

    async def close_position(
        self, symbol: str, size: float, current_price: float
    ) -> HLOrder:
        """Close a position at market-proximate price via aggressive limit."""
        book = await self.get_orderbook(symbol, depth=1)
        # Use best bid/ask with tiny slippage buffer
        close_price = (
            float(book["bids"][0][0]) * 0.999  # closing long: sell just below bid
            if size > 0
            else float(book["asks"][0][0]) * 1.001  # closing short: buy just above ask
        )
        side = "sell" if size > 0 else "buy"
        return await self.place_maker_order(
            symbol, side, abs(size), close_price, reduce_only=True
        )

    # ------------------------------------------------------------------
    # Fee / rebate accounting
    # ------------------------------------------------------------------
    def calculate_rebate(self, filled_value: Decimal) -> Decimal:
        """Calculate rebate earned for a filled maker order."""
        rebate = filled_value * abs(MAKER_REBATE_PCT)
        self.total_rebates_earned += rebate
        return rebate

    def fee_summary(self) -> dict:
        return {
            "total_rebates_earned_usdc": float(self.total_rebates_earned),
            "total_fees_paid_usdc": float(self.total_fees_paid),
            "net_fee_pnl_usdc": float(self.total_rebates_earned - self.total_fees_paid),
            "orders_placed": self.orders_placed,
            "orders_filled": self.orders_filled,
            "maker_rebate_pct": float(MAKER_REBATE_PCT),
        }

    # ------------------------------------------------------------------
    # WebSocket subscription
    # ------------------------------------------------------------------
    async def subscribe_orderbook(
        self, symbol: str, callback: Callable[[dict], None]
    ) -> None:
        """Subscribe to live L2 orderbook updates via WebSocket."""
        key = f"l2Book:{symbol}"
        if key not in self._subscriptions:
            self._subscriptions[key] = []
        self._subscriptions[key].append(callback)

        if self._ws_task is None or self._ws_task.done():
            self._ws_task = asyncio.create_task(self._ws_listener())

    async def subscribe_trades(
        self, symbol: str, callback: Callable[[dict], None]
    ) -> None:
        """Subscribe to live trade stream."""
        key = f"trades:{symbol}"
        if key not in self._subscriptions:
            self._subscriptions[key] = []
        self._subscriptions[key].append(callback)

        if self._ws_task is None or self._ws_task.done():
            self._ws_task = asyncio.create_task(self._ws_listener())

    async def _ws_listener(self) -> None:
        """WebSocket listener with exponential back-off reconnect."""
        import websockets
        backoff = 1
        while True:
            try:
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    self._ws = ws
                    backoff = 1  # reset on successful connect
                    # Subscribe to all registered channels
                    for key in self._subscriptions:
                        sub_type, symbol = key.split(":", 1)
                        await ws.send(json.dumps({
                            "method": "subscribe",
                            "subscription": {
                                "type": sub_type,
                                "coin": symbol,
                            }
                        }))
                    async for raw in ws:
                        msg = json.loads(raw)
                        await self._dispatch_ws(msg)
            except Exception as exc:
                logger.warning(
                    "WS disconnected: %s. Reconnecting in %ds...", exc, backoff
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _dispatch_ws(self, msg: dict) -> None:
        """Route incoming WS message to registered callbacks."""
        channel = msg.get("channel", "")
        data = msg.get("data", {})
        coin = data.get("coin", "") if isinstance(data, dict) else ""

        # L2 book update
        if channel == "l2Book" and coin:
            key = f"l2Book:{coin}"
            for cb in self._subscriptions.get(key, []):
                try:
                    await asyncio.coroutine(cb)(data) if asyncio.iscoroutinefunction(cb) else cb(data)
                except Exception as e:
                    logger.error("l2Book callback error: %s", e)

        # Trade stream
        elif channel == "trades" and coin:
            key = f"trades:{coin}"
            for cb in self._subscriptions.get(key, []):
                try:
                    await asyncio.coroutine(cb)(data) if asyncio.iscoroutinefunction(cb) else cb(data)
                except Exception as e:
                    logger.error("trades callback error: %s", e)
