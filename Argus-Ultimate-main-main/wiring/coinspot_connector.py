"""
Coinspot Australia Connector
Local Australian exchange integration for Sydney traders
"""

import asyncio
import aiohttp
import hmac
import hashlib
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class CoinspotOrder:
    """Coinspot order structure"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: float
    status: str
    created_at: datetime


class CoinspotConnector:
    """
    Coinspot Australia API Connector
    
    Pros:
    - Australian owned & operated
    - Easy AUD deposits
    - Local customer support
    - Simple interface
    
    Cons:
    - Higher fees (0.1-1%)
    - Limited API for advanced trading
    - Lower liquidity than Kraken
    - No WebSocket (REST only)
    """
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://www.coinspot.com.au"
        self.api_url = "https://www.coinspot.com.au/api/v2"
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_connected = False
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
    async def connect(self):
        """Initialize connection to Coinspot"""
        self.session = aiohttp.ClientSession()
        
        try:
            # Test connection with balances
            await self.get_balances()
            self.is_connected = True
            logger.info("✅ Coinspot Australia connected")
            
        except Exception as e:
            logger.error(f"❌ Coinspot connection failed: {e}")
            raise
    
    async def disconnect(self):
        """Close connection"""
        if self.session:
            await self.session.close()
            self.is_connected = False
            logger.info("⏹️ Coinspot disconnected")
    
    def _sign_request(self, endpoint: str, params: Dict = None) -> str:
        """Sign Coinspot API request"""
        if params is None:
            params = {}
        
        # Add nonce
        params['nonce'] = str(int(datetime.now().timestamp() * 1000))
        
        # Create payload string
        payload = json.dumps(params, separators=(',', ':'))
        
        # Sign with HMAC-SHA512
        signature = hmac.new(
            self.api_secret.encode(),
            payload.encode(),
            hashlib.sha512
        ).hexdigest()
        
        return signature, params['nonce']
    
    async def _rate_limit(self):
        """Rate limiting"""
        import time
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    async def _post(self, endpoint: str, params: Dict = None) -> Dict:
        """Make POST request to Coinspot"""
        await self._rate_limit()
        
        if not self.session:
            raise Exception("Not connected")
        
        signature, nonce = self._sign_request(endpoint, params)
        
        headers = {
            'Content-Type': 'application/json',
            'key': self.api_key,
            'sign': signature
        }
        
        url = f"{self.api_url}{endpoint}"
        
        async with self.session.post(url, json=params, headers=headers) as response:
            return await response.json()
    
    async def _get_public(self, endpoint: str) -> Dict:
        """Make public GET request"""
        await self._rate_limit()
        
        if not self.session:
            raise Exception("Not connected")
        
        url = f"{self.api_url}{endpoint}"
        
        async with self.session.get(url) as response:
            return await response.json()
    
    async def get_balances(self) -> Dict[str, float]:
        """Get account balances"""
        try:
            response = await self._post('/my/balances')
            
            if response.get('status') == 'ok':
                balances = {}
                for coin, data in response.get('balances', {}).items():
                    if isinstance(data, dict):
                        balance = float(data.get('balance', 0))
                        if balance > 0:
                            balances[coin] = balance
                    else:
                        # Some coins return direct value
                        balances[coin] = float(data)
                
                return balances
            
            return {}
            
        except Exception as e:
            logger.error(f"Get balances error: {e}")
            return {}
    
    async def get_prices(self) -> Dict[str, Dict]:
        """Get current prices"""
        try:
            response = await self._get_public('/pub/latest')
            
            if response.get('status') == 'ok':
                prices = {}
                for coin, data in response.get('prices', {}).items():
                    prices[coin] = {
                        'bid': float(data.get('bid', 0)),
                        'ask': float(data.get('ask', 0)),
                        'last': float(data.get('last', 0))
                    }
                return prices
            
            return {}
            
        except Exception as e:
            logger.error(f"Get prices error: {e}")
            return {}
    
    async def submit_buy_order(
        self,
        coin: str,
        amount: float,
        price: Optional[float] = None,
        order_type: str = "market"
    ) -> Optional[CoinspotOrder]:
        """Submit buy order"""
        try:
            params = {
                'cointype': coin.upper(),
            }
            
            if order_type == "market":
                # Market order by amount
                params['amount'] = amount
                endpoint = '/my/buy'
            else:
                # Limit order
                params['amount'] = amount
                params['rate'] = price
                endpoint = '/my/buy'
            
            response = await self._post(endpoint, params)
            
            if response.get('status') == 'ok':
                order = CoinspotOrder(
                    order_id=response.get('id', 'unknown'),
                    symbol=f"{coin.upper()}/AUD",
                    side='buy',
                    amount=amount,
                    price=price or float(response.get('rate', 0)),
                    status='pending',
                    created_at=datetime.now()
                )
                
                logger.info(f"📤 Coinspot buy: {amount} {coin} @ {order.price}")
                return order
            else:
                error = response.get('message', 'Unknown error')
                logger.error(f"Buy order failed: {error}")
                return None
                
        except Exception as e:
            logger.error(f"Submit buy error: {e}")
            return None
    
    async def submit_sell_order(
        self,
        coin: str,
        amount: float,
        price: Optional[float] = None,
        order_type: str = "market"
    ) -> Optional[CoinspotOrder]:
        """Submit sell order"""
        try:
            params = {
                'cointype': coin.upper(),
            }
            
            if order_type == "market":
                params['amount'] = amount
                endpoint = '/my/sell'
            else:
                params['amount'] = amount
                params['rate'] = price
                endpoint = '/my/sell'
            
            response = await self._post(endpoint, params)
            
            if response.get('status') == 'ok':
                order = CoinspotOrder(
                    order_id=response.get('id', 'unknown'),
                    symbol=f"{coin.upper()}/AUD",
                    side='sell',
                    amount=amount,
                    price=price or float(response.get('rate', 0)),
                    status='pending',
                    created_at=datetime.now()
                )
                
                logger.info(f"📤 Coinspot sell: {amount} {coin} @ {order.price}")
                return order
            else:
                error = response.get('message', 'Unknown error')
                logger.error(f"Sell order failed: {error}")
                return None
                
        except Exception as e:
            logger.error(f"Submit sell error: {e}")
            return None
    
    async def get_order_history(self, limit: int = 100) -> List[Dict]:
        """Get order history"""
        try:
            response = await self._post('/my/orders/completed', {'limit': limit})
            
            if response.get('status') == 'ok':
                return response.get('orders', [])
            
            return []
            
        except Exception as e:
            logger.error(f"Get order history error: {e}")
            return []
    
    async def get_deposit_address(self, coin: str) -> Optional[str]:
        """Get deposit address for coin"""
        try:
            response = await self._post('/my/coin/deposit', {'cointype': coin.upper()})
            
            if response.get('status') == 'ok':
                return response.get('address')
            
            return None
            
        except Exception as e:
            logger.error(f"Get deposit address error: {e}")
            return None
    
    def get_fees(self) -> Dict:
        """Get Coinspot fee structure"""
        return {
            'maker': 0.001,  # 0.1% (market maker)
            'taker': 0.001,  # 0.1% (market taker)
            'instant_buy': 0.01,  # 1% (instant buy/sell)
            'withdrawal_aud': 0,  # Free AUD withdrawals
            'deposit_aud': 0,     # Free AUD deposits
        }


class CoinspotManager:
    """Manages Coinspot connection for Sydney traders"""
    
    def __init__(self):
        self.connector: Optional[CoinspotConnector] = None
        
    async def connect(self, api_key: str, api_secret: str):
        """Connect to Coinspot"""
        self.connector = CoinspotConnector(api_key, api_secret)
        await self.connector.connect()
    
    async def get_balances(self) -> Dict[str, float]:
        """Get all balances"""
        if self.connector:
            return await self.connector.get_balances()
        return {}
    
    async def get_prices(self) -> Dict[str, Dict]:
        """Get current prices"""
        if self.connector:
            return await self.connector.get_prices()
        return {}
    
    async def buy(self, coin: str, amount_aud: float) -> Optional[CoinspotOrder]:
        """Buy crypto with AUD"""
        if not self.connector:
            return None
        
        # Get current price
        prices = await self.get_prices()
        coin_data = prices.get(coin.upper(), {})
        ask = coin_data.get('ask', 0)
        
        if ask == 0:
            logger.error(f"Cannot get price for {coin}")
            return None
        
        # Calculate amount
        amount_coin = amount_aud / ask
        
        return await self.connector.submit_buy_order(coin, amount_coin)
    
    async def sell(self, coin: str, amount_coin: float) -> Optional[CoinspotOrder]:
        """Sell crypto for AUD"""
        if not self.connector:
            return None
        
        return await self.connector.submit_sell_order(coin, amount_coin)


# Global instance
_coinspot_manager: Optional[CoinspotManager] = None


def get_coinspot_manager() -> CoinspotManager:
    """Get singleton Coinspot manager"""
    global _coinspot_manager
    if _coinspot_manager is None:
        _coinspot_manager = CoinspotManager()
    return _coinspot_manager


async def init_coinspot(api_key: str, api_secret: str):
    """Initialize Coinspot connection"""
    manager = get_coinspot_manager()
    await manager.connect(api_key, api_secret)
    logger.info("✅ Coinspot Australia initialized")
    return manager
