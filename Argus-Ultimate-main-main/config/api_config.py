"""
Argus API Configuration Loader
Loads API keys from .env file and provides them to all systems
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class APIConfig:
    """
    Centralized API configuration for Argus
    Loads from .env file and validates all required keys
    """
    
    def __init__(self, env_file: str = ".env"):
        # Load .env file
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"✅ Loaded API config from {env_file}")
        else:
            logger.warning(f"⚠️  No {env_file} file found. Using environment variables.")
        
        # Exchange APIs
        self.kraken_api_key = os.getenv('KRAKEN_API_KEY')
        self.kraken_api_secret = os.getenv('KRAKEN_API_SECRET')
        self.kraken_sandbox = os.getenv('KRAKEN_SANDBOX', 'true').lower() == 'true'
        
        self.coinbase_api_key = os.getenv('COINBASE_API_KEY')
        self.coinbase_api_secret = os.getenv('COINBASE_API_SECRET')
        self.coinbase_sandbox = os.getenv('COINBASE_SANDBOX', 'true').lower() == 'true'
        
        # Trading mode
        self.trading_mode = os.getenv('TRADING_MODE', 'paper')
        self.paper_initial_balance = float(os.getenv('PAPER_INITIAL_BALANCE', '10000.0'))
        
        # Risk management
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', '0.10'))
        self.max_drawdown = float(os.getenv('MAX_DRAWDOWN', '0.15'))
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', '0.02'))
        self.daily_loss_limit = float(os.getenv('DAILY_LOSS_LIMIT', '500.0'))
        
        # External data APIs
        self.coinglass_api_key = os.getenv('COINGLASS_API_KEY')
        self.whale_alert_api_key = os.getenv('WHALE_ALERT_API_KEY')
        self.cryptocompare_api_key = os.getenv('CRYPTOCOMPARE_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.glassnode_api_key = os.getenv('GLASSNODE_API_KEY')
        
        # Notifications
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.telegram_enabled = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'
        
        # Infrastructure
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', '6379'))
        
        # Quantum
        self.quantum_enabled = os.getenv('QUANTUM_ENABLED', 'true').lower() == 'true'
        self.quantum_backend = os.getenv('QUANTUM_BACKEND', 'numpy')
        
        # Validate
        self._validate()
    
    def _validate(self):
        """Validate critical configuration"""
        errors = []
        warnings = []
        
        # Check Kraken keys
        if not self.kraken_api_key or self.kraken_api_key == 'your_kraken_api_key_here':
            errors.append("KRAKEN_API_KEY not set")
        if not self.kraken_api_secret or self.kraken_api_secret == 'your_kraken_api_secret_here':
            errors.append("KRAKEN_API_SECRET not set")
        
        # Check trading mode
        if self.trading_mode not in ['paper', 'live', 'hybrid']:
            errors.append(f"Invalid TRADING_MODE: {self.trading_mode}")
        
        # Warnings for missing data sources
        if not self.coinglass_api_key:
            warnings.append("COINGLASS_API_KEY not set (funding rates unavailable)")
        if not self.whale_alert_api_key:
            warnings.append("WHALE_ALERT_API_KEY not set (whale tracking unavailable)")
        if not self.news_api_key:
            warnings.append("NEWS_API_KEY not set (sentiment analysis limited)")
        
        # Log results
        if errors:
            logger.error("❌ API Configuration Errors:")
            for error in errors:
                logger.error(f"   - {error}")
            logger.error("   Run: copy .env.example .env and fill in your keys")
        
        if warnings:
            logger.warning("⚠️  API Configuration Warnings:")
            for warning in warnings:
                logger.warning(f"   - {warning}")
        
        if not errors and not warnings:
            logger.info("✅ All API configuration valid")
        
        self.is_valid = len(errors) == 0
    
    def get_exchange_config(self, exchange: str = 'kraken') -> Dict[str, Any]:
        """Get configuration for specific exchange"""
        if exchange == 'kraken':
            return {
                'api_key': self.kraken_api_key,
                'api_secret': self.kraken_api_secret,
                'sandbox': self.kraken_sandbox,
                'trading_mode': self.trading_mode
            }
        elif exchange == 'coinbase':
            return {
                'api_key': self.coinbase_api_key,
                'api_secret': self.coinbase_api_secret,
                'sandbox': self.coinbase_sandbox,
                'trading_mode': self.trading_mode
            }
        else:
            raise ValueError(f"Unknown exchange: {exchange}")
    
    def get_risk_config(self) -> Dict[str, float]:
        """Get risk management configuration"""
        return {
            'max_position_size': self.max_position_size,
            'max_drawdown': self.max_drawdown,
            'stop_loss_pct': self.stop_loss_pct,
            'daily_loss_limit': self.daily_loss_limit
        }
    
    def get_data_sources_config(self) -> Dict[str, Optional[str]]:
        """Get external data source API keys"""
        return {
            'coinglass': self.coinglass_api_key,
            'whale_alert': self.whale_alert_api_key,
            'cryptocompare': self.cryptocompare_api_key,
            'news': self.news_api_key,
            'glassnode': self.glassnode_api_key
        }
    
    def get_trading_mode(self) -> str:
        """Get current trading mode"""
        return self.trading_mode
    
    def is_paper_trading(self) -> bool:
        """Check if in paper trading mode"""
        return self.trading_mode == 'paper'
    
    def is_live_trading(self) -> bool:
        """Check if in live trading mode"""
        return self.trading_mode == 'live'
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'trading_mode': self.trading_mode,
            'kraken_configured': bool(self.kraken_api_key and self.kraken_api_key != 'your_kraken_api_key_here'),
            'coinbase_configured': bool(self.coinbase_api_key and self.coinbase_api_key != 'your_coinbase_api_key_here'),
            'data_sources': {
                'coinglass': bool(self.coinglass_api_key),
                'whale_alert': bool(self.whale_alert_api_key),
                'cryptocompare': bool(self.cryptocompare_api_key),
                'news': bool(self.news_api_key),
                'glassnode': bool(self.glassnode_api_key)
            },
            'risk_limits': self.get_risk_config(),
            'telegram_enabled': self.telegram_enabled,
            'is_valid': self.is_valid
        }


# Global instance
_api_config: Optional[APIConfig] = None


def get_api_config() -> APIConfig:
    """Get the global API configuration instance"""
    global _api_config
    if _api_config is None:
        _api_config = APIConfig()
    return _api_config


def reload_config() -> APIConfig:
    """Reload API configuration from .env"""
    global _api_config
    _api_config = APIConfig()
    return _api_config
