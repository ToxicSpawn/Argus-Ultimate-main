#!/usr/bin/env python3
"""
Configuration Management - Centralized Settings
===============================================

Centralized configuration management for all bot components
with environment variable support and validation.
"""

import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Centralized configuration management

    Features:
    - Environment variable support
    - Configuration validation
    - Multiple configuration sources
    - Type checking and defaults
    """

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config.json"
        self._config: Dict[str, Any] = {}
        self._defaults = self._get_defaults()
        self._validators = self._get_validators()

        # Load configuration
        self.load_config()

        logger.info("Configuration loaded")

    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            # Trading parameters
            "initial_balance": 10000.0,
            "max_position_size": 0.1,
            "risk_per_trade": 0.01,
            "commission_rate": 0.001,
            # Exchange settings
            "exchange": "binance",
            "api_key": os.getenv("API_KEY", ""),
            "api_secret": os.getenv("API_SECRET", ""),
            "testnet": True,
            # Strategy settings
            "strategies": ["rsi_divergence", "bollinger_breakout"],
            "timeframe": "1h",
            "symbols": ["BTC/USDT", "ETH/USDT"],
            # Risk management
            "max_drawdown": 0.15,
            "circuit_breaker_enabled": True,
            "volatility_filter": True,
            # Logging
            "log_level": "INFO",
            "log_file": "logs/bot.log",
            # Performance monitoring
            "enable_monitoring": True,
            "metrics_port": 8000,
            # Database
            "database_url": "sqlite:///data/bot.db",
        }

    def _get_validators(self) -> Dict[str, callable]:
        """Get configuration validators"""
        return {
            "initial_balance": lambda x: isinstance(x, (int, float)) and x > 0,
            "max_position_size": lambda x: isinstance(x, float) and 0 < x <= 1,
            "risk_per_trade": lambda x: isinstance(x, float) and 0 < x <= 0.1,
            "commission_rate": lambda x: isinstance(x, float) and x >= 0,
            "max_drawdown": lambda x: isinstance(x, float) and 0 < x <= 1,
        }

    def load_config(self) -> None:
        """Load configuration from file and environment"""
        # Start with defaults
        self._config = self._defaults.copy()

        # Load from file if exists
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, "r") as f:
                    file_config = json.load(f)
                self._config.update(file_config)
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Could not load config file: {e}")

        # Override with environment variables
        self._load_from_env()

        # Validate configuration
        self._validate_config()

    def _load_from_env(self) -> None:
        """Load configuration from environment variables"""
        env_mappings = {
            "INITIAL_BALANCE": ("initial_balance", float),
            "MAX_POSITION_SIZE": ("max_position_size", float),
            "RISK_PER_TRADE": ("risk_per_trade", float),
            "COMMISSION_RATE": ("commission_rate", float),
            "EXCHANGE": ("exchange", str),
            "API_KEY": ("api_key", str),
            "API_SECRET": ("api_secret", str),
            "TESTNET": ("testnet", lambda x: x.lower() in ("true", "1", "yes")),
            "LOG_LEVEL": ("log_level", str),
            "MAX_DRAWDOWN": ("max_drawdown", float),
        }

        for env_var, (config_key, type_converter) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    self._config[config_key] = type_converter(env_value)
                    logger.debug(f"Set {config_key} from environment: {env_value}")
                except Exception as e:
                    logger.warning(f"Could not convert {env_var}={env_value}: {e}")

    def _validate_config(self) -> None:
        """Validate configuration values"""
        invalid_keys = []

        for key, validator in self._validators.items():
            if key in self._config:
                try:
                    if not validator(self._config[key]):
                        invalid_keys.append(key)
                        logger.warning(f"Invalid value for {key}: {self._config[key]}")
                except Exception as e:
                    invalid_keys.append(key)
                    logger.warning(f"Validation error for {key}: {e}")

        if invalid_keys:
            logger.warning(f"Invalid configuration keys: {invalid_keys}")
            # Reset invalid keys to defaults
            for key in invalid_keys:
                if key in self._defaults:
                    self._config[key] = self._defaults[key]
                    logger.info(f"Reset {key} to default: {self._defaults[key]}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        # Validate if validator exists
        if key in self._validators:
            if not self._validators[key](value):
                raise ValueError(f"Invalid value for {key}: {value}")

        self._config[key] = value
        logger.debug(f"Set configuration {key} = {value}")

    def save_config(self, file_path: Optional[str] = None) -> None:
        """Save current configuration to file"""
        save_path = file_path or self.config_file

        try:
            with open(save_path, "w") as f:
                json.dump(self._config, f, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Could not save configuration: {e}")

    def get_exchange_config(self) -> Dict[str, Any]:
        """Get exchange-specific configuration"""
        return {
            "exchange": self.get("exchange"),
            "api_key": self.get("api_key"),
            "api_secret": self.get("api_secret"),
            "testnet": self.get("testnet"),
            "commission_rate": self.get("commission_rate"),
        }

    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration"""
        return {
            "max_drawdown": self.get("max_drawdown"),
            "risk_per_trade": self.get("risk_per_trade"),
            "max_position_size": self.get("max_position_size"),
            "circuit_breaker_enabled": self.get("circuit_breaker_enabled"),
            "volatility_filter": self.get("volatility_filter"),
        }

    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration"""
        return {
            "initial_balance": self.get("initial_balance"),
            "strategies": self.get("strategies"),
            "timeframe": self.get("timeframe"),
            "symbols": self.get("symbols"),
        }

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return {
            "enable_monitoring": self.get("enable_monitoring"),
            "metrics_port": self.get("metrics_port"),
            "log_level": self.get("log_level"),
            "log_file": self.get("log_file"),
        }

    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            try:
                self.set(key, value)
            except Exception as e:
                logger.warning(f"Could not update {key}: {e}")

    def reset_to_defaults(self) -> None:
        """Reset all configuration to defaults"""
        self._config = self._defaults.copy()
        logger.info("Configuration reset to defaults")

    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration values"""
        return self._config.copy()

    def print_config(self) -> None:
        """Print current configuration"""
        logger.info("=== Current Configuration ===")
        for key, value in sorted(self._config.items()):
            if "secret" in key.lower() or "key" in key.lower():
                logger.info(f"{key}: ***HIDDEN***")
            else:
                logger.info(f"{key}: {value}")
    def validate_exchange_connection(self) -> bool:
        """Validate exchange connection configuration"""
        api_key = self.get("api_key")
        api_secret = self.get("api_secret")

        if not api_key or not api_secret:
            logger.warning("API key and secret not configured")
            return False

        # Basic validation - check length
        if len(api_key) < 10 or len(api_secret) < 10:
            logger.warning("API credentials seem too short")
            return False

        return True
