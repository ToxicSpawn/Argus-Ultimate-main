#!/usr/bin/env python3
"""
Unified Configuration - Singleton Configuration Manager
=======================================================

Unified configuration management system with singleton pattern
for centralized access to all bot settings.
"""

import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading
import logging

logger = logging.getLogger(__name__)


class UnifiedConfig:
    """
    Unified configuration manager using singleton pattern

    Provides centralized access to all configuration settings
    with thread-safe singleton implementation.
    """

    _instance: Optional["UnifiedConfig"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "UnifiedConfig":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self._config: Dict[str, Any] = {}
        self._config_file = "config.json"
        self._env_prefix = "ARGUS_"

        # Load configuration
        self._load_config()

        logger.info("UnifiedConfig initialized (core/config_unified.py)")
        logger.warning(
            "NOTE: This config system (core/config_unified.py) is the SECONDARY config. "
            "The primary config is unified_trading_system.UnifiedConfig which loads unified_config.yaml. "
            "This JSON-based config should only be used for monitoring/dashboard settings. "
            "Trading parameters should be set in unified_config.yaml."
        )

    def _load_config(self) -> None:
        """Load configuration from multiple sources"""
        # Start with defaults
        self._config = self._get_defaults()

        # Load from file
        self._load_from_file()

        # Override with environment variables
        self._load_from_env()

        # Validate configuration
        self._validate_config()

    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            # Trading
            "initial_balance": 10000.0,
            "max_position_size": 0.1,
            "risk_per_trade": 0.01,
            "commission_rate": 0.001,
            "slippage": 0.0005,
            # DCA allocation splits across 3 levels (must sum to 1.0).
            # Level-1 (first entry) gets the largest slice; level-3 the smallest.
            # Override via config.json key "dca_levels_pct" or ARGUS_DCA_LEVELS_PCT env.
            "dca_levels_pct": [0.40, 0.35, 0.25],
            # Exchange
            "exchange": "binance",
            "testnet": True,
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "timeframes": ["1h", "4h", "1d"],
            # Strategies
            "active_strategies": ["rsi_divergence", "bollinger_breakout"],
            "strategy_weights": {"rsi_divergence": 0.6, "bollinger_breakout": 0.4},
            # Risk Management
            "max_drawdown": 0.15,
            "circuit_breaker_enabled": True,
            "volatility_filter_enabled": True,
            "max_consecutive_losses": 5,
            # Monitoring
            "enable_monitoring": True,
            "metrics_port": 8000,
            "log_level": "INFO",
            "alert_email": None,
            # Database
            "database_url": "sqlite:///data/bot.db",
            "backup_enabled": True,
            # Performance
            "max_cpu_usage": 80.0,
            "max_memory_usage": 85.0,
            "performance_monitoring": True,
        }
    def _load_from_file(self) -> None:
        """Load configuration from JSON file"""
        if Path(self._config_file).exists():
            try:
                with open(self._config_file, "r") as f:
                    file_config = json.load(f)
                self._merge_config(file_config)
                logger.info(f"Loaded configuration from {self._config_file}")
            except Exception as e:
                logger.warning(f"Could not load config file: {e}")

    def _load_from_env(self) -> None:
        """Load configuration from environment variables"""
        env_mappings = {
            f"{self._env_prefix}INITIAL_BALANCE": ("initial_balance", float),
            f"{self._env_prefix}MAX_POSITION_SIZE": ("max_position_size", float),
            f"{self._env_prefix}RISK_PER_TRADE": ("risk_per_trade", float),
            f"{self._env_prefix}COMMISSION_RATE": ("commission_rate", float),
            f"{self._env_prefix}EXCHANGE": ("exchange", str),
            f"{self._env_prefix}TESTNET": ("testnet", lambda x: x.lower() in ("true", "1", "yes")),
            f"{self._env_prefix}LOG_LEVEL": ("log_level", str),
            f"{self._env_prefix}METRICS_PORT": ("metrics_port", int),
            f"{self._env_prefix}DATABASE_URL": ("database_url", str),
        }

        for env_var, (config_key, converter) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    self._config[config_key] = converter(env_value)
                    logger.debug(f"Set {config_key} from environment")
                except Exception as e:
                    logger.warning(f"Could not convert {env_var}: {e}")

    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """Merge new configuration with existing"""
        self._config.update(new_config)

    def _validate_config(self) -> None:
        """Validate configuration values"""
        validators = {
            "initial_balance": lambda x: isinstance(x, (int, float)) and x > 0,
            "max_position_size": lambda x: isinstance(x, float) and 0 < x <= 1,
            "risk_per_trade": lambda x: isinstance(x, float) and 0 < x <= 0.1,
            "commission_rate": lambda x: isinstance(x, float) and x >= 0,
            "max_drawdown": lambda x: isinstance(x, float) and 0 < x <= 1,
            "metrics_port": lambda x: isinstance(x, int) and 1000 <= x <= 65535,
            "dca_levels_pct": lambda x: (
                isinstance(x, list)
                and len(x) == 3
                and all(isinstance(v, (int, float)) and v > 0 for v in x)
                and abs(sum(x) - 1.0) < 1e-6
            ),
        }

        invalid_keys = []
        for key, validator in validators.items():
            if key in self._config:
                try:
                    if not validator(self._config[key]):
                        invalid_keys.append(key)
                except Exception:
                    invalid_keys.append(key)

        if invalid_keys:
            logger.warning(f"Invalid configuration values for: {invalid_keys}")
            # Reset to defaults
            for key in invalid_keys:
                if key in self._get_defaults():
                    self._config[key] = self._get_defaults()[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self._config[key] = value
        logger.debug(f"Set configuration {key} = {value}")

    def save_config(self, file_path: Optional[str] = None) -> None:
        """Save current configuration to file"""
        save_path = file_path or self._config_file

        try:
            with open(save_path, "w") as f:
                json.dump(self._config, f, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Could not save configuration: {e}")

    def reload_config(self) -> None:
        """Reload configuration from sources"""
        self._load_config()
        logger.info("Configuration reloaded")

    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading-related configuration"""
        return {
            "initial_balance": self.get("initial_balance"),
            "max_position_size": self.get("max_position_size"),
            "risk_per_trade": self.get("risk_per_trade"),
            "commission_rate": self.get("commission_rate"),
            "slippage": self.get("slippage"),
        }

    def get_exchange_config(self) -> Dict[str, Any]:
        """Get exchange-related configuration"""
        return {
            "exchange": self.get("exchange"),
            "testnet": self.get("testnet"),
            "symbols": self.get("symbols"),
            "timeframes": self.get("timeframes"),
        }

    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration"""
        return {
            "max_drawdown": self.get("max_drawdown"),
            "circuit_breaker_enabled": self.get("circuit_breaker_enabled"),
            "volatility_filter_enabled": self.get("volatility_filter_enabled"),
            "max_consecutive_losses": self.get("max_consecutive_losses"),
        }

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return {
            "enable_monitoring": self.get("enable_monitoring"),
            "metrics_port": self.get("metrics_port"),
            "log_level": self.get("log_level"),
            "alert_email": self.get("alert_email"),
        }

    def get_strategy_config(self) -> Dict[str, Any]:
        """Get strategy configuration"""
        return {
            "active_strategies": self.get("active_strategies"),
            "strategy_weights": self.get("strategy_weights"),
        }

    def print_config(self) -> None:
        """Print current configuration"""
        logger.info("\n=== Unified Configuration ===")
        for key, value in sorted(self._config.items()):
            if any(secret in key.lower() for secret in ["secret", "key", "password"]):
                logger.info(f"{key}: ***HIDDEN***")
            else:
                logger.info(f"{key}: {value}")
    def get_all_config(self) -> Dict[str, Any]:
        """Get complete configuration"""
        return self._config.copy()

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values"""
        self._config.update(updates)
        self._validate_config()
        logger.info(f"Updated {len(updates)} configuration values")

    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults"""
        self._config = self._get_defaults()
        logger.info("Configuration reset to defaults")

    def validate_exchange_connection(self) -> bool:
        """Validate exchange connection configuration"""
        # This would check API keys, connection, etc.
        # For now, just basic validation
        exchange = self.get("exchange")
        if not exchange:
            return False

        # In a real implementation, you might test the connection
        return True

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance monitoring configuration"""
        return {
            "max_cpu_usage": self.get("max_cpu_usage"),
            "max_memory_usage": self.get("max_memory_usage"),
            "performance_monitoring": self.get("performance_monitoring"),
        }

    def get_dca_levels(self) -> List[float]:
        """Return DCA allocation splits. Always 3 elements summing to 1.0."""
        return self.get("dca_levels_pct", [0.40, 0.35, 0.25])


# Legacy convenience singleton expected by some older modules.
config = UnifiedConfig()
