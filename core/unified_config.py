"""
Unified Configuration Manager
=============================

Single source of truth for all Argus configuration.
Replaces multiple config files with unified system.yaml.

Usage:
    from core.unified_config import config
    
    # Get configuration value
    trading_mode = config.get('trading.mode')
    
    # Get with default
    timeout = config.get('execution.timeout', 30)
    
    # Get nested dict
    risk_limits = config.get('risk')
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConfigSource:
    """Configuration source information."""
    name: str
    priority: int
    path: Optional[Path] = None


class UnifiedConfig:
    """
    Unified configuration manager.
    
    Configuration precedence (highest to lowest):
    1. ARGUS_* environment variables
    2. config/local.yaml (gitignored, user-specific)
    3. config/system.yaml (this file)
    4. Default values in code
    """
    
    def __init__(self, config_path: str = "config/system.yaml"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._sources: List[ConfigSource] = []
        self._load_all()
    
    def _load_all(self):
        """Load configuration from all sources."""
        # 1. Load base configuration (lowest priority)
        self._load_base_config()
        
        # 2. Override with local configuration
        self._load_local_config()
        
        # 3. Override with environment variables (highest priority)
        self._load_environment_variables()
        
        logger.info("Configuration loaded successfully")
    
    def _load_base_config(self):
        """Load base configuration from system.yaml."""
        if not self.config_path.exists():
            logger.warning(f"Base config not found: {self.config_path}")
            return
        
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            
            self._sources.append(ConfigSource(
                name="system.yaml",
                priority=1,
                path=self.config_path
            ))
            
            logger.info(f"Loaded base config: {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load base config: {e}")
    
    def _load_local_config(self):
        """Override with local configuration if exists."""
        local_path = self.config_path.parent / "local.yaml"
        
        if not local_path.exists():
            return
        
        try:
            with open(local_path, 'r') as f:
                local_config = yaml.safe_load(f)
            
            if local_config:
                self._deep_merge(self._config, local_config)
                
                self._sources.append(ConfigSource(
                    name="local.yaml",
                    priority=2,
                    path=local_path
                ))
                
                logger.info(f"Loaded local config: {local_path}")
                
        except Exception as e:
            logger.error(f"Failed to load local config: {e}")
    
    def _load_environment_variables(self):
        """Override with ARGUS_* environment variables."""
        env_config = {}
        
        for key, value in os.environ.items():
            if key.startswith('ARGUS_'):
                # Convert ARGUS_TRADING_MODE to trading.mode
                config_key = key[6:].lower().replace('_', '.')
                self._set_by_path(env_config, config_key, self._convert_value(value))
        
        if env_config:
            self._deep_merge(self._config, env_config)
            
            self._sources.append(ConfigSource(
                name="environment",
                priority=3
            ))
            
            logger.info("Loaded environment variables")
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot path.
        
        Args:
            path: Dot-separated path (e.g., 'trading.mode')
            default: Default value if not found
            
        Returns:
            Configuration value or default
            
        Examples:
            >>> config.get('trading.mode')
            'paper'
            
            >>> config.get('risk.max_position_size')
            0.1
            
            >>> config.get('nonexistent.key', 'default')
            'default'
        """
        keys = path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_dict(self, path: str) -> Dict[str, Any]:
        """Get configuration section as dictionary."""
        value = self.get(path, {})
        return value if isinstance(value, dict) else {}
    
    def get_list(self, path: str) -> List[Any]:
        """Get configuration value as list."""
        value = self.get(path, [])
        return value if isinstance(value, list) else []
    
    def get_str(self, path: str, default: str = "") -> str:
        """Get configuration value as string."""
        value = self.get(path, default)
        return str(value) if value is not None else default
    
    def get_int(self, path: str, default: int = 0) -> int:
        """Get configuration value as integer."""
        value = self.get(path, default)
        try:
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def get_float(self, path: str, default: float = 0.0) -> float:
        """Get configuration value as float."""
        value = self.get(path, default)
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def get_bool(self, path: str, default: bool = False) -> bool:
        """Get configuration value as boolean."""
        value = self.get(path, default)
        
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            return value.lower() in ('true', 'yes', '1', 'on')
        
        return bool(value) if value is not None else default
    
    def set(self, path: str, value: Any):
        """
        Set configuration value at runtime.
        
        Note: This only affects the current process and doesn't persist.
        """
        self._set_by_path(self._config, path, value)
        logger.debug(f"Config set: {path} = {value}")
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration as dictionary."""
        return self._config.copy()
    
    def get_sources(self) -> List[ConfigSource]:
        """Get list of configuration sources."""
        return self._sources.copy()
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return errors.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check required fields
        required_paths = [
            'trading.mode',
            'trading.initial_balance',
            'risk.max_position_size',
            'risk.max_drawdown'
        ]
        
        for path in required_paths:
            if self.get(path) is None:
                errors.append(f"Required configuration missing: {path}")
        
        # Validate trading mode
        trading_mode = self.get('trading.mode')
        if trading_mode not in ['paper', 'live', 'hybrid']:
            errors.append(f"Invalid trading mode: {trading_mode}")
        
        # Validate risk limits
        max_position = self.get_float('risk.max_position_size')
        if max_position <= 0 or max_position > 1:
            errors.append(f"Invalid max_position_size: {max_position}")
        
        max_drawdown = self.get_float('risk.max_drawdown')
        if max_drawdown <= 0 or max_drawdown > 1:
            errors.append(f"Invalid max_drawdown: {max_drawdown}")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0
    
    def reload(self):
        """Reload configuration from all sources."""
        self._config = {}
        self._sources = []
        self._load_all()
        logger.info("Configuration reloaded")
    
    def _deep_merge(self, base: Dict, override: Dict):
        """Deep merge override into base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _set_by_path(self, config: Dict, path: str, value: Any):
        """Set value by dot path."""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # Return as string
        return value


# Global configuration instance
config = UnifiedConfig()


def get_config() -> UnifiedConfig:
    """Get global configuration instance."""
    return config


def reload_config():
    """Reload global configuration."""
    config.reload()


# Convenience functions for common config access
def get_trading_mode() -> str:
    """Get current trading mode."""
    return config.get_str('trading.mode', 'paper')


def is_paper_mode() -> bool:
    """Check if running in paper mode."""
    return get_trading_mode() == 'paper'


def is_live_mode() -> bool:
    """Check if running in live mode."""
    return get_trading_mode() == 'live'


def get_initial_balance() -> float:
    """Get initial trading balance."""
    return config.get_float('trading.initial_balance', 10000.0)


def get_risk_limits() -> Dict[str, Any]:
    """Get risk management limits."""
    return config.get_dict('risk')


def get_exchange_config(name: str) -> Optional[Dict[str, Any]]:
    """Get exchange configuration."""
    exchanges = config.get_dict('exchanges')
    
    if name == 'primary':
        return exchanges.get('primary')
    
    backups = exchanges.get('backup', [])
    for exchange in backups:
        if exchange.get('name') == name:
            return exchange
    
    return None
