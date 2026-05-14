"""Argus config package."""
from config.schema import ArgusConfig, load_config
from config.unified_config import UnifiedConfig

__all__ = ["ArgusConfig", "load_config", "UnifiedConfig"]
