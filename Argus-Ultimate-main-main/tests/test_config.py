'''Tests for configuration management.'''

from importlib import util
from pathlib import Path


_CONFIG_PATH = Path(__file__).resolve().parents[1] / "core" / "config.py"
_SPEC = util.spec_from_file_location("legacy_core_config", _CONFIG_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
Config = _MODULE.Config


def test_config_defaults():
    '''Test configuration with defaults.'''
    config = Config()
    assert config is not None


def test_config_get():
    '''Test getting configuration values.'''
    config = Config()
    # Test with default for missing key
    value = config.get("nonexistent_key", "default_value")
    assert value == "default_value"


def test_config_get_known_defaults():
    '''Test that known default values are set.'''
    config = Config()
    assert config.get("initial_balance") == 10000.0
    assert config.get("testnet") is True
    assert config.get("log_level") == "INFO"


def test_config_env_override(monkeypatch):
    '''Test environment variable override.'''
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    config = Config()
    assert config.get("log_level") == "DEBUG"


def test_config_set_and_get():
    '''Test setting and getting values.'''
    config = Config()
    config.set("log_level", "WARNING")
    assert config.get("log_level") == "WARNING"


def test_config_get_exchange_config():
    '''Test exchange configuration retrieval.'''
    config = Config()
    exchange_config = config.get_exchange_config()
    assert "exchange" in exchange_config
    assert "testnet" in exchange_config
