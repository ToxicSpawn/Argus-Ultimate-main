
from risk.unified_risk_manager import UnifiedRiskManager
'''
Tests for Risk Manager
'''


def test_risk_manager_initialization():
    '''Test risk manager initialization'''
    rm = UnifiedRiskManager(initial_capital=10000.0)
    assert rm.initial_capital == 10000.0
    assert rm.current_capital == 10000.0


def test_risk_limits():
    '''Test risk limits'''
    rm = UnifiedRiskManager(initial_capital=10000.0, max_daily_loss=0.02, max_position_loss=0.01)
    assert rm.max_daily_loss == 0.02
    assert rm.max_position_loss == 0.01


# =========================================================================
# Margin Requirement Tracking Tests
# =========================================================================

def test_margin_initial_empty():
    '''Margin requirements start empty'''
    rm = UnifiedRiskManager(initial_capital=10000.0)
    assert rm.get_total_margin() == 0.0
    assert rm.get_free_margin(10000.0) == 10000.0


def test_update_margin_requirement():
    '''update_margin_requirement tracks per-symbol margin'''
    rm = UnifiedRiskManager(initial_capital=10000.0)
    rm.update_margin_requirement("BTC-PERP", 500.0)
    assert rm.get_total_margin() == 500.0
    rm.update_margin_requirement("ETH-PERP", 300.0)
    assert rm.get_total_margin() == 800.0


def test_update_margin_overwrites_previous():
    '''Updating same symbol overwrites the previous margin'''
    rm = UnifiedRiskManager(initial_capital=10000.0)
    rm.update_margin_requirement("BTC-PERP", 500.0)
    rm.update_margin_requirement("BTC-PERP", 700.0)
    assert rm.get_total_margin() == 700.0


def test_clear_margin_on_zero():
    '''Setting margin to 0 removes the symbol'''
    rm = UnifiedRiskManager(initial_capital=10000.0)
    rm.update_margin_requirement("BTC-PERP", 500.0)
    rm.update_margin_requirement("BTC-PERP", 0.0)
    assert rm.get_total_margin() == 0.0
    assert "BTC-PERP" not in rm._margin_requirements


def test_clear_margin_on_negative():
    '''Setting margin to negative removes the symbol'''
    rm = UnifiedRiskManager(initial_capital=10000.0)
    rm.update_margin_requirement("BTC-PERP", 500.0)
    rm.update_margin_requirement("BTC-PERP", -1.0)
    assert rm.get_total_margin() == 0.0


def test_get_free_margin():
    '''Free margin is capital minus total margin'''
    rm = UnifiedRiskManager(initial_capital=10000.0)
    rm.update_margin_requirement("BTC-PERP", 3000.0)
    rm.update_margin_requirement("ETH-PERP", 2000.0)
    assert rm.get_free_margin(10000.0) == 5000.0


def test_check_margin_available_sufficient():
    '''check_margin_available returns True when enough free margin'''
    rm = UnifiedRiskManager(initial_capital=10000.0)
    rm.update_margin_requirement("BTC-PERP", 3000.0)
    assert rm.check_margin_available(5000.0, 10000.0) is True


def test_check_margin_available_insufficient():
    '''check_margin_available returns False when not enough free margin'''
    rm = UnifiedRiskManager(initial_capital=10000.0)
    rm.update_margin_requirement("BTC-PERP", 8000.0)
    assert rm.check_margin_available(5000.0, 10000.0) is False


def test_check_margin_available_exact():
    '''check_margin_available returns True when margin exactly matches free'''
    rm = UnifiedRiskManager(initial_capital=10000.0)
    rm.update_margin_requirement("BTC-PERP", 5000.0)
    assert rm.check_margin_available(5000.0, 10000.0) is True


# =========================================================================
# Pre-trade Risk Check Tests
# =========================================================================

def test_pre_trade_risk_check_approved():
    '''Pre-trade check approves when all conditions pass'''
    rm = UnifiedRiskManager(initial_capital=10000.0)
    approved, reason = rm.pre_trade_risk_check("BTC/AUD", 1000.0)
    assert approved is True
    assert reason == "approved"


def test_pre_trade_risk_check_circuit_breaker():
    '''Pre-trade check rejects when circuit breaker is active'''
    rm = UnifiedRiskManager(initial_capital=10000.0, max_consecutive_losses=2)
    rm.record_trade(-10.0)
    rm.record_trade(-10.0)
    approved, reason = rm.pre_trade_risk_check("BTC/AUD", 1000.0)
    assert approved is False
    assert "circuit_breaker" in reason


def test_pre_trade_risk_check_leverage_exceeded():
    '''Pre-trade check rejects when leverage would be exceeded'''
    rm = UnifiedRiskManager(initial_capital=10000.0, max_leverage=2.0)
    rm.set_total_exposure(18000.0)
    approved, reason = rm.pre_trade_risk_check("BTC/AUD", 5000.0)
    assert approved is False
    assert "leverage" in reason


def test_pre_trade_risk_check_margin_rejected():
    '''Pre-trade check rejects when margin is insufficient'''
    rm = UnifiedRiskManager(initial_capital=10000.0)
    rm.update_margin_requirement("ETH-PERP", 8000.0)
    approved, reason = rm.pre_trade_risk_check(
        "BTC-PERP", 3000.0, required_margin_usd=5000.0, total_capital_usd=10000.0,
    )
    assert approved is False
    assert "margin" in reason


def test_pre_trade_risk_check_margin_approved():
    '''Pre-trade check approves when margin is sufficient'''
    rm = UnifiedRiskManager(initial_capital=10000.0)
    rm.update_margin_requirement("ETH-PERP", 2000.0)
    approved, reason = rm.pre_trade_risk_check(
        "BTC-PERP", 3000.0, required_margin_usd=3000.0, total_capital_usd=10000.0,
    )
    assert approved is True
    assert reason == "approved"


def test_pre_trade_risk_check_uses_current_capital_default():
    '''Pre-trade check uses self.current_capital if total_capital_usd not provided'''
    rm = UnifiedRiskManager(initial_capital=10000.0)
    rm.update_margin_requirement("ETH-PERP", 9000.0)
    approved, reason = rm.pre_trade_risk_check(
        "BTC-PERP", 500.0, required_margin_usd=2000.0,
    )
    assert approved is False
    assert "margin" in reason
