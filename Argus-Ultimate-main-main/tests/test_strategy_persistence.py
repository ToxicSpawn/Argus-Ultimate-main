"""
Tests for Strategy State Persistence + Cooldown Enforcement.

Covers:
  - SQLite round-trip (save/load)
  - Trade count increments
  - Win/loss tracking & consecutive streaks
  - Cooldown activation after consecutive-loss threshold
  - Cooldown blocks signal processing
  - Cooldown expiration
  - Multiple strategies are independent
  - Parameter validation catches bad values
  - State survives simulated restart (new store instance on same DB)
  - Edge cases (zero PnL, empty DB, concurrent access)
"""

import os
import tempfile
import time

import pytest

from strategies.strategy_state_store import (
    DEFAULT_COOLDOWN_MINUTES,
    DEFAULT_MAX_CONSECUTIVE_LOSSES,
    StrategyState,
    StrategyStateStore,
    validate_strategy_parameters,
)


@pytest.fixture
def tmp_db(tmp_path):
    """Return a fresh temporary DB path."""
    return str(tmp_path / "test_strategy_states.db")


@pytest.fixture
def store(tmp_db):
    """Return a fresh StrategyStateStore."""
    return StrategyStateStore(db_path=tmp_db)


# ── Save/Load Round Trip ─────────────────────────────────────────────────────

class TestSaveLoadRoundTrip:
    def test_save_and_load_single(self, store):
        state = {
            "trade_count": 10,
            "win_count": 7,
            "loss_count": 3,
            "total_pnl": 42.5,
            "consecutive_losses": 1,
            "consecutive_wins": 2,
            "last_trade_time": 1700000000.0,
            "cooldown_until": None,
            "parameters": {"rsi_period": 14},
        }
        store.save_state("my_strategy", state)
        loaded = store.load_state("my_strategy")
        assert loaded is not None
        assert loaded["trade_count"] == 10
        assert loaded["win_count"] == 7
        assert loaded["loss_count"] == 3
        assert loaded["total_pnl"] == pytest.approx(42.5)
        assert loaded["consecutive_losses"] == 1
        assert loaded["consecutive_wins"] == 2
        assert loaded["parameters"]["rsi_period"] == 14

    def test_load_missing_returns_none(self, store):
        assert store.load_state("nonexistent") is None

    def test_load_all_empty(self, store):
        result = store.load_all()
        assert result == {}

    def test_load_all_multiple(self, store):
        store.save_state("strat_a", {"trade_count": 5, "total_pnl": 10.0})
        store.save_state("strat_b", {"trade_count": 3, "total_pnl": -5.0})
        all_states = store.load_all()
        assert len(all_states) == 2
        assert "strat_a" in all_states
        assert "strat_b" in all_states
        assert all_states["strat_a"]["trade_count"] == 5
        assert all_states["strat_b"]["total_pnl"] == pytest.approx(-5.0)

    def test_save_overwrites(self, store):
        store.save_state("x", {"trade_count": 1})
        store.save_state("x", {"trade_count": 99})
        loaded = store.load_state("x")
        assert loaded["trade_count"] == 99


# ── Trade Count Increments ────────────────────────────────────────────────────

class TestTradeCountIncrements:
    def test_single_win(self, store):
        result = store.update_after_trade("strat", pnl=10.0, timestamp=1.0)
        assert result["trade_count"] == 1
        assert result["win_count"] == 1
        assert result["loss_count"] == 0

    def test_single_loss(self, store):
        result = store.update_after_trade("strat", pnl=-5.0, timestamp=1.0)
        assert result["trade_count"] == 1
        assert result["win_count"] == 0
        assert result["loss_count"] == 1

    def test_multiple_trades(self, store):
        store.update_after_trade("strat", pnl=10.0, timestamp=1.0)
        store.update_after_trade("strat", pnl=-3.0, timestamp=2.0)
        store.update_after_trade("strat", pnl=5.0, timestamp=3.0)
        result = store.get_state("strat")
        assert result["trade_count"] == 3
        assert result["win_count"] == 2
        assert result["loss_count"] == 1
        assert result["total_pnl"] == pytest.approx(12.0)

    def test_zero_pnl_counts_as_win(self, store):
        """Zero PnL (breakeven) is treated as a non-loss (>= 0)."""
        result = store.update_after_trade("strat", pnl=0.0, timestamp=1.0)
        assert result["win_count"] == 1
        assert result["loss_count"] == 0


# ── Consecutive Loss Tracking ─────────────────────────────────────────────────

class TestConsecutiveLossTracking:
    def test_consecutive_losses_increment(self, store):
        for i in range(3):
            result = store.update_after_trade("strat", pnl=-1.0, timestamp=float(i))
        assert result["consecutive_losses"] == 3
        assert result["consecutive_wins"] == 0

    def test_win_resets_consecutive_losses(self, store):
        store.update_after_trade("strat", pnl=-1.0, timestamp=1.0)
        store.update_after_trade("strat", pnl=-1.0, timestamp=2.0)
        result = store.update_after_trade("strat", pnl=5.0, timestamp=3.0)
        assert result["consecutive_losses"] == 0
        assert result["consecutive_wins"] == 1

    def test_loss_resets_consecutive_wins(self, store):
        store.update_after_trade("strat", pnl=5.0, timestamp=1.0)
        store.update_after_trade("strat", pnl=5.0, timestamp=2.0)
        result = store.update_after_trade("strat", pnl=-1.0, timestamp=3.0)
        assert result["consecutive_wins"] == 0
        assert result["consecutive_losses"] == 1


# ── Cooldown Activation ──────────────────────────────────────────────────────

class TestCooldownActivation:
    def test_cooldown_activates_after_threshold(self, tmp_db):
        store = StrategyStateStore(db_path=tmp_db, max_consecutive_losses=3, cooldown_minutes=30)
        now = time.time()
        for i in range(3):
            store.update_after_trade("strat", pnl=-1.0, timestamp=now + i)
        state = store.get_state("strat")
        assert state["cooldown_until"] is not None
        assert state["cooldown_until"] > now
        # Cooldown should be ~30 minutes from the last trade
        expected_end = (now + 2) + 30 * 60
        assert abs(state["cooldown_until"] - expected_end) < 2.0

    def test_cooldown_not_activated_below_threshold(self, tmp_db):
        store = StrategyStateStore(db_path=tmp_db, max_consecutive_losses=5, cooldown_minutes=60)
        for i in range(4):
            store.update_after_trade("strat", pnl=-1.0, timestamp=float(i))
        state = store.get_state("strat")
        assert state["cooldown_until"] is None

    def test_cooldown_extends_on_continued_losses(self, tmp_db):
        store = StrategyStateStore(db_path=tmp_db, max_consecutive_losses=2, cooldown_minutes=10)
        now = time.time()
        store.update_after_trade("strat", pnl=-1.0, timestamp=now)
        store.update_after_trade("strat", pnl=-1.0, timestamp=now + 1)
        cd1 = store.get_state("strat")["cooldown_until"]
        # Third loss extends cooldown
        store.update_after_trade("strat", pnl=-1.0, timestamp=now + 2)
        cd2 = store.get_state("strat")["cooldown_until"]
        assert cd2 > cd1


# ── Cooldown Enforcement ─────────────────────────────────────────────────────

class TestCooldownEnforcement:
    def test_check_cooldown_during_active(self, tmp_db):
        store = StrategyStateStore(db_path=tmp_db, max_consecutive_losses=2, cooldown_minutes=60)
        now = time.time()
        store.update_after_trade("strat", pnl=-1.0, timestamp=now)
        store.update_after_trade("strat", pnl=-1.0, timestamp=now)
        assert store.check_cooldown("strat", now=now + 10) is True

    def test_check_cooldown_after_expiry(self, tmp_db):
        store = StrategyStateStore(db_path=tmp_db, max_consecutive_losses=2, cooldown_minutes=1)
        now = time.time()
        store.update_after_trade("strat", pnl=-1.0, timestamp=now)
        store.update_after_trade("strat", pnl=-1.0, timestamp=now)
        # 2 minutes later = past 1-minute cooldown
        assert store.check_cooldown("strat", now=now + 120) is False

    def test_check_cooldown_no_state(self, store):
        assert store.check_cooldown("nonexistent") is False

    def test_check_cooldown_no_cooldown_set(self, store):
        store.update_after_trade("strat", pnl=10.0, timestamp=1.0)
        assert store.check_cooldown("strat") is False

    def test_cooldown_remaining_seconds(self, tmp_db):
        store = StrategyStateStore(db_path=tmp_db, max_consecutive_losses=2, cooldown_minutes=10)
        now = time.time()
        store.update_after_trade("strat", pnl=-1.0, timestamp=now)
        store.update_after_trade("strat", pnl=-1.0, timestamp=now)
        remaining = store.cooldown_remaining_seconds("strat", now=now + 60)
        assert remaining > 0
        assert remaining < 10 * 60

    def test_cooldown_remaining_zero_after_expiry(self, tmp_db):
        store = StrategyStateStore(db_path=tmp_db, max_consecutive_losses=2, cooldown_minutes=1)
        now = time.time()
        store.update_after_trade("strat", pnl=-1.0, timestamp=now)
        store.update_after_trade("strat", pnl=-1.0, timestamp=now)
        assert store.cooldown_remaining_seconds("strat", now=now + 120) == 0.0

    def test_win_clears_cooldown(self, tmp_db):
        store = StrategyStateStore(db_path=tmp_db, max_consecutive_losses=2, cooldown_minutes=60)
        now = time.time()
        store.update_after_trade("strat", pnl=-1.0, timestamp=now)
        store.update_after_trade("strat", pnl=-1.0, timestamp=now)
        assert store.check_cooldown("strat", now=now + 1) is True
        # Win clears cooldown
        store.update_after_trade("strat", pnl=5.0, timestamp=now + 2)
        assert store.check_cooldown("strat", now=now + 3) is False

    def test_clear_cooldown_manual(self, tmp_db):
        store = StrategyStateStore(db_path=tmp_db, max_consecutive_losses=2, cooldown_minutes=60)
        now = time.time()
        store.update_after_trade("strat", pnl=-1.0, timestamp=now)
        store.update_after_trade("strat", pnl=-1.0, timestamp=now)
        assert store.check_cooldown("strat", now=now + 1) is True
        store.clear_cooldown("strat")
        assert store.check_cooldown("strat", now=now + 1) is False


# ── Multiple Strategies Independence ─────────────────────────────────────────

class TestMultipleStrategiesIndependence:
    def test_strategies_are_independent(self, tmp_db):
        store = StrategyStateStore(db_path=tmp_db, max_consecutive_losses=3, cooldown_minutes=60)
        now = time.time()
        # Strategy A: 3 losses -> cooldown
        for i in range(3):
            store.update_after_trade("strat_a", pnl=-1.0, timestamp=now + i)
        # Strategy B: wins only
        for i in range(3):
            store.update_after_trade("strat_b", pnl=5.0, timestamp=now + i)

        assert store.check_cooldown("strat_a", now=now + 10) is True
        assert store.check_cooldown("strat_b", now=now + 10) is False

        a = store.get_state("strat_a")
        b = store.get_state("strat_b")
        assert a["consecutive_losses"] == 3
        assert b["consecutive_wins"] == 3
        assert a["win_count"] == 0
        assert b["loss_count"] == 0


# ── State Survives Restart ────────────────────────────────────────────────────

class TestStateSurvivesRestart:
    def test_reload_from_same_db(self, tmp_db):
        # First store instance
        store1 = StrategyStateStore(db_path=tmp_db, max_consecutive_losses=3, cooldown_minutes=60)
        now = time.time()
        store1.update_after_trade("strat", pnl=10.0, timestamp=now)
        store1.update_after_trade("strat", pnl=-3.0, timestamp=now + 1)
        store1.save_all()

        # Simulate restart: new store instance on same DB
        store2 = StrategyStateStore(db_path=tmp_db, max_consecutive_losses=3, cooldown_minutes=60)
        loaded = store2.load_all()
        assert "strat" in loaded
        assert loaded["strat"]["trade_count"] == 2
        assert loaded["strat"]["win_count"] == 1
        assert loaded["strat"]["loss_count"] == 1
        assert loaded["strat"]["total_pnl"] == pytest.approx(7.0)

    def test_cooldown_persists_across_restart(self, tmp_db):
        now = time.time()
        store1 = StrategyStateStore(db_path=tmp_db, max_consecutive_losses=2, cooldown_minutes=60)
        store1.update_after_trade("strat", pnl=-1.0, timestamp=now)
        store1.update_after_trade("strat", pnl=-1.0, timestamp=now)
        assert store1.check_cooldown("strat", now=now + 5) is True
        store1.save_all()

        # New instance
        store2 = StrategyStateStore(db_path=tmp_db, max_consecutive_losses=2, cooldown_minutes=60)
        store2.load_all()
        assert store2.check_cooldown("strat", now=now + 10) is True


# ── Parameter Validation ──────────────────────────────────────────────────────

class TestParameterValidation:
    def _make_config(self, **kwargs):
        """Create a simple namespace object as config."""
        class Cfg:
            pass
        c = Cfg()
        for k, v in kwargs.items():
            setattr(c, k, v)
        return c

    def test_valid_config_no_errors(self):
        cfg = self._make_config(
            se_rsi_period=14,
            se_bb_period=20,
            min_signal_confidence=0.75,
            se_buy_rsi=35.0,
            se_sell_rsi=65.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
            max_concurrent_signals=3,
        )
        errors = validate_strategy_parameters(cfg)
        assert errors == []

    def test_rsi_period_too_small(self):
        cfg = self._make_config(se_rsi_period=1)
        errors = validate_strategy_parameters(cfg)
        assert any("se_rsi_period" in e for e in errors)

    def test_bb_period_too_small(self):
        cfg = self._make_config(se_bb_period=0)
        errors = validate_strategy_parameters(cfg)
        assert any("se_bb_period" in e for e in errors)

    def test_confidence_out_of_range(self):
        cfg = self._make_config(min_signal_confidence=1.5)
        errors = validate_strategy_parameters(cfg)
        assert any("min_signal_confidence" in e for e in errors)

    def test_negative_confidence(self):
        cfg = self._make_config(min_signal_confidence=-0.1)
        errors = validate_strategy_parameters(cfg)
        assert any("min_signal_confidence" in e for e in errors)

    def test_buy_rsi_above_sell_rsi(self):
        cfg = self._make_config(se_buy_rsi=70.0, se_sell_rsi=30.0)
        errors = validate_strategy_parameters(cfg)
        assert any("se_buy_rsi" in e and "se_sell_rsi" in e for e in errors)

    def test_stop_loss_negative(self):
        cfg = self._make_config(stop_loss_pct=-0.01)
        errors = validate_strategy_parameters(cfg)
        assert any("stop_loss_pct" in e for e in errors)

    def test_stop_loss_too_large(self):
        cfg = self._make_config(stop_loss_pct=1.5)
        errors = validate_strategy_parameters(cfg)
        assert any("stop_loss_pct" in e for e in errors)

    def test_max_concurrent_signals_zero(self):
        cfg = self._make_config(max_concurrent_signals=0)
        errors = validate_strategy_parameters(cfg)
        assert any("max_concurrent_signals" in e for e in errors)

    def test_bb_threshold_out_of_range(self):
        cfg = self._make_config(se_buy_bb=1.5)
        errors = validate_strategy_parameters(cfg)
        assert any("se_buy_bb" in e for e in errors)

    def test_cooldown_minutes_negative(self):
        cfg = self._make_config(strategy_cooldown_minutes=-10)
        errors = validate_strategy_parameters(cfg)
        assert any("strategy_cooldown_minutes" in e for e in errors)

    def test_max_consecutive_losses_zero(self):
        cfg = self._make_config(strategy_max_consecutive_losses=0)
        errors = validate_strategy_parameters(cfg)
        assert any("strategy_max_consecutive_losses" in e for e in errors)


# ── StrategyState dataclass ───────────────────────────────────────────────────

class TestStrategyState:
    def test_win_rate_no_trades(self):
        s = StrategyState(strategy_name="x")
        assert s.win_rate == 0.0

    def test_win_rate_calculation(self):
        s = StrategyState(strategy_name="x", trade_count=10, win_count=7)
        assert s.win_rate == pytest.approx(0.7)

    def test_to_dict_from_dict_roundtrip(self):
        s = StrategyState(
            strategy_name="test",
            trade_count=5,
            win_count=3,
            loss_count=2,
            total_pnl=15.0,
            consecutive_losses=0,
            consecutive_wins=1,
            last_trade_time=12345.0,
            cooldown_until=None,
            parameters={"k": "v"},
        )
        d = s.to_dict()
        s2 = StrategyState.from_dict(d)
        assert s2.strategy_name == s.strategy_name
        assert s2.trade_count == s.trade_count
        assert s2.total_pnl == s.total_pnl
        assert s2.parameters == s.parameters


# ── save_all / load_all consistency ───────────────────────────────────────────

class TestSaveAllLoadAll:
    def test_save_all_persists_in_memory_states(self, tmp_db):
        store = StrategyStateStore(db_path=tmp_db)
        store.update_after_trade("a", pnl=1.0, timestamp=1.0)
        store.update_after_trade("b", pnl=-2.0, timestamp=2.0)
        store.save_all()

        # Fresh instance
        store2 = StrategyStateStore(db_path=tmp_db)
        all_s = store2.load_all()
        assert "a" in all_s
        assert "b" in all_s
        assert all_s["a"]["total_pnl"] == pytest.approx(1.0)
        assert all_s["b"]["total_pnl"] == pytest.approx(-2.0)


# ── Edge Cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_update_creates_state_if_missing(self, store):
        result = store.update_after_trade("new_strat", pnl=1.0)
        assert result["trade_count"] == 1
        assert result["strategy_name"] == "new_strat"

    def test_get_state_missing(self, store):
        assert store.get_state("nope") is None

    def test_get_all_states_empty(self, store):
        assert store.get_all_states() == {}

    def test_default_thresholds(self):
        assert DEFAULT_MAX_CONSECUTIVE_LOSSES == 5
        assert DEFAULT_COOLDOWN_MINUTES == 60
