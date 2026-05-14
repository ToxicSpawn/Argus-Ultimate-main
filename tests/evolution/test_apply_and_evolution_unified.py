"""
Tests for evolution.apply_evolved_strategies and evolution.evolution_unified.

Covers: apply_to_config, apply_from_file, fitness min_trades penalty,
evolve_once return structure, dry_run no persist.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest


# Project root on path via conftest
def test_apply_to_config_sets_matching_attributes():
    from evolution.apply_evolved_strategies import apply_to_config

    class C:
        se_buy_rsi: float = 35.0
        se_sell_rsi: float = 65.0
        other: int = 1

    config = C()
    params: Dict[str, Any] = {"se_buy_rsi": 28.0, "se_sell_rsi": 72.0, "nonexistent": 99.0}
    n = apply_to_config(config, params)
    assert n == 2
    assert config.se_buy_rsi == 28.0
    assert config.se_sell_rsi == 72.0
    assert not hasattr(config, "nonexistent") or getattr(config, "other", 1) == 1


def test_apply_from_file_missing_file_returns_zero():
    from evolution.apply_evolved_strategies import apply_from_file

    class C:
        se_buy_rsi: float = 35.0

    n = apply_from_file(C(), path=Path("/nonexistent/evolved_params.json"))
    assert n == 0


def test_apply_from_file_applies_best_params():
    from evolution.apply_evolved_strategies import apply_from_file

    class C:
        se_buy_rsi: float = 35.0
        min_signal_confidence: float = 0.75

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"best_params": {"se_buy_rsi": 25.0, "min_signal_confidence": 0.80}}, f)
        path = f.name
    try:
        config = C()
        n = apply_from_file(config, path=path, key="best_params")
        assert n == 2
        assert config.se_buy_rsi == 25.0
        assert config.min_signal_confidence == 0.80
    finally:
        Path(path).unlink(missing_ok=True)


def test_fitness_returns_penalty_when_trades_below_min_trades():
    try:
        import scripts.paper_loop_30d_unified as _m
    except Exception:
        pytest.skip("scripts.paper_loop_30d_unified not importable")
    from evolution.strategy_genetic_algorithm import make_paper_loop_fitness

    with patch.object(_m, "run_paper_loop", return_value={"trades": 0, "sharpe": 1.0, "return_pct": 1.0}):
        fitness = make_paper_loop_fitness(
            config_path="unified_config.yaml",
            days=1,
            timeframe="1h",
            min_trades=3,
        )
        score = fitness({"se_buy_rsi": 30.0, "se_sell_rsi": 70.0})
        assert score <= -1e8


def test_evolve_once_returns_expected_structure():
    from evolution.evolution_unified import evolve_once

    with patch("evolution.evolution_unified.run_genetic_algorithm") as mock_ga:
        mock_ga.return_value = (
            {"se_buy_rsi": 32.0, "se_sell_rsi": 68.0},
            0.5,
            [0.3, 0.5],
        )
        result = evolve_once(
            generations=1,
            population_size=2,
            fitness_days=1,
            dry_run=True,
            persist=False,
            source="interval",
        )
    assert "best_params" in result
    assert "best_fitness" in result
    assert "history" in result
    assert result["best_fitness"] == 0.5
    assert result["best_params"]["se_buy_rsi"] == 32.0


def test_evolve_once_dry_run_does_not_write_file():
    from evolution.evolution_unified import evolve_once

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "evolved_params.json"
        with patch("evolution.strategy_genetic_algorithm.run_genetic_algorithm") as mock_ga:
            mock_ga.return_value = (
                {"se_buy_rsi": 32.0},
                0.5,
                [0.5],
            )
            evolve_once(
                generations=1,
                population_size=2,
                fitness_days=1,
                dry_run=True,
                persist=True,
                evolved_params_path=str(path),
                source="interval",
            )
        assert not path.exists()


def test_apply_evolved_params_filter_to_config():
    from evolution.evolution_unified import apply_evolved_params

    class C:
        se_buy_rsi: float = 35.0

    config = C()
    params = {"se_buy_rsi": 28.0, "nonexistent_key": 99.0}
    n = apply_evolved_params(config, params, filter_to_config=True)
    assert n == 1
    assert config.se_buy_rsi == 28.0


def test_load_last_best_params_returns_none_for_missing_file():
    from evolution.apply_evolved_strategies import load_last_best_params

    assert load_last_best_params(Path("/nonexistent/evolved.json")) is None


def test_load_last_best_params_returns_dict():
    from evolution.apply_evolved_strategies import load_last_best_params

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"best_params": {"se_buy_rsi": 30.0}}, f)
        path = f.name
    try:
        out = load_last_best_params(path)
        assert out is not None
        assert out.get("se_buy_rsi") == 30.0
    finally:
        Path(path).unlink(missing_ok=True)


def test_rollback_to_previous():
    from evolution.apply_evolved_strategies import get_version_history, rollback_to_previous, write_evolved_params

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "evolved.json"
        hist_dir = Path(tmp) / "evolved_history"
        hist_dir.mkdir(parents=True, exist_ok=True)
        # Write a version into history
        payload = {"best_params": {"se_buy_rsi": 28.0}, "timestamp_utc": "2025-01-01T00:00:00Z"}
        (hist_dir / "evolved_20250101_120000.json").write_text(json.dumps(payload), encoding="utf-8")
        out = rollback_to_previous(path=path, version_history_dir=str(hist_dir), index=0)
        assert out is not None
        assert path.exists()
        data = json.loads(path.read_text())
        assert data.get("best_params", {}).get("se_buy_rsi") == 28.0


def test_backup_if_exists_creates_copy():
    from evolution.apply_evolved_strategies import backup_if_exists, write_evolved_params

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "evolved.json"
        write_evolved_params({"a": 1}, path=path, meta={}, backup_before=False)
        backup_dir = Path(tmp) / "backups"
        backup_if_exists(path, backup_dir=backup_dir)
        backups = list(backup_dir.glob("evolved_params_*.json"))
        assert len(backups) == 1
        data = json.loads(backups[0].read_text())
        assert data.get("best_params", {}).get("a") == 1
