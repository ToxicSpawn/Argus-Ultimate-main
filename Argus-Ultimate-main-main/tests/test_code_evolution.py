"""
Tests for the code evolution stack — true self-modifying code:
  1. CodeGenerator — template-based Python source generation
  2. CodeReviewGate — AST-based safety analysis
  3. CodeSandbox — isolated execution validation
  4. GitVersioner — auto-commit + rollback
  5. ModuleReloader — hot-load generated modules
  6. CodeEvolutionEngine — master coordinator
  7. ComponentRegistry wiring
"""
from __future__ import annotations

import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ──────────────────────────────────────────────────────────────────────────────
# CodeGenerator
# ──────────────────────────────────────────────────────────────────────────────

class TestCodeGenerator(unittest.TestCase):
    def setUp(self):
        from core.code_generator import CodeGenerator, GenerationContext
        self.tmpdir = tempfile.mkdtemp()
        self.gen = CodeGenerator(output_dir=self.tmpdir)
        self.ctx = GenerationContext(
            pattern_id="test_pattern",
            observation_count=100,
            win_rate=0.62,
            avg_pnl_aud=15.0,
            sharpe=1.2,
            target_regime="TRENDING_UP",
            description="Test pattern strategy",
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_generate_threshold_strategy(self):
        strategy = self.gen.generate_threshold_strategy(
            self.ctx, indicator="rsi", threshold=30, direction="below",
        )
        self.assertIsNotNone(strategy)
        self.assertTrue(strategy.file_path.exists())
        self.assertIn("threshold_rsi_below", strategy.name)

    def test_generated_file_is_valid_python(self):
        import ast
        strategy = self.gen.generate_threshold_strategy(
            self.ctx, indicator="rsi", threshold=30, direction="below",
        )
        source = strategy.file_path.read_text(encoding="utf-8")
        # Should parse without errors
        ast.parse(source)

    def test_generated_strategy_can_be_imported(self):
        import importlib.util
        strategy = self.gen.generate_threshold_strategy(
            self.ctx, indicator="rsi", threshold=30, direction="below",
        )
        spec = importlib.util.spec_from_file_location("test_mod", str(strategy.file_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(any(
            name.startswith("GeneratedStrategy_") for name in dir(module)
        ))

    def test_generated_strategy_evaluates(self):
        import importlib.util
        strategy = self.gen.generate_threshold_strategy(
            self.ctx, indicator="rsi", threshold=30, direction="below",
            confidence_base=0.7,
        )
        spec = importlib.util.spec_from_file_location("test_mod", str(strategy.file_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls_name = next(n for n in dir(module) if n.startswith("GeneratedStrategy_"))
        cls = getattr(module, cls_name)
        instance = cls()

        # RSI below threshold → SELL
        result = instance.evaluate({"rsi": 25, "regime": "TRENDING_UP"})
        self.assertEqual(result["action"], "SELL")
        self.assertGreater(result["confidence"], 0)

        # RSI above threshold → HOLD
        result = instance.evaluate({"rsi": 50, "regime": "TRENDING_UP"})
        self.assertEqual(result["action"], "HOLD")

        # Wrong regime → HOLD
        result = instance.evaluate({"rsi": 25, "regime": "RANGING"})
        self.assertEqual(result["action"], "HOLD")

    def test_generate_confluence_strategy(self):
        conditions = [
            {"field": "rsi", "op": "lt", "value": 30},
            {"field": "volume", "op": "gt", "value": 1000},
        ]
        strategy = self.gen.generate_confluence_strategy(
            self.ctx, conditions=conditions, action="BUY",
        )
        self.assertIsNotNone(strategy)
        self.assertIn("confluence", strategy.name)

    def test_generate_crossover_strategy(self):
        strategy = self.gen.generate_crossover_strategy(
            self.ctx, fast_indicator="ema_fast", slow_indicator="ema_slow",
            buy_on="above",
        )
        self.assertIsNotNone(strategy)
        self.assertIn("crossover", strategy.name)

    def test_invalid_indicator_rejected(self):
        # Special characters and reserved words
        strategy = self.gen.generate_threshold_strategy(
            self.ctx, indicator="!!!", threshold=30, direction="below",
        )
        self.assertIsNone(strategy)

    def test_invalid_direction_rejected(self):
        strategy = self.gen.generate_threshold_strategy(
            self.ctx, indicator="rsi", threshold=30, direction="invalid",
        )
        self.assertIsNone(strategy)

    def test_snapshot(self):
        self.gen.generate_threshold_strategy(
            self.ctx, indicator="rsi", threshold=30, direction="below",
        )
        snap = self.gen.snapshot()
        self.assertEqual(snap["stats"]["threshold"], 1)
        self.assertEqual(snap["stats"]["errors"], 0)


# ──────────────────────────────────────────────────────────────────────────────
# CodeReviewGate
# ──────────────────────────────────────────────────────────────────────────────

class TestCodeReviewGate(unittest.TestCase):
    def setUp(self):
        from core.code_review_gate import CodeReviewGate
        self.gate = CodeReviewGate()

    def test_valid_strategy_passes(self):
        good_source = '''
from generated_strategies import BaseGeneratedStrategy

class GoodStrategy(BaseGeneratedStrategy):
    name = "good"
    def evaluate(self, market_state):
        rsi = market_state.get("rsi", 50)
        if rsi < 30:
            return {"action": "BUY", "confidence": 0.7, "reasoning": "low rsi"}
        return {"action": "HOLD", "confidence": 0.0, "reasoning": "no signal"}
'''
        result = self.gate.review_source(good_source)
        self.assertTrue(result.passed)

    def test_forbidden_import_rejected(self):
        bad = '''
import os
class S(BaseGeneratedStrategy):
    pass
'''
        result = self.gate.review_source(bad)
        self.assertFalse(result.passed)
        self.assertTrue(any("os" in v for v in result.violations))

    def test_subprocess_rejected(self):
        bad = '''
import subprocess
class S(BaseGeneratedStrategy):
    pass
'''
        result = self.gate.review_source(bad)
        self.assertFalse(result.passed)

    def test_eval_rejected(self):
        bad = '''
from generated_strategies import BaseGeneratedStrategy
class S(BaseGeneratedStrategy):
    def evaluate(self, m):
        return eval("1+1")
'''
        result = self.gate.review_source(bad)
        self.assertFalse(result.passed)

    def test_exec_rejected(self):
        bad = '''
from generated_strategies import BaseGeneratedStrategy
class S(BaseGeneratedStrategy):
    def evaluate(self, m):
        exec("x=1")
        return {"action": "HOLD", "confidence": 0}
'''
        result = self.gate.review_source(bad)
        self.assertFalse(result.passed)

    def test_open_file_rejected(self):
        bad = '''
from generated_strategies import BaseGeneratedStrategy
class S(BaseGeneratedStrategy):
    def evaluate(self, m):
        f = open("/etc/passwd")
        return {"action": "HOLD", "confidence": 0}
'''
        result = self.gate.review_source(bad)
        self.assertFalse(result.passed)

    def test_infinite_loop_rejected(self):
        bad = '''
from generated_strategies import BaseGeneratedStrategy
class S(BaseGeneratedStrategy):
    def evaluate(self, m):
        while True:
            pass
'''
        result = self.gate.review_source(bad)
        self.assertFalse(result.passed)

    def test_dunder_dict_rejected(self):
        bad = '''
from generated_strategies import BaseGeneratedStrategy
class S(BaseGeneratedStrategy):
    def evaluate(self, m):
        return self.__dict__
'''
        result = self.gate.review_source(bad)
        self.assertFalse(result.passed)

    def test_no_strategy_class_rejected(self):
        bad = '''
def some_function():
    return 1
'''
        result = self.gate.review_source(bad)
        self.assertFalse(result.passed)

    def test_multiple_strategies_rejected(self):
        bad = '''
from generated_strategies import BaseGeneratedStrategy
class S1(BaseGeneratedStrategy):
    pass
class S2(BaseGeneratedStrategy):
    pass
'''
        result = self.gate.review_source(bad)
        self.assertFalse(result.passed)

    def test_snapshot(self):
        snap = self.gate.snapshot()
        self.assertIn("reviewed", snap)


# ──────────────────────────────────────────────────────────────────────────────
# CodeSandbox
# ──────────────────────────────────────────────────────────────────────────────

class TestCodeSandbox(unittest.TestCase):
    def setUp(self):
        from core.code_generator import CodeGenerator, GenerationContext
        from core.code_sandbox import CodeSandbox
        self.tmpdir = tempfile.mkdtemp()
        self.gen = CodeGenerator(output_dir=self.tmpdir)
        self.sandbox = CodeSandbox(n_samples=10)
        self.ctx = GenerationContext(
            pattern_id="sandbox_test",
            observation_count=100,
            win_rate=0.6,
            avg_pnl_aud=10,
            sharpe=1.0,
            target_regime="ANY",
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_valid_strategy_passes_sandbox(self):
        strategy = self.gen.generate_threshold_strategy(
            self.ctx, indicator="rsi", threshold=30, direction="below",
        )
        result = self.sandbox.run(strategy.file_path)
        self.assertTrue(result.passed)
        self.assertGreater(result.samples_passed, 0)

    def test_avg_eval_time_measured(self):
        strategy = self.gen.generate_threshold_strategy(
            self.ctx, indicator="rsi", threshold=30, direction="below",
        )
        result = self.sandbox.run(strategy.file_path)
        self.assertGreaterEqual(result.avg_eval_time_ms, 0)

    def test_actions_distribution_tracked(self):
        strategy = self.gen.generate_threshold_strategy(
            self.ctx, indicator="rsi", threshold=30, direction="below",
        )
        result = self.sandbox.run(strategy.file_path)
        self.assertIsInstance(result.actions_returned, dict)

    def test_missing_file_fails(self):
        result = self.sandbox.run(Path("nonexistent.py"))
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_snapshot(self):
        snap = self.sandbox.snapshot()
        self.assertIn("runs", snap)


# ──────────────────────────────────────────────────────────────────────────────
# GitVersioner
# ──────────────────────────────────────────────────────────────────────────────

class TestGitVersioner(unittest.TestCase):
    def setUp(self):
        from core.git_versioner import GitVersioner
        self.versioner = GitVersioner(dry_run=True)

    def test_initial_state(self):
        snap = self.versioner.snapshot()
        self.assertEqual(snap["total_commits"], 0)

    def test_commit_outside_repo_rejected(self):
        # Try to commit a file outside generated_strategies/
        result = self.versioner.commit_generation(
            file_path=Path("/tmp/external_file.py"),
        )
        self.assertIsNone(result)

    def test_snapshot_format(self):
        snap = self.versioner.snapshot()
        self.assertIn("git_available", snap)
        self.assertIn("by_operation", snap)


# ──────────────────────────────────────────────────────────────────────────────
# ModuleReloader
# ──────────────────────────────────────────────────────────────────────────────

class TestModuleReloader(unittest.TestCase):
    def setUp(self):
        from core.module_reloader import ModuleReloader
        self.tmpdir = tempfile.mkdtemp()
        active = Path(self.tmpdir) / "active"
        active.mkdir()
        self.reloader = ModuleReloader(active_dir=str(active))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_initial_empty(self):
        self.assertEqual(len(self.reloader._loaded), 0)

    def test_scan_empty_directory(self):
        count = self.reloader.scan_and_load()
        self.assertEqual(count, 0)

    def test_snapshot(self):
        snap = self.reloader.snapshot()
        self.assertIn("loaded_count", snap)
        self.assertIn("scan_count", snap)


# ──────────────────────────────────────────────────────────────────────────────
# CodeEvolutionEngine
# ──────────────────────────────────────────────────────────────────────────────

class TestCodeEvolutionEngine(unittest.TestCase):
    def setUp(self):
        from core.code_evolution_engine import CodeEvolutionEngine, EvolutionConfig
        self.tmpdir = tempfile.mkdtemp()
        self.cfg = EvolutionConfig(
            enabled=True,
            generation_cycles=1,
            pattern_min_observations=5,
            pattern_min_win_rate=0.55,
            auto_commit_to_git=False,
            auto_promote_to_active=True,
            candidates_dir=str(Path(self.tmpdir) / "candidates"),
            active_dir=str(Path(self.tmpdir) / "active"),
            graveyard_dir=str(Path(self.tmpdir) / "graveyard"),
        )
        self.engine = CodeEvolutionEngine(config=self.cfg)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_initial_state(self):
        self.assertEqual(self.engine._cycle_count, 0)

    def test_tick_disabled(self):
        from core.code_evolution_engine import EvolutionConfig
        self.engine._config.enabled = False
        result = self.engine.tick(cycle_number=1)
        self.assertFalse(result.get("enabled", True))

    def test_tick_with_no_components(self):
        result = self.engine.tick(cycle_number=1)
        self.assertEqual(result["generated"], [])

    def test_full_pipeline_with_pattern(self):
        from core.code_generator import CodeGenerator
        from core.code_review_gate import CodeReviewGate
        from core.code_sandbox import CodeSandbox
        from core.observation_recorder import ObservationRecorder

        recorder = ObservationRecorder()
        gen = CodeGenerator(output_dir=str(Path(self.tmpdir) / "candidates"))
        gate = CodeReviewGate()
        sandbox = CodeSandbox(n_samples=10)

        self.engine.attach(
            observation_recorder=recorder,
            code_generator=gen,
            code_review_gate=gate,
            code_sandbox=sandbox,
        )

        # Seed observations with strong pattern
        for i in range(50):
            obs = recorder.record_decision(
                symbol="BTC/USD", regime="TRENDING_UP", price=60000.0,
                strategy="momentum", action="BUY", confidence=0.7,
                final_size_pct=0.15,
            )
            recorder.complete_observation(
                obs.obs_id, pnl_aud=10.0 if i % 3 != 0 else -3.0,
            )

        result = self.engine.tick(cycle_number=1)
        self.assertGreater(len(result["generated"]), 0)

    def test_snapshot(self):
        snap = self.engine.snapshot()
        self.assertIn("stats", snap)
        self.assertIn("active_strategies", snap)

    def test_attach(self):
        mock_recorder = MagicMock()
        self.engine.attach(observation_recorder=mock_recorder)
        self.assertIsNotNone(self.engine._observation_recorder)


# ──────────────────────────────────────────────────────────────────────────────
# ComponentRegistry wiring
# ──────────────────────────────────────────────────────────────────────────────

class TestCodeEvolutionWiring(unittest.TestCase):
    def setUp(self):
        from core.component_registry import ComponentRegistry
        self.reg = ComponentRegistry(config=MagicMock())
        self.reg.config = MagicMock()
        self.reg.config.code_evolution = {
            "enabled": True,
            "generation_cycles": 500,
            "pattern_min_observations": 50,
        }

    def test_code_generator_slot(self):
        self.assertTrue(hasattr(self.reg, "code_generator"))

    def test_code_review_gate_slot(self):
        self.assertTrue(hasattr(self.reg, "code_review_gate"))

    def test_code_sandbox_slot(self):
        self.assertTrue(hasattr(self.reg, "code_sandbox"))

    def test_git_versioner_slot(self):
        self.assertTrue(hasattr(self.reg, "git_versioner"))

    def test_module_reloader_slot(self):
        self.assertTrue(hasattr(self.reg, "module_reloader"))

    def test_code_evolution_engine_slot(self):
        self.assertTrue(hasattr(self.reg, "code_evolution_engine"))

    def test_init_methods_exist(self):
        for name in ("code_generator", "code_review_gate", "code_sandbox",
                     "git_versioner", "module_reloader", "code_evolution_engine"):
            self.assertTrue(hasattr(self.reg, f"_init_{name}"))

    def test_init_full_chain(self):
        self.reg._init_observation_recorder()
        self.reg._init_code_generator()
        self.reg._init_code_review_gate()
        self.reg._init_code_sandbox()
        self.reg._init_git_versioner()
        self.reg._init_module_reloader()
        self.reg._init_code_evolution_engine()
        self.assertIsNotNone(self.reg.code_evolution_engine)


# ──────────────────────────────────────────────────────────────────────────────
# Config registration
# ──────────────────────────────────────────────────────────────────────────────

class TestCodeEvolutionConfig(unittest.TestCase):
    def test_code_evolution_key_registered(self):
        from core.config_manager import _KNOWN_TOP_LEVEL_KEYS
        self.assertIn("code_evolution", _KNOWN_TOP_LEVEL_KEYS)


# ──────────────────────────────────────────────────────────────────────────────
# generated_strategies package
# ──────────────────────────────────────────────────────────────────────────────

class TestGeneratedStrategiesPackage(unittest.TestCase):
    def test_base_class_importable(self):
        from generated_strategies import BaseGeneratedStrategy
        self.assertTrue(callable(BaseGeneratedStrategy))

    def test_base_class_has_evaluate(self):
        from generated_strategies import BaseGeneratedStrategy
        instance = BaseGeneratedStrategy()
        result = instance.evaluate({})
        self.assertIn("action", result)

    def test_directories_exist(self):
        for subdir in ("candidates", "active", "graveyard"):
            path = Path("generated_strategies") / subdir
            self.assertTrue(path.exists(), f"{path} should exist")


if __name__ == "__main__":
    unittest.main()
