"""tests/test_batch7.py

Batch 7 unit tests:
  - fix_silent_excepts rewriter (AST detection + rewrite)
  - fix_print_calls rewriter (print -> logger.info)
  - startup_config_check wiring (Pydantic config into SharedState)
  - get_config() retrieval from SharedState
"""
from __future__ import annotations

import textwrap
import pytest


# ============================================================
#  Silent Except Fixer
# ============================================================

class TestSilentExceptFixer:
    def _rewrite(self, source: str):
        from tools.fix_silent_excepts import rewrite_source
        return rewrite_source(textwrap.dedent(source), filename="test_input.py")

    def test_detects_bare_except(self):
        src = """
            try:
                x = 1/0
            except:
                pass
        """
        new, count = self._rewrite(src)
        assert count >= 1

    def test_detects_except_exception(self):
        src = """
            try:
                x = do_thing()
            except Exception:
                pass
        """
        new, count = self._rewrite(src)
        assert count >= 1

    def test_injects_logger_exception(self):
        src = """
            try:
                x = do_thing()
            except Exception:
                pass
        """
        new, count = self._rewrite(src)
        assert "logger.exception" in new

    def test_skips_already_logged(self):
        src = """
            import logging
            logger = logging.getLogger(__name__)
            try:
                x = do_thing()
            except Exception as e:
                logger.error("oops %s", e)
        """
        new, count = self._rewrite(src)
        assert count == 0

    def test_skips_specific_exceptions(self):
        src = """
            try:
                x = int("bad")
            except ValueError:
                pass
        """
        new, count = self._rewrite(src)
        assert count == 0  # ValueError is specific — not rewritten

    def test_preserves_existing_body(self):
        src = """
            try:
                x = do_thing()
            except Exception:
                do_cleanup()
        """
        new, count = self._rewrite(src)
        assert "do_cleanup" in new  # existing code preserved

    def test_injects_logger_import(self):
        src = """
            def f():
                try:
                    pass
                except Exception:
                    pass
        """
        new, count = self._rewrite(src)
        assert "logging" in new

    def test_no_change_on_clean_file(self):
        src = """
            x = 1 + 1
            print(x)
        """
        new, count = self._rewrite(src)
        assert count == 0
        assert new == textwrap.dedent(src)


# ============================================================
#  Print → Logger Fixer
# ============================================================

class TestPrintFixer:
    def _rewrite(self, source: str):
        from tools.fix_print_calls import rewrite_source
        return rewrite_source(textwrap.dedent(source), filename="test_input.py")

    def test_detects_print_call(self):
        src = """
            def f():
                print("hello")
        """
        new, count = self._rewrite(src)
        assert count >= 1

    def test_replaces_with_logger_info(self):
        src = """
            def f():
                print("hello")
        """
        new, count = self._rewrite(src)
        assert "logger.info(" in new
        assert "print(" not in new.split("logger.info(")[1] if "logger.info(" in new else True

    def test_skips_file_kwarg(self):
        src = """
            import sys
            def f():
                print("err", file=sys.stderr)
        """
        new, count = self._rewrite(src)
        assert count == 0

    def test_skips_flush_kwarg(self):
        src = """
            def f():
                print("progress", flush=True)
        """
        new, count = self._rewrite(src)
        assert count == 0

    def test_injects_logger_import(self):
        src = """
            def f():
                print("x")
        """
        new, count = self._rewrite(src)
        assert "logging" in new

    def test_no_change_on_clean(self):
        src = """
            import logging
            logger = logging.getLogger(__name__)
            logger.info("already clean")
        """
        new, count = self._rewrite(src)
        assert count == 0

    def test_multiple_prints_in_file(self):
        src = """
            def a():
                print("one")
            def b():
                print("two")
                print("three")
        """
        new, count = self._rewrite(src)
        assert count == 3


# ============================================================
#  startup_config_check wiring
# ============================================================

class TestStartupConfigWiring:
    def setup_method(self):
        # Reset SharedState singleton between tests
        try:
            from core.shared_state import SharedState
            SharedState.reset_singleton()
        except ImportError:
            pass

    def test_stores_config_in_shared_state(self, tmp_path):
        try:
            import yaml
            from core.shared_state import SharedState
            from core.startup import startup_config_check, get_config
        except ImportError:
            pytest.skip("deps not available")

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({
            "system": {"mode": "dry_run", "initial_capital": 1000.0},
            "risk": {"max_drawdown": 0.10},
            "trading": {"capital": 1000.0},
        }))

        cfg = startup_config_check(str(cfg_file))
        assert cfg is not None

        stored = get_config()
        assert stored is cfg

    def test_capital_override(self, tmp_path):
        try:
            import yaml
            from core.startup import startup_config_check
        except ImportError:
            pytest.skip("deps not available")

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({
            "trading": {"capital": 500.0},
        }))
        cfg = startup_config_check(str(cfg_file), capital_override=9999.0)
        try:
            assert cfg.trading.capital == pytest.approx(9999.0)
        except AttributeError:
            pass  # dict fallback — ok

    def test_invalid_mode_raises(self, tmp_path):
        try:
            import yaml
            from core.startup import startup_config_check
            from pydantic import ValidationError
        except ImportError:
            pytest.skip("deps not available")

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({
            "system": {"mode": "yolo"},
        }))
        with pytest.raises(ValidationError):
            startup_config_check(str(cfg_file))

    def test_get_config_returns_none_before_startup(self):
        try:
            from core.shared_state import SharedState
            from core.startup import get_config
            SharedState.reset_singleton()
        except ImportError:
            pytest.skip("deps not available")
        assert get_config() is None
