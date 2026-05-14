"""
Module Reloader — dynamically loads + hot-reloads generated strategy modules.

After a generated strategy passes all gates and is committed, this module
imports it into the running Python process so ARGUS can use it without restart.

Capabilities:
  - Discover all .py files in generated_strategies/active/
  - Import them into the running process
  - Reload modules when the underlying file changes
  - Track which strategies are loaded
  - Unload retired strategies

This is the bridge between disk-based generated code and the running ARGUS
trading loop. Without this, generated files would just sit on disk unused.
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type

logger = logging.getLogger(__name__)


@dataclass
class LoadedModule:
    """Tracks a dynamically loaded strategy module."""
    module_name: str
    file_path: Path
    file_mtime: float
    strategy_class: Optional[type] = None
    strategy_instance: Optional[Any] = None
    loaded_at: float = field(default_factory=time.time)
    reload_count: int = 0
    error_count: int = 0
    last_error: str = ""


class ModuleReloader:
    """
    Dynamically discovers, imports, and hot-reloads generated strategy modules.

    Usage::

        reloader = ModuleReloader(active_dir="generated_strategies/active")

        # Initial scan
        loaded = reloader.scan_and_load()

        # Periodic check for new files / changed files
        reloader.refresh()

        # Get all loaded strategies
        for name, mod in reloader.loaded.items():
            instance = mod.strategy_instance
            result = instance.evaluate(market_state)

        # Unload a retired strategy
        reloader.unload("strat_name_xyz")
    """

    def __init__(
        self,
        active_dir: str = "generated_strategies/active",
        candidates_dir: str = "generated_strategies/candidates",
    ) -> None:
        self.active_dir = Path(active_dir)
        self.candidates_dir = Path(candidates_dir)
        self.active_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self._loaded: Dict[str, LoadedModule] = {}
        self._scan_count = 0
        self._reload_count = 0
        self._unload_count = 0
        logger.info(
            "ModuleReloader: initialized (active=%s)", self.active_dir,
        )

    @property
    def loaded(self) -> Dict[str, LoadedModule]:
        return self._loaded

    def scan_and_load(self) -> int:
        """
        Scan the active directory for .py files and load any that aren't loaded.
        Returns the number of newly loaded modules.
        """
        self._scan_count += 1
        new_count = 0

        for py_file in self.active_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            module_name = py_file.stem
            if module_name not in self._loaded:
                if self._load_file(py_file):
                    new_count += 1

        return new_count

    def refresh(self) -> Dict[str, int]:
        """
        Check for: new files (load), changed files (reload), missing files (unload).
        Returns counts dict.
        """
        result = {"loaded": 0, "reloaded": 0, "unloaded": 0}

        # Get current files in active directory
        current_files: Dict[str, Path] = {}
        for py_file in self.active_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            current_files[py_file.stem] = py_file

        # Load new files
        for module_name, py_file in current_files.items():
            if module_name not in self._loaded:
                if self._load_file(py_file):
                    result["loaded"] += 1

        # Check for changes (mtime)
        for module_name in list(current_files.keys()):
            if module_name in self._loaded:
                py_file = current_files[module_name]
                loaded = self._loaded[module_name]
                try:
                    current_mtime = py_file.stat().st_mtime
                except OSError:
                    continue
                if current_mtime > loaded.file_mtime:
                    if self._reload_module(module_name, py_file):
                        result["reloaded"] += 1

        # Unload removed files
        for module_name in list(self._loaded.keys()):
            if module_name not in current_files:
                if self._unload_module(module_name):
                    result["unloaded"] += 1

        return result

    def _load_file(self, file_path: Path) -> bool:
        """Load a single .py file as a strategy module."""
        module_name = file_path.stem
        full_module_name = f"generated_strategies.active.{module_name}"

        try:
            spec = importlib.util.spec_from_file_location(
                full_module_name, str(file_path),
            )
            if spec is None or spec.loader is None:
                logger.warning("ModuleReloader: spec failed for %s", file_path)
                return False

            module = importlib.util.module_from_spec(spec)
            sys.modules[full_module_name] = module
            spec.loader.exec_module(module)

            # Find the strategy class
            strategy_class = self._find_strategy_class(module)
            if strategy_class is None:
                logger.warning("ModuleReloader: no strategy class in %s", file_path)
                sys.modules.pop(full_module_name, None)
                return False

            # Instantiate
            try:
                instance = strategy_class()
            except Exception as exc:
                logger.warning(
                    "ModuleReloader: instantiation failed for %s: %s",
                    file_path.name, exc,
                )
                sys.modules.pop(full_module_name, None)
                return False

            # Record
            try:
                mtime = file_path.stat().st_mtime
            except OSError:
                mtime = time.time()

            self._loaded[module_name] = LoadedModule(
                module_name=module_name,
                file_path=file_path,
                file_mtime=mtime,
                strategy_class=strategy_class,
                strategy_instance=instance,
            )
            logger.info("ModuleReloader: loaded %s", module_name)
            return True

        except Exception as exc:
            logger.warning("ModuleReloader: load failed for %s: %s", file_path.name, exc)
            return False

    def _reload_module(self, module_name: str, file_path: Path) -> bool:
        """Reload a previously loaded module from updated file."""
        full_module_name = f"generated_strategies.active.{module_name}"

        if full_module_name not in sys.modules:
            return self._load_file(file_path)

        try:
            mod = sys.modules[full_module_name]
            importlib.reload(mod)

            strategy_class = self._find_strategy_class(mod)
            if strategy_class is None:
                logger.warning(
                    "ModuleReloader: reload lost strategy class for %s", module_name,
                )
                return False

            try:
                instance = strategy_class()
            except Exception as exc:
                logger.warning(
                    "ModuleReloader: reload instantiation failed for %s: %s",
                    module_name, exc,
                )
                return False

            try:
                mtime = file_path.stat().st_mtime
            except OSError:
                mtime = time.time()

            loaded = self._loaded[module_name]
            loaded.strategy_class = strategy_class
            loaded.strategy_instance = instance
            loaded.file_mtime = mtime
            loaded.reload_count += 1
            self._reload_count += 1
            logger.info("ModuleReloader: reloaded %s", module_name)
            return True

        except Exception as exc:
            logger.warning("ModuleReloader: reload failed for %s: %s", module_name, exc)
            loaded = self._loaded.get(module_name)
            if loaded:
                loaded.error_count += 1
                loaded.last_error = str(exc)
            return False

    def _unload_module(self, module_name: str) -> bool:
        """Remove a module from the loaded set."""
        full_module_name = f"generated_strategies.active.{module_name}"
        sys.modules.pop(full_module_name, None)
        if module_name in self._loaded:
            del self._loaded[module_name]
            self._unload_count += 1
            logger.info("ModuleReloader: unloaded %s", module_name)
            return True
        return False

    def unload(self, module_name: str) -> bool:
        """Public unload — explicit retirement."""
        return self._unload_module(module_name)

    def get_strategy(self, module_name: str) -> Optional[Any]:
        """Get a loaded strategy instance by name."""
        loaded = self._loaded.get(module_name)
        return loaded.strategy_instance if loaded else None

    def get_all_strategies(self) -> List[Any]:
        """Get all loaded strategy instances."""
        return [m.strategy_instance for m in self._loaded.values() if m.strategy_instance]

    def evaluate_all(self, market_state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all loaded strategies against the same market state.
        Returns dict of {strategy_name: result_dict}.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for name, mod in self._loaded.items():
            if mod.strategy_instance is None:
                continue
            try:
                result = mod.strategy_instance.evaluate(market_state)
                results[name] = result
            except Exception as exc:
                mod.error_count += 1
                mod.last_error = str(exc)
                results[name] = {"action": "HOLD", "confidence": 0.0, "error": str(exc)}
        return results

    @staticmethod
    def _find_strategy_class(module: Any) -> Optional[type]:
        """Find the BaseGeneratedStrategy subclass in a module."""
        try:
            from generated_strategies import BaseGeneratedStrategy
        except ImportError:
            return None

        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseGeneratedStrategy)
                and attr is not BaseGeneratedStrategy
            ):
                return attr
        return None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "loaded_count": len(self._loaded),
            "scan_count": self._scan_count,
            "reload_count": self._reload_count,
            "unload_count": self._unload_count,
            "loaded_strategies": list(self._loaded.keys()),
        }
