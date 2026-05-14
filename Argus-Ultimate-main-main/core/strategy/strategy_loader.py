"""StrategyLoader — hot-swap file/directory loader — Push 58.

Loads Python files, discovers AbstractStrategy subclasses,
and registers them into a StrategyRegistry. Supports async
file-watching for hot-reload on file changes.

Usage::

    registry = StrategyRegistry()
    loader = StrategyLoader(registry)
    loader.load_file(Path("strategies/my_strategy.py"))
    loader.load_directory(Path("strategies/"))

    # Hot-reload watcher (run as asyncio task)
    asyncio.create_task(loader.watch_directory(Path("strategies/"), interval=5.0))
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from core.strategy.base_strategy import AbstractStrategy
from core.strategy.strategy_registry import StrategyRegistry

logger = logging.getLogger(__name__)


class StrategyLoader:
    """Loads strategy classes from Python files into a StrategyRegistry.

    Parameters
    ----------
    registry : StrategyRegistry
    """

    def __init__(self, registry: StrategyRegistry) -> None:
        self._registry = registry
        self._loaded_files: Dict[Path, float] = {}  # path -> mtime
        self._reload_count = 0
        self._last_reload_at: Optional[float] = None

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def load_file(self, path: Path) -> List[str]:
        """Import a .py file and register all AbstractStrategy subclasses.

        Returns list of registered strategy names.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Strategy file not found: {path}")
        if path.suffix != ".py":
            raise ValueError(f"Expected .py file, got: {path}")

        module_name = f"_argus_strategy_{path.stem}_{id(path)}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except Exception as exc:
            logger.error("StrategyLoader: failed to load %s: %s", path, exc)
            raise

        registered: List[str] = []
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, AbstractStrategy)
                and obj is not AbstractStrategy
                and not inspect.isabstract(obj)
                and obj.__module__ == module_name
            ):
                try:
                    self._registry.register(obj)
                    inst = obj.__new__(obj)
                    AbstractStrategy.__init__(inst)
                    registered.append(inst.metadata.name)
                except Exception as exc:
                    logger.warning("StrategyLoader: skipping %s: %s", obj.__name__, exc)

        mtime = path.stat().st_mtime
        self._loaded_files[path] = mtime
        self._reload_count += 1
        self._last_reload_at = time.time()

        logger.info("StrategyLoader: loaded %s -> %s", path.name, registered)
        return registered

    def load_directory(self, directory: Path) -> Dict[Path, List[str]]:
        """Load all .py files in a directory (non-recursive).

        Returns dict of {path: [strategy_names]}.
        """
        directory = Path(directory).resolve()
        results: Dict[Path, List[str]] = {}
        for py_file in sorted(directory.glob("*.py")):
            if py_file.stem.startswith("_"):
                continue
            try:
                names = self.load_file(py_file)
                results[py_file] = names
            except Exception as exc:
                logger.warning("StrategyLoader: skipping %s: %s", py_file, exc)
                results[py_file] = []
        return results

    # ------------------------------------------------------------------
    # Hot-reload watcher
    # ------------------------------------------------------------------

    async def watch_directory(
        self, directory: Path, interval: float = 5.0
    ) -> None:
        """Async coroutine: poll directory for changed .py files and reload."""
        directory = Path(directory).resolve()
        logger.info(
            "StrategyLoader: watching %s (interval=%.1fs)", directory, interval
        )
        while True:
            await asyncio.sleep(interval)
            if not directory.exists():
                continue
            for py_file in directory.glob("*.py"):
                if py_file.stem.startswith("_"):
                    continue
                try:
                    current_mtime = py_file.stat().st_mtime
                    known_mtime = self._loaded_files.get(py_file)
                    if known_mtime is None or current_mtime > known_mtime:
                        logger.info(
                            "StrategyLoader: hot-reloading %s", py_file.name
                        )
                        self.load_file(py_file)
                except Exception as exc:
                    logger.error(
                        "StrategyLoader: hot-reload error %s: %s", py_file, exc
                    )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def reload_count(self) -> int:
        return self._reload_count

    @property
    def last_reload_at(self) -> Optional[float]:
        return self._last_reload_at

    @property
    def loaded_files(self) -> List[Path]:
        return list(self._loaded_files.keys())
