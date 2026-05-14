"""ConfigWatcher — async file-change watcher with hot-reload — Push 61.

Polls the config file's mtime every `interval` seconds.
On detected change, calls ConfigLoader.reload() and notifies
all registered on_change callbacks.

Usage::

    loader = ConfigLoader()
    loader.load(Path("argus.yaml"))

    watcher = ConfigWatcher(loader, interval=5.0)
    watcher.add_callback(lambda cfg: print("Config reloaded:", cfg.env))
    asyncio.create_task(watcher.watch())
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable, List, Optional

from core.config.config_loader import ConfigLoader
from core.config.config_schema import ArgusConfig

logger = logging.getLogger(__name__)

OnChangeFn = Callable[[ArgusConfig], None]


class ConfigWatcher:
    """Watches a config file for changes and triggers hot-reload.

    Parameters
    ----------
    loader : ConfigLoader
    interval : float
        Poll interval in seconds (default 5.0).
    """

    def __init__(
        self,
        loader: ConfigLoader,
        interval: float = 5.0,
    ) -> None:
        self._loader = loader
        self._interval = interval
        self._callbacks: List[OnChangeFn] = []
        self._reload_count = 0
        self._last_mtime: Optional[float] = None
        self._running = False

    def add_callback(self, fn: OnChangeFn) -> None:
        self._callbacks.append(fn)

    def remove_callback(self, fn: OnChangeFn) -> None:
        self._callbacks = [cb for cb in self._callbacks if cb is not fn]

    async def watch(self) -> None:
        """Coroutine: continuously poll file mtime and hot-reload on change."""
        self._running = True
        path = self._loader._path
        if path is None:
            logger.warning("ConfigWatcher: no file path set; watching disabled")
            return
        logger.info("ConfigWatcher: watching %s (interval=%.1fs)", path, self._interval)
        while self._running:
            await asyncio.sleep(self._interval)
            try:
                if not Path(path).exists():
                    continue
                mtime = Path(path).stat().st_mtime
                if self._last_mtime is None:
                    self._last_mtime = mtime
                    continue
                if mtime > self._last_mtime:
                    self._last_mtime = mtime
                    logger.info("ConfigWatcher: change detected, reloading")
                    new_config = self._loader.reload()
                    self._reload_count += 1
                    for cb in self._callbacks:
                        try:
                            cb(new_config)
                        except Exception as exc:  # noqa: BLE001
                            logger.error("ConfigWatcher: callback error: %s", exc)
            except Exception as exc:  # noqa: BLE001
                logger.error("ConfigWatcher: watch error: %s", exc)

    def stop(self) -> None:
        self._running = False

    @property
    def reload_count(self) -> int:
        return self._reload_count
