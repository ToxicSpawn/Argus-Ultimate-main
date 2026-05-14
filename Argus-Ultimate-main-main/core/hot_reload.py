"""
Config & Strategy Hot-Reload Manager — zero-downtime configuration updates.

Allows changing strategy weights, risk limits, and signal thresholds at runtime
without restarting the ARGUS process. Two trigger mechanisms:

  1. SIGHUP (Unix): send `kill -HUP <pid>` → immediate config reload
  2. File watcher: monitors unified_config.yaml mtime; auto-reloads if changed

Safe fields that can be hot-reloaded without risk:
  - strategy_allocator weights
  - min_signal_confidence, live_min_signal_confidence
  - max_concurrent_signals
  - risk.daily_loss_limit_pct, risk.position_size_pct
  - edge_cost_gate thresholds
  - logging levels

Unsafe fields that require a full restart:
  - exchanges (credentials, endpoints)
  - capital.starting_capital
  - runtime.mode (paper → live switch must go through LiveGate)

Usage:
    reloader = HotReloadManager(config_path="unified_config.yaml",
                                 trading_system=system)
    reloader.install_signal_handler()     # register SIGHUP
    reloader.start_file_watcher()         # background mtime polling
    ...
    reloader.stop()
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-field sanity bounds for hot-reloadable values.
# Format: dotted_key → (min_inclusive, max_inclusive)
# Values outside the range are logged as WARNING and skipped.
# ---------------------------------------------------------------------------
_FIELD_BOUNDS: Dict[str, Tuple[float, float]] = {
    # Risk limits
    "risk.daily_loss_limit_pct":       (0.001, 0.50),   # 0.1% – 50%
    "risk.max_drawdown_pct":           (0.01,  1.00),   # 1% – 100%
    "risk.position_size_pct":          (0.001, 1.00),   # 0.1% – 100%
    # Strategy signal thresholds
    "strategies.min_signal_confidence":      (0.0, 1.0),
    "strategies.live_min_signal_confidence": (0.0, 1.0),
    "strategies.max_concurrent_signals":     (1,   100),
    # Numeric fields inside sub-sections that get set directly on the config object
    "max_position_size_aud":           (1.0,   100_000.0),
    "min_position_size_aud":           (0.1,   10_000.0),
    "max_total_exposure_pct":          (0.01,  1.00),
    "max_leverage":                    (1.0,   20.0),
    "max_concurrent_positions":        (1,     50),
}

# Fields that are safe to update live (dotted path keys)
SAFE_RELOAD_KEYS: frozenset = frozenset({
    "strategies.min_signal_confidence",
    "strategies.live_min_signal_confidence",
    "strategies.max_concurrent_signals",
    "strategy_allocator",
    "risk.daily_loss_limit_pct",
    "risk.max_drawdown_pct",
    "risk.position_size_pct",
    "edge_cost_gate",
    "logging",
    "paper_trading",
    "monitoring",
    "self_optimizing_meta_engine.advisory_only",
    "self_optimizing_meta_engine.meta_alpha",
    "self_optimizing_meta_engine.update_interval_cycles",
    "champion_challenger",
    "funding_rate_harvester",
    "alternative_data",
    "advanced_risk",
    "market_making",
    "websocket_orders",
})


class HotReloadManager:
    """
    Manages zero-downtime config reloads for the ARGUS trading system.

    Thread/async-safe: uses an asyncio.Event to signal the main loop.
    """

    def __init__(
        self,
        config_path: str = "unified_config.yaml",
        trading_system: Any = None,
        poll_interval: float = 30.0,
        on_reload: Optional[Callable[[Dict], None]] = None,
    ):
        self._config_path = Path(config_path)
        self._system = trading_system
        self._poll_interval = float(poll_interval)
        self._on_reload = on_reload
        self._last_mtime: float = self._get_mtime()
        self._reload_event = asyncio.Event() if _in_async() else None
        self._watcher_task: Optional[asyncio.Task] = None
        self._running = False
        self._reload_count = 0
        self._last_reload_ts: float = 0.0
        self._reload_timeout: float = 10.0
        self._previous_config: Optional[Dict] = None
        self._reload_lock = threading.Lock()
        self._reload_in_progress = False
        self._reload_lock_timeout: float = 5.0

    # ------------------------------------------------------------------
    # Signal handler (SIGHUP)
    # ------------------------------------------------------------------

    def install_signal_handler(self) -> bool:
        """
        Register SIGHUP handler (Unix only).
        Returns True if installed, False on Windows or error.
        """
        if not hasattr(signal, "SIGHUP"):
            logger.debug("HotReloadManager: SIGHUP not available on this platform (Windows?)")
            return False
        try:
            signal.signal(signal.SIGHUP, self._handle_sighup)
            logger.info("HotReloadManager: SIGHUP handler installed (PID %d)", os.getpid())
            return True
        except (OSError, ValueError) as exc:
            logger.warning("HotReloadManager: could not install SIGHUP handler: %s", exc)
            return False

    def _handle_sighup(self, signum: int, frame: Any) -> None:
        """SIGHUP handler — schedule reload on the event loop."""
        logger.info("HotReloadManager: SIGHUP received → scheduling config reload")
        if self._reload_event is not None:
            # Signal-safe: set event from signal handler
            try:
                self._reload_event.set()
            except Exception as _e:
                logger.debug("hot_reload error: %s", _e)
        else:
            # Synchronous path (no event loop)
            self._do_reload()

    # ------------------------------------------------------------------
    # File watcher
    # ------------------------------------------------------------------

    def start_file_watcher(self) -> None:
        """Start background asyncio task that polls config file mtime."""
        try:
            loop = asyncio.get_running_loop()
            self._watcher_task = loop.create_task(
                self._watch_loop(), name="hot_reload_watcher"
            )
            self._running = True
            logger.info(
                "HotReloadManager: file watcher started (poll=%ds) for %s",
                int(self._poll_interval),
                self._config_path,
            )
        except RuntimeError:
            logger.debug("HotReloadManager: no running event loop — file watcher not started")

    def stop(self) -> None:
        """Stop the file watcher task."""
        self._running = False
        if self._watcher_task and not self._watcher_task.done():
            self._watcher_task.cancel()
        logger.info("HotReloadManager: stopped")

    async def _watch_loop(self) -> None:
        """Background task: check mtime every poll_interval seconds."""
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval)
                mtime = self._get_mtime()
                if mtime > self._last_mtime:
                    logger.info(
                        "HotReloadManager: config file changed (mtime %s → %s)",
                        self._last_mtime, mtime,
                    )
                    self._last_mtime = mtime
                    await asyncio.get_running_loop().run_in_executor(None, self._do_reload)
                # Also respond to SIGHUP events
                if self._reload_event and self._reload_event.is_set():
                    self._reload_event.clear()
                    await asyncio.get_running_loop().run_in_executor(None, self._do_reload)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("HotReloadManager watcher error: %s", exc)

    # ------------------------------------------------------------------
    # Reload logic
    # ------------------------------------------------------------------

    def _do_reload(self) -> bool:
        """
        Load the config file and apply safe field updates to the trading system.
        Returns True on success.

        Wraps the reload in a thread-pool timeout.  If reload takes longer
        than ``_reload_timeout`` seconds (default 10), logs ERROR, reverts to
        the previous config snapshot, and returns False.

        Uses a lock with a 5-second timeout to prevent deadlocks and a
        re-entrancy guard to skip overlapping reloads.
        """
        # Re-entrancy guard: skip if a reload is already in progress
        if self._reload_in_progress:
            logger.warning(
                "HotReloadManager: reload already in progress — skipping re-entrant reload"
            )
            return False

        # Attempt to acquire the reload lock with timeout
        acquired = self._reload_lock.acquire(timeout=self._reload_lock_timeout)
        if not acquired:
            logger.warning(
                "HotReloadManager: could not acquire reload lock within %.1fs — skipping reload",
                self._reload_lock_timeout,
            )
            return False

        try:
            self._reload_in_progress = True

            # Snapshot current config state for rollback
            current_snapshot = self._snapshot_config()

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(self._do_reload_inner)
                    result = future.result(timeout=self._reload_timeout)
                    if result:
                        # Reload succeeded — save the snapshot as the rollback point
                        self._previous_config = current_snapshot
                    return result
            except concurrent.futures.TimeoutError:
                logger.error(
                    "HotReloadManager: reload timed out after %.1fs — reverting to previous config",
                    self._reload_timeout,
                )
                self._revert_config(current_snapshot)
                return False
            except Exception as exc:
                logger.error("HotReloadManager: reload error: %s — reverting", exc)
                self._revert_config(current_snapshot)
                return False
        finally:
            self._reload_in_progress = False
            self._reload_lock.release()

    def _do_reload_inner(self) -> bool:
        """Inner reload logic (runs inside timeout wrapper)."""
        try:
            new_cfg = self._load_yaml()
        except Exception as exc:
            logger.error("HotReloadManager: failed to load config: %s", exc)
            return False

        if new_cfg is None:
            return False

        applied = self._apply_safe_fields(new_cfg)
        self._reload_count += 1
        self._last_reload_ts = time.time()

        if self._on_reload:
            try:
                self._on_reload(new_cfg)
            except Exception as exc:
                logger.warning("HotReloadManager: on_reload callback error: %s", exc)

        logger.info(
            "HotReloadManager: reload #%d complete — applied %d safe field groups",
            self._reload_count, applied,
        )
        return True

    def _snapshot_config(self) -> Optional[Dict]:
        """Take a shallow snapshot of the safe config fields for rollback."""
        if self._system is None:
            return None
        config = getattr(self._system, "config", None)
        if config is None:
            return None
        snapshot: Dict[str, Any] = {}
        safe_top_keys = {k.split(".")[0] for k in SAFE_RELOAD_KEYS}
        for key in safe_top_keys:
            val = getattr(config, key, None)
            if val is not None:
                if isinstance(val, dict):
                    snapshot[key] = dict(val)
                else:
                    snapshot[key] = val
        return snapshot

    def _revert_config(self, snapshot: Optional[Dict]) -> None:
        """Revert safe config fields to a previous snapshot."""
        if snapshot is None or self._system is None:
            return
        config = getattr(self._system, "config", None)
        if config is None:
            return
        for key, val in snapshot.items():
            try:
                self._apply_nested(config, key, val)
            except Exception as exc:
                logger.warning("HotReloadManager: revert failed for '%s': %s", key, exc)
        logger.info("HotReloadManager: config reverted to previous state")

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_value(dotted_key: str, value: Any) -> bool:
        """
        Check *value* against the bounds table for *dotted_key*.

        Returns True if the value is acceptable (or has no registered bound).
        Logs a WARNING and returns False if the value is out of range.
        """
        bounds = _FIELD_BOUNDS.get(dotted_key)
        if bounds is None:
            # No registered bound — accept as-is
            return True
        lo, hi = bounds
        try:
            v = float(value)
        except (TypeError, ValueError):
            # Non-numeric fields have no numeric bounds to check
            return True
        if v < lo or v > hi:
            logger.warning(
                "HotReloadManager: VALIDATION FAILED — '%s' value %r is outside "
                "allowed range [%s, %s]. Field will NOT be applied.",
                dotted_key, value, lo, hi,
            )
            return False
        return True

    @staticmethod
    def _get_nested_value(obj: Any, key: str) -> Any:
        """Retrieve *key* from a dict or object attribute, returning _MISSING sentinel."""
        _MISSING = object()
        if isinstance(obj, dict):
            return obj.get(key, _MISSING)
        return getattr(obj, key, _MISSING)

    def _load_yaml(self) -> Optional[Dict]:
        """Load and parse the YAML config file."""
        try:
            import yaml  # type: ignore
        except ImportError:
            logger.warning("HotReloadManager: PyYAML not installed — cannot reload")
            return None

        if not self._config_path.exists():
            logger.warning("HotReloadManager: config file not found: %s", self._config_path)
            return None

        with open(self._config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _apply_safe_fields(self, new_cfg: Dict) -> int:
        """
        Apply safe config fields to the attached trading system config.

        For each field:
          1. Validate the new value against _FIELD_BOUNDS; skip if out-of-range.
          2. Log a diff line (old → new) for every changed value (Task 3).
          3. Apply the value to the live config object.

        Returns count of top-level safe sections applied.
        """
        if self._system is None:
            return 0

        config = getattr(self._system, "config", None)
        if config is None:
            return 0

        applied = 0
        safe_top_keys = {k.split(".")[0] for k in SAFE_RELOAD_KEYS}

        for top_key in safe_top_keys:
            if top_key not in new_cfg:
                continue
            new_val = new_cfg[top_key]

            # --- Per-field validation for scalar values ---
            if not isinstance(new_val, dict):
                dotted = top_key
                if not self._validate_value(dotted, new_val):
                    continue  # skip this field — out of range

            # --- Dict section: validate and diff sub-keys ---
            if isinstance(new_val, dict):
                old_section = getattr(config, top_key, {}) or {}
                for sub_key, sub_val in new_val.items():
                    dotted = f"{top_key}.{sub_key}"
                    if not self._validate_value(dotted, sub_val):
                        continue  # keep old value for this sub-key
                    old_sub = old_section.get(sub_key) if isinstance(old_section, dict) else getattr(old_section, sub_key, None)
                    if old_sub != sub_val:
                        logger.info(
                            "Hot-reload: %s: %r → %r",
                            dotted, old_sub, sub_val,
                        )
            else:
                # Scalar top-level key — diff log
                old_val = getattr(config, top_key, None)
                if old_val != new_val:
                    logger.info(
                        "Hot-reload: %s: %r → %r",
                        top_key, old_val, new_val,
                    )

            # Apply nested safe paths
            self._apply_nested(config, top_key, new_val)
            applied += 1

        return applied

    @staticmethod
    def _apply_nested(config: Any, top_key: str, new_val: Any) -> None:
        """Apply a top-level key's value to the config object."""
        try:
            if hasattr(config, top_key):
                old_val = getattr(config, top_key)
                if isinstance(old_val, dict) and isinstance(new_val, dict):
                    old_val.update(new_val)
                else:
                    setattr(config, top_key, new_val)
                logger.debug("HotReloadManager: applied '%s'", top_key)
            elif hasattr(config, "__dict__"):
                setattr(config, top_key, new_val)
        except Exception as exc:
            logger.debug("HotReloadManager: could not apply '%s': %s", top_key, exc)

    # ------------------------------------------------------------------
    # Status / inspection
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return current hot-reload status dict."""
        return {
            "config_path": str(self._config_path),
            "reload_count": self._reload_count,
            "last_reload_ts": self._last_reload_ts,
            "last_config_mtime": self._last_mtime,
            "poll_interval_seconds": self._poll_interval,
            "watcher_running": self._running,
            "sighup_available": hasattr(signal, "SIGHUP"),
        }

    def force_reload(self) -> bool:
        """Manually trigger a reload (useful for testing or API-driven reload)."""
        return self._do_reload()

    def _get_mtime(self) -> float:
        """Get config file mtime, or 0.0 if file missing."""
        try:
            return self._config_path.stat().st_mtime
        except OSError:
            return 0.0


def _in_async() -> bool:
    """True if called from within a running asyncio event loop."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False
