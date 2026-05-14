"""
Configuration Auto-Tuner — reads logs and adjusts config for self-healing.

Rules:
  - >50% trades suppressed by "small_delta" → lower min_delta_pct by 50%
  - >50% trades suppressed by "liquidity_thin" → enable assume_normal_without_l2
  - "TimeoutError" in >30% of cycles → increase latency_spike_ms by 50%
  - "STRATEGY_COOLDOWN" blocking >50% signals → reduce cooldown_minutes
  - Cycle time consistently >120s → reduce trading_pairs count

Features:
  - Dry-run mode (suggest but don't apply)
  - SQLite persistence at data/config_tuning.db
  - Tuning history with timestamps
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@dataclass
class TuningChange:
    """A single config change suggestion or application."""
    key: str
    old_value: Any
    new_value: Any
    reason: str
    applied: bool = False
    timestamp: float = field(default_factory=time.time)


class ConfigAutoTuner:
    """
    Reads recent log data, detects patterns, and suggests/applies config
    changes to improve system performance.
    """

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        dry_run: bool = True,
    ) -> None:
        self.dry_run = dry_run
        self._db_path = Path(db_path or "data/config_tuning.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._pending_changes: List[TuningChange] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_and_tune(
        self,
        log_path: str | Path,
        config_path: str | Path,
        *,
        lookback_lines: int = 5000,
    ) -> List[TuningChange]:
        """
        Read recent log lines, analyze patterns, and produce tuning changes.
        If dry_run is False, changes are applied to the config file.
        Returns list of TuningChange objects.
        """
        log_path = Path(log_path)
        config_path = Path(config_path)

        if not log_path.exists():
            logger.warning("Log file not found: %s", log_path)
            return []

        lines = self._read_recent_lines(log_path, lookback_lines)
        if not lines:
            logger.info("No log lines to analyze")
            return []

        config = self._load_config(config_path)
        if config is None:
            return []

        changes: List[TuningChange] = []

        # Rule 1: small_delta suppression
        change = self._check_small_delta(lines, config)
        if change:
            changes.append(change)

        # Rule 2: liquidity_thin suppression
        change = self._check_liquidity_thin(lines, config)
        if change:
            changes.append(change)

        # Rule 3: TimeoutError rate
        change = self._check_timeout_rate(lines, config)
        if change:
            changes.append(change)

        # Rule 4: STRATEGY_COOLDOWN blocking
        change = self._check_cooldown_blocking(lines, config)
        if change:
            changes.append(change)

        # Rule 5: Slow cycle time
        change = self._check_cycle_time(lines, config)
        if change:
            changes.append(change)

        if changes and not self.dry_run:
            self.apply_changes({c.key: c.new_value for c in changes}, config_path)
            for c in changes:
                c.applied = True

        # Persist to DB
        for c in changes:
            self._save_change(c)

        self._pending_changes = changes
        return changes

    def apply_changes(
        self,
        changes: Dict[str, Any],
        config_path: str | Path,
    ) -> bool:
        """
        Write changes to the config YAML file.
        Keys use dot notation: "execution_engine.latency_spike_ms" etc.
        """
        config_path = Path(config_path)
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            for dotted_key, value in changes.items():
                parts = dotted_key.split(".")
                d = data
                for part in parts[:-1]:
                    if part not in d or not isinstance(d[part], dict):
                        d[part] = {}
                    d = d[part]
                d[parts[-1]] = value
                logger.info("Config tuned: %s = %s", dotted_key, value)

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)

            return True
        except Exception as exc:
            logger.error("Failed to apply config changes: %s", exc)
            return False

    def get_tuning_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent tuning history from DB."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM tuning_history ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            logger.error("DB read error: %s", exc)
            return []

    def get_pending_changes(self) -> List[TuningChange]:
        """Return the last set of pending/applied changes."""
        return list(self._pending_changes)

    # ------------------------------------------------------------------
    # Rule implementations
    # ------------------------------------------------------------------

    def _check_small_delta(
        self, lines: List[str], config: Dict[str, Any]
    ) -> Optional[TuningChange]:
        """If >50% trades suppressed by 'small_delta' → lower min_delta_pct by 50%."""
        total_signals = 0
        small_delta_count = 0
        for line in lines:
            if "signal" in line.lower() or "trade" in line.lower():
                total_signals += 1
            if "small_delta" in line:
                small_delta_count += 1

        if total_signals == 0:
            return None

        ratio = small_delta_count / total_signals
        if ratio <= 0.5:
            return None

        edge_gate = config.get("edge_cost_gate") or {}
        current = float(edge_gate.get("min_delta_pct", 0.5))
        new_val = round(current * 0.5, 4)

        return TuningChange(
            key="edge_cost_gate.min_delta_pct",
            old_value=current,
            new_value=new_val,
            reason=f"small_delta suppressed {ratio:.0%} of signals → halving min_delta_pct",
        )

    def _check_liquidity_thin(
        self, lines: List[str], config: Dict[str, Any]
    ) -> Optional[TuningChange]:
        """If >50% trades suppressed by 'liquidity_thin' → enable assume_normal_without_l2."""
        total_signals = 0
        liq_count = 0
        for line in lines:
            if "signal" in line.lower() or "trade" in line.lower():
                total_signals += 1
            if "liquidity_thin" in line:
                liq_count += 1

        if total_signals == 0:
            return None

        ratio = liq_count / total_signals
        if ratio <= 0.5:
            return None

        exec_cfg = config.get("execution_engine") or {}
        current = bool(exec_cfg.get("assume_normal_without_l2", False))
        if current:
            return None  # Already enabled

        return TuningChange(
            key="execution_engine.assume_normal_without_l2",
            old_value=False,
            new_value=True,
            reason=f"liquidity_thin suppressed {ratio:.0%} of signals → enabling assume_normal_without_l2",
        )

    def _check_timeout_rate(
        self, lines: List[str], config: Dict[str, Any]
    ) -> Optional[TuningChange]:
        """If 'TimeoutError' in >30% of cycles → increase latency_spike_ms by 50%."""
        cycle_count = 0
        timeout_count = 0
        for line in lines:
            if re.search(r"Cycle.*complete", line, re.IGNORECASE):
                cycle_count += 1
            if "TimeoutError" in line:
                timeout_count += 1

        if cycle_count == 0:
            return None

        ratio = timeout_count / cycle_count
        if ratio <= 0.3:
            return None

        exec_cfg = config.get("execution_engine") or {}
        current = int(exec_cfg.get("latency_spike_ms", 500))
        new_val = int(current * 1.5)

        return TuningChange(
            key="execution_engine.latency_spike_ms",
            old_value=current,
            new_value=new_val,
            reason=f"TimeoutError in {ratio:.0%} of cycles → increasing latency_spike_ms",
        )

    def _check_cooldown_blocking(
        self, lines: List[str], config: Dict[str, Any]
    ) -> Optional[TuningChange]:
        """If 'STRATEGY_COOLDOWN' blocking >50% signals → reduce cooldown_minutes."""
        total_signals = 0
        cooldown_count = 0
        for line in lines:
            if "signal" in line.lower():
                total_signals += 1
            if "STRATEGY_COOLDOWN" in line:
                cooldown_count += 1

        if total_signals == 0:
            return None

        ratio = cooldown_count / total_signals
        if ratio <= 0.5:
            return None

        strategies = config.get("strategies") or {}
        current = int(strategies.get("cooldown_minutes", 60))
        new_val = max(5, current // 2)

        return TuningChange(
            key="strategies.cooldown_minutes",
            old_value=current,
            new_value=new_val,
            reason=f"STRATEGY_COOLDOWN blocked {ratio:.0%} of signals → halving cooldown_minutes",
        )

    def _check_cycle_time(
        self, lines: List[str], config: Dict[str, Any]
    ) -> Optional[TuningChange]:
        """If cycle time consistently >120s → reduce trading_pairs count."""
        slow_cycles = 0
        total_cycles = 0
        for line in lines:
            m = re.search(r"Cycle.*complete.*?(\d+(?:\.\d+)?)\s*s", line, re.IGNORECASE)
            if m:
                total_cycles += 1
                if float(m.group(1)) > 120:
                    slow_cycles += 1

        if total_cycles < 5:
            return None

        ratio = slow_cycles / total_cycles
        if ratio <= 0.5:
            return None

        pairs = config.get("trading_pairs") or []
        if isinstance(pairs, list) and len(pairs) > 2:
            new_pairs = pairs[: len(pairs) // 2]
            return TuningChange(
                key="trading_pairs",
                old_value=pairs,
                new_value=new_pairs,
                reason=f"{ratio:.0%} of cycles >120s → reducing trading_pairs from {len(pairs)} to {len(new_pairs)}",
            )

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_recent_lines(self, log_path: Path, n: int) -> List[str]:
        """Read last N lines from a log file."""
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                # Read all lines and take last N (simple for moderate files)
                all_lines = f.readlines()
                return all_lines[-n:] if len(all_lines) > n else all_lines
        except OSError as exc:
            logger.error("Failed to read log: %s", exc)
            return []

    def _load_config(self, config_path: Path) -> Optional[Dict[str, Any]]:
        """Load YAML config file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:
            logger.error("Failed to load config: %s", exc)
            return None

    def _init_db(self) -> None:
        """Create the tuning_history table if it doesn't exist."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tuning_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    key TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    reason TEXT,
                    applied INTEGER DEFAULT 0
                )
            """)
            conn.commit()
            conn.close()
        except sqlite3.Error as exc:
            logger.error("DB init error: %s", exc)

    def _save_change(self, change: TuningChange) -> None:
        """Persist a tuning change to SQLite."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute(
                """INSERT INTO tuning_history (timestamp, key, old_value, new_value, reason, applied)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    change.timestamp,
                    change.key,
                    json.dumps(change.old_value),
                    json.dumps(change.new_value),
                    change.reason,
                    1 if change.applied else 0,
                ),
            )
            conn.commit()
            conn.close()
        except sqlite3.Error as exc:
            logger.error("DB write error: %s", exc)
