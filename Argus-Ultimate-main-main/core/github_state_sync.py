#!/usr/bin/env python3
"""
core/github_state_sync.py — Argus v6.4.0
==========================================
Git-based state synchronisation layer for the dual-node setup.

The R7525 (LIVE_SERVER) pushes trading state to an orphan Git branch called
``argus-state`` every N seconds.  The PC (PAPER_PC) pulls from that branch to
stay up-to-date without ever touching the main working tree.

All operations use ``git worktree`` so the main branch checkout is never
disturbed.  API keys, secrets, and mnemonics are stripped from config
snapshots before they are committed.

Usage:
    from core.github_state_sync import GitHubStateSync, SyncConfig

    sync = GitHubStateSync(SyncConfig(
        repo_path=".",
        state_dir="data/node_state",
        node_id="r7525-live",
        push_interval_s=300,
        auto_push=True,
    ))
    asyncio.run(sync.init_state_branch())
    asyncio.run(sync.start_sync_loop(NodeRole.LIVE_SERVER))
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Fields whose values must be redacted in config snapshots
_SECRET_KEYWORDS = {"key", "secret", "password", "token", "mnemonic", "passphrase", "seed"}


# ---------------------------------------------------------------------------
# NodeRole import shim  (avoids circular import)
# ---------------------------------------------------------------------------

try:
    from core.node_orchestrator import NodeRole  # type: ignore[import]
except ImportError:
    class NodeRole(str, Enum):  # type: ignore[no-redef]
        LIVE_SERVER = "live_server"
        PAPER_PC    = "paper_pc"
        STANDALONE  = "standalone"


# ---------------------------------------------------------------------------
# SyncConfig
# ---------------------------------------------------------------------------

@dataclass
class SyncConfig:
    """
    Configuration for :class:`GitHubStateSync`.

    Attributes
    ----------
    repo_path:
        Absolute or relative path to the root of the Git repository.
    state_branch:
        Orphan branch used exclusively for state files.
    state_dir:
        Local directory where state JSON files live before being committed.
    node_id:
        Identifier of *this* node (used in health file naming).
    push_interval_s:
        Seconds between automatic pushes (LIVE_SERVER).
    pull_interval_s:
        Seconds between automatic pulls (PAPER_PC).
    auto_push:
        If *True*, start push loop when role is LIVE_SERVER.
    auto_pull:
        If *True*, start pull loop when role is PAPER_PC.
    git_user_email:
        Git commit author e-mail (defaults to a bot address if empty).
    git_user_name:
        Git commit author name.
    remote_name:
        Name of the Git remote (default ``"origin"``).
    worktree_dir:
        Temporary directory used for the git worktree.  A fresh tmpdir is
        created on each push if left as the default empty string.
    """
    repo_path:       str   = "."
    state_branch:    str   = "argus-state"
    state_dir:       str   = "data/node_state"
    node_id:         str   = "argus-node"
    push_interval_s: float = 300.0
    pull_interval_s: float = 300.0
    auto_push:       bool  = True
    auto_pull:       bool  = True
    git_user_email:  str   = "argus-bot@argus.local"
    git_user_name:   str   = "Argus State Bot"
    remote_name:     str   = "origin"
    worktree_dir:    str   = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_git(
    args: List[str],
    cwd: str,
    check: bool = True,
    capture: bool = True,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    """
    Run a ``git`` sub-command and return the :class:`subprocess.CompletedProcess`.

    Parameters
    ----------
    args:
        Sub-command and arguments, e.g. ``["status", "--short"]``.
    cwd:
        Working directory for the git command.
    check:
        If *True* (default) raise :exc:`subprocess.CalledProcessError` on
        non-zero exit.
    capture:
        If *True* (default) capture stdout/stderr.
    env:
        Additional environment variables.  Merged with current ``os.environ``.
    """
    merged_env = {**os.environ}
    if env:
        merged_env.update(env)
    merged_env.setdefault("GIT_TERMINAL_PROMPT", "0")

    cmd = ["git"] + args
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        capture_output=capture,
        text=True,
        env=merged_env,
        timeout=60,
    )


def _strip_secrets(data: Any) -> Any:
    """
    Recursively traverse *data* (dict / list / scalar) and replace the
    value of any key whose name contains a secret keyword with
    ``"<REDACTED>"``.

    Secret keywords (case-insensitive): key, secret, password, token,
    mnemonic, passphrase, seed.
    """
    if isinstance(data, dict):
        return {
            k: ("<REDACTED>" if _is_secret_key(k) else _strip_secrets(v))
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [_strip_secrets(item) for item in data]
    return data


def _is_secret_key(key: str) -> bool:
    key_lower = key.lower()
    return any(kw in key_lower for kw in _SECRET_KEYWORDS)


# ---------------------------------------------------------------------------
# GitHubStateSync
# ---------------------------------------------------------------------------

class GitHubStateSync:
    """
    Manages bidirectional state exchange between Argus nodes via a dedicated
    Git branch.

    The live R7525 node pushes a snapshot of capital, positions, orders, PnL,
    health, and (sanitised) config to the ``argus-state`` branch.  The PC node
    pulls those files to drive its paper-trading comparison engine and
    monitoring dashboard.

    All push operations happen in a throw-away ``git worktree`` so the main
    working tree is never modified.

    Parameters
    ----------
    config:
        :class:`SyncConfig` instance.
    """

    def __init__(self, config: SyncConfig) -> None:
        self.config = config
        self._repo_path = Path(config.repo_path).resolve()
        self._state_dir = Path(config.state_dir)
        self._last_push: float = 0.0
        self._last_pull: float = 0.0
        self._push_count: int  = 0
        self._pull_count: int  = 0
        self._last_error: str  = ""
        self._branch_ready: bool = False

        # Ensure state directory exists
        self._state_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "GitHubStateSync initialised — repo=%s branch=%s node=%s",
            self._repo_path, config.state_branch, config.node_id,
        )

    # ------------------------------------------------------------------
    # Branch initialisation
    # ------------------------------------------------------------------

    async def init_state_branch(self) -> None:
        """
        Ensure the ``argus-state`` orphan branch exists both locally and on
        the remote.

        If the branch already exists (local or remote), this is a no-op.
        Otherwise an orphan branch is created with an initial empty commit.
        """
        try:
            # Check if branch exists locally
            result = _run_git(
                ["branch", "--list", self.config.state_branch],
                cwd=str(self._repo_path),
                check=False,
            )
            branch_exists_local = self.config.state_branch in result.stdout

            # Check remote
            remote_result = _run_git(
                ["ls-remote", "--heads", self.config.remote_name, self.config.state_branch],
                cwd=str(self._repo_path),
                check=False,
            )
            branch_exists_remote = self.config.state_branch in remote_result.stdout

            if branch_exists_local or branch_exists_remote:
                logger.info("State branch '%s' already exists.", self.config.state_branch)
                if not branch_exists_local and branch_exists_remote:
                    # Fetch the remote branch locally
                    _run_git(
                        ["fetch", self.config.remote_name, self.config.state_branch],
                        cwd=str(self._repo_path),
                    )
                    _run_git(
                        ["branch", "--track", self.config.state_branch,
                         f"{self.config.remote_name}/{self.config.state_branch}"],
                        cwd=str(self._repo_path),
                        check=False,
                    )
                self._branch_ready = True
                return

            # Create orphan branch via a worktree to avoid touching HEAD
            with tempfile.TemporaryDirectory(prefix="argus_init_") as tmpdir:
                _run_git(
                    ["worktree", "add", "--orphan", "-b", self.config.state_branch, tmpdir],
                    cwd=str(self._repo_path),
                )
                try:
                    # Write a README to the orphan branch
                    readme = Path(tmpdir) / "README.md"
                    readme.write_text(
                        "# Argus State Branch\n\n"
                        "Auto-managed by `core/github_state_sync.py`.\n"
                        "Do not edit manually.\n"
                    )
                    env = {
                        "GIT_AUTHOR_EMAIL":    self.config.git_user_email,
                        "GIT_AUTHOR_NAME":     self.config.git_user_name,
                        "GIT_COMMITTER_EMAIL": self.config.git_user_email,
                        "GIT_COMMITTER_NAME":  self.config.git_user_name,
                    }
                    _run_git(["add", "README.md"], cwd=tmpdir, env=env)
                    _run_git(
                        ["commit", "-m", "chore: init argus-state branch"],
                        cwd=tmpdir, env=env,
                    )
                    # Push orphan branch
                    _run_git(
                        ["push", self.config.remote_name, self.config.state_branch],
                        cwd=tmpdir,
                    )
                finally:
                    _run_git(
                        ["worktree", "remove", "--force", tmpdir],
                        cwd=str(self._repo_path),
                        check=False,
                    )

            logger.info("Created orphan branch '%s' on remote.", self.config.state_branch)
            self._branch_ready = True

        except subprocess.CalledProcessError as exc:
            self._last_error = str(exc)
            logger.error("init_state_branch failed: %s\n%s", exc, exc.stderr)
            raise

    # ------------------------------------------------------------------
    # Push state
    # ------------------------------------------------------------------

    async def push_state(self, state: Dict[str, Any]) -> None:
        """
        Commit and push a state snapshot to the ``argus-state`` branch.

        The state dict should contain at minimum the keys consumed by the
        :meth:`_build_state_files` method.  Extra keys are silently ignored.

        Internally this method uses a temporary ``git worktree`` to avoid
        touching the main branch checkout.

        Parameters
        ----------
        state:
            Dict with keys ``capital``, ``pnl_history``, ``positions``,
            ``orders``, ``health``, ``config``.
        """
        worktree_root = (
            Path(self.config.worktree_dir)
            if self.config.worktree_dir
            else None
        )
        tmp_ctx = (
            tempfile.TemporaryDirectory(prefix="argus_push_")
            if worktree_root is None
            else _NullContext(str(worktree_root))
        )

        with tmp_ctx as tmpdir:
            try:
                await self._ensure_worktree(tmpdir)
                self._write_state_files(tmpdir, state)
                committed = await self._git_commit_push(tmpdir, state)
                if committed:
                    self._last_push = time.time()
                    self._push_count += 1
                    self._update_last_sync_marker("push")
                    logger.info(
                        "State pushed to branch '%s' (#%d)",
                        self.config.state_branch, self._push_count,
                    )
            except subprocess.CalledProcessError as exc:
                self._last_error = f"push error: {exc.stderr}"
                logger.error("push_state failed: %s", exc.stderr)
                raise
            finally:
                _run_git(
                    ["worktree", "remove", "--force", tmpdir],
                    cwd=str(self._repo_path),
                    check=False,
                )

    async def _ensure_worktree(self, tmpdir: str) -> None:
        """Add a worktree for the state branch at *tmpdir*."""
        # First fetch latest state branch
        _run_git(
            ["fetch", self.config.remote_name, self.config.state_branch],
            cwd=str(self._repo_path),
            check=False,
        )
        # Try to add worktree for existing branch
        result = _run_git(
            ["worktree", "add", tmpdir, self.config.state_branch],
            cwd=str(self._repo_path),
            check=False,
        )
        if result.returncode != 0:
            # Branch might not exist yet — create it
            _run_git(
                ["worktree", "add", "--orphan", "-b", self.config.state_branch, tmpdir],
                cwd=str(self._repo_path),
            )

    def _write_state_files(self, tmpdir: str, state: Dict[str, Any]) -> None:
        """Write all state JSON files into the worktree directory."""
        base = Path(tmpdir)

        def write_json(filename: str, data: Any) -> None:
            (base / filename).write_text(json.dumps(data, indent=2, default=str))

        # capital_state.json
        capital = state.get("capital", {
            "total_usd": 0.0,
            "available_usd": 0.0,
            "locked_usd": 0.0,
            "equity_usd": 0.0,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        # Ensure required keys are present
        for req_key in ("total_usd", "available_usd", "locked_usd", "equity_usd"):
            capital.setdefault(req_key, 0.0)
        capital["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json("capital_state.json", capital)

        # pnl_history.json
        write_json("pnl_history.json", state.get("pnl_history", []))

        # active_positions.json
        write_json("active_positions.json", state.get("positions", []))

        # active_orders.json
        write_json("active_orders.json", state.get("orders", []))

        # health_{node_id}.json
        health = state.get("health", {
            "node_id": self.config.node_id,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        write_json(f"health_{self.config.node_id}.json", health)

        # config_snapshot.json — secrets stripped
        raw_config = state.get("config", {})
        safe_config = _strip_secrets(raw_config)
        safe_config["_snapshot_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        safe_config["_node_id"]      = self.config.node_id
        write_json("config_snapshot.json", safe_config)

    async def _git_commit_push(self, tmpdir: str, state: Dict[str, Any]) -> bool:
        """
        Stage all changed files, commit, and push.

        Returns *True* if a commit was made (i.e. there were changes),
        *False* if the tree was clean.
        """
        env = {
            "GIT_AUTHOR_EMAIL":    self.config.git_user_email,
            "GIT_AUTHOR_NAME":     self.config.git_user_name,
            "GIT_COMMITTER_EMAIL": self.config.git_user_email,
            "GIT_COMMITTER_NAME":  self.config.git_user_name,
        }

        _run_git(["add", "-A"], cwd=tmpdir, env=env)

        # Check if there is anything to commit
        status = _run_git(["status", "--porcelain"], cwd=tmpdir, env=env)
        if not status.stdout.strip():
            logger.debug("push_state: nothing to commit — tree is clean.")
            return False

        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        msg = f"state: {self.config.node_id} snapshot @ {ts}"
        _run_git(["commit", "-m", msg], cwd=tmpdir, env=env)
        _run_git(
            ["push", self.config.remote_name, self.config.state_branch],
            cwd=tmpdir,
        )
        return True

    # ------------------------------------------------------------------
    # Pull state
    # ------------------------------------------------------------------

    async def pull_state(self) -> Dict[str, Any]:
        """
        Read the current state from the ``argus-state`` branch WITHOUT
        switching branches in the main worktree.

        Uses ``git show <branch>:<file>`` to read file contents directly
        from the object store.

        Returns
        -------
        dict
            Keys: ``capital``, ``pnl_history``, ``positions``, ``orders``,
            ``health``, ``config``, ``_pulled_utc``.
        """
        # Fetch latest remote state branch
        _run_git(
            ["fetch", self.config.remote_name, self.config.state_branch],
            cwd=str(self._repo_path),
            check=False,
        )

        ref = f"{self.config.remote_name}/{self.config.state_branch}"

        def read_json(filename: str, default: Any = None) -> Any:
            try:
                result = _run_git(
                    ["show", f"{ref}:{filename}"],
                    cwd=str(self._repo_path),
                )
                return json.loads(result.stdout)
            except (subprocess.CalledProcessError, json.JSONDecodeError):
                return default if default is not None else {}

        pulled: Dict[str, Any] = {
            "capital":     read_json("capital_state.json",     {}),
            "pnl_history": read_json("pnl_history.json",       []),
            "positions":   read_json("active_positions.json",  []),
            "orders":      read_json("active_orders.json",     []),
            "health":      read_json(f"health_{self.config.node_id}.json", {}),
            "config":      read_json("config_snapshot.json",   {}),
            "_pulled_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        self._last_pull = time.time()
        self._pull_count += 1
        self._update_last_sync_marker("pull")
        logger.info(
            "State pulled from branch '%s' (#%d)",
            self.config.state_branch, self._pull_count,
        )
        return pulled

    # ------------------------------------------------------------------
    # Sync loop
    # ------------------------------------------------------------------

    async def start_sync_loop(self, node_role: "NodeRole") -> None:
        """
        Start the appropriate sync loop for the given *node_role*.

        * **LIVE_SERVER** — push every ``push_interval_s`` seconds (if
          ``auto_push`` is *True*).
        * **PAPER_PC** — pull every ``pull_interval_s`` seconds (if
          ``auto_pull`` is *True*).
        * **STANDALONE** — no-op; returns immediately.

        This coroutine runs indefinitely (until cancelled).

        Parameters
        ----------
        node_role:
            :class:`NodeRole` of the calling node.
        """
        if node_role == NodeRole.LIVE_SERVER and self.config.auto_push:
            logger.info(
                "Starting state PUSH loop (interval=%.0f s)",
                self.config.push_interval_s,
            )
            await self._push_loop()
        elif node_role == NodeRole.PAPER_PC and self.config.auto_pull:
            logger.info(
                "Starting state PULL loop (interval=%.0f s)",
                self.config.pull_interval_s,
            )
            await self._pull_loop()
        else:
            logger.info("Sync loop not started (role=%s).", node_role)

    async def _push_loop(self) -> None:
        """Periodically collect state and push it."""
        while True:
            try:
                state = await self._collect_local_state()
                await self.push_state(state)
            except Exception as exc:
                self._last_error = str(exc)
                logger.error("Sync push loop error: %s", exc)
            await asyncio.sleep(self.config.push_interval_s)

    async def _pull_loop(self) -> None:
        """Periodically pull state from remote."""
        while True:
            try:
                await self.pull_state()
            except Exception as exc:
                self._last_error = str(exc)
                logger.error("Sync pull loop error: %s", exc)
            await asyncio.sleep(self.config.pull_interval_s)

    async def _collect_local_state(self) -> Dict[str, Any]:
        """
        Read JSON state files from the local ``state_dir`` and assemble
        them into the state dict expected by :meth:`push_state`.
        """
        def load(fname: str, default: Any = None) -> Any:
            p = self._state_dir / fname
            if not p.exists():
                return default if default is not None else {}
            try:
                return json.loads(p.read_text())
            except Exception:
                return default if default is not None else {}

        return {
            "capital":     load("capital_state.json",    {
                "total_usd": 0.0, "available_usd": 0.0,
                "locked_usd": 0.0, "equity_usd": 0.0,
            }),
            "pnl_history": load("pnl_history.json",      []),
            "positions":   load("active_positions.json", []),
            "orders":      load("active_orders.json",    []),
            "health":      load(f"health_{self.config.node_id}.json", {}),
            "config":      load("config_snapshot.json",  {}),
        }

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def get_sync_status(self) -> str:
        """
        Return a one-line human-readable status string.

        Format: ``"push: N, last: 2024-01-01T12:00:00Z, error: none"``
        """
        last_push = (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._last_push))
            if self._last_push else "never"
        )
        last_pull = (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._last_pull))
            if self._last_pull else "never"
        )
        error_str = self._last_error or "none"
        return (
            f"push_count={self._push_count} last_push={last_push} "
            f"pull_count={self._pull_count} last_pull={last_pull} "
            f"error={error_str}"
        )

    def get_remote_node_health(self) -> Dict[str, Any]:
        """
        Read the remote node's health file from the ``argus-state`` branch
        without pulling all state.

        Returns an empty dict if the branch or file does not exist.
        """
        ref = f"{self.config.remote_name}/{self.config.state_branch}"
        try:
            _run_git(
                ["fetch", self.config.remote_name, self.config.state_branch],
                cwd=str(self._repo_path),
                check=False,
            )
            result = _run_git(
                ["show", f"{ref}:health_{self.config.node_id}.json"],
                cwd=str(self._repo_path),
            )
            return json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            return {}

    def get_last_push_time(self) -> float:
        """Return the Unix timestamp of the last successful push (0 if never)."""
        return self._last_push

    def get_last_pull_time(self) -> float:
        """Return the Unix timestamp of the last successful pull (0 if never)."""
        return self._last_pull

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_last_sync_marker(self, direction: str) -> None:
        """Write a last-sync timestamp to the local state_dir."""
        marker = self._state_dir / "last_sync.txt"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            marker.write_text(f"{direction}:{ts}")
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "GitHubStateSync":
        await self.init_state_branch()
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"GitHubStateSync(node={self.config.node_id!r}, "
            f"branch={self.config.state_branch!r}, "
            f"push_count={self._push_count}, pull_count={self._pull_count})"
        )


# ---------------------------------------------------------------------------
# Null context manager (used when worktree_dir is pre-specified)
# ---------------------------------------------------------------------------

class _NullContext:
    """Context manager that does nothing — yields the pre-supplied path."""

    def __init__(self, path: str) -> None:
        self._path = path

    def __enter__(self) -> str:
        return self._path

    def __exit__(self, *_: Any) -> None:
        pass
