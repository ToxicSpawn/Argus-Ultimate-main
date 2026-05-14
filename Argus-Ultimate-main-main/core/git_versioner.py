"""
Git Versioner — auto-commits generated strategy files with rollback support.

Every generated strategy that passes review + sandbox gets committed to git
with a structured message. If the strategy later fails in production, we can
roll back via git revert. This gives full audit trail and reversibility.

Operations:
  - commit_generation()  — commit a new generated file
  - tag_active()          — tag a strategy as currently active
  - rollback()            — revert to previous version
  - list_history()        — show all generated files in git log

The versioner ONLY commits files in generated_strategies/. It never touches
the rest of the ARGUS codebase.
"""
from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CommitRecord:
    """Record of one git commit by the versioner."""
    timestamp: float
    sha: str
    file_path: Path
    message: str
    operation: str  # "generate", "promote", "retire", "rollback"


class GitVersioner:
    """
    Manages git commits for generated strategies.

    Usage::

        versioner = GitVersioner(repo_path=".")

        # Commit a new generation
        sha = versioner.commit_generation(
            file_path=Path("generated_strategies/candidates/foo.py"),
            metadata={"sharpe": 1.2, "win_rate": 0.6},
        )

        # Promote to active
        versioner.commit_promotion(file_path)

        # Rollback if needed
        versioner.rollback(sha)
    """

    DEFAULT_COMMIT_MSG_PREFIX = "[code-evolution]"

    def __init__(
        self,
        repo_path: str = ".",
        author_name: str = "ARGUS Code Evolution",
        author_email: str = "argus@local",
        dry_run: bool = False,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.author_name = author_name
        self.author_email = author_email
        self.dry_run = dry_run
        self._commits: List[CommitRecord] = []
        self._git_available = self._check_git_available()
        if not self._git_available:
            logger.warning("GitVersioner: git not available, running in record-only mode")
        else:
            logger.info("GitVersioner: initialized at %s", self.repo_path)

    def _check_git_available(self) -> bool:
        """Check if git is installed and the repo is initialized."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=str(self.repo_path),
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError):
            return False

    def _run_git(self, args: List[str], timeout: float = 10.0) -> Optional[str]:
        """Run a git command and return stdout, or None on failure."""
        if not self._git_available:
            return None
        if self.dry_run:
            logger.debug("GitVersioner [dry-run]: git %s", " ".join(args))
            return ""
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=str(self.repo_path),
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode != 0:
                logger.debug(
                    "GitVersioner: git %s failed: %s",
                    " ".join(args), result.stderr.strip(),
                )
                return None
            return result.stdout.strip()
        except subprocess.SubprocessError as exc:
            logger.debug("GitVersioner: git command error: %s", exc)
            return None

    def commit_generation(
        self,
        file_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Stage and commit a newly generated strategy file.
        Returns commit SHA or None on failure.
        """
        if not file_path.exists():
            logger.warning("GitVersioner: file does not exist: %s", file_path)
            return None

        # Verify path is within generated_strategies/
        try:
            rel_path = file_path.resolve().relative_to(self.repo_path)
        except ValueError:
            logger.warning("GitVersioner: refusing to commit file outside repo: %s", file_path)
            return None

        if "generated_strategies" not in str(rel_path):
            logger.warning(
                "GitVersioner: refusing to commit file outside generated_strategies/: %s",
                rel_path,
            )
            return None

        # Build commit message
        meta = metadata or {}
        message = self._build_commit_message("generate", file_path, meta)

        # Stage the file
        if self._run_git(["add", str(rel_path)]) is None:
            return None

        # Commit
        commit_args = [
            "commit",
            "-m", message,
            "--author", f"{self.author_name} <{self.author_email}>",
        ]
        result = self._run_git(commit_args)
        if result is None:
            return None

        # Get the SHA
        sha = self._run_git(["rev-parse", "HEAD"])
        if sha is None:
            return None
        sha = sha.strip()

        record = CommitRecord(
            timestamp=time.time(),
            sha=sha,
            file_path=file_path,
            message=message,
            operation="generate",
        )
        self._commits.append(record)
        logger.info("GitVersioner: committed %s as %s", file_path.name, sha[:8])
        return sha

    def commit_promotion(
        self,
        file_path: Path,
        from_dir: str = "candidates",
        to_dir: str = "active",
    ) -> Optional[str]:
        """
        Move a strategy from candidates/ to active/ and commit.
        Returns the new commit SHA.
        """
        if not file_path.exists():
            return None

        new_path = file_path.parent.parent / to_dir / file_path.name
        new_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            file_path.rename(new_path)
        except OSError as exc:
            logger.warning("GitVersioner: rename failed: %s", exc)
            return None

        message = self._build_commit_message("promote", new_path, {"from": from_dir, "to": to_dir})

        try:
            old_rel = file_path.resolve().relative_to(self.repo_path)
            new_rel = new_path.resolve().relative_to(self.repo_path)
        except ValueError:
            return None

        if self._run_git(["add", str(old_rel), str(new_rel)]) is None:
            return None

        commit_args = [
            "commit",
            "-m", message,
            "--author", f"{self.author_name} <{self.author_email}>",
        ]
        if self._run_git(commit_args) is None:
            return None

        sha = self._run_git(["rev-parse", "HEAD"])
        if sha is None:
            return None
        sha = sha.strip()

        self._commits.append(CommitRecord(
            timestamp=time.time(),
            sha=sha,
            file_path=new_path,
            message=message,
            operation="promote",
        ))
        logger.info("GitVersioner: promoted %s as %s", new_path.name, sha[:8])
        return sha

    def commit_retirement(
        self,
        file_path: Path,
        reason: str,
        from_dir: str = "active",
        to_dir: str = "graveyard",
    ) -> Optional[str]:
        """Move a strategy from active/ to graveyard/ and commit."""
        if not file_path.exists():
            return None

        new_path = file_path.parent.parent / to_dir / file_path.name
        new_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            file_path.rename(new_path)
        except OSError as exc:
            logger.warning("GitVersioner: rename failed: %s", exc)
            return None

        message = self._build_commit_message("retire", new_path, {"reason": reason})

        try:
            old_rel = file_path.resolve().relative_to(self.repo_path)
            new_rel = new_path.resolve().relative_to(self.repo_path)
        except ValueError:
            return None

        if self._run_git(["add", str(old_rel), str(new_rel)]) is None:
            return None

        if self._run_git([
            "commit", "-m", message,
            "--author", f"{self.author_name} <{self.author_email}>",
        ]) is None:
            return None

        sha = self._run_git(["rev-parse", "HEAD"])
        if sha is None:
            return None
        sha = sha.strip()

        self._commits.append(CommitRecord(
            timestamp=time.time(),
            sha=sha,
            file_path=new_path,
            message=message,
            operation="retire",
        ))
        logger.warning("GitVersioner: retired %s — %s", new_path.name, reason)
        return sha

    def rollback(self, sha: str) -> bool:
        """
        Revert a specific commit. Useful when a generated strategy
        causes problems in production.
        """
        if not sha:
            return False
        result = self._run_git(["revert", "--no-edit", sha])
        if result is None:
            return False
        new_sha = self._run_git(["rev-parse", "HEAD"])
        if new_sha:
            self._commits.append(CommitRecord(
                timestamp=time.time(),
                sha=new_sha.strip(),
                file_path=Path("<rollback>"),
                message=f"rollback of {sha}",
                operation="rollback",
            ))
        logger.warning("GitVersioner: rolled back %s", sha[:8])
        return True

    def list_history(self, n: int = 20) -> List[Dict[str, Any]]:
        """List the last N commits made by the versioner."""
        recent = self._commits[-n:]
        return [
            {
                "timestamp": c.timestamp,
                "sha": c.sha[:8] if c.sha else "",
                "file": str(c.file_path.name) if c.file_path else "",
                "operation": c.operation,
                "message": c.message[:100],
            }
            for c in reversed(recent)
        ]

    def _build_commit_message(
        self,
        operation: str,
        file_path: Path,
        metadata: Dict[str, Any],
    ) -> str:
        """Build a structured commit message."""
        lines = [
            f"{self.DEFAULT_COMMIT_MSG_PREFIX} {operation}: {file_path.name}",
            "",
        ]
        if metadata:
            for k, v in metadata.items():
                lines.append(f"{k}: {v}")
        lines.append("")
        lines.append("Auto-generated by ARGUS code_evolution_engine")
        return "\n".join(lines)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "git_available": self._git_available,
            "dry_run": self.dry_run,
            "total_commits": len(self._commits),
            "by_operation": {
                op: sum(1 for c in self._commits if c.operation == op)
                for op in ("generate", "promote", "retire", "rollback")
            },
        }
