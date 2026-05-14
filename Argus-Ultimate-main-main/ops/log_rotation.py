"""
Log Rotation & Archival -- configures rotating file handlers and manages log files.

Features:
  - RotatingFileHandler with gzip compression of rotated files
  - Old log archival and cleanup
  - Disk usage reporting
"""
from __future__ import annotations

import gzip
import logging
import os
import shutil
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class _GzipRotatingHandler(RotatingFileHandler):
    """RotatingFileHandler that gzip-compresses rotated log files."""

    def __init__(self, *args, compress: bool = True, **kwargs):
        self._compress = compress
        super().__init__(*args, **kwargs)

    def doRollover(self) -> None:
        super().doRollover()
        if not self._compress:
            return
        # Compress the most recently rotated file
        rotated = f"{self.baseFilename}.1"
        if os.path.exists(rotated):
            gz_path = rotated + ".gz"
            try:
                with open(rotated, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(rotated)
            except Exception as exc:
                logger.debug("gzip compression failed for %s: %s", rotated, exc)


def configure_log_rotation(
    log_dir: str = "logs",
    max_bytes: int = 50_000_000,
    backup_count: int = 10,
    compress: bool = True,
) -> logging.Handler:
    """Set up a RotatingFileHandler with optional gzip compression.

    Args:
        log_dir:      Directory for log files (created if needed).
        max_bytes:    Max size per log file before rotation.
        backup_count: Number of rotated backups to keep.
        compress:     Gzip-compress rotated files.

    Returns:
        The configured logging handler (already added to root logger).
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / "argus.log"

    handler = _GzipRotatingHandler(
        str(log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        compress=compress,
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.addHandler(handler)
    logger.info(
        "Log rotation configured: dir=%s max_bytes=%d backups=%d compress=%s",
        log_dir, max_bytes, backup_count, compress,
    )
    return handler


def archive_old_logs(
    log_dir: str = "logs",
    archive_dir: str = "logs/archive",
    max_age_days: int = 30,
) -> Dict[str, int]:
    """Move old log files to an archive directory and delete very old archives.

    Args:
        log_dir:       Source directory with log files.
        archive_dir:   Destination archive directory.
        max_age_days:  Delete archived files older than this many days.

    Returns:
        Dict with ``archived`` and ``deleted`` counts.
    """
    log_path = Path(log_dir)
    archive_path = Path(archive_dir)
    archive_path.mkdir(parents=True, exist_ok=True)

    now = time.time()
    cutoff_archive = now - (7 * 86400)  # move files older than 7 days
    cutoff_delete = now - (max_age_days * 86400)

    archived = 0
    deleted = 0

    # Move old logs to archive
    if log_path.exists():
        for f in log_path.iterdir():
            if not f.is_file():
                continue
            if f.suffix not in (".log", ".gz"):
                continue
            if f.stat().st_mtime < cutoff_archive:
                dest = archive_path / f.name
                try:
                    shutil.move(str(f), str(dest))
                    archived += 1
                except Exception as exc:
                    logger.debug("Failed to archive %s: %s", f, exc)

    # Delete very old archived files
    for f in archive_path.iterdir():
        if not f.is_file():
            continue
        if f.stat().st_mtime < cutoff_delete:
            try:
                f.unlink()
                deleted += 1
            except Exception as exc:
                logger.debug("Failed to delete %s: %s", f, exc)

    logger.info("Log archival: archived=%d, deleted=%d", archived, deleted)
    return {"archived": archived, "deleted": deleted}


def get_log_disk_usage(log_dir: str = "logs") -> Dict[str, object]:
    """Return disk usage statistics for the log directory.

    Returns:
        Dict with ``total_bytes``, ``file_count``, ``oldest_file``, ``newest_file``.
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return {
            "total_bytes": 0,
            "file_count": 0,
            "oldest_file": None,
            "newest_file": None,
        }

    total_bytes = 0
    file_count = 0
    oldest_mtime: Optional[float] = None
    newest_mtime: Optional[float] = None
    oldest_name: Optional[str] = None
    newest_name: Optional[str] = None

    for f in log_path.rglob("*"):
        if not f.is_file():
            continue
        stat = f.stat()
        total_bytes += stat.st_size
        file_count += 1
        if oldest_mtime is None or stat.st_mtime < oldest_mtime:
            oldest_mtime = stat.st_mtime
            oldest_name = str(f)
        if newest_mtime is None or stat.st_mtime > newest_mtime:
            newest_mtime = stat.st_mtime
            newest_name = str(f)

    return {
        "total_bytes": total_bytes,
        "file_count": file_count,
        "oldest_file": oldest_name,
        "newest_file": newest_name,
    }
