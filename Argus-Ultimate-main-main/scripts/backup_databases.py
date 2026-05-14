"""
ARGUS Database Backup Script.

Finds all .db files in the data/ directory, copies each to
data/backups/YYYY-MM-DD/filename.db, and prunes backups older than
KEEP_DAYS (default 7).

Can be run manually::

    py scripts/backup_databases.py

Or scheduled (cron example — daily at 02:00)::

    0 2 * * * cd /path/to/argus && py scripts/backup_databases.py >> logs/backup.log 2>&1

Windows Task Scheduler example::

    py C:\\argus\\scripts\\backup_databases.py
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KEEP_DAYS: int = 7          # How many days of backups to keep
DATA_DIR: str = "data"      # Root directory to search for .db files
BACKUP_ROOT: str = os.path.join(DATA_DIR, "backups")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [backup] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("argus.backup")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_db_files(root: Path) -> list[Path]:
    """
    Recursively find all .db files under ``root``, excluding those already
    inside the backups directory to avoid backing up backups.
    """
    backup_abs = root / "backups"
    found: list[Path] = []
    for db_path in root.rglob("*.db"):
        # Skip anything inside the backups subdirectory
        try:
            db_path.relative_to(backup_abs)
            continue  # This path is inside backups/
        except ValueError:
            pass  # Not inside backups — include it
        found.append(db_path)
    return sorted(found)


def _backup_today(
    db_files: list[Path],
    backup_root: Path,
    root: Path,
) -> tuple[int, int]:
    """
    Copy each db file into ``backup_root/YYYY-MM-DD/``.

    Files are placed in subdirectories mirroring their path relative to
    ``root``, so data/fills.db → backups/2026-03-14/fills.db and
    data/subdir/trades.db → backups/2026-03-14/subdir/trades.db.

    Returns (files_backed_up, files_skipped).
    """
    today_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    dest_dir = backup_root / today_str
    dest_dir.mkdir(parents=True, exist_ok=True)

    backed_up = 0
    skipped = 0

    for db_path in db_files:
        try:
            rel = db_path.relative_to(root)
        except ValueError:
            rel = Path(db_path.name)

        dest_path = dest_dir / rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(str(db_path), str(dest_path))
            size_kb = dest_path.stat().st_size / 1024
            logger.info(
                "Backed up: %s → %s (%.1f KB)",
                db_path,
                dest_path.relative_to(backup_root),
                size_kb,
            )
            backed_up += 1
        except OSError as exc:
            logger.warning("Failed to back up %s: %s", db_path, exc)
            skipped += 1

    return backed_up, skipped


def _prune_old_backups(backup_root: Path, keep_days: int) -> int:
    """
    Delete backup date-directories older than ``keep_days`` days.

    Only removes directories whose names match YYYY-MM-DD.
    Returns the number of directories deleted.
    """
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=keep_days)
    cutoff_date = cutoff.date()
    deleted = 0

    if not backup_root.exists():
        return 0

    for entry in sorted(backup_root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            dir_date = datetime.strptime(entry.name, "%Y-%m-%d").date()
        except ValueError:
            continue  # Skip non-date directories

        if dir_date < cutoff_date:
            try:
                shutil.rmtree(str(entry))
                logger.info("Pruned old backup: %s", entry.name)
                deleted += 1
            except OSError as exc:
                logger.warning("Failed to prune %s: %s", entry, exc)

    return deleted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Run the full backup cycle: discover → copy → prune.

    Returns exit code (0 = success, 1 = no databases found, 2 = all failed).
    """
    # Resolve paths relative to the script's parent directory (project root)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_root = project_root / DATA_DIR
    backup_root = project_root / BACKUP_ROOT

    if not data_root.exists():
        logger.error("Data directory not found: %s", data_root)
        return 1

    logger.info("ARGUS database backup starting")
    logger.info("  Data root : %s", data_root)
    logger.info("  Backup root: %s", backup_root)
    logger.info("  Retention : %d days", KEEP_DAYS)

    # 1. Discover
    db_files = _find_db_files(data_root)
    if not db_files:
        logger.warning("No .db files found under %s — nothing to back up", data_root)
        return 1

    logger.info("Found %d database file(s):", len(db_files))
    for f in db_files:
        logger.info("  %s", f.relative_to(project_root))

    # 2. Back up
    backed_up, skipped = _backup_today(db_files, backup_root, data_root)
    logger.info("Backup complete: %d copied, %d skipped", backed_up, skipped)

    # 3. Prune
    pruned = _prune_old_backups(backup_root, KEEP_DAYS)
    if pruned:
        logger.info("Pruned %d old backup directory(ies)", pruned)
    else:
        logger.info("No old backups to prune (retention=%d days)", KEEP_DAYS)

    if backed_up == 0 and skipped > 0:
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
