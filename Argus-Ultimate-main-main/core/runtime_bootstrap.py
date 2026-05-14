from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable


def ensure_runtime_dirs(directories: Iterable[str]) -> None:
    """Create runtime directories if missing."""
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)


def configure_file_and_stdout_logging(log_path: str) -> None:
    """Configure default process logging to file + stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
