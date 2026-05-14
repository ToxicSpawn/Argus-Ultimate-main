"""Deprecated compatibility wrapper. Use argus_live.governance instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "The top-level governance package is deprecated; import from argus_live.governance instead.",
    DeprecationWarning,
    stacklevel=2,
)

from argus_live.governance import *  # noqa: F401,F403
