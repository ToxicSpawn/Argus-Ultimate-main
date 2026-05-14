from __future__ import annotations

BLOCKED_PREFIXES = (
    "quantum",
    "research",
    "core.ARGUS",
)

BLOCKED_MODULES = (
    "enhanced_trading_launcher",
)


def assert_live_import_allowed(module_name: str) -> None:
    if module_name in BLOCKED_MODULES:
        raise RuntimeError(f"Blocked live import: {module_name}")
    for prefix in BLOCKED_PREFIXES:
        if module_name == prefix or module_name.startswith(prefix + "."):
            raise RuntimeError(f"Blocked live import: {module_name}")
