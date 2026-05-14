"""EnvResolver — resolves ${VAR} and ${VAR:-default} in config values — Push 61.

Recursively walks dict/list structures and replaces string placeholders
with values from os.environ.

Examples::

    ${ARGUS_API_KEY}           -> os.environ['ARGUS_API_KEY']
    ${ARGUS_PORT:-8080}        -> os.environ.get('ARGUS_PORT', '8080')
    ${MISSING_VAR:-fallback}   -> 'fallback'

"""
from __future__ import annotations

import os
import re
from typing import Any

_PLACEHOLDER = re.compile(r"\$\{([^}]+)\}")


class EnvResolver:
    """Resolves environment variable placeholders in configuration dicts."""

    def __init__(self, environ: dict = None) -> None:  # type: ignore[assignment]
        self._env = environ if environ is not None else os.environ

    def resolve(self, value: Any) -> Any:
        """Recursively resolve placeholders in value."""
        if isinstance(value, str):
            return self._resolve_string(value)
        elif isinstance(value, dict):
            return {k: self.resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.resolve(item) for item in value]
        return value

    def resolve_dict(self, d: dict) -> dict:
        """Resolve all placeholders in a config dictionary."""
        return {k: self.resolve(v) for k, v in d.items()}

    def _resolve_string(self, s: str) -> str:
        def replacer(match: re.Match) -> str:
            expr = match.group(1)
            if ":-" in expr:
                var, default = expr.split(":-", 1)
                return self._env.get(var.strip(), default)
            else:
                var = expr.strip()
                return self._env.get(var, match.group(0))  # leave unresolved if missing
        return _PLACEHOLDER.sub(replacer, s)

    @staticmethod
    def from_env() -> "EnvResolver":
        return EnvResolver(os.environ)
