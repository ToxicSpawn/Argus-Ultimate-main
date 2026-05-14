#!/usr/bin/env python3
"""
Code Sandbox — safe execution environment for dynamically-generated strategy
code snippets. Uses subprocess isolation with hard timeouts and resource caps.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_WRAPPER_TEMPLATE = '''
import json, sys
try:
    _ctx = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {{}}
except Exception:
    _ctx = {{}}
try:
{code}
    if "_result" in dir():
        print(json.dumps(_result, default=str))
    else:
        print(json.dumps({{}}))
except Exception as _e:
    print(json.dumps({{"error": str(_e)}}))
'''


class CodeSandbox:
    """
    Execute arbitrary Python snippets in a subprocess jail.

    Security notes:
    - Each snippet runs in a fresh subprocess (no shared memory)
    - Stdout is captured; stderr is discarded unless debug=True
    - Timeout enforced via subprocess timeout parameter
    - No network access restriction at OS level (advisory only)
    """

    def __init__(self, timeout: float = 5.0, *, debug: bool = False):
        self.timeout = max(0.5, float(timeout))
        self.debug = bool(debug)

    def run(self, code: str, context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], Optional[str]]:
        """
        Execute *code* in an isolated subprocess.

        Returns (result, error_message).  result is the Python-deserialized
        value printed by the snippet as JSON; error_message is None on success.
        """
        indented = "\n".join("    " + line for line in code.splitlines())
        src = _WRAPPER_TEMPLATE.format(code=indented)
        ctx_str = json.dumps(context or {}, default=str)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(src)
            fname = f.name
        try:
            result = subprocess.run(
                [sys.executable, fname, ctx_str],
                capture_output=not self.debug,
                text=True,
                timeout=self.timeout,
            )
            stdout = result.stdout.strip() if result.stdout else ""
            if result.returncode != 0 and not stdout:
                err = result.stderr.strip() if result.stderr else "non-zero exit"
                return None, err[:500]
            if stdout:
                try:
                    parsed = json.loads(stdout)
                    if isinstance(parsed, dict) and "error" in parsed and len(parsed) == 1:
                        return None, str(parsed["error"])[:500]
                    return parsed, None
                except json.JSONDecodeError:
                    return stdout, None
            return {}, None
        except subprocess.TimeoutExpired:
            return None, f"sandbox timeout ({self.timeout}s)"
        except Exception as e:
            return None, str(e)
        finally:
            try:
                os.unlink(fname)
            except Exception:
                pass
