"""
Module isolation tests — verifies that independent layers don't import from each other.

Layer 1 (Trading System): unified_trading_system, core/component_registry, unified_execution_engine
Layer 2 (Core Modules): portfolio_manager, kelly_sizing, strategy_scanner, etc.
Layer 3 (Institutional): argus_live/, argus/

Rule: Layer 2 and 3 must NEVER import from Layer 1.
"""

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

# Layer 1 modules that should NOT be imported by Layer 2/3
FORBIDDEN_IMPORTS = {
    "unified_trading_system",
    "unified_execution_engine",
    "core.component_registry",
}

# Layer 2: independent core modules
LAYER2_MODULES = [
    "core/portfolio_manager.py",
    "core/kelly_sizing.py",
    "core/strategy_scanner.py",
    "core/strategy_validator.py",
    "core/performance_scorecard.py",
    "core/implementation_shortfall.py",
    "core/async_write_queue.py",
    "core/position_reconciler.py",
    "core/protocols.py",
    "core/domain_config.py",
]

# Layer 3: institutional modules
LAYER3_DIRS = [
    "argus/truth",
    "argus/replay",
    "argus/drift",
    "argus/executive",
    "argus/chaos",
    "argus/execution_analytics",
]


def _get_imports(filepath: Path) -> set:
    """Extract all import module names from a Python file."""
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
                imports.add(node.module)
    return imports


class TestLayer2Isolation(unittest.TestCase):
    """Layer 2 core modules must not import from Layer 1."""

    def test_no_forbidden_imports(self):
        violations = []
        for rel_path in LAYER2_MODULES:
            filepath = REPO_ROOT / rel_path
            if not filepath.exists():
                continue
            imports = _get_imports(filepath)
            for forbidden in FORBIDDEN_IMPORTS:
                if forbidden in imports:
                    violations.append(f"{rel_path} imports {forbidden}")

        self.assertEqual(violations, [], f"Layer 2 isolation violated:\n" + "\n".join(violations))


class TestLayer3Isolation(unittest.TestCase):
    """Layer 3 institutional modules must not import from Layer 1."""

    def test_argus_modules_isolated(self):
        violations = []
        for dir_path in LAYER3_DIRS:
            full_dir = REPO_ROOT / dir_path
            if not full_dir.exists():
                continue
            for py_file in full_dir.rglob("*.py"):
                imports = _get_imports(py_file)
                for forbidden in FORBIDDEN_IMPORTS:
                    if forbidden in imports:
                        rel = py_file.relative_to(REPO_ROOT)
                        violations.append(f"{rel} imports {forbidden}")

        self.assertEqual(violations, [], f"Layer 3 isolation violated:\n" + "\n".join(violations))

    def test_argus_live_modules_isolated(self):
        violations = []
        argus_live_dir = REPO_ROOT / "argus_live"
        if not argus_live_dir.exists():
            self.skipTest("argus_live/ not found")
        for py_file in argus_live_dir.rglob("*.py"):
            imports = _get_imports(py_file)
            for forbidden in FORBIDDEN_IMPORTS:
                if forbidden in imports:
                    rel = py_file.relative_to(REPO_ROOT)
                    violations.append(f"{rel} imports {forbidden}")

        self.assertEqual(violations, [], f"argus_live/ isolation violated:\n" + "\n".join(violations))


class TestLayer2SelfContained(unittest.TestCase):
    """Each Layer 2 module must not import from other Layer 2 modules."""

    def test_no_cross_imports(self):
        module_names = {Path(p).stem for p in LAYER2_MODULES}
        violations = []
        for rel_path in LAYER2_MODULES:
            filepath = REPO_ROOT / rel_path
            if not filepath.exists():
                continue
            this_name = filepath.stem
            imports = _get_imports(filepath)
            for other in module_names:
                if other != this_name and (other in imports or f"core.{other}" in imports):
                    violations.append(f"{rel_path} imports core.{other}")

        self.assertEqual(violations, [], f"Layer 2 cross-imports:\n" + "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
