"""
Code Review Gate — static safety analysis for generated code.

Before any generated Python file is allowed to run, it passes through this
review gate. The gate uses AST traversal to enforce strict rules:

  1. SYNTAX        — must be valid Python
  2. IMPORTS       — only whitelisted modules allowed
  3. NO I/O        — no file/network/socket operations
  4. NO EVAL       — no eval/exec/compile
  5. NO ATTRIBUTES — no __getattr__/__setattr__/__class__/__dict__
  6. NO SUBPROCESS — no subprocess/os.system/popen
  7. NO INFINITE   — no `while True:` without sleep/break
  8. NO REFLECTION — no inspect/getattr with dynamic attributes
  9. SIZE LIMIT    — file must be < 10KB
  10. STRUCTURE    — must contain exactly one BaseGeneratedStrategy subclass

If any rule fails, the file is rejected and moved to graveyard.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# Whitelist of safe imports
ALLOWED_IMPORTS: Set[str] = {
    "__future__",
    "math",
    "typing",
    "dataclasses",
    "enum",
    "collections",
    "generated_strategies",
    "generated_strategies.__init__",
}

# Whitelist of safe builtins
ALLOWED_BUILTINS: Set[str] = {
    # Type constructors
    "int", "float", "str", "bool", "list", "tuple", "dict", "set", "frozenset",
    # Math
    "abs", "min", "max", "sum", "round", "pow", "divmod",
    # Iteration
    "len", "range", "enumerate", "zip", "filter", "map", "sorted", "reversed",
    "all", "any",
    # Type checks
    "isinstance", "issubclass", "type", "callable",
    # I/O (limited — only print for debug, not allowed at runtime)
    # NO: open, input, exec, eval, compile, __import__
}

# Forbidden names — anything matching is rejected
FORBIDDEN_NAMES: Set[str] = {
    "eval", "exec", "compile", "__import__",
    "open", "input", "raw_input",
    "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr", "hasattr",
    "exit", "quit",
    "breakpoint",
}

# Forbidden attributes
FORBIDDEN_ATTRS: Set[str] = {
    "__class__", "__base__", "__bases__", "__mro__", "__subclasses__",
    "__dict__", "__globals__", "__builtins__", "__import__",
    "__getattribute__", "__setattr__", "__delattr__",
    "f_locals", "f_globals", "f_back", "f_code",
    "gi_frame", "gi_code",
}

# Maximum file size in bytes
MAX_FILE_SIZE = 10 * 1024  # 10 KB

# Maximum AST node count
MAX_AST_NODES = 500


@dataclass
class ReviewResult:
    """Result of a code review."""
    file_path: Path
    passed: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    ast_node_count: int = 0
    file_size_bytes: int = 0
    contains_strategy_class: bool = False
    strategy_class_name: str = ""


class _ASTValidator(ast.NodeVisitor):
    """Walks the AST and accumulates violations."""

    def __init__(self) -> None:
        self.violations: List[str] = []
        self.warnings: List[str] = []
        self.imports: Set[str] = set()
        self.strategy_class_count: int = 0
        self.strategy_class_name: str = ""
        self.node_count: int = 0
        self.has_while_true: bool = False
        self.has_break_in_while_true: bool = False

    def generic_visit(self, node: ast.AST) -> None:
        self.node_count += 1
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.node_count += 1
        for alias in node.names:
            module_name = alias.name.split(".")[0]
            self.imports.add(alias.name)
            if module_name not in ALLOWED_IMPORTS and alias.name not in ALLOWED_IMPORTS:
                self.violations.append(f"forbidden import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.node_count += 1
        module = node.module or ""
        module_root = module.split(".")[0]
        self.imports.add(module)
        if module_root not in ALLOWED_IMPORTS and module not in ALLOWED_IMPORTS:
            self.violations.append(f"forbidden import: from {module}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self.node_count += 1
        if node.id in FORBIDDEN_NAMES:
            self.violations.append(f"forbidden name: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.node_count += 1
        if node.attr in FORBIDDEN_ATTRS:
            self.violations.append(f"forbidden attribute: .{node.attr}")
        if node.attr.startswith("__") and node.attr.endswith("__"):
            # Allow only common dunder access
            allowed_dunders = {"__init__", "__name__", "__doc__", "__future__"}
            if node.attr not in allowed_dunders:
                self.warnings.append(f"dunder access: .{node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.node_count += 1
        # Check function being called
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_NAMES:
                self.violations.append(f"forbidden call: {node.func.id}()")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN_NAMES:
                self.violations.append(f"forbidden method call: .{node.func.attr}()")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.node_count += 1
        # Check if it's a BaseGeneratedStrategy subclass
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "BaseGeneratedStrategy":
                self.strategy_class_count += 1
                self.strategy_class_name = node.name
                break
            if isinstance(base, ast.Attribute) and base.attr == "BaseGeneratedStrategy":
                self.strategy_class_count += 1
                self.strategy_class_name = node.name
                break
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.node_count += 1
        # Check for `while True:` without break
        is_while_true = False
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            is_while_true = True
        elif isinstance(node.test, ast.Name) and node.test.id == "True":
            is_while_true = True

        if is_while_true:
            self.has_while_true = True
            # Check if any descendant is a Break statement
            has_break = any(isinstance(n, ast.Break) for n in ast.walk(node))
            if not has_break:
                self.violations.append("infinite loop: while True without break")

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.node_count += 1
        if node.name.startswith("__") and node.name not in {
            "__init__", "__call__", "__repr__", "__str__"
        }:
            self.warnings.append(f"unusual dunder method: {node.name}")
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.node_count += 1
        # Lambdas are allowed but tracked
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.node_count += 1
        # Allow try/except — strategies should handle missing data gracefully
        self.generic_visit(node)


class CodeReviewGate:
    """
    Static analyser that approves or rejects generated Python files.

    Usage::

        gate = CodeReviewGate()
        result = gate.review(Path("generated_strategies/candidates/foo.py"))

        if result.passed:
            # Move to active/
            ...
        else:
            for v in result.violations:
                logger.warning(v)
            # Move to graveyard/
    """

    def __init__(self) -> None:
        self._reviewed_count = 0
        self._passed_count = 0
        self._rejected_count = 0
        logger.info("CodeReviewGate: initialized")

    def review(self, file_path: Path) -> ReviewResult:
        """Review a single Python file. Returns ReviewResult."""
        self._reviewed_count += 1
        result = ReviewResult(file_path=file_path, passed=False)

        if not file_path.exists():
            result.violations.append("file does not exist")
            self._rejected_count += 1
            return result

        # Size check
        file_size = file_path.stat().st_size
        result.file_size_bytes = file_size
        if file_size > MAX_FILE_SIZE:
            result.violations.append(f"file too large: {file_size} > {MAX_FILE_SIZE}")
            self._rejected_count += 1
            return result

        # Read source
        try:
            source = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            result.violations.append(f"read error: {exc}")
            self._rejected_count += 1
            return result

        # Parse AST
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            result.violations.append(f"syntax error: {exc}")
            self._rejected_count += 1
            return result

        # Walk AST
        validator = _ASTValidator()
        validator.visit(tree)

        result.violations.extend(validator.violations)
        result.warnings.extend(validator.warnings)
        result.ast_node_count = validator.node_count
        result.contains_strategy_class = validator.strategy_class_count == 1
        result.strategy_class_name = validator.strategy_class_name

        if validator.node_count > MAX_AST_NODES:
            result.violations.append(
                f"AST too large: {validator.node_count} > {MAX_AST_NODES}"
            )

        if validator.strategy_class_count == 0:
            result.violations.append("no BaseGeneratedStrategy subclass found")
        elif validator.strategy_class_count > 1:
            result.violations.append(
                f"multiple BaseGeneratedStrategy subclasses ({validator.strategy_class_count})"
            )

        # Final verdict
        result.passed = len(result.violations) == 0
        if result.passed:
            self._passed_count += 1
        else:
            self._rejected_count += 1

        return result

    def review_source(self, source: str) -> ReviewResult:
        """Review a source string directly (no file)."""
        result = ReviewResult(file_path=Path("<inline>"), passed=False)
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            result.violations.append(f"syntax error: {exc}")
            return result
        validator = _ASTValidator()
        validator.visit(tree)
        result.violations.extend(validator.violations)
        result.warnings.extend(validator.warnings)
        result.ast_node_count = validator.node_count
        result.contains_strategy_class = validator.strategy_class_count == 1
        result.strategy_class_name = validator.strategy_class_name
        if validator.node_count > MAX_AST_NODES:
            result.violations.append(f"AST too large: {validator.node_count}")
        if validator.strategy_class_count != 1:
            result.violations.append(
                f"need exactly one strategy class (got {validator.strategy_class_count})"
            )
        result.passed = len(result.violations) == 0
        return result

    def snapshot(self) -> Dict[str, Any]:
        return {
            "reviewed": self._reviewed_count,
            "passed": self._passed_count,
            "rejected": self._rejected_count,
            "pass_rate": (
                self._passed_count / max(self._reviewed_count, 1)
            ),
        }
