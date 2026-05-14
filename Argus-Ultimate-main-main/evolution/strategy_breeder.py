"""
Strategy DNA Crossover — breeds new strategy parameter sets from high-fitness
parents using genetic operators (uniform crossover + Gaussian mutation).

Each strategy's parameters are treated as a "genome" — a flat dict of numeric
and categorical values.  The breeder maintains lineage so you can trace which
parents contributed to a given offspring.

Persistence: SQLite at ``data/strategy_breeding.db``.

Usage::

    breeder = StrategyBreeder()
    breeder.register_strategy("mom_v1", {"lookback": 20, "threshold": 0.6}, fitness=1.5)
    breeder.register_strategy("mom_v2", {"lookback": 30, "threshold": 0.4}, fitness=2.1)
    offspring = breeder.breed_generation(top_k=2, offspring=5)
"""

from __future__ import annotations

import copy
import json
import logging
import random
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_DB_DIR = Path("data")
_DEFAULT_DB_NAME = "strategy_breeding.db"


class StrategyBreeder:
    """Breeds new strategy parameter sets via genetic crossover and mutation.

    Parameters
    ----------
    db_path : str | None
        SQLite persistence path.  Defaults to ``data/strategy_breeding.db``.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        if db_path is None:
            db_dir = _DEFAULT_DB_DIR
            db_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = str(db_dir / _DEFAULT_DB_NAME)
        else:
            self._db_path = str(db_path)

        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._init_db()
        logger.info("StrategyBreeder initialised, db=%s", self._db_path)

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS strategies (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        name        TEXT    NOT NULL UNIQUE,
                        params_json TEXT    NOT NULL,
                        fitness     REAL    NOT NULL DEFAULT 0.0,
                        parent_a    TEXT    DEFAULT NULL,
                        parent_b    TEXT    DEFAULT NULL,
                        generation  INTEGER NOT NULL DEFAULT 0,
                        created_at  TEXT    NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_strategies_fitness
                    ON strategies(fitness DESC)
                """)
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_strategy(
        self,
        name: str,
        params: Dict[str, Any],
        fitness: float,
        parent_a: Optional[str] = None,
        parent_b: Optional[str] = None,
        generation: int = 0,
    ) -> None:
        """Register a strategy genome (parameter set) with its fitness score.

        Parameters
        ----------
        name : str
            Unique strategy identifier.
        params : dict
            Strategy parameters (must be JSON-serialisable).
        fitness : float
            Fitness / performance score.
        parent_a, parent_b : str | None
            Names of parent strategies (for lineage tracking).
        generation : int
            Which breeding generation this strategy belongs to.
        """
        now = datetime.now(timezone.utc).isoformat()
        params_json = json.dumps(params)

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO strategies
                       (name, params_json, fitness, parent_a, parent_b, generation, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (name, params_json, fitness, parent_a, parent_b, generation, now),
                )
                conn.commit()
            finally:
                conn.close()

        logger.debug("Registered strategy %s (fitness=%.4f, gen=%d)", name, fitness, generation)

    def crossover(self, parent_a: str, parent_b: str) -> Dict[str, Any]:
        """Uniform crossover of two parent strategies' parameters.

        For each parameter key, the value is randomly drawn from one parent.
        If a key exists in only one parent, it is always included.

        Parameters
        ----------
        parent_a, parent_b : str
            Names of registered parent strategies.

        Returns
        -------
        dict
            Child parameter set.
        """
        params_a = self._get_params(parent_a)
        params_b = self._get_params(parent_b)

        if params_a is None or params_b is None:
            raise ValueError(f"Parent not found: a={parent_a}, b={parent_b}")

        all_keys = set(params_a.keys()) | set(params_b.keys())
        child: Dict[str, Any] = {}

        for key in all_keys:
            in_a = key in params_a
            in_b = key in params_b
            if in_a and in_b:
                child[key] = params_a[key] if self._rng.random() < 0.5 else params_b[key]
            elif in_a:
                child[key] = params_a[key]
            else:
                child[key] = params_b[key]

        logger.debug("Crossover(%s, %s) → %d keys", parent_a, parent_b, len(child))
        return child

    def mutate(
        self,
        params: Dict[str, Any],
        mutation_rate: float = 0.1,
    ) -> Dict[str, Any]:
        """Apply Gaussian perturbation to numeric parameters.

        Non-numeric values are left unchanged.  Boolean values are flipped
        with probability ``mutation_rate``.

        Parameters
        ----------
        params : dict
            Parameter set to mutate.
        mutation_rate : float
            Probability of mutating each parameter.

        Returns
        -------
        dict
            Mutated parameter set (new dict, original unchanged).
        """
        mutated = copy.deepcopy(params)

        for key, value in mutated.items():
            if self._rng.random() >= mutation_rate:
                continue

            if isinstance(value, bool):
                mutated[key] = not value
            elif isinstance(value, int):
                # Integer: add Gaussian noise scaled to ~10% of value (min ±1)
                scale = max(1, abs(value) // 10)
                delta = round(self._rng.gauss(0, scale))
                mutated[key] = value + delta
            elif isinstance(value, float):
                # Float: Gaussian noise scaled to ~10% of value (min ±0.01)
                scale = max(0.01, abs(value) * 0.1)
                mutated[key] = value + self._rng.gauss(0, scale)
            # Strings, lists, etc. are left as-is

        return mutated

    def breed_generation(
        self,
        top_k: int = 5,
        offspring: int = 10,
        mutation_rate: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Breed a new generation from the top-performing strategies.

        1. Select the ``top_k`` strategies by fitness.
        2. Generate ``offspring`` children via random pairwise crossover + mutation.

        Parameters
        ----------
        top_k : int
            Number of top parents to select.
        offspring : int
            Number of children to produce.
        mutation_rate : float
            Mutation rate applied to each child.

        Returns
        -------
        list[dict]
            List of offspring parameter dicts.
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT name, params_json, fitness, generation FROM strategies ORDER BY fitness DESC LIMIT ?",
                    (top_k,),
                ).fetchall()
            finally:
                conn.close()

        if len(rows) < 2:
            logger.warning("breed_generation: need at least 2 strategies, have %d", len(rows))
            return []

        parents = [(r["name"], json.loads(r["params_json"]), r["fitness"]) for r in rows]
        max_gen = max(r["generation"] for r in rows)
        new_gen = max_gen + 1
        children: List[Dict[str, Any]] = []

        for i in range(offspring):
            pa, pb = self._rng.sample(parents, 2)
            child_params = self.crossover(pa[0], pb[0])
            child_params = self.mutate(child_params, mutation_rate)

            child_name = f"bred_gen{new_gen}_{i}"
            self.register_strategy(
                name=child_name,
                params=child_params,
                fitness=0.0,
                parent_a=pa[0],
                parent_b=pb[0],
                generation=new_gen,
            )
            children.append(child_params)

        logger.info(
            "Bred generation %d: %d offspring from %d parents",
            new_gen, len(children), len(parents),
        )
        return children

    def get_lineage(self, strategy_name: str) -> List[str]:
        """Trace the ancestry of a strategy through its breeding lineage.

        Parameters
        ----------
        strategy_name : str
            Strategy to trace.

        Returns
        -------
        list[str]
            List of ancestor strategy names (breadth-first), starting with
            the immediate parents.
        """
        ancestors: List[str] = []
        visited: set = set()
        queue = [strategy_name]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            with self._lock:
                conn = self._connect()
                try:
                    row = conn.execute(
                        "SELECT parent_a, parent_b FROM strategies WHERE name = ?",
                        (current,),
                    ).fetchone()
                finally:
                    conn.close()

            if row is None:
                continue

            for parent in [row["parent_a"], row["parent_b"]]:
                if parent and parent not in visited:
                    ancestors.append(parent)
                    queue.append(parent)

        return ancestors

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_params(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve params for a strategy by name."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT params_json FROM strategies WHERE name = ?", (name,)
                ).fetchone()
            finally:
                conn.close()

        if row is None:
            return None
        return json.loads(row["params_json"])
