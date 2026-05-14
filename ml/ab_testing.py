from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

_SUPPORTED_METRICS = {"sharpe", "returns", "accuracy", "precision"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ab_tests (
    test_id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    winner TEXT,
    promoted_model TEXT,
    abort_reason TEXT,
    analysis_json TEXT
);

CREATE TABLE IF NOT EXISTS ab_assignments (
    test_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    assigned_model TEXT NOT NULL,
    assigned_at REAL NOT NULL,
    outcome_recorded INTEGER NOT NULL DEFAULT 0,
    segments_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (test_id, request_id)
);

CREATE TABLE IF NOT EXISTS ab_outcomes (
    outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    model TEXT NOT NULL,
    metric_value REAL NOT NULL,
    reward REAL NOT NULL,
    recorded_at REAL NOT NULL,
    segments_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_ab_assignments_test_id ON ab_assignments(test_id);
CREATE INDEX IF NOT EXISTS idx_ab_outcomes_test_id ON ab_outcomes(test_id);
CREATE INDEX IF NOT EXISTS idx_ab_outcomes_test_model ON ab_outcomes(test_id, model);
"""


@dataclass
class ABTestConfig:
    test_name: str
    model_a_name: str
    model_b_name: str
    traffic_split: float
    primary_metric: str
    minimum_samples: int = 1000
    significance_level: float = 0.05
    test_duration_hours: int = 24


@dataclass
class ABTestResult:
    test_name: str
    model_a_performance: Dict[str, Any]
    model_b_performance: Dict[str, Any]
    winner: str
    p_value: float
    confidence_interval: Tuple[float, float]
    statistical_power: float
    recommendation: str


@dataclass
class ChampionChallengerConfig:
    """Configuration for a production champion/challenger model challenge."""

    challenge_name: str
    champion_model: str
    challenger_model: str
    primary_metric: str
    traffic_to_champion: float = 0.9
    minimum_samples: int = 1000
    significance_level: float = 0.05
    test_duration_hours: int = 24


@dataclass
class ChampionChallengerDecision:
    """Deterministic assignment result for one request."""

    test_id: str
    request_id: str
    assigned_arm: str
    assigned_model: str
    is_challenger: bool


@dataclass
class ChampionChallengerSummary:
    """Current champion/challenger analysis summary."""

    test_id: str
    status: str
    winner: str
    promoted_model: Optional[str]
    recommendation: str
    champion_samples: int
    challenger_samples: int
    champion_mean: float
    challenger_mean: float
    challenger_probability_best: float
    p_value: float
    stop: bool
    stop_reason: str


class ChampionChallengerManager:
    """Trading-friendly wrapper around ABTestEngine for model promotion gates."""

    def __init__(
        self,
        db_path: str = "data/ab_testing.db",
        *,
        engine: Optional["ABTestEngine"] = None,
    ) -> None:
        self.engine = engine or ABTestEngine(db_path=db_path)

    def start_challenge(self, config: ChampionChallengerConfig) -> str:
        """Create a champion/challenger challenge and return its test id."""
        if config.champion_model == config.challenger_model:
            raise ValueError("champion_model and challenger_model must be different")

        test_config = ABTestConfig(
            test_name=config.challenge_name,
            model_a_name=config.champion_model,
            model_b_name=config.challenger_model,
            traffic_split=config.traffic_to_champion,
            primary_metric=config.primary_metric,
            minimum_samples=config.minimum_samples,
            significance_level=config.significance_level,
            test_duration_hours=config.test_duration_hours,
        )
        return self.engine.create_test(test_config)

    def assign_model(self, test_id: str, request_id: str) -> ChampionChallengerDecision:
        """Assign a request to the champion or challenger model."""
        assigned_arm = self.engine.assign_model(test_id, request_id)
        config = self._get_config(test_id)
        assigned_model = config.model_a_name if assigned_arm == "A" else config.model_b_name
        return ChampionChallengerDecision(
            test_id=test_id,
            request_id=request_id,
            assigned_arm=assigned_arm,
            assigned_model=assigned_model,
            is_challenger=assigned_arm == "B",
        )

    def record_outcome(
        self,
        test_id: str,
        request_id: str,
        model_name: str,
        metric_value: float,
    ) -> None:
        """Record an outcome by model name rather than A/B arm."""
        config = self._get_config(test_id)
        if model_name == config.model_a_name:
            arm = "A"
        elif model_name == config.model_b_name:
            arm = "B"
        else:
            raise ValueError(f"model_name '{model_name}' is not part of challenge '{test_id}'")
        self.engine.record_outcome(test_id, request_id, arm, metric_value)

    def evaluate(self, test_id: str) -> ChampionChallengerSummary:
        """Analyze a challenge and return a promotion-oriented summary."""
        result = self.engine.analyze_test(test_id)
        stop_state = self.engine.check_stopping_conditions(test_id)
        test_row = self.engine._get_test_row(test_id)
        config = self.engine._config_from_row(test_row)

        promoted_model: Optional[str]
        if result.winner == "A":
            promoted_model = config.model_a_name
        elif result.winner == "B":
            promoted_model = config.model_b_name
        else:
            promoted_model = None

        return ChampionChallengerSummary(
            test_id=test_id,
            status=str(test_row["status"]),
            winner=result.winner,
            promoted_model=promoted_model,
            recommendation=result.recommendation,
            champion_samples=int(result.model_a_performance.get("sample_count", 0)),
            challenger_samples=int(result.model_b_performance.get("sample_count", 0)),
            champion_mean=float(result.model_a_performance.get("mean", 0.0)),
            challenger_mean=float(result.model_b_performance.get("mean", 0.0)),
            challenger_probability_best=float(
                result.model_b_performance.get("bayesian_probability_best", 0.5)
            ),
            p_value=float(result.p_value),
            stop=bool(stop_state["stop"]),
            stop_reason=str(stop_state["reason"]),
        )

    def promote_if_significant(
        self,
        test_id: str,
        *,
        require_challenger_win: bool = True,
    ) -> Dict[str, Any]:
        """Promote only when the statistical gate selects an acceptable winner."""
        summary = self.evaluate(test_id)
        if require_challenger_win and summary.winner != "B":
            return {
                "test_id": test_id,
                "promoted": False,
                "winner": summary.winner,
                "reason": "challenger_not_decisive",
            }
        if summary.winner == "inconclusive":
            return {
                "test_id": test_id,
                "promoted": False,
                "winner": summary.winner,
                "reason": "winner_inconclusive",
            }
        return self.engine.promote_winner(test_id)

    def _get_config(self, test_id: str) -> ABTestConfig:
        test_row = self.engine._get_test_row(test_id)
        return self.engine._config_from_row(test_row)


class ABTestEngine:
    """Persistent A/B testing engine for ML model evaluation."""

    def __init__(
        self,
        db_path: str = "data/ab_testing.db",
        *,
        bayesian_enabled: bool = True,
        early_stopping_enabled: bool = True,
        bandit_enabled: bool = True,
        segment_parser: Optional[Callable[[str], Dict[str, str]]] = None,
        bayesian_samples: int = 5000,
        early_stopping_probability: float = 0.95,
    ) -> None:
        self.db_path = db_path
        self.bayesian_enabled = bool(bayesian_enabled)
        self.early_stopping_enabled = bool(early_stopping_enabled)
        self.bandit_enabled = bool(bandit_enabled)
        self.segment_parser = segment_parser or self._default_segment_parser
        self.bayesian_samples = max(int(bayesian_samples), 1000)
        self.early_stopping_probability = min(max(float(early_stopping_probability), 0.5), 0.999)
        self._lock = threading.RLock()

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def create_test(self, config: ABTestConfig) -> str:
        """Create and persist a new A/B test."""
        self._validate_config(config)

        test_id = uuid.uuid4().hex
        now = time.time()

        with self._lock:
            con = self._connect()
            try:
                con.execute(
                    """
                    INSERT INTO ab_tests (test_id, config_json, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (test_id, json.dumps(asdict(config), sort_keys=True), "active", now, now),
                )
                con.commit()
            finally:
                con.close()

        logger.info(
            "ABTestEngine: created test %s (%s vs %s)",
            config.test_name,
            config.model_a_name,
            config.model_b_name,
        )
        return test_id

    def assign_model(self, test_id: str, request_id: str) -> str:
        """Assign request traffic to model A or B."""
        with self._lock:
            test_row = self._get_test_row(test_id)
            if test_row["status"] != "active":
                raise ValueError(f"test '{test_id}' is not active")

            existing = self._get_assignment(test_id, request_id)
            if existing is not None:
                return str(existing["assigned_model"])

            config = self._config_from_row(test_row)
            assigned_model = self._choose_model(test_id, request_id, config)
            segments = self.segment_parser(request_id)
            now = time.time()

            con = self._connect()
            try:
                con.execute(
                    """
                    INSERT INTO ab_assignments
                    (test_id, request_id, assigned_model, assigned_at, outcome_recorded, segments_json)
                    VALUES (?, ?, ?, ?, 0, ?)
                    """,
                    (test_id, request_id, assigned_model, now, json.dumps(segments, sort_keys=True)),
                )
                con.commit()
            finally:
                con.close()

        logger.debug(
            "ABTestEngine: assigned request %s to model %s for test %s",
            request_id,
            assigned_model,
            test_id,
        )
        return assigned_model

    def record_outcome(self, test_id: str, request_id: str, model: str, metric_value: float) -> None:
        """Persist one observed outcome for the test."""
        metric_value = float(metric_value)

        with self._lock:
            test_row = self._get_test_row(test_id)
            config = self._config_from_row(test_row)
            self._validate_model_letter(model)

            assignment = self._get_assignment(test_id, request_id)
            if assignment is None:
                segments = self.segment_parser(request_id)
                assignment_model = model
                con = self._connect()
                try:
                    con.execute(
                        """
                        INSERT INTO ab_assignments
                        (test_id, request_id, assigned_model, assigned_at, outcome_recorded, segments_json)
                        VALUES (?, ?, ?, ?, 0, ?)
                        """,
                        (test_id, request_id, model, time.time(), json.dumps(segments, sort_keys=True)),
                    )
                    con.commit()
                finally:
                    con.close()
            else:
                assignment_model = str(assignment["assigned_model"])
                segments = self._decode_segments(assignment["segments_json"])

            if assignment_model != model:
                logger.warning(
                    "ABTestEngine: outcome model mismatch for request %s on test %s (assigned=%s recorded=%s)",
                    request_id,
                    test_id,
                    assignment_model,
                    model,
                )

            reward = self._normalise_reward(config.primary_metric, metric_value)
            now = time.time()

            con = self._connect()
            try:
                con.execute(
                    """
                    INSERT INTO ab_outcomes
                    (test_id, request_id, model, metric_value, reward, recorded_at, segments_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        test_id,
                        request_id,
                        model,
                        metric_value,
                        reward,
                        now,
                        json.dumps(segments, sort_keys=True),
                    ),
                )
                con.execute(
                    """
                    UPDATE ab_assignments
                    SET outcome_recorded = 1
                    WHERE test_id = ? AND request_id = ?
                    """,
                    (test_id, request_id),
                )
                con.execute(
                    "UPDATE ab_tests SET updated_at = ? WHERE test_id = ?",
                    (now, test_id),
                )
                con.commit()
            finally:
                con.close()

        logger.debug(
            "ABTestEngine: recorded outcome for test %s request %s model %s metric %.6f",
            test_id,
            request_id,
            model,
            metric_value,
        )

    def analyze_test(self, test_id: str) -> ABTestResult:
        """Analyze collected outcomes and persist the latest result."""
        with self._lock:
            test_row = self._get_test_row(test_id)
            config = self._config_from_row(test_row)
            outcomes = self._load_outcomes(test_id)

        values_a = [row["metric_value"] for row in outcomes if row["model"] == "A"]
        values_b = [row["metric_value"] for row in outcomes if row["model"] == "B"]
        rewards_a = [row["reward"] for row in outcomes if row["model"] == "A"]
        rewards_b = [row["reward"] for row in outcomes if row["model"] == "B"]

        p_value, power = self.calculate_statistical_significance(values_a, values_b)
        confidence_interval = self._calculate_confidence_interval(values_a, values_b, config.significance_level)

        posterior_b_better = self._bayesian_probability_b_better(rewards_a, rewards_b)
        posterior_a_better = 1.0 - posterior_b_better

        model_a_performance = self._summarise_performance(
            values=values_a,
            rewards=rewards_a,
            model_name=config.model_a_name,
            primary_metric=config.primary_metric,
            outcomes=outcomes,
            model="A",
        )
        model_b_performance = self._summarise_performance(
            values=values_b,
            rewards=rewards_b,
            model_name=config.model_b_name,
            primary_metric=config.primary_metric,
            outcomes=outcomes,
            model="B",
        )

        model_a_performance["bayesian_probability_best"] = round(posterior_a_better, 6)
        model_b_performance["bayesian_probability_best"] = round(posterior_b_better, 6)

        winner = self._determine_winner(
            values_a=values_a,
            values_b=values_b,
            p_value=p_value,
            confidence_interval=confidence_interval,
            significance_level=config.significance_level,
            posterior_b_better=posterior_b_better,
        )
        recommendation = self._build_recommendation(
            config=config,
            values_a=values_a,
            values_b=values_b,
            winner=winner,
            p_value=p_value,
            power=power,
            confidence_interval=confidence_interval,
            posterior_b_better=posterior_b_better,
        )

        result = ABTestResult(
            test_name=config.test_name,
            model_a_performance=model_a_performance,
            model_b_performance=model_b_performance,
            winner=winner,
            p_value=float(p_value),
            confidence_interval=(float(confidence_interval[0]), float(confidence_interval[1])),
            statistical_power=float(power),
            recommendation=recommendation,
        )

        self._persist_analysis(test_id, result)
        return result

    def calculate_statistical_significance(
        self,
        values_a: Sequence[float],
        values_b: Sequence[float],
    ) -> Tuple[float, float]:
        """Return frequentist p-value and approximate statistical power."""
        arr_a = self._to_array(values_a)
        arr_b = self._to_array(values_b)
        if arr_a.size < 2 or arr_b.size < 2:
            return 1.0, 0.0

        try:
            _, p_value = stats.ttest_ind(arr_a, arr_b, equal_var=False, nan_policy="omit")
            if math.isnan(p_value):
                p_value = 1.0
        except Exception as exc:
            logger.warning("ABTestEngine: significance calculation failed: %s", exc)
            p_value = 1.0

        power = self._estimate_statistical_power(arr_a, arr_b)
        return float(p_value), float(power)

    def check_stopping_conditions(self, test_id: str) -> Dict[str, Any]:
        """Return the current stop decision and reason."""
        with self._lock:
            test_row = self._get_test_row(test_id)
            config = self._config_from_row(test_row)
            outcomes = self._load_outcomes(test_id)

        values_a = [row["metric_value"] for row in outcomes if row["model"] == "A"]
        values_b = [row["metric_value"] for row in outcomes if row["model"] == "B"]
        total_samples = len(values_a) + len(values_b)
        elapsed_hours = max(0.0, (time.time() - float(test_row["created_at"])) / 3600.0)

        result = self.analyze_test(test_id)
        min_samples_met = total_samples >= config.minimum_samples
        duration_met = elapsed_hours >= config.test_duration_hours
        ci_excludes_zero = result.confidence_interval[0] > 0.0 or result.confidence_interval[1] < 0.0
        early_sample_floor = max(50, int(config.minimum_samples * 0.25))
        enough_for_early_stop = total_samples >= early_sample_floor

        early_stop = (
            self.early_stopping_enabled
            and enough_for_early_stop
            and result.winner in {"A", "B"}
            and ci_excludes_zero
            and (
                result.p_value <= config.significance_level
                or result.model_a_performance.get("bayesian_probability_best", 0.0) >= self.early_stopping_probability
                or result.model_b_performance.get("bayesian_probability_best", 0.0) >= self.early_stopping_probability
            )
        )

        stop = early_stop or (min_samples_met and duration_met)
        if early_stop:
            reason = "early_stop_clear_winner"
        elif min_samples_met and duration_met:
            reason = "planned_duration_complete"
        elif min_samples_met:
            reason = "minimum_samples_met_waiting_for_duration"
        elif duration_met:
            reason = "duration_met_waiting_for_samples"
        else:
            reason = "continue_collecting_data"

        return {
            "test_id": test_id,
            "test_name": config.test_name,
            "status": test_row["status"],
            "stop": stop,
            "reason": reason,
            "winner": result.winner,
            "total_samples": total_samples,
            "samples_a": len(values_a),
            "samples_b": len(values_b),
            "minimum_samples_met": min_samples_met,
            "duration_elapsed_hours": round(elapsed_hours, 4),
            "duration_requirement_met": duration_met,
            "early_stop": early_stop,
            "p_value": result.p_value,
            "statistical_power": result.statistical_power,
            "confidence_interval": result.confidence_interval,
        }

    def promote_winner(self, test_id: str) -> Dict[str, Any]:
        """Mark the winning model as promoted if the result is decisive."""
        result = self.analyze_test(test_id)
        with self._lock:
            test_row = self._get_test_row(test_id)
            config = self._config_from_row(test_row)

            if result.winner == "A":
                promoted_model = config.model_a_name
            elif result.winner == "B":
                promoted_model = config.model_b_name
            else:
                return {
                    "test_id": test_id,
                    "test_name": config.test_name,
                    "promoted": False,
                    "winner": result.winner,
                    "reason": "winner_inconclusive",
                }

            now = time.time()
            con = self._connect()
            try:
                con.execute(
                    """
                    UPDATE ab_tests
                    SET status = ?, winner = ?, promoted_model = ?, updated_at = ?
                    WHERE test_id = ?
                    """,
                    ("promoted", result.winner, promoted_model, now, test_id),
                )
                con.commit()
            finally:
                con.close()

        logger.info("ABTestEngine: promoted %s for test %s", promoted_model, test_id)
        return {
            "test_id": test_id,
            "test_name": result.test_name,
            "promoted": True,
            "winner": result.winner,
            "promoted_model": promoted_model,
        }

    def abort_test(self, test_id: str, reason: str) -> None:
        """Abort an active test and persist the reason."""
        with self._lock:
            self._get_test_row(test_id)
            con = self._connect()
            try:
                con.execute(
                    """
                    UPDATE ab_tests
                    SET status = ?, abort_reason = ?, updated_at = ?
                    WHERE test_id = ?
                    """,
                    ("aborted", reason, time.time(), test_id),
                )
                con.commit()
            finally:
                con.close()

        logger.warning("ABTestEngine: aborted test %s (%s)", test_id, reason)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.row_factory = sqlite3.Row
        return con

    def _init_schema(self) -> None:
        with self._lock:
            con = self._connect()
            try:
                con.executescript(_SCHEMA)
                con.commit()
            finally:
                con.close()

    def _validate_config(self, config: ABTestConfig) -> None:
        if not config.test_name.strip():
            raise ValueError("test_name must not be empty")
        if not config.model_a_name.strip() or not config.model_b_name.strip():
            raise ValueError("model names must not be empty")
        if config.model_a_name == config.model_b_name:
            raise ValueError("model_a_name and model_b_name must be different")
        if not 0.0 < float(config.traffic_split) < 1.0:
            raise ValueError("traffic_split must be between 0 and 1")
        if config.primary_metric not in _SUPPORTED_METRICS:
            raise ValueError(f"primary_metric must be one of {_SUPPORTED_METRICS}")
        if int(config.minimum_samples) <= 0:
            raise ValueError("minimum_samples must be positive")
        if not 0.0 < float(config.significance_level) < 1.0:
            raise ValueError("significance_level must be between 0 and 1")
        if int(config.test_duration_hours) <= 0:
            raise ValueError("test_duration_hours must be positive")

    def _validate_model_letter(self, model: str) -> None:
        if model not in {"A", "B"}:
            raise ValueError("model must be 'A' or 'B'")

    def _get_test_row(self, test_id: str) -> sqlite3.Row:
        con = self._connect()
        try:
            row = con.execute(
                "SELECT * FROM ab_tests WHERE test_id = ?",
                (test_id,),
            ).fetchone()
        finally:
            con.close()
        if row is None:
            raise KeyError(f"unknown test_id '{test_id}'")
        return row

    def _get_assignment(self, test_id: str, request_id: str) -> Optional[sqlite3.Row]:
        con = self._connect()
        try:
            return con.execute(
                "SELECT * FROM ab_assignments WHERE test_id = ? AND request_id = ?",
                (test_id, request_id),
            ).fetchone()
        finally:
            con.close()

    def _config_from_row(self, row: sqlite3.Row) -> ABTestConfig:
        payload = json.loads(str(row["config_json"]))
        return ABTestConfig(**payload)

    def _choose_model(self, test_id: str, request_id: str, config: ABTestConfig) -> str:
        if self.bandit_enabled:
            bandit_choice = self._choose_model_bandit(test_id)
            if bandit_choice is not None:
                return bandit_choice

        fraction = self._stable_fraction(f"{test_id}:{request_id}")
        return "A" if fraction < float(config.traffic_split) else "B"

    def _choose_model_bandit(self, test_id: str) -> Optional[str]:
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT model, reward FROM ab_outcomes WHERE test_id = ?",
                (test_id,),
            ).fetchall()
        finally:
            con.close()

        rewards_a = [float(row["reward"]) for row in rows if row["model"] == "A"]
        rewards_b = [float(row["reward"]) for row in rows if row["model"] == "B"]
        total = len(rewards_a) + len(rewards_b)
        if total < 10:
            return None

        alpha_a = 1.0 + sum(rewards_a)
        beta_a = 1.0 + len(rewards_a) - sum(rewards_a)
        alpha_b = 1.0 + sum(rewards_b)
        beta_b = 1.0 + len(rewards_b) - sum(rewards_b)

        sample_a = float(np.random.beta(max(alpha_a, 1e-6), max(beta_a, 1e-6)))
        sample_b = float(np.random.beta(max(alpha_b, 1e-6), max(beta_b, 1e-6)))
        return "A" if sample_a >= sample_b else "B"

    def _stable_fraction(self, key: str) -> float:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        value = int(digest[:12], 16)
        return value / float(16 ** 12 - 1)

    def _normalise_reward(self, metric_name: str, metric_value: float) -> float:
        if metric_name in {"accuracy", "precision"} and 0.0 <= metric_value <= 1.0:
            reward = metric_value
        elif metric_name == "returns" and -1.0 <= metric_value <= 1.0:
            reward = (metric_value + 1.0) / 2.0
        else:
            reward = 1.0 / (1.0 + math.exp(-metric_value))
        return float(min(max(reward, 1e-6), 1.0 - 1e-6))

    def _load_outcomes(self, test_id: str) -> List[Dict[str, Any]]:
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT * FROM ab_outcomes WHERE test_id = ? ORDER BY recorded_at ASC, outcome_id ASC",
                (test_id,),
            ).fetchall()
        finally:
            con.close()

        return [
            {
                "request_id": str(row["request_id"]),
                "model": str(row["model"]),
                "metric_value": float(row["metric_value"]),
                "reward": float(row["reward"]),
                "recorded_at": float(row["recorded_at"]),
                "segments": self._decode_segments(row["segments_json"]),
            }
            for row in rows
        ]

    def _summarise_performance(
        self,
        *,
        values: Sequence[float],
        rewards: Sequence[float],
        model_name: str,
        primary_metric: str,
        outcomes: Sequence[Dict[str, Any]],
        model: str,
    ) -> Dict[str, Any]:
        arr = self._to_array(values)
        reward_arr = self._to_array(rewards)
        summary: Dict[str, Any] = {
            "model_name": model_name,
            "primary_metric": primary_metric,
            "sample_count": int(arr.size),
            "mean": float(np.mean(arr)) if arr.size else 0.0,
            "median": float(np.median(arr)) if arr.size else 0.0,
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "min": float(np.min(arr)) if arr.size else 0.0,
            "max": float(np.max(arr)) if arr.size else 0.0,
            "total": float(np.sum(arr)) if arr.size else 0.0,
            "win_rate": float(np.mean(arr > 0.0)) if arr.size else 0.0,
            "mean_reward": float(np.mean(reward_arr)) if reward_arr.size else 0.0,
            "segments": self._build_segment_analysis(outcomes, model),
        }
        return summary

    def _build_segment_analysis(
        self,
        outcomes: Sequence[Dict[str, Any]],
        model: str,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        segment_buckets: Dict[str, Dict[str, List[float]]] = {}
        for row in outcomes:
            if row["model"] != model:
                continue
            for key, value in row["segments"].items():
                segment_buckets.setdefault(key, {}).setdefault(value, []).append(float(row["metric_value"]))

        summary: Dict[str, Dict[str, Dict[str, float]]] = {}
        for key, values_by_segment in segment_buckets.items():
            summary[key] = {}
            for segment_value, segment_values in values_by_segment.items():
                arr = self._to_array(segment_values)
                summary[key][segment_value] = {
                    "count": float(arr.size),
                    "mean": float(np.mean(arr)) if arr.size else 0.0,
                    "median": float(np.median(arr)) if arr.size else 0.0,
                    "win_rate": float(np.mean(arr > 0.0)) if arr.size else 0.0,
                }
        return summary

    def _determine_winner(
        self,
        *,
        values_a: Sequence[float],
        values_b: Sequence[float],
        p_value: float,
        confidence_interval: Tuple[float, float],
        significance_level: float,
        posterior_b_better: float,
    ) -> str:
        mean_a = float(np.mean(self._to_array(values_a))) if values_a else 0.0
        mean_b = float(np.mean(self._to_array(values_b))) if values_b else 0.0

        if self.bayesian_enabled:
            if posterior_b_better >= 1.0 - significance_level:
                return "B"
            if posterior_b_better <= significance_level:
                return "A"

        significant = p_value <= significance_level
        ci_excludes_zero = confidence_interval[0] > 0.0 or confidence_interval[1] < 0.0
        if significant and ci_excludes_zero:
            if mean_b > mean_a:
                return "B"
            if mean_a > mean_b:
                return "A"
        return "inconclusive"

    def _build_recommendation(
        self,
        *,
        config: ABTestConfig,
        values_a: Sequence[float],
        values_b: Sequence[float],
        winner: str,
        p_value: float,
        power: float,
        confidence_interval: Tuple[float, float],
        posterior_b_better: float,
    ) -> str:
        total_samples = len(values_a) + len(values_b)
        if winner == "A":
            return (
                f"Promote {config.model_a_name}; model A leads on {config.primary_metric} "
                f"with p={p_value:.4f}, power={power:.3f}, and CI={confidence_interval}."
            )
        if winner == "B":
            return (
                f"Promote {config.model_b_name}; model B leads on {config.primary_metric} "
                f"with p={p_value:.4f}, power={power:.3f}, and posterior={posterior_b_better:.3f}."
            )
        if total_samples < config.minimum_samples:
            return (
                f"Keep running the test; only {total_samples} samples collected "
                f"vs required {config.minimum_samples}."
            )
        if power < 0.8:
            return (
                f"Result is inconclusive due to low power ({power:.3f}); extend the test "
                f"or increase traffic to improve sensitivity."
            )
        return (
            f"Result remains inconclusive despite sufficient samples; retain the current champion "
            f"until a clearer difference in {config.primary_metric} emerges."
        )

    def _calculate_confidence_interval(
        self,
        values_a: Sequence[float],
        values_b: Sequence[float],
        significance_level: float,
    ) -> Tuple[float, float]:
        arr_a = self._to_array(values_a)
        arr_b = self._to_array(values_b)
        if arr_a.size == 0 or arr_b.size == 0:
            return 0.0, 0.0

        mean_diff = float(np.mean(arr_b) - np.mean(arr_a))
        if arr_a.size < 2 or arr_b.size < 2:
            return mean_diff, mean_diff

        var_term = (np.var(arr_a, ddof=1) / arr_a.size) + (np.var(arr_b, ddof=1) / arr_b.size)
        se = math.sqrt(max(var_term, 0.0))
        if se <= 0.0:
            return mean_diff, mean_diff

        z_value = float(stats.norm.ppf(1.0 - significance_level / 2.0))
        margin = z_value * se
        return mean_diff - margin, mean_diff + margin

    def _estimate_statistical_power(self, arr_a: np.ndarray, arr_b: np.ndarray, alpha: float = 0.05) -> float:
        n_a = int(arr_a.size)
        n_b = int(arr_b.size)
        if n_a < 2 or n_b < 2:
            return 0.0

        var_a = float(np.var(arr_a, ddof=1))
        var_b = float(np.var(arr_b, ddof=1))
        pooled_denom = ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1)
        pooled_std = math.sqrt(max(pooled_denom, 1e-12))
        effect_size = abs(float(np.mean(arr_b) - np.mean(arr_a))) / pooled_std
        if effect_size <= 0.0:
            return 0.0

        n_eff = (n_a * n_b) / max(n_a + n_b, 1)
        z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
        z_effect = effect_size * math.sqrt(max(n_eff, 1e-12))
        power = stats.norm.cdf(z_effect - z_alpha) + stats.norm.cdf(-z_effect - z_alpha)
        return float(min(max(power, 0.0), 1.0))

    def _bayesian_probability_b_better(
        self,
        rewards_a: Sequence[float],
        rewards_b: Sequence[float],
    ) -> float:
        if not self.bayesian_enabled:
            return 0.5

        reward_a = self._to_array(rewards_a)
        reward_b = self._to_array(rewards_b)
        alpha_a = 1.0 + float(np.sum(reward_a))
        beta_a = 1.0 + float(reward_a.size - np.sum(reward_a))
        alpha_b = 1.0 + float(np.sum(reward_b))
        beta_b = 1.0 + float(reward_b.size - np.sum(reward_b))

        samples_a = np.random.beta(max(alpha_a, 1e-6), max(beta_a, 1e-6), size=self.bayesian_samples)
        samples_b = np.random.beta(max(alpha_b, 1e-6), max(beta_b, 1e-6), size=self.bayesian_samples)
        return float(np.mean(samples_b > samples_a))

    def _persist_analysis(self, test_id: str, result: ABTestResult) -> None:
        payload = json.dumps(asdict(result), sort_keys=True, default=str)
        now = time.time()
        con = self._connect()
        try:
            con.execute(
                """
                UPDATE ab_tests
                SET analysis_json = ?, winner = ?, updated_at = ?
                WHERE test_id = ?
                """,
                (payload, result.winner, now, test_id),
            )
            con.commit()
        finally:
            con.close()

    def _decode_segments(self, raw_segments: Any) -> Dict[str, str]:
        if not raw_segments:
            return {}
        try:
            decoded = json.loads(str(raw_segments))
        except Exception:
            logger.debug("ABTestEngine: failed to decode segments payload %r", raw_segments)
            return {}
        if not isinstance(decoded, dict):
            return {}
        return {str(key): str(value) for key, value in decoded.items()}

    def _default_segment_parser(self, request_id: str) -> Dict[str, str]:
        segments: Dict[str, str] = {}
        for token in str(request_id).split("|"):
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                segments[key] = value
        return segments

    def _to_array(self, values: Sequence[float]) -> np.ndarray:
        if not values:
            return np.array([], dtype=float)
        return np.asarray(list(values), dtype=float)
