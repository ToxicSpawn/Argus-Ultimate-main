from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChampionProfile:
    profile_id: str
    created_ts: float
    source_bundle_path: str
    config_hash: str
    strategy_set: List[str]
    version_label: str
    status: str  # active | archived


@dataclass(slots=True)
class ChallengerProfile:
    profile_id: str
    created_ts: float
    source_bundle_path: str
    parent_champion_id: str
    strategy_set: List[str]
    version_label: str
    status: str  # candidate | shadow_running | rejected | promoted
    evaluation_window_start: float
    evaluation_window_end: float


@dataclass(slots=True)
class PromotionDecision:
    champion_id: str
    challenger_id: str
    decision_ts: float
    decision: str  # promote | reject | hold
    reasons: List[str]
    metrics_summary: Dict[str, Any]
    promotion_score: float
    safety_checks_passed: bool


class ChampionChallengerEngine:
    """
    Advisory-first champion/challenger engine.

    This engine never places trades and does not modify execution flow directly.
    It only persists profiles, evaluates challengers from Strategy Evaluation metrics,
    and emits auditable promotion decisions plus promotion artifacts.
    """

    VALID_CHAMPION_STATUS = {"active", "archived"}
    VALID_CHALLENGER_STATUS = {"candidate", "shadow_running", "rejected", "promoted"}
    VALID_DECISIONS = {"promote", "reject", "hold"}

    def __init__(
        self,
        *,
        db_path: str = "data/champion_challenger.db",
        artifacts_dir: str = "deploy/promotions",
        enabled: bool = True,
        advisory_only: bool = True,
        min_trades_for_promotion: int = 10,
        max_drawdown_pct_for_promotion: float = 0.12,
        require_expectancy_improvement: bool = True,
        require_profit_factor_improvement: bool = False,
        require_sharpe_like_improvement: bool = True,
        promotion_weights: Optional[Dict[str, float]] = None,
        persist_interval_cycles: int = 10,
    ) -> None:
        self.db_path = str(db_path or "data/champion_challenger.db")
        self.artifacts_dir = str(artifacts_dir or "deploy/promotions")
        self.enabled = bool(enabled)
        self.advisory_only = bool(advisory_only)
        self.min_trades_for_promotion = max(1, int(min_trades_for_promotion or 10))
        self.max_drawdown_pct_for_promotion = self._normalize_drawdown_limit(max_drawdown_pct_for_promotion)
        self.require_expectancy_improvement = bool(require_expectancy_improvement)
        self.require_profit_factor_improvement = bool(require_profit_factor_improvement)
        self.require_sharpe_like_improvement = bool(require_sharpe_like_improvement)
        self.persist_interval_cycles = max(1, int(persist_interval_cycles or 10))
        self._last_persist_cycle = 0

        weights = dict(promotion_weights or {})
        self.promotion_weights: Dict[str, float] = {
            "net_pnl": float(weights.get("net_pnl", 1.0) or 1.0),
            "expectancy": float(weights.get("expectancy", 1.0) or 1.0),
            "profit_factor": float(weights.get("profit_factor", 0.75) or 0.75),
            "sharpe_like": float(weights.get("sharpe_like", 1.0) or 1.0),
            "drawdown_penalty": float(weights.get("drawdown_penalty", 1.25) or 1.25),
            "fee_penalty": float(weights.get("fee_penalty", 0.5) or 0.5),
        }

        self._champions: Dict[str, ChampionProfile] = {}
        self._challengers: Dict[str, ChallengerProfile] = {}
        self._decisions: List[PromotionDecision] = []

        self._init_schema()
        self.load_from_db()

    @staticmethod
    def _normalize_drawdown_limit(value: float) -> float:
        v = max(0.0, float(value or 0.0))
        if v <= 1.0:
            return v * 100.0
        return v

    @staticmethod
    def _safe_json_dumps(payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=True, default=str)

    @staticmethod
    def _safe_json_loads(payload: Any, fallback: Any) -> Any:
        if not payload:
            return fallback
        try:
            return json.loads(payload)
        except Exception:
            return fallback

    def _connect(self) -> sqlite3.Connection:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.db_path, timeout=15.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        return con

    def _init_schema(self) -> None:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS champion_profiles (
                    profile_id TEXT PRIMARY KEY,
                    created_ts REAL NOT NULL,
                    source_bundle_path TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    strategy_set_json TEXT NOT NULL,
                    version_label TEXT NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_champion_profiles_status ON champion_profiles(status)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS challenger_profiles (
                    profile_id TEXT PRIMARY KEY,
                    created_ts REAL NOT NULL,
                    source_bundle_path TEXT NOT NULL,
                    parent_champion_id TEXT NOT NULL,
                    strategy_set_json TEXT NOT NULL,
                    version_label TEXT NOT NULL,
                    status TEXT NOT NULL,
                    evaluation_window_start REAL NOT NULL,
                    evaluation_window_end REAL NOT NULL
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_challenger_profiles_status ON challenger_profiles(status)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS promotion_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    champion_id TEXT NOT NULL,
                    challenger_id TEXT NOT NULL,
                    decision_ts REAL NOT NULL,
                    decision TEXT NOT NULL,
                    reasons_json TEXT NOT NULL,
                    metrics_summary_json TEXT NOT NULL,
                    promotion_score REAL NOT NULL,
                    safety_checks_passed INTEGER NOT NULL,
                    artifact_path TEXT
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_promotion_decisions_challenger ON promotion_decisions(challenger_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_promotion_decisions_ts ON promotion_decisions(decision_ts)")
            con.commit()

    def load_from_db(self) -> None:
        self._champions.clear()
        self._challengers.clear()
        self._decisions.clear()
        with self._connect() as con:
            cur = con.cursor()
            for row in cur.execute("SELECT * FROM champion_profiles"):
                profile = ChampionProfile(
                    profile_id=str(row["profile_id"]),
                    created_ts=float(row["created_ts"]),
                    source_bundle_path=str(row["source_bundle_path"] or ""),
                    config_hash=str(row["config_hash"] or ""),
                    strategy_set=list(self._safe_json_loads(row["strategy_set_json"], [])),
                    version_label=str(row["version_label"] or ""),
                    status=str(row["status"] or "archived"),
                )
                self._champions[profile.profile_id] = profile

            for row in cur.execute("SELECT * FROM challenger_profiles"):
                profile = ChallengerProfile(
                    profile_id=str(row["profile_id"]),
                    created_ts=float(row["created_ts"]),
                    source_bundle_path=str(row["source_bundle_path"] or ""),
                    parent_champion_id=str(row["parent_champion_id"] or ""),
                    strategy_set=list(self._safe_json_loads(row["strategy_set_json"], [])),
                    version_label=str(row["version_label"] or ""),
                    status=str(row["status"] or "candidate"),
                    evaluation_window_start=float(row["evaluation_window_start"] or 0.0),
                    evaluation_window_end=float(row["evaluation_window_end"] or 0.0),
                )
                self._challengers[profile.profile_id] = profile

            for row in cur.execute("SELECT * FROM promotion_decisions ORDER BY decision_ts ASC"):
                decision = PromotionDecision(
                    champion_id=str(row["champion_id"] or ""),
                    challenger_id=str(row["challenger_id"] or ""),
                    decision_ts=float(row["decision_ts"]),
                    decision=str(row["decision"] or "hold"),
                    reasons=list(self._safe_json_loads(row["reasons_json"], [])),
                    metrics_summary=dict(self._safe_json_loads(row["metrics_summary_json"], {})),
                    promotion_score=float(row["promotion_score"] or 0.0),
                    safety_checks_passed=bool(int(row["safety_checks_passed"] or 0)),
                )
                artifact_path = str(row["artifact_path"] or "")
                if artifact_path:
                    decision.metrics_summary.setdefault("artifact_path", artifact_path)
                self._decisions.append(decision)

    def persist_to_db(self) -> None:
        with self._connect() as con:
            cur = con.cursor()
            for profile in self._champions.values():
                cur.execute(
                    """
                    INSERT INTO champion_profiles
                    (profile_id, created_ts, source_bundle_path, config_hash, strategy_set_json, version_label, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(profile_id) DO UPDATE SET
                        created_ts=excluded.created_ts,
                        source_bundle_path=excluded.source_bundle_path,
                        config_hash=excluded.config_hash,
                        strategy_set_json=excluded.strategy_set_json,
                        version_label=excluded.version_label,
                        status=excluded.status
                    """,
                    (
                        profile.profile_id,
                        float(profile.created_ts),
                        str(profile.source_bundle_path or ""),
                        str(profile.config_hash or ""),
                        self._safe_json_dumps(list(profile.strategy_set or [])),
                        str(profile.version_label or ""),
                        str(profile.status or "archived"),
                    ),
                )
            for profile in self._challengers.values():
                cur.execute(
                    """
                    INSERT INTO challenger_profiles
                    (profile_id, created_ts, source_bundle_path, parent_champion_id, strategy_set_json, version_label, status, evaluation_window_start, evaluation_window_end)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(profile_id) DO UPDATE SET
                        created_ts=excluded.created_ts,
                        source_bundle_path=excluded.source_bundle_path,
                        parent_champion_id=excluded.parent_champion_id,
                        strategy_set_json=excluded.strategy_set_json,
                        version_label=excluded.version_label,
                        status=excluded.status,
                        evaluation_window_start=excluded.evaluation_window_start,
                        evaluation_window_end=excluded.evaluation_window_end
                    """,
                    (
                        profile.profile_id,
                        float(profile.created_ts),
                        str(profile.source_bundle_path or ""),
                        str(profile.parent_champion_id or ""),
                        self._safe_json_dumps(list(profile.strategy_set or [])),
                        str(profile.version_label or ""),
                        str(profile.status or "candidate"),
                        float(profile.evaluation_window_start),
                        float(profile.evaluation_window_end),
                    ),
                )
            con.commit()

    def maybe_persist(self, *, cycle_id: int) -> None:
        if int(cycle_id) <= 0:
            return
        if int(cycle_id) - int(self._last_persist_cycle) < int(self.persist_interval_cycles):
            return
        self.persist_to_db()
        self._last_persist_cycle = int(cycle_id)

    def get_active_champion(self) -> Optional[ChampionProfile]:
        active = [p for p in self._champions.values() if p.status == "active"]
        if active:
            active.sort(key=lambda p: p.created_ts, reverse=True)
            return active[0]
        if not self._champions:
            return None
        fallback = list(self._champions.values())
        fallback.sort(key=lambda p: p.created_ts, reverse=True)
        return fallback[0]

    def register_champion(
        self,
        *,
        profile_id: str,
        source_bundle_path: str,
        config_hash: str,
        strategy_set: List[str],
        version_label: str,
        status: str = "active",
    ) -> ChampionProfile:
        now = float(time.time())
        status_n = str(status or "active").strip().lower()
        if status_n not in self.VALID_CHAMPION_STATUS:
            status_n = "active"
        if status_n == "active":
            for pid, profile in list(self._champions.items()):
                if profile.status == "active" and pid != profile_id:
                    self._champions[pid] = ChampionProfile(**{**asdict(profile), "status": "archived"})
        profile = ChampionProfile(
            profile_id=str(profile_id),
            created_ts=now,
            source_bundle_path=str(source_bundle_path or ""),
            config_hash=str(config_hash or ""),
            strategy_set=[str(s) for s in list(strategy_set or []) if str(s).strip()],
            version_label=str(version_label or ""),
            status=status_n,
        )
        self._champions[profile.profile_id] = profile
        self.persist_to_db()
        logger.info("champion profile registered: %s", profile.profile_id)
        return profile

    def register_challenger(
        self,
        *,
        profile_id: str,
        source_bundle_path: str,
        parent_champion_id: str,
        strategy_set: List[str],
        version_label: str,
        status: str = "candidate",
    ) -> ChallengerProfile:
        now = float(time.time())
        status_n = str(status or "candidate").strip().lower()
        if status_n not in self.VALID_CHALLENGER_STATUS:
            status_n = "candidate"
        profile = ChallengerProfile(
            profile_id=str(profile_id),
            created_ts=now,
            source_bundle_path=str(source_bundle_path or ""),
            parent_champion_id=str(parent_champion_id or ""),
            strategy_set=[str(s) for s in list(strategy_set or []) if str(s).strip()],
            version_label=str(version_label or ""),
            status=status_n,
            evaluation_window_start=now,
            evaluation_window_end=0.0,
        )
        self._challengers[profile.profile_id] = profile
        self.persist_to_db()
        logger.info("challenger registered: %s", profile.profile_id)
        return profile

    def list_challengers(self, *, status: Optional[str] = None) -> List[ChallengerProfile]:
        rows = list(self._challengers.values())
        if status:
            rows = [r for r in rows if r.status == str(status).strip().lower()]
        rows.sort(key=lambda p: p.created_ts, reverse=True)
        return rows

    def list_pending_promotion_decisions(self, *, limit: int = 20) -> List[PromotionDecision]:
        rows = [d for d in self._decisions if d.decision == "hold"]
        rows.sort(key=lambda d: d.decision_ts, reverse=True)
        return rows[: max(1, int(limit or 20))]

    def best_challengers_by_promotion_score(self, *, limit: int = 5) -> List[PromotionDecision]:
        latest_by_challenger: Dict[str, PromotionDecision] = {}
        for decision in self._decisions:
            latest_by_challenger[decision.challenger_id] = decision
        rows = list(latest_by_challenger.values())
        rows.sort(key=lambda d: float(d.promotion_score), reverse=True)
        return rows[: max(1, int(limit or 5))]

    def rejected_challengers(self, *, limit: int = 10) -> List[PromotionDecision]:
        rows = [d for d in self._decisions if d.decision == "reject"]
        rows.sort(key=lambda d: d.decision_ts, reverse=True)
        return rows[: max(1, int(limit or 10))]

    @staticmethod
    def _collect_profile_metrics(strategy_engine: Any, strategy_set: List[str], regime_label: Optional[str]) -> Dict[str, Any]:
        strategies = [str(s) for s in list(strategy_set or []) if str(s).strip()]
        breakdown: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []

        total_trades = 0
        total_net = 0.0
        total_fees = 0.0
        max_drawdown = 0.0
        expectancy_weighted = 0.0
        profit_factor_weighted = 0.0
        sharpe_like_weighted = 0.0

        for strategy_name in strategies:
            metric = None
            try:
                metric = strategy_engine.get_metrics(
                    strategy_name=strategy_name,
                    symbol=None,
                    regime_label=(regime_label or None),
                )
            except Exception:
                metric = None
            if metric is None:
                missing.append(strategy_name)
                continue

            trades = int(getattr(metric, "trades_count", 0) or 0)
            net_pnl = float(getattr(metric, "net_pnl_aud", 0.0) or 0.0)
            fees = float(getattr(metric, "total_fees_aud", 0.0) or 0.0)
            dd = float(getattr(metric, "max_drawdown_pct", 0.0) or 0.0)
            expectancy = float(getattr(metric, "expectancy", 0.0) or 0.0)
            profit_factor = float(getattr(metric, "profit_factor", 0.0) or 0.0)
            sharpe_like = float(getattr(metric, "sharpe_like_score", 0.0) or 0.0)

            breakdown[strategy_name] = {
                "trades_count": trades,
                "net_pnl_aud": net_pnl,
                "total_fees_aud": fees,
                "max_drawdown_pct": dd,
                "expectancy": expectancy,
                "profit_factor": profit_factor,
                "sharpe_like_score": sharpe_like,
            }

            weight = max(1, trades)
            total_trades += trades
            total_net += net_pnl
            total_fees += fees
            max_drawdown = max(max_drawdown, dd)
            expectancy_weighted += expectancy * weight
            profit_factor_weighted += profit_factor * weight
            sharpe_like_weighted += sharpe_like * weight

        denom = max(1, total_trades)
        return {
            "strategy_count": len(strategies),
            "covered_strategy_count": len(breakdown),
            "missing_strategies": missing,
            "trades_count": int(total_trades),
            "net_pnl_aud": float(total_net),
            "total_fees_aud": float(total_fees),
            "max_drawdown_pct": float(max_drawdown),
            "expectancy": float(expectancy_weighted / denom),
            "profit_factor": float(profit_factor_weighted / denom),
            "sharpe_like_score": float(sharpe_like_weighted / denom),
            "strategy_breakdown": breakdown,
        }

    def _promotion_score(self, champion: Dict[str, Any], challenger: Dict[str, Any]) -> float:
        w = self.promotion_weights
        score = 0.0
        score += w["net_pnl"] * (float(challenger.get("net_pnl_aud", 0.0)) - float(champion.get("net_pnl_aud", 0.0)))
        score += w["expectancy"] * (float(challenger.get("expectancy", 0.0)) - float(champion.get("expectancy", 0.0)))
        score += w["profit_factor"] * (
            float(challenger.get("profit_factor", 0.0)) - float(champion.get("profit_factor", 0.0))
        )
        score += w["sharpe_like"] * (
            float(challenger.get("sharpe_like_score", 0.0)) - float(champion.get("sharpe_like_score", 0.0))
        )
        score -= w["drawdown_penalty"] * max(
            0.0,
            float(challenger.get("max_drawdown_pct", 0.0)) - float(champion.get("max_drawdown_pct", 0.0)),
        )
        score -= w["fee_penalty"] * max(
            0.0,
            float(challenger.get("total_fees_aud", 0.0)) - float(champion.get("total_fees_aud", 0.0)),
        )
        return float(score)

    def _evaluate_rules(
        self,
        *,
        champion_metrics: Dict[str, Any],
        challenger_metrics: Dict[str, Any],
        critical_safety_checks_passed: bool,
    ) -> tuple[str, List[str], bool]:
        reasons: List[str] = []

        trades = int(challenger_metrics.get("trades_count", 0) or 0)
        if trades < self.min_trades_for_promotion:
            reasons.append(f"insufficient_trades:{trades}<{self.min_trades_for_promotion}")
            return "hold", reasons, False

        if not critical_safety_checks_passed:
            reasons.append("critical_safety_checks_failed")
            return "reject", reasons, False

        missing = list(challenger_metrics.get("missing_strategies", []) or [])
        covered = int(challenger_metrics.get("covered_strategy_count", 0) or 0)
        if covered <= 0:
            reasons.append("incomplete_metrics:no_strategy_coverage")
            return "hold", reasons, False
        if missing:
            reasons.append(f"incomplete_metrics:missing={','.join(missing)}")

        challenger_dd = float(challenger_metrics.get("max_drawdown_pct", 0.0) or 0.0)
        if challenger_dd > self.max_drawdown_pct_for_promotion:
            reasons.append(
                f"drawdown_too_high:{challenger_dd:.4f}>{self.max_drawdown_pct_for_promotion:.4f}"
            )
            return "reject", reasons, False

        champion_expectancy = float(champion_metrics.get("expectancy", 0.0) or 0.0)
        challenger_expectancy = float(challenger_metrics.get("expectancy", 0.0) or 0.0)
        if self.require_expectancy_improvement and challenger_expectancy <= champion_expectancy:
            reasons.append(f"weak_expectancy:{challenger_expectancy:.6f}<={champion_expectancy:.6f}")
            return "reject", reasons, False

        champion_pf = float(champion_metrics.get("profit_factor", 0.0) or 0.0)
        challenger_pf = float(challenger_metrics.get("profit_factor", 0.0) or 0.0)
        if self.require_profit_factor_improvement and challenger_pf <= champion_pf:
            reasons.append(f"weak_profit_factor:{challenger_pf:.6f}<={champion_pf:.6f}")
            return "reject", reasons, False

        champion_sharpe = float(champion_metrics.get("sharpe_like_score", 0.0) or 0.0)
        challenger_sharpe = float(challenger_metrics.get("sharpe_like_score", 0.0) or 0.0)
        if self.require_sharpe_like_improvement and challenger_sharpe <= champion_sharpe:
            reasons.append(f"weak_sharpe_like:{challenger_sharpe:.6f}<={champion_sharpe:.6f}")
            return "reject", reasons, False

        return "promote", reasons, True

    def _persist_decision(self, decision: PromotionDecision, artifact_path: Optional[str]) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO promotion_decisions
                (champion_id, challenger_id, decision_ts, decision, reasons_json, metrics_summary_json, promotion_score, safety_checks_passed, artifact_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(decision.champion_id),
                    str(decision.challenger_id),
                    float(decision.decision_ts),
                    str(decision.decision),
                    self._safe_json_dumps(list(decision.reasons or [])),
                    self._safe_json_dumps(dict(decision.metrics_summary or {})),
                    float(decision.promotion_score),
                    1 if bool(decision.safety_checks_passed) else 0,
                    str(artifact_path or ""),
                ),
            )
            con.commit()

    def _write_promotion_artifact(self, decision: PromotionDecision) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(decision.decision_ts))
        dir_name = f"{ts}_{decision.challenger_id}"
        out_dir = Path(self.artifacts_dir) / dir_name
        out_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "decision": asdict(decision),
            "advisory_only": bool(self.advisory_only),
            "generated_ts": float(time.time()),
        }
        decision_json = out_dir / "promotion_decision.json"
        decision_json.write_text(self._safe_json_dumps(payload), encoding="utf-8")

        notes = [
            "# ARGUS Champion/Challenger Promotion Decision",
            "",
            f"- champion_id: {decision.champion_id}",
            f"- challenger_id: {decision.challenger_id}",
            f"- decision: {decision.decision}",
            f"- promotion_score: {decision.promotion_score:.6f}",
            f"- safety_checks_passed: {bool(decision.safety_checks_passed)}",
            f"- advisory_only: {bool(self.advisory_only)}",
            "",
            "## Reasons",
        ]
        for reason in list(decision.reasons or []):
            notes.append(f"- {reason}")
        notes.append("")
        notes.append("## Metrics Summary")
        notes.append("```json")
        notes.append(json.dumps(decision.metrics_summary, ensure_ascii=True, indent=2, default=str))
        notes.append("```")
        (out_dir / "promotion_note.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
        return str(out_dir)

    def evaluate_challenger(
        self,
        *,
        challenger_id: str,
        strategy_evaluation_engine: Any,
        regime_label: Optional[str] = None,
        critical_safety_checks_passed: bool = True,
    ) -> PromotionDecision:
        now = float(time.time())
        challenger = self._challengers.get(str(challenger_id))
        champion = self.get_active_champion()

        if challenger is None:
            decision = PromotionDecision(
                champion_id="",
                challenger_id=str(challenger_id),
                decision_ts=now,
                decision="hold",
                reasons=["challenger_not_found"],
                metrics_summary={},
                promotion_score=0.0,
                safety_checks_passed=False,
            )
            self._persist_decision(decision, artifact_path=None)
            self._decisions.append(decision)
            return decision

        if champion is None:
            decision = PromotionDecision(
                champion_id="",
                challenger_id=challenger.profile_id,
                decision_ts=now,
                decision="hold",
                reasons=["champion_not_found"],
                metrics_summary={},
                promotion_score=0.0,
                safety_checks_passed=False,
            )
            self._persist_decision(decision, artifact_path=None)
            self._decisions.append(decision)
            return decision

        champion_metrics = self._collect_profile_metrics(
            strategy_engine=strategy_evaluation_engine,
            strategy_set=champion.strategy_set,
            regime_label=regime_label,
        )
        challenger_metrics = self._collect_profile_metrics(
            strategy_engine=strategy_evaluation_engine,
            strategy_set=challenger.strategy_set,
            regime_label=regime_label,
        )

        proposed_decision, reasons, safety_ok = self._evaluate_rules(
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
            critical_safety_checks_passed=critical_safety_checks_passed,
        )

        score = self._promotion_score(champion_metrics, challenger_metrics)
        if proposed_decision == "promote" and score <= 0.0:
            proposed_decision = "reject"
            reasons.append(f"non_positive_promotion_score:{score:.6f}")
            safety_ok = False

        if proposed_decision == "promote" and self.advisory_only:
            reasons.append("advisory_only_no_auto_promotion")

        challenger_status = challenger.status
        if proposed_decision == "reject":
            challenger_status = "rejected"
        elif challenger.status == "candidate":
            challenger_status = "shadow_running"

        updated = ChallengerProfile(
            **{
                **asdict(challenger),
                "status": challenger_status,
                "evaluation_window_end": now,
            }
        )
        self._challengers[updated.profile_id] = updated

        summary = {
            "regime_label": (regime_label or ""),
            "champion": champion_metrics,
            "challenger": challenger_metrics,
            "delta": {
                "net_pnl_aud": float(challenger_metrics.get("net_pnl_aud", 0.0)) - float(champion_metrics.get("net_pnl_aud", 0.0)),
                "expectancy": float(challenger_metrics.get("expectancy", 0.0)) - float(champion_metrics.get("expectancy", 0.0)),
                "profit_factor": float(challenger_metrics.get("profit_factor", 0.0)) - float(champion_metrics.get("profit_factor", 0.0)),
                "sharpe_like_score": float(challenger_metrics.get("sharpe_like_score", 0.0)) - float(champion_metrics.get("sharpe_like_score", 0.0)),
                "max_drawdown_pct": float(challenger_metrics.get("max_drawdown_pct", 0.0)) - float(champion_metrics.get("max_drawdown_pct", 0.0)),
                "total_fees_aud": float(challenger_metrics.get("total_fees_aud", 0.0)) - float(champion_metrics.get("total_fees_aud", 0.0)),
            },
        }

        decision = PromotionDecision(
            champion_id=champion.profile_id,
            challenger_id=challenger.profile_id,
            decision_ts=now,
            decision=proposed_decision,
            reasons=list(reasons),
            metrics_summary=summary,
            promotion_score=float(score),
            safety_checks_passed=bool(safety_ok),
        )

        artifact_path = None
        if decision.decision == "promote":
            artifact_path = self._write_promotion_artifact(decision)
            decision.metrics_summary["artifact_path"] = artifact_path
            logger.info(
                "promotion decision generated: champion=%s challenger=%s score=%.6f artifact=%s",
                decision.champion_id,
                decision.challenger_id,
                float(decision.promotion_score),
                artifact_path,
            )
        else:
            logger.info(
                "promotion decision generated: champion=%s challenger=%s decision=%s score=%.6f",
                decision.champion_id,
                decision.challenger_id,
                decision.decision,
                float(decision.promotion_score),
            )

        self._persist_decision(decision, artifact_path=artifact_path)
        self._decisions.append(decision)
        self.persist_to_db()
        return decision

    def evaluate_all(
        self,
        *,
        strategy_evaluation_engine: Any,
        regime_label: Optional[str] = None,
    ) -> List[PromotionDecision]:
        if not self.enabled:
            return []
        decisions: List[PromotionDecision] = []
        for challenger in self.list_challengers():
            if challenger.status not in {"candidate", "shadow_running"}:
                continue
            try:
                decision = self.evaluate_challenger(
                    challenger_id=challenger.profile_id,
                    strategy_evaluation_engine=strategy_evaluation_engine,
                    regime_label=regime_label,
                    critical_safety_checks_passed=True,
                )
                decisions.append(decision)
            except Exception as e:
                logger.warning("challenger evaluation failed for %s: %s", challenger.profile_id, e)
        return decisions
