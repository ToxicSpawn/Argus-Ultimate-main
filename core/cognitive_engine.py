#!/usr/bin/env python3
"""
Cognitive Architecture — 5-stage autonomous thinking loop.

PERCEIVE  -> gather all inputs (prices, regime, signals, risk, sentiment, execution)
ANALYZE   -> identify patterns, anomalies, opportunities, threats
PLAN      -> generate multiple action plans with expected outcomes
DECIDE    -> select best plan via multi-criteria decision analysis
LEARN     -> record outcome, update beliefs, adjust future decisions

Maintains a Bayesian belief system that updates with each observation.
SQLite persistence at data/cognitive_engine.db.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CognitiveResult:
    """Output of a single cognitive cycle."""
    perception: Dict[str, Any] = field(default_factory=dict)
    analysis: Dict[str, Any] = field(default_factory=dict)
    plans: List[Dict[str, Any]] = field(default_factory=list)
    chosen_plan: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reasoning: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Belief:
    """A single probabilistic belief about the market."""
    name: str
    probability: float  # 0..1
    evidence_count: int = 0
    last_updated: str = ""
    history: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bayesian helpers
# ---------------------------------------------------------------------------

def _bayesian_update(prior: float, likelihood: float, evidence_positive: bool) -> float:
    """Simple Bayesian update.  Returns posterior probability."""
    prior = max(0.01, min(0.99, prior))
    if evidence_positive:
        numerator = likelihood * prior
        denominator = likelihood * prior + (1.0 - likelihood) * (1.0 - prior)
    else:
        numerator = (1.0 - likelihood) * prior
        denominator = (1.0 - likelihood) * prior + likelihood * (1.0 - prior)
    if denominator == 0:
        return prior
    return max(0.01, min(0.99, numerator / denominator))


# ---------------------------------------------------------------------------
# CognitiveEngine
# ---------------------------------------------------------------------------

class CognitiveEngine:
    """
    Autonomous 5-stage cognitive loop for ARGUS.

    Parameters
    ----------
    config : dict, optional
        ``cognitive_engine`` section from unified_config.yaml.
    db_path : str, optional
        Override SQLite database path.
    """

    # Default tuning knobs
    _DEFAULTS: Dict[str, Any] = {
        "enabled": True,
        "db_path": "data/cognitive_engine.db",
        "max_plans": 5,
        "belief_prior": 0.5,
        "belief_likelihood": 0.7,
        "learning_rate": 0.1,
        "pattern_lookback": 50,
        "min_confidence_to_trade": 0.35,
        "anomaly_z_threshold": 2.5,
        "opportunity_min_score": 0.3,
        "threat_max_drawdown_pct": 5.0,
        "diversification_weight": 0.15,
        "risk_adjusted_return_weight": 0.50,
        "confidence_weight": 0.35,
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        db_path: Optional[str] = None,
    ) -> None:
        cfg = dict(self._DEFAULTS)
        if config:
            cfg.update(config)
        self._cfg = cfg
        self._enabled: bool = bool(cfg.get("enabled", True))
        self._max_plans: int = int(cfg.get("max_plans", 5))
        self._belief_prior: float = float(cfg.get("belief_prior", 0.5))
        self._belief_likelihood: float = float(cfg.get("belief_likelihood", 0.7))
        self._learning_rate: float = float(cfg.get("learning_rate", 0.1))
        self._pattern_lookback: int = int(cfg.get("pattern_lookback", 50))
        self._min_confidence: float = float(cfg.get("min_confidence_to_trade", 0.35))
        self._anomaly_z: float = float(cfg.get("anomaly_z_threshold", 2.5))
        self._opp_min: float = float(cfg.get("opportunity_min_score", 0.3))
        self._threat_dd: float = float(cfg.get("threat_max_drawdown_pct", 5.0))

        # Weights for plan scoring (MCDA)
        self._w_rar: float = float(cfg.get("risk_adjusted_return_weight", 0.50))
        self._w_conf: float = float(cfg.get("confidence_weight", 0.35))
        self._w_div: float = float(cfg.get("diversification_weight", 0.15))

        # Belief system: name -> Belief
        self._beliefs: Dict[str, Belief] = {}
        self._lock = threading.Lock()

        # Outcome history for learning
        self._outcome_history: List[Dict[str, Any]] = []

        # SQLite
        db = db_path or str(cfg.get("db_path", "data/cognitive_engine.db"))
        self._db_path = db
        self._init_db()
        self._load_beliefs()

        logger.info("CognitiveEngine initialised (enabled=%s, db=%s)", self._enabled, self._db_path)

    # ------------------------------------------------------------------
    # SQLite persistence
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS beliefs (
                    name TEXT PRIMARY KEY,
                    probability REAL NOT NULL,
                    evidence_count INTEGER NOT NULL DEFAULT 0,
                    last_updated TEXT NOT NULL,
                    history TEXT NOT NULL DEFAULT '[]'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cognitive_cycles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    perception TEXT,
                    analysis TEXT,
                    plans TEXT,
                    chosen_plan TEXT,
                    confidence REAL,
                    reasoning TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id INTEGER,
                    plan_id TEXT,
                    expected_return REAL,
                    actual_return REAL,
                    error REAL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.commit()

    def _load_beliefs(self) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute("SELECT name, probability, evidence_count, last_updated, history FROM beliefs").fetchall()
            for name, prob, count, updated, hist_json in rows:
                try:
                    history = json.loads(hist_json) if hist_json else []
                except Exception:
                    history = []
                self._beliefs[name] = Belief(
                    name=name, probability=prob, evidence_count=count,
                    last_updated=updated, history=history,
                )
        except Exception as exc:
            logger.warning("CognitiveEngine: failed to load beliefs — %s", exc)

    def _save_belief(self, belief: Belief) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO beliefs (name, probability, evidence_count, last_updated, history) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (belief.name, belief.probability, belief.evidence_count,
                     belief.last_updated, json.dumps(belief.history[-200:])),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("CognitiveEngine: failed to save belief %s — %s", belief.name, exc)

    def _persist_cycle(self, result: CognitiveResult) -> Optional[int]:
        try:
            with sqlite3.connect(self._db_path) as conn:
                cur = conn.execute(
                    "INSERT INTO cognitive_cycles (timestamp, perception, analysis, plans, chosen_plan, confidence, reasoning) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (result.timestamp, json.dumps(result.perception), json.dumps(result.analysis),
                     json.dumps(result.plans), json.dumps(result.chosen_plan),
                     result.confidence, result.reasoning),
                )
                conn.commit()
                return cur.lastrowid
        except Exception as exc:
            logger.warning("CognitiveEngine: persist failed — %s", exc)
            return None

    # ------------------------------------------------------------------
    # Belief system
    # ------------------------------------------------------------------

    def get_beliefs(self) -> Dict[str, float]:
        """Return current belief probabilities."""
        with self._lock:
            return {name: b.probability for name, b in self._beliefs.items()}

    def update_belief(self, belief_name: str, evidence: float = 0.7, direction: bool = True) -> float:
        """
        Bayesian update of a named belief.

        Parameters
        ----------
        belief_name : str
            Identifier (e.g. "btc_trending").
        evidence : float
            Likelihood ratio for the evidence (0..1).
        direction : bool
            True = evidence supports belief, False = evidence opposes.

        Returns
        -------
        float
            Posterior probability.
        """
        with self._lock:
            if belief_name not in self._beliefs:
                self._beliefs[belief_name] = Belief(
                    name=belief_name, probability=self._belief_prior,
                    last_updated=datetime.now(timezone.utc).isoformat(),
                )
            b = self._beliefs[belief_name]
            posterior = _bayesian_update(b.probability, evidence, direction)
            b.probability = posterior
            b.evidence_count += 1
            b.last_updated = datetime.now(timezone.utc).isoformat()
            b.history.append(posterior)
            if len(b.history) > 200:
                b.history = b.history[-200:]
            self._save_belief(b)
            return posterior

    def reset_belief(self, belief_name: str) -> None:
        """Reset a belief to the prior."""
        with self._lock:
            if belief_name in self._beliefs:
                self._beliefs[belief_name].probability = self._belief_prior
                self._beliefs[belief_name].history.append(self._belief_prior)
                self._save_belief(self._beliefs[belief_name])

    # ------------------------------------------------------------------
    # The 5-stage cognitive loop
    # ------------------------------------------------------------------

    def think(self, market_state: Dict[str, Any]) -> CognitiveResult:
        """
        Execute one full cognitive cycle.

        Parameters
        ----------
        market_state : dict
            Keys may include: prices, regime, signals, risk_metrics, sentiment,
            execution_quality, positions, portfolio, orderbook.

        Returns
        -------
        CognitiveResult
        """
        ts = datetime.now(timezone.utc).isoformat()

        # Stage 1: PERCEIVE
        perception = self._perceive(market_state)

        # Stage 2: ANALYZE
        analysis = self._analyze(perception, market_state)

        # Stage 3: PLAN
        plans = self._plan(analysis, market_state)

        # Stage 4: DECIDE
        chosen_plan, confidence, reasoning = self._decide(plans, analysis)

        # Stage 5: LEARN (deferred — we record the decision; actual outcome arrives later)
        self._learn_from_decision(analysis)

        result = CognitiveResult(
            perception=perception,
            analysis=analysis,
            plans=plans,
            chosen_plan=chosen_plan,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=ts,
        )
        self._persist_cycle(result)
        return result

    # ------------------------------------------------------------------
    # Stage 1 — PERCEIVE
    # ------------------------------------------------------------------

    def _perceive(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Gather and normalise all inputs into a structured perception."""
        perception: Dict[str, Any] = {}

        # Prices
        prices = state.get("prices", {})
        perception["prices"] = prices
        perception["price_changes"] = {}
        for sym, data in prices.items():
            if isinstance(data, dict):
                perception["price_changes"][sym] = data.get("change_pct", 0.0)
            elif isinstance(data, (int, float)):
                perception["price_changes"][sym] = 0.0

        # Regime
        perception["regime"] = state.get("regime", "unknown")

        # Signals
        perception["signals"] = state.get("signals", {})
        perception["signal_count"] = len(state.get("signals", {}))

        # Risk
        risk = state.get("risk_metrics", {})
        perception["drawdown_pct"] = risk.get("drawdown_pct", 0.0)
        perception["portfolio_var"] = risk.get("var", 0.0)
        perception["exposure_pct"] = risk.get("exposure_pct", 0.0)

        # Sentiment
        perception["sentiment"] = state.get("sentiment", {})
        perception["fear_greed"] = state.get("sentiment", {}).get("fear_greed", 50)

        # Execution quality
        perception["execution_quality"] = state.get("execution_quality", {})
        perception["avg_slippage_bps"] = state.get("execution_quality", {}).get("avg_slippage_bps", 0.0)

        # Positions
        perception["open_positions"] = len(state.get("positions", {}))
        perception["positions"] = state.get("positions", {})

        # Volatility
        perception["volatility"] = state.get("volatility", {})

        return perception

    # ------------------------------------------------------------------
    # Stage 2 — ANALYZE
    # ------------------------------------------------------------------

    def _analyze(self, perception: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Identify patterns, anomalies, opportunities, and threats."""
        analysis: Dict[str, Any] = {
            "patterns": [],
            "anomalies": [],
            "opportunities": [],
            "threats": [],
            "regime_assessment": "",
            "market_mood": "neutral",
        }

        # --- Regime assessment ---
        regime = perception.get("regime", "unknown")
        analysis["regime_assessment"] = regime

        # --- Pattern detection ---
        price_changes = perception.get("price_changes", {})
        if price_changes:
            avg_change = sum(price_changes.values()) / max(len(price_changes), 1)
            if avg_change > 1.0:
                analysis["patterns"].append({"type": "broad_rally", "strength": avg_change})
                analysis["market_mood"] = "bullish"
            elif avg_change < -1.0:
                analysis["patterns"].append({"type": "broad_selloff", "strength": abs(avg_change)})
                analysis["market_mood"] = "bearish"

            # Divergence: some symbols up, some down
            ups = [s for s, c in price_changes.items() if c > 0.5]
            downs = [s for s, c in price_changes.items() if c < -0.5]
            if ups and downs:
                analysis["patterns"].append({"type": "divergence", "up": ups, "down": downs})

        # --- Anomaly detection ---
        if price_changes and len(price_changes) >= 2:
            values = list(price_changes.values())
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance) if variance > 0 else 1e-9
            for sym, change in price_changes.items():
                z = abs(change - mean) / std if std > 1e-9 else 0.0
                if z > self._anomaly_z:
                    analysis["anomalies"].append({"symbol": sym, "z_score": round(z, 2), "change_pct": change})

        # --- Opportunity identification ---
        signals = perception.get("signals", {})
        for sym, sig in signals.items():
            score = 0.0
            if isinstance(sig, dict):
                score = sig.get("score", sig.get("strength", 0.0))
            elif isinstance(sig, (int, float)):
                score = float(sig)
            if abs(score) >= self._opp_min:
                direction = "long" if score > 0 else "short"
                analysis["opportunities"].append({
                    "symbol": sym, "direction": direction,
                    "signal_score": round(score, 4),
                })

        # --- Threat detection ---
        dd = perception.get("drawdown_pct", 0.0)
        if dd > self._threat_dd:
            analysis["threats"].append({"type": "drawdown", "severity": "high", "value": dd})
        elif dd > self._threat_dd * 0.6:
            analysis["threats"].append({"type": "drawdown", "severity": "moderate", "value": dd})

        slippage = perception.get("avg_slippage_bps", 0.0)
        if slippage > 15.0:
            analysis["threats"].append({"type": "execution_degradation", "slippage_bps": slippage})

        fear = perception.get("fear_greed", 50)
        if fear < 20:
            analysis["threats"].append({"type": "extreme_fear", "value": fear})
        elif fear > 80:
            analysis["threats"].append({"type": "extreme_greed", "value": fear})

        return analysis

    # ------------------------------------------------------------------
    # Stage 3 — PLAN
    # ------------------------------------------------------------------

    def _plan(self, analysis: Dict[str, Any], state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate candidate action plans."""
        plans: List[Dict[str, Any]] = []
        threats = analysis.get("threats", [])
        opps = analysis.get("opportunities", [])
        mood = analysis.get("market_mood", "neutral")

        # Plan 0: Hold / do nothing
        plans.append({
            "id": "hold",
            "action": "hold",
            "description": "Maintain current positions, no new trades",
            "expected_return_pct": 0.0,
            "risk_pct": 0.0,
            "confidence": 0.5,
            "diversification_score": 0.5,
        })

        # Threat-driven plans
        high_threats = [t for t in threats if t.get("severity") == "high"]
        if high_threats:
            plans.append({
                "id": "de_risk",
                "action": "reduce_exposure",
                "description": "Reduce all positions by 50% due to high threat level",
                "expected_return_pct": 0.0,
                "risk_pct": -2.0,
                "confidence": 0.7,
                "diversification_score": 0.3,
                "trigger_threats": [t["type"] for t in high_threats],
            })

        # Opportunity-driven plans
        for opp in opps[:self._max_plans - len(plans)]:
            sym = opp["symbol"]
            direction = opp["direction"]
            score = abs(opp["signal_score"])
            expected_ret = score * 1.5  # rough heuristic
            risk = expected_ret * 0.6
            plans.append({
                "id": f"trade_{sym}_{direction}",
                "action": f"open_{direction}",
                "symbol": sym,
                "direction": direction,
                "description": f"Open {direction} on {sym} (signal={opp['signal_score']:.3f})",
                "expected_return_pct": round(expected_ret, 3),
                "risk_pct": round(risk, 3),
                "confidence": round(min(score + 0.2, 0.95), 3),
                "diversification_score": 0.5,
            })

        # Mood-driven aggregate plan
        if mood == "bullish" and not high_threats:
            plans.append({
                "id": "increase_exposure",
                "action": "increase_exposure",
                "description": "Market mood bullish — increase overall position sizing",
                "expected_return_pct": 0.5,
                "risk_pct": 0.3,
                "confidence": 0.55,
                "diversification_score": 0.4,
            })
        elif mood == "bearish":
            plans.append({
                "id": "defensive",
                "action": "defensive_mode",
                "description": "Market mood bearish — tighten stops, reduce sizing",
                "expected_return_pct": 0.0,
                "risk_pct": -1.0,
                "confidence": 0.6,
                "diversification_score": 0.6,
            })

        return plans[:self._max_plans]

    # ------------------------------------------------------------------
    # Stage 4 — DECIDE (Multi-Criteria Decision Analysis)
    # ------------------------------------------------------------------

    def _decide(
        self, plans: List[Dict[str, Any]], analysis: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float, str]:
        """Score each plan and select the best."""
        if not plans:
            return {}, 0.0, "No plans generated"

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for plan in plans:
            expected_ret = plan.get("expected_return_pct", 0.0)
            risk = max(plan.get("risk_pct", 0.01), 0.01)
            rar = expected_ret / abs(risk) if abs(risk) > 1e-9 else expected_ret * 10.0
            conf = plan.get("confidence", 0.5)
            div = plan.get("diversification_score", 0.5)

            # Normalise rar to 0..1 range
            rar_norm = min(max(rar / 5.0, 0.0), 1.0)

            score = (
                self._w_rar * rar_norm
                + self._w_conf * conf
                + self._w_div * div
            )

            # Threat penalty: if there are high threats and plan increases exposure
            threats = analysis.get("threats", [])
            high_threats = [t for t in threats if t.get("severity") == "high"]
            if high_threats and plan.get("action", "").startswith("open_"):
                score *= 0.5

            scored.append((score, plan))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_plan = scored[0]
        confidence = min(best_score, 1.0)

        # Build reasoning
        reasons = []
        regime = analysis.get("regime_assessment", "unknown")
        reasons.append(f"Regime: {regime}")
        n_opp = len(analysis.get("opportunities", []))
        n_threat = len(analysis.get("threats", []))
        reasons.append(f"Opportunities: {n_opp}, Threats: {n_threat}")
        reasons.append(f"Selected plan '{best_plan.get('id', '?')}' with score {best_score:.3f}")
        if best_plan.get("action") == "hold":
            reasons.append("Holding due to insufficient edge or high risk")
        reasoning = "; ".join(reasons)

        return best_plan, round(confidence, 4), reasoning

    # ------------------------------------------------------------------
    # Stage 5 — LEARN
    # ------------------------------------------------------------------

    def _learn_from_decision(self, analysis: Dict[str, Any]) -> None:
        """Update beliefs based on current analysis."""
        mood = analysis.get("market_mood", "neutral")
        if mood == "bullish":
            self.update_belief("market_bullish", 0.7, True)
            self.update_belief("market_bearish", 0.7, False)
        elif mood == "bearish":
            self.update_belief("market_bearish", 0.7, True)
            self.update_belief("market_bullish", 0.7, False)

        # High-threat belief
        threats = analysis.get("threats", [])
        high = any(t.get("severity") == "high" for t in threats)
        self.update_belief("high_risk_environment", 0.7, high)

    def record_outcome(self, plan_id: str, expected_return: float, actual_return: float) -> None:
        """
        Record the actual outcome of a previous plan to improve future decisions.
        """
        error = actual_return - expected_return
        entry = {
            "plan_id": plan_id,
            "expected_return": expected_return,
            "actual_return": actual_return,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._outcome_history.append(entry)
        if len(self._outcome_history) > 500:
            self._outcome_history = self._outcome_history[-500:]

        # Persist
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT INTO outcomes (plan_id, expected_return, actual_return, error, timestamp) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (plan_id, expected_return, actual_return, error, entry["timestamp"]),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("CognitiveEngine: outcome persist failed — %s", exc)

        # Learn: update optimism/pessimism belief
        if error > 0:
            self.update_belief("predictions_optimistic", 0.6, False)
        else:
            self.update_belief("predictions_optimistic", 0.6, True)

    def get_prediction_accuracy(self) -> Dict[str, float]:
        """Return stats on how accurate our plans have been."""
        if not self._outcome_history:
            return {"count": 0, "mae": 0.0, "bias": 0.0, "direction_accuracy": 0.0}
        errors = [o["error"] for o in self._outcome_history]
        mae = sum(abs(e) for e in errors) / len(errors)
        bias = sum(errors) / len(errors)
        direction_hits = sum(
            1 for o in self._outcome_history
            if (o["expected_return"] >= 0 and o["actual_return"] >= 0) or
               (o["expected_return"] < 0 and o["actual_return"] < 0)
        )
        dir_acc = direction_hits / len(self._outcome_history)
        return {
            "count": len(self._outcome_history),
            "mae": round(mae, 6),
            "bias": round(bias, 6),
            "direction_accuracy": round(dir_acc, 4),
        }
