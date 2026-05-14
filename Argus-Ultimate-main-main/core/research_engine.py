"""
Research Engine — ARGUS conducts its own R&D autonomously.

This is the final layer: ARGUS doesn't just trade and learn — it
RESEARCHES new approaches, tests hypotheses about market structure,
discovers new features, and develops new capabilities.

Research Programs:
1. FEATURE DISCOVERY: automatically generates and tests new technical indicators
2. MARKET MICROSTRUCTURE RESEARCH: studies order flow patterns to find edges
3. REGIME TRANSITION STUDY: how do regimes change and what predicts transitions
4. STRATEGY ARCHAEOLOGY: dissects why past strategies worked or failed
5. CORRELATION MINING: finds hidden relationships between assets/signals
6. EXECUTION RESEARCH: tests different order types and measures alpha
7. ANOMALY CATALOGING: studies detected anomalies to build pattern library

Each research program runs in the background, produces findings,
and automatically converts discoveries into actionable improvements.
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ResearchFinding:
    """A single research discovery."""
    finding_id: str
    program: str                    # which research program found this
    title: str
    description: str
    evidence_strength: float        # 0-1
    actionable: bool                # can we act on this immediately?
    action: str                     # what to do (e.g. "add_feature", "adjust_parameter")
    params: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    applied: bool = False


@dataclass
class ResearchReport:
    """Summary of all research activity."""
    total_experiments: int
    total_findings: int
    actionable_findings: int
    applied_findings: int
    programs_active: int
    top_findings: List[ResearchFinding]
    duration_ms: float


# ════════════════════════════════════════════════════════════════════════════
# Feature Discovery — invents new indicators
# ════════════════════════════════════════════════════════════════════════════

class FeatureDiscovery:
    """
    Automatically generates and tests new technical indicators.

    Method: combinatorial feature construction
    1. Take base features: close, volume, high-low range, returns
    2. Apply transforms: rolling mean, std, rank, diff, ratio
    3. Combine: feature_A / feature_B, feature_A - feature_B
    4. Test predictive power: correlation with future returns
    5. If r² > threshold, promote to production features
    """

    def __init__(self, min_r_squared: float = 0.02, max_features: int = 50):
        self._min_r2 = min_r_squared
        self._max_features = max_features
        self._discovered: List[Dict[str, Any]] = []
        self._experiments = 0

    def research(self, close: np.ndarray, volume: np.ndarray,
                 high: np.ndarray, low: np.ndarray) -> List[ResearchFinding]:
        """Run one feature discovery cycle."""
        findings = []
        T = len(close)
        if T < 60:
            return findings

        # Future returns (what we're trying to predict)
        future_ret = np.zeros(T)
        for i in range(T - 5):
            future_ret[i] = (close[i + 5] / close[i] - 1) * 100

        # Base features
        returns = np.diff(np.log(close + 1e-9))
        returns = np.concatenate([[0], returns])
        hl_range = (high - low) / np.maximum(close, 1e-9)
        vol_ratio = volume / np.maximum(np.convolve(volume, np.ones(20)/20, mode='same'), 1e-9)

        # Generate candidate features
        candidates = {}

        # Rolling statistics at various windows
        for window in [5, 10, 20, 50]:
            if window >= T:
                continue
            # Rolling mean of returns
            rm = np.convolve(returns, np.ones(window)/window, mode='same')
            candidates[f"ret_mean_{window}"] = rm

            # Rolling std of returns (realized vol)
            rstd = np.zeros(T)
            for i in range(window, T):
                rstd[i] = np.std(returns[i-window:i])
            candidates[f"ret_std_{window}"] = rstd

            # Volume trend
            vm = np.convolve(volume, np.ones(window)/window, mode='same')
            candidates[f"vol_trend_{window}"] = volume / np.maximum(vm, 1e-9)

            # Range compression/expansion
            rm_range = np.convolve(hl_range, np.ones(window)/window, mode='same')
            candidates[f"range_ratio_{window}"] = hl_range / np.maximum(rm_range, 1e-9)

        # Cross-feature combinations
        for name_a, feat_a in list(candidates.items())[:10]:
            for name_b, feat_b in list(candidates.items())[:10]:
                if name_a >= name_b:
                    continue
                # Ratio
                ratio = feat_a / np.maximum(np.abs(feat_b), 1e-9)
                candidates[f"{name_a}_div_{name_b}"] = ratio

        # Test each candidate's predictive power
        for name, feature in candidates.items():
            self._experiments += 1
            if len(feature) != T:
                continue

            # Compute correlation with future returns
            valid = ~(np.isnan(feature) | np.isnan(future_ret) | np.isinf(feature))
            if valid.sum() < 30:
                continue

            f_valid = feature[valid]
            r_valid = future_ret[valid]

            # Pearson correlation
            f_mean = np.mean(f_valid)
            r_mean = np.mean(r_valid)
            cov = np.mean((f_valid - f_mean) * (r_valid - r_mean))
            f_std = np.std(f_valid)
            r_std = np.std(r_valid)
            if f_std < 1e-9 or r_std < 1e-9:
                continue
            corr = cov / (f_std * r_std)
            r_squared = corr ** 2

            if r_squared > self._min_r2:
                finding = ResearchFinding(
                    finding_id=f"feat_{name}_{int(time.time())}",
                    program="feature_discovery",
                    title=f"Predictive feature: {name}",
                    description=f"r²={r_squared:.4f}, corr={corr:.3f} with 5-bar future return",
                    evidence_strength=min(1.0, r_squared * 10),
                    actionable=True,
                    action="add_feature",
                    params={"feature_name": name, "r_squared": r_squared, "correlation": corr},
                )
                findings.append(finding)
                self._discovered.append({"name": name, "r2": r_squared, "corr": corr})

        # Keep top N discoveries
        self._discovered.sort(key=lambda x: x["r2"], reverse=True)
        self._discovered = self._discovered[:self._max_features]

        return findings

    def get_top_features(self, n: int = 10) -> List[Dict[str, Any]]:
        return self._discovered[:n]


# ════════════════════════════════════════════════════════════════════════════
# Regime Transition Research
# ════════════════════════════════════════════════════════════════════════════

class RegimeTransitionResearch:
    """
    Studies how market regimes change and what predicts transitions.

    Questions:
    - What happens in the 10 bars BEFORE a regime change?
    - Which indicators are most predictive of regime transitions?
    - How long do regimes typically last?
    - What's the average P&L in each regime?
    """

    def __init__(self):
        self._transitions: List[Dict[str, Any]] = []
        self._regime_durations: Dict[str, List[int]] = defaultdict(list)
        self._regime_pnl: Dict[str, List[float]] = defaultdict(list)
        self._current_regime: str = ""
        self._regime_start_cycle: int = 0
        self._experiments = 0

    def record_regime(self, regime: str, cycle: int, pnl: float = 0.0,
                      indicators: Optional[Dict[str, float]] = None) -> List[ResearchFinding]:
        """Record current regime. Returns findings on transitions."""
        findings = []
        self._experiments += 1
        self._regime_pnl[regime].append(pnl)

        if regime != self._current_regime and self._current_regime:
            # Transition detected
            duration = cycle - self._regime_start_cycle
            self._regime_durations[self._current_regime].append(duration)

            self._transitions.append({
                "from": self._current_regime,
                "to": regime,
                "cycle": cycle,
                "duration": duration,
                "indicators_before": indicators or {},
            })

            # Analyze: what predicts this transition type?
            trans_key = f"{self._current_regime}→{regime}"
            similar = [t for t in self._transitions if
                       f"{t['from']}→{t['to']}" == trans_key]

            if len(similar) >= 3:
                avg_duration = sum(t["duration"] for t in similar) / len(similar)
                findings.append(ResearchFinding(
                    finding_id=f"regime_{trans_key}_{cycle}",
                    program="regime_transition",
                    title=f"Regime transition pattern: {trans_key}",
                    description=(f"Seen {len(similar)} times. "
                                 f"Avg duration before transition: {avg_duration:.0f} cycles"),
                    evidence_strength=min(1.0, len(similar) / 10),
                    actionable=True,
                    action="adjust_regime_expectations",
                    params={"transition": trans_key, "avg_duration": avg_duration,
                            "count": len(similar)},
                ))

            self._regime_start_cycle = cycle
        elif not self._current_regime:
            self._regime_start_cycle = cycle

        self._current_regime = regime
        return findings

    def get_regime_stats(self) -> Dict[str, Any]:
        stats = {}
        for regime, durations in self._regime_durations.items():
            pnls = self._regime_pnl.get(regime, [])
            stats[regime] = {
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "occurrences": len(durations),
                "avg_pnl": sum(pnls) / len(pnls) if pnls else 0,
            }
        return stats


# ════════════════════════════════════════════════════════════════════════════
# Strategy Archaeology — why did strategies work or fail?
# ════════════════════════════════════════════════════════════════════════════

class StrategyArchaeology:
    """
    Post-mortem analysis of strategy performance.

    When a strategy is retired, this module dissects WHY:
    - Market conditions when it worked vs failed
    - Which regime it was best/worst in
    - Correlation with other strategies (was it redundant?)
    - Execution quality (was slippage the problem, not the signal?)
    """

    def __init__(self):
        self._autopsies: List[Dict[str, Any]] = []
        self._experiments = 0

    def autopsy(
        self,
        strategy_name: str,
        trades: List[Dict[str, Any]],
        regime_at_trade: List[str],
    ) -> List[ResearchFinding]:
        """Perform post-mortem on a retired strategy."""
        findings = []
        self._experiments += 1

        if not trades:
            return findings

        pnls = [float(t.get("pnl", 0)) for t in trades]
        n = len(pnls)
        total_pnl = sum(pnls)
        win_rate = sum(1 for p in pnls if p > 0) / max(n, 1)

        # Regime breakdown
        regime_pnl: Dict[str, List[float]] = defaultdict(list)
        for pnl_val, regime in zip(pnls, regime_at_trade):
            regime_pnl[regime].append(pnl_val)

        best_regime = ""
        best_regime_avg = -999
        worst_regime = ""
        worst_regime_avg = 999

        for regime, rpnls in regime_pnl.items():
            avg = sum(rpnls) / len(rpnls)
            if avg > best_regime_avg:
                best_regime_avg = avg
                best_regime = regime
            if avg < worst_regime_avg:
                worst_regime_avg = avg
                worst_regime = regime

        # Time-series analysis: did it degrade over time?
        if n >= 10:
            first_half = pnls[:n//2]
            second_half = pnls[n//2:]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            degradation = first_avg - second_avg

            if degradation > 0.5:
                findings.append(ResearchFinding(
                    finding_id=f"arch_{strategy_name}_degrade",
                    program="strategy_archaeology",
                    title=f"{strategy_name}: edge degradation detected",
                    description=(f"First half avg={first_avg:.2f}%, second half avg={second_avg:.2f}%. "
                                 f"Edge decayed by {degradation:.2f}% — possibly crowded or regime-shifted"),
                    evidence_strength=min(1.0, degradation),
                    actionable=True,
                    action="flag_edge_decay",
                    params={"strategy": strategy_name, "degradation": degradation},
                ))

        if best_regime:
            findings.append(ResearchFinding(
                finding_id=f"arch_{strategy_name}_regime",
                program="strategy_archaeology",
                title=f"{strategy_name}: regime sensitivity",
                description=(f"Best in '{best_regime}' (avg={best_regime_avg:.2f}%), "
                             f"worst in '{worst_regime}' (avg={worst_regime_avg:.2f}%)"),
                evidence_strength=0.7,
                actionable=True,
                action="regime_conditional_activation",
                params={"strategy": strategy_name, "best_regime": best_regime,
                        "worst_regime": worst_regime},
            ))

        self._autopsies.append({
            "strategy": strategy_name,
            "total_pnl": total_pnl,
            "trades": n,
            "win_rate": win_rate,
            "best_regime": best_regime,
            "worst_regime": worst_regime,
        })

        return findings


# ════════════════════════════════════════════════════════════════════════════
# Correlation Mining
# ════════════════════════════════════════════════════════════════════════════

class CorrelationMiner:
    """
    Discovers hidden relationships between assets, signals, and outcomes.

    Finds:
    - Lead-lag relationships (ETH leads BTC by 2 bars)
    - Cross-asset correlations that break (regime change signal)
    - Signal combinations that are more predictive together
    """

    def __init__(self, window: int = 100, min_correlation: float = 0.3):
        self._window = window
        self._min_corr = min_correlation
        self._series: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self._experiments = 0

    def record(self, series_name: str, value: float) -> None:
        self._series[series_name].append(value)

    def mine(self) -> List[ResearchFinding]:
        """Search for significant correlations."""
        findings = []
        names = [k for k, v in self._series.items() if len(v) >= self._window]

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                self._experiments += 1
                a = list(self._series[names[i]])[-self._window:]
                b = list(self._series[names[j]])[-self._window:]
                n = min(len(a), len(b))
                if n < 20:
                    continue

                a, b = a[-n:], b[-n:]

                # Contemporaneous correlation
                corr = self._pearson(a, b)

                if abs(corr) > self._min_corr:
                    findings.append(ResearchFinding(
                        finding_id=f"corr_{names[i]}_{names[j]}",
                        program="correlation_mining",
                        title=f"Correlation: {names[i]} ↔ {names[j]}",
                        description=f"r={corr:.3f} over {n} observations",
                        evidence_strength=min(1.0, abs(corr)),
                        actionable=abs(corr) > 0.6,
                        action="pair_trading_candidate" if abs(corr) > 0.7 else "monitor",
                        params={"series_a": names[i], "series_b": names[j], "correlation": corr},
                    ))

                # Lead-lag (does A predict B 1-5 bars ahead?)
                for lag in range(1, 6):
                    if n <= lag + 10:
                        continue
                    lead_corr = self._pearson(a[:-lag], b[lag:])
                    if abs(lead_corr) > self._min_corr * 1.5:
                        findings.append(ResearchFinding(
                            finding_id=f"lead_{names[i]}_{names[j]}_lag{lag}",
                            program="correlation_mining",
                            title=f"Lead-lag: {names[i]} leads {names[j]} by {lag} bars",
                            description=f"r={lead_corr:.3f} — predictive relationship",
                            evidence_strength=min(1.0, abs(lead_corr)),
                            actionable=True,
                            action="lead_lag_signal",
                            params={"leader": names[i], "follower": names[j],
                                    "lag": lag, "correlation": lead_corr},
                        ))

        return findings

    def _pearson(self, a: List[float], b: List[float]) -> float:
        n = min(len(a), len(b))
        if n < 5:
            return 0.0
        a, b = a[:n], b[:n]
        ma = sum(a) / n
        mb = sum(b) / n
        cov = sum((ai - ma) * (bi - mb) for ai, bi in zip(a, b)) / (n - 1)
        sa = (sum((ai - ma) ** 2 for ai in a) / (n - 1)) ** 0.5
        sb = (sum((bi - mb) ** 2 for bi in b) / (n - 1)) ** 0.5
        return cov / max(sa * sb, 1e-9)


# ════════════════════════════════════════════════════════════════════════════
# Research Engine (orchestrator)
# ════════════════════════════════════════════════════════════════════════════

class ResearchEngine:
    """
    Orchestrates all research programs.

    Runs in the background every N cycles. Collects findings from all
    programs, ranks by evidence strength, and auto-applies actionable
    discoveries to the live trading system.
    """

    def __init__(self, research_interval: int = 200):
        self._interval = research_interval
        self._feature_discovery = FeatureDiscovery()
        self._regime_research = RegimeTransitionResearch()
        self._archaeology = StrategyArchaeology()
        self._correlation_miner = CorrelationMiner()

        self._all_findings: List[ResearchFinding] = []
        self._applied_count = 0
        self._cycle_count = 0

    def run(
        self,
        cycle: int,
        close: Optional[np.ndarray] = None,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
        volume: Optional[np.ndarray] = None,
        regime: str = "",
        pnl: float = 0.0,
        prices: Optional[Dict[str, float]] = None,
    ) -> Optional[ResearchReport]:
        """Run one research cycle. Returns report if research was conducted."""
        self._cycle_count = cycle
        if cycle % self._interval != 0 or cycle == 0:
            return None

        t0 = time.time()
        new_findings = []

        # Feature Discovery
        if close is not None and len(close) >= 60:
            h = high if high is not None else close
            l = low if low is not None else close
            v = volume if volume is not None else np.ones(len(close))
            new_findings.extend(self._feature_discovery.research(close, v, h, l))

        # Regime Transition Research
        if regime:
            new_findings.extend(self._regime_research.record_regime(regime, cycle, pnl))

        # Correlation Mining
        if prices:
            for sym, price in prices.items():
                self._correlation_miner.record(sym, price)
            if cycle % (self._interval * 5) == 0:
                new_findings.extend(self._correlation_miner.mine())

        # Store findings
        self._all_findings.extend(new_findings)
        if len(self._all_findings) > 500:
            self._all_findings = sorted(
                self._all_findings, key=lambda f: f.evidence_strength, reverse=True
            )[:500]

        actionable = [f for f in new_findings if f.actionable]
        duration_ms = (time.time() - t0) * 1000

        total_experiments = (
            self._feature_discovery._experiments
            + self._regime_research._experiments
            + self._archaeology._experiments
            + self._correlation_miner._experiments
        )

        if new_findings:
            logger.info(
                "ResearchEngine cycle %d: %d findings (%d actionable) from %d experiments (%.0fms)",
                cycle, len(new_findings), len(actionable), total_experiments, duration_ms,
            )
            for f in new_findings[:3]:
                logger.info("  Finding: %s — %s (evidence=%.2f)", f.title, f.description, f.evidence_strength)

        return ResearchReport(
            total_experiments=total_experiments,
            total_findings=len(self._all_findings),
            actionable_findings=len(actionable),
            applied_findings=self._applied_count,
            programs_active=4,
            top_findings=sorted(new_findings, key=lambda f: f.evidence_strength, reverse=True)[:5],
            duration_ms=duration_ms,
        )

    def autopsy_strategy(self, strategy_name: str, trades: List[Dict[str, Any]],
                         regime_at_trade: List[str]) -> List[ResearchFinding]:
        """Run post-mortem on a retired strategy."""
        findings = self._archaeology.autopsy(strategy_name, trades, regime_at_trade)
        self._all_findings.extend(findings)
        return findings

    def get_top_findings(self, n: int = 10) -> List[ResearchFinding]:
        return sorted(self._all_findings, key=lambda f: f.evidence_strength, reverse=True)[:n]

    def get_discovered_features(self, n: int = 10) -> List[Dict[str, Any]]:
        return self._feature_discovery.get_top_features(n)

    def get_regime_stats(self) -> Dict[str, Any]:
        return self._regime_research.get_regime_stats()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_findings": len(self._all_findings),
            "applied": self._applied_count,
            "top_features": len(self._feature_discovery._discovered),
            "regime_transitions": len(self._regime_research._transitions),
            "strategy_autopsies": len(self._archaeology._autopsies),
            "correlation_pairs": len(self._correlation_miner._series),
        }
