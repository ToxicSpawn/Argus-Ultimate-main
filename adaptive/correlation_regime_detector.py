"""Real-time correlation regime detection and analysis.

Features:
- Multi-asset correlation matrix computation
- Correlation regime classification (low, normal, high, crisis, decoupling)
- Pairwise correlation tracking and change detection
- Sector correlation analysis and rotation detection
- Diversification benefit quantification
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CorrelationRegime(str, Enum):
    LOW_CORRELATION = "low_correlation"
    NORMAL_CORRELATION = "normal_correlation"
    HIGH_CORRELATION = "high_correlation"
    CRISIS_CORRELATION = "crisis_correlation"
    DECOUPLING = "decoupling"


@dataclass(frozen=True)
class CorrelationMatrix:
    assets: List[str]
    matrix: np.ndarray
    timestamp: datetime
    avg_correlation: float
    max_correlation: float
    min_correlation: float


@dataclass(frozen=True)
class CorrelationRegimeSnapshot:
    regime: CorrelationRegime
    correlation_matrix: CorrelationMatrix
    regime_score: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RotationSignal:
    sector_from: str
    sector_to: str
    strength: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class CorrelationTracker:
    """Tracks rolling correlations between multiple assets."""

    def __init__(
        self,
        assets: List[str],
        window: int = 60,
        min_periods: int = 20,
    ) -> None:
        self._assets = list(assets)
        self._n_assets = len(assets)
        self._window = window
        self._min_periods = min_periods
        self._returns_buffer: Dict[str, deque] = {
            a: deque(maxlen=window) for a in assets
        }
        self._last_matrix: Optional[CorrelationMatrix] = None
        self._previous_matrix: Optional[CorrelationMatrix] = None

    def update(self, returns: Dict[str, float]) -> CorrelationMatrix:
        for asset, ret in returns.items():
            if asset in self._returns_buffer:
                self._returns_buffer[asset].append(ret)

        self._previous_matrix = self._last_matrix
        self._last_matrix = self._compute_correlation()
        return self._last_matrix

    def get_rolling_correlation(self, window: Optional[int] = None) -> CorrelationMatrix:
        w = window or self._window
        if w > self._window:
            logger.warning(
                "Requested window %d exceeds buffer size %d", w, self._window
            )
            w = self._window

        current_window = self._window
        self._window = w
        try:
            matrix = self._compute_correlation()
        finally:
            self._window = current_window
        return matrix

    def get_pairwise_correlation(self, asset1: str, asset2: str) -> float:
        if asset1 not in self._returns_buffer or asset2 not in self._returns_buffer:
            logger.warning("Asset not found: %s or %s", asset1, asset2)
            return 0.0

        r1 = list(self._returns_buffer[asset1])
        r2 = list(self._returns_buffer[asset2])

        min_len = min(len(r1), len(r2))
        if min_len < self._min_periods:
            return 0.0

        r1 = r1[-min_len:]
        r2 = r2[-min_len:]

        corr = np.corrcoef(r1, r2)[0, 1]
        return float(corr) if np.isfinite(corr) else 0.0

    def get_correlation_changes(self) -> Dict[Tuple[str, str], float]:
        if self._previous_matrix is None or self._last_matrix is None:
            return {}

        changes: Dict[Tuple[str, str], float] = {}
        prev_mat = self._previous_matrix.matrix
        curr_mat = self._last_matrix.matrix
        assets = self._assets

        for i in range(self._n_assets):
            for j in range(i + 1, self._n_assets):
                key = (assets[i], assets[j])
                changes[key] = float(curr_mat[i, j] - prev_mat[i, j])

        return changes

    def _compute_correlation(self) -> CorrelationMatrix:
        valid_assets = []
        returns_matrix = []

        for asset in self._assets:
            data = list(self._returns_buffer[asset])
            if len(data) >= self._min_periods:
                valid_assets.append(asset)
                returns_matrix.append(data)

        if len(valid_assets) < 2:
            return CorrelationMatrix(
                assets=self._assets,
                matrix=np.eye(self._n_assets),
                timestamp=datetime.utcnow(),
                avg_correlation=0.0,
                max_correlation=0.0,
                min_correlation=0.0,
            )

        returns_array = np.array(returns_matrix)
        corr_matrix = np.corrcoef(returns_array)

        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

        off_diag = corr_matrix[~np.eye(len(valid_assets), dtype=bool)]
        avg_corr = float(np.mean(off_diag))
        max_corr = float(np.max(off_diag))
        min_corr = float(np.min(off_diag))

        full_matrix = np.eye(self._n_assets)
        asset_idx = {a: i for i, a in enumerate(self._assets)}

        for i, a1 in enumerate(valid_assets):
            for j, a2 in enumerate(valid_assets):
                if a1 in asset_idx and a2 in asset_idx:
                    full_matrix[asset_idx[a1], asset_idx[a2]] = corr_matrix[i, j]

        return CorrelationMatrix(
            assets=self._assets,
            matrix=full_matrix,
            timestamp=datetime.utcnow(),
            avg_correlation=avg_corr,
            max_correlation=max_corr,
            min_correlation=min_corr,
        )


class RegimeClassifier:
    """Classifies correlation regimes based on correlation matrix statistics."""

    def __init__(
        self,
        low_threshold: float = 0.2,
        normal_high_threshold: float = 0.5,
        high_threshold: float = 0.7,
        crisis_threshold: float = 0.85,
        decoupling_threshold: float = -0.1,
        lookback: int = 50,
    ) -> None:
        self._low_threshold = low_threshold
        self._normal_high_threshold = normal_high_threshold
        self._high_threshold = high_threshold
        self._crisis_threshold = crisis_threshold
        self._decoupling_threshold = decoupling_threshold
        self._lookback = lookback
        self._regime_history: deque = deque(maxlen=lookback)
        self._transition_counts: Dict[Tuple[CorrelationRegime, CorrelationRegime], int] = {}

    def classify_regime(self, corr_matrix: CorrelationMatrix) -> CorrelationRegime:
        avg_corr = corr_matrix.avg_correlation
        max_corr = corr_matrix.max_correlation
        min_corr = corr_matrix.min_correlation

        regime = CorrelationRegime.NORMAL_CORRELATION

        if avg_corr >= self._crisis_threshold:
            regime = CorrelationRegime.CRISIS_CORRELATION
        elif avg_corr >= self._high_threshold:
            regime = CorrelationRegime.HIGH_CORRELATION
        elif avg_corr >= self._normal_high_threshold:
            regime = CorrelationRegime.NORMAL_CORRELATION
        elif avg_corr >= self._low_threshold:
            regime = CorrelationRegime.NORMAL_CORRELATION
        elif min_corr < self._decoupling_threshold and max_corr > self._high_threshold:
            regime = CorrelationRegime.DECOUPLING
        else:
            regime = CorrelationRegime.LOW_CORRELATION

        return regime

    def compute_regime_score(self, corr_matrix: CorrelationMatrix) -> float:
        avg_corr = corr_matrix.avg_correlation
        max_corr = corr_matrix.max_correlation
        min_corr = corr_matrix.min_correlation
        corr_spread = max_corr - min_corr

        score = avg_corr * 0.5
        score += max_corr * 0.3
        score -= corr_spread * 0.2

        return float(np.clip(score, -1.0, 1.0))

    def detect_regime_change(
        self, current: CorrelationRegime, previous: CorrelationRegime
    ) -> bool:
        if current == previous:
            return False

        key = (previous, current)
        self._transition_counts[key] = self._transition_counts.get(key, 0) + 1

        return True

    def get_regime_transition_probability(
        self,
    ) -> Dict[CorrelationRegime, float]:
        if not self._regime_history:
            return {r: 0.2 for r in CorrelationRegime}

        counts: Dict[CorrelationRegime, int] = {r: 0 for r in CorrelationRegime}
        for regime in self._regime_history:
            counts[regime] = counts.get(regime, 0) + 1

        total = sum(counts.values())
        if total == 0:
            return {r: 0.2 for r in CorrelationRegime}

        return {r: c / total for r, c in counts.items()}

    def record_regime(self, regime: CorrelationRegime) -> None:
        self._regime_history.append(regime)


class SectorCorrelationAnalyzer:
    """Analyzes correlations between sectors and detects sector rotation."""

    def __init__(
        self,
        window: int = 60,
        min_periods: int = 20,
        rotation_threshold: float = 0.15,
    ) -> None:
        self._window = window
        self._min_periods = min_periods
        self._rotation_threshold = rotation_threshold
        self._sector_returns: Dict[str, deque] = {}
        self._sector_correlations: Dict[str, float] = {}
        self._previous_sector_correlations: Dict[str, float] = {}
        self._rotation_history: deque = deque(maxlen=100)

    def compute_sector_correlations(
        self, returns: Dict[str, List[float]], sectors: Dict[str, List[str]]
    ) -> Dict[str, float]:
        sector_avg_returns: Dict[str, List[float]] = {}

        for sector, assets in sectors.items():
            if sector not in self._sector_returns:
                self._sector_returns[sector] = deque(maxlen=self._window)

            valid_returns = []
            for asset in assets:
                if asset in returns and len(returns[asset]) > 0:
                    valid_returns.append(returns[asset])

            if valid_returns:
                avg_ret = np.mean(valid_returns, axis=0).tolist()
                sector_avg_returns[sector] = avg_ret
                for r in avg_ret:
                    self._sector_returns[sector].append(r)

        self._previous_sector_correlations = self._sector_correlations.copy()
        self._sector_correlations = {}

        for sector, avg_returns in sector_avg_returns.items():
            if len(avg_returns) >= self._min_periods:
                vol = float(np.std(avg_returns))
                self._sector_correlations[sector] = vol if vol > 0 else 0.0

        return self._sector_correlations

    def detect_sector_rotation(self) -> List[RotationSignal]:
        if not self._previous_sector_correlations or not self._sector_correlations:
            return []

        signals: List[RotationSignal] = []
        timestamp = datetime.utcnow()

        sectors = sorted(self._sector_correlations.keys())

        for i, s1 in enumerate(sectors):
            for s2 in sectors[i + 1 :]:
                corr_change = self._sector_correlations.get(s1, 0) - self._sector_correlations.get(s2, 0)

                if abs(corr_change) >= self._rotation_threshold:
                    from_sector = s2 if corr_change > 0 else s1
                    to_sector = s1 if corr_change > 0 else s2
                    strength = abs(corr_change)

                    signal = RotationSignal(
                        sector_from=from_sector,
                        sector_to=to_sector,
                        strength=strength,
                        timestamp=timestamp,
                        metadata={"correlation_change": corr_change},
                    )
                    signals.append(signal)
                    self._rotation_history.append(signal)

        return signals

    def get_inter_sector_correlation(self, sector1: str, sector2: str) -> float:
        if (
            sector1 not in self._sector_returns
            or sector2 not in self._sector_returns
        ):
            return 0.0

        r1 = list(self._sector_returns[sector1])
        r2 = list(self._sector_returns[sector2])

        min_len = min(len(r1), len(r2))
        if min_len < self._min_periods:
            return 0.0

        r1 = r1[-min_len:]
        r2 = r2[-min_len:]

        corr = np.corrcoef(r1, r2)[0, 1]
        return float(corr) if np.isfinite(corr) else 0.0


class CorrelationRegimeDetector:
    """Main detector for correlation regime detection and analysis."""

    def __init__(
        self,
        assets: List[str],
        window: int = 60,
        min_periods: int = 20,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config or {}
        self._assets = list(assets)
        self._window = window
        self._min_periods = min_periods

        self._tracker = CorrelationTracker(
            assets=assets, window=window, min_periods=min_periods
        )
        self._classifier = RegimeClassifier(
            low_threshold=self.config.get("low_threshold", 0.2),
            normal_high_threshold=self.config.get("normal_high_threshold", 0.5),
            high_threshold=self.config.get("high_threshold", 0.7),
            crisis_threshold=self.config.get("crisis_threshold", 0.85),
            decoupling_threshold=self.config.get("decoupling_threshold", -0.1),
            lookback=self.config.get("lookback", 50),
        )
        self._sector_analyzer = SectorCorrelationAnalyzer(
            window=self.config.get("sector_window", 60),
            min_periods=self.config.get("sector_min_periods", 20),
            rotation_threshold=self.config.get("rotation_threshold", 0.15),
        )

        self._current_regime: Optional[CorrelationRegime] = None
        self._regime_history: deque = deque(maxlen=1000)
        self._snapshot_history: deque = deque(maxlen=1000)

    def update(self, returns: Dict[str, float]) -> CorrelationRegime:
        corr_matrix = self._tracker.update(returns)
        regime = self._classifier.classify_regime(corr_matrix)
        score = self._classifier.compute_regime_score(corr_matrix)

        self._classifier.record_regime(regime)

        if self._current_regime is not None:
            changed = self._classifier.detect_regime_change(regime, self._current_regime)
            if changed:
                logger.info(
                    "Correlation regime change: %s -> %s",
                    self._current_regime.value,
                    regime.value,
                )

        self._current_regime = regime

        snapshot = CorrelationRegimeSnapshot(
            regime=regime,
            correlation_matrix=corr_matrix,
            regime_score=score,
            timestamp=corr_matrix.timestamp,
            metadata={
                "avg_correlation": corr_matrix.avg_correlation,
                "max_correlation": corr_matrix.max_correlation,
                "min_correlation": corr_matrix.min_correlation,
            },
        )
        self._regime_history.append(regime)
        self._snapshot_history.append(snapshot)

        return regime

    def get_current_regime(self) -> CorrelationRegime:
        if self._current_regime is None:
            return CorrelationRegime.NORMAL_CORRELATION
        return self._current_regime

    def get_regime_history(self) -> List[CorrelationRegimeSnapshot]:
        return list(self._snapshot_history)

    def get_diversification_benefit(self) -> float:
        if self._current_regime is None:
            return 0.0

        regime = self._current_regime

        benefit_map = {
            CorrelationRegime.LOW_CORRELATION: 0.8,
            CorrelationRegime.NORMAL_CORRELATION: 0.5,
            CorrelationRegime.HIGH_CORRELATION: 0.2,
            CorrelationRegime.CRISIS_CORRELATION: 0.0,
            CorrelationRegime.DECOUPLING: 0.6,
        }

        return benefit_map.get(regime, 0.5)

    def get_correlation_tracker(self) -> CorrelationTracker:
        return self._tracker

    def get_regime_classifier(self) -> RegimeClassifier:
        return self._classifier

    def get_sector_analyzer(self) -> SectorCorrelationAnalyzer:
        return self._sector_analyzer
