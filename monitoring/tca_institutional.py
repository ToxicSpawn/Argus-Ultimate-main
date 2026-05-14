"""Institutional Transaction Cost Analysis (TCA) with venue analytics.

Provides venue-level execution analysis, market-impact estimation, reporting,
and execution-quality scoring for institutional workflows.
"""

from __future__ import annotations

import csv
import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

_BPS_MULTIPLIER = 10_000.0
_DEFAULT_ADV = 1_000_000.0


class TCAMetric(str, Enum):
    """Supported transaction-cost metrics."""

    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    MARKET_IMPACT = "market_impact"
    TIMING_COST = "timing_cost"
    OPPORTUNITY_COST = "opportunity_cost"
    TOTAL_COST = "total_cost"


@dataclass
class FillQuality:
    """Execution-quality record for a single fill."""

    fill_id: str
    venue: str
    timestamp: datetime
    side: str
    quantity: float
    price: float
    benchmark_price: float
    arrival_price: float
    implementation_shortfall: float
    market_impact: float
    timing_cost: float

    def __post_init__(self) -> None:
        self.fill_id = str(self.fill_id).strip()
        self.venue = str(self.venue).strip()
        self.side = str(self.side).strip().lower()

        if not self.fill_id:
            raise ValueError("fill_id must be a non-empty string")
        if not self.venue:
            raise ValueError("venue must be a non-empty string")
        if self.side not in {"buy", "sell"}:
            raise ValueError(f"side must be 'buy' or 'sell', got {self.side!r}")

        self.quantity = _require_positive(self.quantity, "quantity")
        self.price = _require_positive(self.price, "price")
        self.benchmark_price = _require_positive(self.benchmark_price, "benchmark_price")
        self.arrival_price = _require_positive(self.arrival_price, "arrival_price")
        self.implementation_shortfall = float(self.implementation_shortfall)
        self.market_impact = float(self.market_impact)
        self.timing_cost = float(self.timing_cost)

    @property
    def total_cost(self) -> float:
        """Return total measured cost in basis points."""
        return self.implementation_shortfall + self.market_impact + self.timing_cost

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable dictionary for reporting/export."""
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        payload["total_cost"] = round(self.total_cost, 6)
        return payload


class VenueAnalyzer:
    """Collect and compare venue-level fill analytics."""

    def __init__(
        self,
        fills: Optional[Iterable[FillQuality]] = None,
        venue_order_stats: Optional[Mapping[str, Mapping[str, int]]] = None,
    ) -> None:
        self._fills_by_venue: DefaultDict[str, List[FillQuality]] = defaultdict(list)
        self._order_stats: DefaultDict[str, Dict[str, int]] = defaultdict(
            lambda: {"attempted": 0, "filled": 0, "rejected": 0}
        )
        self._seeded_stats_venues: set[str] = set()

        if venue_order_stats:
            for venue, stats in venue_order_stats.items():
                venue_name = str(venue)
                attempted = max(0, int(stats.get("attempted", 0)))
                filled = max(0, int(stats.get("filled", 0)))
                rejected = max(0, int(stats.get("rejected", 0)))
                self._order_stats[venue_name] = {
                    "attempted": max(attempted, filled + rejected),
                    "filled": filled,
                    "rejected": rejected,
                }
                self._seeded_stats_venues.add(venue_name)

        for fill in fills or []:
            self._store_fill(fill, update_stats=fill.venue not in self._seeded_stats_venues)

    def add_fill(self, fill: FillQuality) -> None:
        """Record a fill against its venue."""
        self._store_fill(fill, update_stats=True)

    def _store_fill(self, fill: FillQuality, update_stats: bool) -> None:
        """Store a fill and optionally update order statistics."""
        self._fills_by_venue[fill.venue].append(fill)
        if update_stats:
            stats = self._order_stats[fill.venue]
            stats["attempted"] += 1
            stats["filled"] += 1

    def get_venue_metrics(self, venue: str) -> Dict[str, Any]:
        """Return aggregated performance metrics for a venue."""
        venue_name = str(venue).strip()
        if not venue_name:
            raise ValueError("venue must be a non-empty string")

        fills = self._fills_by_venue.get(venue_name, [])
        stats = self._order_stats[venue_name]

        if not fills and stats["attempted"] == 0:
            raise KeyError(f"No data recorded for venue {venue_name!r}")

        implementation_shortfalls = [fill.implementation_shortfall for fill in fills]
        market_impacts = [fill.market_impact for fill in fills]
        timing_costs = [fill.timing_cost for fill in fills]
        total_costs = [fill.total_cost for fill in fills]
        quantities = [fill.quantity for fill in fills]
        notional = sum(fill.quantity * fill.price for fill in fills)
        side_mix = {
            "buy": sum(1 for fill in fills if fill.side == "buy"),
            "sell": sum(1 for fill in fills if fill.side == "sell"),
        }

        return {
            "venue": venue_name,
            "fill_count": len(fills),
            "total_quantity": round(sum(quantities), 6),
            "total_notional": round(notional, 6),
            "average_quantity": round(_mean(quantities), 6),
            "average_implementation_shortfall": round(_mean(implementation_shortfalls), 6),
            "average_market_impact": round(_mean(market_impacts), 6),
            "average_timing_cost": round(_mean(timing_costs), 6),
            "average_total_cost": round(_mean(total_costs), 6),
            "cost_volatility": round(_population_stddev(total_costs), 6),
            "fill_rate": round(self.compute_fill_rate(venue_name), 4),
            "reject_rate": round(self.compute_reject_rate(venue_name), 4),
            "side_mix": side_mix,
            "last_fill_timestamp": max((fill.timestamp for fill in fills), default=None),
        }

    def compare_venues(self) -> Dict[str, Any]:
        """Compare all venues and rank them by execution cost."""
        venues = sorted(set(self._fills_by_venue) | set(self._order_stats))
        if not venues:
            return {"venues": {}, "best_venue": None, "worst_venue": None, "ranked_venues": []}

        comparison: Dict[str, Dict[str, Any]] = {}
        ranked: List[Dict[str, Any]] = []
        for venue in venues:
            metrics = self.get_venue_metrics(venue)
            comparison[venue] = metrics
            ranked.append(metrics)

        ranked.sort(
            key=lambda item: (
                item["average_total_cost"],
                -item["fill_rate"],
                item["cost_volatility"],
            )
        )
        return {
            "venues": comparison,
            "best_venue": ranked[0]["venue"] if ranked else None,
            "worst_venue": ranked[-1]["venue"] if ranked else None,
            "ranked_venues": ranked,
        }

    def get_best_venue(self, side: str, size: float) -> Optional[str]:
        """Recommend the best venue for the given side and order size."""
        requested_side = str(side).strip().lower()
        if requested_side not in {"buy", "sell"}:
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        order_size = _require_positive(size, "size")

        best_venue: Optional[str] = None
        best_score: Optional[float] = None

        for venue in sorted(set(self._fills_by_venue) | set(self._order_stats)):
            fills = [fill for fill in self._fills_by_venue.get(venue, []) if fill.side == requested_side]
            if not fills:
                fills = self._fills_by_venue.get(venue, [])
            if not fills:
                continue

            metrics = self.get_venue_metrics(venue)
            size_alignment = _mean([abs(fill.quantity - order_size) / order_size for fill in fills])
            score = (
                metrics["average_total_cost"]
                + metrics["cost_volatility"] * 0.5
                + size_alignment * 10.0
                + metrics["reject_rate"] * 0.1
                - metrics["fill_rate"] * 0.02
            )
            if best_score is None or score < best_score:
                best_score = score
                best_venue = venue

        return best_venue

    def compute_fill_rate(self, venue: str) -> float:
        """Return fill rate percentage for a venue."""
        stats = self._order_stats[str(venue)]
        attempted = stats["attempted"]
        if attempted <= 0:
            return 0.0
        return stats["filled"] / attempted * 100.0

    def compute_reject_rate(self, venue: str) -> float:
        """Return reject rate percentage for a venue."""
        stats = self._order_stats[str(venue)]
        attempted = stats["attempted"]
        if attempted <= 0:
            return 0.0
        return stats["rejected"] / attempted * 100.0


class MarketImpactModel:
    """Fit and apply a lightweight Almgren-Chriss-style impact model."""

    def __init__(
        self,
        eta: float = 5.0,
        gamma: float = 12.0,
        default_adv: float = _DEFAULT_ADV,
    ) -> None:
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.default_adv = _require_positive(default_adv, "default_adv")
        self.is_fitted = False

    def fit_model(self, fills: Sequence[Any]) -> Dict[str, float]:
        """Fit an Almgren-Chriss-like model from historical fills."""
        observations: List[tuple[float, float]] = []

        for fill in fills:
            quantity = _safe_float(_get_field(fill, "quantity"), default=0.0)
            impact = _safe_float(_get_field(fill, "market_impact"), default=0.0)
            adv = _safe_float(_get_field(fill, "adv", self.default_adv), default=self.default_adv)
            if quantity <= 0 or adv <= 0:
                continue

            participation = max(1e-9, min(1.0, quantity / adv))
            observations.append((participation, impact))

        if len(observations) < 2:
            logger.warning("Insufficient fills to fit market-impact model; using defaults")
            self.is_fitted = False
            return {"eta": round(self.eta, 6), "gamma": round(self.gamma, 6), "observations": len(observations)}

        s_xx = s_xs = s_ss = s_xy = s_sy = 0.0
        for participation, impact in observations:
            linear = participation
            root = math.sqrt(participation)
            s_xx += linear * linear
            s_xs += linear * root
            s_ss += root * root
            s_xy += linear * impact
            s_sy += root * impact

        determinant = s_xx * s_ss - s_xs * s_xs
        if abs(determinant) < 1e-12:
            logger.warning("Degenerate impact observations; reverting to average fit")
            avg_participation = _mean([item[0] for item in observations])
            avg_impact = _mean([item[1] for item in observations])
            scale = math.sqrt(avg_participation) + avg_participation
            coefficient = avg_impact / scale if scale > 0 else self.gamma
            self.eta = max(0.0, coefficient * 0.5)
            self.gamma = max(0.0, coefficient * 0.5)
        else:
            self.eta = max(0.0, (s_xy * s_ss - s_sy * s_xs) / determinant)
            self.gamma = max(0.0, (s_sy * s_xx - s_xy * s_xs) / determinant)

        self.is_fitted = True
        return {"eta": round(self.eta, 6), "gamma": round(self.gamma, 6), "observations": len(observations)}

    def predict_impact(self, side: str, quantity: float, adv: float) -> float:
        """Predict market impact in basis points."""
        requested_side = str(side).strip().lower()
        if requested_side not in {"buy", "sell"}:
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

        order_quantity = _require_positive(quantity, "quantity")
        daily_volume = _require_positive(adv, "adv")
        participation = min(1.0, order_quantity / daily_volume)
        return round(self.eta * participation + self.gamma * math.sqrt(participation), 6)

    def compute_implementation_shortfall(
        self,
        order: Mapping[str, Any],
        fills: Sequence[FillQuality],
    ) -> Dict[str, float]:
        """Compute realized and opportunity-cost shortfall for an order."""
        side = str(order.get("side", "buy")).strip().lower()
        if side not in {"buy", "sell"}:
            raise ValueError(f"order side must be 'buy' or 'sell', got {side!r}")

        order_quantity = _require_positive(_safe_float(order.get("quantity"), 0.0), "order.quantity")
        arrival_price = _require_positive(_safe_float(order.get("arrival_price"), 0.0), "order.arrival_price")

        executed_quantity = sum(fill.quantity for fill in fills)
        remaining_quantity = max(0.0, order_quantity - executed_quantity)
        weighted_fill_price = (
            sum(fill.price * fill.quantity for fill in fills) / executed_quantity if executed_quantity > 0 else arrival_price
        )

        if side == "buy":
            realized_bps = ((weighted_fill_price - arrival_price) / arrival_price) * _BPS_MULTIPLIER
            fallback_price = max((fill.benchmark_price for fill in fills), default=arrival_price)
            opportunity_bps = ((fallback_price - arrival_price) / arrival_price) * _BPS_MULTIPLIER
        else:
            realized_bps = ((arrival_price - weighted_fill_price) / arrival_price) * _BPS_MULTIPLIER
            fallback_price = min((fill.benchmark_price for fill in fills), default=arrival_price)
            opportunity_bps = ((arrival_price - fallback_price) / arrival_price) * _BPS_MULTIPLIER

        fill_ratio = min(1.0, executed_quantity / order_quantity)
        realized_component = max(0.0, realized_bps) * fill_ratio
        opportunity_component = max(0.0, opportunity_bps) * (remaining_quantity / order_quantity)
        total_shortfall = realized_component + opportunity_component
        return {
            "implementation_shortfall_bps": round(total_shortfall, 6),
            "realized_shortfall_bps": round(max(0.0, realized_bps), 6),
            "opportunity_cost_bps": round(opportunity_component, 6),
            "executed_quantity": round(executed_quantity, 6),
            "remaining_quantity": round(remaining_quantity, 6),
            "fill_ratio": round(fill_ratio, 6),
        }


class TCADashboard:
    """Generate institutional TCA reports and compliance exports."""

    def __init__(
        self,
        fills: Optional[Iterable[FillQuality]] = None,
        venue_order_stats: Optional[Mapping[str, Mapping[str, int]]] = None,
        impact_model: Optional[MarketImpactModel] = None,
    ) -> None:
        self.fills: List[FillQuality] = []
        self.venue_analyzer = VenueAnalyzer(venue_order_stats=venue_order_stats)
        self.impact_model = impact_model or MarketImpactModel()

        for fill in fills or []:
            self.add_fill(fill)

    def add_fill(self, fill: FillQuality) -> None:
        """Add a fill to the dashboard dataset."""
        if not isinstance(fill, FillQuality):
            raise TypeError("fill must be an instance of FillQuality")
        self.fills.append(fill)
        self.venue_analyzer.add_fill(fill)

    def generate_daily_report(self) -> Dict[str, Any]:
        """Return a daily summary of TCA metrics."""
        if not self.fills:
            return {"daily_metrics": {}, "summary": {"fill_count": 0, "venues": []}}

        daily_metrics: Dict[str, Dict[str, float]] = {}
        grouped: DefaultDict[date, List[FillQuality]] = defaultdict(list)
        for fill in self.fills:
            grouped[fill.timestamp.date()].append(fill)

        for trading_day, fills in sorted(grouped.items()):
            daily_metrics[trading_day.isoformat()] = {
                TCAMetric.IMPLEMENTATION_SHORTFALL.value: round(_mean([f.implementation_shortfall for f in fills]), 6),
                TCAMetric.MARKET_IMPACT.value: round(_mean([f.market_impact for f in fills]), 6),
                TCAMetric.TIMING_COST.value: round(_mean([f.timing_cost for f in fills]), 6),
                TCAMetric.TOTAL_COST.value: round(_mean([f.total_cost for f in fills]), 6),
                "fill_count": float(len(fills)),
                "notional": round(sum(f.price * f.quantity for f in fills), 6),
            }

        return {
            "daily_metrics": daily_metrics,
            "summary": {
                "fill_count": len(self.fills),
                "venues": sorted({fill.venue for fill in self.fills}),
                "average_total_cost": round(_mean([fill.total_cost for fill in self.fills]), 6),
            },
        }

    def generate_venue_comparison(self) -> Dict[str, Any]:
        """Return a venue-by-venue execution comparison report."""
        return self.venue_analyzer.compare_venues()

    def generate_cost_attribution(self) -> Dict[str, Any]:
        """Break execution cost into metric-level and venue-level attribution."""
        if not self.fills:
            return {"totals": {}, "weights": {}, "by_venue": {}}

        totals = {
            TCAMetric.IMPLEMENTATION_SHORTFALL.value: sum(fill.implementation_shortfall for fill in self.fills),
            TCAMetric.MARKET_IMPACT.value: sum(fill.market_impact for fill in self.fills),
            TCAMetric.TIMING_COST.value: sum(fill.timing_cost for fill in self.fills),
            TCAMetric.OPPORTUNITY_COST.value: 0.0,
            TCAMetric.TOTAL_COST.value: sum(fill.total_cost for fill in self.fills),
        }
        total_cost = totals[TCAMetric.TOTAL_COST.value]
        weights = {
            key: round((value / total_cost) if total_cost else 0.0, 6)
            for key, value in totals.items()
            if key != TCAMetric.TOTAL_COST.value
        }

        by_venue = {
            venue: {
                TCAMetric.IMPLEMENTATION_SHORTFALL.value: round(sum(fill.implementation_shortfall for fill in fills), 6),
                TCAMetric.MARKET_IMPACT.value: round(sum(fill.market_impact for fill in fills), 6),
                TCAMetric.TIMING_COST.value: round(sum(fill.timing_cost for fill in fills), 6),
                TCAMetric.TOTAL_COST.value: round(sum(fill.total_cost for fill in fills), 6),
            }
            for venue, fills in sorted(self.venue_analyzer._fills_by_venue.items())
        }

        return {
            "totals": {key: round(value, 6) for key, value in totals.items()},
            "weights": weights,
            "by_venue": by_venue,
        }

    def export_to_csv(self, path: str) -> str:
        """Export fill-level TCA records to CSV for compliance workflows."""
        output_path = Path(path)
        if not output_path.parent.exists():
            raise FileNotFoundError(f"Parent directory does not exist for {path!r}")

        fieldnames = [
            "fill_id",
            "venue",
            "timestamp",
            "side",
            "quantity",
            "price",
            "benchmark_price",
            "arrival_price",
            "implementation_shortfall",
            "market_impact",
            "timing_cost",
            "total_cost",
        ]

        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for fill in sorted(self.fills, key=lambda item: item.timestamp):
                writer.writerow(fill.to_dict())

        logger.info("Exported %d TCA fills to %s", len(self.fills), output_path)
        return str(output_path)


class ExecutionQualityScorer:
    """Score fills and venues on a 0-100 institutional-quality scale."""

    def __init__(self, venue_analyzer: VenueAnalyzer) -> None:
        if not isinstance(venue_analyzer, VenueAnalyzer):
            raise TypeError("venue_analyzer must be an instance of VenueAnalyzer")
        self.venue_analyzer = venue_analyzer

    def score_fill(self, fill: FillQuality) -> float:
        """Score a single fill from 0 to 100, where higher is better."""
        if not isinstance(fill, FillQuality):
            raise TypeError("fill must be an instance of FillQuality")

        total_cost = max(0.0, fill.total_cost)
        market_deviation = abs(fill.price - fill.benchmark_price) / fill.benchmark_price * _BPS_MULTIPLIER
        cost_penalty = min(85.0, total_cost * 1.8)
        deviation_penalty = min(10.0, market_deviation * 0.2)
        size_penalty = min(5.0, max(0.0, fill.quantity - 100_000.0) / 100_000.0)
        score = 100.0 - cost_penalty - deviation_penalty - size_penalty
        return round(max(0.0, min(100.0, score)), 2)

    def score_venue(self, venue: str) -> float:
        """Score a venue using fill quality and reliability statistics."""
        metrics = self.venue_analyzer.get_venue_metrics(venue)
        fills = self.venue_analyzer._fills_by_venue.get(venue, [])

        fill_score = _mean([self.score_fill(fill) for fill in fills]) if fills else 50.0
        score = fill_score + metrics["fill_rate"] * 0.1 - metrics["reject_rate"] * 0.5
        return round(max(0.0, min(100.0, score)), 2)

    def get_improvement_suggestions(self) -> List[str]:
        """Return practical suggestions for improving execution quality."""
        suggestions: List[str] = []
        comparison = self.venue_analyzer.compare_venues()
        ranked_venues = comparison.get("ranked_venues", [])

        if not ranked_venues:
            return ["Collect more venue-level fill history before generating recommendations."]

        worst = ranked_venues[-1]
        best = ranked_venues[0]
        if worst["average_total_cost"] > best["average_total_cost"]:
            suggestions.append(
                f"Route more flow away from {worst['venue']} toward {best['venue']} to reduce average total cost."
            )

        for venue_metrics in ranked_venues:
            venue = venue_metrics["venue"]
            if venue_metrics["reject_rate"] > 5.0:
                suggestions.append(
                    f"Investigate reject handling for {venue}; reject rate is {venue_metrics['reject_rate']:.2f}%."
                )
            if venue_metrics["average_market_impact"] > venue_metrics["average_timing_cost"] * 1.5:
                suggestions.append(
                    f"Reduce participation rate on {venue}; market impact dominates timing cost."
                )
            if venue_metrics["cost_volatility"] > 5.0:
                suggestions.append(
                    f"Execution quality is inconsistent on {venue}; tighten venue-selection thresholds."
                )

        return suggestions or ["Execution quality is stable across venues; continue monitoring for drift."]


def _require_positive(value: float, field_name: str) -> float:
    numeric_value = float(value)
    if numeric_value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value!r}")
    return numeric_value


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _population_stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _get_field(item: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(field_name, default)
    return getattr(item, field_name, default)
