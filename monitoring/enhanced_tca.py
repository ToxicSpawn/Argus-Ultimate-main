"""Enhanced Transaction Cost Analysis (TCA) module for Argus.

Provides granular cost decomposition, market-impact modelling, venue
comparison, and reporting dashboards for institutional-grade execution
analytics.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_BPS = 10_000.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TCAComponents:
    """Individual cost components for a single order."""
    implementation_shortfall: float = 0.0
    market_impact: float = 0.0
    timing_cost: float = 0.0
    spread_cost: float = 0.0
    commission: float = 0.0
    opportunity_cost: float = 0.0


@dataclass
class TCABreakdown:
    """Full TCA breakdown for a single order."""
    total_cost: float = 0.0
    cost_bps: float = 0.0
    components: TCAComponents = field(default_factory=TCAComponents)
    component_weights: Dict[str, float] = field(default_factory=dict)

    def compute_weights(self) -> Dict[str, float]:
        """Compute relative weight of each cost component."""
        comp = asdict(self.components)
        total = self.total_cost if self.total_cost != 0 else 1.0
        weights = {k: abs(v) / abs(total) for k, v in comp.items()}
        self.component_weights = {k: round(v, 4) for k, v in weights.items()}
        return self.component_weights


@dataclass
class TimingResult:
    """Result of execution timing analysis."""
    avg_timing_cost_bps: float = 0.0
    best_execution_window: str = ""
    worst_execution_window: str = ""
    timing_efficiency_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VenueComparison:
    """Comparison of execution quality across venues."""
    venues: List[str] = field(default_factory=list)
    avg_cost_bps: Dict[str, float] = field(default_factory=dict)
    avg_fill_rate: Dict[str, float] = field(default_factory=dict)
    avg_latency_ms: Dict[str, float] = field(default_factory=dict)
    best_venue: str = ""
    worst_venue: str = ""
    score_ranking: Dict[str, float] = field(default_factory=dict)


@dataclass
class FillQuality:
    """Fill quality metrics for a set of trades."""
    fill_rate: float = 0.0
    avg_slippage_bps: float = 0.0
    avg_fill_time_ms: float = 0.0
    price_improvement_bps: float = 0.0
    partial_fills: int = 0
    total_orders: int = 0


@dataclass
class TCAReport:
    """Aggregated TCA report over a period."""
    period: Tuple[datetime, datetime] = (datetime.min, datetime.min)
    total_orders: int = 0
    total_volume: float = 0.0
    avg_cost_bps: float = 0.0
    cost_by_venue: Dict[str, float] = field(default_factory=dict)
    cost_by_strategy: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class TCATrend:
    """Trend of TCA metrics over a rolling window."""
    window_days: int = 30
    avg_cost_bps: List[float] = field(default_factory=list)
    dates: List[datetime] = field(default_factory=list)
    trend_direction: str = "flat"
    trend_slope: float = 0.0


@dataclass
class PlotData:
    """Data suitable for plotting cost breakdowns."""
    labels: List[str] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    title: str = ""
    chart_type: str = "bar"


# ---------------------------------------------------------------------------
# ArrivalPriceAnalyzer
# ---------------------------------------------------------------------------


class ArrivalPriceAnalyzer:
    """Analyse execution performance against arrival-price benchmarks."""

    @staticmethod
    def compute_arrival_price(order: Dict[str, Any]) -> float:
        """Compute the arrival price for an order.

        Uses the first available price from the order metadata:
        ``arrival_price``, ``decision_price``, or ``intended_price``.
        """
        for key in ("arrival_price", "decision_price", "intended_price", "price"):
            val = order.get(key)
            if val is not None:
                return float(val)
        logger.warning("No arrival price found in order %s", order.get("order_id", "unknown"))
        return 0.0

    @staticmethod
    def compute_is(order: Dict[str, Any], arrival_price: float) -> float:
        """Compute implementation shortfall in price units."""
        fill_price = float(order.get("fill_price", 0.0))
        side = order.get("side", "buy").lower()
        if arrival_price == 0:
            return 0.0
        if side == "buy":
            return fill_price - arrival_price
        return arrival_price - fill_price

    @staticmethod
    def compute_is_bps(order: Dict[str, Any], arrival_price: float) -> float:
        """Compute implementation shortfall in basis points."""
        if arrival_price == 0:
            return 0.0
        is_val = ArrivalPriceAnalyzer.compute_is(order, arrival_price)
        return abs(is_val) / arrival_price * _BPS

    @staticmethod
    def benchmark_vs_vwap(order: Dict[str, Any], vwap: float) -> float:
        """Return slippage vs VWAP in basis points."""
        fill_price = float(order.get("fill_price", 0.0))
        if vwap == 0:
            return 0.0
        side = order.get("side", "buy").lower()
        if side == "buy":
            return (fill_price - vwap) / vwap * _BPS
        return (vwap - fill_price) / fill_price * _BPS

    @staticmethod
    def benchmark_vs_twap(order: Dict[str, Any], twap: float) -> float:
        """Return slippage vs TWAP in basis points."""
        fill_price = float(order.get("fill_price", 0.0))
        if twap == 0:
            return 0.0
        side = order.get("side", "buy").lower()
        if side == "buy":
            return (fill_price - twap) / twap * _BPS
        return (twap - fill_price) / fill_price * _BPS


# ---------------------------------------------------------------------------
# MarketImpactModel
# ---------------------------------------------------------------------------


class MarketImpactModel:
    """Market impact estimation models."""

    @staticmethod
    def square_root_model(order_size: float, adv: float, volatility: float) -> float:
        """Square-root market impact model.

        impact = volatility * sqrt(order_size / adv)
        """
        if adv <= 0 or order_size <= 0:
            return 0.0
        participation = order_size / adv
        return volatility * np.sqrt(participation)

    @staticmethod
    def almgren_chriss_model(order_size: float, adv: float, params: Dict[str, float]) -> float:
        """Almgren-Chriss market impact model.

        Permanent impact: eta * (order_size / adv)
        Temporary impact: gamma * sqrt(order_size / adv)

        params should contain:
            eta (float): permanent impact coefficient
            gamma (float): temporary impact coefficient
        """
        if adv <= 0 or order_size <= 0:
            return 0.0
        participation = order_size / adv
        eta = params.get("eta", 1.0)
        gamma = params.get("gamma", 0.5)
        permanent = eta * participation
        temporary = gamma * np.sqrt(participation)
        return permanent + temporary

    @staticmethod
    def temporary_impact(order_size: float, duration: float) -> float:
        """Estimate temporary market impact.

        Shorter duration with larger orders creates more temporary impact.
        """
        if duration <= 0 or order_size <= 0:
            return 0.0
        return 0.1 * order_size / duration

    @staticmethod
    def permanent_impact(order_size: float, volume_participation: float) -> float:
        """Estimate permanent market impact.

        Scales linearly with order size and participation rate.
        """
        if order_size <= 0 or volume_participation <= 0:
            return 0.0
        return 0.05 * order_size * volume_participation


# ---------------------------------------------------------------------------
# TimingAnalyzer
# ---------------------------------------------------------------------------


class TimingAnalyzer:
    """Analyse execution timing and identify optimal execution windows."""

    @staticmethod
    def compute_timing_cost(arrival_price: float, decision_price: float) -> float:
        """Compute timing cost between decision and arrival."""
        if decision_price == 0:
            return 0.0
        return abs(arrival_price - decision_price) / decision_price * _BPS

    @staticmethod
    def analyze_execution_timing(
        trades: List[Dict[str, Any]],
        market_data: Dict[str, Any],
    ) -> TimingResult:
        """Analyse execution timing across a set of trades."""
        if not trades:
            return TimingResult()

        timing_costs = []
        hourly_costs: Dict[int, List[float]] = {}

        for trade in trades:
            fill_price = float(trade.get("fill_price", 0.0))
            decision_price = float(trade.get("decision_price", fill_price))
            timestamp = trade.get("timestamp")

            tc = TimingAnalyzer.compute_timing_cost(fill_price, decision_price)
            timing_costs.append(tc)

            if timestamp is not None:
                if isinstance(timestamp, datetime):
                    hour = timestamp.hour
                elif isinstance(timestamp, (int, float)):
                    hour = datetime.fromtimestamp(timestamp).hour
                else:
                    hour = 0
                hourly_costs.setdefault(hour, []).append(tc)

        avg_tc = float(np.mean(timing_costs)) if timing_costs else 0.0

        best_hour = min(hourly_costs, key=lambda h: np.mean(hourly_costs[h])) if hourly_costs else -1
        worst_hour = max(hourly_costs, key=lambda h: np.mean(hourly_costs[h])) if hourly_costs else -1

        efficiency = 0.0
        if avg_tc > 0:
            efficiency = max(0.0, min(100.0, 100.0 - avg_tc))

        return TimingResult(
            avg_timing_cost_bps=round(avg_tc, 4),
            best_execution_window=f"{best_hour:02d}:00-{best_hour+1:02d}:00" if best_hour >= 0 else "N/A",
            worst_execution_window=f"{worst_hour:02d}:00-{worst_hour+1:02d}:00" if worst_hour >= 0 else "N/A",
            timing_efficiency_score=round(efficiency, 2),
            details={"hourly_costs": {str(k): round(float(np.mean(v)), 4) for k, v in hourly_costs.items()}},
        )

    @staticmethod
    def optimal_execution_time(
        order: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> datetime:
        """Estimate the optimal execution time for an order.

        Uses volatility and volume profiles from market_data to find
        the window with lowest expected impact.
        """
        volatility_profile = market_data.get("volatility_by_hour", {})
        volume_profile = market_data.get("volume_by_hour", {})

        if not volatility_profile and not volume_profile:
            return datetime.now() + timedelta(hours=1)

        scores: Dict[int, float] = {}
        for hour in range(24):
            vol = volatility_profile.get(hour, 1.0)
            vol_norm = float(vol)
            vol_score = vol_norm

            vol_val = volume_profile.get(hour, 0.0)
            vol_score = vol_score / (float(vol_val) + 1e-9)
            scores[hour] = vol_score

        best_hour = min(scores, key=scores.get) if scores else 10
        now = datetime.now()
        return now.replace(hour=best_hour, minute=0, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# VenueAnalyzer
# ---------------------------------------------------------------------------


class VenueAnalyzer:
    """Compare execution quality across trading venues."""

    @staticmethod
    def compare_venues(venue_trades: Dict[str, List[Dict[str, Any]]]) -> VenueComparison:
        """Compare multiple venues based on their trade history."""
        if not venue_trades:
            return VenueComparison()

        avg_cost: Dict[str, float] = {}
        avg_fill_rate: Dict[str, float] = {}
        avg_latency: Dict[str, float] = {}

        for venue, trades in venue_trades.items():
            if not trades:
                continue

            costs = []
            fill_times = []
            filled = 0

            for t in trades:
                fp = float(t.get("fill_price", 0.0))
                dp = float(t.get("decision_price", fp))
                if dp > 0:
                    costs.append(abs(fp - dp) / dp * _BPS)
                if t.get("status", "filled") == "filled":
                    filled += 1
                ft = float(t.get("fill_time_ms", 0.0))
                fill_times.append(ft)

            avg_cost[venue] = round(float(np.mean(costs)), 4) if costs else 0.0
            avg_fill_rate[venue] = round(filled / len(trades) * 100, 2) if trades else 0.0
            avg_latency[venue] = round(float(np.mean(fill_times)), 2) if fill_times else 0.0

        best = min(avg_cost, key=avg_cost.get) if avg_cost else ""
        worst = max(avg_cost, key=avg_cost.get) if avg_cost else ""

        ranking = {v: VenueAnalyzer.compute_venue_score_from_metrics(
            avg_cost.get(v, 0.0),
            avg_fill_rate.get(v, 0.0),
            avg_latency.get(v, 0.0),
        ) for v in avg_cost}

        return VenueComparison(
            venues=list(venue_trades.keys()),
            avg_cost_bps=avg_cost,
            avg_fill_rate=avg_fill_rate,
            avg_latency_ms=avg_latency,
            best_venue=best,
            worst_venue=worst,
            score_ranking=ranking,
        )

    @staticmethod
    def compute_venue_score(venue: Dict[str, Any]) -> float:
        """Compute a composite score for a single venue."""
        cost_bps = float(venue.get("avg_cost_bps", 0.0))
        fill_rate = float(venue.get("avg_fill_rate", 0.0))
        latency_ms = float(venue.get("avg_latency_ms", 0.0))
        return VenueAnalyzer.compute_venue_score_from_metrics(cost_bps, fill_rate, latency_ms)

    @staticmethod
    def compute_venue_score_from_metrics(
        cost_bps: float,
        fill_rate: float,
        latency_ms: float,
    ) -> float:
        """Composite venue score: 0-100."""
        cost_score = max(0.0, 50.0 - cost_bps * 2.0)
        fill_score = fill_rate * 0.3
        latency_score = max(0.0, 20.0 - latency_ms / 50.0)
        return round(min(100.0, cost_score + fill_score + latency_score), 2)

    @staticmethod
    def analyze_fill_quality(trades: List[Dict[str, Any]]) -> FillQuality:
        """Analyse fill quality for a set of trades."""
        if not trades:
            return FillQuality()

        total = len(trades)
        filled = sum(1 for t in trades if t.get("status", "filled") == "filled")
        partials = sum(1 for t in trades if t.get("status") == "partial")

        slippages = []
        fill_times = []
        improvements = []

        for t in trades:
            fp = float(t.get("fill_price", 0.0))
            dp = float(t.get("decision_price", fp))
            lp = float(t.get("limit_price", 0.0))
            if dp > 0:
                slippages.append(abs(fp - dp) / dp * _BPS)
            if lp > 0 and fp > 0:
                improvements.append(abs(lp - fp) / lp * _BPS)
            ft = float(t.get("fill_time_ms", 0.0))
            fill_times.append(ft)

        return FillQuality(
            fill_rate=round(filled / total * 100, 2) if total > 0 else 0.0,
            avg_slippage_bps=round(float(np.mean(slippages)), 4) if slippages else 0.0,
            avg_fill_time_ms=round(float(np.mean(fill_times)), 2) if fill_times else 0.0,
            price_improvement_bps=round(float(np.mean(improvements)), 4) if improvements else 0.0,
            partial_fills=partials,
            total_orders=total,
        )


# ---------------------------------------------------------------------------
# TCADashboard
# ---------------------------------------------------------------------------


class TCADashboard:
    """Generate TCA reports, trends, and visualisation data."""

    def __init__(self) -> None:
        self.arrival_analyzer = ArrivalPriceAnalyzer()
        self.impact_model = MarketImpactModel()
        self.timing_analyzer = TimingAnalyzer()
        self.venue_analyzer = VenueAnalyzer()
        logger.info("TCADashboard initialised")

    def generate_report(self, orders: List[Dict[str, Any]]) -> TCAReport:
        """Generate a comprehensive TCA report from a list of orders."""
        if not orders:
            return TCAReport()

        timestamps = []
        for o in orders:
            ts = o.get("timestamp")
            if isinstance(ts, datetime):
                timestamps.append(ts)
            elif isinstance(ts, (int, float)):
                timestamps.append(datetime.fromtimestamp(ts))

        if timestamps:
            period_start = min(timestamps)
            period_end = max(timestamps)
        else:
            period_start = datetime.now() - timedelta(days=30)
            period_end = datetime.now()

        total_volume = 0.0
        cost_values = []
        cost_by_venue: Dict[str, List[float]] = {}
        cost_by_strategy: Dict[str, List[float]] = {}

        for order in orders:
            fill_price = float(order.get("fill_price", 0.0))
            quantity = float(order.get("quantity", 0.0))
            total_volume += fill_price * quantity

            arrival = self.arrival_analyzer.compute_arrival_price(order)
            is_bps = self.arrival_analyzer.compute_is_bps(order, arrival)
            cost_values.append(is_bps)

            venue = order.get("venue", "unknown")
            cost_by_venue.setdefault(venue, []).append(is_bps)

            strategy = order.get("strategy", "unknown")
            cost_by_strategy.setdefault(strategy, []).append(is_bps)

        avg_cost = float(np.mean(cost_values)) if cost_values else 0.0

        avg_by_venue = {
            v: round(float(np.mean(costs)), 4) for v, costs in cost_by_venue.items()
        }
        avg_by_strategy = {
            s: round(float(np.mean(costs)), 4) for s, costs in cost_by_strategy.items()
        }

        recommendations = self._generate_recommendations(
            avg_cost, avg_by_venue, avg_by_strategy
        )

        return TCAReport(
            period=(period_start, period_end),
            total_orders=len(orders),
            total_volume=round(total_volume, 2),
            avg_cost_bps=round(avg_cost, 4),
            cost_by_venue=avg_by_venue,
            cost_by_strategy=avg_by_strategy,
            recommendations=recommendations,
        )

    def plot_cost_breakdown(self, orders: List[Dict[str, Any]]) -> PlotData:
        """Generate data for a cost breakdown plot."""
        if not orders:
            return PlotData(title="TCA Cost Breakdown")

        components = TCAComponents()
        for order in orders:
            arrival = self.arrival_analyzer.compute_arrival_price(order)
            fp = float(order.get("fill_price", 0.0))
            dp = float(order.get("decision_price", fp))
            bid = float(order.get("bid", fp))
            ask = float(order.get("ask", fp))
            commission = float(order.get("commission", 0.0))

            is_val = abs(self.arrival_analyzer.compute_is(order, arrival))
            components.implementation_shortfall += is_val
            components.timing_cost += self.timing_analyzer.compute_timing_cost(fp, dp)
            components.spread_cost += abs(ask - bid) / ((ask + bid) / 2) * _BPS if (ask + bid) > 0 else 0.0
            components.commission += commission

        comp_dict = asdict(components)
        labels = list(comp_dict.keys())
        values = [round(v, 4) for v in comp_dict.values()]

        return PlotData(
            labels=labels,
            values=values,
            title="TCA Cost Breakdown",
            chart_type="bar",
        )

    def compute_trend(self, orders: List[Dict[str, Any]], window: int = 30) -> TCATrend:
        """Compute rolling TCA trend over a window of days."""
        if not orders:
            return TCATrend(window_days=window)

        daily_costs: Dict[str, List[float]] = {}
        for order in orders:
            ts = order.get("timestamp")
            if isinstance(ts, datetime):
                date_key = ts.strftime("%Y-%m-%d")
            elif isinstance(ts, (int, float)):
                date_key = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            else:
                continue

            arrival = self.arrival_analyzer.compute_arrival_price(order)
            is_bps = self.arrival_analyzer.compute_is_bps(order, arrival)
            daily_costs.setdefault(date_key, []).append(is_bps)

        sorted_dates = sorted(daily_costs.keys())[-window:]
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in sorted_dates]
        avg_costs = [
            round(float(np.mean(daily_costs[d])), 4) for d in sorted_dates
        ]

        trend_direction = "flat"
        trend_slope = 0.0
        if len(avg_costs) >= 2:
            x = np.arange(len(avg_costs))
            y = np.array(avg_costs)
            coeffs = np.polyfit(x, y, 1)
            trend_slope = round(float(coeffs[0]), 6)
            if trend_slope > 0.01:
                trend_direction = "increasing"
            elif trend_slope < -0.01:
                trend_direction = "decreasing"

        return TCATrend(
            window_days=window,
            avg_cost_bps=avg_costs,
            dates=dates,
            trend_direction=trend_direction,
            trend_slope=trend_slope,
        )

    def export_report(self, orders: List[Dict[str, Any]], format: str = "json") -> str:
        """Export TCA report in the specified format."""
        report = self.generate_report(orders)
        data = {
            "period": {
                "start": report.period[0].isoformat(),
                "end": report.period[1].isoformat(),
            },
            "total_orders": report.total_orders,
            "total_volume": report.total_volume,
            "avg_cost_bps": report.avg_cost_bps,
            "cost_by_venue": report.cost_by_venue,
            "cost_by_strategy": report.cost_by_strategy,
            "recommendations": report.recommendations,
        }

        if format == "json":
            return json.dumps(data, indent=2, default=str)

        if format == "csv":
            lines = ["metric,value"]
            lines.append(f"period_start,{report.period[0].isoformat()}")
            lines.append(f"period_end,{report.period[1].isoformat()}")
            lines.append(f"total_orders,{report.total_orders}")
            lines.append(f"total_volume,{report.total_volume}")
            lines.append(f"avg_cost_bps,{report.avg_cost_bps}")
            for venue, cost in report.cost_by_venue.items():
                lines.append(f"venue_{venue}_cost_bps,{cost}")
            for strategy, cost in report.cost_by_strategy.items():
                lines.append(f"strategy_{strategy}_cost_bps,{cost}")
            return "\n".join(lines)

        logger.warning("Unsupported export format: %s, falling back to JSON", format)
        return json.dumps(data, indent=2, default=str)

    @staticmethod
    def _generate_recommendations(
        avg_cost: float,
        cost_by_venue: Dict[str, float],
        cost_by_strategy: Dict[str, float],
    ) -> List[str]:
        """Generate actionable recommendations based on TCA analysis."""
        recs = []

        if avg_cost > 10.0:
            recs.append("Average execution cost exceeds 10 bps — consider using limit orders or TWAP/VWAP algorithms")

        if cost_by_venue:
            best = min(cost_by_venue, key=cost_by_venue.get)
            worst = max(cost_by_venue, key=cost_by_venue.get)
            if cost_by_venue[worst] - cost_by_venue[best] > 5.0:
                recs.append(
                    f"Venue '{worst}' costs {cost_by_venue[worst] - cost_by_venue[best]:.2f} bps more than '{best}' — route orders to {best}"
                )

        if cost_by_strategy:
            highest = max(cost_by_strategy, key=cost_by_strategy.get)
            if cost_by_strategy[highest] > 15.0:
                recs.append(
                    f"Strategy '{highest}' has high execution costs ({cost_by_strategy[highest]:.2f} bps) — review execution parameters"
                )

        if not recs:
            recs.append("Execution costs are within acceptable range")

        return recs
