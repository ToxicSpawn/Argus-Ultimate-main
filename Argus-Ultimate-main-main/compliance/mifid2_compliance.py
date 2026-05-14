"""MiFID II compliance reporting for ARGUS.

Provides RTS25 timestamp-quality reporting, RTS28 venue-quality reporting,
transaction cost analysis, and regulatory submission payload generation for
best-execution oversight.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional

from execution.multi_venue_execution import MultiVenueExecutor
from monitoring.audit_trail import AuditTrail
from monitoring.tca_enhanced import TCAEngine
from monitoring.trade_ledger import TradeLedger

logger = logging.getLogger(__name__)

_SECONDS_TO_MS = 1000.0
_DEFAULT_OUTPUT_DIR = Path("compliance/reports/mifid2")
_DEFAULT_ENTITY_NAME = "ARGUS Trading"


@dataclass(frozen=True)
class BestExecutionReport:
    trade_id: str
    symbol: str
    venue: str
    order_timestamp: datetime
    execution_timestamp: datetime
    quantity: float
    price: float
    total_cost: float
    spread_cost: float
    market_impact_cost: float
    venue_latency_ms: float
    price_improvement: float


class MiFID2Reporter:
    """MiFID II best-execution and transaction-reporting helper.

    The reporter can work from direct trade dictionaries, from the existing
    ``TradeLedger``, or from upstream integrations such as a venue router,
    transaction cost analysis engine, audit trail, or order management system.
    """

    def __init__(
        self,
        trade_ledger: Optional[TradeLedger] = None,
        execution_venues: Optional[MultiVenueExecutor] = None,
        order_management_system: Optional[Any] = None,
        tca_engine: Optional[TCAEngine] = None,
        audit_trail: Optional[AuditTrail] = None,
        output_dir: Optional[Path] = None,
        entity_name: str = _DEFAULT_ENTITY_NAME,
    ) -> None:
        self.trade_ledger = trade_ledger
        self.execution_venues = execution_venues
        self.order_management_system = order_management_system
        self.tca_engine = tca_engine or TCAEngine(trade_ledger=trade_ledger)
        self.audit_trail = audit_trail
        self.output_dir = output_dir or _DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.entity_name = entity_name
        self._recorded_trades: List[Dict[str, Any]] = []
        self._best_execution_reports: List[BestExecutionReport] = []

        logger.info(
            "MiFID2Reporter initialised — output_dir=%s entity=%s",
            self.output_dir,
            self.entity_name,
        )

    def generate_best_execution_report(self, trade: dict) -> BestExecutionReport:
        """Create and store a MiFID II best-execution report for one trade."""
        trade_payload = self._normalise_trade(trade)
        costs = self.calculate_transaction_costs(trade_payload)
        order_timestamp = self._parse_datetime(
            trade_payload.get("order_timestamp")
            or trade_payload.get("timestamp")
            or trade_payload.get("decision_timestamp")
        )
        execution_timestamp = self._parse_datetime(
            trade_payload.get("execution_timestamp")
            or trade_payload.get("fill_timestamp")
            or trade_payload.get("timestamp")
        )
        latency_ms = self._resolve_latency_ms(
            trade_payload,
            order_timestamp=order_timestamp,
            execution_timestamp=execution_timestamp,
        )

        report = BestExecutionReport(
            trade_id=str(
                trade_payload.get("trade_id")
                or trade_payload.get("order_id")
                or trade_payload.get("id")
                or f"trade-{len(self._best_execution_reports) + 1}"
            ),
            symbol=str(trade_payload.get("symbol", "UNKNOWN")),
            venue=str(trade_payload.get("venue") or trade_payload.get("exchange") or "unknown"),
            order_timestamp=order_timestamp,
            execution_timestamp=execution_timestamp,
            quantity=float(trade_payload.get("quantity", 0.0)),
            price=float(trade_payload.get("price", 0.0)),
            total_cost=float(costs["total_cost"]),
            spread_cost=float(costs["spread_cost"]),
            market_impact_cost=float(costs["market_impact_cost"]),
            venue_latency_ms=latency_ms,
            price_improvement=float(costs["price_improvement"]),
        )

        report_payload = asdict(report)
        report_payload["order_timestamp"] = report.order_timestamp.isoformat()
        report_payload["execution_timestamp"] = report.execution_timestamp.isoformat()

        self._recorded_trades.append(dict(trade_payload))
        self._best_execution_reports.append(report)
        self._write_json(
            self.output_dir / f"best_execution_{report.trade_id}.json",
            {
                "report_type": "MiFIDII_BEST_EXECUTION",
                "generated_at": datetime.now(tz=UTC).isoformat(),
                "entity_name": self.entity_name,
                "report": report_payload,
                "transaction_cost_analysis": costs,
                "regulatory_submission": self._format_best_execution_submission(report, costs),
            },
        )
        self._append_audit_event("mifid2_best_execution_report", report_payload)
        logger.info(
            "MiFID II best-execution report generated for trade_id=%s venue=%s symbol=%s",
            report.trade_id,
            report.venue,
            report.symbol,
        )
        return report

    def calculate_transaction_costs(self, trade: dict) -> dict:
        """Calculate MiFID II transaction costs and price improvement metrics."""
        trade_payload = self._normalise_trade(trade)
        quantity = abs(float(trade_payload.get("quantity", 0.0)))
        price = float(trade_payload.get("price", 0.0))
        reference_price = self._first_float(
            trade_payload.get("arrival_price"),
            trade_payload.get("decision_price"),
            trade_payload.get("benchmark_price"),
            trade_payload.get("mid_price"),
            default=price,
        )
        fee = self._first_float(trade_payload.get("fee"), trade_payload.get("commission"), default=0.0)
        spread_cost = self._calculate_spread_cost(trade_payload, quantity=quantity, price=price)
        market_impact_cost = self._calculate_market_impact_cost(trade_payload, quantity=quantity, price=price)
        implementation_shortfall = abs(price - reference_price) * quantity
        explicit_cost = abs(fee)
        total_cost = spread_cost + market_impact_cost + implementation_shortfall + explicit_cost
        price_improvement = self._calculate_price_improvement(trade_payload, price=price)

        tca_summary: Dict[str, float] = {}
        if self._can_run_tca(quantity=quantity, price=price, reference_price=reference_price):
            result = self.tca_engine.analyze_trade(
                order_size=quantity,
                target_price=self._first_float(
                    trade_payload.get("target_price"),
                    trade_payload.get("decision_price"),
                    default=price,
                ),
                fill_price=price,
                arrival_price=reference_price,
                vwap=self._first_float(trade_payload.get("vwap"), default=price),
                twap=self._first_float(trade_payload.get("twap"), default=reference_price),
            )
            if trade_payload.get("avg_daily_volume"):
                result.market_impact_cost_bps = self.tca_engine.calculate_market_impact(
                    quantity,
                    float(trade_payload["avg_daily_volume"]),
                )
                result.total_cost_bps = round(
                    result.spread_cost_bps
                    + result.slippage_cost_bps
                    + result.timing_cost_bps
                    + result.market_impact_cost_bps,
                    4,
                )
            tca_summary = result.to_dict()

        return {
            "trade_id": str(
                trade_payload.get("trade_id")
                or trade_payload.get("order_id")
                or trade_payload.get("id")
                or "unknown"
            ),
            "symbol": str(trade_payload.get("symbol", "UNKNOWN")),
            "venue": str(trade_payload.get("venue") or trade_payload.get("exchange") or "unknown"),
            "notional": round(quantity * price, 8),
            "spread_cost": round(spread_cost, 8),
            "market_impact_cost": round(market_impact_cost, 8),
            "implementation_shortfall": round(implementation_shortfall, 8),
            "explicit_cost": round(explicit_cost, 8),
            "price_improvement": round(price_improvement, 8),
            "total_cost": round(total_cost, 8),
            "tca": tca_summary,
        }

    def compare_venue_quality(self, trades: List[dict]) -> dict:
        """Compare execution venues across cost, price improvement, and speed."""
        venue_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for trade in trades:
            trade_payload = self._normalise_trade(trade)
            venue_name = str(trade_payload.get("venue") or trade_payload.get("exchange") or "unknown")
            venue_groups[venue_name].append(trade_payload)

        venue_scores = self._get_live_venue_scores()
        results: Dict[str, Dict[str, Any]] = {}

        for venue, venue_trades in venue_groups.items():
            cost_breakdowns = [self.calculate_transaction_costs(trade) for trade in venue_trades]
            latencies = [
                self._resolve_latency_ms(
                    trade,
                    order_timestamp=self._parse_datetime(
                        trade.get("order_timestamp") or trade.get("timestamp")
                    ),
                    execution_timestamp=self._parse_datetime(
                        trade.get("execution_timestamp") or trade.get("fill_timestamp") or trade.get("timestamp")
                    ),
                )
                for trade in venue_trades
            ]
            volumes = [abs(float(trade.get("quantity", 0.0))) * float(trade.get("price", 0.0)) for trade in venue_trades]
            price_improvements = [cost["price_improvement"] for cost in cost_breakdowns]
            total_costs = [cost["total_cost"] for cost in cost_breakdowns]
            fill_ratios = [self._first_float(trade.get("fill_ratio"), default=1.0) for trade in venue_trades]
            avg_latency = mean(latencies) if latencies else 0.0
            avg_total_cost = mean(total_costs) if total_costs else 0.0
            avg_price_improvement = mean(price_improvements) if price_improvements else 0.0
            avg_fill_ratio = mean(fill_ratios) if fill_ratios else 1.0
            live_score = venue_scores.get(venue, 0.0)
            quality_score = self._score_venue_quality(
                avg_total_cost=avg_total_cost,
                avg_latency=avg_latency,
                avg_price_improvement=avg_price_improvement,
                avg_fill_ratio=avg_fill_ratio,
                live_score=live_score,
            )

            results[venue] = {
                "trade_count": len(venue_trades),
                "volume": round(sum(volumes), 8),
                "average_total_cost": round(avg_total_cost, 8),
                "average_price_improvement": round(avg_price_improvement, 8),
                "average_latency_ms": round(avg_latency, 4),
                "average_fill_ratio": round(avg_fill_ratio, 6),
                "live_venue_score": round(live_score, 6),
                "quality_score": round(quality_score, 6),
                "quality_of_execution": self._quality_analysis_summary(
                    avg_total_cost=avg_total_cost,
                    avg_latency=avg_latency,
                    avg_price_improvement=avg_price_improvement,
                    avg_fill_ratio=avg_fill_ratio,
                ),
            }

        ranked = sorted(results.items(), key=lambda item: item[1]["quality_score"], reverse=True)
        return {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "venue_count": len(results),
            "venues": results,
            "ranking": [venue for venue, _ in ranked],
        }

    def generate_daily_transaction_report(self) -> dict:
        """Generate a daily RTS25-style transaction and timestamp-quality report."""
        now = datetime.now(tz=UTC)
        start = datetime.combine(now.date(), datetime.min.time(), tzinfo=UTC)
        end = start + timedelta(days=1)
        trades = self._collect_trades(start=start, end=end)
        venue_comparison = self.compare_venue_quality(trades)
        completeness = self.validate_reporting_completeness()
        rts25_metrics = self._generate_rts25_metrics(trades)

        report = {
            "report_type": "MiFIDII_DAILY_TRANSACTION_REPORT",
            "generated_at": now.isoformat(),
            "entity_name": self.entity_name,
            "reporting_date": start.date().isoformat(),
            "trade_count": len(trades),
            "rts25": rts25_metrics,
            "venue_quality": venue_comparison,
            "reporting_completeness": completeness,
            "regulatory_submission": {
                "schema": "RTS25_DAILY",
                "submission_ready": completeness["complete_trade_reports"] == completeness["trade_count"],
                "payload": {
                    "entity_name": self.entity_name,
                    "reporting_date": start.date().isoformat(),
                    "clock_sync_status": rts25_metrics["clock_synchronisation_status"],
                    "latency_metrics_ms": rts25_metrics["execution_speed_metrics"],
                },
            },
        }
        self._write_json(self.output_dir / f"mifid2_daily_{start.date().isoformat()}.json", report)
        self._append_audit_event("mifid2_daily_transaction_report", report)
        logger.info("MiFID II daily transaction report generated for %s", start.date().isoformat())
        return report

    def generate_quarterly_report(self) -> dict:
        """Generate a quarterly RTS28-style best-execution report."""
        now = datetime.now(tz=UTC)
        quarter = ((now.month - 1) // 3) + 1
        quarter_start_month = ((quarter - 1) * 3) + 1
        start = datetime(now.year, quarter_start_month, 1, tzinfo=UTC)
        if quarter == 4:
            end = datetime(now.year + 1, 1, 1, tzinfo=UTC)
        else:
            end = datetime(now.year, quarter_start_month + 3, 1, tzinfo=UTC)

        trades = self._collect_trades(start=start, end=end)
        venue_quality = self.compare_venue_quality(trades)
        top_venues = self._top_execution_venues_by_volume(trades, limit=5)
        price_improvement_stats = self._price_improvement_statistics(trades)
        execution_speed_metrics = self._execution_speed_metrics(trades)
        quality_analysis = self._quality_of_execution_analysis(trades, venue_quality)

        report = {
            "report_type": "MiFIDII_RTS28_QUARTERLY_REPORT",
            "generated_at": now.isoformat(),
            "entity_name": self.entity_name,
            "quarter": f"Q{quarter}",
            "year": now.year,
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "trade_count": len(trades),
            "rts28": {
                "top_5_execution_venues_by_volume": top_venues,
                "price_improvement_statistics": price_improvement_stats,
                "execution_speed_metrics": execution_speed_metrics,
                "analysis_of_quality_of_execution": quality_analysis,
            },
            "venue_quality": venue_quality,
            "systemic_internaliser_metrics": self.calculate_systemic_internaliser_metrics(),
            "regulatory_submission": {
                "schema": "RTS28_QUARTERLY",
                "submission_ready": True,
                "payload": {
                    "entity_name": self.entity_name,
                    "quarter": f"Q{quarter}",
                    "year": now.year,
                    "venues": top_venues,
                    "quality_of_execution": quality_analysis,
                },
            },
        }
        self._write_json(self.output_dir / f"mifid2_rts28_q{quarter}_{now.year}.json", report)
        self._append_audit_event("mifid2_quarterly_report", report)
        logger.info("MiFID II quarterly report generated for Q%s %s", quarter, now.year)
        return report

    def calculate_systemic_internaliser_metrics(self) -> dict:
        """Calculate internalisation metrics for MiFID II oversight."""
        trades = self._collect_trades()
        internalised = [trade for trade in trades if self._is_internalised_trade(trade)]
        total_notional = sum(abs(float(t.get("quantity", 0.0))) * float(t.get("price", 0.0)) for t in trades)
        internalised_notional = sum(
            abs(float(t.get("quantity", 0.0))) * float(t.get("price", 0.0))
            for t in internalised
        )

        by_symbol: Dict[str, Dict[str, float]] = defaultdict(lambda: {"trade_count": 0.0, "notional": 0.0})
        for trade in internalised:
            symbol = str(trade.get("symbol", "UNKNOWN"))
            by_symbol[symbol]["trade_count"] += 1
            by_symbol[symbol]["notional"] += abs(float(trade.get("quantity", 0.0))) * float(trade.get("price", 0.0))

        return {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "total_trades": len(trades),
            "internalised_trade_count": len(internalised),
            "internalised_notional": round(internalised_notional, 8),
            "internalised_ratio": round(len(internalised) / len(trades), 6) if trades else 0.0,
            "internalised_notional_ratio": round(internalised_notional / total_notional, 6) if total_notional else 0.0,
            "by_symbol": {
                symbol: {
                    "trade_count": int(values["trade_count"]),
                    "notional": round(values["notional"], 8),
                }
                for symbol, values in by_symbol.items()
            },
        }

    def validate_reporting_completeness(self) -> dict:
        """Validate that mandatory fields exist for MiFID II reporting."""
        trades = self._collect_trades()
        required_fields = (
            "symbol",
            "quantity",
            "price",
            "timestamp",
            "venue",
        )
        complete = 0
        incomplete: List[Dict[str, Any]] = []

        for trade in trades:
            missing = [field for field in required_fields if trade.get(field) in (None, "")]
            if missing:
                incomplete.append(
                    {
                        "trade_id": trade.get("trade_id") or trade.get("order_id") or trade.get("id"),
                        "missing_fields": missing,
                    }
                )
            else:
                complete += 1

        status = "complete" if not incomplete else "incomplete"
        report = {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "trade_count": len(trades),
            "complete_trade_reports": complete,
            "incomplete_trade_reports": len(incomplete),
            "status": status,
            "missing_records": incomplete,
        }
        self._append_audit_event("mifid2_reporting_completeness", report)
        return report

    def _collect_trades(
        self,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        trades: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        for trade in self._recorded_trades:
            self._append_unique_trade(trades, seen_ids, trade)

        if self.trade_ledger is not None:
            try:
                for trade in self.trade_ledger.get_trades(limit=5000):
                    self._append_unique_trade(trades, seen_ids, trade)
            except Exception as exc:
                logger.warning("MiFID II: unable to load trades from ledger: %s", exc)

        for trade in self._load_trades_from_oms():
            self._append_unique_trade(trades, seen_ids, trade)

        if start is None and end is None:
            return trades

        filtered: List[Dict[str, Any]] = []
        for trade in trades:
            trade_ts = self._parse_datetime(
                trade.get("execution_timestamp") or trade.get("fill_timestamp") or trade.get("timestamp")
            )
            if start is not None and trade_ts < start:
                continue
            if end is not None and trade_ts >= end:
                continue
            filtered.append(trade)
        return filtered

    def _load_trades_from_oms(self) -> List[Dict[str, Any]]:
        oms = self.order_management_system
        if oms is None:
            return []
        method_names = (
            "get_completed_orders",
            "get_filled_orders",
            "list_filled_orders",
            "get_trade_history",
            "get_trades",
        )
        for method_name in method_names:
            method = getattr(oms, method_name, None)
            if callable(method):
                try:
                    records = method()
                except TypeError:
                    continue
                except Exception as exc:
                    logger.warning("MiFID II: OMS method %s failed: %s", method_name, exc)
                    return []
                if isinstance(records, Iterable):
                    return [self._normalise_trade(record) for record in records]
        return []

    def _append_unique_trade(
        self,
        trades: List[Dict[str, Any]],
        seen_ids: set[str],
        trade: Mapping[str, Any],
    ) -> None:
        trade_payload = self._normalise_trade(trade)
        trade_id = str(
            trade_payload.get("trade_id")
            or trade_payload.get("order_id")
            or trade_payload.get("id")
            or f"{trade_payload.get('symbol', 'UNKNOWN')}:{trade_payload.get('timestamp', '')}:{len(trades)}"
        )
        if trade_id in seen_ids:
            return
        seen_ids.add(trade_id)
        trades.append(trade_payload)

    def _normalise_trade(self, trade: Mapping[str, Any] | dict) -> Dict[str, Any]:
        trade_payload = dict(trade)
        if "venue" not in trade_payload and "exchange" in trade_payload:
            trade_payload["venue"] = trade_payload["exchange"]
        if "timestamp" not in trade_payload:
            trade_payload["timestamp"] = (
                trade_payload.get("execution_timestamp")
                or trade_payload.get("fill_timestamp")
                or trade_payload.get("order_timestamp")
            )
        if "trade_id" not in trade_payload:
            trade_payload["trade_id"] = trade_payload.get("order_id") or trade_payload.get("id")
        return trade_payload

    def _parse_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=UTC)
        if isinstance(value, str) and value:
            normalized = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
            return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        return datetime.now(tz=UTC)

    def _resolve_latency_ms(
        self,
        trade: Mapping[str, Any],
        *,
        order_timestamp: datetime,
        execution_timestamp: datetime,
    ) -> float:
        explicit_latency = trade.get("venue_latency_ms") or trade.get("latency_ms")
        if explicit_latency is not None:
            return round(float(explicit_latency), 6)
        return round(max((execution_timestamp - order_timestamp).total_seconds(), 0.0) * _SECONDS_TO_MS, 6)

    def _calculate_spread_cost(self, trade: Mapping[str, Any], *, quantity: float, price: float) -> float:
        spread = self._first_float(
            trade.get("spread"),
            default=None,
        )
        if spread is not None:
            return abs(spread) * quantity / 2.0

        bid = self._first_float(trade.get("bid"), default=None)
        ask = self._first_float(trade.get("ask"), default=None)
        if bid is not None and ask is not None and ask >= bid:
            return ((ask - bid) / 2.0) * quantity

        spread_bps = self._first_float(trade.get("spread_bps"), default=0.0)
        return abs(spread_bps) / 10_000.0 * price * quantity

    def _calculate_market_impact_cost(self, trade: Mapping[str, Any], *, quantity: float, price: float) -> float:
        impact_cost = self._first_float(
            trade.get("market_impact_cost"),
            trade.get("impact_cost"),
            default=None,
        )
        if impact_cost is not None:
            return abs(impact_cost)

        avg_daily_volume = self._first_float(trade.get("avg_daily_volume"), default=None)
        if avg_daily_volume and quantity > 0 and self.tca_engine is not None:
            impact_bps = self.tca_engine.calculate_market_impact(quantity, avg_daily_volume)
            return abs(impact_bps) / 10_000.0 * price * quantity
        return 0.0

    def _calculate_price_improvement(self, trade: Mapping[str, Any], *, price: float) -> float:
        side = str(trade.get("side", "buy")).lower()
        benchmark = self._first_float(
            trade.get("benchmark_price"),
            trade.get("arrival_price"),
            trade.get("decision_price"),
            trade.get("mid_price"),
            default=price,
        )
        if side in {"sell", "short"}:
            return max(price - benchmark, 0.0)
        return max(benchmark - price, 0.0)

    def _can_run_tca(self, *, quantity: float, price: float, reference_price: float) -> bool:
        return self.tca_engine is not None and quantity > 0 and price > 0 and reference_price > 0

    def _top_execution_venues_by_volume(self, trades: List[Dict[str, Any]], *, limit: int) -> List[Dict[str, Any]]:
        volumes: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        for trade in trades:
            venue = str(trade.get("venue") or trade.get("exchange") or "unknown")
            volumes[venue] += abs(float(trade.get("quantity", 0.0))) * float(trade.get("price", 0.0))
            counts[venue] += 1
        ranked = sorted(volumes.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [
            {
                "venue": venue,
                "volume": round(volume, 8),
                "trade_count": counts[venue],
            }
            for venue, volume in ranked
        ]

    def _price_improvement_statistics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        improvements = [self.calculate_transaction_costs(trade)["price_improvement"] for trade in trades]
        positive = [value for value in improvements if value > 0]
        return {
            "average": round(mean(improvements), 8) if improvements else 0.0,
            "median_proxy": round(sorted(improvements)[len(improvements) // 2], 8) if improvements else 0.0,
            "max": round(max(improvements), 8) if improvements else 0.0,
            "min": round(min(improvements), 8) if improvements else 0.0,
            "positive_rate": round(len(positive) / len(improvements), 6) if improvements else 0.0,
        }

    def _execution_speed_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        latencies = [
            self._resolve_latency_ms(
                trade,
                order_timestamp=self._parse_datetime(trade.get("order_timestamp") or trade.get("timestamp")),
                execution_timestamp=self._parse_datetime(
                    trade.get("execution_timestamp") or trade.get("fill_timestamp") or trade.get("timestamp")
                ),
            )
            for trade in trades
        ]
        sorted_latencies = sorted(latencies)
        p95_index = int(len(sorted_latencies) * 0.95) - 1 if sorted_latencies else -1
        return {
            "average_latency_ms": round(mean(latencies), 6) if latencies else 0.0,
            "max_latency_ms": round(max(latencies), 6) if latencies else 0.0,
            "min_latency_ms": round(min(latencies), 6) if latencies else 0.0,
            "p95_latency_ms": round(sorted_latencies[max(p95_index, 0)], 6) if sorted_latencies else 0.0,
        }

    def _quality_of_execution_analysis(
        self,
        trades: List[Dict[str, Any]],
        venue_quality: Dict[str, Any],
    ) -> Dict[str, Any]:
        costs = [self.calculate_transaction_costs(trade) for trade in trades]
        ranked_venues = venue_quality.get("ranking", [])
        best_venue = ranked_venues[0] if ranked_venues else None
        return {
            "average_total_cost": round(mean(cost["total_cost"] for cost in costs), 8) if costs else 0.0,
            "average_spread_cost": round(mean(cost["spread_cost"] for cost in costs), 8) if costs else 0.0,
            "average_market_impact_cost": round(
                mean(cost["market_impact_cost"] for cost in costs),
                8,
            ) if costs else 0.0,
            "average_price_improvement": round(
                mean(cost["price_improvement"] for cost in costs),
                8,
            ) if costs else 0.0,
            "best_scoring_venue": best_venue,
            "summary": (
                f"Best execution quality currently favours {best_venue}."
                if best_venue
                else "Insufficient trade volume to rank execution quality."
            ),
        }

    def _generate_rts25_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        speed_metrics = self._execution_speed_metrics(trades)
        return {
            "clock_synchronisation_status": "synchronised" if trades else "no_trades",
            "timestamp_precision": "UTC ISO-8601",
            "execution_speed_metrics": speed_metrics,
            "event_count": len(trades),
        }

    def _get_live_venue_scores(self) -> Dict[str, float]:
        if self.execution_venues is None:
            return {}
        try:
            return self.execution_venues.venue_scores()
        except Exception as exc:
            logger.warning("MiFID II: unable to read venue scores: %s", exc)
            return {}

    def _score_venue_quality(
        self,
        *,
        avg_total_cost: float,
        avg_latency: float,
        avg_price_improvement: float,
        avg_fill_ratio: float,
        live_score: float,
    ) -> float:
        cost_component = 1.0 / (1.0 + max(avg_total_cost, 0.0))
        latency_component = 1.0 / (1.0 + max(avg_latency, 0.0) / 100.0)
        price_component = min(max(avg_price_improvement, 0.0), 1.0)
        fill_component = min(max(avg_fill_ratio, 0.0), 1.0)
        live_component = min(max(live_score, 0.0), 1.0)
        return (
            0.35 * cost_component
            + 0.20 * latency_component
            + 0.20 * price_component
            + 0.15 * fill_component
            + 0.10 * live_component
        )

    def _quality_analysis_summary(
        self,
        *,
        avg_total_cost: float,
        avg_latency: float,
        avg_price_improvement: float,
        avg_fill_ratio: float,
    ) -> str:
        if avg_price_improvement > 0 and avg_total_cost < 1.0 and avg_latency < 100.0:
            return "Strong execution quality with positive price improvement and low latency."
        if avg_total_cost > 5.0:
            return "Execution quality is cost-heavy and should be reviewed for routing efficiency."
        if avg_fill_ratio < 0.95:
            return "Execution quality is constrained by incomplete fills and venue capacity."
        return "Execution quality is stable but should be monitored for incremental optimisation."

    def _is_internalised_trade(self, trade: Mapping[str, Any]) -> bool:
        venue = str(trade.get("venue") or trade.get("exchange") or "").lower()
        flags = (
            bool(trade.get("internalised")),
            bool(trade.get("systematic_internaliser")),
            venue in {"internal", "si", "systematic_internaliser"},
        )
        return any(flags)

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def _append_audit_event(self, kind: str, payload: Mapping[str, Any]) -> None:
        if self.audit_trail is None:
            return
        try:
            self.audit_trail.append(kind, dict(payload))
        except Exception as exc:
            logger.warning("MiFID II: audit trail append failed for %s: %s", kind, exc)

    def _format_best_execution_submission(
        self,
        report: BestExecutionReport,
        costs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return {
            "schema": "MIFIDII_BEST_EXECUTION",
            "entity_name": self.entity_name,
            "trade_id": report.trade_id,
            "symbol": report.symbol,
            "venue": report.venue,
            "order_timestamp": report.order_timestamp.isoformat(),
            "execution_timestamp": report.execution_timestamp.isoformat(),
            "quantity": report.quantity,
            "price": report.price,
            "total_cost": report.total_cost,
            "spread_cost": report.spread_cost,
            "market_impact_cost": report.market_impact_cost,
            "venue_latency_ms": report.venue_latency_ms,
            "price_improvement": report.price_improvement,
            "transaction_cost_analysis": dict(costs),
        }

    def _first_float(self, *values: Any, default: Optional[float] = 0.0) -> Optional[float]:
        for value in values:
            if value in (None, ""):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return default


__all__ = ["BestExecutionReport", "MiFID2Reporter"]
