"""Enhanced Transaction Cost Analysis (TCA) engine for Argus.

Provides richer execution-quality analytics than ``monitoring.tca`` and can
analyse both direct trade inputs and rows loaded from the existing SQLite trade
ledger.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional

from monitoring.trade_ledger import TradeLedger

logger = logging.getLogger(__name__)

_BPS_MULTIPLIER = 10_000.0
_DEFAULT_ADV = 1_000_000.0
_DEFAULT_IMPACT_COEFFICIENT = 15.0


@dataclass
class TCAResult:
    spread_cost_bps: float
    slippage_cost_bps: float
    timing_cost_bps: float
    market_impact_cost_bps: float
    total_cost_bps: float
    implementation_shortfall: float

    def to_dict(self) -> Dict[str, float]:
        """Return a serialisable representation of the TCA result."""
        return {key: round(float(value), 6) for key, value in asdict(self).items()}


class TCAEngine:
    """Enhanced transaction cost analysis engine.

    Parameters
    ----------
    trade_ledger:
        Optional existing trade ledger instance used for ledger-based analysis.
    default_avg_daily_volume:
        Fallback ADV used by the square-root market-impact model.
    impact_coefficient_bps:
        Square-root impact coefficient in basis points.
    """

    def __init__(
        self,
        trade_ledger: Optional[TradeLedger] = None,
        default_avg_daily_volume: float = _DEFAULT_ADV,
        impact_coefficient_bps: float = _DEFAULT_IMPACT_COEFFICIENT,
    ) -> None:
        self.trade_ledger = trade_ledger
        self.default_avg_daily_volume = self._require_positive(
            default_avg_daily_volume,
            "default_avg_daily_volume",
        )
        self.impact_coefficient_bps = self._require_positive(
            impact_coefficient_bps,
            "impact_coefficient_bps",
        )
        logger.info(
            "TCAEngine initialised — default_adv=%.2f, impact_coefficient_bps=%.2f",
            self.default_avg_daily_volume,
            self.impact_coefficient_bps,
        )

    def analyze_trade(
        self,
        order_size: float,
        target_price: float,
        fill_price: float,
        arrival_price: float,
        vwap: float,
        twap: float,
    ) -> TCAResult:
        """Analyse a single trade and return a full TCA cost breakdown."""
        validated_order_size = self._require_positive(order_size, "order_size")
        validated_target = self._require_positive(target_price, "target_price")
        validated_fill = self._require_positive(fill_price, "fill_price")
        validated_arrival = self._require_positive(arrival_price, "arrival_price")
        validated_vwap = self._require_positive(vwap, "vwap")
        validated_twap = self._require_positive(twap, "twap")

        spread_cost_bps = self._absolute_bps(validated_fill, validated_target)
        slippage_cost_bps = self.calculate_vwap_slippage(validated_fill, validated_vwap)
        timing_cost_bps = self._absolute_bps(validated_arrival, validated_twap)
        market_impact_cost_bps = self.calculate_market_impact(
            validated_order_size,
            self.default_avg_daily_volume,
        )
        total_cost_bps = (
            spread_cost_bps
            + slippage_cost_bps
            + timing_cost_bps
            + market_impact_cost_bps
        )
        implementation_shortfall = self.calculate_implementation_shortfall(
            validated_fill,
            validated_arrival,
        )

        result = TCAResult(
            spread_cost_bps=round(spread_cost_bps, 4),
            slippage_cost_bps=round(slippage_cost_bps, 4),
            timing_cost_bps=round(timing_cost_bps, 4),
            market_impact_cost_bps=round(market_impact_cost_bps, 4),
            total_cost_bps=round(total_cost_bps, 4),
            implementation_shortfall=round(implementation_shortfall, 8),
        )
        logger.debug("TCA analysed trade — order_size=%.4f result=%s", order_size, result)
        return result

    def calculate_implementation_shortfall(
        self,
        fill_price: float,
        arrival_price: float,
    ) -> float:
        """Return absolute implementation shortfall in price units."""
        validated_fill = self._require_positive(fill_price, "fill_price")
        validated_arrival = self._require_positive(arrival_price, "arrival_price")
        return abs(validated_fill - validated_arrival)

    def calculate_vwap_slippage(self, fill_price: float, vwap: float) -> float:
        """Return absolute slippage versus VWAP in basis points."""
        validated_fill = self._require_positive(fill_price, "fill_price")
        validated_vwap = self._require_positive(vwap, "vwap")
        return self._absolute_bps(validated_fill, validated_vwap)

    def calculate_market_impact(self, order_size: float, avg_daily_volume: float) -> float:
        """Estimate market impact in basis points using a square-root model."""
        validated_order_size = self._require_positive(order_size, "order_size")
        validated_adv = self._require_positive(avg_daily_volume, "avg_daily_volume")
        participation = min(1.0, validated_order_size / validated_adv)
        impact_bps = self.impact_coefficient_bps * math.sqrt(participation)
        return round(impact_bps, 4)

    def generate_tca_report(self, trades: List[TCAResult]) -> dict:
        """Aggregate a list of TCA results into a report dictionary."""
        if not trades:
            return {
                "trade_count": 0,
                "average_costs_bps": {},
                "total_cost_bps": 0.0,
                "average_implementation_shortfall": 0.0,
                "max_total_cost_bps": 0.0,
                "min_total_cost_bps": 0.0,
            }

        spread_costs = [trade.spread_cost_bps for trade in trades]
        slippage_costs = [trade.slippage_cost_bps for trade in trades]
        timing_costs = [trade.timing_cost_bps for trade in trades]
        impact_costs = [trade.market_impact_cost_bps for trade in trades]
        total_costs = [trade.total_cost_bps for trade in trades]
        shortfalls = [trade.implementation_shortfall for trade in trades]

        return {
            "trade_count": len(trades),
            "average_costs_bps": {
                "spread": round(mean(spread_costs), 4),
                "slippage": round(mean(slippage_costs), 4),
                "timing": round(mean(timing_costs), 4),
                "market_impact": round(mean(impact_costs), 4),
                "total": round(mean(total_costs), 4),
            },
            "total_cost_bps": round(sum(total_costs), 4),
            "average_implementation_shortfall": round(mean(shortfalls), 8),
            "max_total_cost_bps": round(max(total_costs), 4),
            "min_total_cost_bps": round(min(total_costs), 4),
        }

    def track_benchmark_comparison(
        self,
        fill_price: float,
        benchmarks: dict,
    ) -> dict:
        """Compare a fill price against benchmark prices in basis points."""
        validated_fill = self._require_positive(fill_price, "fill_price")
        comparisons: Dict[str, float] = {}

        for benchmark_name, benchmark_value in benchmarks.items():
            try:
                validated_benchmark = self._require_positive(
                    float(benchmark_value),
                    f"benchmark[{benchmark_name}]",
                )
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Skipping invalid benchmark %s=%r: %s",
                    benchmark_name,
                    benchmark_value,
                    exc,
                )
                continue
            comparisons[str(benchmark_name)] = round(
                self._absolute_bps(validated_fill, validated_benchmark),
                4,
            )

        return comparisons

    def analyze_ledger_trade(
        self,
        trade: Mapping[str, Any],
        *,
        target_price: Optional[float] = None,
        arrival_price: Optional[float] = None,
        vwap: Optional[float] = None,
        twap: Optional[float] = None,
        avg_daily_volume: Optional[float] = None,
    ) -> TCAResult:
        """Analyse a trade row loaded from ``monitoring.trade_ledger``.

        The existing ledger does not store TCA-specific benchmarks, so callers may
        pass them explicitly. Missing values fall back to the ledger price.
        """
        try:
            order_size = float(trade.get("quantity", 0.0))
            fill_price = float(trade.get("price", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid ledger trade payload: {trade!r}") from exc

        resolved_target = float(target_price if target_price is not None else fill_price)
        resolved_arrival = float(arrival_price if arrival_price is not None else fill_price)
        resolved_vwap = float(vwap if vwap is not None else fill_price)
        resolved_twap = float(twap if twap is not None else fill_price)
        result = self.analyze_trade(
            order_size=order_size,
            target_price=resolved_target,
            fill_price=fill_price,
            arrival_price=resolved_arrival,
            vwap=resolved_vwap,
            twap=resolved_twap,
        )

        if avg_daily_volume is not None:
            result.market_impact_cost_bps = self.calculate_market_impact(order_size, avg_daily_volume)
            result.total_cost_bps = round(
                result.spread_cost_bps
                + result.slippage_cost_bps
                + result.timing_cost_bps
                + result.market_impact_cost_bps,
                4,
            )
        return result

    def analyze_trades_from_ledger(
        self,
        *,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        limit: int = 500,
        target_price: Optional[float] = None,
        arrival_price: Optional[float] = None,
        vwap: Optional[float] = None,
        twap: Optional[float] = None,
        avg_daily_volume: Optional[float] = None,
    ) -> dict:
        """Load trades from the existing ledger and return an aggregate TCA report."""
        if self.trade_ledger is None:
            raise RuntimeError("Trade ledger integration requested but no trade_ledger was provided")

        ledger_trades = self.trade_ledger.get_trades(
            symbol=symbol,
            strategy=strategy,
            limit=limit,
        )
        results = [
            self.analyze_ledger_trade(
                trade,
                target_price=target_price,
                arrival_price=arrival_price,
                vwap=vwap,
                twap=twap,
                avg_daily_volume=avg_daily_volume,
            )
            for trade in ledger_trades
        ]
        report = self.generate_tca_report(results)
        report["ledger_trade_count"] = len(ledger_trades)
        report["symbol"] = symbol
        report["strategy"] = strategy
        return report

    def build_ledger_tca_note(self, tca_result: TCAResult, existing_notes: Optional[str] = None) -> str:
        """Return a ledger-friendly note string containing TCA analytics."""
        payload = {"tca": tca_result.to_dict()}
        note_suffix = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        if existing_notes:
            return f"{existing_notes} | {note_suffix}"
        return note_suffix

    @staticmethod
    def _require_positive(value: float, field_name: str) -> float:
        numeric_value = float(value)
        if numeric_value <= 0:
            raise ValueError(f"{field_name} must be positive, got {value!r}")
        return numeric_value

    @staticmethod
    def _absolute_bps(observed_price: float, benchmark_price: float) -> float:
        return abs(observed_price - benchmark_price) / benchmark_price * _BPS_MULTIPLIER
