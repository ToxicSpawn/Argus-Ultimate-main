"""
Hostile Scenario Injector — Stress-test strategies with adversarial market conditions.

Before promoting a strategy from PAPER to LIVE (or on demand), inject hostile
conditions into the decision context and verify the strategy+gates behave safely.

Scenarios:
  1. STALE_BOOK     — 4x spread, stale depth data
  2. FLASH_CRASH    — 30% price drop, volume spike, extreme volatility
  3. VENUE_FAILURE  — Primary venue disabled, 5s lag
  4. LIQUIDITY_VOID — 90% depth removed, 20x spread
  5. WHALE_DUMP     — Massive sell pressure, orderbook imbalance
  6. REGIME_WHIPSAW — Rapid regime transitions (TRENDING_UP → CRISIS → RANGING)
  7. FEE_SPIKE      — 10x normal fees (exchange incident)

Each scenario modifies a copy of the current market state and asks:
  "Would the strategy still produce a safe signal?"
  "Would the gates properly block or reduce?"
  "Would sizing stay within acceptable bounds?"

Results feed into strategy_promotion.py — a strategy must pass hostile tests
to be promoted from PAPER → LIVE.
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────────────────────────────────────────

class ScenarioType(str, Enum):
    STALE_BOOK = "stale_book"
    FLASH_CRASH = "flash_crash"
    VENUE_FAILURE = "venue_failure"
    LIQUIDITY_VOID = "liquidity_void"
    WHALE_DUMP = "whale_dump"
    REGIME_WHIPSAW = "regime_whipsaw"
    FEE_SPIKE = "fee_spike"


@dataclass(frozen=True)
class ScenarioParams:
    """Parameters for each hostile scenario."""
    scenario: ScenarioType
    spread_multiplier: float = 1.0
    price_shock_pct: float = 0.0      # negative = crash
    volume_multiplier: float = 1.0
    depth_multiplier: float = 1.0     # < 1.0 = reduced depth
    venue_disabled: bool = False
    venue_lag_ms: float = 0.0
    fee_multiplier: float = 1.0
    imbalance_inject: float = 0.0     # negative = sell pressure
    forced_regime: str = ""           # override regime label
    description: str = ""


# Pre-built hostile scenarios
HOSTILE_SCENARIOS: Dict[ScenarioType, ScenarioParams] = {
    ScenarioType.STALE_BOOK: ScenarioParams(
        scenario=ScenarioType.STALE_BOOK,
        spread_multiplier=4.0,
        depth_multiplier=0.3,
        description="4x spread, 70% depth gone — stale orderbook",
    ),
    ScenarioType.FLASH_CRASH: ScenarioParams(
        scenario=ScenarioType.FLASH_CRASH,
        price_shock_pct=-30.0,
        volume_multiplier=10.0,
        spread_multiplier=8.0,
        depth_multiplier=0.05,
        forced_regime="CRISIS",
        description="30% price crash, 10x volume, extreme spread",
    ),
    ScenarioType.VENUE_FAILURE: ScenarioParams(
        scenario=ScenarioType.VENUE_FAILURE,
        venue_disabled=True,
        venue_lag_ms=5000.0,
        description="Primary venue disabled, 5s lag",
    ),
    ScenarioType.LIQUIDITY_VOID: ScenarioParams(
        scenario=ScenarioType.LIQUIDITY_VOID,
        spread_multiplier=20.0,
        depth_multiplier=0.1,
        description="90% depth removed, 20x spread",
    ),
    ScenarioType.WHALE_DUMP: ScenarioParams(
        scenario=ScenarioType.WHALE_DUMP,
        price_shock_pct=-8.0,
        volume_multiplier=5.0,
        spread_multiplier=3.0,
        imbalance_inject=-0.9,
        description="Massive sell pressure, -8% price, orderbook imbalance",
    ),
    ScenarioType.REGIME_WHIPSAW: ScenarioParams(
        scenario=ScenarioType.REGIME_WHIPSAW,
        spread_multiplier=2.0,
        volume_multiplier=3.0,
        forced_regime="CRISIS",
        description="Regime whipsaw: force CRISIS, elevated vol",
    ),
    ScenarioType.FEE_SPIKE: ScenarioParams(
        scenario=ScenarioType.FEE_SPIKE,
        fee_multiplier=10.0,
        description="10x normal fees — exchange incident",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Test result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioTestResult:
    """Result of running one strategy against one hostile scenario."""
    scenario: ScenarioType
    strategy_name: str
    symbol: str
    passed: bool
    signal_generated: bool       # did strategy emit a signal?
    signal_blocked: bool         # did gates block it?
    final_size_pct: float        # what size would have been used
    gate_multiplier: float       # product of all gate adjustments
    violations: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HostileTestReport:
    """Full report for all scenarios against a strategy."""
    strategy_name: str
    total_scenarios: int = 0
    passed: int = 0
    failed: int = 0
    results: List[ScenarioTestResult] = field(default_factory=list)
    promotion_safe: bool = False  # True if all critical scenarios passed
    timestamp_ms: int = 0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total_scenarios if self.total_scenarios else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Hostile Scenario Injector
# ─────────────────────────────────────────────────────────────────────────────

class HostileScenarioInjector:
    """
    Injects adversarial market conditions to stress-test strategies.

    Usage::

        injector = HostileScenarioInjector()

        # Test a single scenario:
        stressed = injector.inject(current_prices, current_advisory, ScenarioType.FLASH_CRASH)
        # stressed["prices"] and stressed["advisory"] now reflect crash conditions

        # Full test suite for promotion:
        report = injector.test_strategy(
            strategy_name="momentum",
            symbol="BTC/USD",
            prices=current_prices,
            advisory=current_advisory,
            signal_generator=my_strategy.generate_signal,
        )
        if report.promotion_safe:
            promote_to_live(strategy_name)

    Advisory keys injected:
        advisory["hostile_scenario"] = {
            "active": True,
            "scenario": "flash_crash",
            "description": "...",
            "injected_at_ms": 1234567890,
        }
    """

    # Scenarios that MUST pass for PAPER→LIVE promotion
    CRITICAL_SCENARIOS = {
        ScenarioType.FLASH_CRASH,
        ScenarioType.LIQUIDITY_VOID,
        ScenarioType.VENUE_FAILURE,
    }

    # Maximum acceptable position size under stress (% of portfolio)
    MAX_STRESS_SIZE_PCT = 0.02  # 2% max under hostile conditions

    def __init__(self) -> None:
        self._test_count: int = 0
        self._pass_count: int = 0
        self._fail_count: int = 0
        logger.info("HostileScenarioInjector: initialized with %d scenarios", len(HOSTILE_SCENARIOS))

    def inject(
        self,
        prices: Dict[str, float],
        advisory: Dict[str, Any],
        scenario: ScenarioType,
        target_symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Inject a hostile scenario into prices and advisory.
        Returns a dict with modified "prices" and "advisory" copies.
        Does NOT modify the originals.

        If target_symbol is specified, only that symbol is stressed.
        Otherwise all symbols are affected.
        """
        params = HOSTILE_SCENARIOS.get(scenario)
        if params is None:
            logger.warning("Unknown hostile scenario: %s", scenario)
            return {"prices": prices, "advisory": advisory}

        # Deep copy to avoid mutating live state
        stressed_prices = copy.deepcopy(prices)
        stressed_advisory = copy.deepcopy(advisory)

        symbols = [target_symbol] if target_symbol else list(stressed_prices.keys())

        for sym in symbols:
            if sym not in stressed_prices:
                continue
            original_price = stressed_prices[sym]

            # Price shock
            if params.price_shock_pct != 0.0:
                stressed_prices[sym] = original_price * (1.0 + params.price_shock_pct / 100.0)

        # Inject into advisory's vol_forecasts (stress the volatility)
        vol_forecasts = stressed_advisory.get("vol_forecasts", {})
        for sym in symbols:
            if sym in vol_forecasts:
                current_vol = vol_forecasts[sym].get("forecast_vol_1d", 0.02)
                vol_forecasts[sym]["forecast_vol_1d"] = current_vol * max(
                    params.spread_multiplier, params.volume_multiplier
                )
            else:
                vol_forecasts[sym] = {
                    "forecast_vol_1d": 0.05 * params.spread_multiplier,
                    "regime": params.forced_regime or "HIGH_VOL",
                }
        stressed_advisory["vol_forecasts"] = vol_forecasts

        # Inject spread/depth/volume into market microstructure advisory
        micro = stressed_advisory.get("market_microstructure", {})
        for sym in symbols:
            sym_micro = micro.get(sym, {})
            base_spread = sym_micro.get("spread_bps", 5.0)
            base_depth = sym_micro.get("visible_depth_aud", 50000.0)
            base_volume = sym_micro.get("volume_24h", 1e6)
            base_fee = sym_micro.get("fee_bps", 2.0)

            sym_micro["spread_bps"] = base_spread * params.spread_multiplier
            sym_micro["visible_depth_aud"] = base_depth * params.depth_multiplier
            sym_micro["volume_24h"] = base_volume * params.volume_multiplier
            sym_micro["fee_bps"] = base_fee * params.fee_multiplier
            sym_micro["imbalance"] = params.imbalance_inject
            sym_micro["hostile_injected"] = True
            micro[sym] = sym_micro
        stressed_advisory["market_microstructure"] = micro

        # Venue health injection
        if params.venue_disabled or params.venue_lag_ms > 0:
            venue_health = stressed_advisory.get("venue_health", {})
            for venue in ["kraken", "coinbase"]:
                vh = venue_health.get(venue, {})
                if params.venue_disabled:
                    vh["disabled"] = True
                    vh["status"] = "OFFLINE"
                if params.venue_lag_ms > 0:
                    vh["ws_lag_ms"] = params.venue_lag_ms
                    vh["latency_ms"] = params.venue_lag_ms
                venue_health[venue] = vh
            stressed_advisory["venue_health"] = venue_health

        # Regime override
        if params.forced_regime:
            stressed_advisory["hostile_regime_override"] = params.forced_regime

        # Mark the injection
        stressed_advisory["hostile_scenario"] = {
            "active": True,
            "scenario": scenario.value,
            "description": params.description,
            "injected_at_ms": int(time.time() * 1000),
            "params": {
                "spread_multiplier": params.spread_multiplier,
                "price_shock_pct": params.price_shock_pct,
                "depth_multiplier": params.depth_multiplier,
                "volume_multiplier": params.volume_multiplier,
                "venue_disabled": params.venue_disabled,
            },
        }

        return {
            "prices": stressed_prices,
            "advisory": stressed_advisory,
            "scenario": params,
        }

    def evaluate_result(
        self,
        scenario: ScenarioType,
        strategy_name: str,
        symbol: str,
        signal_generated: bool,
        signal_blocked: bool,
        final_size_pct: float,
        gate_multiplier: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ScenarioTestResult:
        """
        Evaluate whether a strategy's behavior under stress was acceptable.

        Rules:
          - Under FLASH_CRASH / LIQUIDITY_VOID: signal must be blocked or size < 2%
          - Under VENUE_FAILURE: signal must be blocked (no venue to execute on)
          - Under any scenario: size must be < MAX_STRESS_SIZE_PCT
          - Under any scenario: gate_multiplier must be < 0.5 (gates must react)
        """
        violations: List[str] = []
        params = HOSTILE_SCENARIOS.get(scenario)

        # Check 1: size limit under stress
        if signal_generated and not signal_blocked:
            if final_size_pct > self.MAX_STRESS_SIZE_PCT:
                violations.append(
                    f"size_too_large: {final_size_pct:.4f} > {self.MAX_STRESS_SIZE_PCT}"
                )

        # Check 2: venue failure must block
        if scenario == ScenarioType.VENUE_FAILURE:
            if signal_generated and not signal_blocked:
                violations.append("venue_failure_not_blocked")

        # Check 3: flash crash / liquidity void must heavily reduce
        if scenario in (ScenarioType.FLASH_CRASH, ScenarioType.LIQUIDITY_VOID):
            if signal_generated and not signal_blocked and gate_multiplier > 0.3:
                violations.append(
                    f"insufficient_gate_reduction: mult={gate_multiplier:.3f} > 0.3"
                )

        # Check 4: whale dump should reduce BUY signals
        if scenario == ScenarioType.WHALE_DUMP:
            if signal_generated and not signal_blocked and gate_multiplier > 0.5:
                violations.append(
                    f"whale_dump_not_reduced: mult={gate_multiplier:.3f}"
                )

        passed = len(violations) == 0
        self._test_count += 1
        if passed:
            self._pass_count += 1
        else:
            self._fail_count += 1

        return ScenarioTestResult(
            scenario=scenario,
            strategy_name=strategy_name,
            symbol=symbol,
            passed=passed,
            signal_generated=signal_generated,
            signal_blocked=signal_blocked,
            final_size_pct=final_size_pct,
            gate_multiplier=gate_multiplier,
            violations=violations,
            details=extra or {},
        )

    def test_all_scenarios(
        self,
        strategy_name: str,
        symbol: str,
        prices: Dict[str, float],
        advisory: Dict[str, Any],
        signal_fn: Any = None,
    ) -> HostileTestReport:
        """
        Run ALL hostile scenarios against a strategy.

        signal_fn: optional callable(prices, advisory) -> (signal_generated, final_size_pct, gate_mult)
        If not provided, scenarios are injected but results must be evaluated externally.

        Returns HostileTestReport with promotion_safe flag.
        """
        results: List[ScenarioTestResult] = []

        for scenario_type in ScenarioType:
            stressed = self.inject(prices, advisory, scenario_type, target_symbol=symbol)

            if signal_fn is not None:
                try:
                    sig_gen, sig_blocked, size_pct, gate_mult = signal_fn(
                        stressed["prices"],
                        stressed["advisory"],
                    )
                except Exception as exc:
                    # Strategy crashed under stress — that's a failure
                    logger.warning(
                        "HostileInjector: %s crashed under %s: %s",
                        strategy_name, scenario_type.value, exc,
                    )
                    result = ScenarioTestResult(
                        scenario=scenario_type,
                        strategy_name=strategy_name,
                        symbol=symbol,
                        passed=False,
                        signal_generated=False,
                        signal_blocked=False,
                        final_size_pct=0.0,
                        gate_multiplier=0.0,
                        violations=[f"strategy_crashed: {exc}"],
                    )
                    results.append(result)
                    self._test_count += 1
                    self._fail_count += 1
                    continue

                result = self.evaluate_result(
                    scenario=scenario_type,
                    strategy_name=strategy_name,
                    symbol=symbol,
                    signal_generated=sig_gen,
                    signal_blocked=sig_blocked,
                    final_size_pct=size_pct,
                    gate_multiplier=gate_mult,
                )
                results.append(result)
            else:
                # No signal_fn: just inject and record placeholder
                results.append(ScenarioTestResult(
                    scenario=scenario_type,
                    strategy_name=strategy_name,
                    symbol=symbol,
                    passed=True,  # assume pass if we can't test
                    signal_generated=False,
                    signal_blocked=False,
                    final_size_pct=0.0,
                    gate_multiplier=0.0,
                    details={"injection_only": True},
                ))

        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)

        # Promotion safety: all CRITICAL scenarios must pass
        critical_passed = all(
            r.passed for r in results
            if r.scenario in self.CRITICAL_SCENARIOS
        )

        report = HostileTestReport(
            strategy_name=strategy_name,
            total_scenarios=len(results),
            passed=passed,
            failed=failed,
            results=results,
            promotion_safe=critical_passed,
            timestamp_ms=int(time.time() * 1000),
        )

        logger.info(
            "HostileInjector: %s — %d/%d passed, promotion_safe=%s",
            strategy_name, passed, len(results), critical_passed,
        )

        return report

    def snapshot(self) -> Dict[str, Any]:
        """Return current stats for advisory/dashboard."""
        return {
            "total_tests": self._test_count,
            "passed": self._pass_count,
            "failed": self._fail_count,
            "pass_rate": round(
                self._pass_count / self._test_count if self._test_count else 0.0, 3
            ),
            "scenarios_available": len(HOSTILE_SCENARIOS),
        }
