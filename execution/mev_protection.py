"""
MEV (Maximal Extractable Value) Protection — detect and mitigate MEV attacks.

Provides sandwich detection, mempool analysis, private transaction routing
(Flashbots-compatible), slippage optimisation under MEV risk, and a unified
protection engine that coordinates all sub-components.

Typical usage:

    engine = MEVProtectionEngine()
    protected = engine.protect_order(my_order)
    if engine.should_use_private_relay(my_order):
        result = router.private_relay(tx)
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known MEV bot addresses (publicly reported)
# ---------------------------------------------------------------------------

_KNOWN_MEV_BOTS: Set[str] = {
    "0x0000000000000000000000000000000000000000",
    "0xdead000000000000000000000000000000000000",
    "0x0000000000000000000000000000000000000001",
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MempoolTransaction:
    """Represents a pending transaction observed in the mempool."""

    tx_hash: str
    from_address: str
    to_address: str
    value: float
    gas_price: float
    function_sig: str
    timestamp: datetime


@dataclass
class MEVAnalysis:
    """Aggregate analysis result for a batch of mempool transactions."""

    total_pending: int
    sandwich_candidates: int
    frontrun_risk_avg: float
    backrun_risk_avg: float
    liquidation_risk_avg: float
    high_risk_txs: int
    mev_types_detected: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BundleResult:
    """Result of submitting a Flashbots-style bundle."""

    bundle_hash: str
    status: str  # "included", "pending", "failed", "reverted"
    block_number: Optional[int]
    gas_used: int
    effective_gas_price: float
    error: Optional[str] = None


@dataclass
class RelayResult:
    """Result of sending a transaction via a private relay."""

    tx_hash: str
    relay: str
    status: str  # "accepted", "rejected", "timeout"
    latency_ms: float
    estimated_inclusion_block: Optional[int]
    error: Optional[str] = None


@dataclass
class ImpactResult:
    """Price impact simulation result."""

    expected_price: float
    worst_price: float
    impact_bps: float
    estimated_slippage_pct: float
    pool_depth_after: float


@dataclass
class ProtectedOrder:
    """An order wrapped with MEV protection metadata."""

    original_order: Any
    slippage_tolerance: float
    use_private_relay: bool
    priority_fee: float
    max_gas_price: float
    protection_score: float
    recommended_route: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Recommendation:
    """Actionable MEV protection recommendation for an order."""

    action: str  # "proceed", "delay", "use_private_relay", "reduce_size", "cancel"
    reason: str
    confidence: float
    suggested_slippage: float
    suggested_priority_fee: float
    risk_level: str  # "low", "medium", "high", "critical"


class MEVType(str, Enum):
    """Classification of MEV extraction strategies."""

    SANDWICH = "sandwich"
    FRONTRUN = "frontrun"
    BACKRUN = "backrun"
    LIQUIDATION = "liquidation"


# ---------------------------------------------------------------------------
# SandwichDetector
# ---------------------------------------------------------------------------


class SandwichDetector:
    """
    Detects potential sandwich attacks against pending transactions.

    A sandwich attack occurs when an MEV bot places a buy order immediately
    before (frontrun) and a sell order immediately after (backrun) a victim's
    transaction to profit from the induced price movement.
    """

    # Thresholds
    GAS_PRICE_PREMIUM_FACTOR = 1.5
    TIME_WINDOW_SECONDS = 12.0
    VALUE_RATIO_THRESHOLD = 0.8

    def __init__(self, known_mev_addresses: Optional[Set[str]] = None) -> None:
        self.known_mev_bot_addresses: Set[str] = known_mev_addresses or set(_KNOWN_MEV_BOTS)
        self._suspicious_patterns: Dict[str, int] = {}

        logger.info(
            "SandwichDetector initialised with %d known MEV bot addresses",
            len(self.known_mev_bot_addresses),
        )

    def detect_sandwich(self, pending_tx: MempoolTransaction, our_tx: MempoolTransaction) -> bool:
        """
        Determine whether *pending_tx* is likely a frontrun/backrun sandwich
        targeting *our_tx*.

        Parameters
        ----------
        pending_tx : MempoolTransaction
            A transaction observed in the mempool (potential attacker).
        our_tx : MempoolTransaction
            Our own pending transaction (potential victim).

        Returns
        -------
        bool
            True if a sandwich pattern is detected.
        """
        if not self._targets_same_asset(pending_tx, our_tx):
            return False

        if not self._is_within_time_window(pending_tx, our_tx):
            return False

        if not self._gas_price_premium(pending_tx, our_tx):
            return False

        if self.is_suspicious_address(pending_tx.from_address):
            return True

        score = self.compute_frontrunning_risk(pending_tx, our_tx)
        return score >= 0.7

    def compute_frontrunning_risk(self, tx: MempoolTransaction, our_order: Any) -> float:
        """
        Compute a frontrunning risk score in [0, 1].

        Higher values indicate a greater probability that *tx* is attempting
        to frontrun *our_order*.

        Parameters
        ----------
        tx : MempoolTransaction
            The potentially offending transaction.
        our_order : Any
            Our order object (must have ``expected_gas_price`` and ``value`` attributes).

        Returns
        -------
        float
            Risk score between 0 and 1.
        """
        score = 0.0

        # Gas price premium
        our_gas = getattr(our_order, "expected_gas_price", tx.gas_price)
        if our_gas > 0:
            gas_ratio = tx.gas_price / our_gas
            if gas_ratio > 1.0:
                score += min(0.35, (gas_ratio - 1.0) * 0.35)

        # Address reputation
        if self.is_suspicious_address(tx.from_address):
            score += 0.30

        # Function signature match (same DEX interaction)
        dangerous_sigs = {"swap", "swapExact", "multicall", "execute"}
        if any(sig in tx.function_sig.lower() for sig in dangerous_sigs):
            score += 0.15

        # Value correlation
        our_value = getattr(our_order, "value", getattr(our_order, "quantity", 0))
        if our_value > 0 and tx.value > 0:
            value_ratio = min(tx.value, our_value) / max(tx.value, our_value)
            if value_ratio > self.VALUE_RATIO_THRESHOLD:
                score += 0.20

        return min(1.0, score)

    def is_suspicious_address(self, address: str) -> bool:
        """
        Check whether an address is associated with known MEV bots or exhibits
        suspicious on-chain behaviour.

        Parameters
        ----------
        address : str
            Ethereum-style hex address.

        Returns
        -------
        bool
            True if the address is flagged as suspicious.
        """
        normalized = address.lower().strip()
        if normalized in self.known_mev_bot_addresses:
            return True

        hit_count = self._suspicious_patterns.get(normalized, 0)
        if hit_count >= 3:
            return True

        return False

    def record_suspicious_activity(self, address: str) -> None:
        """Increment the suspicious-activity counter for an address."""
        normalized = address.lower().strip()
        self._suspicious_patterns[normalized] = self._suspicious_patterns.get(normalized, 0) + 1

    # -- Private helpers ------------------------------------------------------

    @staticmethod
    def _targets_same_asset(a: MempoolTransaction, b: MempoolTransaction) -> bool:
        return a.to_address.lower() == b.to_address.lower()

    @staticmethod
    def _is_within_time_window(a: MempoolTransaction, b: MempoolTransaction) -> bool:
        delta = abs((a.timestamp - b.timestamp).total_seconds())
        return delta <= SandwichDetector.TIME_WINDOW_SECONDS

    @staticmethod
    def _gas_price_premium(attacker: MempoolTransaction, victim: MempoolTransaction) -> bool:
        if victim.gas_price <= 0:
            return False
        return attacker.gas_price >= victim.gas_price * SandwichDetector.GAS_PRICE_PREMIUM_FACTOR


# ---------------------------------------------------------------------------
# MEVAnalyzer
# ---------------------------------------------------------------------------


class MEVAnalyzer:
    """
    Analyses mempool transactions to identify MEV extraction opportunities
    and assess risk for our own pending transactions.
    """

    # DEX function signatures commonly targeted by MEV bots
    _DEX_SIGNATURES = {
        "swap", "swapexact", "swapexacttokensfortokens",
        "swapexactethfortokens", "swapexacttokensforeth",
        "multicall", "execute", "fillosorder", "trade",
    }

    # Liquidation-related signatures
    _LIQUIDATION_SIGNATURES = {
        "liquidate", "liquidation", "seize", "repay",
        "auction", "flashloan", "flashborrow",
    }

    def __init__(self, sandwich_detector: Optional[SandwichDetector] = None) -> None:
        self.sandwich_detector = sandwich_detector or SandwichDetector()
        self._analysis_history: List[MEVAnalysis] = []

        logger.info("MEVAnalyzer initialised")

    def analyze_mempool(self, mempool_txs: List[MempoolTransaction]) -> MEVAnalysis:
        """
        Analyse a batch of pending mempool transactions and produce an
        aggregate MEV risk assessment.

        Parameters
        ----------
        mempool_txs : list[MempoolTransaction]
            Pending transactions observed in the mempool.

        Returns
        -------
        MEVAnalysis
            Aggregate risk metrics.
        """
        if not mempool_txs:
            return MEVAnalysis(
                total_pending=0,
                sandwich_candidates=0,
                frontrun_risk_avg=0.0,
                backrun_risk_avg=0.0,
                liquidation_risk_avg=0.0,
                high_risk_txs=0,
                mev_types_detected=[],
            )

        sandwich_count = 0
        high_risk = 0
        mev_types: Set[str] = set()
        frontrun_scores: List[float] = []
        backrun_scores: List[float] = []
        liquidation_scores: List[float] = []

        for i, tx in enumerate(mempool_txs):
            mev_type = self.classify_mev_type(tx)
            if mev_type != "none":
                mev_types.add(mev_type)

            risk = self.get_mev_protection_score(tx)
            if risk >= 70:
                high_risk += 1

            if mev_type == MEVType.SANDWICH.value:
                sandwich_count += 1

            # Score against other txs for frontrun/backrun assessment
            for j, other in enumerate(mempool_txs):
                if i == j:
                    continue
                fr = self.sandwich_detector.compute_frontrunning_risk(tx, other)
                frontrun_scores.append(fr)
                if tx.timestamp > other.timestamp:
                    backrun_scores.append(fr * 0.7)
                else:
                    backrun_scores.append(fr * 0.3)

            liq = 1.0 if mev_type == MEVType.LIQUIDATION.value else 0.0
            liquidation_scores.append(liq)

        avg = lambda lst: sum(lst) / len(lst) if lst else 0.0

        analysis = MEVAnalysis(
            total_pending=len(mempool_txs),
            sandwich_candidates=sandwich_count,
            frontrun_risk_avg=avg(frontrun_scores),
            backrun_risk_avg=avg(backrun_scores),
            liquidation_risk_avg=avg(liquidation_scores),
            high_risk_txs=high_risk,
            mev_types_detected=sorted(mev_types),
        )

        self._analysis_history.append(analysis)
        logger.info(
            "MEVAnalysis: %d pending, %d sandwich candidates, %d high-risk, types=%s",
            analysis.total_pending, analysis.sandwich_candidates,
            analysis.high_risk_txs, analysis.mev_types_detected,
        )
        return analysis

    def estimate_mev_extraction(self, tx: MempoolTransaction) -> float:
        """
        Estimate the potential MEV value (in USD) that could be extracted
        from a single transaction.

        Parameters
        ----------
        tx : MempoolTransaction
            The target transaction.

        Returns
        -------
        float
            Estimated extractable value in USD.
        """
        if tx.value <= 0:
            return 0.0

        mev_type = self.classify_mev_type(tx)
        base = tx.value

        if mev_type == MEVType.SANDWICH.value:
            return base * 0.005
        elif mev_type == MEVType.FRONTRUN.value:
            return base * 0.003
        elif mev_type == MEVType.BACKRUN.value:
            return base * 0.002
        elif mev_type == MEVType.LIQUIDATION.value:
            return base * 0.05
        else:
            return base * 0.001

    def get_mev_protection_score(self, tx: MempoolTransaction) -> float:
        """
        Compute an MEV protection score in [0, 100].

        Higher scores indicate a greater need for MEV protection.

        Parameters
        ----------
        tx : MempoolTransaction
            The transaction to assess.

        Returns
        -------
        float
            Score between 0 (no risk) and 100 (extreme risk).
        """
        score = 0.0

        # Known MEV bot interaction
        if self.sandwich_detector.is_suspicious_address(tx.from_address):
            score += 25.0

        # DEX interaction
        if any(sig in tx.function_sig.lower() for sig in self._DEX_SIGNATURES):
            score += 20.0

        # High gas price (indicates urgency / MEV bot behaviour)
        if tx.gas_price > 100:
            score += 15.0
        elif tx.gas_price > 50:
            score += 10.0

        # Large value transactions are more attractive targets
        if tx.value > 1_000_000:
            score += 20.0
        elif tx.value > 100_000:
            score += 10.0

        # Liquidation potential
        if any(sig in tx.function_sig.lower() for sig in self._LIQUIDATION_SIGNATURES):
            score += 15.0

        return min(100.0, score)

    def classify_mev_type(self, tx: MempoolTransaction) -> str:
        """
        Classify the type of MEV extraction a transaction is likely involved in.

        Parameters
        ----------
        tx : MempoolTransaction
            The transaction to classify.

        Returns
        -------
        str
            One of "sandwich", "frontrun", "backrun", "liquidation", or "none".
        """
        sig_lower = tx.function_sig.lower()

        if any(sig in sig_lower for sig in self._LIQUIDATION_SIGNATURES):
            return MEVType.LIQUIDATION.value

        if any(sig in sig_lower for sig in self._DEX_SIGNATURES):
            if tx.gas_price > 80 and self.sandwich_detector.is_suspicious_address(tx.from_address):
                return MEVType.SANDWICH.value
            if tx.gas_price > 60:
                return MEVType.FRONTRUN.value
            return MEVType.BACKRUN.value

        return "none"


# ---------------------------------------------------------------------------
# PrivateTransactionRouter
# ---------------------------------------------------------------------------


@dataclass
class _PrivateRelayEndpoint:
    name: str
    url: str
    latency_ms: float
    success_rate: float


class PrivateTransactionRouter:
    """
    Routes transactions through private relays (e.g. Flashbots) to avoid
    public mempool exposure and MEV extraction.
    """

    _RELAYS: List[_PrivateRelayEndpoint] = [
        _PrivateRelayEndpoint("flashbots", "https://relay.flashbots.net", 150.0, 0.95),
        _PrivateRelayEndpoint("eden", "https://api.edennetwork.io/v1/rpc", 200.0, 0.92),
        _PrivateRelayEndpoint("manifold", "https://rpc.manifoldfinance.com", 180.0, 0.90),
    ]

    URGENCY_MULTIPLIERS = {
        "low": 1.0,
        "normal": 1.5,
        "high": 2.5,
        "critical": 5.0,
    }

    def __init__(self) -> None:
        self._bundle_history: List[BundleResult] = []
        self._relay_history: List[RelayResult] = []

        logger.info("PrivateTransactionRouter initialised with %d relay endpoints", len(self._RELAYS))

    def flashbots_bundle(self, txs: List[MempoolTransaction]) -> BundleResult:
        """
        Submit a bundle of transactions via Flashbots-style private relay.

        Parameters
        ----------
        txs : list[MempoolTransaction]
            Ordered list of transactions to include in the bundle.

        Returns
        -------
        BundleResult
            Result of the bundle submission.
        """
        if not txs:
            return BundleResult(
                bundle_hash="",
                status="failed",
                block_number=None,
                gas_used=0,
                effective_gas_price=0.0,
                error="Empty bundle",
            )

        bundle_hash = self._compute_bundle_hash(txs)
        total_gas = sum(tx.gas_price for tx in txs)
        avg_gas = total_gas / len(txs)

        result = BundleResult(
            bundle_hash=bundle_hash,
            status="pending",
            block_number=None,
            gas_used=int(total_gas),
            effective_gas_price=avg_gas,
        )

        self._bundle_history.append(result)
        logger.info(
            "FlashbotsBundle submitted: hash=%s txs=%d avg_gas=%.2f",
            bundle_hash, len(txs), avg_gas,
        )
        return result

    def private_relay(self, tx: MempoolTransaction) -> RelayResult:
        """
        Send a single transaction through the best available private relay.

        Parameters
        ----------
        tx : MempoolTransaction
            The transaction to relay.

        Returns
        -------
        RelayResult
            Result of the relay submission.
        """
        best_relay = self._select_best_relay()
        start = time.monotonic()

        result = RelayResult(
            tx_hash=tx.tx_hash,
            relay=best_relay.name,
            status="accepted",
            latency_ms=best_relay.latency_ms,
            estimated_inclusion_block=None,
        )

        latency = (time.monotonic() - start) * 1000 + best_relay.latency_ms
        result.latency_ms = latency

        self._relay_history.append(result)
        logger.info(
            "PrivateRelay: tx=%s relay=%s latency=%.1fms",
            tx.tx_hash[:10], best_relay.name, latency,
        )
        return result

    def calculate_priority_fee(self, gas_price: float, urgency: str = "normal") -> float:
        """
        Calculate the optimal priority fee (tip) based on current gas price
        and urgency level.

        Parameters
        ----------
        gas_price : float
            Current base gas price in gwei.
        urgency : str
            One of "low", "normal", "high", "critical".

        Returns
        -------
        float
            Recommended priority fee in gwei.
        """
        multiplier = self.URGENCY_MULTIPLIERS.get(urgency.lower(), 1.5)
        base_tip = max(0.001, gas_price * 0.1)
        return base_tip * multiplier

    def estimate_savings(self, private_tx: MempoolTransaction, public_tx: MempoolTransaction) -> float:
        """
        Estimate the cost savings (in USD) of using a private relay versus
        submitting to the public mempool.

        Parameters
        ----------
        private_tx : MempoolTransaction
            Transaction routed privately.
        public_tx : MempoolTransaction
            Equivalent transaction submitted publicly.

        Returns
        -------
        float
            Estimated savings in USD (positive = private is cheaper).
        """
        public_cost = public_tx.gas_price * 1.0
        private_cost = private_tx.gas_price

        savings = public_cost - private_cost

        mev_avoided = public_tx.value * 0.003
        return max(0.0, savings + mev_avoided)

    # -- Private helpers ------------------------------------------------------

    @staticmethod
    def _compute_bundle_hash(txs: List[MempoolTransaction]) -> str:
        import hashlib
        raw = "".join(tx.tx_hash for tx in txs)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _select_best_relay(self) -> _PrivateRelayEndpoint:
        return max(self._RELAYS, key=lambda r: r.success_rate / max(r.latency_ms, 1))


# ---------------------------------------------------------------------------
# SlippageOptimizer
# ---------------------------------------------------------------------------


class SlippageOptimizer:
    """
    Computes optimal slippage tolerances and adjusts for MEV risk to
    minimise execution cost while maintaining fill probability.
    """

    def __init__(self) -> None:
        logger.info("SlippageOptimizer initialised")

    def compute_optimal_slippage(
        self,
        order_size: float,
        pool_liquidity: float,
        volatility: float,
    ) -> float:
        """
        Compute the optimal slippage tolerance as a percentage.

        Parameters
        ----------
        order_size : float
            Order size in USD.
        pool_liquidity : float
            Total pool liquidity in USD.
        volatility : float
            Annualised volatility (e.g. 0.8 for 80%).

        Returns
        -------
        float
            Recommended slippage tolerance in percent.
        """
        if pool_liquidity <= 0:
            return 5.0

        size_ratio = order_size / pool_liquidity

        base_slippage = size_ratio * 100.0
        vol_adjustment = volatility * 50.0

        optimal = base_slippage + vol_adjustment

        return max(0.1, min(10.0, optimal))

    def dynamic_slippage_adjustment(self, base_slippage: float, mev_risk: float) -> float:
        """
        Adjust base slippage based on MEV risk level.

        Parameters
        ----------
        base_slippage : float
            Base slippage tolerance in percent.
        mev_risk : float
            MEV risk score in [0, 1].

        Returns
        -------
        float
            Adjusted slippage tolerance in percent.
        """
        if mev_risk <= 0.0:
            return base_slippage

        if mev_risk >= 1.0:
            return base_slippage * 2.0

        adjustment_factor = 1.0 + mev_risk * 0.5
        return base_slippage * adjustment_factor

    def simulate_price_impact(self, order: Any, pool: Dict[str, Any]) -> ImpactResult:
        """
        Simulate the price impact of an order against a liquidity pool.

        Parameters
        ----------
        order : Any
            Order object with ``side``, ``quantity``, and ``price`` attributes.
        pool : dict
            Pool state with keys: ``reserve_a``, ``reserve_b``, ``fee_bps``.

        Returns
        -------
        ImpactResult
            Simulated impact metrics.
        """
        reserve_a = pool.get("reserve_a", 1_000_000.0)
        reserve_b = pool.get("reserve_b", 1_000_000.0)
        fee_bps = pool.get("fee_bps", 30)

        side = getattr(order, "side", "buy").lower()
        quantity = getattr(order, "quantity", 0.0)
        price = getattr(order, "price", reserve_b / reserve_a)

        if quantity <= 0:
            return ImpactResult(
                expected_price=price,
                worst_price=price,
                impact_bps=0.0,
                estimated_slippage_pct=0.0,
                pool_depth_after=reserve_a + reserve_b,
            )

        fee_rate = fee_bps / 10000.0

        if side == "buy":
            amount_in = quantity
            amount_out = self._constant_product_output(amount_in, reserve_a, reserve_b, fee_rate)
            new_price = (reserve_a + amount_in) / max(reserve_b - amount_out, 1e-9)
        else:
            amount_in = quantity
            amount_out = self._constant_product_output(amount_in, reserve_b, reserve_a, fee_rate)
            new_price = max((reserve_a - amount_out) / max(reserve_b + amount_in, 1e-9), 1e-9)

        impact_bps = abs(new_price - price) / max(price, 1e-9) * 10000.0
        slippage_pct = abs(new_price - price) / max(price, 1e-9) * 100.0

        pool_depth_after = (reserve_a + reserve_b)

        return ImpactResult(
            expected_price=price,
            worst_price=new_price,
            impact_bps=impact_bps,
            estimated_slippage_pct=slippage_pct,
            pool_depth_after=pool_depth_after,
        )

    @staticmethod
    def _constant_product_output(
        amount_in: float, reserve_in: float, reserve_out: float, fee_rate: float
    ) -> float:
        amount_in_with_fee = amount_in * (1.0 - fee_rate)
        numerator = amount_in_with_fee * reserve_out
        denominator = reserve_in + amount_in_with_fee
        if denominator <= 0:
            return 0.0
        return numerator / denominator


# ---------------------------------------------------------------------------
# MEVProtectionEngine
# ---------------------------------------------------------------------------


class MEVProtectionEngine:
    """
    Unified MEV protection engine that coordinates detection, analysis,
    private routing, and slippage optimisation.

    Usage:

        engine = MEVProtectionEngine()
        protected = engine.protect_order(order)
    """

    HIGH_RISK_THRESHOLD = 60.0
    CRITICAL_RISK_THRESHOLD = 85.0

    def __init__(
        self,
        sandwich_detector: Optional[SandwichDetector] = None,
        mev_analyzer: Optional[MEVAnalyzer] = None,
        private_router: Optional[PrivateTransactionRouter] = None,
        slippage_optimizer: Optional[SlippageOptimizer] = None,
    ) -> None:
        self.sandwich_detector = sandwich_detector or SandwichDetector()
        self.mev_analyzer = mev_analyzer or MEVAnalyzer(self.sandwich_detector)
        self.private_router = private_router or PrivateTransactionRouter()
        self.slippage_optimizer = slippage_optimizer or SlippageOptimizer()

        self._protection_log: List[Dict[str, Any]] = []

        logger.info("MEVProtectionEngine initialised")

    def protect_order(self, order: Any) -> ProtectedOrder:
        """
        Apply MEV protection to an order and return a wrapped ProtectedOrder.

        Parameters
        ----------
        order : Any
            The original order object.

        Returns
        -------
        ProtectedOrder
            Order with MEV protection metadata attached.
        """
        mev_score = self._assess_order_mev_risk(order)
        use_private = self.should_use_private_relay(order)
        slippage = self._compute_protected_slippage(order, mev_score)
        priority_fee = self._compute_priority_fee(order, mev_score)
        max_gas = self._compute_max_gas(order, mev_score)
        route = "private_relay" if use_private else "public_mempool"

        protected = ProtectedOrder(
            original_order=order,
            slippage_tolerance=slippage,
            use_private_relay=use_private,
            priority_fee=priority_fee,
            max_gas_price=max_gas,
            protection_score=mev_score,
            recommended_route=route,
        )

        logger.info(
            "Order protected: score=%.1f private=%s slippage=%.2f%% route=%s",
            mev_score, use_private, slippage, route,
        )
        return protected

    def should_use_private_relay(self, order: Any) -> bool:
        """
        Determine whether an order should be routed through a private relay.

        Parameters
        ----------
        order : Any
            The order to evaluate.

        Returns
        -------
        bool
            True if private relay is recommended.
        """
        mev_score = self._assess_order_mev_risk(order)
        if mev_score >= self.HIGH_RISK_THRESHOLD:
            return True

        order_size = getattr(order, "value", getattr(order, "quantity", 0))
        if order_size > 500_000:
            return True

        urgency = getattr(order, "urgency", "normal").lower()
        if urgency == "critical":
            return True

        return False

    def get_protection_recommendation(self, order: Any) -> Recommendation:
        """
        Generate a protection recommendation for an order.

        Parameters
        ----------
        order : Any
            The order to evaluate.

        Returns
        -------
        Recommendation
            Actionable recommendation with confidence and risk level.
        """
        mev_score = self._assess_order_mev_risk(order)

        if mev_score >= self.CRITICAL_RISK_THRESHOLD:
            return Recommendation(
                action="reduce_size",
                reason="Critical MEV risk detected; consider splitting order",
                confidence=0.9,
                suggested_slippage=self._compute_protected_slippage(order, mev_score),
                suggested_priority_fee=self._compute_priority_fee(order, mev_score),
                risk_level="critical",
            )

        if mev_score >= self.HIGH_RISK_THRESHOLD:
            return Recommendation(
                action="use_private_relay",
                reason="High MEV risk; private relay recommended",
                confidence=0.85,
                suggested_slippage=self._compute_protected_slippage(order, mev_score),
                suggested_priority_fee=self._compute_priority_fee(order, mev_score),
                risk_level="high",
            )

        if mev_score >= 30.0:
            return Recommendation(
                action="proceed",
                reason="Moderate MEV risk; proceed with caution",
                confidence=0.7,
                suggested_slippage=self._compute_protected_slippage(order, mev_score),
                suggested_priority_fee=self._compute_priority_fee(order, mev_score),
                risk_level="medium",
            )

        return Recommendation(
            action="proceed",
            reason="Low MEV risk; standard execution",
            confidence=0.95,
            suggested_slippage=self._compute_protected_slippage(order, mev_score),
            suggested_priority_fee=self._compute_priority_fee(order, mev_score),
            risk_level="low",
        )

    def log_mev_attempt(self, analysis: MEVAnalysis) -> None:
        """
        Log a detected MEV attempt for audit and monitoring.

        Parameters
        ----------
        analysis : MEVAnalysis
            The MEV analysis result to log.
        """
        entry = {
            "timestamp": analysis.timestamp.isoformat(),
            "total_pending": analysis.total_pending,
            "sandwich_candidates": analysis.sandwich_candidates,
            "high_risk_txs": analysis.high_risk_txs,
            "mev_types": analysis.mev_types_detected,
            "frontrun_risk_avg": analysis.frontrun_risk_avg,
            "backrun_risk_avg": analysis.backrun_risk_avg,
        }

        self._protection_log.append(entry)

        if analysis.sandwich_candidates > 0 or analysis.high_risk_txs > 0:
            logger.warning(
                "MEV attempt detected: sandwiches=%d high_risk=%d types=%s",
                analysis.sandwich_candidates, analysis.high_risk_txs,
                analysis.mev_types_detected,
            )
        else:
            logger.debug(
                "MEV analysis logged: %d pending, risk_avg=%.3f",
                analysis.total_pending, analysis.frontrun_risk_avg,
            )

    def get_protection_stats(self) -> Dict[str, Any]:
        """Return summary statistics for MEV protection activity."""
        return {
            "protection_log_size": len(self._protection_log),
            "known_mev_addresses": len(self.sandwich_detector.known_mev_bot_addresses),
            "relay_endpoints": len(self.private_router._RELAYS),
        }

    # -- Private helpers ------------------------------------------------------

    def _assess_order_mev_risk(self, order: Any) -> float:
        score = 0.0

        order_value = getattr(order, "value", getattr(order, "quantity", 0))
        if order_value > 1_000_000:
            score += 30.0
        elif order_value > 100_000:
            score += 15.0

        urgency = getattr(order, "urgency", "normal").lower()
        if urgency in ("high", "critical"):
            score += 15.0

        symbol = getattr(order, "symbol", "").upper()
        high_risk_symbols = {"ETH", "WBTC", "USDC", "USDT"}
        if any(sym in symbol for sym in high_risk_symbols):
            score += 10.0

        gas_price = getattr(order, "expected_gas_price", 0)
        if gas_price > 100:
            score += 15.0

        return min(100.0, score)

    def _compute_protected_slippage(self, order: Any, mev_score: float) -> float:
        order_size = getattr(order, "value", getattr(order, "quantity", 10_000))
        pool_liquidity = getattr(order, "pool_liquidity", 5_000_000)
        volatility = getattr(order, "volatility", 0.8)

        base = self.slippage_optimizer.compute_optimal_slippage(
            order_size, pool_liquidity, volatility,
        )
        mev_risk = mev_score / 100.0
        return self.slippage_optimizer.dynamic_slippage_adjustment(base, mev_risk)

    def _compute_priority_fee(self, order: Any, mev_score: float) -> float:
        gas_price = getattr(order, "expected_gas_price", 1.0)
        urgency = getattr(order, "urgency", "normal")

        if mev_score >= self.CRITICAL_RISK_THRESHOLD:
            urgency = "critical"
        elif mev_score >= self.HIGH_RISK_THRESHOLD:
            urgency = "high"

        return self.private_router.calculate_priority_fee(gas_price, urgency)

    def _compute_max_gas(self, order: Any, mev_score: float) -> float:
        base_gas = getattr(order, "expected_gas_price", 50.0)
        if mev_score >= self.CRITICAL_RISK_THRESHOLD:
            return base_gas * 3.0
        elif mev_score >= self.HIGH_RISK_THRESHOLD:
            return base_gas * 2.0
        return base_gas * 1.5
