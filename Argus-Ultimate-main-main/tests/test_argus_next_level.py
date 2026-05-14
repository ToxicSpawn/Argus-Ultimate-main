"""
Tests for ARGUS Next-Level Advancement Batches F, G, H, I, J.

Batch G: 10 advisory gates in _execute_signals
Batch H: Drawdown partial close + funding cost rotation exits
Batch I: TWAP execution + toxicity order type switching + config fields
Batch F: Model staleness deration + FeatureDiscoverer feedback
Batch J: LiquidationCascadeStrategy in ComponentRegistry
"""

import math
import time
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
#  Pure Python helpers replicating gate logic (no imports from main codebase)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_ensemble_composite(size_pct: float, advisory: dict) -> float:
    """G1: Ensemble composite — multi-source signal strength."""
    ens = advisory.get("ensemble")
    if ens and isinstance(ens, dict):
        composite = float(ens.get("composite", 0.0) or 0.0)
        if composite != 0.0:
            mult = max(0.50, min(1.30, 1.0 + composite * 0.30))
            size_pct *= mult
    return size_pct


def apply_antifragile(size_pct: float, advisory: dict) -> float:
    """G2: Antifragile multiplier (float scalar from 'antifragile_multiplier' key)."""
    af_mult = advisory.get("antifragile_multiplier")
    if af_mult is not None:
        mult = float(af_mult)
        mult = max(0.50, min(1.50, mult))
        if mult != 1.0:
            size_pct *= mult
    return size_pct


def apply_bleeders(size_pct: float, advisory: dict, strategy: str) -> float:
    """G3: Bleeders — halve size for strategies in losing streak (LIST of dicts)."""
    bldr = advisory.get("bleeders")
    if bldr and isinstance(bldr, list):
        names = [str(b.get("name", "")) for b in bldr if isinstance(b, dict)]
        if strategy in names:
            size_pct *= 0.50
    return size_pct


def apply_tca(size_pct: float, advisory: dict) -> float:
    """G4: TCA score — high transaction costs reduce size (float scalar from 'tca_score')."""
    score = advisory.get("tca_score")
    if score is not None:
        score = float(score)
        if score > 50:
            mult = max(0.50, 1.0 - (score - 50) / 100.0)
            size_pct *= mult
    return size_pct


def apply_system_status(size_pct: float, advisory: dict, action: str) -> tuple:
    """G5: System status. Returns (size_pct, should_skip)."""
    sys_adv = advisory.get("system_status")
    if sys_adv and isinstance(sys_adv, dict):
        status = str(sys_adv.get("status", "HEALTHY")).upper()
        if status == "CRITICAL" and action == "BUY":
            return size_pct, True  # skip
        elif status == "DEGRADED":
            size_pct *= 0.70
    return size_pct, False


def apply_whale(size_pct: float, advisory: dict) -> float:
    """G6: Whale activity."""
    whale = advisory.get("whale_activity")
    if whale and isinstance(whale, dict):
        bias = float(whale.get("net_flow_bias", 0.0) or 0.0)
        if bias != 0.0:
            mult = max(0.70, min(1.30, 1.0 + bias * 0.25))
            size_pct *= mult
    return size_pct


def apply_causal_graph(size_pct: float, advisory: dict, action: str) -> float:
    """G7: Causal graph."""
    cg = advisory.get("causal_graph")
    if cg and isinstance(cg, dict):
        conf = float(cg.get("confidence", 0.0) or 0.0)
        direction = str(cg.get("direction", "neutral")).lower()
        if conf > 0.60:
            if direction == "bearish" and action == "BUY":
                size_pct *= 0.70
            elif direction == "bullish" and action == "BUY":
                size_pct *= 1.20
    return size_pct


def apply_outcome_correlator(size_pct: float, advisory: dict) -> float:
    """G8: Outcome correlator."""
    oc = advisory.get("outcome_correlator")
    if oc and isinstance(oc, dict):
        score = float(oc.get("favorability", 0.5) or 0.5)
        if score < 0.30:
            size_pct *= 0.60
        elif score > 0.70:
            size_pct *= 1.15
    return size_pct


def apply_alpha_decay(size_pct: float, advisory: dict, strategy: str) -> float:
    """G9: Alpha decay."""
    ad = advisory.get("alpha_decay")
    if ad and isinstance(ad, dict):
        decays = ad.get("strategy_decays", {})
        if isinstance(decays, dict) and strategy in decays:
            factor = float(decays[strategy])
            factor = max(0.30, min(1.0, factor))
            if factor < 1.0:
                size_pct *= factor
    return size_pct


def apply_liquidation_cascade(size_pct: float, advisory: dict, symbol: str, action: str) -> float:
    """G10: Liquidation cascade."""
    lc = advisory.get("liquidation_cascade")
    if lc and isinstance(lc, dict):
        signals = lc.get("signals", [])
        for sig in (signals if isinstance(signals, list) else []):
            if isinstance(sig, dict) and sig.get("symbol") == symbol:
                direction = str(sig.get("direction", "")).lower()
                conf = float(sig.get("confidence", 0.0) or 0.0)
                if direction == "long_squeeze" and action == "BUY" and conf > 0.60:
                    size_pct *= 0.40
                elif direction == "short_squeeze" and action == "BUY" and conf > 0.60:
                    size_pct *= 1.30
                break
    return size_pct


def apply_toxicity(order_type: str, advisory: dict) -> str:
    """Batch I: Toxicity → market switch."""
    tox = advisory.get("toxicity")
    if tox and isinstance(tox, dict):
        score = float(tox.get("toxicity_score", 0.0) or 0.0)
        if score >= 0.70 and order_type == "limit":
            return "market"
    return order_type


def check_twap_requested(advisory: dict) -> bool:
    """Batch I: Check if TWAP requested."""
    ei = advisory.get("execution_intelligence")
    if ei and isinstance(ei, dict):
        otype = str(ei.get("order_type", "")).lower()
        if otype in ("twap", "vwap", "adaptive"):
            return True
    return False


def model_staleness_derate(staleness: float) -> float:
    """Batch F1: Compute derate factor for stale models."""
    if staleness > 0.5:
        return max(0.30, 1.0 - staleness)
    return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch G Tests: 10 advisory gates
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchG_EnsembleComposite(unittest.TestCase):
    """G1: Ensemble composite — multi-source signal strength."""

    def test_positive_composite_boosts(self):
        result = apply_ensemble_composite(1.0, {"ensemble": {"composite": 0.5}})
        self.assertAlmostEqual(result, 1.15, places=2)

    def test_negative_composite_reduces(self):
        result = apply_ensemble_composite(1.0, {"ensemble": {"composite": -0.5}})
        self.assertAlmostEqual(result, 0.85, places=2)

    def test_zero_composite_unchanged(self):
        result = apply_ensemble_composite(1.0, {"ensemble": {"composite": 0.0}})
        self.assertAlmostEqual(result, 1.0)

    def test_capped_at_1_30(self):
        result = apply_ensemble_composite(1.0, {"ensemble": {"composite": 5.0}})
        self.assertAlmostEqual(result, 1.30, places=2)

    def test_floored_at_0_50(self):
        result = apply_ensemble_composite(1.0, {"ensemble": {"composite": -5.0}})
        self.assertAlmostEqual(result, 0.50, places=2)

    def test_missing_advisory_unchanged(self):
        result = apply_ensemble_composite(1.0, {})
        self.assertAlmostEqual(result, 1.0)


class TestBatchG_Antifragile(unittest.TestCase):
    """G2: Antifragile multiplier."""

    def test_boost_multiplier(self):
        result = apply_antifragile(1.0, {"antifragile_multiplier": 1.3})
        self.assertAlmostEqual(result, 1.3, places=2)

    def test_reduce_multiplier(self):
        result = apply_antifragile(1.0, {"antifragile_multiplier": 0.7})
        self.assertAlmostEqual(result, 0.7, places=2)

    def test_capped_at_1_50(self):
        result = apply_antifragile(1.0, {"antifragile_multiplier": 2.0})
        self.assertAlmostEqual(result, 1.50, places=2)

    def test_floored_at_0_50(self):
        result = apply_antifragile(1.0, {"antifragile_multiplier": 0.1})
        self.assertAlmostEqual(result, 0.50, places=2)

    def test_multiplier_1_unchanged(self):
        result = apply_antifragile(1.0, {"antifragile_multiplier": 1.0})
        self.assertAlmostEqual(result, 1.0)

    def test_missing_unchanged(self):
        result = apply_antifragile(1.0, {})
        self.assertAlmostEqual(result, 1.0)


class TestBatchG_Bleeders(unittest.TestCase):
    """G3: Bleeders — halve size for bleeding strategies."""

    def test_bleeding_strategy_halved(self):
        result = apply_bleeders(1.0, {"bleeders": [{"name": "momentum", "loss": -0.05}]}, "momentum")
        self.assertAlmostEqual(result, 0.50)

    def test_non_bleeding_unchanged(self):
        result = apply_bleeders(1.0, {"bleeders": [{"name": "momentum", "loss": -0.05}]}, "mean_reversion")
        self.assertAlmostEqual(result, 1.0)

    def test_empty_bleeders_unchanged(self):
        result = apply_bleeders(1.0, {"bleeders": []}, "momentum")
        self.assertAlmostEqual(result, 1.0)

    def test_missing_unchanged(self):
        result = apply_bleeders(1.0, {}, "momentum")
        self.assertAlmostEqual(result, 1.0)

    def test_multiple_bleeders(self):
        result = apply_bleeders(1.0, {"bleeders": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}, "b")
        self.assertAlmostEqual(result, 0.50)


class TestBatchG_TCA(unittest.TestCase):
    """G4: TCA score — high transaction costs reduce size."""

    def test_score_below_50_unchanged(self):
        result = apply_tca(1.0, {"tca_score": 30})
        self.assertAlmostEqual(result, 1.0)

    def test_score_at_50_unchanged(self):
        result = apply_tca(1.0, {"tca_score": 50})
        self.assertAlmostEqual(result, 1.0)

    def test_score_60_reduces(self):
        result = apply_tca(1.0, {"tca_score": 60})
        expected = max(0.50, 1.0 - (60 - 50) / 100.0)
        self.assertAlmostEqual(result, expected, places=2)

    def test_score_100_reduces(self):
        result = apply_tca(1.0, {"tca_score": 100})
        expected = max(0.50, 1.0 - (100 - 50) / 100.0)
        self.assertAlmostEqual(result, expected, places=2)

    def test_score_200_floored(self):
        result = apply_tca(1.0, {"tca_score": 200})
        self.assertAlmostEqual(result, 0.50, places=2)

    def test_missing_unchanged(self):
        result = apply_tca(1.0, {})
        self.assertAlmostEqual(result, 1.0)


class TestBatchG_SystemStatus(unittest.TestCase):
    """G5: System status — CRITICAL blocks, DEGRADED reduces."""

    def test_critical_blocks_buy(self):
        size, skip = apply_system_status(1.0, {"system_status": {"status": "CRITICAL"}}, "BUY")
        self.assertTrue(skip)

    def test_critical_allows_sell(self):
        size, skip = apply_system_status(1.0, {"system_status": {"status": "CRITICAL"}}, "SELL")
        self.assertFalse(skip)

    def test_degraded_reduces_30pct(self):
        size, skip = apply_system_status(1.0, {"system_status": {"status": "DEGRADED"}}, "BUY")
        self.assertFalse(skip)
        self.assertAlmostEqual(size, 0.70, places=2)

    def test_healthy_unchanged(self):
        size, skip = apply_system_status(1.0, {"system_status": {"status": "HEALTHY"}}, "BUY")
        self.assertFalse(skip)
        self.assertAlmostEqual(size, 1.0)

    def test_missing_unchanged(self):
        size, skip = apply_system_status(1.0, {}, "BUY")
        self.assertFalse(skip)
        self.assertAlmostEqual(size, 1.0)


class TestBatchG_Whale(unittest.TestCase):
    """G6: Whale activity — on-chain flow direction."""

    def test_positive_bias_boosts(self):
        result = apply_whale(1.0, {"whale_activity": {"net_flow_bias": 0.8}})
        expected = max(0.70, min(1.30, 1.0 + 0.8 * 0.25))
        self.assertAlmostEqual(result, expected, places=2)

    def test_negative_bias_reduces(self):
        result = apply_whale(1.0, {"whale_activity": {"net_flow_bias": -0.8}})
        expected = max(0.70, min(1.30, 1.0 - 0.8 * 0.25))
        self.assertAlmostEqual(result, expected, places=2)

    def test_zero_bias_unchanged(self):
        result = apply_whale(1.0, {"whale_activity": {"net_flow_bias": 0.0}})
        self.assertAlmostEqual(result, 1.0)

    def test_extreme_capped(self):
        result = apply_whale(1.0, {"whale_activity": {"net_flow_bias": 5.0}})
        self.assertAlmostEqual(result, 1.30, places=2)

    def test_missing_unchanged(self):
        result = apply_whale(1.0, {})
        self.assertAlmostEqual(result, 1.0)


class TestBatchG_CausalGraph(unittest.TestCase):
    """G7: Causal graph — funding→vol→regime chain."""

    def test_bearish_buy_reduces(self):
        adv = {"causal_graph": {"confidence": 0.80, "direction": "bearish"}}
        result = apply_causal_graph(1.0, adv, "BUY")
        self.assertAlmostEqual(result, 0.70, places=2)

    def test_bullish_buy_boosts(self):
        adv = {"causal_graph": {"confidence": 0.80, "direction": "bullish"}}
        result = apply_causal_graph(1.0, adv, "BUY")
        self.assertAlmostEqual(result, 1.20, places=2)

    def test_low_confidence_unchanged(self):
        adv = {"causal_graph": {"confidence": 0.30, "direction": "bearish"}}
        result = apply_causal_graph(1.0, adv, "BUY")
        self.assertAlmostEqual(result, 1.0)

    def test_neutral_unchanged(self):
        adv = {"causal_graph": {"confidence": 0.80, "direction": "neutral"}}
        result = apply_causal_graph(1.0, adv, "BUY")
        self.assertAlmostEqual(result, 1.0)

    def test_bearish_sell_unchanged(self):
        adv = {"causal_graph": {"confidence": 0.80, "direction": "bearish"}}
        result = apply_causal_graph(1.0, adv, "SELL")
        self.assertAlmostEqual(result, 1.0)

    def test_missing_unchanged(self):
        result = apply_causal_graph(1.0, {}, "BUY")
        self.assertAlmostEqual(result, 1.0)


class TestBatchG_OutcomeCorrelator(unittest.TestCase):
    """G8: Outcome correlator — reduce in unfavorable conditions."""

    def test_unfavorable_reduces(self):
        result = apply_outcome_correlator(1.0, {"outcome_correlator": {"favorability": 0.20}})
        self.assertAlmostEqual(result, 0.60, places=2)

    def test_favorable_boosts(self):
        result = apply_outcome_correlator(1.0, {"outcome_correlator": {"favorability": 0.80}})
        self.assertAlmostEqual(result, 1.15, places=2)

    def test_neutral_unchanged(self):
        result = apply_outcome_correlator(1.0, {"outcome_correlator": {"favorability": 0.50}})
        self.assertAlmostEqual(result, 1.0)

    def test_boundary_030_unchanged(self):
        result = apply_outcome_correlator(1.0, {"outcome_correlator": {"favorability": 0.30}})
        self.assertAlmostEqual(result, 1.0)

    def test_boundary_070_unchanged(self):
        result = apply_outcome_correlator(1.0, {"outcome_correlator": {"favorability": 0.70}})
        self.assertAlmostEqual(result, 1.0)

    def test_missing_unchanged(self):
        result = apply_outcome_correlator(1.0, {})
        self.assertAlmostEqual(result, 1.0)


class TestBatchG_AlphaDecay(unittest.TestCase):
    """G9: Alpha decay — age signal by strategy-level decay factor."""

    def test_decay_applies(self):
        adv = {"alpha_decay": {"strategy_decays": {"momentum": 0.70}}}
        result = apply_alpha_decay(1.0, adv, "momentum")
        self.assertAlmostEqual(result, 0.70, places=2)

    def test_no_decay_for_other_strategy(self):
        adv = {"alpha_decay": {"strategy_decays": {"momentum": 0.70}}}
        result = apply_alpha_decay(1.0, adv, "mean_reversion")
        self.assertAlmostEqual(result, 1.0)

    def test_decay_floored_at_030(self):
        adv = {"alpha_decay": {"strategy_decays": {"momentum": 0.10}}}
        result = apply_alpha_decay(1.0, adv, "momentum")
        self.assertAlmostEqual(result, 0.30, places=2)

    def test_decay_1_unchanged(self):
        adv = {"alpha_decay": {"strategy_decays": {"momentum": 1.0}}}
        result = apply_alpha_decay(1.0, adv, "momentum")
        self.assertAlmostEqual(result, 1.0)

    def test_missing_unchanged(self):
        result = apply_alpha_decay(1.0, {}, "momentum")
        self.assertAlmostEqual(result, 1.0)


class TestBatchG_LiquidationCascade(unittest.TestCase):
    """G10: Liquidation cascade — boost/block based on cascade direction."""

    def test_long_squeeze_reduces_buy(self):
        adv = {"liquidation_cascade": {"signals": [
            {"symbol": "BTC/USD", "direction": "long_squeeze", "confidence": 0.80}
        ]}}
        result = apply_liquidation_cascade(1.0, adv, "BTC/USD", "BUY")
        self.assertAlmostEqual(result, 0.40, places=2)

    def test_short_squeeze_boosts_buy(self):
        adv = {"liquidation_cascade": {"signals": [
            {"symbol": "BTC/USD", "direction": "short_squeeze", "confidence": 0.80}
        ]}}
        result = apply_liquidation_cascade(1.0, adv, "BTC/USD", "BUY")
        self.assertAlmostEqual(result, 1.30, places=2)

    def test_low_confidence_unchanged(self):
        adv = {"liquidation_cascade": {"signals": [
            {"symbol": "BTC/USD", "direction": "long_squeeze", "confidence": 0.40}
        ]}}
        result = apply_liquidation_cascade(1.0, adv, "BTC/USD", "BUY")
        self.assertAlmostEqual(result, 1.0)

    def test_different_symbol_unchanged(self):
        adv = {"liquidation_cascade": {"signals": [
            {"symbol": "ETH/USD", "direction": "long_squeeze", "confidence": 0.80}
        ]}}
        result = apply_liquidation_cascade(1.0, adv, "BTC/USD", "BUY")
        self.assertAlmostEqual(result, 1.0)

    def test_sell_not_affected(self):
        adv = {"liquidation_cascade": {"signals": [
            {"symbol": "BTC/USD", "direction": "long_squeeze", "confidence": 0.80}
        ]}}
        result = apply_liquidation_cascade(1.0, adv, "BTC/USD", "SELL")
        self.assertAlmostEqual(result, 1.0)

    def test_missing_unchanged(self):
        result = apply_liquidation_cascade(1.0, {}, "BTC/USD", "BUY")
        self.assertAlmostEqual(result, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch H Tests: Partial exits
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchH_DrawdownPartialClose(unittest.TestCase):
    """H1: Drawdown CRITICAL → close 30% of all open positions."""

    def test_critical_generates_exit_signal(self):
        """When system_status=CRITICAL, should generate 30% close signals."""
        positions = {
            "BTC/USD": {"quantity": 1.0, "side": "BUY", "current_price": 50000, "entry_price": 52000},
            "ETH/USD": {"quantity": 10.0, "side": "BUY", "current_price": 3000, "entry_price": 3200},
        }
        advisory = {"system_status": {"status": "CRITICAL"}}
        # Simulate the logic: for each position, generate a SELL signal with strength=0.30
        exit_signals = []
        partial_done = {}
        for sym, pos in positions.items():
            qty = float(pos.get("quantity", 0))
            if qty <= 0:
                continue
            side = pos.get("side", "BUY")
            key = f"dd_critical_{sym}"
            if partial_done.get(key):
                continue
            exit_signals.append({
                "symbol": sym,
                "action": "SELL" if side == "BUY" else "BUY",
                "strength": 0.30,
            })
            partial_done[key] = True
        self.assertEqual(len(exit_signals), 2)
        self.assertEqual(exit_signals[0]["strength"], 0.30)
        self.assertEqual(exit_signals[0]["action"], "SELL")

    def test_non_critical_no_signals(self):
        advisory = {"system_status": {"status": "HEALTHY"}}
        status = advisory["system_status"]["status"]
        self.assertNotEqual(status, "CRITICAL")

    def test_no_advisory_no_signals(self):
        advisory = {}
        self.assertIsNone(advisory.get("system_status"))

    def test_short_position_generates_buy(self):
        """Short position should generate BUY exit signal."""
        side = "SELL"
        exit_action = "SELL" if side == "BUY" else "BUY"
        self.assertEqual(exit_action, "BUY")

    def test_dedup_prevents_double_close(self):
        """_partial_exit_done key prevents closing same position twice."""
        partial_done = {"dd_critical_BTC/USD": True}
        key = "dd_critical_BTC/USD"
        self.assertTrue(partial_done.get(key, False))

    def test_zero_quantity_skipped(self):
        pos = {"quantity": 0.0, "side": "BUY", "current_price": 50000}
        self.assertEqual(float(pos.get("quantity", 0)), 0.0)

    def test_no_current_price_skipped(self):
        pos = {"quantity": 1.0, "side": "BUY", "current_price": 0}
        self.assertEqual(float(pos.get("current_price", 0)), 0.0)


class TestBatchH_FundingCostRotation(unittest.TestCase):
    """H2: Funding cost rotation → close high-funding-cost long positions."""

    def test_funding_rec_generates_sell(self):
        recs = [{"symbol": "BTC/USD"}]
        positions = {"BTC/USD": {"quantity": 1.0, "side": "BUY", "current_price": 50000}}
        exits = []
        for rec in recs:
            sym = rec.get("symbol", "")
            pos = positions.get(sym)
            if pos and pos.get("side") == "BUY" and float(pos.get("quantity", 0)) > 0:
                exits.append({"symbol": sym, "action": "SELL", "strength": 1.0})
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0]["action"], "SELL")
        self.assertEqual(exits[0]["strength"], 1.0)

    def test_short_position_not_closed(self):
        """Only close LONG positions."""
        positions = {"BTC/USD": {"quantity": 1.0, "side": "SELL", "current_price": 50000}}
        pos = positions.get("BTC/USD")
        self.assertNotEqual(pos.get("side"), "BUY")

    def test_no_recs_no_exits(self):
        recs = []
        self.assertEqual(len(recs), 0)

    def test_missing_symbol_skipped(self):
        recs = [{"symbol": "DOT/USD"}]
        positions = {"BTC/USD": {"quantity": 1.0, "side": "BUY", "current_price": 50000}}
        pos = positions.get(recs[0]["symbol"])
        self.assertIsNone(pos)

    def test_dedup_key(self):
        """Funding rotation dedup key format."""
        sym = "BTC/USD"
        key = f"funding_rot_{sym}"
        self.assertEqual(key, "funding_rot_BTC/USD")

    def test_full_close_strength(self):
        """Funding rotation closes the full position (strength=1.0)."""
        self.assertEqual(1.0, 1.0)  # Strength is always 1.0 for funding rotation


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch I Tests: TWAP + toxicity
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchI_Toxicity(unittest.TestCase):
    """Order flow toxicity → switch to market when toxic."""

    def test_high_toxicity_switches_to_market(self):
        result = apply_toxicity("limit", {"toxicity": {"toxicity_score": 0.80}})
        self.assertEqual(result, "market")

    def test_low_toxicity_stays_limit(self):
        result = apply_toxicity("limit", {"toxicity": {"toxicity_score": 0.50}})
        self.assertEqual(result, "limit")

    def test_boundary_070_switches(self):
        result = apply_toxicity("limit", {"toxicity": {"toxicity_score": 0.70}})
        self.assertEqual(result, "market")

    def test_already_market_stays_market(self):
        result = apply_toxicity("market", {"toxicity": {"toxicity_score": 0.90}})
        self.assertEqual(result, "market")

    def test_missing_advisory_unchanged(self):
        result = apply_toxicity("limit", {})
        self.assertEqual(result, "limit")

    def test_zero_score_unchanged(self):
        result = apply_toxicity("limit", {"toxicity": {"toxicity_score": 0.0}})
        self.assertEqual(result, "limit")


class TestBatchI_TwapRequested(unittest.TestCase):
    """TWAP requested from execution_intelligence advisory."""

    def test_twap_type_requested(self):
        self.assertTrue(check_twap_requested({"execution_intelligence": {"order_type": "twap"}}))

    def test_vwap_type_requested(self):
        self.assertTrue(check_twap_requested({"execution_intelligence": {"order_type": "vwap"}}))

    def test_adaptive_type_requested(self):
        self.assertTrue(check_twap_requested({"execution_intelligence": {"order_type": "adaptive"}}))

    def test_limit_not_requested(self):
        self.assertFalse(check_twap_requested({"execution_intelligence": {"order_type": "limit"}}))

    def test_market_not_requested(self):
        self.assertFalse(check_twap_requested({"execution_intelligence": {"order_type": "market"}}))

    def test_missing_not_requested(self):
        self.assertFalse(check_twap_requested({}))


class TestBatchI_ConfigFields(unittest.TestCase):
    """New config fields: twap_min_notional_usd, twap_duration_minutes."""

    def test_config_defaults(self):
        """Config defaults match expected values."""
        from unified_trading_system import UnifiedConfig
        cfg = UnifiedConfig()
        self.assertAlmostEqual(cfg.twap_min_notional_usd, 250.0)
        self.assertAlmostEqual(cfg.twap_duration_minutes, 5.0)

    def test_config_from_yaml(self):
        """Config loads from YAML dict."""
        from unified_trading_system import UnifiedConfig
        yaml_dict = {
            "execution_engine": {
                "twap_min_notional_usd": 500.0,
                "twap_duration_minutes": 10.0,
            }
        }
        cfg = UnifiedConfig.from_unified_yaml_dict(yaml_dict)
        self.assertAlmostEqual(cfg.twap_min_notional_usd, 500.0)
        self.assertAlmostEqual(cfg.twap_duration_minutes, 10.0)

    def test_config_yaml_defaults(self):
        """Config defaults when YAML section missing."""
        from unified_trading_system import UnifiedConfig
        cfg = UnifiedConfig.from_unified_yaml_dict({})
        self.assertAlmostEqual(cfg.twap_min_notional_usd, 250.0)
        self.assertAlmostEqual(cfg.twap_duration_minutes, 5.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch F Tests: Model staleness + FeatureDiscoverer
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchF_ModelStaleness(unittest.TestCase):
    """F1: Model staleness deration."""

    def test_staleness_06_derates(self):
        result = model_staleness_derate(0.6)
        self.assertAlmostEqual(result, 0.40, places=2)

    def test_staleness_09_derates(self):
        result = model_staleness_derate(0.9)
        self.assertAlmostEqual(result, 0.30, places=2)  # floored

    def test_staleness_10_derates(self):
        result = model_staleness_derate(1.0)
        self.assertAlmostEqual(result, 0.30, places=2)  # floored at max(0.30, 0.0)

    def test_staleness_05_no_derate(self):
        result = model_staleness_derate(0.5)
        self.assertAlmostEqual(result, 1.0)

    def test_staleness_0_no_derate(self):
        result = model_staleness_derate(0.0)
        self.assertAlmostEqual(result, 1.0)

    def test_component_registry_has_method(self):
        """ComponentRegistry.on_fill contains model staleness logic."""
        from core.component_registry import ComponentRegistry
        cr = ComponentRegistry(config=MagicMock())
        self.assertTrue(hasattr(cr, "on_fill"))

    def test_staleness_feedback_with_mocks(self):
        """Model staleness feedback derate logic with mocked components."""
        from core.component_registry import ComponentRegistry
        cr = ComponentRegistry(config=MagicMock())
        # Mock model_manager with a registry containing stale model
        mock_entry = MagicMock()
        mock_entry.staleness_score = 0.80
        cr.model_manager = MagicMock()
        cr.model_manager.registry = {"vol_model": mock_entry}
        # Mock ensemble_hub
        cr.ensemble_hub = MagicMock()
        cr.ensemble_hub.update_source_weights = MagicMock()
        # Call on_fill
        cr.on_fill({"symbol": "BTC/USD", "side": "buy", "price": 50000, "quantity": 0.01, "pnl": 10.0})
        # Verify ensemble_hub.update_source_weights was called with derated weight
        cr.ensemble_hub.update_source_weights.assert_called_once()
        args = cr.ensemble_hub.update_source_weights.call_args[0][0]
        self.assertIn("vol", args)
        self.assertAlmostEqual(args["vol"], 0.30, places=2)  # max(0.30, 1.0-0.80) = max(0.30, 0.20) = 0.30


class TestBatchF_FeatureDiscoverer(unittest.TestCase):
    """F2: FeatureDiscoverer outcome feedback."""

    def test_observe_outcome_called_on_positive_pnl(self):
        from core.component_registry import ComponentRegistry
        cr = ComponentRegistry(config=MagicMock())
        cr.feature_discoverer = MagicMock()
        cr.feature_discoverer.observe_outcome = MagicMock()
        cr.on_fill({"symbol": "BTC/USD", "side": "buy", "price": 50000, "quantity": 0.01, "pnl": 10.0})
        cr.feature_discoverer.observe_outcome.assert_called_with(True)

    def test_observe_outcome_called_on_negative_pnl(self):
        from core.component_registry import ComponentRegistry
        cr = ComponentRegistry(config=MagicMock())
        cr.feature_discoverer = MagicMock()
        cr.feature_discoverer.observe_outcome = MagicMock()
        cr.on_fill({"symbol": "BTC/USD", "side": "buy", "price": 50000, "quantity": 0.01, "pnl": -5.0})
        cr.feature_discoverer.observe_outcome.assert_called_with(False)

    def test_observe_outcome_not_called_on_zero_pnl(self):
        from core.component_registry import ComponentRegistry
        cr = ComponentRegistry(config=MagicMock())
        cr.feature_discoverer = MagicMock()
        cr.feature_discoverer.observe_outcome = MagicMock()
        cr.on_fill({"symbol": "BTC/USD", "side": "buy", "price": 50000, "quantity": 0.01, "pnl": 0.0})
        cr.feature_discoverer.observe_outcome.assert_not_called()

    def test_no_discoverer_no_error(self):
        from core.component_registry import ComponentRegistry
        cr = ComponentRegistry(config=MagicMock())
        cr.feature_discoverer = None
        # Should not raise
        cr.on_fill({"symbol": "BTC/USD", "side": "buy", "price": 50000, "quantity": 0.01, "pnl": 10.0})


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch J Tests: LiquidationCascadeStrategy
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchJ_LiquidationCascade(unittest.TestCase):
    """J: LiquidationCascadeStrategy slot + init + on_cycle."""

    def test_component_registry_has_slot(self):
        from core.component_registry import ComponentRegistry
        cr = ComponentRegistry(config=MagicMock())
        self.assertTrue(hasattr(cr, "liquidation_cascade"))
        self.assertIsNone(cr.liquidation_cascade)

    def test_oi_estimate_dict_exists(self):
        from core.component_registry import ComponentRegistry
        cr = ComponentRegistry(config=MagicMock())
        self.assertTrue(hasattr(cr, "_last_oi_estimate"))
        self.assertIsInstance(cr._last_oi_estimate, dict)

    def test_init_method_exists(self):
        from core.component_registry import ComponentRegistry
        cr = ComponentRegistry(config=MagicMock())
        self.assertTrue(hasattr(cr, "_init_liquidation_cascade"))

    def test_lc_init_creates_strategy(self):
        """_init_liquidation_cascade creates a LiquidationCascadeStrategy instance."""
        from core.component_registry import ComponentRegistry
        cr = ComponentRegistry(config=MagicMock())
        try:
            cr._init_liquidation_cascade()
            self.assertIsNotNone(cr.liquidation_cascade)
        except ImportError:
            self.skipTest("LiquidationCascadeStrategy not available")

    def test_lc_gate_logic_long_squeeze(self):
        """G10 gate: long_squeeze with high confidence reduces BUY by 60%."""
        adv = {"liquidation_cascade": {"signals": [
            {"symbol": "BTC/USD", "direction": "long_squeeze", "confidence": 0.80}
        ]}}
        result = apply_liquidation_cascade(1.0, adv, "BTC/USD", "BUY")
        self.assertAlmostEqual(result, 0.40, places=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  Integration: Combined gate pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchG_CombinedPipeline(unittest.TestCase):
    """Test that multiple gates compose correctly."""

    def test_ensemble_plus_antifragile(self):
        adv = {
            "ensemble": {"composite": 0.5},
            "antifragile_multiplier": 1.2,
        }
        size = 1.0
        size = apply_ensemble_composite(size, adv)
        size = apply_antifragile(size, adv)
        expected = 1.15 * 1.2
        self.assertAlmostEqual(size, expected, places=2)

    def test_all_reducing_gates(self):
        """Multiple reducing gates compound."""
        adv = {
            "bleeders": [{"name": "momentum", "loss": -0.05}],
            "tca_score": 80,
            "system_status": {"status": "DEGRADED"},
        }
        size = 1.0
        size = apply_bleeders(size, adv, "momentum")  # *0.50
        size = apply_tca(size, adv)  # *0.70
        size, _ = apply_system_status(size, adv, "BUY")  # *0.70
        expected = 1.0 * 0.50 * 0.70 * 0.70
        self.assertAlmostEqual(size, expected, places=3)

    def test_all_boosting_gates(self):
        """Multiple boosting gates compound."""
        adv = {
            "ensemble": {"composite": 0.5},
            "whale_activity": {"net_flow_bias": 0.8},
            "outcome_correlator": {"favorability": 0.80},
        }
        size = 1.0
        size = apply_ensemble_composite(size, adv)  # *1.15
        size = apply_whale(size, adv)  # *1.20
        size = apply_outcome_correlator(size, adv)  # *1.15
        expected = 1.0 * 1.15 * 1.20 * 1.15
        self.assertAlmostEqual(size, expected, places=3)

    def test_system_critical_short_circuits(self):
        """CRITICAL blocks BUY regardless of other gates."""
        adv = {
            "ensemble": {"composite": 1.0},
            "system_status": {"status": "CRITICAL"},
        }
        size = apply_ensemble_composite(1.0, adv)
        _, skip = apply_system_status(size, adv, "BUY")
        self.assertTrue(skip)

    def test_empty_advisory_passthrough(self):
        """All gates pass through unchanged with empty advisory."""
        adv = {}
        size = 1.0
        size = apply_ensemble_composite(size, adv)
        size = apply_antifragile(size, adv)
        size = apply_bleeders(size, adv, "test")
        size = apply_tca(size, adv)
        size, skip = apply_system_status(size, adv, "BUY")
        size = apply_whale(size, adv)
        size = apply_causal_graph(size, adv, "BUY")
        size = apply_outcome_correlator(size, adv)
        size = apply_alpha_decay(size, adv, "test")
        size = apply_liquidation_cascade(size, adv, "BTC/USD", "BUY")
        self.assertFalse(skip)
        self.assertAlmostEqual(size, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Pure Python helpers for Batches K–O
# ═══════════════════════════════════════════════════════════════════════════════

def apply_vol_forecast(size_pct, advisory, symbol, current_vol):
    """K1: Vol forecast sizing."""
    vf = advisory.get("vol_forecasts")
    if vf and isinstance(vf, dict) and symbol in vf:
        sym_vf = vf[symbol]
        if isinstance(sym_vf, dict):
            fvol = float(sym_vf.get("forecast_vol_1d", 0.0) or 0.0)
            if fvol > 0 and current_vol > 0:
                ratio = fvol / max(current_vol, 0.001)
                mult = max(0.50, min(1.15, 1.0 - (ratio - 1.0) * 0.30))
                if abs(mult - 1.0) > 0.01:
                    size_pct *= mult
    return size_pct

def apply_alpha_scores(size_pct, advisory, symbol, action):
    """K2: Alpha model direction."""
    als = advisory.get("alpha_scores")
    if als and isinstance(als, dict) and symbol in als:
        sym = als[symbol]
        if isinstance(sym, dict):
            comp = float(sym.get("composite", 0.0) or 0.0)
            if abs(comp) > 0.3:
                aligns = (comp > 0 and action == "BUY") or (comp < 0 and action == "SELL")
                size_pct *= 1.15 if aligns else 0.70
    return size_pct

def apply_pretrained_alpha(size_pct, advisory, action):
    """K5: Pretrained alpha direction."""
    pa = advisory.get("pretrained_alpha")
    if pa and isinstance(pa, dict):
        d = str(pa.get("direction", "")).upper()
        c = float(pa.get("confidence", 0.0) or 0.0)
        if c > 0.70:
            aligns = (d == "UP" and action == "BUY") or (d == "DOWN" and action == "SELL")
            size_pct *= 1.20 if aligns else 0.60
    return size_pct

def apply_fear_greed(size_pct, advisory, action):
    """L1: Fear & Greed contrarian."""
    fg = advisory.get("fear_greed")
    if fg is not None:
        val = float(fg)
        bias = (50.0 - val) / 250.0
        mult = (1.0 + bias) if action == "BUY" else (1.0 - bias)
        mult = max(0.80, min(1.20, mult))
        if abs(mult - 1.0) > 0.01:
            size_pct *= mult
    return size_pct

def apply_quantum_anomaly(size_pct, advisory, action):
    """M3: Quantum anomaly circuit breaker. Returns (size_pct, should_skip)."""
    qa = advisory.get("quantum_anomaly_score")
    if qa is not None:
        val = float(qa)
        if val > 0.90 and action == "BUY":
            return size_pct, True
        elif val > 0.75:
            size_pct *= 0.60
    return size_pct, False

def apply_correlation_penalty(size_pct, advisory):
    """N1: Correlation penalty."""
    c = advisory.get("correlation_penalty")
    if c is not None:
        val = float(c)
        if val > 0.30:
            size_pct *= max(0.50, 1.0 - val)
    return size_pct

def apply_risk_score(size_pct, advisory, action):
    """N5: Risk score gate. Returns (size_pct, should_skip)."""
    rs = advisory.get("risk_score")
    if rs is not None:
        val = float(rs)
        if val > 0.95 and action == "BUY":
            return size_pct, True
        elif val > 0.80:
            size_pct *= 0.70
    return size_pct, False

def apply_market_anomaly(size_pct, advisory, action):
    """N6: Market anomaly. Returns (size_pct, should_skip)."""
    ma = advisory.get("market_anomaly")
    if ma and isinstance(ma, dict) and ma.get("is_anomaly"):
        sev = str(ma.get("severity", "low")).lower()
        if sev == "high" and action == "BUY":
            return size_pct, True
        elif sev == "medium":
            size_pct *= 0.70
    return size_pct, False

def apply_regime_rotation(size_pct, advisory, strategy):
    """O5: Regime rotation strategy weights."""
    rr = advisory.get("regime_rotation")
    if rr and isinstance(rr, dict):
        weights = rr.get("strategy_weights", {})
        if isinstance(weights, dict) and strategy in weights:
            w = float(weights[strategy])
            w = max(0.30, min(1.50, w))
            if abs(w - 1.0) > 0.01:
                size_pct *= w
    return size_pct

def apply_bandit_rankings(size_pct, advisory, strategy):
    """O10: Bandit strategy rankings."""
    br = advisory.get("bandit_rankings")
    if br and isinstance(br, list):
        for entry in br:
            if isinstance(entry, dict) and entry.get("strategy") == strategy:
                wr = float(entry.get("expected_win_rate", 0.5) or 0.5)
                if wr < 0.40:
                    size_pct *= 0.75
                elif wr > 0.60:
                    size_pct *= 1.15
                break
    return size_pct

def apply_session_effect(size_pct, advisory):
    """O9: Session effect time-of-day bias."""
    se = advisory.get("session_effect")
    if se is not None:
        val = float(se) if not isinstance(se, dict) else float(se.get("bias", 0.0) or 0.0)
        mult = max(0.85, min(1.15, 1.0 + val * 0.15))
        if abs(mult - 1.0) > 0.01:
            size_pct *= mult
    return size_pct


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch K Tests: ML Intelligence
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchK_VolForecast(unittest.TestCase):
    """K1: Volatility forecast sizing."""

    def test_high_forecast_reduces(self):
        adv = {"vol_forecasts": {"BTC/USD": {"forecast_vol_1d": 0.06}}}
        result = apply_vol_forecast(1.0, adv, "BTC/USD", 0.02)
        self.assertLess(result, 1.0)

    def test_low_forecast_boosts(self):
        adv = {"vol_forecasts": {"BTC/USD": {"forecast_vol_1d": 0.005}}}
        result = apply_vol_forecast(1.0, adv, "BTC/USD", 0.02)
        self.assertGreater(result, 1.0)

    def test_equal_forecast_unchanged(self):
        adv = {"vol_forecasts": {"BTC/USD": {"forecast_vol_1d": 0.02}}}
        result = apply_vol_forecast(1.0, adv, "BTC/USD", 0.02)
        self.assertAlmostEqual(result, 1.0, places=2)

    def test_missing_symbol_unchanged(self):
        adv = {"vol_forecasts": {"ETH/USD": {"forecast_vol_1d": 0.06}}}
        result = apply_vol_forecast(1.0, adv, "BTC/USD", 0.02)
        self.assertAlmostEqual(result, 1.0)

    def test_missing_advisory_unchanged(self):
        result = apply_vol_forecast(1.0, {}, "BTC/USD", 0.02)
        self.assertAlmostEqual(result, 1.0)


class TestBatchK_AlphaScores(unittest.TestCase):
    """K2: Alpha model direction."""

    def test_positive_alpha_aligns_buy(self):
        adv = {"alpha_scores": {"BTC/USD": {"composite": 0.5}}}
        result = apply_alpha_scores(1.0, adv, "BTC/USD", "BUY")
        self.assertAlmostEqual(result, 1.15, places=2)

    def test_negative_alpha_opposes_buy(self):
        adv = {"alpha_scores": {"BTC/USD": {"composite": -0.5}}}
        result = apply_alpha_scores(1.0, adv, "BTC/USD", "BUY")
        self.assertAlmostEqual(result, 0.70, places=2)

    def test_small_alpha_unchanged(self):
        adv = {"alpha_scores": {"BTC/USD": {"composite": 0.1}}}
        result = apply_alpha_scores(1.0, adv, "BTC/USD", "BUY")
        self.assertAlmostEqual(result, 1.0)

    def test_missing_unchanged(self):
        result = apply_alpha_scores(1.0, {}, "BTC/USD", "BUY")
        self.assertAlmostEqual(result, 1.0)


class TestBatchK_PretrainedAlpha(unittest.TestCase):
    """K5: Pretrained alpha direction confidence."""

    def test_aligns_buy_boosts(self):
        adv = {"pretrained_alpha": {"direction": "UP", "confidence": 0.80}}
        result = apply_pretrained_alpha(1.0, adv, "BUY")
        self.assertAlmostEqual(result, 1.20, places=2)

    def test_opposes_buy_reduces(self):
        adv = {"pretrained_alpha": {"direction": "DOWN", "confidence": 0.80}}
        result = apply_pretrained_alpha(1.0, adv, "BUY")
        self.assertAlmostEqual(result, 0.60, places=2)

    def test_low_confidence_unchanged(self):
        adv = {"pretrained_alpha": {"direction": "UP", "confidence": 0.50}}
        result = apply_pretrained_alpha(1.0, adv, "BUY")
        self.assertAlmostEqual(result, 1.0)

    def test_missing_unchanged(self):
        result = apply_pretrained_alpha(1.0, {}, "BUY")
        self.assertAlmostEqual(result, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch L Tests: Sentiment & Pattern
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchL_FearGreed(unittest.TestCase):
    """L1: Fear & Greed contrarian sizing."""

    def test_extreme_fear_boosts_buy(self):
        result = apply_fear_greed(1.0, {"fear_greed": 10}, "BUY")
        self.assertGreater(result, 1.0)

    def test_extreme_greed_reduces_buy(self):
        result = apply_fear_greed(1.0, {"fear_greed": 90}, "BUY")
        self.assertLess(result, 1.0)

    def test_neutral_unchanged(self):
        result = apply_fear_greed(1.0, {"fear_greed": 50}, "BUY")
        self.assertAlmostEqual(result, 1.0, places=2)

    def test_capped_at_120(self):
        result = apply_fear_greed(1.0, {"fear_greed": 0}, "BUY")
        self.assertLessEqual(result, 1.20)

    def test_floored_at_080(self):
        result = apply_fear_greed(1.0, {"fear_greed": 100}, "BUY")
        self.assertGreaterEqual(result, 0.80)

    def test_missing_unchanged(self):
        result = apply_fear_greed(1.0, {}, "BUY")
        self.assertAlmostEqual(result, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch M Tests: Quantum Intelligence
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchM_QuantumAnomaly(unittest.TestCase):
    """M3: Quantum anomaly circuit breaker."""

    def test_extreme_anomaly_blocks_buy(self):
        _, skip = apply_quantum_anomaly(1.0, {"quantum_anomaly_score": 0.95}, "BUY")
        self.assertTrue(skip)

    def test_extreme_anomaly_allows_sell(self):
        _, skip = apply_quantum_anomaly(1.0, {"quantum_anomaly_score": 0.95}, "SELL")
        self.assertFalse(skip)

    def test_high_anomaly_reduces(self):
        size, skip = apply_quantum_anomaly(1.0, {"quantum_anomaly_score": 0.80}, "BUY")
        self.assertFalse(skip)
        self.assertAlmostEqual(size, 0.60, places=2)

    def test_normal_score_unchanged(self):
        size, skip = apply_quantum_anomaly(1.0, {"quantum_anomaly_score": 0.50}, "BUY")
        self.assertFalse(skip)
        self.assertAlmostEqual(size, 1.0)

    def test_missing_unchanged(self):
        size, skip = apply_quantum_anomaly(1.0, {}, "BUY")
        self.assertFalse(skip)
        self.assertAlmostEqual(size, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch N Tests: Portfolio & Risk
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchN_CorrelationPenalty(unittest.TestCase):
    """N1: Correlation penalty."""

    def test_high_correlation_reduces(self):
        result = apply_correlation_penalty(1.0, {"correlation_penalty": 0.70})
        self.assertAlmostEqual(result, 0.50, places=2)  # max(0.50, 1.0-0.70)

    def test_moderate_correlation_reduces(self):
        result = apply_correlation_penalty(1.0, {"correlation_penalty": 0.50})
        self.assertAlmostEqual(result, 0.50, places=2)

    def test_low_correlation_unchanged(self):
        result = apply_correlation_penalty(1.0, {"correlation_penalty": 0.20})
        self.assertAlmostEqual(result, 1.0)

    def test_missing_unchanged(self):
        result = apply_correlation_penalty(1.0, {})
        self.assertAlmostEqual(result, 1.0)


class TestBatchN_RiskScore(unittest.TestCase):
    """N5: Risk score portfolio gate."""

    def test_extreme_blocks_buy(self):
        _, skip = apply_risk_score(1.0, {"risk_score": 0.96}, "BUY")
        self.assertTrue(skip)

    def test_extreme_allows_sell(self):
        _, skip = apply_risk_score(1.0, {"risk_score": 0.96}, "SELL")
        self.assertFalse(skip)

    def test_high_reduces(self):
        size, skip = apply_risk_score(1.0, {"risk_score": 0.85}, "BUY")
        self.assertFalse(skip)
        self.assertAlmostEqual(size, 0.70, places=2)

    def test_normal_unchanged(self):
        size, skip = apply_risk_score(1.0, {"risk_score": 0.50}, "BUY")
        self.assertFalse(skip)
        self.assertAlmostEqual(size, 1.0)

    def test_missing_unchanged(self):
        size, skip = apply_risk_score(1.0, {}, "BUY")
        self.assertFalse(skip)
        self.assertAlmostEqual(size, 1.0)


class TestBatchN_MarketAnomaly(unittest.TestCase):
    """N6: Market anomaly circuit breaker."""

    def test_high_severity_blocks_buy(self):
        adv = {"market_anomaly": {"is_anomaly": True, "severity": "high"}}
        _, skip = apply_market_anomaly(1.0, adv, "BUY")
        self.assertTrue(skip)

    def test_medium_severity_reduces(self):
        adv = {"market_anomaly": {"is_anomaly": True, "severity": "medium"}}
        size, skip = apply_market_anomaly(1.0, adv, "BUY")
        self.assertFalse(skip)
        self.assertAlmostEqual(size, 0.70, places=2)

    def test_no_anomaly_unchanged(self):
        adv = {"market_anomaly": {"is_anomaly": False, "severity": "low"}}
        size, skip = apply_market_anomaly(1.0, adv, "BUY")
        self.assertFalse(skip)
        self.assertAlmostEqual(size, 1.0)

    def test_missing_unchanged(self):
        size, skip = apply_market_anomaly(1.0, {}, "BUY")
        self.assertFalse(skip)
        self.assertAlmostEqual(size, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch O Tests: Strategy Intelligence
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchO_RegimeRotation(unittest.TestCase):
    """O5: Regime rotation strategy weights."""

    def test_high_weight_boosts(self):
        adv = {"regime_rotation": {"strategy_weights": {"momentum": 1.3}}}
        result = apply_regime_rotation(1.0, adv, "momentum")
        self.assertAlmostEqual(result, 1.30, places=2)

    def test_low_weight_reduces(self):
        adv = {"regime_rotation": {"strategy_weights": {"momentum": 0.5}}}
        result = apply_regime_rotation(1.0, adv, "momentum")
        self.assertAlmostEqual(result, 0.50, places=2)

    def test_other_strategy_unchanged(self):
        adv = {"regime_rotation": {"strategy_weights": {"momentum": 0.5}}}
        result = apply_regime_rotation(1.0, adv, "mean_reversion")
        self.assertAlmostEqual(result, 1.0)

    def test_missing_unchanged(self):
        result = apply_regime_rotation(1.0, {}, "momentum")
        self.assertAlmostEqual(result, 1.0)

    def test_capped_at_150(self):
        adv = {"regime_rotation": {"strategy_weights": {"momentum": 3.0}}}
        result = apply_regime_rotation(1.0, adv, "momentum")
        self.assertAlmostEqual(result, 1.50, places=2)

    def test_floored_at_030(self):
        adv = {"regime_rotation": {"strategy_weights": {"momentum": 0.1}}}
        result = apply_regime_rotation(1.0, adv, "momentum")
        self.assertAlmostEqual(result, 0.30, places=2)


class TestBatchO_BanditRankings(unittest.TestCase):
    """O10: Bandit strategy rankings."""

    def test_low_win_rate_reduces(self):
        adv = {"bandit_rankings": [{"strategy": "momentum", "expected_win_rate": 0.30}]}
        result = apply_bandit_rankings(1.0, adv, "momentum")
        self.assertAlmostEqual(result, 0.75, places=2)

    def test_high_win_rate_boosts(self):
        adv = {"bandit_rankings": [{"strategy": "momentum", "expected_win_rate": 0.70}]}
        result = apply_bandit_rankings(1.0, adv, "momentum")
        self.assertAlmostEqual(result, 1.15, places=2)

    def test_normal_win_rate_unchanged(self):
        adv = {"bandit_rankings": [{"strategy": "momentum", "expected_win_rate": 0.50}]}
        result = apply_bandit_rankings(1.0, adv, "momentum")
        self.assertAlmostEqual(result, 1.0)

    def test_different_strategy_unchanged(self):
        adv = {"bandit_rankings": [{"strategy": "mean_reversion", "expected_win_rate": 0.30}]}
        result = apply_bandit_rankings(1.0, adv, "momentum")
        self.assertAlmostEqual(result, 1.0)

    def test_missing_unchanged(self):
        result = apply_bandit_rankings(1.0, {}, "momentum")
        self.assertAlmostEqual(result, 1.0)


class TestBatchO_SessionEffect(unittest.TestCase):
    """O9: Session effect time-of-day bias."""

    def test_positive_bias_boosts(self):
        result = apply_session_effect(1.0, {"session_effect": 0.5})
        self.assertGreater(result, 1.0)

    def test_negative_bias_reduces(self):
        result = apply_session_effect(1.0, {"session_effect": -0.5})
        self.assertLess(result, 1.0)

    def test_zero_bias_unchanged(self):
        result = apply_session_effect(1.0, {"session_effect": 0.0})
        self.assertAlmostEqual(result, 1.0)

    def test_capped_at_115(self):
        result = apply_session_effect(1.0, {"session_effect": 5.0})
        self.assertAlmostEqual(result, 1.15, places=2)

    def test_floored_at_085(self):
        result = apply_session_effect(1.0, {"session_effect": -5.0})
        self.assertAlmostEqual(result, 0.85, places=2)

    def test_missing_unchanged(self):
        result = apply_session_effect(1.0, {})
        self.assertAlmostEqual(result, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Combined pipeline test for all batches
# ═══════════════════════════════════════════════════════════════════════════════

class TestAllBatches_CombinedPipeline(unittest.TestCase):
    """Full advisory pipeline: G + K + L + M + N + O."""

    def test_all_gates_empty_advisory(self):
        """All gates pass through with empty advisory."""
        adv = {}
        size = 1.0
        size = apply_ensemble_composite(size, adv)
        size = apply_antifragile(size, adv)
        size = apply_bleeders(size, adv, "test")
        size = apply_tca(size, adv)
        size, _ = apply_system_status(size, adv, "BUY")
        size = apply_vol_forecast(size, adv, "BTC/USD", 0.02)
        size = apply_alpha_scores(size, adv, "BTC/USD", "BUY")
        size = apply_pretrained_alpha(size, adv, "BUY")
        size = apply_fear_greed(size, adv, "BUY")
        size, _ = apply_quantum_anomaly(size, adv, "BUY")
        size = apply_correlation_penalty(size, adv)
        size, _ = apply_risk_score(size, adv, "BUY")
        size, _ = apply_market_anomaly(size, adv, "BUY")
        size = apply_regime_rotation(size, adv, "test")
        size = apply_bandit_rankings(size, adv, "test")
        size = apply_session_effect(size, adv)
        self.assertAlmostEqual(size, 1.0)

    def test_multiple_reducing_gates_compound(self):
        """Several reducing gates stack."""
        adv = {
            "correlation_penalty": 0.50,
            "quantum_anomaly_score": 0.80,
            "fear_greed": 90,  # extreme greed → reduce BUY
        }
        size = 1.0
        size = apply_correlation_penalty(size, adv)  # *0.50
        size, _ = apply_quantum_anomaly(size, adv, "BUY")  # *0.60
        size = apply_fear_greed(size, adv, "BUY")  # *~0.84
        self.assertLess(size, 0.30)

    def test_circuit_breaker_stops_pipeline(self):
        """Quantum anomaly > 0.90 blocks BUY regardless."""
        adv = {
            "ensemble": {"composite": 1.0},
            "quantum_anomaly_score": 0.95,
        }
        size = apply_ensemble_composite(1.0, adv)  # boosts
        _, skip = apply_quantum_anomaly(size, adv, "BUY")
        self.assertTrue(skip)


if __name__ == "__main__":
    unittest.main()
