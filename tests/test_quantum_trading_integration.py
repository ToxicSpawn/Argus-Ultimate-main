"""
End-to-end trading integration tests for the quantum overhaul.

Phase D5 (critical — addresses risk R5: no integration test exists for the
``_execute_signals`` size flow before the Phase C wiring).

These tests verify that:
- Quantum VaR (Phase C3) actually halves order size on tail risk
- Quantum signal classifier (Phase C3) applies continuous quality scaling
- Quantum portfolio weights (Phase C4) are consumed in size_pct
- Quantum annealer strategy mask (Phase C5) blocks deselected strategies

Without these tests, regressions to the trading-decision wiring would be silent.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pytest


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _make_config(**overrides) -> SimpleNamespace:
    cfg = SimpleNamespace(
        starting_capital_aud=1000.0,
        aud_to_usd=0.65,
        trading_pairs=["BTC/USD", "ETH/USD"],
        initial_price_history=None,
        compliance_wash_sale_block=False,
        use_quantum_var_position_cap=True,
        max_tail_loss_fraction=0.05,
        use_quantum_portfolio_weights=True,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_registry(cfg=None):
    from core.component_registry import ComponentRegistry
    return ComponentRegistry(cfg or _make_config())


# ═════════════════════════════════════════════════════════════════════════════
# Phase C3 — Quantum VaR continuous mapping in pre_order_check
# ═════════════════════════════════════════════════════════════════════════════


class TestQuantumVaRSizeFlow:
    """Verify Phase C3 quantum VaR gate halves size on severe tail risk."""

    def test_severe_tail_risk_halves_size_factor(self):
        reg = _make_registry()
        reg._quantum_risk_engine = MagicMock()
        reg._quantum_risk_engine.quantum_var.return_value = {
            "var": -0.07,
            "cvar": -0.08,
            "method": "sobol_importance_sampling",
            "n_paths": 5000,
        }
        reg._price_history = {
            "BTC/USD": list(np.linspace(40000, 41000, 50)),
        }

        result = reg.pre_order_check("BTC/USD", "buy", 500.0)
        # New continuous mapping: exp(12 * -0.08) = exp(-0.96) = 0.383
        # Plus the position cap may reduce further
        assert result["size_factor"] < 0.5, (
            f"size_factor={result['size_factor']:.3f} should be < 0.5 on severe risk"
        )
        assert any("QuantumVaR" in r for r in result["reasons"])

    def test_var_cap_applied_when_flag_enabled(self):
        cfg = _make_config(
            use_quantum_var_position_cap=True,
            max_tail_loss_fraction=0.02,  # tight 2% cap
            starting_capital_aud=1000.0,
        )
        reg = _make_registry(cfg)
        reg._quantum_risk_engine = MagicMock()
        reg._quantum_risk_engine.quantum_var.return_value = {
            "var": -0.10,
            "cvar": -0.15,  # 15% CVaR loss → would violate 2% cap on $500
            "method": "sobol",
            "n_paths": 5000,
        }
        reg._price_history = {
            "BTC/USD": list(np.linspace(40000, 41000, 50)),
        }

        result = reg.pre_order_check("BTC/USD", "buy", 500.0)
        # Cap factor: max_tail_loss_aud=$20, denom=$500*0.15=$75 → cap=20/75=0.27
        # Plus the continuous mapping factor
        assert result["size_factor"] < 0.3
        # Look for cap reason
        cap_reasons = [r for r in result["reasons"] if "QuantumVaR cap" in r]
        assert len(cap_reasons) >= 1, "Expected QuantumVaR cap reason"

    def test_no_var_effect_in_deadband(self):
        """CVaR within ±0.02 should not trigger any QuantumVaR effect."""
        reg = _make_registry()
        reg._quantum_risk_engine = MagicMock()
        reg._quantum_risk_engine.quantum_var.return_value = {
            "var": -0.01,
            "cvar": -0.01,  # within deadband
            "method": "sobol",
            "n_paths": 5000,
        }
        reg._price_history = {
            "BTC/USD": list(np.linspace(40000, 41000, 50)),
        }

        result = reg.pre_order_check("BTC/USD", "buy", 500.0)
        var_reasons = [r for r in result["reasons"] if "QuantumVaR" in r]
        assert len(var_reasons) == 0


# ═════════════════════════════════════════════════════════════════════════════
# Phase C3 — Quantum signal classifier continuous mapping
# ═════════════════════════════════════════════════════════════════════════════


class TestQuantumKernelSizeFlow:
    """Verify Phase C3 quantum signal classifier applies continuous scaling."""

    def test_high_quality_increases_size(self):
        reg = _make_registry()
        reg._quantum_signal_classifier = MagicMock()
        reg._quantum_signal_classifier.n_features = 4
        reg._quantum_signal_classifier.predict_signal_quality.return_value = {
            "quality": 0.9,
            "confidence": 0.8,
        }
        result = reg.pre_order_check("BTC/USD", "buy", 500.0)
        # quality=0.9 → factor = 0.3 + 0.9 = 1.2 (cap)
        kernel_reasons = [r for r in result["reasons"] if "QuantumKernel" in r]
        assert len(kernel_reasons) == 1
        assert result["size_factor"] >= 1.0

    def test_low_quality_reduces_size(self):
        reg = _make_registry()
        reg._quantum_signal_classifier = MagicMock()
        reg._quantum_signal_classifier.n_features = 4
        reg._quantum_signal_classifier.predict_signal_quality.return_value = {
            "quality": 0.1,
            "confidence": 0.8,
        }
        result = reg.pre_order_check("BTC/USD", "buy", 500.0)
        # quality=0.1 → factor = 0.3 + 0.1 = 0.4
        assert result["size_factor"] <= 0.5

    def test_low_confidence_skips_gate(self):
        """Quality factor should NOT apply when confidence is too low."""
        reg = _make_registry()
        reg._quantum_signal_classifier = MagicMock()
        reg._quantum_signal_classifier.n_features = 4
        reg._quantum_signal_classifier.predict_signal_quality.return_value = {
            "quality": 0.1,
            "confidence": 0.1,  # below 0.3 threshold
        }
        result = reg.pre_order_check("BTC/USD", "buy", 500.0)
        kernel_reasons = [r for r in result["reasons"] if "QuantumKernel" in r]
        assert len(kernel_reasons) == 0


# ═════════════════════════════════════════════════════════════════════════════
# Phase C4 — Quantum portfolio weight consumption
# ═════════════════════════════════════════════════════════════════════════════


class TestQuantumPortfolioWeightConsumer:
    """Verify Phase C4 _execute_signals helpers consume quantum_portfolio."""

    def test_helper_imports_cleanly(self):
        """The helper that consumes the advisory must be importable."""
        from core.execute_signals_helpers import _apply_intelligence_gates
        assert callable(_apply_intelligence_gates)

    def test_quantum_portfolio_advisory_weighting(self):
        """When advisory weights favor BTC heavily, size_pct should increase."""
        from core.execute_signals_helpers import _apply_intelligence_gates

        # Mock self with the minimum surface needed by the helper
        mock_self = MagicMock()
        mock_self.config = _make_config(use_quantum_portfolio_weights=True)
        mock_self.component_registry = None
        mock_self._mtf_bias = {}
        mock_self._latest_regime_label = "NORMAL"
        # No live quantum portfolio dict
        mock_self._quantum_portfolio_weights = None
        mock_self._strategy_active_mask = None
        mock_self._get_current_vol = MagicMock(return_value=0.005)

        sig_fields = {
            "symbol": "BTC/USD",
            "action": "BUY",
            "entry_price": 50000.0,
            "source_strategy": "test_strat",
            "source": "test",
        }
        ctx = {}
        # Build advisory with quantum_portfolio favoring BTC
        advisory = {
            "quantum_portfolio": {
                "weights": [0.6, 0.4],
                "weights_by_symbol": {"BTC/USD": 0.6, "ETH/USD": 0.4},
                "symbols": ["BTC/USD", "ETH/USD"],
                "method": "qaoa_in_repo_simulator",
                "sharpe": 1.5,
                "n_assets": 2,
            },
        }

        # Initial size = 0.05; expected = 0.05 * (0.6 * 2) = 0.05 * 1.2 = 0.06
        # (clamped by helper's range; weight*n=1.2 caps at 1.5)
        size_pct, sizing_method = _apply_intelligence_gates(
            mock_self, sig_fields, 0.05, advisory, "init", ctx,
        )
        # Should be at least slightly increased from 0.05
        assert size_pct > 0.05
        assert "qportfolio" in sizing_method


# ═════════════════════════════════════════════════════════════════════════════
# Phase C5 — Quantum annealer strategy mask consumption
# ═════════════════════════════════════════════════════════════════════════════


class TestQuantumAnnealerMask:
    """Verify Phase C5 strategy mask blocks deselected strategies."""

    def test_helper_blocks_deselected_strategy(self):
        from core.execute_signals_helpers import _apply_intelligence_gates

        mock_self = MagicMock()
        mock_self.config = _make_config()
        mock_self.component_registry = None
        mock_self._mtf_bias = {}
        mock_self._latest_regime_label = "NORMAL"
        mock_self._quantum_portfolio_weights = None
        mock_self._strategy_active_mask = {"strat_a"}  # only strat_a allowed
        mock_self._get_current_vol = MagicMock(return_value=0.005)

        sig_fields = {
            "symbol": "BTC/USD",
            "action": "BUY",
            "entry_price": 50000.0,
            "source_strategy": "strat_b",  # deselected
            "source": "test",
        }
        size_pct, sizing_method = _apply_intelligence_gates(
            mock_self, sig_fields, 0.05, {}, "init", {},
        )
        assert size_pct == 0.0
        assert "BLOCKED" in sizing_method
        assert "quantum_annealer_deselected" in sizing_method

    def test_helper_allows_selected_strategy(self):
        from core.execute_signals_helpers import _apply_intelligence_gates

        mock_self = MagicMock()
        mock_self.config = _make_config()
        mock_self.component_registry = None
        mock_self._mtf_bias = {}
        mock_self._latest_regime_label = "NORMAL"
        mock_self._quantum_portfolio_weights = None
        mock_self._strategy_active_mask = {"strat_a", "strat_b"}
        mock_self._get_current_vol = MagicMock(return_value=0.005)

        sig_fields = {
            "symbol": "BTC/USD",
            "action": "BUY",
            "entry_price": 50000.0,
            "source_strategy": "strat_a",
            "source": "test",
        }
        size_pct, sizing_method = _apply_intelligence_gates(
            mock_self, sig_fields, 0.05, {}, "init", {},
        )
        assert size_pct > 0.0
        assert "BLOCKED" not in sizing_method


# ═════════════════════════════════════════════════════════════════════════════
# Phase C2 — Real returns feed to _quantum_portfolio_optimizer
# ═════════════════════════════════════════════════════════════════════════════


class TestQuantumPortfolioRealDataFeed:
    """Verify Phase C2: optimizer receives real returns, not synthetic stubs."""

    def test_on_cycle_uses_real_price_history(self):
        reg = _make_registry()
        reg._cycle_count = 99  # next on_cycle will be cycle 100, modulo 100 fires

        # Mock optimizer
        reg._quantum_portfolio_optimizer = MagicMock()
        reg._quantum_portfolio_optimizer.optimize_weights.return_value = {
            "weights": np.array([0.5, 0.5]),
            "method": "test_method",
            "sharpe": 1.0,
        }

        # Provide real price history
        reg._price_history = {
            "BTC/USD": list(np.linspace(40000, 41000, 50)),
            "ETH/USD": list(np.linspace(2500, 2600, 50)),
        }

        prices = {"BTC/USD": 41000.0, "ETH/USD": 2600.0}
        # Reset cycle counter so on_cycle fires the optimizer block
        reg._cycle_count = 0
        advisory = reg.on_cycle(prices, regime="TRENDING_UP")

        # Verify the optimizer was called with REAL data (not synthetic [0.001]*n)
        call = reg._quantum_portfolio_optimizer.optimize_weights.call_args
        assert call is not None
        mu, sigma = call[0][0], call[0][1]
        # Real returns from log price diff: should NOT be all 0.001
        assert not np.allclose(mu, 0.001)
        # Covariance should be a real cov matrix (not eye*0.01)
        assert not np.allclose(sigma, np.eye(len(mu)) * 0.01)

        # Advisory should include weights_by_symbol
        assert "quantum_portfolio" in advisory
        qp = advisory["quantum_portfolio"]
        assert "weights_by_symbol" in qp
        assert "n_assets" in qp


# ═════════════════════════════════════════════════════════════════════════════
# Phase C1 — production simulator uses QAOA v2 by default
# ═════════════════════════════════════════════════════════════════════════════


class TestProductionSimulatorQAOAv2:
    """Verify Phase C1: _build_portfolio_optimization_circuit uses real QAOA."""

    def test_default_variant_is_qaoa_v2(self):
        import os
        from quantum.production_quantum_simulator import ARGUSQuantumSimulator

        # Save and restore env
        old = os.environ.pop("ARGUS_QPORTFOLIO_VARIANT", None)
        try:
            sim = ARGUSQuantumSimulator()
            assets = ["BTC", "ETH", "XRP", "ADA"]
            returns = np.random.default_rng(42).normal(0.001, 0.02, (50, 4))
            cov = np.cov(returns.T)
            qc = sim._build_portfolio_optimization_circuit(4, returns, cov, 0.02)
            # The QAOA v2 circuit must include RZZ (off-diagonal coupling) gates
            from quantum_simulator import GateType
            gate_types = {op.gate for op in qc.operations}
            assert GateType.RZZ in gate_types, "QAOA v2 should include RZZ gates"
        finally:
            if old is not None:
                os.environ["ARGUS_QPORTFOLIO_VARIANT"] = old

    def test_legacy_variant_uses_old_circuit(self):
        import os
        from quantum.production_quantum_simulator import ARGUSQuantumSimulator

        old = os.environ.get("ARGUS_QPORTFOLIO_VARIANT")
        os.environ["ARGUS_QPORTFOLIO_VARIANT"] = "legacy"
        try:
            sim = ARGUSQuantumSimulator()
            assets = ["BTC", "ETH", "XRP", "ADA"]
            returns = np.random.default_rng(42).normal(0.001, 0.02, (50, 4))
            cov = np.cov(returns.T)
            qc = sim._build_portfolio_optimization_circuit(4, returns, cov, 0.02)
            # Legacy circuit has only RZ + RX + CNOT — no RZZ
            from quantum_simulator import GateType
            gate_types = {op.gate for op in qc.operations}
            assert GateType.RZZ not in gate_types
            assert GateType.RZ in gate_types or GateType.RX in gate_types
        finally:
            if old is None:
                os.environ.pop("ARGUS_QPORTFOLIO_VARIANT", None)
            else:
                os.environ["ARGUS_QPORTFOLIO_VARIANT"] = old
