"""
Tests for unified_language_orchestrator.py - verifies determinism after
removal of hash-based randomization from in-process calculation functions.
"""
from __future__ import annotations

import pytest


LANGUAGES = [
    "rust", "cpp", "go", "haskell", "r", "julia", "python", "typescript",
]

PRICE_DATA = {"prices": [100.0 + i * 0.5 for i in range(60)]}
RETURN_DATA = {"returns": [0.001 * (i % 5 - 2) for i in range(50)]}
SIGNAL_DATA = {"confidence": 0.75, "score": 0.80}
CTX_DATA = {"signals": 3, "cash_balance_aud": 500.0, "portfolio_value_aud": 1000.0}
OB_DATA = {
    "bids": [[49990.0, 1.5], [49980.0, 2.0]],
    "asks": [[50010.0, 1.0], [50020.0, 2.5]],
}


class TestVolatilityDeterminism:
    @pytest.mark.parametrize("language", LANGUAGES)
    def test_same_result_on_repeated_call(self, language: str):
        from unified_language_orchestrator import _in_process_volatility_estimate
        r1 = _in_process_volatility_estimate(PRICE_DATA, language)
        r2 = _in_process_volatility_estimate(PRICE_DATA, language)
        assert r1["volatility_annual_bps"] == r2["volatility_annual_bps"], (
            f"Non-deterministic volatility for {language}: {r1} vs {r2}"
        )

    def test_all_languages_return_positive_vol(self):
        from unified_language_orchestrator import _in_process_volatility_estimate
        for lang in LANGUAGES:
            r = _in_process_volatility_estimate(PRICE_DATA, lang)
            assert r["volatility_annual_bps"] >= 0.0, f"{lang}: negative volatility"


class TestSignalScoreDeterminism:
    @pytest.mark.parametrize("language", LANGUAGES)
    def test_same_result_on_repeated_call(self, language: str):
        from unified_language_orchestrator import _in_process_signal_score
        r1 = _in_process_signal_score(SIGNAL_DATA, language)
        r2 = _in_process_signal_score(SIGNAL_DATA, language)
        assert r1["score_delta"] == r2["score_delta"], (
            f"Non-deterministic signal score for {language}"
        )

    def test_score_reflects_base_score(self):
        from unified_language_orchestrator import _in_process_signal_score
        # score_delta should be base_score * weight, always non-negative for positive signal
        for lang in LANGUAGES:
            r = _in_process_signal_score({"confidence": 0.8, "score": 0.8}, lang)
            assert r["score_delta"] >= 0.0, f"{lang}: negative score_delta for positive signal"


class TestRegimeDeterminism:
    @pytest.mark.parametrize("language", LANGUAGES)
    def test_same_regime_on_repeated_call(self, language: str):
        from unified_language_orchestrator import _in_process_regime_estimate
        r1 = _in_process_regime_estimate(PRICE_DATA, language)
        r2 = _in_process_regime_estimate(PRICE_DATA, language)
        assert r1["regime"] == r2["regime"], (
            f"Non-deterministic regime for {language}: {r1['regime']} vs {r2['regime']}"
        )

    def test_all_languages_agree_on_clear_trend(self):
        """Strongly trending data should produce same regime across all languages."""
        from unified_language_orchestrator import _in_process_regime_estimate
        trending = {"prices": [100.0 * (1.02 ** i) for i in range(40)]}
        regimes = set()
        for lang in LANGUAGES:
            r = _in_process_regime_estimate(trending, lang)
            regimes.add(r["regime"])
        # All languages must agree on the same regime for clearly trending data
        assert len(regimes) == 1, f"Languages disagree on regime for trending data: {regimes}"


class TestCyclePlanDeterminism:
    @pytest.mark.parametrize("language", LANGUAGES)
    def test_same_boost_on_repeated_call(self, language: str):
        from unified_language_orchestrator import _in_process_cycle_plan
        r1 = _in_process_cycle_plan(CTX_DATA, language)
        r2 = _in_process_cycle_plan(CTX_DATA, language)
        assert r1["cycle_boost"] == r2["cycle_boost"], (
            f"Non-deterministic cycle_boost for {language}"
        )
