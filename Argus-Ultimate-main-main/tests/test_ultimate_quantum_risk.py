#!/usr/bin/env py
"""
Smoke test for Ultimate Quantum Risk Engine.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import math
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s')
logger = logging.getLogger(__name__)

from risk.ultimate_quantum_risk import UltimateQuantumRiskEngine


def test_ultimate_quantum_risk():
    """Test the Ultimate Quantum Risk Engine."""
    print("\n" + "=" * 70)
    print("  ULTIMATE QUANTUM RISK ENGINE - SMOKE TEST")
    print("=" * 70 + "\n")

    # Initialize engine
    engine = UltimateQuantumRiskEngine(
        n_qubits=8,
        bond_dimension=32,
        annealing_iterations=200,
    )

    # Simulate market data for multiple assets
    print("1. Simulating quantum market data...")
    random.seed(42)
    
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]
    base_prices = {"BTC/USDT": 65000, "ETH/USDT": 3500, "SOL/USDT": 150, "AVAX/USDT": 40}
    
    for tick in range(200):
        for symbol in symbols:
            price = base_prices[symbol]
            # Random walk with correlation
            shock = random.gauss(0, 1)
            ret = 0.0001 + 0.02 * shock
            new_price = price * (1 + ret)
            base_prices[symbol] = new_price
            
            volume = random.uniform(1000, 10000)
            engine.update(symbol, new_price, volume)
    
    print(f"   Simulated {len(symbols)} assets x 200 ticks")
    print(f"   Final prices: BTC=${base_prices['BTC/USDT']:,.0f}, "
          f"ETH=${base_prices['ETH/USDT']:,.0f}")

    # Test quantum status
    print("\n2. Quantum System Status:")
    status = engine.get_quantum_status()
    for key, value in status.items():
        print(f"   {key}: {value}")

    # Test risk assessment
    print("\n3. Running Quantum Risk Assessment...")
    assessment = engine.assess_risk(
        portfolio_equity=100000,
        positions={"BTC/USDT": 50000, "ETH/USDT": 30000, "SOL/USDT": 15000, "AVAX/USDT": 5000},
    )

    print("\n4. Classical Risk Metrics:")
    print(f"   VaR 95%:       {assessment.var_95:.4%}")
    print(f"   VaR 99%:       {assessment.var_99:.4%}")
    print(f"   CVaR 95%:      {assessment.cvar_95:.4%}")
    print(f"   CVaR 99%:      {assessment.cvar_99:.4%}")
    print(f"   Max Drawdown:  {assessment.max_drawdown:.2%}")

    print("\n5. Quantum-Enhanced Risk Metrics:")
    print(f"   Quantum VaR 95%:  {assessment.quantum_var_95:.4%}")
    print(f"   Quantum VaR 99%:  {assessment.quantum_var_99:.4%}")
    print(f"   Entanglement Risk: {assessment.entanglement_risk:.4f}")
    print(f"   Coherence Risk:    {assessment.coherence_risk:.4f}")
    print(f"   Superposition Scenarios: {assessment.superposition_scenarios}")

    print("\n6. Quantum Optimization Results:")
    anneal = assessment.annealing_result
    print(f"   Optimal Energy:     {anneal.optimal_energy:.6f}")
    print(f"   Ground State Prob:  {anneal.ground_state_probability:.2%}")
    print(f"   Annealing Time:     {anneal.annealing_time:.3f}s")
    print(f"   Convergence Rate:   {anneal.convergence_rate:.4f}")

    print("\n7. Tensor Network State:")
    tn = assessment.tensor_network_state
    print(f"   Bond Dimensions:    {tn.bond_dimensions[:5]}...")
    print(f"   Truncation Error:   {tn.truncation_error:.2e}")
    print(f"   Total Bond Dim:     {tn.total_bond_dimension}")

    print("\n8. Quantum State:")
    qs = assessment.quantum_risk_state
    print(f"   Entanglement Entropy: {qs.entanglement_entropy:.4f}")
    print(f"   Coherence Time:       {qs.coherence_time:.2f}s")
    print(f"   Fidelity:             {qs.fidelity:.2%}")

    print("\n9. Overall Assessment:")
    print(f"   Risk Score:    {assessment.risk_score:.1f}/100")
    print(f"   Risk Level:    {assessment.risk_level}")
    print(f"   Confidence:    {assessment.confidence:.2%}")
    print(f"   Quantum Speedup: {assessment.quantum_speedup:.1f}x")
    print(f"\n   Recommendation:")
    print(f"   {assessment.recommendation}")

    # Test entanglement detection
    print("\n10. Entanglement Correlation Analysis:")
    for i, sym1 in enumerate(symbols):
        for sym2 in symbols[i+1:]:
            corr = engine.entanglement_correlator.detect_non_classical_correlation(sym1, sym2)
            print(f"   {sym1[:7]} <-> {sym2[:7]}: "
                  f"classical={corr['classical_corr']:.3f}, "
                  f"entanglement={corr['entanglement']:.3f}, "
                  f"non-classical={corr['non_classical_ratio']:.3f}")

    # Test high-stress scenario
    print("\n11. Testing High-Stress Scenario (simulated crash)...")
    for _ in range(50):
        crash_price = base_prices["BTC/USDT"] * 0.99  # 1% crash per tick
        base_prices["BTC/USDT"] = crash_price
        engine.update("BTC/USDT", crash_price, 50000)
    
    stress_assessment = engine.assess_risk(
        portfolio_equity=85000,  # 15% loss
        positions={"BTC/USDT": 40000},
    )
    
    print(f"   After simulated crash:")
    print(f"   Risk Score:  {stress_assessment.risk_score:.1f}/100")
    print(f"   Risk Level:  {stress_assessment.risk_level}")
    print(f"   Quantum VaR: {stress_assessment.quantum_var_99:.4%}")
    print(f"   Recommendation: {stress_assessment.recommendation}")

    print("\n" + "=" * 70)
    print("  ULTIMATE QUANTUM RISK ENGINE - TEST COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_ultimate_quantum_risk()
