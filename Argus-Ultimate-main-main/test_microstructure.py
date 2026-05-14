import sys
sys.path.insert(0, r"F:\Argus-Ultimate-main")
from analytics.market_microstructure import *
from datetime import datetime, timezone

e = MarketMicrostructureEngine()

# Test 1: VPIN
for i in range(200):
    e.vpin_calculator.update(50000 + i * 0.1, 100.0, i % 2 == 0)
vpin = e.vpin_calculator.calculate_vpin()
assert 0 <= vpin <= 1, f"VPIN out of range: {vpin}"
print(f"1. VPIN: {vpin:.4f} (toxicity: {e.vpin_calculator.get_toxicity_level().value})")

# Test 2: Order Flow
result = e.order_flow_analyzer.add_trade(50000.0, 50.0, "buy")
assert isinstance(result, OrderFlowMetrics)
imbalance = e.order_flow_analyzer.calculate_imbalance()
assert -1 <= imbalance <= 1
kyle = e.order_flow_analyzer.calculate_kyle_lambda()
trader = e.order_flow_analyzer.classify_trader_type()
print(f"2. Order Flow: imbalance={imbalance:.4f}, kyle_lambda={kyle:.6f}, trader={trader.value}")

# Test 3: Iceberg
icebergs = e.iceberg_detector.detect_iceberg(
    [(50000.0, 10.0), (49999.0, 10.0)],
    [(50001.0, 10.0), (50002.0, 10.0)]
)
hidden = e.iceberg_detector.estimate_hidden_size(50000.0, "bid", 10.0)
pattern = e.iceberg_detector.track_iceberg_pattern(50000.0, "bid")
detected = pattern.get("detected")
print(f"3. Iceberg: hidden_size={hidden:.2f}, pattern_detected={detected}")

# Test 4: Market Impact
impact = e.impact_modeler.estimate_impact(1000.0, 100000.0, 50000.0)
assert isinstance(impact, MarketImpactEstimate)
schedule = e.impact_modeler.optimize_schedule(1000.0, 100000.0, 50000.0)
assert len(schedule) > 0
print(f"4. Impact: total={impact.total_impact:.6f}, slices={impact.optimal_slices}, schedule_len={len(schedule)}")

# Test 5: Adverse Selection
adv = e.adverse_selection_detector.measure_adverse_selection(
    50001.0, 100.0, "buy", 50000.0, 50000.5
)
assert isinstance(adv, AdverseSelectionMetrics)
toxic = e.adverse_selection_detector.is_toxic_flow()
route = e.adverse_selection_detector.route_order()
print(f"5. Adverse Selection: cost={adv.adverse_selection_cost:.6f}, toxic={toxic}, route={route}")

# Test 6: Order Book Imbalance
ob = e.order_book_imbalance.calculate_ob_imbalance(
    [(50000.0, 100.0), (49999.0, 200.0)],
    [(50001.0, 50.0), (50002.0, 150.0)]
)
assert isinstance(ob, OrderBookImbalance)
pressure = e.order_book_imbalance.get_book_pressure(
    [(50000.0, 100.0)], [(50001.0, 50.0)]
)
print(f"6. OB Imbalance: weighted={ob.weighted_imbalance:.4f}, bid_pressure={ob.bid_pressure:.2f}")

# Test 7: Spread
decomp = e.spread_analyzer.decompose_spread(50001.0, 50000.0, 50002.0, 50000.5, "buy")
assert isinstance(decomp, SpreadDecomposition)
eff = e.spread_analyzer.calculate_effective_spread(50001.0, 50000.0, 50002.0, "buy")
real = e.spread_analyzer.calculate_realized_spread(50001.0, 50000.0, 50002.0, 50000.5, "buy")
print(f"7. Spread: effective={eff:.4f}, realized={real:.4f}, impact={decomp.price_impact_component:.4f}")

# Test 8: Regime
regime = e.regime_detector.detect_regime(
    vpin=0.3, order_imbalance=0.2, spread_bps=10.0,
    trade_rate=5.0, avg_trade_size=100.0, book_imbalance=0.1, price_change=0.05
)
assert isinstance(regime, RegimeResult)
duration = e.regime_detector.get_regime_duration()
prediction = e.regime_detector.predict_regime_change()
print(f"8. Regime: {regime.current_regime.value} (confidence={regime.regime_confidence:.4f}, duration={duration})")

# Test process_trade
trade_result = e.process_trade(50000.0, 100.0, "buy", 49999.0, 50001.0)
assert "vpin" in trade_result
print(f"\nprocess_trade: vpin={trade_result['vpin']:.4f}")

# Test process_orderbook
ob_result = e.process_orderbook([(50000.0, 100.0)], [(50001.0, 50.0)])
assert "icebergs" in ob_result
print(f"process_orderbook: spread_bps={ob_result['spread_bps']:.2f}")

# Test analyze_execution
exec_result = e.analyze_execution(5000.0, 500000.0, 50000.0)
assert "execution_schedule" in exec_result
slippage = exec_result["total_expected_slippage_bps"]
print(f"analyze_execution: slippage_bps={slippage:.2f}")

# Test comprehensive summary
summary = e.get_comprehensive_summary()
print(f"\nSummary keys: {list(summary.keys())}")
print("\nAll 8 components verified successfully!")
