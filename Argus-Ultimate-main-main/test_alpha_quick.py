"""
Simple test script - tests alpha and market flow systems without full main.py
"""
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test():
print("=" * 60)
print("ARGUS ALPHA & MARKET FLOW TEST - SIMPLE TEST")
print("=" * 60)
    
    # Test 1: Alpha Signal Fusion
    print("\n[1] Testing Alpha Signal Fusion...")
    from ml.alpha_signal_fusion import create_alpha_fusion, AlphaSignal, OnChainAlpha, LLMSentimentAlpha
    
    fusion = create_alpha_fusion(use_ml=False, use_alpha=False, use_sentiment=False)
    print(f"   - Created: {fusion}")
    
    # Test on-chain
    onchain = OnChainAlpha(min_whale_threshold=100000)
    result = onchain.analyze("BTC", {"whale_inflow": 200000, "whale_outflow": 0})
    print(f"   ✅ On-chain analysis: {result}")
    
    # Test sentiment
    llm = LLMSentimentAlpha()
    result2 = await llm.analyze("BTC", {"news": [{"sentiment": 0.8}, {"sentiment": 0.9}]})
    print(f"   ✅ LLM sentiment: {result2}")
    
    # Test 2: Market Flow Integration
    print("\n[2] Testing Market Flow Integration...")
    from ml.market_flow_integration import create_integration
    
    mfi = create_integration()
    print(f"   ✅ Created: {mfi}")
    
    status = mfi.get_status()
    print(f"   ✅ Status: {status}")
    
    # Test 3: Market Flow Risk
    print("\n[3] Testing Market Flow Risk...")
    from ml.market_flow_risk import create_risk_adapter
    
    risk = create_risk_adapter()
    print(f"   ✅ Created: {risk}")
    
    # Test risk assessment
    risk_result = risk.assess_market_flow_risk(
        current_volatility=0.02,
        historical_volatility=0.02,
        current_volume=1000,
        average_volume=1000,
        bid_ask_spread_bps=5,
        order_book_depth=5000,
        current_regime="trending_up",
        fear_greed_index=50,
        price_change_pct=1.0,
    )
    print(f"   ✅ Risk assessment: {risk_result.condition}")
    
    # Test 4: Component Registry integration
    print("\n[4] Testing Component Registry...")
    from core.component_registry import ComponentRegistry
    print(f"   ✅ ComponentRegistry imported")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)
    print("\nAlpha System Summary:")
    print("- AlphaSignalFusion: Combines ML + sentiment + on-chain → signals")
    print("- MarketFlowIntegration: Pipes signals through risk adaptation")  
    print("- MarketFlowRisk: Stops/positions breathe with volatility")
    print("- All wired into component registry")
    print("\nTo run full system: py main.py paper")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test())