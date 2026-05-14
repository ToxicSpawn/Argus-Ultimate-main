"""
Test API Setup - Verify Your Configuration Works
Run this to check if all API keys are configured correctly
"""

import asyncio
import sys
from pathlib import Path

// Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from config.api_config import get_api_config


def test_api_setup():
    """Test if API configuration is valid"""
    print("=" * 70)
    print("🔌 ARGUS API CONFIGURATION TEST")
    print("=" * 70)
    
    // Load config
    print("\n1. Loading configuration from .env...")
    try:
        config = get_api_config()
        print("   ✅ Configuration loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load configuration: {e}")
        print("\n   Make sure you have created a .env file:")
        print("   copy .env.example .env")
        return False
    
    // Check validity
    print("\n2. Checking configuration validity...")
    if config.is_valid:
        print("   ✅ Configuration is valid")
    else:
        print("   ⚠️  Configuration has errors (see above)")
        print("\n   Fix: Edit .env file and add your API keys")
    
    // Show summary
    print("\n3. Configuration Summary:")
    summary = config.get_summary()
    
    print(f"   Trading Mode: {summary['trading_mode']}")
    print(f"   Kraken Configured: {'✅ Yes' if summary['kraken_configured'] else '❌ No'}")
    print(f"   Coinbase Configured: {'✅ Yes' if summary['coinbase_configured'] else '❌ No (optional)'}")
    
    print("\n4. Data Sources:")
    for source, configured in summary['data_sources'].items():
        status = "✅" if configured else "⚠️"
        print(f"   {status} {source}: {'configured' if configured else 'not configured'}")
    
    print("\n5. Risk Management:")
    risk = summary['risk_limits']
    print(f"   Max Position: {risk['max_position_size']*100:.0f}%")
    print(f"   Max Drawdown: {risk['max_drawdown']*100:.0f}%")
    print(f"   Daily Loss Limit: ${risk['daily_loss_limit']}")
    print(f"   Stop Loss: {risk['stop_loss_pct']*100:.1f}%")
    
    // Paper vs Live warning
    print("\n6. Trading Mode Check:")
    if summary['trading_mode'] == 'paper':
        print("   ✅ PAPER TRADING MODE (Safe - fake money)")
        print("   💡 All trades will be simulated")
    elif summary['trading_mode'] == 'live':
        print("   ⚠️  LIVE TRADING MODE (Real money!)")
        print("   🔴 Make sure you know what you're doing!")
    
    // Final status
    print("\n" + "=" * 70)
    if summary['kraken_configured'] and summary['trading_mode'] == 'paper':
        print("✅ READY FOR PAPER TRADING")
        print("   Run: python -m wiring.argus_omega_supreme")
    elif summary['kraken_configured'] and summary['trading_mode'] == 'live':
        print("⚠️  READY FOR LIVE TRADING")
        print("   🔴 Make sure you have tested with paper trading first!")
        print("   💡 Start with small amount ($100)")
    else:
        print("❌ NOT READY - Missing API Keys")
        print("   1. copy .env.example .env")
        print("   2. Edit .env and add your Kraken API keys")
        print("   3. Run this test again")
    print("=" * 70)
    
    return summary['kraken_configured']


def test_data_flow():
    """Test if data pipeline can start"""
    print("\n" + "=" * 70)
    print("🌊 TESTING DATA PIPELINE")
    print("=" * 70)
    
    try:
        from wiring.argus_realtime_data_flow import get_realtime_data_flow
        
        print("\n1. Creating data pipeline...")
        flow = get_realtime_data_flow()
        print("   ✅ Data pipeline created")
        
        print("\n2. Pipeline Statistics:")
        stats = flow.get_pipeline_stats()
        print(f"   Systems registered: {stats['data_flow']['systems_active']}")
        print(f"   Data sources: {stats['data_flow']['data_sources']}")
        
        print("\n3. Status:")
        print(f"   Pipeline status: {stats['status']}")
        
        print("\n✅ Data pipeline ready to start")
        return True
        
    except Exception as e:
        print(f"\n❌ Data pipeline error: {e}")
        return False


def main():
    """Run all tests"""
    api_ok = test_api_setup()
    
    if api_ok:
        print("\n" + "=" * 70)
        print("🚀 NEXT STEPS:")
        print("=" * 70)
        print("1. Start paper trading:")
        print("   python -m wiring.argus_omega_supreme")
        print("\n2. Or run the full test:")
        print("   python test_argus_live.py")
        print("\n3. Monitor for 1 week, then switch to live:")
        print("   Edit .env: TRADING_MODE=live")
    else:
        print("\n" + "=" * 70)
        print("🔧 FIX NEEDED:")
        print("=" * 70)
        print("1. copy .env.example .env")
        print("2. Get Kraken API key from https://www.kraken.com/")
        print("3. Edit .env and paste your keys")
        print("4. Run this test again")


if __name__ == "__main__":
    main()
