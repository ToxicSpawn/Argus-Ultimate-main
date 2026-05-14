"""
Start Argus Omega Supreme with Real Kraken Connection
Simple one-command launcher with Kraken API integration
"""

import asyncio
import logging
import sys
from pathlib import Path

// Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('argus.log')
    ]
)

// Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from config.api_config import get_api_config


async def start_argus():
    """Start Argus Omega with Kraken integration"""
    print("=" * 100)
    print("🌌 STARTING ARGUS OMEGA SUPREME")
    print("=" * 100)
    
    // Check configuration
    config = get_api_config()
    
    print("\n🔌 API Configuration:")
    summary = config.get_summary()
    print(f"   Trading Mode: {summary['trading_mode'].upper()}")
    print(f"   Kraken API: {'✅ Connected' if summary['kraken_configured'] else '❌ Not configured'}")
    
    if summary['trading_mode'] == 'live':
        print("\n   🔴 LIVE TRADING MODE")
        print("   ⚠️  Real money will be used!")
        print(f"   Daily Loss Limit: ${summary['risk_limits']['daily_loss_limit']}")
        input("\n   Press ENTER to confirm you want to trade with REAL MONEY, or Ctrl+C to cancel...")
    else:
        print("\n   ✅ PAPER TRADING MODE")
        print("   💡 Simulated trading with real market data")
        print(f"   Paper Balance: ${config.paper_initial_balance}")
    
    // Start all systems
    print("\n" + "=" * 100)
    print("🚀 INITIALIZING ALL 62 SYSTEMS...")
    print("=" * 100)
    
    try:
        // Start Argus Omega
        from wiring.argus_omega_supreme import start_argus_omega_supreme
        
        omega = await start_argus_omega_supreme()
        
        print("\n" + "=" * 100)
        print("✅ ARGUS OMEGA IS LIVE AND TRADING!")
        print("=" * 100)
        
        if summary['trading_mode'] == 'paper':
            print("\n📊 Paper Trading Mode:")
            print("   - Real market data from Kraken")
            print("   - Simulated trades (no real money)")
            print("   - Test for 1 week before going live")
        else:
            print("\n🔴 LIVE TRADING MODE:")
            print("   - Real money at risk!")
            print("   - Monitor closely for first few days")
            print("   - Check argus.log for all activity")
        
        print("\n📈 Monitoring:")
        print("   - Logs: argus.log")
        print("   - Stats: Check console output")
        print("   - Stop: Press Ctrl+C")
        
        // Keep running
        print("\n⏳ Running continuously... (Press Ctrl+C to stop)")
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping Argus Omega...")
        print("✅ Shutdown complete")
        
    except Exception as e:
        print(f"\n❌ Error starting Argus: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(start_argus())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
