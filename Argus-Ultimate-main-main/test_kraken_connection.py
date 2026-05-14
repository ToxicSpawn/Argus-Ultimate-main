"""
Test Kraken API Connection
Verify your API keys work before starting Argus
"""

import asyncio
import sys
from pathlib import Path

// Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from config.api_config import get_api_config


async def test_kraken_connection():
    """Test if Kraken API connection works"""
    print("=" * 70)
    print("🔌 TESTING KRAKEN API CONNECTION")
    print("=" * 70)
    
    // Load config
    config = get_api_config()
    
    print("\n1. Checking API keys...")
    if not config.kraken_api_key or config.kraken_api_key == 'your_kraken_api_key_here':
        print("   ❌ Kraken API key not found in .env")
        print("\n   Fix: Edit .env and add your keys:")
        print("   KRAKEN_API_KEY=K-youractualkeyhere")
        print("   KRAKEN_API_SECRET=youractualsecrethere")
        return False
    
    print(f"   ✅ API Key found: {config.kraken_api_key[:10]}...")
    print(f"   ✅ Sandbox mode: {config.kraken_sandbox}")
    print(f"   ✅ Trading mode: {config.trading_mode}")
    
    // Try to connect to Kraken
    print("\n2. Testing Kraken API connection...")
    
    try:
        // Import kraken library (install if needed)
        try:
            import krakenex
        except ImportError:
            print("   Installing krakenex library...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "krakenex"])
            import krakenex
        
        // Create Kraken API instance
        kraken = krakenex.API()
        kraken.key = config.kraken_api_key
        kraken.secret = config.kraken_api_secret
        
        // Test connection - get server time
        print("   Connecting to Kraken...")
        response = kraken.query_public('Time')
        
        if 'error' in response and response['error']:
            print(f"   ❌ Connection failed: {response['error']}")
            return False
        
        server_time = response['result']['unixtime']
        print(f"   ✅ Connected! Server time: {server_time}")
        
        // Test authenticated endpoint (balance)
        print("\n3. Testing authenticated access...")
        balance = kraken.query_private('Balance')
        
        if 'error' in balance and balance['error']:
            error_msg = balance['error'][0]
            if 'Invalid key' in error_msg:
                print(f"   ❌ Invalid API key: {error_msg}")
                print("\n   Fix: Check your API key in .env file")
                return False
            elif 'Invalid signature' in error_msg:
                print(f"   ❌ Invalid API secret: {error_msg}")
                print("\n   Fix: Check your API secret in .env file")
                return False
            else:
                print(f"   ⚠️  Warning: {error_msg}")
        else:
            print("   ✅ Authenticated access successful!")
            print(f"   Available balances: {list(balance['result'].keys())}")
        
        // Get account info
        print("\n4. Getting account information...")
        trade_balance = kraken.query_private('TradeBalance')
        
        if 'result' in trade_balance:
            result = trade_balance['result']
            print(f"   Equivalent Balance: {result.get('eb', 'N/A')}")
            print(f"   Trade Balance: {result.get('tb', 'N/A')}")
            print(f"   Margin Open Positions: {result.get('mop', 'N/A')}")
        
        // Show what would happen
        print("\n5. Trading Mode Check:")
        if config.trading_mode == 'paper':
            print("   ✅ PAPER TRADING (Safe - no real money)")
            print("   Trades will be SIMULATED")
            print("   Real market data will be used")
        elif config.trading_mode == 'live':
            print("   🔴 LIVE TRADING (Real money!)")
            print(f"   Daily loss limit: ${config.daily_loss_limit}")
            print(f"   Max position: {config.max_position_size*100}%")
            print("   ⚠️  Make sure you want to trade with real money!")
        
        print("\n" + "=" * 70)
        print("✅ KRAKEN API CONNECTION SUCCESSFUL!")
        print("=" * 70)
        print("\nYou can now start Argus:")
        print("   python -m wiring.argus_omega_supreme")
        print("\nOr run with test capital:")
        print("   python test_argus_live.py")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Missing library: {e}")
        print("\n   Fix: pip install krakenex")
        return False
        
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        print("\n   Troubleshooting:")
        print("   1. Check internet connection")
        print("   2. Verify API keys are correct")
        print("   3. Check if Kraken is online")
        return False


def show_env_example():
    """Show what a proper .env looks like"""
    print("\n" + "=" * 70)
    print("📄 EXAMPLE .env FILE:")
    print("=" * 70)
    print("""
# REQUIRED - Kraken API Keys (get from https://www.kraken.com/)
KRAKEN_API_KEY=K-1234567890abcdef1234567890abcdef
KRAKEN_API_SECRET=abcdefghijklmnopqrstuvwxyz1234567890abcdef=
KRAKEN_SANDBOX=true

# Trading Mode: paper = safe testing, live = real money
TRADING_MODE=paper
PAPER_INITIAL_BALANCE=1000.0

# Risk Management
MAX_POSITION_SIZE=0.05      # 5% max per trade
STOP_LOSS_PCT=0.02          # 2% stop loss
DAILY_LOSS_LIMIT=100.0      # $100/day max
    """)


if __name__ == "__main__":
    success = asyncio.run(test_kraken_connection())
    
    if not success:
        show_env_example()
        sys.exit(1)
    else:
        print("\n🚀 Ready to start Argus Omega!")
