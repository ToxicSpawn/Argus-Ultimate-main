"""
Live Liquidity Scanner - Continuously scans for highest liquidity pairs.

Run standalone: py -m ml.liquidity_scanner_live
"""

import asyncio
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class LiveLiquidityScanner:
    """Continuously scan for highest liquidity pairs."""
    
    def __init__(self, scan_interval: int = 60):  # seconds
        self.scan_interval = scan_interval
        self._running = False
        self._scanner = None
        self._rankings = []
        
    async def initialize(self):
        """Initialize the scanner."""
        try:
            from execution.multi_pair_liquidity_scanner import create_liquidity_scanner
            self._scanner = create_liquidity_scanner()
            logger.info("✅ Liquidity scanner initialized")
        except Exception as e:
            logger.error(f"Scanner init error: {e}")
            
    async def scan_once(self):
        """Run one scan cycle."""
        if not self._scanner:
            return
            
        try:
            # Scan all pairs
            results = await self._scanner.scan_all_pairs()
            
            # Get top pairs by volume
            rankings = self._scanner.get_volume_rankings(limit=10)
            
            print(f"\n=== {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC ===")
            print("Top 10 by Volume:")
            for i, (symbol, score) in enumerate(rankings[:10], 1):
                print(f"  {i}. {symbol}: ${score/1e6:.1f}M")
                
            self._rankings = rankings
            return rankings
            
        except Exception as e:
            logger.error(f"Scan error: {e}")
            
    async def run_continuously(self):
        """Run scanner continuously."""
        self._running = True
        await self.initialize()
        
        logger.info(f"🔄 Starting continuous scan every {self.scan_interval}s...")
        
        while self._running:
            await self.scan_once()
            await asyncio.sleep(self.scan_interval)


async def main():
    """Main entry."""
    scanner = LiveLiquidityScanner(scan_interval=30)  # 30 second intervals
    
    # Single scan first
    print("=== Initial Liquidity Scan ===")
    await scanner.initialize()
    await scanner.scan_once()
    
    # Or run continuously with Ctrl+C to stop
    # await scanner.run_continuously()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    asyncio.run(main())