"""
Collect on-chain data for ML training.

Uses the on-chain intelligence engine to generate features:
- Whale activity signals
- Exchange flow analysis
- DeFi TVL and risks
- Smart money following
"""

import json
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.onchain_intelligence import OnChainIntelligenceEngine, get_onchain_intelligence_engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# Sample on-chain metrics (simulated for demo - real data would come from APIs)
SAMPLE_WHALE_ALERTS = [
    {"token": "BTC", "whale_inflow": 2500, "whale_outflow": 1800, "significant": True},
    {"token": "ETH", "whale_inflow": 35000, "whale_outflow": 28000, "significant": True},
    {"token": "BNB", "whale_inflow": 15000, "whale_outflow": 12000, "significant": False},
    {"token": "SOL", "whale_inflow": 85000, "whale_outflow": 92000, "significant": False},
    {"token": "XRP", "whale_inflow": 5000000, "whale_outflow": 4800000, "significant": True},
]

SAMPLE_EXCHANGE_FLOWS = [
    {"token": "BTC", "exchange_inflow": 45000, "exchange_outflow": 52000, "net_flow": -7000},
    {"token": "ETH", "exchange_inflow": 180000, "exchange_outflow": 165000, "net_flow": 15000},
    {"token": "BNB", "exchange_inflow": 85000, "exchange_outflow": 92000, "net_flow": -7000},
    {"token": "SOL", "exchange_inflow": 120000, "exchange_outflow": 145000, "net_flow": -25000},
    {"token": "XRP", "exchange_inflow": 8500000, "exchange_outflow": 7800000, "net_flow": 700000},
]

SAMPLE_DEFI_TVL = [
    {"protocol": "Uniswap", "tvl_usd": 3500000000, "change_24h": 2.5},
    {"protocol": "Aave", "tvl_usd": 8500000000, "change_24h": -1.2},
    {"protocol": "Compound", "tvl_usd": 1200000000, "change_24h": 0.8},
    {"protocol": "Curve", "tvl_usd": 1800000000, "change_24h": 3.2},
    {"protocol": "SushiSwap", "tvl_usd": 850000000, "change_24h": -2.5},
]


def collect_onchain_data(output_path: str = "data/onchain/onchain_data.pkl"):
    """Collect on-chain data for ML training."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing on-chain intelligence engine...")
    engine = get_onchain_intelligence_engine()
    
    onchain_data = {
        "whale_alerts": [],
        "exchange_flows": [],
        "defi_tvl": [],
        "whale_signal": 0.0,
        "exchange_signal": 0.0,
        "defi_signal": 0.0,
        "combined_signal": 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    # Process whale alerts
    total_whale_in = 0
    total_whale_out = 0
    significant_count = 0
    
    for alert in SAMPLE_WHALE_ALERTS:
        total_whale_in += alert["whale_inflow"]
        total_whale_out += alert["whale_outflow"]
        if alert["significant"]:
            significant_count += 1
    
    net_whale = total_whale_in - total_whale_out
    onchain_data["whale_alerts"] = SAMPLE_WHALE_ALERTS
    onchain_data["whale_signal"] = float(net_whale / max(total_whale_in, 1))
    logger.info(f"  Whale signal: {onchain_data['whale_signal']:.2%}")
    
    # Process exchange flows
    total_net_flow = sum(f["net_flow"] for f in SAMPLE_EXCHANGE_FLOWS)
    onchain_data["exchange_flows"] = SAMPLE_EXCHANGE_FLOWS
    onchain_data["exchange_signal"] = float(total_net_flow / 1000000)  # Normalize
    logger.info(f"  Exchange signal: {onchain_data['exchange_signal']:.2f}")
    
    # Process DeFi TVL
    avg_defi_change = sum(d["change_24h"] for d in SAMPLE_DEFI_TVL) / len(SAMPLE_DEFI_TVL)
    onchain_data["defi_tvl"] = SAMPLE_DEFI_TVL
    onchain_data["defi_signal"] = avg_defi_change / 100  # Normalize
    logger.info(f"  DeFi signal: {onchain_data['defi_signal']:.2%}")
    
    # Combined signal (weighted average)
    onchain_data["combined_signal"] = (
        onchain_data["whale_signal"] * 0.4 +
        onchain_data["exchange_signal"] * 0.4 +
        onchain_data["defi_signal"] * 0.2
    )
    logger.info(f"  Combined signal: {onchain_data['combined_signal']:.2%}")
    
    # Save
    with open(output_path, 'wb') as f:
        pickle.dump(onchain_data, f)
    
    logger.info(f"On-chain data saved to {output_path}")
    return onchain_data


def create_onchain_features(df, onchain_data):
    """Add on-chain features to a DataFrame."""
    if not onchain_data:
        df["whale_signal"] = 0.0
        df["exchange_signal"] = 0.0
        df["defi_signal"] = 0.0
        df["combined_onchain_signal"] = 0.0
        return df
    
    df["whale_signal"] = onchain_data.get("whale_signal", 0.0)
    df["exchange_signal"] = onchain_data.get("exchange_signal", 0.0)
    df["defi_signal"] = onchain_data.get("defi_signal", 0.0)
    df["combined_onchain_signal"] = onchain_data.get("combined_signal", 0.0)
    
    return df


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("ON-CHAIN DATA COLLECTION")
    logger.info("="*60)
    
    collect_onchain_data()
    logger.info("Done!")