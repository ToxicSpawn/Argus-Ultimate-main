"""
Collect sentiment data for ML training.

Uses FinBERT to analyze market-related text and saves sentiment scores
for integration with the ML pipeline.
"""

import json
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.llm_sentiment_enhanced import LLMEnsembleSentiment

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# Sample market headlines for sentiment analysis
SAMPLE_HEADLINES = [
    # Positive
    "Bitcoin ETF approval sparks institutional rally",
    "Ethereum merge success boosts network efficiency",
    "Solana TVL reaches new all-time high",
    "Ripple wins SEC case clarification",
    "Cardano smart contracts adoption accelerates",
    "BNB burn removes supply from market",
    "加密货币ETF批准引发机构反弹",
    "以太坊合并成功提升网络效率",
    # Negative
    "Bitcoin flash crash wipes $1B in longs",
    "SEC delays ETF decision again",
    "Terra collapse triggers regulatory probe",
    "Celsius insolvency shakes confidence",
    "FTX audit reveals massive hole",
    "加密货币闪崩清空多头",
    "监管机构延迟ETF决定",
    # Neutral
    "BTC trades in tight range",
    "Volume stays flat on weekend",
    "Miners hodl through volatility",
    "交易所報告穩定流入",
]


def collect_sentiment_data(output_path: str = "data/sentiment/sentiment_data.pkl"):
    """Collect sentiment data from sample headlines."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing sentiment analyzer...")
    analyzer = LLMEnsembleSentiment()
    
    logger.info(f"Analyzing {len(SAMPLE_HEADLINES)} headlines...")
    sentiment_data = []
    
    for i, headline in enumerate(SAMPLE_HEADLINES):
        try:
            result = analyzer.analyze_text(headline)
            sentiment_data.append({
                "text": headline,
                "sentiment": result.sentiment,
                "confidence": result.confidence,
                "positive": result.positive_score,
                "negative": result.negative_score,
                "neutral": result.neutral_score,
                "signed": result.signed_score,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            logger.info(f"  [{i+1}/{len(SAMPLE_HEADLINES)}] {headline[:40]}... → {result.sentiment} ({result.confidence:.2f})")
        except Exception as e:
            logger.warning(f"  Failed to analyze: {headline[:40]}... → {e}")
    
    # Save
    with open(output_path, 'wb') as f:
        pickle.dump(sentiment_data, f)
    
    # Summary
    positive = sum(1 for s in sentiment_data if s["sentiment"] == "positive")
    negative = sum(1 for s in sentiment_data if s["sentiment"] == "negative")
    neutral = sum(1 for s in sentiment_data if s["sentiment"] == "neutral")
    
    logger.info(f"Sentiment data saved to {output_path}")
    logger.info(f"  Positive: {positive}, Negative: {negative}, Neutral: {neutral}")
    logger.info(f"  Mean confidence: {sum(s['confidence'] for s in sentiment_data)/len(sentiment_data):.2%}")
    
    return sentiment_data


def create_sentiment_features(df, sentiment_data):
    """Add sentiment features to a DataFrame."""
    if not sentiment_data:
        df["sentiment_signal"] = 0.0
        df["sentiment_confidence"] = 0.0
        return df
    
    # Average sentiment from recent headlines
    avg_positive = sum(s["positive"] for s in sentiment_data) / len(sentiment_data)
    avg_negative = sum(s["negative"] for s in sentiment_data) / len(sentiment_data)
    avg_confidence = sum(s["confidence"] for s in sentiment_data) / len(sentiment_data)
    avg_signed = sum(s["signed"] for s in sentiment_data) / len(sentiment_data)
    
    df["sentiment_signal"] = avg_signed
    df["sentiment_confidence"] = avg_confidence
    df["sentiment_positive"] = avg_positive
    df["sentiment_negative"] = avg_negative
    
    return df


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("SENTIMENT DATA COLLECTION")
    logger.info("="*60)
    
    collect_sentiment_data()
    logger.info("Done!")