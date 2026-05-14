"""
Train GNN model for cross-asset learning.

Trains a Graph Neural Network to learn relationships between
BTC, ETH, BNB, SOL, and XRP for improved signal generation.
"""

import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def load_historical_data():
    with open('data/historical/historical_data.pkl', 'rb') as f:
        return pickle.load(f)


def create_multi_asset_graph(data):
    """Create a multi-asset graph for GNN."""
    symbols = list(data.keys())[:5]  # BTC, ETH, BNB, SOL, XRP
    returns_data = {}
    
    for symbol in symbols:
        df = pd.DataFrame(data[symbol]['1h'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('datetime').sort_index()
        
        returns = df['close'].pct_change().fillna(0).values
        returns_data[symbol] = returns
    
    # Create correlation matrix
    min_len = min(len(v) for v in returns_data.values())
    returns_array = np.array([returns_data[s][:min_len] for s in symbols])
    
    correlations = np.corrcoef(returns_array)
    
    return symbols, returns_array, correlations


def main():
    logger.info("=" * 70)
    logger.info("TRAINING GNN MODEL")
    logger.info("=" * 70)
    
    # Load data
    data = load_historical_data()
    symbols, returns, correlations = create_multi_asset_graph(data)
    
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Correlation matrix shape: {correlations.shape}")
    logger.info(f"Returns shape: {returns.shape}")
    
    # Create edge index from correlation matrix
    threshold = 0.3
    edge_index = []
    edge_weight = []
    
    for i in range(len(symbols)):
        for j in range(len(symbols)):
            if i != j and abs(correlations[i, j]) > threshold:
                edge_index.append([i, j])
                edge_weight.append(correlations[i, j])
    
    logger.info(f"Edges: {len(edge_index)}")
    
    # Try to train GNN
    try:
        from ml.gnn_trainer import GNNTrainer, GraphConfig
        import torch
        
        config = GraphConfig(
            tickers=symbols,
            lookback=24,
            correlation_threshold=threshold,
            gnn_type='gat',
            hidden_channels=32,
            num_layers=2,
            dropout=0.3,
        )
        
        trainer = GNNTrainer(config)
        
        # Create labels (next period return direction)
        labels = (returns[:, 1:] > 0).astype(int).flatten()
        features = returns[:, :-1].T.flatten()
        
        # Remove NaN
        valid = ~np.isnan(features) & ~np.isnan(labels)
        features = features[valid].reshape(-1, len(symbols))
        labels = labels[valid]
        
        logger.info(f"Training with {len(features)} samples")
        
        # Train
        metrics = trainer.train(features, labels, epochs=20, batch_size=32)
        
        # Save
        output_dir = Path("data/models_mtf")
        output_dir.mkdir(exist_ok=True)
        trainer.save(output_dir / 'gnn_model.pt')
        
        logger.info(f"GNN training complete: {metrics}")
        
    except Exception as e:
        logger.warning(f"GNN training skipped: {e}")
    
    # Save GNN features
    output_dir = Path("data/models_mtf")
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / 'gnn_correlations.npy', correlations)
    np.save(output_dir / 'gnn_edge_index.npy', np.array(edge_index))
    np.save(output_dir / 'gnn_edge_weight.npy', np.array(edge_weight))
    
    logger.info("GNN data saved to data/models_mtf/")
    logger.info("Done!")


if __name__ == "__main__":
    main()