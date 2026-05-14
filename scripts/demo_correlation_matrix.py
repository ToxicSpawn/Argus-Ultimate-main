"""
Demonstration of DynamicCorrelationMatrix

This script shows how the DynamicCorrelationMatrix works with:
1. Different market regimes
2. Correlation updates from asset returns
3. Diversification metrics
4. Concentration alerts
"""

import logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from core.real_time_learning.orchestrator import RealTimeLearningOrchestrator
from core.real_time_learning.correlation_matrix import DynamicCorrelationMatrix

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_correlated_returns(assets, base_returns, correlation_matrix):
    """Generate correlated returns based on a base return series and correlation matrix"""
    # Convert to covariance matrix (simplified)
    n = len(assets)
    cov_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                cov_matrix[i,j] = 0.01  # Variance
            else:
                cov_matrix[i,j] = correlation_matrix[i,j] * np.sqrt(0.01 * 0.01)
    
    # Generate correlated returns using Cholesky decomposition
    L = np.linalg.cholesky(cov_matrix)
    random_shocks = np.random.normal(0, 1, (len(base_returns), n))
    correlated_returns = np.dot(random_shocks, L.T)
    
    # Scale to match base returns
    for i in range(n):
        correlated_returns[:,i] = correlated_returns[:,i] * np.std(base_returns) / np.std(correlated_returns[:,i])
        correlated_returns[:,i] = correlated_returns[:,i] + np.mean(base_returns)
    
    # Create return dictionary
    return_dict = {}
    for i, asset in enumerate(assets):
        return_dict[asset] = list(correlated_returns[:,i])
    
    return return_dict


def run_demo():
    """Run the dynamic correlation matrix demonstration"""
    
    # Create orchestrator and correlation matrix
    orchestrator = RealTimeLearningOrchestrator()
    corr_matrix = DynamicCorrelationMatrix()
    
    # Set update frequency to 1 for demo purposes
    corr_matrix.update_frequency = 1
    
    # Register the component
    orchestrator.register_component(corr_matrix)
    
    # Initialize with 3 assets
    assets = ["BTC", "ETH", "SOL"]
    corr_matrix.initialize_assets(assets)
    
    # Define base correlation matrices for different regimes
    regime_correlations = {
        'stable': np.array([
            [1.0, 0.6, 0.4],
            [0.6, 1.0, 0.5],
            [0.4, 0.5, 1.0]
        ]),
        'volatile': np.array([
            [1.0, 0.8, 0.7],
            [0.8, 1.0, 0.85],
            [0.7, 0.85, 1.0]
        ]),
        'trending': np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ]),
        'range': np.array([
            [1.0, 0.7, 0.6],
            [0.7, 1.0, 0.75],
            [0.6, 0.75, 1.0]
        ])
    }
    
    # Generate base returns (random walk)
    np.random.seed(42)
    base_returns = np.random.normal(0.001, 0.02, 100)
    
    print("=== Dynamic Correlation Matrix Demo ===\n")
    
    # Test different market regimes
    regimes = ['stable', 'volatile', 'trending', 'range']
    
    for regime in regimes:
        print(f"\n--- Testing {regime.upper()} Regime ---")
        
        # Generate correlated returns for this regime
        returns_data = generate_correlated_returns(assets, base_returns, regime_correlations[regime])
        
        # Create market data with regime characteristics
        market_data = {
            'volatility': 0.01,
            'trend_strength': 0.2
        }
        
        # Set explicit values to ensure regime detection works
        if regime == 'volatile':
            market_data['volatility'] = 0.03  # Above volatile threshold (0.02)
            market_data['trend_strength'] = 0.3
        elif regime == 'trending':
            market_data['volatility'] = 0.01
            market_data['trend_strength'] = 0.6  # Above trending threshold (0.5)
        elif regime == 'range':
            market_data['volatility'] = 0.005  # Below range threshold (0.008)
            market_data['trend_strength'] = 0.05
        elif regime == 'stable':
            market_data['volatility'] = 0.015  # Between thresholds
            market_data['trend_strength'] = 0.3
        
        # Create input data
        data = {
            'asset_returns': returns_data,
            'market_data': market_data
        }
        
        # Process through orchestrator
        results = orchestrator.on_market_data(data)
        
        # Get current state
        params = corr_matrix.get_params()
        current_regime = params['current_regime']
        concentration = params['portfolio_concentration']
        
        print(f"Detected regime: {current_regime}")
        print(f"Portfolio concentration: {concentration:.2f}")
        print(f"Diversification score: {corr_matrix.get_diversification_score():.2f}")
        
        print("\nCurrent correlation matrix:")
        for (a1, a2), corr in sorted(params['current_matrix'].items()):
            print(f"  {a1}-{a2}: {corr:.2f}")
        
        # Check for alerts
        alerts = corr_matrix.get_concentration_alerts()
        if alerts:
            print(f"\nAlerts ({len(alerts)}):")
            for alert in alerts:
                print(f"  {alert['level'].upper()}: {alert['message']}")
        else:
            print("\nNo alerts")
        
        print("\n" + "="*50)
    
    # Show final state
    print("\n=== Final System State ===")
    print(f"Matrix history length: {len(corr_matrix.matrix_history)}")
    print(f"Final diversification score: {corr_matrix.get_diversification_score():.2f}")
    
    # Show regime-adjusted correlation
    print("\nRegime-adjusted correlation examples:")
    for asset1 in assets:
        for asset2 in assets:
            if asset1 != asset2:
                corr = corr_matrix.get_regime_adjusted_correlation(asset1, asset2)
                print(f"  {asset1}-{asset2}: {corr:.2f}")


if __name__ == "__main__":
    run_demo()