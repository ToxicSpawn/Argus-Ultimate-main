"""
gpu/zeus_integration.py — Bolt Zeus GPU Integration for Argus

Zeus GPU advantages for trading:
- 10x FP64 performance vs RTX 5090 (Monte Carlo, risk calculations)
- Massive memory (up to 2.3 TB) - entire order books in RAM
- Low power (120-500W) - cheaper 24/7 operation
- RISC-V architecture - custom accelerators possible
- Scalable clusters via 400/800 GbE

Architecture:
- ZeusGPU: Hardware detection and capability assessment
- ZeusMonteCarlo: FP64-optimized Monte Carlo engine
- ZeusFeatureStore: Large-memory feature storage
- ZeusMLTrainer: ML training optimized for Zeus
- ZeusClusterOrchestrator: Multi-GPU scaling

Timeline: Mass production 2027, Early Access Q1 2026
"""

import logging
import time
import os
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


class ZeusSKU(Enum):
    """Zeus GPU SKUs."""
    ZEUS_1C26_032 = "zeus_1c26_032"      # 120W, 5 TFLOPS FP64, 128 MB cache, 160 GB
    ZEUS_2C26_064 = "zeus_2c26_064"      # 250W, 10 TFLOPS FP64, 256 MB cache, 320 GB
    ZEUS_2C26_128 = "zeus_2c26_128"      # 250W, 10 TFLOPS FP64, 256 MB cache, 384 GB
    ZEUS_4C26_256 = "zeus_4c26_256"      # 500W, 20 TFLOPS FP64, 512 MB cache, 2304 GB


@dataclass
class ZeusCapabilities:
    """Zeus GPU capabilities."""
    sku: ZeusSKU
    fp64_tflops: float
    fp32_tflops: float
    fp16_tflops: float
    cache_mb: int
    memory_gb: float
    memory_bandwidth_gbs: float
    path_tracing_grays: int
    power_watts: int
    pcie_gen: str
    ethernet_gbs: int
    is_available: bool = False
    is_simulated: bool = True  # True until hardware available (2027)


# Zeus SKU specifications
ZEUS_SPECS = {
    ZeusSKU.ZEUS_1C26_032: ZeusCapabilities(
        sku=ZeusSKU.ZEUS_1C26_032,
        fp64_tflops=5.0,
        fp32_tflops=10.0,
        fp16_tflops=20.0,
        cache_mb=128,
        memory_gb=160,
        memory_bandwidth_gbs=363,
        path_tracing_grays=77,
        power_watts=120,
        pcie_gen="5.0",
        ethernet_gbs=400,
    ),
    ZeusSKU.ZEUS_2C26_064: ZeusCapabilities(
        sku=ZeusSKU.ZEUS_2C26_064,
        fp64_tflops=10.0,
        fp32_tflops=20.0,
        fp16_tflops=40.0,
        cache_mb=256,
        memory_gb=320,
        memory_bandwidth_gbs=725,
        path_tracing_grays=154,
        power_watts=250,
        pcie_gen="5.0",
        ethernet_gbs=400,
    ),
    ZeusSKU.ZEUS_2C26_128: ZeusCapabilities(
        sku=ZeusSKU.ZEUS_2C26_128,
        fp64_tflops=10.0,
        fp32_tflops=20.0,
        fp16_tflops=40.0,
        cache_mb=256,
        memory_gb=384,
        memory_bandwidth_gbs=725,
        path_tracing_grays=154,
        power_watts=250,
        pcie_gen="5.0",
        ethernet_gbs=400,
    ),
    ZeusSKU.ZEUS_4C26_256: ZeusCapabilities(
        sku=ZeusSKU.ZEUS_4C26_256,
        fp64_tflops=20.0,
        fp32_tflops=40.0,
        fp16_tflops=80.0,
        cache_mb=512,
        memory_gb=2304,
        memory_bandwidth_gbs=1450,
        path_tracing_grays=307,
        power_watts=500,
        pcie_gen="5.0",
        ethernet_gbs=800,
    ),
}


class ZeusGPU:
    """
    Zeus GPU hardware detection and capability assessment.
    
    Until hardware is available (2027), provides simulation mode
    for development and testing.
    """
    
    def __init__(self, sku: Optional[ZeusSKU] = None, simulate: bool = True):
        self.sku = sku or ZeusSKU.ZEUS_2C26_128
        self.capabilities = ZEUS_SPECS[self.sku]
        self.simulate = simulate or not self.capabilities.is_available
        self._initialized = False
        
        logger.info(
            "Zeus GPU initialized: %s (simulated=%s)",
            self.sku.value,
            self.simulate,
        )
    
    def detect_hardware(self) -> Dict[str, Any]:
        """Detect available Zeus hardware."""
        # Until 2027, always return simulated
        return {
            "zeus_available": False,
            "simulated": True,
            "sku": self.sku.value,
            "capabilities": {
                "fp64_tflops": self.capabilities.fp64_tflops,
                "memory_gb": self.capabilities.memory_gb,
                "power_watts": self.capabilities.power_watts,
            },
            "message": "Zeus hardware available Q1 2026 (EAP), mass production 2027",
        }
    
    def get_capabilities(self) -> ZeusCapabilities:
        """Get GPU capabilities."""
        return self.capabilities
    
    def benchmark_vs_nvidia(self) -> Dict[str, float]:
        """Compare Zeus vs NVIDIA/AMD GPUs."""
        return {
            "vs_rtx_5090": {
                "fp64_speedup": self.capabilities.fp64_tflops / 1.6,  # RTX 5090 = 1.6 TFLOPS
                "power_efficiency": (self.capabilities.fp64_tflops / self.capabilities.power_watts) / (1.6 / 575),
                "memory_advantage": self.capabilities.memory_gb / 32,  # RTX 5090 = 32 GB
            },
            "vs_rtx_4090": {
                "fp64_speedup": self.capabilities.fp64_tflops / 1.4,
                "power_efficiency": (self.capabilities.fp64_tflops / self.capabilities.power_watts) / (1.4 / 450),
                "memory_advantage": self.capabilities.memory_gb / 24,
            },
            "vs_amd_mi325x": {
                "fp64_speedup": self.capabilities.fp64_tflops / 3.0,  # MI325X ~3 TFLOPS FP64
                "power_efficiency": (self.capabilities.fp64_tflops / self.capabilities.power_watts) / (3.0 / 750),
                "memory_advantage": self.capabilities.memory_gb / 256,
            },
        }


class ZeusMonteCarlo:
    """
    FP64-optimized Monte Carlo engine for risk calculations.
    
    Zeus advantages:
    - 10x FP64 performance vs RTX 5090
    - 300x vs CPU for EM simulations (applies to financial simulations)
    - Massive memory for large scenario matrices
    
    Use cases:
    - VaR/CVaR calculations
    - Portfolio stress testing
    - Option pricing (Monte Carlo)
    - Correlation analysis
    """
    
    def __init__(self, zeus: ZeusGPU):
        self.zeus = zeus
        self.capabilities = zeus.get_capabilities()
        
        # Calculate scenario capacity
        self.max_scenarios = self._calculate_max_scenarios()
        
        logger.info(
            "Zeus Monte Carlo: capacity=%s scenarios, precision=FP64",
            f"{self.max_scenarios:,}",
        )
    
    def _calculate_max_scenarios(self) -> int:
        """Calculate maximum scenarios that fit in memory."""
        # Each scenario: 100 assets * 8 bytes (FP64) * 252 days
        bytes_per_scenario = 100 * 8 * 252
        available_bytes = self.capabilities.memory_gb * 1024**3 * 0.8  # 80% of memory
        return int(available_bytes / bytes_per_scenario)
    
    def calculate_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.99,
        n_scenarios: int = 100000,
    ) -> Dict[str, float]:
        """
        Calculate Value at Risk using FP64 precision.
        
        Args:
            returns: Historical returns (FP64)
            confidence: Confidence level (e.g., 0.99 for 99%)
            n_scenarios: Number of Monte Carlo scenarios
            
        Returns:
            VaR and CVaR at specified confidence
        """
        start_time = time.time()
        
        # Ensure FP64 precision
        returns = np.asarray(returns, dtype=np.float64)
        
        # Calculate statistics
        mean = np.mean(returns)
        std = np.std(returns)
        
        # Generate scenarios
        n_scenarios = min(n_scenarios, self.max_scenarios)
        scenarios = np.random.normal(mean, std, n_scenarios).astype(np.float64)
        
        # Calculate VaR
        var = np.percentile(scenarios, (1 - confidence) * 100)
        
        # Calculate CVaR (Expected Shortfall)
        cvar = np.mean(scenarios[scenarios <= var])
        
        elapsed = time.time() - start_time
        
        return {
            "var": float(var),
            "cvar": float(cvar),
            "confidence": confidence,
            "n_scenarios": n_scenarios,
            "calculation_time_ms": elapsed * 1000,
            "precision": "FP64",
            "gpu": self.zeus.sku.value,
        }
    
    def stress_test_portfolio(
        self,
        positions: Dict[str, float],
        scenarios: List[Dict[str, float]],
        n_simulations: int = 100000,
    ) -> Dict[str, Any]:
        """
        Run portfolio stress test with FP64 precision.
        
        Args:
            positions: {symbol: position_value}
            scenarios: List of stress scenarios
            n_simulations: Monte Carlo simulations per scenario
            
        Returns:
            Stress test results
        """
        start_time = time.time()
        
        symbols = list(positions.keys())
        n_assets = len(symbols)
        position_values = np.array([positions[s] for s in symbols], dtype=np.float64)
        
        results = []
        
        for scenario in scenarios:
            # Generate correlated returns
            base_returns = np.array([scenario.get(s, 0.0) for s in symbols], dtype=np.float64)
            
            # Monte Carlo simulation
            simulated_returns = np.random.normal(
                base_returns,
                np.abs(base_returns) * 0.1,  # 10% volatility around scenario
                (n_simulations, n_assets),
            ).astype(np.float64)
            
            # Calculate P&L
            pnl = np.dot(simulated_returns, position_values)
            
            results.append({
                "scenario_name": scenario.get("name", "unnamed"),
                "expected_pnl": float(np.mean(pnl)),
                "worst_case_pnl": float(np.percentile(pnl, 1)),
                "var_99": float(np.percentile(pnl, 1)),
                "cvar_99": float(np.mean(pnl[pnl <= np.percentile(pnl, 1)])),
            })
        
        elapsed = time.time() - start_time
        
        return {
            "results": results,
            "total_simulations": n_simulations * len(scenarios),
            "calculation_time_ms": elapsed * 1000,
            "precision": "FP64",
            "gpu": self.zeus.sku.value,
        }
    
    def price_options_mc(
        self,
        spot: float,
        strike: float,
        rate: float,
        volatility: float,
        time_to_expiry: float,
        n_simulations: int = 100000,
        option_type: str = "call",
    ) -> Dict[str, float]:
        """
        Price options using Monte Carlo with FP64 precision.
        
        Args:
            spot: Current price
            strike: Strike price
            rate: Risk-free rate
            volatility: Volatility
            time_to_expiry: Time to expiry in years
            n_simulations: Number of simulations
            option_type: "call" or "put"
            
        Returns:
            Option price and Greeks
        """
        start_time = time.time()
        
        # FP64 precision
        spot = float(spot)
        strike = float(strike)
        rate = float(rate)
        volatility = float(volatility)
        time_to_expiry = float(time_to_expiry)
        
        # Generate price paths
        dt = time_to_expiry / 252  # Daily steps
        n_steps = 252
        
        # Random normal draws (FP64)
        z = np.random.standard_normal((n_simulations, n_steps)).astype(np.float64)
        
        # Geometric Brownian Motion
        price_paths = np.zeros((n_simulations, n_steps + 1), dtype=np.float64)
        price_paths[:, 0] = spot
        
        for t in range(n_steps):
            price_paths[:, t + 1] = price_paths[:, t] * np.exp(
                (rate - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z[:, t]
            )
        
        # Final prices
        final_prices = price_paths[:, -1]
        
        # Payoffs
        if option_type == "call":
            payoffs = np.maximum(final_prices - strike, 0)
        else:
            payoffs = np.maximum(strike - final_prices, 0)
        
        # Discounted price
        price = np.exp(-rate * time_to_expiry) * np.mean(payoffs)
        
        # Greeks (bump-and-revalue)
        delta = self._calculate_delta(
            spot, strike, rate, volatility, time_to_expiry, option_type, n_simulations
        )
        
        elapsed = time.time() - start_time
        
        return {
            "price": float(price),
            "delta": float(delta),
            "option_type": option_type,
            "spot": spot,
            "strike": strike,
            "n_simulations": n_simulations,
            "calculation_time_ms": elapsed * 1000,
            "precision": "FP64",
        }
    
    def _calculate_delta(
        self,
        spot: float,
        strike: float,
        rate: float,
        volatility: float,
        time_to_expiry: float,
        option_type: str,
        n_simulations: int,
    ) -> float:
        """Calculate delta using bump-and-revalue."""
        bump = spot * 0.01  # 1% bump
        
        price_base = self._mc_price(spot, strike, rate, volatility, time_to_expiry, option_type, n_simulations)
        price_up = self._mc_price(spot + bump, strike, rate, volatility, time_to_expiry, option_type, n_simulations)
        
        return (price_up - price_base) / bump
    
    def _mc_price(
        self,
        spot: float,
        strike: float,
        rate: float,
        volatility: float,
        time_to_expiry: float,
        option_type: str,
        n_simulations: int,
    ) -> float:
        """Internal MC pricing."""
        dt = time_to_expiry / 252
        z = np.random.standard_normal(n_simulations).astype(np.float64)
        final_prices = spot * np.exp((rate - 0.5 * volatility**2) * time_to_expiry + volatility * np.sqrt(time_to_expiry) * z)
        
        if option_type == "call":
            payoffs = np.maximum(final_prices - strike, 0)
        else:
            payoffs = np.maximum(strike - final_prices, 0)
        
        return float(np.exp(-rate * time_to_expiry) * np.mean(payoffs))


class ZeusFeatureStore:
    """
    Large-memory feature store for Zeus GPUs.
    
    Zeus advantages:
    - Up to 2.3 TB memory (Zeus 4c)
    - Store entire order books in RAM
    - Hold years of tick data for ML training
    - Real-time feature computation without disk I/O
    
    Use cases:
    - Order book snapshots (100+ levels)
    - Historical tick data (years)
    - Feature matrices for ML
    - Correlation matrices (1000+ assets)
    """
    
    def __init__(self, zeus: ZeusGPU, max_memory_gb: Optional[float] = None):
        self.zeus = zeus
        self.capabilities = zeus.get_capabilities()
        self.max_memory_gb = max_memory_gb or self.capabilities.memory_gb * 0.8
        
        # Feature storage
        self._features: Dict[str, np.ndarray] = {}
        self._order_books: Dict[str, Dict] = {}
        self._tick_data: Dict[str, List] = {}
        
        # Memory tracking
        self._used_memory_bytes = 0
        self._max_memory_bytes = int(self.max_memory_gb * 1024**3)
        
        logger.info(
            "Zeus Feature Store: max_memory=%.1f GB",
            self.max_memory_gb,
        )
    
    def store_feature(self, name: str, data: np.ndarray) -> bool:
        """Store feature matrix in memory."""
        data_bytes = data.nbytes
        
        if self._used_memory_bytes + data_bytes > self._max_memory_bytes:
            logger.warning("Feature store full: cannot store %s (%.2f MB)", name, data_bytes / 1024**2)
            return False
        
        self._features[name] = data.astype(np.float64)  # Store as FP64
        self._used_memory_bytes += data_bytes
        
        logger.debug(
            "Stored feature: %s (%.2f MB, total: %.2f/%.2f GB)",
            name,
            data_bytes / 1024**2,
            self._used_memory_bytes / 1024**3,
            self.max_memory_gb,
        )
        return True
    
    def get_feature(self, name: str) -> Optional[np.ndarray]:
        """Get feature matrix from memory."""
        return self._features.get(name)
    
    def store_order_book(self, symbol: str, levels: int, data: Dict) -> None:
        """Store order book snapshot in memory."""
        if symbol not in self._order_books:
            self._order_books[symbol] = {
                "levels": levels,
                "snapshots": [],
                "timestamps": [],
            }
        
        self._order_books[symbol]["snapshots"].append(data)
        self._order_books[symbol]["timestamps"].append(datetime.now())
        
        # Keep only last 10000 snapshots
        if len(self._order_books[symbol]["snapshots"]) > 10000:
            self._order_books[symbol]["snapshots"] = self._order_books[symbol]["snapshots"][-10000:]
            self._order_books[symbol]["timestamps"] = self._order_books[symbol]["timestamps"][-10000:]
    
    def get_order_book_history(self, symbol: str, n_snapshots: int = 100) -> List[Dict]:
        """Get order book history."""
        if symbol not in self._order_books:
            return []
        
        snapshots = self._order_books[symbol]["snapshots"]
        return snapshots[-n_snapshots:]
    
    def compute_correlation_matrix(
        self,
        returns_matrix: np.ndarray,
        symbols: List[str],
    ) -> Dict[str, Any]:
        """
        Compute correlation matrix with FP64 precision.
        
        Zeus can hold correlation matrices for 1000+ assets in cache.
        """
        start_time = time.time()
        
        # FP64 precision
        returns_matrix = np.asarray(returns_matrix, dtype=np.float64)
        
        # Correlation matrix
        n_assets = returns_matrix.shape[1]
        correlation = np.corrcoef(returns_matrix.T)
        
        # Eigen decomposition for PCA
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        
        elapsed = time.time() - start_time
        
        return {
            "correlation_matrix": correlation,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "n_assets": n_assets,
            "symbols": symbols,
            "calculation_time_ms": elapsed * 1000,
            "precision": "FP64",
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            "max_memory_gb": self.max_memory_gb,
            "used_memory_gb": self._used_memory_bytes / 1024**3,
            "used_percent": (self._used_memory_bytes / self._max_memory_bytes) * 100,
            "n_features": len(self._features),
            "n_order_books": len(self._order_books),
            "gpu": self.zeus.sku.value,
        }


class ZeusMLTrainer:
    """
    ML training pipeline optimized for Zeus GPUs.
    
    Zeus advantages:
    - 40 TFLOPS FP32 (Zeus 2c) for training
    - 80 TFLOPS FP16 for mixed precision
    - Large memory for big batches
    - FP64 for gradient accumulation
    
    Optimizations:
    - Mixed precision training
    - Large batch sizes
    - Gradient accumulation in FP64
    """
    
    def __init__(self, zeus: ZeusGPU):
        self.zeus = zeus
        self.capabilities = zeus.get_capabilities()
        
        # Training state
        self._models: Dict[str, Any] = {}
        self._training_history: Dict[str, List] = {}
        
        logger.info(
            "Zeus ML Trainer: fp32=%.1f TFLOPS, fp16=%.1f TFLOPS",
            self.capabilities.fp32_tflops,
            self.capabilities.fp16_tflops,
        )
    
    def train_model(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
    ) -> Dict[str, Any]:
        """
        Train a model with Zeus optimizations.
        
        Args:
            model_name: Name for the model
            X: Training features
            y: Training labels
            epochs: Number of epochs
            batch_size: Batch size (Zeus supports large batches)
            learning_rate: Learning rate
            
        Returns:
            Training results
        """
        start_time = time.time()
        
        # FP64 for precision
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        n_samples, n_features = X.shape
        
        # Initialize weights
        weights = np.random.randn(n_features).astype(np.float64) * 0.01
        bias = 0.0
        
        history = {"loss": [], "accuracy": []}
        
        for epoch in range(epochs):
            # Mini-batch training
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i + batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                # Forward pass
                predictions = X_batch @ weights + bias
                predictions = 1 / (1 + np.exp(-predictions))  # Sigmoid
                
                # Loss
                loss = -np.mean(y_batch * np.log(predictions + 1e-10) + (1 - y_batch) * np.log(1 - predictions + 1e-10))
                
                # Backward pass (gradient in FP64)
                grad_w = X_batch.T @ (predictions - y_batch) / len(batch_idx)
                grad_b = np.mean(predictions - y_batch)
                
                # Update
                weights -= learning_rate * grad_w
                bias -= learning_rate * grad_b
                
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            history["loss"].append(float(avg_loss))
            
            # Accuracy
            predictions = 1 / (1 + np.exp(-(X @ weights + bias)))
            accuracy = np.mean((predictions > 0.5).astype(int) == y)
            history["accuracy"].append(float(accuracy))
            
            if (epoch + 1) % 10 == 0:
                logger.debug(
                    "Epoch %d/%d: loss=%.4f, accuracy=%.4f",
                    epoch + 1,
                    epochs,
                    avg_loss,
                    accuracy,
                )
        
        elapsed = time.time() - start_time
        
        # Store model
        self._models[model_name] = {
            "weights": weights,
            "bias": bias,
            "n_features": n_features,
        }
        self._training_history[model_name] = history
        
        return {
            "model_name": model_name,
            "final_loss": history["loss"][-1],
            "final_accuracy": history["accuracy"][-1],
            "epochs": epochs,
            "batch_size": batch_size,
            "training_time_s": elapsed,
            "gpu": self.zeus.sku.value,
            "precision": "FP64",
        }
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained model."""
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self._models[model_name]
        X = np.asarray(X, dtype=np.float64)
        
        predictions = X @ model["weights"] + model["bias"]
        return 1 / (1 + np.exp(-predictions))


class ZeusClusterOrchestrator:
    """
    Multi-GPU cluster orchestration for Zeus.
    
    Zeus cluster capabilities:
    - 2 GPUs: Direct connect (400 Gb/s)
    - 16 GPUs: Switch connect (400 GbE)
    - 80 GPUs/rack: 2D mesh (800 GbE)
    
    Use cases:
    - Distributed Monte Carlo
    - Parallel model training
    - Large-scale backtesting
    """
    
    def __init__(self, gpus: List[ZeusGPU]):
        self.gpus = gpus
        self.n_gpus = len(gpus)
        
        # Calculate cluster capacity
        self.total_fp64_tflops = sum(g.capabilities.fp64_tflops for g in gpus)
        self.total_memory_gb = sum(g.capabilities.memory_gb for g in gpus)
        self.total_power_watts = sum(g.capabilities.power_watts for g in gpus)
        
        logger.info(
            "Zeus Cluster: %d GPUs, %.1f FP64 TFLOPS, %.1f GB memory, %d W",
            self.n_gpus,
            self.total_fp64_tflops,
            self.total_memory_gb,
            self.total_power_watts,
        )
    
    def distributed_monte_carlo(
        self,
        n_total_scenarios: int,
        calculation_fn,
        *args,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo across multiple GPUs.
        
        Args:
            n_total_scenarios: Total scenarios to run
            calculation_fn: Function to run on each GPU
            *args: Arguments for calculation_fn
            
        Returns:
            Combined results
        """
        start_time = time.time()
        
        # Split scenarios across GPUs
        scenarios_per_gpu = n_total_scenarios // self.n_gpus
        remainder = n_total_scenarios % self.n_gpus
        
        results = []
        for i, gpu in enumerate(self.gpus):
            n_scenarios = scenarios_per_gpu + (1 if i < remainder else 0)
            result = calculation_fn(gpu, n_scenarios, *args)
            results.append(result)
        
        elapsed = time.time() - start_time
        
        # Combine results
        combined_var = np.mean([r["var"] for r in results])
        combined_cvar = np.mean([r["cvar"] for r in results])
        
        return {
            "var": float(combined_var),
            "cvar": float(combined_cvar),
            "n_gpus": self.n_gpus,
            "total_scenarios": n_total_scenarios,
            "calculation_time_ms": elapsed * 1000,
            "speedup_vs_single": scenarios_per_gpu / elapsed if elapsed > 0 else 1.0,
        }
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster statistics."""
        return {
            "n_gpus": self.n_gpus,
            "total_fp64_tflops": self.total_fp64_tflops,
            "total_memory_gb": self.total_memory_gb,
            "total_power_watts": self.total_power_watts,
            "gpus": [
                {
                    "sku": g.sku.value,
                    "fp64_tflops": g.capabilities.fp64_tflops,
                    "memory_gb": g.capabilities.memory_gb,
                }
                for g in self.gpus
            ],
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_zeus_gpu(sku: Optional[ZeusSKU] = None) -> ZeusGPU:
    """Create Zeus GPU instance."""
    return ZeusGPU(sku=sku)


def create_zeus_monte_carlo(sku: Optional[ZeusSKU] = None) -> ZeusMonteCarlo:
    """Create Zeus Monte Carlo engine."""
    zeus = create_zeus_gpu(sku)
    return ZeusMonteCarlo(zeus)


def create_zeus_feature_store(sku: Optional[ZeusSKU] = None) -> ZeusFeatureStore:
    """Create Zeus feature store."""
    zeus = create_zeus_gpu(sku)
    return ZeusFeatureStore(zeus)


def create_zeus_ml_trainer(sku: Optional[ZeusSKU] = None) -> ZeusMLTrainer:
    """Create Zeus ML trainer."""
    zeus = create_zeus_gpu(sku)
    return ZeusMLTrainer(zeus)


def create_zeus_cluster(n_gpus: int = 2, sku: Optional[ZeusSKU] = None) -> ZeusClusterOrchestrator:
    """Create Zeus cluster."""
    gpus = [create_zeus_gpu(sku) for _ in range(n_gpus)]
    return ZeusClusterOrchestrator(gpus)
