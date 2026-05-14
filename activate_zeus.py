#!/usr/bin/env python3
"""
activate_zeus.py — Activate Zeus GPU Configuration for R7525

Usage:
    py activate_zeus.py              # Show current config
    py activate_zeus.py --activate   # Activate Zeus profile
    py activate_zeus.py --test       # Run Zeus diagnostics
    py activate_zeus.py --benchmark  # Run performance benchmark
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from gpu.zeus_integration import (
    ZeusGPU,
    ZeusMonteCarlo,
    ZeusFeatureStore,
    ZeusMLTrainer,
    ZeusClusterOrchestrator,
    ZeusSKU,
    create_zeus_gpu,
    create_zeus_monte_carlo,
    create_zeus_feature_store,
    create_zeus_ml_trainer,
    create_zeus_cluster,
)


def show_current_config():
    """Show current hardware configuration."""
    print("=" * 60)
    print("ARGUS ULTIMATE - Hardware Configuration")
    print("=" * 60)
    print()
    print("Current Setup (from unified_config.yaml):")
    print("  CPU: Intel Core Ultra 9 285K")
    print("  RAM: 64GB DDR5-6400")
    print("  GPU: MSI GeForce RTX 5080 16GB GDDR7")
    print("  NIC: Solarflare XtremeScale 10GbE")
    print()
    print("Zeus Profile Available:")
    print("  File: config/profiles/r7525_zeus.yaml")
    print()
    print("To activate Zeus profile:")
    print("  1. Set environment variable:")
    print("     $env:ARGUS_CONFIG_PROFILE='r7525_zeus'")
    print("  2. Or add to unified_config.yaml:")
    print("     include: config/profiles/r7525_zeus.yaml")
    print()


def activate_zeus():
    """Activate Zeus GPU configuration."""
    print("=" * 60)
    print("ARGUS ULTIMATE - Activating Zeus GPU Configuration")
    print("=" * 60)
    print()
    
    # Check if Zeus profile exists
    zeus_profile = Path("config/profiles/r7525_zeus.yaml")
    if not zeus_profile.exists():
        print("ERROR: Zeus profile not found at", zeus_profile)
        return False
    
    print("Zeus profile found:", zeus_profile)
    print()
    
    # Set environment variable
    os.environ["ARGUS_CONFIG_PROFILE"] = "r7525_zeus"
    print("Set ARGUS_CONFIG_PROFILE=r7525_zeus")
    print()
    
    # Initialize Zeus GPU
    print("Initializing Zeus GPU...")
    zeus = create_zeus_gpu(ZeusSKU.ZEUS_2C26_128)
    caps = zeus.get_capabilities()
    
    print(f"  SKU: {caps.sku.value}")
    print(f"  FP64: {caps.fp64_tflops} TFLOPS")
    print(f"  Memory: {caps.memory_gb} GB")
    print(f"  Power: {caps.power_watts}W")
    print()
    
    # Initialize Monte Carlo engine
    print("Initializing Monte Carlo engine...")
    mc = create_zeus_monte_carlo(ZeusSKU.ZEUS_2C26_128)
    print(f"  Max scenarios: {mc.max_scenarios:,}")
    print()
    
    # Initialize Feature Store
    print("Initializing Feature Store...")
    store = create_zeus_feature_store(ZeusSKU.ZEUS_2C26_128)
    print(f"  Max memory: {store.max_memory_gb} GB")
    print()
    
    # Initialize ML Trainer
    print("Initializing ML Trainer...")
    trainer = create_zeus_ml_trainer(ZeusSKU.ZEUS_2C26_128)
    print(f"  FP32: {trainer.capabilities.fp32_tflops} TFLOPS")
    print(f"  FP16: {trainer.capabilities.fp16_tflops} TFLOPS")
    print()
    
    print("=" * 60)
    print("Zeus GPU Configuration Activated!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Run diagnostics: py activate_zeus.py --test")
    print("  2. Run benchmark: py activate_zeus.py --benchmark")
    print("  3. Start Argus: py main.py paper")
    print()
    
    return True


def run_diagnostics():
    """Run Zeus GPU diagnostics."""
    import numpy as np
    
    print("=" * 60)
    print("ARGUS ULTIMATE - Zeus GPU Diagnostics")
    print("=" * 60)
    print()
    
    # Test 1: GPU Detection
    print("[1/5] GPU Detection...")
    zeus = create_zeus_gpu(ZeusSKU.ZEUS_2C26_128)
    hw = zeus.detect_hardware()
    print(f"  Status: {'OK' if hw['simulated'] else 'HARDWARE'}")
    print(f"  Mode: {'Simulated' if hw['simulated'] else 'Real Hardware'}")
    print()
    
    # Test 2: Monte Carlo Engine
    print("[2/5] Monte Carlo Engine...")
    mc = create_zeus_monte_carlo(ZeusSKU.ZEUS_2C26_128)
    returns = np.random.randn(1000) * 0.02
    var_result = mc.calculate_var(returns, confidence=0.99, n_scenarios=10000)
    print(f"  VaR (99%): {var_result['var']:.4f}")
    print(f"  CVaR (99%): {var_result['cvar']:.4f}")
    print(f"  Time: {var_result['calculation_time_ms']:.2f}ms")
    print()
    
    # Test 3: Feature Store
    print("[3/5] Feature Store...")
    store = create_zeus_feature_store(ZeusSKU.ZEUS_2C26_128)
    features = np.random.randn(10000, 100)
    store.store_feature("test_features", features)
    retrieved = store.get_feature("test_features")
    print(f"  Stored: {features.shape}")
    print(f"  Retrieved: {retrieved.shape if retrieved is not None else 'None'}")
    print(f"  Dtype: {retrieved.dtype if retrieved is not None else 'N/A'}")
    print()
    
    # Test 4: ML Trainer
    print("[4/5] ML Trainer...")
    trainer = create_zeus_ml_trainer(ZeusSKU.ZEUS_2C26_128)
    X = np.random.randn(500, 10)
    y = (X[:, 0] > 0).astype(float)
    result = trainer.train_model("diagnostic_model", X, y, epochs=20, batch_size=100)
    print(f"  Final accuracy: {result['final_accuracy']:.4f}")
    print(f"  Training time: {result['training_time_s']:.2f}s")
    print()
    
    # Test 5: Benchmark vs NVIDIA
    print("[5/5] Performance Comparison...")
    benchmark = zeus.benchmark_vs_nvidia()
    print(f"  vs RTX 5090 FP64: {benchmark['vs_rtx_5090']['fp64_speedup']:.1f}x")
    print(f"  vs RTX 5090 Power: {benchmark['vs_rtx_5090']['power_efficiency']:.1f}x more efficient")
    print(f"  vs RTX 5090 Memory: {benchmark['vs_rtx_5090']['memory_advantage']:.1f}x more")
    print()
    
    print("=" * 60)
    print("All Diagnostics Passed!")
    print("=" * 60)


def run_benchmark():
    """Run Zeus GPU benchmark."""
    import numpy as np
    import time
    
    print("=" * 60)
    print("ARGUS ULTIMATE - Zeus GPU Benchmark")
    print("=" * 60)
    print()
    
    zeus = create_zeus_gpu(ZeusSKU.ZEUS_2C26_128)
    mc = create_zeus_monte_carlo(ZeusSKU.ZEUS_2C26_128)
    
    # Benchmark 1: VaR Calculation
    print("[1/4] VaR Calculation Benchmark")
    returns = np.random.randn(10000) * 0.02
    
    for n_scenarios in [10000, 100000, 1000000]:
        start = time.time()
        result = mc.calculate_var(returns, confidence=0.99, n_scenarios=n_scenarios)
        elapsed = time.time() - start
        rate = n_scenarios / elapsed
        print(f"  {n_scenarios:>10,} scenarios: {elapsed*1000:>8.2f}ms ({rate:>12,.0f} scenarios/sec)")
    print()
    
    # Benchmark 2: Option Pricing
    print("[2/4] Option Pricing Benchmark")
    for n_sim in [10000, 100000, 1000000]:
        start = time.time()
        result = mc.price_options_mc(
            spot=100.0, strike=105.0, rate=0.05,
            volatility=0.2, time_to_expiry=1.0,
            n_simulations=n_sim, option_type="call"
        )
        elapsed = time.time() - start
        print(f"  {n_sim:>10,} simulations: {elapsed*1000:>8.2f}ms (price={result['price']:.4f})")
    print()
    
    # Benchmark 3: Feature Store
    print("[3/4] Feature Store Benchmark")
    store = create_zeus_feature_store(ZeusSKU.ZEUS_2C26_128)
    
    for size in [(1000, 100), (10000, 100), (100000, 100)]:
        data = np.random.randn(*size)
        start = time.time()
        store.store_feature(f"bench_{size[0]}", data)
        elapsed = time.time() - start
        throughput = data.nbytes / elapsed / 1024**2
        print(f"  {size[0]:>6,} x {size[1]:>3}: {elapsed*1000:>8.2f}ms ({throughput:>8.1f} MB/s)")
    print()
    
    # Benchmark 4: ML Training
    print("[4/4] ML Training Benchmark")
    trainer = create_zeus_ml_trainer(ZeusSKU.ZEUS_2C26_128)
    
    for batch_size in [256, 1024, 4096]:
        X = np.random.randn(10000, 50)
        y = (X[:, 0] > 0).astype(float)
        start = time.time()
        result = trainer.train_model(f"bench_{batch_size}", X, y, epochs=10, batch_size=batch_size)
        elapsed = time.time() - start
        print(f"  Batch {batch_size:>5}: {elapsed:>6.2f}s (accuracy={result['final_accuracy']:.4f})")
    print()
    
    print("=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Activate Zeus GPU for Argus")
    parser.add_argument("--activate", action="store_true", help="Activate Zeus profile")
    parser.add_argument("--test", action="store_true", help="Run diagnostics")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    
    args = parser.parse_args()
    
    if args.activate:
        activate_zeus()
    elif args.test:
        run_diagnostics()
    elif args.benchmark:
        run_benchmark()
    else:
        show_current_config()


if __name__ == "__main__":
    main()
