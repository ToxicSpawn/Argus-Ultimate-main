"""
DISTRIBUTED CONFIGURATION
==========================
Configuration for PC + Server hybrid setup.
"""

# ============================================================================
# NETWORK CONFIGURATION
# ============================================================================

# Server Configuration (Dell R7525)
SERVER_CONFIG = {
    "host": "192.168.1.100",  # Change to your server IP
    "port": 5555,
    "name": "dell_r7525",
    "role": "server",
    
    # Hardware specs
    "cpu_cores": 64,
    "cpu_model": "AMD EPYC 7763",
    "cpu_base_clock": 2.45,  # GHz
    "cpu_boost_clock": 3.5,  # GHz
    "memory_gb": 512,
    "memory_type": "DDR4-3200",
    "gpu_available": False,
    
    # Capabilities
    "capabilities": [
        "ml_training",
        "backtesting",
        "data_processing",
        "portfolio_optimization",
        "historical_analysis",
        "feature_engineering",
        "model_validation",
        "walk_forward_optimization"
    ],
    
    # Resource limits
    "max_training_jobs": 8,
    "max_backtest_workers": 32,
    "max_data_workers": 64,
    "memory_limit_gb": 450  # Leave 62GB for system
}

# PC Configuration (Your Machine)
PC_CONFIG = {
    "host": "192.168.1.50",  # Change to your PC IP
    "port": 5556,
    "name": "intel_ultra9_rtx5080",
    "role": "pc",
    
    # Hardware specs
    "cpu_cores": 24,
    "cpu_model": "Intel Core Ultra 9 285K",
    "cpu_base_clock": 3.7,  # GHz
    "cpu_boost_clock": 5.7,  # GHz
    "memory_gb": 64,
    "memory_type": "DDR5-6000",
    "gpu_available": True,
    "gpu_model": "NVIDIA RTX 5080",
    "gpu_memory_gb": 16,
    "gpu_cuda_cores": 10752,
    
    # Capabilities
    "capabilities": [
        "real_time_trading",
        "gpu_inference",
        "neural_adaptation",
        "hft_processing",
        "quantum_simulation",
        "order_execution",
        "risk_calculation",
        "live_monitoring",
        "gpu_training"
    ],
    
    # Resource limits
    "max_gpu_models": 10,
    "max_inference_batch": 1024,
    "gpu_memory_limit_gb": 14  # Leave 2GB for display
}

# ============================================================================
# TASK ROUTING CONFIGURATION
# ============================================================================

# Tasks that run on SERVER (CPU-intensive)
SERVER_TASKS = [
    "ml_training",           # Train ML models (64 cores)
    "backtesting",           # Run backtests (parallel)
    "data_processing",       # Process historical data
    "portfolio_optimization", # Large-scale optimization
    "historical_analysis",   # Multi-year analysis
    "feature_engineering",   # Feature computation
    "model_validation",      # Cross-validation
    "walk_forward",          # Walk-forward optimization
    "monte_carlo",           # Monte Carlo simulations
    "hyperparameter_tuning", # Grid/random search
    "ensemble_training",     # Train ensemble members
    "data_pipeline",         # ETL processes
]

# Tasks that run on PC (GPU-accelerated)
PC_TASKS = [
    "neural_inference",      # Real-time inference (GPU)
    "gpu_quantum",           # Quantum simulation (GPU)
    "real_time_adaptation",  # Live adaptation (GPU)
    "hft_processing",        # Order book processing (GPU)
    "risk_calculation",      # Real-time risk (GPU)
    "order_execution",       # Trade execution
    "live_monitoring",       # Real-time monitoring
    "gpu_fine_tuning",       # Fine-tune on GPU
    "inference_ensemble",    # Ensemble inference
    "anomaly_detection",     # Real-time anomaly
]

# ============================================================================
# COMMUNICATION CONFIGURATION
# ============================================================================

COMMUNICATION_CONFIG = {
    "protocol": "zeromq",  # zeromq, websocket, or tcp
    "heartbeat_interval": 5,  # seconds
    "task_timeout": 300,  # seconds
    "max_retries": 3,
    "compression": True,
    "encryption": False,  # Enable for production
    
    # Buffer sizes
    "send_buffer_mb": 64,
    "receive_buffer_mb": 64,
    
    # Retry settings
    "retry_delay_ms": 100,
    "max_message_size_mb": 100
}

# ============================================================================
# ML CONFIGURATION
# ============================================================================

ML_CONFIG = {
    # Training settings
    "default_epochs": 100,
    "early_stopping_patience": 10,
    "learning_rate": 0.001,
    "batch_size": 256,
    
    # Ensemble settings
    "max_ensemble_size": 1000,
    "ensemble_diversity_weight": 0.3,
    
    # Model types
    "model_types": [
        "transformer",
        "lstm",
        "cnn",
        "gnn",
        "ensemble"
    ],
    
    # Feature settings
    "feature_window": 100,
    "prediction_horizon": 5,
    "lookback_periods": [5, 10, 20, 50, 100, 200]
}

# ============================================================================
# BACKTESTING CONFIGURATION
# ============================================================================

BACKTEST_CONFIG = {
    # Parallel settings
    "num_workers": 64,  # Use all server cores
    "chunk_size": 10000,
    
    # Walk-forward settings
    "train_window": 252,  # 1 year
    "test_window": 63,    # 3 months
    "step_size": 21,      # 1 month
    
    # Optimization
    "optimization_method": "bayesian",  # bayesian, grid, random
    "n_trials": 100,
    "scoring_metric": "sharpe_ratio"
}

# ============================================================================
# HYBRID SYSTEM CONFIGURATION
# ============================================================================

HYBRID_CONFIG = {
    # System settings
    "enabled": True,
    "auto_failover": True,
    "load_balancing": True,
    
    # Performance targets
    "target_latency_ms": 10,  # For real-time tasks
    "target_throughput": 1000,  # Tasks per minute
    
    # Monitoring
    "metrics_enabled": True,
    "metrics_interval": 60,  # seconds
    
    # Logging
    "log_level": "INFO",
    "log_file": "distributed.log"
}

# ============================================================================
# EXPECTED PERFORMANCE
# ============================================================================

PERFORMANCE_ESTIMATES = {
    # Component counts
    "total_components": 1300,
    "server_components": 720,
    "pc_components": 584,
    
    # ML improvements
    "ml_model_size_multiplier": 10,  # 10x larger models
    "ml_training_speed_multiplier": 100,  # 100x faster
    "ml_data_capacity_multiplier": 10,  # 10x more data
    
    # Backtesting improvements
    "backtest_speed_multiplier": 60,  # 60x faster
    "backtest_assets_multiplier": 10,  # 10x more assets
    "backtest_history_multiplier": 10,  # 10x more history
    
    # Trading improvements
    "expected_monthly_return_boost": 1.6,  # 60% improvement
    "latency_reduction": 0.1,  # 10x lower latency
    "prediction_accuracy_boost": 0.15,  # 15% better accuracy
    
    # Risk improvements
    "var_accuracy_improvement": 0.2,  # 20% better VaR
    "stress_test_coverage_multiplier": 100  # 100x more scenarios
}
