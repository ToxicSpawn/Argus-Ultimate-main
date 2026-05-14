# Quantum Bot - Quick Start Guide 🚀

## 🎯 One Import, All Features!

All quantum computer files are now merged into one unified module!

---

## 💻 Quick Start

### Single Import
```python
from quantum.quantum_unified_all import get_quantum_bot
```

### Use Any Performance Level
```python
# Standard (50-100ms)
quantum = await get_quantum_bot('standard')

# Ultra-Fast (<1ms cached)
quantum = await get_quantum_bot('ultra_fast')

# Peak (<0.5ms cached, JIT/GPU)
quantum = await get_quantum_bot('peak')

# Extreme (<0.01ms cached, 50k jobs/sec)
quantum = await get_quantum_bot('extreme')

# Realistic (real hardware behavior)
quantum = await get_quantum_bot('realistic')
```

---

## 📝 Example Usage

```python
import numpy as np
from quantum.quantum_unified_all import get_quantum_bot

# Get quantum bot
quantum = await get_quantum_bot(performance_level='extreme')

# Optimize portfolio
result = await quantum.optimize_portfolio(
    assets=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
    expected_returns=np.array([0.1, 0.15, 0.12]),
    covariance_matrix=cov_matrix,
    risk_tolerance=0.5
)

print(f"Weights: {result['weights']}")
print(f"Expected Return: {result['expected_return']:.4f}")
print(f"Cost: ${result['cost']:.2f}")  # Always $0.00
```

---

## 🎯 All Methods

```python
# Portfolio optimization
result = await quantum.optimize_portfolio(...)

# Risk analysis
risk = await quantum.analyze_risk(...)

# Game theory
game = await quantum.game_theory_analysis(...)

# Statistics
stats = quantum.get_stats()

# Hardware info (realistic mode)
hardware = quantum.get_hardware_info()
```

---

## ✅ That's It!

One import, all features, 100% FREE! 🎉
