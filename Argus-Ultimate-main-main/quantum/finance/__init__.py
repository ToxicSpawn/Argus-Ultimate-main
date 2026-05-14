"""
Quantum finance algorithms.

- ``multi_period``: multi-period portfolio optimization via QAOA-DP
- ``options``: option pricing via QAE (European, Asian, barrier, lookback)
- ``execution``: Almgren-Chriss optimal execution with QAOA
- ``qpca``: Quantum Principal Component Analysis (factor model)
- ``cointegration``: quantum cointegration testing for pairs trading
"""

from .multi_period import multi_period_portfolio_qaoa
from .options import (
    european_option_price,
    asian_option_price,
    barrier_option_price,
    lookback_option_price,
)
from .execution import almgren_chriss_qaoa
from .qpca import quantum_pca
from .cointegration import quantum_cointegration_test

__all__ = [
    "multi_period_portfolio_qaoa",
    "european_option_price",
    "asian_option_price",
    "barrier_option_price",
    "lookback_option_price",
    "almgren_chriss_qaoa",
    "quantum_pca",
    "quantum_cointegration_test",
]
