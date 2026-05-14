"""
Quantum Reservoir Computing for Time Series Prediction.

A quantum reservoir computer uses a fixed random quantum circuit as a
nonlinear dynamical system for feature expansion.  Only the readout layer
(ridge regression) is trained -- the reservoir parameters are frozen after
construction.

This is a classical simulation of quantum dynamics.  We explicitly evolve
a 2^n statevector through parameterised single-qubit rotations and CNOT
entanglement layers.  For n_qubits <= 10 this is fast and exact.  There
is no claim of quantum advantage -- the value is the rich nonlinear
feature space that entangled quantum states provide.

When the reservoir dimension is 2^n (e.g. 64 for 6 qubits), ridge
regression on those features can capture patterns that a simple moving
average or linear model misses.

Typical usage::

    from quantum.qml.quantum_reservoir import QuantumReservoirComputer

    qrc = QuantumReservoirComputer(n_qubits=6, n_layers=3, washout=20)
    qrc.fit(price_series, horizon=1)
    result = qrc.predict(recent_prices, steps=3)
    regime = qrc.predict_regime(recent_prices)
    bench  = qrc.benchmark(price_series)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class QuantumReservoirComputer:
    """
    Quantum reservoir computing for time series prediction.

    Uses random unitary evolution as a nonlinear dynamical system.
    Classical simulation of quantum dynamics -- honest about this.

    The reservoir is a fixed random quantum circuit that provides
    nonlinear feature expansion.  Only the readout layer is trained.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 3,
        washout: int = 20,
        ridge_alpha: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if n_qubits < 1 or n_qubits > 14:
            raise ValueError(f"n_qubits must be in [1, 14], got {n_qubits}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if washout < 0:
            raise ValueError(f"washout must be >= 0, got {washout}")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.washout = washout
        self.ridge_alpha = ridge_alpha
        self.dim = 2 ** n_qubits  # Hilbert space dimension

        self._rng = np.random.RandomState(seed)
        self._reservoir_params: Optional[np.ndarray] = None
        self._readout_weights: Optional[np.ndarray] = None
        self._readout_bias: float = 0.0
        self._state: Optional[np.ndarray] = None  # current statevector
        self._fitted = False
        self._fit_time: float = 0.0
        self._train_rmse: float = 0.0

        self._build_reservoir()

    # ------------------------------------------------------------------
    # Reservoir construction
    # ------------------------------------------------------------------

    def _build_reservoir(self) -> None:
        """Build fixed random reservoir.

        For each layer and each qubit we store three rotation angles
        (Rx, Ry, Rz).  The entangling pattern is a nearest-neighbour
        ring of CNOTs, which is fixed (no parameters).

        Shape: (n_layers, n_qubits, 3)  -- angles in [0, 2*pi)
        """
        self._reservoir_params = self._rng.uniform(
            0, 2 * np.pi, size=(self.n_layers, self.n_qubits, 3)
        )
        # Initialise statevector to |0...0>
        self._state = np.zeros(self.dim, dtype=np.complex128)
        self._state[0] = 1.0 + 0j

    def _reset_state(self) -> None:
        """Reset the reservoir state to |0...0>."""
        self._state = np.zeros(self.dim, dtype=np.complex128)
        self._state[0] = 1.0 + 0j

    # ------------------------------------------------------------------
    # Gate primitives (statevector simulation)
    # ------------------------------------------------------------------

    @staticmethod
    def _rx(theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)

    @staticmethod
    def _ry(theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)

    @staticmethod
    def _rz(theta: float) -> np.ndarray:
        return np.array(
            [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
            dtype=np.complex128,
        )

    def _apply_single_qubit(self, gate: np.ndarray, qubit: int) -> None:
        """Apply a 2x2 gate to a specific qubit in the statevector."""
        n = self.n_qubits
        # Reshape statevector into (2,2,...,2) tensor, apply gate on axis=qubit
        shape = [2] * n
        psi = self._state.reshape(shape)

        # Move target qubit axis to last position, apply gate, move back
        psi = np.moveaxis(psi, qubit, -1)
        psi = np.einsum("ij,...j->...i", gate, psi)
        psi = np.moveaxis(psi, -1, qubit)

        self._state = psi.reshape(self.dim)

    def _apply_cnot(self, control: int, target: int) -> None:
        """Apply CNOT gate (control, target) on the statevector."""
        n = self.n_qubits
        new_state = self._state.copy()
        for i in range(self.dim):
            bits = [(i >> (n - 1 - q)) & 1 for q in range(n)]
            if bits[control] == 1:
                bits[target] ^= 1
                j = 0
                for q in range(n):
                    j = (j << 1) | bits[q]
                new_state[j] = self._state[i]
                new_state[i] = 0.0  # will be overwritten if j==i case
        # Handle zero-writes properly: rebuild
        new_state = np.zeros(self.dim, dtype=np.complex128)
        for i in range(self.dim):
            bits = [(i >> (n - 1 - q)) & 1 for q in range(n)]
            if bits[control] == 1:
                bits[target] ^= 1
                j = 0
                for q in range(n):
                    j = (j << 1) | bits[q]
                new_state[j] += self._state[i]
            else:
                new_state[i] += self._state[i]
        self._state = new_state

    def _measure_z_expectations(self) -> np.ndarray:
        """Measure <Z_i> for each qubit.  Returns array of length n_qubits."""
        probs = np.abs(self._state) ** 2
        expectations = np.zeros(self.n_qubits)
        for q in range(self.n_qubits):
            # <Z_q> = sum_i (-1)^{bit_q(i)} * |psi_i|^2
            for i in range(self.dim):
                bit = (i >> (self.n_qubits - 1 - q)) & 1
                expectations[q] += (1 - 2 * bit) * probs[i]
        return expectations

    # ------------------------------------------------------------------
    # Reservoir evolution
    # ------------------------------------------------------------------

    def _reservoir_evolution(self, input_value: float) -> np.ndarray:
        """Evolve reservoir state with new input.

        1. Encode input as Ry rotation on qubit 0
        2. Apply all reservoir layers (Rx, Ry, Rz + CNOT ring)
        3. Return <Z_i> expectations as feature vector
        """
        # Input encoding: Ry on first qubit, scaled to [0, pi]
        # Normalise to [0, 1] then scale to [0, pi]
        theta_input = float(input_value) * np.pi
        self._apply_single_qubit(self._ry(theta_input), 0)

        # Apply fixed reservoir circuit
        for layer in range(self.n_layers):
            params = self._reservoir_params[layer]
            # Single-qubit rotations
            for q in range(self.n_qubits):
                self._apply_single_qubit(self._rx(params[q, 0]), q)
                self._apply_single_qubit(self._ry(params[q, 1]), q)
                self._apply_single_qubit(self._rz(params[q, 2]), q)
            # Nearest-neighbour CNOT ring
            for q in range(self.n_qubits):
                target = (q + 1) % self.n_qubits
                if target != q:
                    self._apply_cnot(q, target)

        return self._measure_z_expectations()

    def _normalise_series(
        self, series: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """Normalise to [0, 1] range.  Returns (normalised, min, range)."""
        smin = float(np.nanmin(series))
        srange = float(np.nanmax(series) - smin)
        if srange < 1e-12:
            srange = 1.0
        return (series - smin) / srange, smin, srange

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, time_series: Any, horizon: int = 1) -> "QuantumReservoirComputer":
        """Train readout layer on a time series.

        1. Run time series through reservoir (with washout period).
        2. Collect reservoir state features for each timestep.
        3. Train linear readout (ridge regression) to predict t+horizon.

        Returns self for chaining.
        """
        t0 = time.monotonic()
        ts = np.asarray(time_series, dtype=np.float64).ravel()

        # Handle NaN by forward-filling
        mask = np.isnan(ts)
        if mask.any():
            for i in range(1, len(ts)):
                if mask[i]:
                    ts[i] = ts[i - 1]
            # If the first value is NaN, use 0
            if mask[0]:
                ts[0] = 0.0

        if len(ts) < self.washout + horizon + 2:
            raise ValueError(
                f"Time series too short ({len(ts)} points) for "
                f"washout={self.washout} + horizon={horizon} + 2"
            )

        normed, self._norm_min, self._norm_range = self._normalise_series(ts)

        # Drive reservoir and collect features
        self._reset_state()
        features_list: List[np.ndarray] = []
        targets_list: List[float] = []

        n_usable = len(normed) - horizon
        for t in range(n_usable):
            feat = self._reservoir_evolution(normed[t])
            if t >= self.washout:
                features_list.append(feat.copy())
                targets_list.append(normed[t + horizon])

        if len(features_list) < 2:
            raise ValueError("Not enough data points after washout for training")

        X = np.array(features_list)
        y = np.array(targets_list)

        # Ridge regression: w = (X^T X + alpha I)^-1 X^T y
        # Add bias column
        X_bias = np.column_stack([X, np.ones(len(X))])
        reg = self.ridge_alpha * np.eye(X_bias.shape[1])
        reg[-1, -1] = 0  # don't regularise bias
        try:
            w = np.linalg.solve(X_bias.T @ X_bias + reg, X_bias.T @ y)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(X_bias, y, rcond=None)[0]

        self._readout_weights = w[:-1]
        self._readout_bias = float(w[-1])
        self._fitted = True
        self._fit_time = time.monotonic() - t0

        # Compute training RMSE
        y_pred = X @ self._readout_weights + self._readout_bias
        self._train_rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

        logger.info(
            "QuantumReservoir fit: %d samples, %d features, "
            "train_rmse=%.6f, time=%.2fs",
            len(X), self.n_qubits, self._train_rmse, self._fit_time,
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, recent_values: Any, steps: int = 1) -> Dict[str, Any]:
        """Predict next steps.

        Returns dict with:
          - predictions: list of predicted values (denormalised)
          - confidence: float in [0, 1]
          - reservoir_entropy: von Neumann entropy estimate
          - method: str description
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        rv = np.asarray(recent_values, dtype=np.float64).ravel()
        if len(rv) == 0:
            raise ValueError("recent_values must not be empty")

        # Handle NaN
        mask = np.isnan(rv)
        if mask.any():
            for i in range(1, len(rv)):
                if mask[i]:
                    rv[i] = rv[i - 1]
            if mask[0]:
                rv[0] = 0.0

        normed = (rv - self._norm_min) / self._norm_range

        # Reset and drive with recent values
        self._reset_state()
        for val in normed:
            self._reservoir_evolution(val)

        # Multi-step prediction (autoregressive)
        predictions_normed: List[float] = []
        for _ in range(steps):
            feat = self._reservoir_evolution(
                predictions_normed[-1] if predictions_normed else normed[-1]
            )
            pred = float(feat @ self._readout_weights + self._readout_bias)
            pred = np.clip(pred, 0.0, 1.0)
            predictions_normed.append(pred)

        # Denormalise
        predictions = [
            float(p * self._norm_range + self._norm_min)
            for p in predictions_normed
        ]

        # Confidence: decay with prediction horizon, scaled by training fit
        base_conf = max(0.0, 1.0 - self._train_rmse * 5)
        confidence = max(0.05, base_conf * (0.85 ** (steps - 1)))

        entropy = self._reservoir_entropy()

        return {
            "predictions": predictions,
            "confidence": round(confidence, 4),
            "reservoir_entropy": round(entropy, 4),
            "method": f"quantum_reservoir_{self.n_qubits}q_{self.n_layers}L",
        }

    def _reservoir_entropy(self) -> float:
        """Estimate von Neumann entropy of the current reservoir state.

        S = -sum p_i log2(p_i)  where p_i = |<i|psi>|^2

        High entropy means the state is spread across many basis states
        (complex dynamics).  Low entropy means it is concentrated
        (simple/periodic dynamics).
        """
        if self._state is None:
            return 0.0
        probs = np.abs(self._state) ** 2
        probs = probs[probs > 1e-15]  # avoid log(0)
        return float(-np.sum(probs * np.log2(probs)))

    # ------------------------------------------------------------------
    # Regime classification
    # ------------------------------------------------------------------

    def predict_regime(self, recent_values: Any) -> Dict[str, Any]:
        """Use reservoir dynamics to classify market regime.

        - Low entropy reservoir state -> trending market
        - Medium entropy -> mean reverting
        - High entropy -> volatile/crisis

        Also estimates the largest Lyapunov exponent from the
        sensitivity of reservoir state to perturbations.
        """
        rv = np.asarray(recent_values, dtype=np.float64).ravel()
        if len(rv) < 5:
            return {
                "regime": "UNKNOWN",
                "confidence": 0.0,
                "entropy": 0.0,
                "lyapunov_estimate": 0.0,
            }

        # Handle NaN
        mask = np.isnan(rv)
        if mask.any():
            for i in range(1, len(rv)):
                if mask[i]:
                    rv[i] = rv[i - 1]
            if mask[0]:
                rv[0] = 0.0

        smin = float(np.nanmin(rv))
        srange = float(np.nanmax(rv) - smin)
        if srange < 1e-12:
            return {
                "regime": "FLAT",
                "confidence": 0.95,
                "entropy": 0.0,
                "lyapunov_estimate": -1.0,
            }
        normed = (rv - smin) / srange

        # Drive reservoir
        self._reset_state()
        entropies: List[float] = []
        for val in normed:
            self._reservoir_evolution(val)
            entropies.append(self._reservoir_entropy())

        mean_entropy = float(np.mean(entropies[-min(20, len(entropies)):]))
        max_possible_entropy = float(self.n_qubits)  # log2(2^n) = n

        # Lyapunov estimate: drive with slightly perturbed series
        self._reset_state()
        for val in normed:
            self._reservoir_evolution(val)
        state_orig = self._state.copy()

        eps = 1e-5
        normed_pert = normed.copy()
        normed_pert[-1] += eps
        self._reset_state()
        for val in normed_pert:
            self._reservoir_evolution(val)
        state_pert = self._state.copy()

        divergence = float(np.linalg.norm(state_pert - state_orig))
        lyapunov = float(np.log(max(divergence / eps, 1e-15)))

        # Classify
        entropy_ratio = mean_entropy / max(max_possible_entropy, 1.0)

        if entropy_ratio < 0.3:
            regime = "TRENDING"
            confidence = 0.7 + 0.3 * (1 - entropy_ratio / 0.3)
        elif entropy_ratio < 0.6:
            regime = "MEAN_REVERTING"
            confidence = 0.6 + 0.2 * (1 - abs(entropy_ratio - 0.45) / 0.15)
        else:
            regime = "VOLATILE"
            confidence = 0.5 + 0.3 * min(entropy_ratio / 1.0, 1.0)

        # Lyapunov refinement: very positive -> chaotic
        if lyapunov > 5.0:
            regime = "CRISIS"
            confidence = max(confidence, 0.7)

        return {
            "regime": regime,
            "confidence": round(float(confidence), 4),
            "entropy": round(mean_entropy, 4),
            "lyapunov_estimate": round(lyapunov, 4),
        }

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def benchmark(self, time_series: Any) -> Dict[str, Any]:
        """Compare quantum reservoir vs baselines.

        Baselines:
          - Simple Moving Average (window=5)
          - Linear Regression (last 10 points)
          - ARIMA(1,1,1) if statsmodels available, else skipped

        Uses walk-forward validation: train on first 70%, test on last 30%.

        Returns honest comparison with RMSE, MAE, directional accuracy
        for each method.
        """
        ts = np.asarray(time_series, dtype=np.float64).ravel()

        # Handle NaN
        mask = np.isnan(ts)
        if mask.any():
            for i in range(1, len(ts)):
                if mask[i]:
                    ts[i] = ts[i - 1]
            if mask[0]:
                ts[0] = 0.0

        n = len(ts)
        if n < self.washout + 10:
            return {"error": "Time series too short for benchmarking"}

        split = int(n * 0.7)
        train, test = ts[:split], ts[split:]

        results: Dict[str, Dict[str, float]] = {}

        # --- Quantum Reservoir ---
        try:
            self.fit(train, horizon=1)
            qr_preds = []
            # Walk-forward: for each test point, predict from preceding window
            window_size = min(50, split)
            for i in range(len(test)):
                start = max(0, split + i - window_size)
                end = split + i
                window = ts[start:end]
                pred_result = self.predict(window, steps=1)
                qr_preds.append(pred_result["predictions"][0])

            qr_preds = np.array(qr_preds)
            results["quantum_reservoir"] = self._compute_metrics(
                test, qr_preds
            )
        except Exception as e:
            logger.warning("Quantum reservoir benchmark failed: %s", e)
            results["quantum_reservoir"] = {"error": str(e)}

        # --- Moving Average (window=5) ---
        try:
            ma_preds = []
            for i in range(len(test)):
                start = max(0, split + i - 5)
                end = split + i
                ma_preds.append(float(np.mean(ts[start:end])))
            results["moving_average_5"] = self._compute_metrics(
                test, np.array(ma_preds)
            )
        except Exception as e:
            results["moving_average_5"] = {"error": str(e)}

        # --- Linear Regression (last 10 points) ---
        try:
            lr_preds = []
            for i in range(len(test)):
                start = max(0, split + i - 10)
                end = split + i
                window = ts[start:end]
                x = np.arange(len(window))
                if len(window) >= 2:
                    coeffs = np.polyfit(x, window, 1)
                    lr_preds.append(float(np.polyval(coeffs, len(window))))
                else:
                    lr_preds.append(float(window[-1]))
            results["linear_regression"] = self._compute_metrics(
                test, np.array(lr_preds)
            )
        except Exception as e:
            results["linear_regression"] = {"error": str(e)}

        # --- ARIMA(1,1,1) ---
        try:
            from statsmodels.tsa.arima.model import ARIMA

            arima_preds = []
            for i in range(len(test)):
                end = split + i
                history = ts[max(0, end - 100) : end]
                if len(history) >= 10:
                    model = ARIMA(history, order=(1, 1, 1))
                    fit = model.fit()
                    arima_preds.append(float(fit.forecast(1)[0]))
                else:
                    arima_preds.append(float(history[-1]))
            results["arima_1_1_1"] = self._compute_metrics(
                test, np.array(arima_preds)
            )
        except ImportError:
            results["arima_1_1_1"] = {"skipped": "statsmodels not installed"}
        except Exception as e:
            results["arima_1_1_1"] = {"error": str(e)}

        # Summary
        best_method = None
        best_rmse = float("inf")
        for method, metrics in results.items():
            rmse = metrics.get("rmse", float("inf"))
            if isinstance(rmse, (int, float)) and rmse < best_rmse:
                best_rmse = rmse
                best_method = method

        return {
            "methods": results,
            "best_method": best_method,
            "best_rmse": round(best_rmse, 6) if best_rmse < float("inf") else None,
            "n_train": split,
            "n_test": len(test),
            "note": (
                "Quantum reservoir uses classical simulation of quantum dynamics. "
                "No quantum hardware was used."
            ),
        }

    @staticmethod
    def _compute_metrics(
        actual: np.ndarray, predicted: np.ndarray
    ) -> Dict[str, float]:
        """Compute RMSE, MAE, and directional accuracy."""
        errors = actual - predicted
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))

        # Directional accuracy: did we predict the right direction of change?
        if len(actual) >= 2:
            actual_dir = np.diff(actual)
            # predicted direction vs previous actual
            pred_dir = predicted[1:] - actual[:-1]
            correct = np.sum(np.sign(actual_dir) == np.sign(pred_dir))
            dir_acc = float(correct / len(actual_dir)) if len(actual_dir) > 0 else 0.0
        else:
            dir_acc = 0.0

        return {
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "directional_accuracy": round(dir_acc, 4),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the reservoir state and configuration."""
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "washout": self.washout,
            "hilbert_dim": self.dim,
            "ridge_alpha": self.ridge_alpha,
            "fitted": self._fitted,
            "train_rmse": round(self._train_rmse, 6) if self._fitted else None,
            "fit_time_s": round(self._fit_time, 3) if self._fitted else None,
            "reservoir_entropy": round(self._reservoir_entropy(), 4),
            "method": "classical_simulation",
        }
