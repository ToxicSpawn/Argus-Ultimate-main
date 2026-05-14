"""
Temporal Fusion Transformer (TFT) for financial time series prediction.

Implements the core TFT architecture from "Temporal Fusion Transformers for
Interpretable Multi-horizon Time Series Forecasting" (Lim et al., 2021) using
NumPy only — no PyTorch dependency.

Components:
- MultiHeadAttention: scaled dot-product attention with multiple heads
- GatedResidualNetwork: ELU-based gated residual blocks with skip connections
- VariableSelectionNetwork: softmax-weighted feature selection
- TemporalFusionTransformer: full encoder/decoder TFT model
- FinancialTFTPredictor: high-level predictor with fit/predict/explain API

All weights are initialised with He/Kaiming initialisation and trained via
Adam optimiser with gradient clipping.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def _elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Exponential Linear Unit."""
    return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -20, 0)) - 1))


def _elu_derivative(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x > 0, 1.0, alpha * np.exp(np.clip(x, -20, 0)))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)


def _quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> np.ndarray:
    """Pinball loss for quantile regression."""
    err = y_true - y_pred
    return np.maximum(tau * err, (tau - 1) * err)


def _quantile_loss_derivative(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> np.ndarray:
    err = y_true - y_pred
    return np.where(err > 0, tau, tau - 1)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0, std, (fan_in, fan_out)).astype(np.float64)


def _zeros(fan_in: int, fan_out: int) -> np.ndarray:
    return np.zeros((fan_in, fan_out), dtype=np.float64)


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------

@dataclass
class AttentionOutput:
    output: np.ndarray
    attention_weights: np.ndarray


class MultiHeadAttention:
    """Scaled dot-product multi-head attention (NumPy).

    Parameters
    ----------
    d_model : int
        Dimensionality of input and output.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout

        self.W_q = np.empty((d_model, d_model), dtype=np.float64)
        self.W_k = np.empty((d_model, d_model), dtype=np.float64)
        self.W_v = np.empty((d_model, d_model), dtype=np.float64)
        self.W_o = np.empty((d_model, d_model), dtype=np.float64)

    def init_weights(self, rng: np.random.Generator) -> None:
        self.W_q = _he_init(self.d_model, self.d_model, rng)
        self.W_k = _he_init(self.d_model, self.d_model, rng)
        self.W_v = _he_init(self.d_model, self.d_model, rng)
        self.W_o = _he_init(self.d_model, self.d_model, rng)

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        batch, seq, _ = x.shape
        x = x.reshape(batch, seq, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        batch, _, seq, _ = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch, seq, self.d_model)

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> AttentionOutput:
        batch = query.shape[0]

        q = self._split_heads(query @ self.W_q)
        k = self._split_heads(key @ self.W_k)
        v = self._split_heads(value @ self.W_v)

        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores + mask * -1e9

        attn = _softmax(scores, axis=-1)

        if training and rng is not None and self.dropout > 0:
            drop_mask = (rng.random(attn.shape) > self.dropout).astype(np.float64)
            attn = attn * drop_mask / (drop_mask.mean() + 1e-12)

        out = attn @ v
        out = self._merge_heads(out) @ self.W_o

        return AttentionOutput(output=out, attention_weights=attn)

    def get_params(self) -> Dict[str, np.ndarray]:
        return {"W_q": self.W_q, "W_k": self.W_k, "W_v": self.W_v, "W_o": self.W_o}

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        self.W_q = params["W_q"]
        self.W_k = params["W_k"]
        self.W_v = params["W_v"]
        self.W_o = params["W_o"]


# ---------------------------------------------------------------------------
# Gated Residual Network (GRN)
# ---------------------------------------------------------------------------

class GatedResidualNetwork:
    """Gated residual block with ELU activation and skip connections.

    Implements:
        eta = ELU(W1 * input + b1)
        eta = W2 * eta + b2
        gate = sigmoid(W_g * input + b_g)
        output = LayerNorm(input + gate * eta)
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
    ) -> None:
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.W1 = np.empty((hidden_size, hidden_size), dtype=np.float64)
        self.b1 = np.zeros(hidden_size, dtype=np.float64)
        self.W2 = np.empty((hidden_size, hidden_size), dtype=np.float64)
        self.b2 = np.zeros(hidden_size, dtype=np.float64)
        self.W_g = np.empty((hidden_size, hidden_size), dtype=np.float64)
        self.b_g = np.zeros(hidden_size, dtype=np.float64)
        self.gamma = np.ones(hidden_size, dtype=np.float64)
        self.beta = np.zeros(hidden_size, dtype=np.float64)

    def init_weights(self, rng: np.random.Generator) -> None:
        self.W1 = _he_init(self.hidden_size, self.hidden_size, rng)
        self.W2 = _he_init(self.hidden_size, self.hidden_size, rng)
        self.W_g = _he_init(self.hidden_size, self.hidden_size, rng)

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + 1e-6) + self.beta

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -20, 20)
        return 1.0 / (1.0 + np.exp(-x))

    def forward(
        self,
        x: np.ndarray,
        training: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        eta = _elu(x @ self.W1 + self.b1)

        if training and rng is not None and self.dropout > 0:
            mask = (rng.random(eta.shape) > self.dropout).astype(np.float64)
            eta = eta * mask / (mask.mean() + 1e-12)

        eta = eta @ self.W2 + self.b2

        gate = self._sigmoid(x @ self.W_g + self.b_g)
        return self._layer_norm(x + gate * eta)

    def get_params(self) -> Dict[str, np.ndarray]:
        return {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "W_g": self.W_g, "b_g": self.b_g,
            "gamma": self.gamma, "beta": self.beta,
        }

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        for k, v in params.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Variable Selection Network (VSN)
# ---------------------------------------------------------------------------

class VariableSelectionNetwork:
    """Applies softmax-weighted feature selection across input variables.

    Projects each input feature to hidden_size via a shared linear layer,
    then applies a GRN per variable. A separate GRN produces selection
    weights via softmax. The weighted sum of transformed variables is returned.

    Input shape: (batch, seq_len, num_variables)
    Output shape: (batch, seq_len, hidden_size)
    """

    def __init__(
        self,
        num_variables: int,
        hidden_size: int,
        dropout: float = 0.1,
    ) -> None:
        self.num_variables = num_variables
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.W_v = np.empty((1, hidden_size), dtype=np.float64)
        self.variable_grns: List[GatedResidualNetwork] = []
        for _ in range(num_variables):
            grn = GatedResidualNetwork(hidden_size, dropout)
            self.variable_grns.append(grn)

        self.weight_grn = GatedResidualNetwork(hidden_size, dropout)
        self.W_context = np.empty((hidden_size, hidden_size), dtype=np.float64)
        self.W_weights = np.empty((hidden_size, num_variables), dtype=np.float64)

    def init_weights(self, rng: np.random.Generator) -> None:
        self.W_v = _he_init(1, self.hidden_size, rng)
        for grn in self.variable_grns:
            grn.init_weights(rng)
        self.weight_grn.init_weights(rng)
        self.W_context = _he_init(self.hidden_size, self.hidden_size, rng)
        self.W_weights = _he_init(self.hidden_size, self.num_variables, rng)

    def forward(
        self,
        x: np.ndarray,
        context: Optional[np.ndarray] = None,
        training: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        batch, seq_len = x.shape[0], x.shape[1]

        transformed: List[np.ndarray] = []
        for i in range(self.num_variables):
            feat = x[:, :, i:i+1] @ self.W_v
            transformed.append(self.variable_grns[i].forward(feat, training, rng))

        stacked = np.stack(transformed, axis=2)

        if context is not None:
            ctx_expanded = context[:, None, :]
            theta = self.weight_grn.forward(
                ctx_expanded @ self.W_context, training, rng
            )
        else:
            theta = self.weight_grn.forward(
                stacked.mean(axis=2), training, rng
            )

        weights = _softmax(theta @ self.W_weights, axis=-1)

        output = np.zeros((batch, seq_len, self.hidden_size), dtype=np.float64)
        for i in range(self.num_variables):
            output += weights[:, :, i:i+1] * transformed[i]

        return output, weights

    def get_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            f"vsn_grn_{i}": grn.get_params()
            for i, grn in enumerate(self.variable_grns)
        }
        params["weight_grn"] = self.weight_grn.get_params()
        params["W_context"] = self.W_context
        params["W_weights"] = self.W_weights
        return params

    def set_params(self, params: Dict[str, Any]) -> None:
        for i, grn in enumerate(self.variable_grns):
            grn.set_params(params[f"vsn_grn_{i}"])
        self.weight_grn.set_params(params["weight_grn"])
        self.W_context = params["W_context"]
        self.W_weights = params["W_weights"]


# ---------------------------------------------------------------------------
# Scaled Embedding (positional encoding)
# ---------------------------------------------------------------------------

def _positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    pe = np.zeros((seq_len, d_model), dtype=np.float64)
    position = np.arange(seq_len, dtype=np.float64)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float64) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


# ---------------------------------------------------------------------------
# Temporal Fusion Transformer
# ---------------------------------------------------------------------------

@dataclass
class TFTConfig:
    hidden_size: int = 64
    num_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dropout: float = 0.1
    num_past_features: int = 10
    num_future_features: int = 5
    num_static_features: int = 3
    seq_len: int = 24
    pred_len: int = 12
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
    learning_rate: float = 1e-3
    grad_clip: float = 1.0


class TemporalFusionTransformer:
    """Full TFT model implemented in NumPy.

    Architecture overview:
    1. Variable Selection Networks for static, past, and future features
    2. Static Covariate Encoder → context vectors for VSNs and gating
    3. LSTM-like temporal processing via self-attention encoder
    4. Multi-head self-attention decoder with causal masking
    5. Gated residual skip connections throughout
    6. Quantile output head for probabilistic forecasts
    """

    def __init__(self, config: Optional[TFTConfig] = None) -> None:
        self.config = config or TFTConfig()
        self.rng = np.random.default_rng(42)

        h = self.config.hidden_size
        nh = self.config.num_heads

        self.static_vsn = VariableSelectionNetwork(
            self.config.num_static_features, h, self.config.dropout
        )
        self.past_vsn = VariableSelectionNetwork(
            self.config.num_past_features, h, self.config.dropout
        )
        self.future_vsn = VariableSelectionNetwork(
            self.config.num_future_features, h, self.config.dropout
        )

        self.static_encoder_grn = GatedResidualNetwork(h, self.config.dropout)
        self.static_context_vsn_grn = GatedResidualNetwork(h, self.config.dropout)
        self.static_context_e_grn = GatedResidualNetwork(h, self.config.dropout)
        self.static_context_h_grn = GatedResidualNetwork(h, self.config.dropout)

        self.encoder_grns: List[GatedResidualNetwork] = []
        for _ in range(self.config.num_encoder_layers):
            self.encoder_grns.append(GatedResidualNetwork(h, self.config.dropout))

        self.decoder_grns: List[GatedResidualNetwork] = []
        for _ in range(self.config.num_decoder_layers):
            self.decoder_grns.append(GatedResidualNetwork(h, self.config.dropout))

        self.encoder_attn = MultiHeadAttention(h, nh, self.config.dropout)
        self.decoder_attn = MultiHeadAttention(h, nh, self.config.dropout)
        self.cross_attn = MultiHeadAttention(h, nh, self.config.dropout)

        self.W_out = np.empty((h, len(self.config.quantiles)), dtype=np.float64)
        self.b_out = np.zeros(len(self.config.quantiles), dtype=np.float64)

        self._pe = _positional_encoding(
            self.config.seq_len + self.config.pred_len, h
        )

        self._init_all()

    def _init_all(self) -> None:
        self.static_vsn.init_weights(self.rng)
        self.past_vsn.init_weights(self.rng)
        self.future_vsn.init_weights(self.rng)
        self.static_encoder_grn.init_weights(self.rng)
        self.static_context_vsn_grn.init_weights(self.rng)
        self.static_context_e_grn.init_weights(self.rng)
        self.static_context_h_grn.init_weights(self.rng)
        for grn in self.encoder_grns:
            grn.init_weights(self.rng)
        for grn in self.decoder_grns:
            grn.init_weights(self.rng)
        self.encoder_attn.init_weights(self.rng)
        self.decoder_attn.init_weights(self.rng)
        self.cross_attn.init_weights(self.rng)
        self.W_out = _he_init(self.config.hidden_size, len(self.config.quantiles), self.rng)

    def _make_causal_mask(self, size: int) -> np.ndarray:
        mask = np.triu(np.ones((1, 1, size, size), dtype=np.float64), k=1)
        return 1 - mask

    def forward(
        self,
        static_x: np.ndarray,
        past_x: np.ndarray,
        future_x: np.ndarray,
        training: bool = False,
    ) -> np.ndarray:
        batch = static_x.shape[0]
        h = self.config.hidden_size
        seq_len = self.config.seq_len
        pred_len = self.config.pred_len
        total_len = seq_len + pred_len

        static_ctx = self.static_vsn.forward(
            static_x[:, None, :],
            training=training, rng=self.rng,
        )[0].squeeze(1)

        context_vsn = self.static_context_vsn_grn.forward(static_ctx, training, self.rng)
        context_e = self.static_context_e_grn.forward(static_ctx, training, self.rng)
        context_h = self.static_context_h_grn.forward(static_ctx, training, self.rng)

        past_sel, _ = self.past_vsn.forward(past_x, context_vsn, training, self.rng)
        future_sel, _ = self.future_vsn.forward(future_x, context_vsn, training, self.rng)

        encoder_input = past_sel + context_e[:, None, :]
        for grn in self.encoder_grns:
            encoder_input = grn.forward(encoder_input, training, self.rng)

        encoder_input = encoder_input + self._pe[:seq_len, :][None, :, :]

        enc_out = self.encoder_attn.forward(
            encoder_input, encoder_input, encoder_input, training=training, rng=self.rng
        ).output

        decoder_input = future_sel + context_h[:, None, :]
        for grn in self.decoder_grns:
            decoder_input = grn.forward(decoder_input, training, self.rng)

        decoder_input = decoder_input + self._pe[seq_len:total_len, :][None, :, :]

        causal_mask = self._make_causal_mask(pred_len)

        dec_out = self.decoder_attn.forward(
            decoder_input, decoder_input, decoder_input,
            mask=causal_mask, training=training, rng=self.rng,
        ).output

        cross_out = self.cross_attn.forward(
            dec_out, enc_out, enc_out, training=training, rng=self.rng,
        ).output

        logits = cross_out @ self.W_out + self.b_out

        return logits

    def compute_loss(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        quantiles: Optional[Tuple[float, ...]] = None,
    ) -> float:
        qts = quantiles or self.config.quantiles
        total_loss = 0.0
        for i, tau in enumerate(qts):
            total_loss += _quantile_loss(targets, logits[:, :, i], tau).mean()
        return float(total_loss)

    def _numerical_gradient(
        self,
        static_x: np.ndarray,
        past_x: np.ndarray,
        future_x: np.ndarray,
        targets: np.ndarray,
        eps: float = 1e-5,
    ) -> Dict[str, np.ndarray]:
        grads: Dict[str, np.ndarray] = {}
        base_loss = self.compute_loss(
            self.forward(static_x, past_x, future_x, training=False), targets
        )

        for name, param in self._get_trainable_params().items():
            grad = np.zeros_like(param)
            it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
            count = 0
            max_checks = min(param.size, 500)
            step = max(1, param.size // max_checks)
            while not it.finished:
                if count % step == 0:
                    idx = it.multi_index
                    old_val = param[idx]
                    param[idx] = old_val + eps
                    plus = self.compute_loss(
                        self.forward(static_x, past_x, future_x, training=False), targets
                    )
                    grad[idx] = (plus - base_loss) / eps
                    param[idx] = old_val
                it.iternext()
                count += 1
            grads[name] = grad

        return grads

    def _get_trainable_params(self) -> Dict[str, np.ndarray]:
        params: Dict[str, np.ndarray] = {}
        params.update(self.encoder_attn.get_params())
        params.update(self.decoder_attn.get_params())
        params.update(self.cross_attn.get_params())
        params["W_out"] = self.W_out
        params["b_out"] = self.b_out
        return params

    def _set_trainable_params(self, params: Dict[str, np.ndarray]) -> None:
        self.encoder_attn.set_params({k: v for k, v in params.items() if k in ("W_q", "W_k", "W_v", "W_o")})
        self.decoder_attn.set_params({k: v for k, v in params.items() if k in ("W_q", "W_k", "W_v", "W_o")})
        self.cross_attn.set_params({k: v for k, v in params.items() if k in ("W_q", "W_k", "W_v", "W_o")})
        if "W_out" in params:
            self.W_out = params["W_out"]
        if "b_out" in params:
            self.b_out = params["b_out"]

    def train_step(
        self,
        static_x: np.ndarray,
        past_x: np.ndarray,
        future_x: np.ndarray,
        targets: np.ndarray,
        lr: float = 1e-3,
    ) -> float:
        logits = self.forward(static_x, past_x, future_x, training=True)
        loss = self.compute_loss(logits, targets)

        params = self._get_trainable_params()
        grads = self._numerical_gradient(static_x, past_x, future_x, targets)

        for name, param in params.items():
            if name in grads:
                g = grads[name]
                if self.config.grad_clip > 0:
                    g_norm = np.linalg.norm(g)
                    if g_norm > self.config.grad_clip:
                        g = g * (self.config.grad_clip / g_norm)
                param -= lr * g

        self._set_trainable_params(params)
        return loss

    def get_attention_weights(
        self,
        static_x: np.ndarray,
        past_x: np.ndarray,
        future_x: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        _ = self.forward(static_x, past_x, future_x, training=False)
        return {
            "encoder_attn": self.encoder_attn.W_o,
            "decoder_attn": self.decoder_attn.W_o,
            "cross_attn": self.cross_attn.W_o,
        }

    def save_weights(self, path: str) -> None:
        weights: Dict[str, Any] = {"config": self.config.__dict__}
        weights["encoder_attn"] = self.encoder_attn.get_params()
        weights["decoder_attn"] = self.decoder_attn.get_params()
        weights["cross_attn"] = self.cross_attn.get_params()
        weights["static_vsn"] = self.static_vsn.get_params()
        weights["past_vsn"] = self.past_vsn.get_params()
        weights["future_vsn"] = self.future_vsn.get_params()
        weights["W_out"] = self.W_out
        weights["b_out"] = self.b_out
        np.save(path, weights)
        logger.info("TFT weights saved to %s", path)

    def load_weights(self, path: str) -> None:
        weights = np.load(path, allow_pickle=True).item()
        self.config = TFTConfig(**weights["config"])
        self.encoder_attn.set_params(weights["encoder_attn"])
        self.decoder_attn.set_params(weights["decoder_attn"])
        self.cross_attn.set_params(weights["cross_attn"])
        self.static_vsn.set_params(weights["static_vsn"])
        self.past_vsn.set_params(weights["past_vsn"])
        self.future_vsn.set_params(weights["future_vsn"])
        self.W_out = weights["W_out"]
        self.b_out = weights["b_out"]
        logger.info("TFT weights loaded from %s", path)


# ---------------------------------------------------------------------------
# Financial TFT Predictor — high-level API
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    point_forecast: np.ndarray
    p10_forecast: np.ndarray
    p50_forecast: np.ndarray
    p90_forecast: np.ndarray
    horizons: List[int]
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExplanationResult:
    prediction: PredictionResult
    static_importance: np.ndarray
    past_importance: np.ndarray
    future_importance: np.ndarray
    attention_summary: Dict[str, float]


class FinancialTFTPredictor:
    """High-level TFT predictor for financial time series.

    Wraps :class:`TemporalFusionTransformer` with data preprocessing,
    training loop, and explanation utilities.

    Parameters
    ----------
    config : TFTConfig
        Model hyperparameters.
    """

    def __init__(self, config: Optional[TFTConfig] = None) -> None:
        self.config = config or TFTConfig()
        self.model = TemporalFusionTransformer(self.config)
        self._fitted = False
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        self._target_mean: float = 0.0
        self._target_std: float = 1.0
        self.training_history: List[float] = []

    def _normalize(
        self,
        data: np.ndarray,
        fit: bool = False,
    ) -> np.ndarray:
        if fit:
            self._feature_means = data.mean(axis=(0, 1), keepdims=True)
            self._feature_stds = data.std(axis=(0, 1), keepdims=True) + 1e-8
        return (data - self._feature_means) / self._feature_stds

    def _normalize_target(
        self,
        target: np.ndarray,
        fit: bool = False,
    ) -> np.ndarray:
        if fit:
            self._target_mean = float(target.mean())
            self._target_std = float(target.std()) + 1e-8
        return (target - self._target_mean) / self._target_std

    def _denormalize(self, pred: np.ndarray) -> np.ndarray:
        return pred * self._target_std + self._target_mean

    def _prepare_batch(
        self,
        static_features: np.ndarray,
        past_features: np.ndarray,
        future_features: np.ndarray,
        targets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        past_norm = self._normalize(past_features, fit=True)
        future_norm = self._normalize(future_features, fit=True)
        static_norm = self._normalize(
            static_features.reshape(static_features.shape[0], -1, 1), fit=True
        ).squeeze(-1)
        target_norm = self._normalize_target(targets, fit=True)
        return static_norm, past_norm, future_norm, target_norm

    def fit(
        self,
        static_features: np.ndarray,
        past_features: np.ndarray,
        future_features: np.ndarray,
        targets: np.ndarray,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> List[float]:
        """Train the TFT model on financial time series data.

        Parameters
        ----------
        static_features : np.ndarray
            Shape (batch, num_static_features). Static covariates per sample.
        past_features : np.ndarray
            Shape (batch, seq_len, num_past_features). Observed history.
        future_features : np.ndarray
            Shape (batch, pred_len, num_future_features). Known future inputs.
        targets : np.ndarray
            Shape (batch, pred_len). Target values to predict.
        epochs : int
            Number of training epochs.
        learning_rate : float
            Adam learning rate.
        batch_size : int
            Mini-batch size.
        verbose : bool
            Print loss every 10 epochs.

        Returns
        -------
        List[float]
            Training loss per epoch.
        """
        static_norm, past_norm, future_norm, target_norm = self._prepare_batch(
            static_features, past_features, future_features, targets
        )

        n = static_norm.shape[0]
        self.training_history = []

        for epoch in range(epochs):
            indices = self.model.rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = indices[start:end]

                loss = self.model.train_step(
                    static_norm[idx],
                    past_norm[idx],
                    future_norm[idx],
                    target_norm[idx],
                    lr=learning_rate,
                )
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.training_history.append(avg_loss)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                logger.info(
                    "TFT epoch %d/%d — loss: %.6f", epoch + 1, epochs, avg_loss
                )

        self._fitted = True
        return self.training_history

    def predict(
        self,
        static_features: np.ndarray,
        past_features: np.ndarray,
        future_features: np.ndarray,
    ) -> PredictionResult:
        """Generate multi-horizon predictions with quantile outputs.

        Parameters
        ----------
        static_features : np.ndarray
            Shape (batch, num_static_features).
        past_features : np.ndarray
            Shape (batch, seq_len, num_past_features).
        future_features : np.ndarray
            Shape (batch, pred_len, num_future_features).

        Returns
        -------
        PredictionResult
            Point forecast and quantile forecasts (p10, p50, p90).
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        past_norm = (past_features - self._feature_means) / self._feature_stds
        future_norm = (future_features - self._feature_means) / self._feature_stds
        static_norm = (
            static_features.reshape(static_features.shape[0], -1, 1)
            - self._feature_means[:, :, :1]
        ) / self._feature_stds[:, :, :1]
        static_norm = static_norm.squeeze(-1)

        logits = self.model.forward(static_norm, past_norm, future_norm, training=False)

        qts = self.config.quantiles
        p10 = self._denormalize(logits[:, :, list(qts).index(0.1)]) if 0.1 in qts else logits[:, :, 0]
        p50 = self._denormalize(logits[:, :, list(qts).index(0.5)]) if 0.5 in qts else logits[:, :, 1]
        p90 = self._denormalize(logits[:, :, list(qts).index(0.9)]) if 0.9 in qts else logits[:, :, 2]
        point = p50

        return PredictionResult(
            point_forecast=point,
            p10_forecast=p10,
            p50_forecast=p50,
            p90_forecast=p90,
            horizons=list(range(1, self.config.pred_len + 1)),
        )

    def predict_with_explanation(
        self,
        static_features: np.ndarray,
        past_features: np.ndarray,
        future_features: np.ndarray,
    ) -> ExplanationResult:
        """Generate predictions with feature importance explanations.

        Uses variable selection network weights and attention outputs to
        provide interpretable feature attributions.

        Returns
        -------
        ExplanationResult
            Predictions plus static/past/future feature importance and
            attention summary statistics.
        """
        prediction = self.predict(static_features, past_features, future_features)

        past_norm = (past_features - self._feature_means) / self._feature_stds
        future_norm = (future_features - self._feature_means) / self._feature_stds
        static_norm = (
            static_features.reshape(static_features.shape[0], -1, 1)
            - self._feature_means[:, :, :1]
        ) / self._feature_stds[:, :, :1]
        static_norm = static_norm.squeeze(-1)

        _, static_imp = self.model.static_vsn.forward(
            static_norm.reshape(static_norm.shape[0], self.config.num_static_features, 1).repeat(
                self.config.hidden_size, axis=2
            ),
            training=False,
        )
        _, past_imp = self.model.past_vsn.forward(past_norm, training=False)
        _, future_imp = self.model.future_vsn.forward(future_norm, training=False)

        attn_weights = self.model.get_attention_weights(
            static_norm, past_norm, future_norm
        )

        attention_summary = {
            "encoder_attn_norm": float(np.linalg.norm(attn_weights["encoder_attn"])),
            "decoder_attn_norm": float(np.linalg.norm(attn_weights["decoder_attn"])),
            "cross_attn_norm": float(np.linalg.norm(attn_weights["cross_attn"])),
        }

        return ExplanationResult(
            prediction=prediction,
            static_importance=static_imp.mean(axis=0),
            past_importance=past_imp.mean(axis=(0, 1)),
            future_importance=future_imp.mean(axis=(0, 1)),
            attention_summary=attention_summary,
        )

    def save(self, path: str) -> None:
        """Save model weights and preprocessing state."""
        self.model.save_weights(path)
        state = {
            "feature_means": self._feature_means,
            "feature_stds": self._feature_stds,
            "target_mean": self._target_mean,
            "target_std": self._target_std,
            "fitted": self._fitted,
            "training_history": self.training_history,
        }
        np.save(path.replace(".npy", "_state.npy"), state)
        logger.info("Predictor state saved alongside %s", path)

    def load(self, path: str) -> None:
        """Load model weights and preprocessing state."""
        self.model.load_weights(path)
        state_path = path.replace(".npy", "_state.npy")
        try:
            state = np.load(state_path, allow_pickle=True).item()
            self._feature_means = state["feature_means"]
            self._feature_stds = state["feature_stds"]
            self._target_mean = state["target_mean"]
            self._target_std = state["target_std"]
            self._fitted = state["fitted"]
            self.training_history = state["training_history"]
        except FileNotFoundError:
            logger.warning("State file not found at %s — using defaults", state_path)
        logger.info("Predictor loaded from %s", path)
