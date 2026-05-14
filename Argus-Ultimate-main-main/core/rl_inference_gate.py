"""
core/rl_inference_gate.py
=========================
Live PyTorch ArgusAI inference gate.

Feature flags
-------------
* config  : cfg.ai.rl_enabled must be True
* env var : ARGUS_RL_INFERENCE=1

Both must be set before the gate activates. When either is absent the
gate returns a safe pass-through (size_mult=1.0, skip_trade=False) so
the rest of the pipeline is completely unaffected.

Architecture
------------
Wraps the live ArgusAI model (ml/argus_ai/model.py):

  DirectionHead  -> 3 classes: 0=flat/skip  1=long  2=short
  SizeHead       -> Beta(alpha, beta) -> mean position size in [0, 1]
  ConfidenceHead -> MC-Dropout mean + std

RLInferenceGate.infer() returns RLDecision:
  skip_trade      True when DirectionHead says flat with high confidence
  size_multiplier [0.1, 2.0] derived from SizeHead mean
  confidence      ConfidenceHead mean scalar
  direction       "long" | "short" | "flat"
  latency_ms      wall-clock inference time

Wiring
------
Instantiate once at startup and register on component_registry:

    gate = RLInferenceGate.from_config(cfg)
    component_registry.rl_inference_gate = gate

In the signal loop after compute_position_size():

    gate = getattr(system.component_registry, "rl_inference_gate", None)
    if gate is not None and gate.enabled:
        obs = RLInferenceGate.build_obs(sig_fields, ctx)
        decision = gate.infer(obs)
        if decision.skip_trade:
            continue
        size_pct *= decision.size_multiplier
        sizing_method += f"+rl_gate({decision.direction},{decision.confidence:.2f})"

Checkpoint
----------
Expects models/rl_policy.pt saved by scripts/export_rl_checkpoint.py:

    torch.save({
        "model_state_dict" : model.state_dict(),
        "argus_ai_kwargs"  : {...},   # passed to ArgusAI(**kwargs)
        "obs_dim"          : 64,
        "n_actions"        : 3,
    }, "models/rl_policy.pt")
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

_ENV_FLAG        = "ARGUS_RL_INFERENCE"
_CHECKPOINT_NAME = "rl_policy.pt"
_OBS_DIM         = 64


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class RLDecision:
    direction     : Literal["long", "short", "flat"]
    size_multiplier: float          # [0.1, 2.0]
    confidence    : float           # [0.0, 1.0]
    skip_trade    : bool
    latency_ms    : float
    direction_probs: List[float] = field(default_factory=list)
    source        : str = "rl_inference_gate"


_PASSTHROUGH = RLDecision(
    direction="long",
    size_multiplier=1.0,
    confidence=0.5,
    skip_trade=False,
    latency_ms=0.0,
    source="rl_gate_disabled",
)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

class RLInferenceGate:
    """
    Live PyTorch inference gate wrapping ArgusAI.

    Thread-safe: a single threading.Lock guards model.eval() forward calls.
    The model is kept in eval() mode permanently; RLTuner updates happen
    on a separate training thread and swap state_dict under the same lock.
    """

    # Direction index mapping
    _DIR_MAP: Dict[int, Literal["flat", "long", "short"]] = {
        0: "flat",
        1: "long",
        2: "short",
    }
    # Size multiplier map: SizeHead mean in [0,1] -> position multiplier
    # mean < 0.20 -> reduce, mean > 0.60 -> boost, else neutral
    _SIZE_BREAKPOINTS = [(0.20, 0.5), (0.40, 0.85), (0.60, 1.0), (0.80, 1.4), (1.01, 1.8)]

    def __init__(
        self,
        model_dir        : str  = "models/",
        enabled          : bool = True,
        device           : str  = "cpu",
        flat_threshold   : float = 0.55,
        mc_samples       : int  = 5,
    ) -> None:
        """
        Args:
            model_dir      : Directory containing rl_policy.pt
            enabled        : Overridden to False when ARGUS_RL_INFERENCE!=1
            device         : "cpu" or "cuda:N"
            flat_threshold : If P(flat) > this and P(flat) > max(long,short)
                             → skip_trade=True
            mc_samples     : MC-Dropout forward passes for confidence estimate
        """
        self.model_dir      = Path(model_dir)
        self.device         = device
        self.flat_threshold = flat_threshold
        self.mc_samples     = mc_samples

        _env_ok   = os.environ.get(_ENV_FLAG, "").lower() in ("1", "true", "yes")
        self.enabled = enabled and _env_ok

        self._lock        : threading.Lock = threading.Lock()
        self._model       : Any = None
        self._loaded      : bool = False
        self._load_error  : Optional[str] = None
        self._checkpoint  : Optional[Path] = None
        self._obs_dim     : int = _OBS_DIM

        if self.enabled:
            self._load()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Any) -> "RLInferenceGate":
        """
        Build from ArgusConfig.  Reads:
            cfg.ai.rl_enabled
            cfg.ai.model_dir
            cfg.ai.gpu_index       (optional)
            cfg.ai.inference_batch_size  (unused here but logged)
        """
        ai        = getattr(cfg, "ai", None)
        enabled   = bool(getattr(ai, "rl_enabled",  False))     if ai else False
        model_dir = str(getattr(ai,  "model_dir",   "models/")) if ai else "models/"
        gpu_idx   = getattr(ai, "gpu_index", None)               if ai else None
        device    = f"cuda:{gpu_idx}" if gpu_idx is not None else "cpu"
        logger.info(
            "RLInferenceGate.from_config: enabled=%s model_dir=%s device=%s",
            enabled, model_dir, device,
        )
        return cls(model_dir=model_dir, enabled=enabled, device=device)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        ckpt = self.model_dir / _CHECKPOINT_NAME
        if not ckpt.exists():
            self._load_error = f"no_checkpoint:{ckpt}"
            logger.info(
                "RLInferenceGate: no checkpoint at %s — "
                "run scripts/export_rl_checkpoint.py to generate one", ckpt,
            )
            return
        try:
            import torch
            from ml.argus_ai.model import ArgusAI

            data = torch.load(str(ckpt), map_location=self.device)
            kwargs = data.get("argus_ai_kwargs", {})
            self._obs_dim = int(data.get("obs_dim", _OBS_DIM))

            model = ArgusAI(**kwargs)
            model.load_state_dict(data["model_state_dict"])
            model.to(self.device)
            model.eval()

            self._model   = model
            self._loaded  = True
            self._checkpoint = ckpt
            logger.info(
                "RLInferenceGate: loaded checkpoint %s (obs_dim=%d, device=%s)",
                ckpt, self._obs_dim, self.device,
            )
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning("RLInferenceGate: failed to load checkpoint: %s", exc)

    def reload(self) -> bool:
        """Hot-reload checkpoint (called by RLTuner after an update)."""
        with self._lock:
            self._loaded = False
            self._load()
            return self._loaded

    def swap_state_dict(self, state_dict: Any) -> None:
        """Atomically swap in a freshly-trained state_dict from RLTuner."""
        with self._lock:
            if self._model is not None:
                self._model.load_state_dict(state_dict)
                logger.debug("RLInferenceGate: state_dict swapped in-place")

    # ------------------------------------------------------------------
    # Observation builder (static — usable without instantiating gate)
    # ------------------------------------------------------------------

    @staticmethod
    def build_obs(
        sig_fields : dict,
        ctx        : dict,
        extra      : Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Build a float32 observation vector from signal fields + execution
        context.  Pads / truncates to _OBS_DIM (64).

        Feature order (must match export_rl_checkpoint.py training features):
         0  confidence
         1  strength
         2  sig_age  / 120   (normalised to [0,1])
         3  age_urgency
         4  num_confirmations / 10
         5  1.0 if BUY else 0.0
         6  regime_pos_mult
         7  regime_stop_mult
         8  session_mult
         9  macro_event_imminent
        10  daily_loss_exceeded
        11  var_breach
        12  portfolio_value / 100_000
        13+ extra features (optional)
        """
        base = [
            float(sig_fields.get("confidence",          0.5)),
            float(sig_fields.get("strength",             0.5)),
            float(sig_fields.get("_sig_age",             0.0)) / 120.0,
            float(sig_fields.get("_age_urgency",         0.5)),
            float(sig_fields.get("_num_confirmations",   0))   / 10.0,
            1.0 if sig_fields.get("action") == "BUY" else 0.0,
            float(ctx.get("regime_pos_mult",  1.0)),
            float(ctx.get("regime_stop_mult", 1.0)),
            float(ctx.get("session_mult",      1.0)),
            1.0 if ctx.get("macro_event_imminent") else 0.0,
            1.0 if ctx.get("daily_loss_exceeded")  else 0.0,
            1.0 if ctx.get("var_breach")           else 0.0,
            float(ctx.get("portfolio_value", 10_000.0)) / 100_000.0,
        ]
        if extra:
            base.extend([float(x) for x in extra])
        arr = np.array(base, dtype=np.float32)
        if len(arr) < _OBS_DIM:
            arr = np.pad(arr, (0, _OBS_DIM - len(arr)))
        return arr[:_OBS_DIM]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, obs: Any) -> RLDecision:
        """
        Run a forward pass through ArgusAI and return an RLDecision.

        obs: array-like of shape (_OBS_DIM,) or compatible.
             Use RLInferenceGate.build_obs() to construct.

        Falls back to _PASSTHROUGH on any error so the calling pipeline
        is never interrupted by a gate failure.
        """
        t0 = time.perf_counter()

        if not self.enabled or not self._loaded or self._model is None:
            return _PASSTHROUGH

        try:
            import torch
            import torch.nn.functional as F

            obs_arr = np.asarray(obs, dtype=np.float32).flatten()
            if len(obs_arr) < self._obs_dim:
                obs_arr = np.pad(obs_arr, (0, self._obs_dim - len(obs_arr)))
            obs_arr = obs_arr[:self._obs_dim]

            # ArgusAI expects regime_ids + at least one modality.
            # We feed the observation vector as a 1-step LOB sequence
            # (padded from 64 -> 128 dims) and a neutral regime.
            with self._lock:
                self._model.eval()
                with torch.no_grad():
                    # Build minimal inputs
                    lob_input = np.zeros(128, dtype=np.float32)
                    lob_input[:self._obs_dim] = obs_arr
                    lob_t = torch.tensor(
                        lob_input, dtype=torch.float32, device=self.device
                    ).unsqueeze(0).unsqueeze(0)  # (1, 1, 128)

                    regime_ids = torch.zeros(1, dtype=torch.long, device=self.device)

                    # MC-Dropout passes for confidence
                    dir_probs_list  = []
                    size_means_list = []
                    conf_means_list = []

                    # First pass: model in eval (no dropout)
                    out = self._model(regime_ids=regime_ids, lob=lob_t)
                    dir_probs_list.append(
                        F.softmax(out.direction_logits, dim=-1)[0].cpu().numpy()
                    )
                    size_means_list.append(float(out.size_mean[0].item()))
                    conf_means_list.append(float(out.confidence_mean[0].item()))

                    # Extra MC passes with dropout enabled
                    if self.mc_samples > 1:
                        self._model.train()  # enables dropout for MC
                        for _ in range(self.mc_samples - 1):
                            out_mc = self._model(regime_ids=regime_ids, lob=lob_t)
                            dir_probs_list.append(
                                F.softmax(out_mc.direction_logits, dim=-1)[0].cpu().numpy()
                            )
                            size_means_list.append(float(out_mc.size_mean[0].item()))
                            conf_means_list.append(float(out_mc.confidence_mean[0].item()))
                        self._model.eval()

            # Aggregate MC samples
            dir_probs = np.mean(dir_probs_list, axis=0)   # (3,)
            size_mean = float(np.mean(size_means_list))
            confidence = float(np.mean(conf_means_list))
            confidence = max(0.0, min(1.0, confidence))

            # Direction
            best_action = int(np.argmax(dir_probs))
            direction   = self._DIR_MAP.get(best_action, "flat")
            skip_trade  = (
                best_action == 0
                and float(dir_probs[0]) > self.flat_threshold
            )

            # Size multiplier from Beta mean
            size_mult = 1.0
            for threshold, mult in self._SIZE_BREAKPOINTS:
                if size_mean < threshold:
                    size_mult = mult
                    break
            size_mult = max(0.1, min(2.0, size_mult))

            # If direction opposes signal action, halve multiplier
            # (the caller knows if it's a BUY or SELL signal)
            # Note: caller must check decision.direction vs sig_fields["action"]

            latency_ms = (time.perf_counter() - t0) * 1000.0
            logger.debug(
                "RLInferenceGate: dir=%s probs=%s size_mean=%.3f "
                "size_mult=%.2f conf=%.3f skip=%s latency=%.2fms",
                direction, np.round(dir_probs, 3).tolist(),
                size_mean, size_mult, confidence, skip_trade, latency_ms,
            )

            return RLDecision(
                direction=direction,
                size_multiplier=size_mult,
                confidence=confidence,
                skip_trade=skip_trade,
                latency_ms=latency_ms,
                direction_probs=dir_probs.tolist(),
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            logger.warning("RLInferenceGate.infer: error — %s; returning passthrough", exc)
            return RLDecision(
                direction="long",
                size_multiplier=1.0,
                confidence=0.5,
                skip_trade=False,
                latency_ms=latency_ms,
                source="rl_gate_error",
            )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def status(self) -> dict:
        return {
            "enabled"        : self.enabled,
            "loaded"         : self._loaded,
            "checkpoint"     : str(self._checkpoint) if self._checkpoint else None,
            "load_error"     : self._load_error,
            "device"         : self.device,
            "flat_threshold" : self.flat_threshold,
            "mc_samples"     : self.mc_samples,
        }
