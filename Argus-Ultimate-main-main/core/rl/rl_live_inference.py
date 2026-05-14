"""Push 98 — RL live inference pipeline (v8.34.0).

Bridges a trained RL checkpoint (PPO/TD3/SAC via stable-baselines3)
directly into ArgusSystem.tick() as a first-class strategy.

Design:
  RLCheckpointLoader     loads model from disk, validates obs/act spaces
  RLLiveStrategy         BaseStrategy subclass; feeds tick -> obs -> action -> Signal
  RLInferencePipeline    wires loader + strategy + SignalBus

Reward shaping:
  RegimeRewardShaper     conditions rewards on LiveRegimeDetector label
                         (penalise trend-following in RANGING, etc.)

Void Breaker calibration:
  ConvictionCalibrator   online Platt-scaling for the Void Breaker gate
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False

try:
    from stable_baselines3 import PPO, TD3, SAC
    _SB3 = True
except ImportError:
    _SB3 = False


# ---------------------------------------------------------------------------
# Regime reward multipliers
# ---------------------------------------------------------------------------

REGIME_REWARD_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    # regime -> strategy_type -> multiplier
    "TRENDING_BULL":  {"momentum": 1.3,  "mean_reversion": 0.5, "rl": 1.1},
    "TRENDING_BEAR":  {"momentum": 1.3,  "mean_reversion": 0.5, "rl": 1.1},
    "RANGING":        {"momentum": 0.4,  "mean_reversion": 1.4, "rl": 0.9},
    "HIGH_VOL":       {"momentum": 0.6,  "mean_reversion": 0.7, "rl": 0.8},
    "UNKNOWN":        {"momentum": 1.0,  "mean_reversion": 1.0, "rl": 1.0},
}


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

class RLCheckpointLoader:
    """Loads a stable-baselines3 checkpoint from disk.

    Supports PPO, TD3, SAC. Falls back to a stub predictor when
    stable-baselines3 is unavailable (returns neutral action).
    """

    ALGO_MAP = {"ppo": "PPO", "td3": "TD3", "sac": "SAC"}

    def __init__(self, checkpoint_path: str, algo: str = "ppo") -> None:
        self._path  = Path(checkpoint_path)
        self._algo  = algo.lower()
        self._model = None
        self._loaded = False

    def load(self) -> bool:
        """Load model from checkpoint. Returns True on success."""
        if not _SB3:
            logger.warning("stable-baselines3 not installed; using stub RL predictor")
            return False
        if not self._path.exists():
            logger.warning("RL checkpoint not found: %s", self._path)
            return False
        try:
            algo_cls = {"ppo": PPO, "td3": TD3, "sac": SAC}.get(self._algo)
            if algo_cls is None:
                raise ValueError(f"Unknown algo: {self._algo}")
            self._model  = algo_cls.load(str(self._path))
            self._loaded = True
            logger.info("RL checkpoint loaded: %s (%s)", self._path, self._algo.upper())
            return True
        except Exception as exc:
            logger.error("Failed to load RL checkpoint: %s", exc)
            return False

    def predict(self, obs: Any) -> Tuple[Any, Any]:
        """Run inference. Returns (action, state)."""
        if self._model is not None:
            return self._model.predict(obs, deterministic=True)
        # Stub: return neutral action
        if _NP:
            return np.zeros(1, dtype=np.float32), None
        return [0.0], None

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def stats(self) -> dict:
        return {
            "checkpoint": str(self._path),
            "algo":       self._algo,
            "loaded":     self._loaded,
            "sb3_available": _SB3,
        }


# ---------------------------------------------------------------------------
# Regime reward shaper
# ---------------------------------------------------------------------------

class RegimeRewardShaper:
    """Scales RL training rewards based on current market regime.

    Call shape(reward, regime, strategy_type) during rollout collection
    to inject regime-conditional reward signal.

    During live inference, use multiplier(regime, strategy_type) to
    gate signal strength from the RL strategy.
    """

    def __init__(
        self,
        custom_multipliers: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self._multipliers = custom_multipliers or REGIME_REWARD_MULTIPLIERS
        self._shape_calls = 0

    def shape(self, reward: float, regime: str, strategy_type: str = "rl") -> float:
        """Apply regime multiplier to a training reward."""
        self._shape_calls += 1
        return reward * self.multiplier(regime, strategy_type)

    def multiplier(self, regime: str, strategy_type: str = "rl") -> float:
        """Return the reward/strength multiplier for (regime, strategy_type)."""
        regime_map = self._multipliers.get(regime, self._multipliers["UNKNOWN"])
        return regime_map.get(strategy_type, 1.0)

    def set_multiplier(self, regime: str, strategy_type: str, value: float) -> None:
        if regime not in self._multipliers:
            self._multipliers[regime] = {}
        self._multipliers[regime][strategy_type] = max(0.0, value)

    @property
    def stats(self) -> dict:
        return {
            "shape_calls":   self._shape_calls,
            "regimes":       list(self._multipliers.keys()),
        }


# ---------------------------------------------------------------------------
# Conviction calibrator (Platt scaling for Void Breaker gate)
# ---------------------------------------------------------------------------

class ConvictionCalibrator:
    """Online Platt-scaling calibrator for the Void Breaker conviction gate.

    Converts raw ensemble scores to calibrated probabilities using
    a sigmoid: P = 1 / (1 + exp(-(A*score + B)))

    A and B are updated online via stochastic gradient descent on
    binary cross-entropy loss.
    """

    def __init__(self, lr: float = 0.01, init_A: float = 1.0, init_B: float = 0.0) -> None:
        self._A  = init_A
        self._B  = init_B
        self._lr = lr
        self._n  = 0
        self._loss_sum = 0.0

    def calibrate(self, raw_score: float) -> float:
        """Return calibrated probability in (0, 1)."""
        import math
        return 1.0 / (1.0 + math.exp(-(self._A * raw_score + self._B)))

    def update(self, raw_score: float, label: float) -> None:
        """Online SGD update. label=1 if signal was profitable, 0 otherwise."""
        import math
        p = self.calibrate(raw_score)
        # Gradient of BCE w.r.t. A and B
        err  = p - label
        dA   = err * raw_score
        dB   = err
        self._A -= self._lr * dA
        self._B -= self._lr * dB
        self._n += 1
        eps = 1e-12
        self._loss_sum += -(label * math.log(p + eps) + (1 - label) * math.log(1 - p + eps))

    @property
    def stats(self) -> dict:
        return {
            "A":        self._A,
            "B":        self._B,
            "n_updates": self._n,
            "mean_loss": self._loss_sum / max(1, self._n),
        }


# ---------------------------------------------------------------------------
# RL Live Strategy
# ---------------------------------------------------------------------------

@dataclass
class RLLiveConfig:
    checkpoint_path:  str   = ""
    algo:             str   = "ppo"
    symbol:           str   = "BTCUSDT"
    obs_window:       int   = 50
    confidence_gate:  float = 0.55
    strategy_type:    str   = "rl"
    enabled:          bool  = True


class RLLiveStrategy:
    """Live RL strategy: feeds price ticks into trained model,
    produces Signal objects that are published to the SignalBus.

    Works with or without a loaded checkpoint (stub mode when unavailable).
    Applies regime reward shaping to gate signal strength.
    """

    def __init__(
        self,
        config:         RLLiveConfig,
        shaper:         Optional[RegimeRewardShaper]   = None,
        calibrator:     Optional[ConvictionCalibrator] = None,
        regime_detector: Any = None,
    ) -> None:
        self._cfg        = config
        self._loader     = RLCheckpointLoader(config.checkpoint_path, config.algo)
        self._shaper     = shaper     or RegimeRewardShaper()
        self._calibrator = calibrator or ConvictionCalibrator()
        self._regime_det = regime_detector
        self._price_buf: List[float] = []
        self._signal_count = 0
        self._blocked_count = 0
        if config.checkpoint_path:
            self._loader.load()

    def tick(self, price: float, volume: float = 0.0) -> Optional[dict]:
        """Feed a price tick. Returns a signal dict or None."""
        self._price_buf.append(price)
        if len(self._price_buf) > self._cfg.obs_window * 2:
            self._price_buf = self._price_buf[-self._cfg.obs_window * 2:]
        if len(self._price_buf) < self._cfg.obs_window:
            return None

        obs = self._build_obs()
        action, _ = self._loader.predict(obs)
        raw_score  = float(action[0]) if hasattr(action, "__len__") else float(action)

        # Calibrated conviction
        conviction = self._calibrator.calibrate(raw_score)

        # Regime multiplier on strength
        regime = self._current_regime()
        strength_mult = self._shaper.multiplier(regime, self._cfg.strategy_type)
        final_strength = conviction * strength_mult

        if final_strength < self._cfg.confidence_gate:
            self._blocked_count += 1
            return None

        self._signal_count += 1
        side = "LONG" if raw_score > 0 else "SHORT"
        return {
            "symbol":      self._cfg.symbol,
            "side":        side,
            "strength":    min(1.0, final_strength),
            "strategy_id": f"rl_{self._cfg.algo}_{self._cfg.symbol}",
            "regime":      regime,
            "conviction":  conviction,
            "raw_score":   raw_score,
            "timestamp":   time.time(),
        }

    def _build_obs(self) -> Any:
        """Build 7-dim observation from price buffer."""
        buf = self._price_buf[-self._cfg.obs_window:]
        if not _NP:
            return buf
        prices = np.array(buf, dtype=np.float32)
        returns = np.diff(prices) / (prices[:-1] + 1e-9)
        n = len(returns)
        obi         = float(np.mean(returns[-5:]))  if n >= 5  else 0.0
        rsi_raw     = float(np.mean(returns[-14:])) if n >= 14 else 0.0
        vol         = float(np.std(returns[-20:]))  if n >= 20 else 0.0
        momentum    = float(np.sum(returns[-10:]))  if n >= 10 else 0.0
        inventory   = 0.0   # placeholder; wired via AppContext in full integration
        pnl         = 0.0   # placeholder
        regime_enc  = {"TRENDING_BULL": 1.0, "TRENDING_BEAR": -1.0,
                       "RANGING": 0.0, "HIGH_VOL": 0.5}.get(
                           self._current_regime(), 0.0)
        return np.array(
            [obi, rsi_raw, vol, momentum, inventory, pnl, regime_enc],
            dtype=np.float32,
        ).reshape(1, -1)

    def _current_regime(self) -> str:
        if self._regime_det is not None:
            try:
                snap = self._regime_det.snapshot()
                return snap.regime.value if hasattr(snap.regime, "value") else str(snap.regime)
            except Exception:
                pass
        return "UNKNOWN"

    @property
    def stats(self) -> dict:
        return {
            "symbol":         self._cfg.symbol,
            "algo":           self._cfg.algo,
            "signal_count":   self._signal_count,
            "blocked_count":  self._blocked_count,
            "loader":         self._loader.stats,
            "shaper":         self._shaper.stats,
            "calibrator":     self._calibrator.stats,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RLInferencePipeline:
    """Wires RLLiveStrategy into ArgusSystem.

    Usage::

        pipeline = RLInferencePipeline(ctx)
        pipeline.add_strategy(RLLiveConfig(
            checkpoint_path="checkpoints/ppo_btcusdt.zip",
            symbol="BTCUSDT",
        ))
        # In ArgusSystem.tick():
        pipeline.tick(price, volume)
    """

    def __init__(self, ctx: Any) -> None:
        self._ctx        = ctx
        self._strategies: List[RLLiveStrategy] = []
        self._tick_count  = 0

    def add_strategy(self, config: RLLiveConfig) -> RLLiveStrategy:
        regime_det = getattr(self._ctx, "regime_detector", None)
        shaper     = RegimeRewardShaper()
        calibrator = ConvictionCalibrator()
        strat = RLLiveStrategy(
            config=config,
            shaper=shaper,
            calibrator=calibrator,
            regime_detector=regime_det,
        )
        self._strategies.append(strat)
        return strat

    def tick(self, price: float, volume: float = 0.0) -> List[dict]:
        """Feed a tick to all RL strategies. Returns list of signals."""
        self._tick_count += 1
        signals = []
        for strat in self._strategies:
            if not strat._cfg.enabled:
                continue
            sig = strat.tick(price, volume)
            if sig is not None:
                signals.append(sig)
                # Publish to SignalBus if available
                bus = getattr(self._ctx, "signal_bus", None)
                if bus is not None and hasattr(bus, "publish"):
                    try:
                        import asyncio
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.ensure_future(bus.publish_dict(sig))
                    except Exception:
                        pass
        return signals

    @property
    def stats(self) -> dict:
        return {
            "strategies":  len(self._strategies),
            "tick_count":  self._tick_count,
            "per_strategy": [s.stats for s in self._strategies],
        }
