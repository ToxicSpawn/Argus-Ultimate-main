#!/usr/bin/env python3
"""
scripts/export_rl_checkpoint.py
================================
Train ArgusAI on synthetic paper-trade rollouts and export
`models/rl_policy.pt` so that RLInferenceGate can load it.

Usage
-----
    # Quick smoke-test (~30 s, CPU)
    python scripts/export_rl_checkpoint.py --episodes 200 --device cpu

    # Full pre-train on GPU (RTX 3070 Ti, ~10 min)
    python scripts/export_rl_checkpoint.py --episodes 5000 --device cuda:0

    # Validate the checkpoint loads cleanly
    python scripts/export_rl_checkpoint.py --validate-only

What it does
------------
1.  Instantiates a lightweight ArgusAI (2-layer backbone, 256d) suitable
    for inference-gate pre-training without a full dataset.
2.  Generates synthetic "signal → fill → reward" rollouts using a
    configurable paper-trade simulator.
3.  Runs PPO updates via the existing RLTuner.
4.  Saves the final state_dict + metadata to models/rl_policy.pt.
5.  Optionally validates the saved checkpoint by loading it through
    RLInferenceGate and running a forward pass.

Synthetic reward function
-------------------------
    r = pnl_pct - 0.5 * |max_drawdown| - 0.001 * commission

This is intentionally simple — the gate learns to size down on low-
confidence signals and skip during flat/volatile regimes.  After
live deployment, the RLTuner continues online updates using real fills.
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger("export_rl_checkpoint")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIR  = ROOT / "models"
CHECKPOINT_NAME    = "rl_policy.pt"
OBS_DIM            = 64
N_ACTIONS          = 3          # 0=flat  1=long  2=short
D_MODEL            = 256        # keep small for pre-train speed
N_LAYERS           = 2
N_HEADS            = 4
COMMISSION_RATE    = 0.001


# ---------------------------------------------------------------------------
# Synthetic rollout
# ---------------------------------------------------------------------------

def _make_obs(rng: random.Random) -> np.ndarray:
    """Generate a plausible 64-dim observation vector."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0]  = rng.uniform(0.3, 1.0)   # confidence
    obs[1]  = rng.uniform(0.3, 1.0)   # strength
    obs[2]  = rng.uniform(0.0, 1.0)   # age / 120
    obs[3]  = rng.uniform(0.0, 1.0)   # age_urgency
    obs[4]  = rng.uniform(0.0, 1.0)   # num_confirmations / 10
    obs[5]  = rng.choice([0.0, 1.0])  # BUY flag
    obs[6]  = rng.uniform(0.5, 1.5)   # regime_pos_mult
    obs[7]  = rng.uniform(0.8, 1.2)   # regime_stop_mult
    obs[8]  = rng.uniform(0.8, 1.2)   # session_mult
    obs[9]  = rng.choice([0.0, 0.0, 0.0, 1.0])  # macro_event (rare)
    obs[10] = rng.choice([0.0, 0.0, 0.0, 1.0])  # daily_loss_exceeded
    obs[11] = rng.choice([0.0, 0.0, 1.0])        # var_breach
    obs[12] = rng.uniform(0.05, 0.5)              # portfolio_value / 100k
    return obs


def _simulate_reward(
    obs: np.ndarray,
    action: int,
    rng: random.Random,
) -> float:
    """
    Simple reward simulator:
    - action 0 (flat): small negative reward when signal was high-conviction
    - action 1 (long): positive when conf*strength > 0.5 and obs[5]=1 (BUY)
    - action 2 (short): positive when conf*strength > 0.5 and obs[5]=0 (SELL)
    Commission always deducted for non-flat actions.
    """
    confidence  = float(obs[0])
    strength    = float(obs[1])
    is_buy      = float(obs[5]) > 0.5
    regime_mult = float(obs[6])
    macro       = float(obs[9]) > 0.5
    var_breach  = float(obs[11]) > 0.5

    signal_quality = confidence * strength * regime_mult

    if action == 0:  # flat / skip
        # Penalise skipping a high-quality signal, reward skipping bad ones
        r = -signal_quality * 0.3 + (1.0 - signal_quality) * 0.1
    else:
        # Trade reward: pnl sampled around signal_quality
        pnl_mean = signal_quality * 0.02 - 0.005
        if (action == 1 and not is_buy) or (action == 2 and is_buy):
            pnl_mean -= 0.015  # direction mismatch penalty
        if macro or var_breach:
            pnl_mean -= 0.01   # high-risk context
        pnl = rng.gauss(pnl_mean, 0.01)
        r   = pnl - COMMISSION_RATE

    # Small noise
    r += rng.gauss(0, 0.001)
    return float(r)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    episodes : int,
    device   : str,
    seed     : int,
    save_dir : Path,
) -> Path:
    import torch
    from ml.argus_ai.model import ArgusAI
    from ml.argus_ai.rl_tuner import RLTuner

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    argus_ai_kwargs = dict(
        d_model    = D_MODEL,
        n_heads    = N_HEADS,
        n_layers   = N_LAYERS,
        mc_samples = 5,
        dropout    = 0.1,
        cot_enabled = False,   # no Redis needed for pre-train
        crisis_gate = True,
    )

    logger.info("Instantiating ArgusAI (d_model=%d, n_layers=%d)", D_MODEL, N_LAYERS)
    model  = ArgusAI(**argus_ai_kwargs).to(device)
    tuner  = RLTuner(
        model          = model,
        lr             = 3e-4,
        clip_ratio     = 0.2,
        entropy_coef   = 0.02,
        value_coef     = 0.5,
        gamma          = 0.99,
        lambda_dd      = 1.0,
        max_dd_gate    = 0.30,
        update_every   = 32,
        buffer_size    = 512,
        device         = device,
    )

    total_reward = 0.0
    best_reward  = float("-inf")
    window: List[float] = []
    WINDOW = 100

    logger.info("Starting PPO pre-train: %d episodes, device=%s", episodes, device)

    for ep in range(1, episodes + 1):
        obs     = _make_obs(rng)
        obs_t   = torch.tensor(obs, dtype=torch.float32, device=device)

        # Build minimal ArgusAI inputs from obs vector
        lob_input = np.zeros(128, dtype=np.float32)
        lob_input[:OBS_DIM] = obs
        lob_t = torch.tensor(
            lob_input, dtype=torch.float32, device=device
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, 128)
        regime_ids = torch.zeros(1, dtype=torch.long, device=device)

        model.eval()
        with torch.no_grad():
            out      = model(regime_ids=regime_ids, lob=lob_t)
            logits   = out.direction_logits[0]           # (3,)
            log_probs = F.log_softmax(logits, dim=-1)
            probs     = F.softmax(logits, dim=-1)

        # Sample action
        action   = int(torch.multinomial(probs, 1).item())
        log_prob = float(log_probs[action].item())

        # Simulate reward
        reward = _simulate_reward(obs, action, rng)
        total_reward += reward
        window.append(reward)
        if len(window) > WINDOW:
            window.pop(0)

        # Push to tuner (triggers PPO update every 32 steps)
        backbone_repr = out.backbone_repr[0].detach()   # (512,) or (D_MODEL,)
        tuner.push(
            state_repr       = backbone_repr,
            action           = action,
            reward           = reward,
            log_prob         = log_prob,
            regime_id        = 0,
            done             = True,
            current_drawdown = 0.0,
        )

        if ep % 100 == 0:
            avg = sum(window) / max(len(window), 1)
            logger.info(
                "Episode %5d/%d | avg_reward(last%d)=%.5f | total=%.4f | updates=%d",
                ep, episodes, WINDOW, avg, total_reward, tuner.total_updates,
            )
            if avg > best_reward:
                best_reward = avg

    # Save
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / CHECKPOINT_NAME
    torch.save({
        "model_state_dict" : model.state_dict(),
        "argus_ai_kwargs"  : argus_ai_kwargs,
        "obs_dim"          : OBS_DIM,
        "n_actions"        : N_ACTIONS,
        "episodes_trained" : episodes,
        "final_avg_reward" : best_reward,
        "tuner_updates"    : tuner.total_updates,
    }, str(ckpt_path))

    logger.info("Checkpoint saved to %s (best_avg_reward=%.5f)", ckpt_path, best_reward)
    return ckpt_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(ckpt_path: Path) -> None:
    """Load the checkpoint through RLInferenceGate and run a forward pass."""
    import os
    os.environ["ARGUS_RL_INFERENCE"] = "1"
    from core.rl_inference_gate import RLInferenceGate

    gate = RLInferenceGate(
        model_dir = str(ckpt_path.parent),
        enabled   = True,
        device    = "cpu",
    )
    assert gate.status["loaded"], f"Checkpoint failed to load: {gate.status['load_error']}"

    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0] = 0.8   # confidence
    obs[1] = 0.7   # strength
    obs[5] = 1.0   # BUY
    obs[6] = 1.0   # regime_pos_mult
    obs[8] = 1.05  # session_mult
    obs[12] = 0.2  # portfolio_value

    decision = gate.infer(obs)
    logger.info(
        "Validation forward pass: direction=%s size_mult=%.2f conf=%.3f skip=%s latency=%.2fms",
        decision.direction, decision.size_multiplier,
        decision.confidence, decision.skip_trade, decision.latency_ms,
    )
    assert not decision.skip_trade or decision.size_multiplier >= 0.1, \
        "Gate returned invalid decision"
    logger.info("Validation PASSED")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-train ArgusAI RL gate and export models/rl_policy.pt"
    )
    parser.add_argument("--episodes",      type=int,   default=1000,
                        help="Number of synthetic training episodes (default: 1000)")
    parser.add_argument("--device",        type=str,   default="cpu",
                        help="Torch device: cpu | cuda:0 | cuda:1 (default: cpu)")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--model-dir",     type=str,   default=str(DEFAULT_MODEL_DIR),
                        help="Output directory for the checkpoint")
    parser.add_argument("--validate-only", action="store_true",
                        help="Skip training, just validate an existing checkpoint")
    args = parser.parse_args()

    save_dir  = Path(args.model_dir)
    ckpt_path = save_dir / CHECKPOINT_NAME

    if args.validate_only:
        if not ckpt_path.exists():
            logger.error("No checkpoint at %s — run without --validate-only first", ckpt_path)
            sys.exit(1)
        validate(ckpt_path)
        return

    ckpt_path = train(
        episodes = args.episodes,
        device   = args.device,
        seed     = args.seed,
        save_dir = save_dir,
    )
    validate(ckpt_path)


if __name__ == "__main__":
    main()
