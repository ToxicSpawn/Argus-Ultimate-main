"""Push 67 — V2 ControllerConfig: YAML-serialisable strategy config.

Every V2 controller is fully described by its ControllerConfig.
Supports YAML round-trip for Hummingbot-style config files.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ControllerConfig:
    """Base configuration for all V2 strategy controllers."""

    # Identity
    controller_name: str = "BaseController"
    controller_type: str = "generic"   # "rl" | "pmm" | "arb" | "dca"
    enabled: bool = True

    # Trading
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    initial_equity: float = 10_000.0
    max_positions: int = 3

    # Execution
    use_position_executor: bool = True
    use_dca: bool = False
    dca_n_levels: int = 3
    dca_level_spread: float = 0.005

    # Fee / spread
    base_bid_spread: float = 0.001
    base_ask_spread: float = 0.001
    min_profitability: float = 0.001
    order_refresh_secs: float = 30.0

    # Risk
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    partial_tp_pct: float = 0.02
    trailing_stop_pct: float = 0.015
    use_trailing: bool = True
    use_partial_tp: bool = True

    # Extra params (strategy-specific)
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ControllerConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        init_kwargs = {k: v for k, v in d.items() if k in known}
        return cls(**init_kwargs)

    def to_yaml_str(self) -> str:
        """Simple YAML serialisation (no pyyaml dependency)."""
        lines = [f"# Argus V2 Controller Config — {self.controller_name}"]
        for k, v in self.to_dict().items():
            if isinstance(v, dict):
                lines.append(f"{k}:")
                for dk, dv in v.items():
                    lines.append(f"  {dk}: {json.dumps(dv)}")
            elif isinstance(v, list):
                lines.append(f"{k}: {json.dumps(v)}")
            else:
                lines.append(f"{k}: {json.dumps(v)}")
        return "\n".join(lines)


@dataclass
class RLControllerConfig(ControllerConfig):
    """Config for the RL-powered V2 controller."""
    controller_name: str = "RLController"
    controller_type: str = "rl"
    algorithm: str = "PPO"          # "PPO" | "TD3" | "SAC"
    model_path: str = "models/rl/argus_PPO_final"
    normalizer_path: str = "models/rl/vec_normalize.pkl"
    min_conviction: float = 0.30
    use_dca: bool = True
    use_position_executor: bool = True
    retrain_every_n_trades: int = 1000
