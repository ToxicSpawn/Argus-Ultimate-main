"""Push 67 — RLController: V2 controller wiring RLStrategy signals
to PositionExecutor + DCAExecutor pipeline.

Flow:
  on_bar(bar)
    -> RLStrategy.predict(bar)  -> signal dict
    -> KellySizer.size()        -> position_usd
    -> AdverseSelectionGate     -> fill_recommended
    -> DCAExecutor (if enabled) -> tranche levels
    -> PositionExecutorEngine   -> open position
    -> create_actions_proposal  -> ExecutorAction list
"""
from __future__ import annotations

from typing import Any, List

from core.strategy.v2.strategy_controller import StrategyController, ExecutorAction, ActionType
from core.strategy.v2.controller_config import RLControllerConfig
from core.execution.position_executor import PositionExecutor, PositionExecutorEngine
from core.execution.dca_executor import DCAPlan, DCAExecutorEngine
from core.execution.fee_adjuster import FeeAdjuster
from core.risk.kelly_sizer import KellySizer
from core.risk.per_strategy_risk import PerStrategyRisk


class RLController(StrategyController):
    """V2 controller: RL signals -> position lifecycle management."""

    name = "RLController"

    def __init__(self, config: RLControllerConfig | None = None):
        cfg = config or RLControllerConfig()
        super().__init__(cfg)
        self.rl_cfg = cfg

        # Sub-systems
        self._position_engine = PositionExecutorEngine(
            max_positions=cfg.max_positions
        )
        self._dca_engine = DCAExecutorEngine()
        self._fee_adjuster = FeeAdjuster(
            base_bid_spread=cfg.base_bid_spread,
            base_ask_spread=cfg.base_ask_spread,
            min_profitability=cfg.min_profitability,
        )
        self._kelly = KellySizer()
        self._per_strategy_risk = PerStrategyRisk()
        self._trade_returns: List[float] = []
        self._rl_strategy = None  # lazy-loaded when model_path is valid
        self._equity: float = cfg.initial_equity

    # ------------------------------------------------------------------
    # V2 override: create_actions_proposal
    # ------------------------------------------------------------------

    async def create_actions_proposal(
        self, bar: Any
    ) -> List[ExecutorAction]:
        actions: List[ExecutorAction] = []

        # 1. Evaluate existing positions
        prices = {bar.symbol: bar.close}
        closed = self._position_engine.evaluate_all(prices)
        for pos in closed:
            self._trade_returns.append(pos.total_pnl / pos.size_usd)
            action_type = (
                ActionType.CLOSE_LONG if pos.side == "buy"
                else ActionType.CLOSE_SHORT
            )
            actions.append(ExecutorAction(
                action_type=action_type,
                symbol=pos.symbol,
                metadata={"reason": pos.status.value, "pnl": pos.total_pnl},
            ))

        # 2. DCA evaluation
        self._dca_engine.evaluate_all(prices)

        # 3. Per-strategy risk gate
        active = self._per_strategy_risk.update(
            self.name, self._equity, 0.0
        )
        if not active:
            return actions

        # 4. RL signal (skip if model not loaded)
        if self._rl_strategy is None:
            return actions

        pnl_norm = (self._equity - self.rl_cfg.initial_equity) / self.rl_cfg.initial_equity
        signal = self._rl_strategy.predict(bar, pnl_norm=pnl_norm)
        if signal is None:
            return actions

        # 5. Kelly sizing
        kelly_result = self._kelly.size_from_trades(
            self._equity, self._trade_returns
        )
        size_usd = kelly_result.position_usd
        if size_usd <= 0:
            size_usd = self._equity * 0.01  # fallback: 1% of equity

        # 6. Fee gate
        if not self._fee_adjuster.is_profitable(
            bar.close, bar.close * (1 + 0.004), signal["side"]
        ):
            return actions

        # 7. Propose open action
        action_type = (
            ActionType.OPEN_LONG if signal["side"] == "buy"
            else ActionType.OPEN_SHORT
        )
        actions.append(ExecutorAction(
            action_type=action_type,
            symbol=bar.symbol,
            size_usd=size_usd,
            price=bar.close,
            confidence=signal["confidence"],
            metadata={
                "algorithm": signal.get("algorithm", "PPO"),
                "target_position": signal.get("target_position", 0.0),
                "kelly_fraction": kelly_result.safe_fraction,
            },
        ))

        # 8. Open DCA plan if enabled
        if self.rl_cfg.use_dca:
            plan = DCAPlan(
                symbol=bar.symbol,
                side=signal["side"],
                total_usd=size_usd,
                n_levels=self.rl_cfg.dca_n_levels,
                level_spread=self.rl_cfg.dca_level_spread,
                strategy_name=self.name,
            )
            plan.build(bar.close)
            self._dca_engine.add_plan(plan)

        return actions

    # ------------------------------------------------------------------
    # 1-second tick: refresh fee tier, evaluate DCA
    # ------------------------------------------------------------------

    async def on_tick(self) -> None:
        # Could refresh fee tier from exchange API here
        pass

    # ------------------------------------------------------------------
    # Wiring helpers
    # ------------------------------------------------------------------

    def set_rl_strategy(self, strategy) -> None:
        """Inject loaded RLStrategy instance."""
        self._rl_strategy = strategy

    def update_equity(self, equity: float) -> None:
        self._equity = equity

    @property
    def open_positions(self) -> int:
        return len(self._position_engine.get_open())

    @property
    def active_dca_plans(self) -> int:
        return len(self._dca_engine.get_active())
