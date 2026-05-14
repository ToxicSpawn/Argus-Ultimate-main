from __future__ import annotations

from pathlib import Path
from typing import Any

from argus_live.config.config_lock import ConfigLock, lock_config, write_config_artifacts
from argus_live.execution.intent_builder import build_intent
from argus_live.ledger.event_journal import JournalEvent
from argus_live.risk.capital_scaling import derive_capital_scaling_policy


class ExecutionEntrypoint:
    def __init__(
        self,
        runtime,
        manifest_hash: str,
        proving_lifecycle=None,
        profile: str = "paper",
        config_payload: dict[str, Any] | None = None,
        config_artifact_dir: str | None = None,
        previous_config_payload: dict[str, Any] | None = None,
        strict_governance: bool = True,
    ):
        self.runtime = runtime
        self.manifest_hash = manifest_hash
        self.proving_lifecycle = proving_lifecycle
        self.profile = profile
        self.config_payload = config_payload or {
            'profile': profile,
            'ladder_stage': 'PAPER' if str(profile).lower() == 'paper' else str(profile).upper(),
            'manifest_hash': manifest_hash,
            'runtime_path': 'intent->liquidity->constitution->governance->route->submit->reconcile->attribute->replay_audit->proving_review',
            'strict_governance': strict_governance,
        }
        self.previous_config_payload = previous_config_payload
        self.config_artifact_dir = config_artifact_dir or str(Path('artifacts') / 'config')
        self.strict_governance = strict_governance
        self._config_lock: ConfigLock = lock_config(self.config_payload)
        self._config_locked_emitted = False
        self._bootstrap_policy = derive_capital_scaling_policy(
            equity=float(self.config_payload.get('bootstrap_equity', 0.0) or 0.0),
            ladder_stage=str(self.config_payload.get('ladder_stage', 'PAPER')),
            replay_ok=True,
            execution_alpha_bps=float(self.config_payload.get('bootstrap_execution_alpha_bps', 0.0) or 0.0),
            critical_incident_count=0,
            regression_overall_score=float(self.config_payload.get('bootstrap_regression_overall_score', 100.0) or 100.0),
            max_safe_aum=float(self.config_payload.get('max_safe_aum')) if self.config_payload.get('max_safe_aum') is not None else None,
        )

        if self.strict_governance and getattr(self.runtime, 'governance', None) is None:
            raise RuntimeError('Governance coordinator required for strict_governance execution entrypoint')
        if str(self.config_payload.get('ladder_stage', 'PAPER')).upper() in {'LIVE_SAFE', 'SCALE'} and self._bootstrap_policy.scale_blocked:
            raise RuntimeError(f"Live-safe bootstrap blocked: {', '.join(self._bootstrap_policy.blockers or [self._bootstrap_policy.throttle_reason or 'scale readiness failure'])}")
        if hasattr(self.runtime, 'strict_governance'):
            self.runtime.strict_governance = strict_governance

    @property
    def config_hash(self) -> str:
        return self._config_lock.config_hash

    def _emit_config_lock(self, intent_id: str) -> None:
        if self._config_locked_emitted:
            return
        config_path, diff_path = write_config_artifacts(
            artifact_dir=self.config_artifact_dir,
            run_id=self.manifest_hash,
            config_lock=self._config_lock,
            previous_config=self.previous_config_payload,
        )
        self.runtime.journal.append(
            JournalEvent.new(
                'CONFIG_LOCKED',
                intent_id,
                {
                    'run_id': self.manifest_hash,
                    'config_hash': self._config_lock.config_hash,
                    'profile': self.profile,
                    'runtime_path': self.config_payload.get('runtime_path'),
                    'config_artifact_path': str(config_path),
                    'config_diff_artifact_path': str(diff_path),
                    'bootstrap_scale_readiness_score': self._bootstrap_policy.scale_readiness_score,
                    'bootstrap_scale_blocked': self._bootstrap_policy.scale_blocked,
                    'bootstrap_max_safe_aum': self._bootstrap_policy.max_safe_aum,
                },
            )
        )
        self._config_locked_emitted = True

    def _assert_replay_clear(self) -> None:
        store = getattr(self.runtime, 'replay_audit_store', None)
        if store is None:
            return
        latest = store.latest_for_run(self.manifest_hash) or store.latest()
        if latest is not None and int(latest.mismatch_count) > 0:
            raise RuntimeError('Replay mismatch present for current runtime; refusing to submit new intents')

    def submit_order(self, *, symbol: str, side: str, quantity: float, strategy_id: str, price: float, equity: float, symbol_notional_after: float, cluster_notional_after: float, gross_notional_after: float, top_of_book_notional: float, spread_bps: float, volatility_bps: float, fee_rate: float, allow_market_orders: bool = False, market_state: dict[str, Any] | None = None) -> str:
        intent = build_intent(symbol=symbol, side=side, quantity=quantity, strategy_id=strategy_id, manifest_hash=self.manifest_hash, limit_price=price)
        self._assert_replay_clear()
        self._emit_config_lock(intent.intent_id)
        market_state = dict(market_state or {})
        market_state.setdefault('ladder_stage', self.config_payload.get('ladder_stage', 'PAPER'))
        self.runtime.process_intent(
            intent,
            equity=equity,
            symbol_notional_after=symbol_notional_after,
            cluster_notional_after=cluster_notional_after,
            gross_notional_after=gross_notional_after,
            top_of_book_notional=top_of_book_notional,
            spread_bps=spread_bps,
            volatility_bps=volatility_bps,
            fee_rate=fee_rate,
            allow_market_orders=allow_market_orders,
            config_hash=self._config_lock.config_hash,
            market_state=market_state,
        )
        if self.proving_lifecycle is not None:
            result = self.proving_lifecycle.refresh(run_id=self.manifest_hash, trade_ledger=self.runtime.trade_ledger)
            self.runtime.journal.append(JournalEvent.new('PROVING_REVIEW_REFRESHED', intent.intent_id, {'run_id': result.run_id, 'report_path': result.report_path, 'promotion_decision': result.review.promotion_decision, 'trade_count': result.review.trade_count, 'active_blockers': result.review.active_blockers, 'regression_summary_path': result.regression_summary_path, 'regression_overall_score': result.review.regression_overall_score, 'regression_pass_rate': result.review.regression_pass_rate, 'scale_readiness_score': result.review.scale_readiness_score, 'scale_blocked': result.review.scale_blocked, 'control_surface_path': result.control_surface_path}))
        return intent.intent_id
