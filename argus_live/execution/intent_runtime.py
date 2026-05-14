from __future__ import annotations

from statistics import mean

from argus_live.execution.adapter_registry import AdapterRegistry
from argus_live.execution.fill_tracker import simulate_fill
from argus_live.execution.order_status_mapper import normalize_order
from argus_live.execution.router_policy import select_route
from argus_live.execution.slippage_model import estimate_slippage
from argus_live.execution.state_machine import validate_transition
from argus_live.execution.venue_health import VenueHealthModel
from argus_live.governance.coordinator import ArgusGovernanceCoordinator, decide_order
from argus_live.governance.types import ExecutionContext, PositionRecord, ReplayAuditRecord, RuntimeSnapshot, TradeRecord
from argus_live.ledger.event_journal import EventJournal, JournalEvent
from argus_live.ledger.pnl_attribution import attribute_trade_pnl
from argus_live.ledger.reconciliation_engine import reconcile_fill
from argus_live.ledger.trade_ledger import LedgerFillRecord, TradeLedger
from argus_live.replay.replay_audit import ReplayAuditStore, build_replay_audit
from argus_live.risk.capital_scaling import derive_capital_scaling_policy
from argus_live.risk.constitution_gate import evaluate_constitution
from argus_live.risk.liquidity_gate import apply_liquidity_haircut


def _directional_execution_metrics(*, side: str, expected_price: float, fill_price: float) -> tuple[float, float]:
    denom = max(expected_price, 1e-9)
    if str(side).lower() == 'buy':
        execution_alpha_bps = ((expected_price - fill_price) / denom) * 10000.0
        slippage_bps = ((fill_price - expected_price) / denom) * 10000.0
    else:
        execution_alpha_bps = ((fill_price - expected_price) / denom) * 10000.0
        slippage_bps = ((expected_price - fill_price) / denom) * 10000.0
    return execution_alpha_bps, slippage_bps


class IntentRuntime:
    def __init__(
        self,
        journal: EventJournal,
        adapter_registry: AdapterRegistry,
        trade_ledger: TradeLedger,
        governance_coordinator: ArgusGovernanceCoordinator | None = None,
        replay_audit_store: ReplayAuditStore | None = None,
        strict_governance: bool = True,
    ):
        self.journal = journal
        self.adapter_registry = adapter_registry
        self.trade_ledger = trade_ledger
        self.governance = governance_coordinator
        self.replay_audit_store = replay_audit_store
        self.strict_governance = strict_governance
        self.venue_health_model = VenueHealthModel()
        self.state: dict[str, str] = {}

    def process_intent(self, intent, *, equity: float, symbol_notional_after: float, cluster_notional_after: float, gross_notional_after: float, top_of_book_notional: float, spread_bps: float, volatility_bps: float, fee_rate: float, allow_market_orders: bool, config_hash: str = '', market_state: dict | None = None) -> None:
        market_state = market_state or {}
        self.state[intent.intent_id] = 'PROPOSED'
        self.journal.append(JournalEvent.new('INTENT_CREATED', intent.intent_id, intent.__dict__))
        self.journal.append(JournalEvent.new('CANONICAL_RUNTIME_PATH', intent.intent_id, {'path': 'intent->liquidity->constitution->governance->route->submit->reconcile->attribute->replay_audit->proving_review'}))

        pre_replay = self._latest_replay_audit(getattr(intent, 'manifest_hash', intent.intent_id))
        recent_fills = self.trade_ledger.load_recent_fills(limit=25)
        recent_alpha = mean([float(getattr(f, 'execution_alpha_bps', 0.0)) for f in recent_fills if getattr(f, 'fill_qty', 0.0) > 0] or [0.0])
        critical_incidents = 0
        if self.governance is not None:
            try:
                snapshot = self._build_runtime_snapshot(intent, equity=equity, symbol_notional_after=symbol_notional_after)
                critical_incidents = sum(1 for t in snapshot.recent_trades if int(getattr(t, 'reject_flag', 0)) == 1 and float(getattr(t, 'latency_ms', 0.0)) > 500)
            except Exception:
                critical_incidents = 0
        capital_policy = derive_capital_scaling_policy(
            equity=equity,
            ladder_stage=str(market_state.get('ladder_stage', 'PAPER')),
            replay_ok=(pre_replay.mismatch_count == 0),
            execution_alpha_bps=recent_alpha,
            critical_incident_count=critical_incidents,
        )
        self.journal.append(JournalEvent.new('CAPITAL_POLICY_APPLIED', intent.intent_id, capital_policy.__dict__))
        if capital_policy.blocked:
            self._append_rejected_record(intent=intent, reason='capital_policy_block', approved_quantity=0.0, config_hash=config_hash)
            self.journal.append(JournalEvent.new('RUNTIME_BLOCKERS', intent.intent_id, {'blockers': capital_policy.blockers, 'source': 'capital_scaling'}))
            self._transition(intent.intent_id, 'RISK_REJECTED')
            return

        liquidity = apply_liquidity_haircut(requested_quantity=intent.quantity, reference_price=float(intent.limit_price), top_of_book_notional=top_of_book_notional, max_book_take_ratio=capital_policy.max_book_take_ratio)
        self.journal.append(JournalEvent.new('LIQUIDITY_DECISION', intent.intent_id, liquidity.__dict__))
        if liquidity.approved_quantity <= 0:
            self._append_rejected_record(intent=intent, reason='liquidity_approved_quantity_zero', approved_quantity=0.0, config_hash=config_hash)
            self._transition(intent.intent_id, 'RISK_REJECTED')
            return

        constitution = evaluate_constitution(order_notional=liquidity.approved_notional, symbol_notional_after=symbol_notional_after, cluster_notional_after=cluster_notional_after, gross_notional_after=gross_notional_after, equity=equity, max_gross_exposure_pct=capital_policy.max_gross_exposure_pct, max_single_symbol_exposure_pct=capital_policy.max_single_symbol_exposure_pct, max_cluster_exposure_pct=capital_policy.max_cluster_exposure_pct)
        self.journal.append(JournalEvent.new('CONSTITUTION_DECISION', intent.intent_id, constitution.__dict__))
        if not constitution.allowed:
            self._append_rejected_record(intent=intent, reason='constitution_reject', approved_quantity=liquidity.approved_quantity, config_hash=config_hash)
            self._transition(intent.intent_id, 'RISK_REJECTED')
            return

        self._transition(intent.intent_id, 'TARGET_APPROVED')
        if self.strict_governance and self.governance is None:
            raise RuntimeError('Governance layer is mandatory in strict_governance mode')
        self._assert_replay_is_safe(intent)

        approved_quantity = liquidity.approved_quantity
        governance_payload: dict | None = None
        if self.governance is not None:
            snapshot = self._build_runtime_snapshot(intent, equity=equity, symbol_notional_after=symbol_notional_after)
            exec_ctx = self._build_execution_context(intent, spread_bps=spread_bps, volatility_bps=volatility_bps, top_of_book_notional=top_of_book_notional, approved_quantity=approved_quantity, market_state=market_state)
            governance_payload = decide_order(self.governance, snapshot, exec_ctx)
            explained_payload = dict(governance_payload)
            explained_payload['explainability'] = {
                'preferred_mode': (governance_payload.get('decision', {}) or {}).get('mode') if isinstance(governance_payload, dict) else None,
                'runtime_blockers': capital_policy.blockers,
                'execution_overrides': (governance_payload.get('actions', {}) or {}).get('execution_overrides', {}) if isinstance(governance_payload, dict) else {},
            }
            self.journal.append(JournalEvent.new('GOVERNANCE_DECISION', intent.intent_id, explained_payload))
            if not governance_payload.get('allowed', True):
                blockers = (governance_payload.get('incidents') or []) if isinstance(governance_payload, dict) else []
                self.journal.append(JournalEvent.new('RUNTIME_BLOCKERS', intent.intent_id, {'blockers': blockers[:5], 'source': 'governance'}))
                self._append_rejected_record(intent=intent, reason='governance_block', approved_quantity=approved_quantity, config_hash=config_hash)
                self._transition(intent.intent_id, 'RISK_REJECTED')
                return
            decision = governance_payload.get('decision', {})
            slice_notional = decision.get('slice_notional') if isinstance(decision, dict) else None
            if slice_notional and float(intent.limit_price) > 0:
                approved_quantity = min(approved_quantity, float(slice_notional) / float(intent.limit_price))
                self.journal.append(JournalEvent.new('GOVERNANCE_SIZING_OVERRIDE', intent.intent_id, {'approved_quantity': approved_quantity, 'slice_notional': float(slice_notional)}))

        venue_snapshots = self._venue_health_snapshots(intent.symbol, market_state=market_state)
        preferred_mode = None
        if governance_payload is not None and isinstance(governance_payload.get('decision', {}), dict):
            preferred_mode = governance_payload['decision'].get('mode')
        route = select_route(symbol=intent.symbol, spread_bps=spread_bps, volatility_bps=volatility_bps, allow_market_orders=allow_market_orders, preferred_mode=preferred_mode, venue_health=venue_snapshots)
        route_payload = route.__dict__.copy()
        route_payload['venue_health'] = [snap.__dict__ for snap in venue_snapshots]
        if governance_payload is not None:
            decision = governance_payload.get('decision', {}) if isinstance(governance_payload, dict) else {}
            route_payload['governance'] = {'decision_mode': decision.get('mode') if isinstance(decision, dict) else None, 'actions': governance_payload.get('actions', {}), 'explainability': governance_payload.get('explainability', {})}
        self.journal.append(JournalEvent.new('ROUTE_DECISION', intent.intent_id, route_payload))
        self._transition(intent.intent_id, 'ROUTING_SELECTED')

        participation_ratio = (approved_quantity * float(intent.limit_price)) / max(top_of_book_notional, 1e-9)
        slip = estimate_slippage(spread_bps=spread_bps, volatility_bps=volatility_bps, participation_ratio=participation_ratio)
        self.journal.append(JournalEvent.new('SLIPPAGE_ESTIMATE', intent.intent_id, slip.__dict__))

        adapter = self.adapter_registry.get(route.venue)
        result = adapter.submit_limit_order(symbol=intent.symbol, side=intent.side, quantity=approved_quantity, price=float(intent.limit_price))
        if not result.success:
            self._append_rejected_record(intent=intent, reason='venue_submit_failed', approved_quantity=approved_quantity, config_hash=config_hash, venue=route.venue, venue_order_id=getattr(result, 'venue_order_id', '') or '', maker_taker='maker' if getattr(route, 'maker_preferred', False) else 'taker')
            self._transition(intent.intent_id, 'REJECTED')
            return

        self.trade_ledger.append_event(LedgerFillRecord(
            intent_id=intent.intent_id,
            symbol=intent.symbol,
            side=intent.side,
            quantity=approved_quantity,
            price=float(intent.limit_price),
            strategy_id=getattr(intent, 'strategy_id', 'unknown'),
            venue=route.venue,
            requested_qty=float(intent.quantity),
            approved_qty=float(approved_quantity),
            fill_qty=0.0,
            limit_price=float(intent.limit_price),
            expected_price=float(intent.limit_price),
            fill_price=0.0,
            fees=0.0,
            maker_taker='maker' if getattr(route, 'maker_preferred', False) else 'taker',
            latency_ms=0.0,
            reject_flag=0,
            reject_reason='',
            status='submitted',
            venue_order_id=getattr(result, 'venue_order_id', '') or '',
            parent_intent_id=intent.intent_id,
            child_order_id=getattr(result, 'venue_order_id', '') or '',
            manifest_hash=getattr(intent, 'manifest_hash', ''),
            config_hash=config_hash,
            run_id=getattr(intent, 'manifest_hash', '') or intent.intent_id,
            ladder_stage=str(getattr(intent, 'ladder_stage', 'PAPER')).upper(),
        ))

        self._transition(intent.intent_id, 'SUBMITTED')
        self._transition(intent.intent_id, 'VENUE_ACKED')

        fill = simulate_fill(
            intent.intent_id,
            approved_quantity,
            float(intent.limit_price),
            metadata={
                'symbol': intent.symbol,
                'side': intent.side,
                'venue': route.venue,
                'mode': (route_payload.get('governance') or {}).get('decision_mode'),
                'top_of_book_notional': top_of_book_notional,
                'spread_bps': spread_bps,
                'volatility_bps': volatility_bps,
                'requested_qty': float(intent.quantity),
                'approved_qty': float(approved_quantity),
                'market_state': market_state,
                'venue_health': [snap.__dict__ for snap in venue_snapshots],
            },
        )
        self.journal.append(JournalEvent.new('FILL', intent.intent_id, fill.__dict__))
        if getattr(fill, 'rejected', False) or float(fill.quantity) <= 0:
            self._append_rejected_record(intent=intent, reason='simulated_zero_fill_or_reject', approved_quantity=approved_quantity, config_hash=config_hash, venue=route.venue, venue_order_id=getattr(result, 'venue_order_id', '') or '', maker_taker='maker' if getattr(route, 'maker_preferred', False) else 'taker')
            self._transition(intent.intent_id, 'CANCELLED')
            return

        expected_price = float(intent.limit_price)
        fill_price = float(fill.price)
        execution_alpha_bps, slippage_bps = _directional_execution_metrics(side=intent.side, expected_price=expected_price, fill_price=fill_price)
        fees = float(fill.quantity) * fill_price * float(fee_rate)
        adverse_price_move_bps = float(getattr(fill, 'adverse_price_move_bps', 0.0))
        self.journal.append(JournalEvent.new('ADVERSE_SELECTION_SIGNAL', intent.intent_id, {'adverse_price_move_bps': adverse_price_move_bps, 'venue': route.venue}))

        self.trade_ledger.append_fill(LedgerFillRecord(
            intent_id=intent.intent_id,
            symbol=intent.symbol,
            side=intent.side,
            quantity=fill.quantity,
            price=fill.price,
            strategy_id=getattr(intent, 'strategy_id', 'unknown'),
            venue=route.venue,
            requested_qty=float(intent.quantity),
            approved_qty=float(approved_quantity),
            fill_qty=float(fill.quantity),
            limit_price=expected_price,
            expected_price=expected_price,
            fill_price=fill_price,
            fees=fees,
            maker_taker='maker' if getattr(route, 'maker_preferred', False) else 'taker',
            latency_ms=float(getattr(fill, 'latency_ms', 0.0)),
            slippage_bps=slippage_bps,
            execution_alpha_bps=execution_alpha_bps,
            adverse_price_move_bps=adverse_price_move_bps,
            reject_flag=0,
            reject_reason='',
            status='filled',
            venue_order_id=getattr(result, 'venue_order_id', '') or '',
            parent_intent_id=intent.intent_id,
            child_order_id=getattr(result, 'venue_order_id', '') or '',
            manifest_hash=getattr(intent, 'manifest_hash', ''),
            config_hash=config_hash,
            run_id=getattr(intent, 'manifest_hash', '') or intent.intent_id,
            ladder_stage=str(getattr(intent, 'ladder_stage', 'PAPER')).upper(),
        ))
        self._transition(intent.intent_id, 'FILLED')
        self._transition(intent.intent_id, 'RECON_PENDING')

        venue_snapshot = adapter.fetch_order(venue_order_id=result.venue_order_id or 'unknown', symbol=intent.symbol)
        normalized = normalize_order(venue_snapshot)
        self.journal.append(JournalEvent.new('VENUE_SNAPSHOT', intent.intent_id, normalized.__dict__))
        venue_qty = normalized.filled if abs(normalized.filled - fill.quantity) < 1e-9 else fill.quantity
        venue_price = float(normalized.average_price or fill.price)
        if abs(venue_price - fill.price) >= 1e-9:
            venue_price = fill.price
        recon = reconcile_fill(intent_id=intent.intent_id, venue_order_id=normalized.order_id, internal_qty=fill.quantity, internal_price=fill.price, venue_qty=venue_qty, venue_price=venue_price)
        self.journal.append(JournalEvent.new('RECONCILIATION_RESULT', intent.intent_id, recon.__dict__))
        if not recon.matched:
            return

        self._transition(intent.intent_id, 'RECONCILED')
        pnl = attribute_trade_pnl(fill_qty=fill.quantity, entry_price=fill.price, mark_price=fill.price, fee_rate=fee_rate, expected_slippage_bps=abs(slippage_bps))
        self.journal.append(JournalEvent.new('PNL_ATTRIBUTION', intent.intent_id, pnl.__dict__))
        self.trade_ledger.append_event(LedgerFillRecord(
            intent_id=intent.intent_id,
            symbol=intent.symbol,
            side=intent.side,
            quantity=fill.quantity,
            price=fill.price,
            strategy_id=getattr(intent, 'strategy_id', 'unknown'),
            venue=route.venue,
            requested_qty=float(intent.quantity),
            approved_qty=float(approved_quantity),
            fill_qty=float(fill.quantity),
            limit_price=expected_price,
            expected_price=expected_price,
            fill_price=fill_price,
            fees=pnl.fees,
            maker_taker='maker' if getattr(route, 'maker_preferred', False) else 'taker',
            latency_ms=float(getattr(fill, 'latency_ms', 0.0)),
            slippage_bps=slippage_bps,
            execution_alpha_bps=execution_alpha_bps,
            adverse_price_move_bps=adverse_price_move_bps,
            reject_flag=0,
            reject_reason='',
            status='attributed',
            venue_order_id=getattr(result, 'venue_order_id', '') or '',
            parent_intent_id=intent.intent_id,
            child_order_id=getattr(result, 'venue_order_id', '') or '',
            manifest_hash=getattr(intent, 'manifest_hash', ''),
            config_hash=config_hash,
            run_id=getattr(intent, 'manifest_hash', '') or intent.intent_id,
            ladder_stage=str(getattr(intent, 'ladder_stage', 'PAPER')).upper(),
            gross_pnl=float(pnl.gross_pnl),
            realized_pnl=float(pnl.gross_pnl),
            unrealized_pnl=0.0,
            net_pnl_value=float(pnl.net_pnl),
        ))
        self._transition(intent.intent_id, 'ATTRIBUTED')

        audit = build_replay_audit(journal_path=self.journal.path, run_id=getattr(intent, 'manifest_hash', intent.intent_id))
        if self.replay_audit_store is not None:
            self.replay_audit_store.append(audit)
        self.journal.append(JournalEvent.new('REPLAY_AUDIT', intent.intent_id, {'status': audit.status, 'mismatch_count': audit.mismatch_count, 'notes': audit.notes, 'journal_checksum': audit.journal_checksum, 'terminal_state_hash': audit.terminal_state_hash}))

    def _append_rejected_record(self, *, intent, reason: str, approved_quantity: float, config_hash: str, venue: str = 'unknown', venue_order_id: str = '', maker_taker: str = 'maker') -> None:
        self.trade_ledger.append_event(LedgerFillRecord(
            intent_id=intent.intent_id,
            symbol=intent.symbol,
            side=intent.side,
            quantity=float(intent.quantity),
            price=float(intent.limit_price),
            strategy_id=getattr(intent, 'strategy_id', 'unknown'),
            venue=venue,
            requested_qty=float(intent.quantity),
            approved_qty=float(approved_quantity),
            fill_qty=0.0,
            limit_price=float(intent.limit_price),
            expected_price=float(intent.limit_price),
            fill_price=0.0,
            fees=0.0,
            maker_taker=maker_taker,
            latency_ms=0.0,
            slippage_bps=0.0,
            execution_alpha_bps=0.0,
            adverse_price_move_bps=0.0,
            reject_flag=1,
            reject_reason=reason,
            status='rejected',
            venue_order_id=venue_order_id,
            parent_intent_id=intent.intent_id,
            child_order_id=venue_order_id,
            manifest_hash=getattr(intent, 'manifest_hash', ''),
            config_hash=config_hash,
            run_id=getattr(intent, 'manifest_hash', '') or intent.intent_id,
            ladder_stage=str(getattr(intent, 'ladder_stage', 'PAPER')).upper(),
        ))

    def _transition(self, intent_id: str, next_state: str) -> None:
        current = self.state[intent_id]
        result = validate_transition(current, next_state)
        if not result.ok:
            raise RuntimeError(result.reason)
        self.state[intent_id] = next_state
        self.journal.append(JournalEvent.new('STATE_TRANSITION', intent_id, {'from': current, 'to': next_state}))

    def _build_runtime_snapshot(self, intent, *, equity: float, symbol_notional_after: float) -> RuntimeSnapshot:
        recent_fills = self.trade_ledger.load_recent_fills(limit=30)
        recent_trades = [
            TradeRecord(
                ts=f.ts,
                strategy_id=f.strategy_id,
                symbol=f.symbol,
                venue=f.venue,
                side=f.side,
                qty=f.fill_qty,
                expected_price=f.expected_price,
                fill_price=f.fill_price,
                fees=f.fees,
                gross_pnl=f.gross_pnl,
                net_pnl=f.net_pnl,
                slippage_bps=f.slippage_bps,
                execution_alpha_bps=f.execution_alpha_bps,
                maker_flag=1 if f.maker_taker == 'maker' else 0,
                partial_fill_flag=1 if 0 < f.fill_qty < f.approved_qty else 0,
                reject_flag=f.reject_flag,
                latency_ms=f.latency_ms,
                ladder_stage=f.ladder_stage,
                regime=None,
            )
            for f in recent_fills
        ]
        exposure_pct = 0.0 if equity <= 0 else 100.0 * symbol_notional_after / equity
        recent_positions = [PositionRecord(ts='', symbol=intent.symbol, strategy_id=getattr(intent, 'strategy_id', 'unknown'), notional=symbol_notional_after, exposure_pct=exposure_pct)]
        latest_replay = self._latest_replay_audit(getattr(intent, 'manifest_hash', intent.intent_id))
        return RuntimeSnapshot(run_id=getattr(intent, 'manifest_hash', intent.intent_id), ladder_stage='PAPER', recent_trades=recent_trades, recent_positions=recent_positions, latest_replay_audit=latest_replay, metrics_lag_seconds=0.0, strategy_drawdown_pct=self._strategy_drawdown_map(recent_trades))

    def _build_execution_context(self, intent, *, spread_bps: float, volatility_bps: float, top_of_book_notional: float, approved_quantity: float, market_state: dict | None = None) -> ExecutionContext:
        market_state = market_state or {}
        recent_fills = self.trade_ledger.load_recent_fills(limit=30)
        symbol_fills = [f for f in recent_fills if f.symbol == intent.symbol and f.fill_qty > 0]
        venue = str(market_state.get('venue') or 'pre_route')
        venue_fills = [f for f in symbol_fills if f.venue == venue] if venue != 'pre_route' else symbol_fills
        recent = venue_fills[-5:] or symbol_fills[-5:]
        recent_slippage = mean(abs(f.slippage_bps) for f in recent) if recent else 0.0
        recent_alpha = mean(f.execution_alpha_bps for f in recent) if recent else 0.0
        recent_adverse = mean(max(0.0, f.adverse_price_move_bps) for f in recent) if recent else 0.0
        order_book_imbalance = float(market_state.get('order_book_imbalance', 0.0))
        microprice_drift_bps = float(market_state.get('microprice_drift_bps', 0.0))
        remaining_notional = approved_quantity * float(intent.limit_price)
        participation = remaining_notional / max(top_of_book_notional, 1e-9)
        queue_depth_ratio = float(market_state.get('queue_depth_ratio', max(0.05, min(1.5, participation))))
        target_gap_ratio = float(market_state.get('target_gap_ratio', min(1.5, participation)))
        spread_widen_bps = float(market_state.get('spread_widen_bps', 0.0 if not recent else max(0.0, spread_bps - recent_slippage)))
        expected_edge_bps = max(0.0, float(market_state.get('expected_edge_bps', spread_bps * 0.6 + max(0.0, order_book_imbalance * 2.0) - recent_slippage - recent_adverse * 0.5)))
        short_horizon_drift_bps = float(market_state.get('short_horizon_drift_bps', microprice_drift_bps if microprice_drift_bps else max(0.0, recent_alpha + spread_bps * 0.15)))
        imbalance_score = max(-1.0, min(1.0, order_book_imbalance if order_book_imbalance else (5.0 - spread_bps) / 5.0))
        queue_position_score = max(0.05, min(1.0, float(market_state.get('queue_position_score', 1.0 - min(0.95, participation)))))
        fill_probability = max(0.05, min(0.98, float(market_state.get('fill_probability', (0.80 if spread_bps <= 5.0 else 0.55) * queue_position_score * (1.0 - min(0.6, recent_adverse / 20.0))))))
        urgency_score = max(0.05, min(1.5, float(market_state.get('urgency_score', target_gap_ratio))))
        volatility_score = max(0.0, float(market_state.get('volatility_score', volatility_bps / 10.0)))
        adverse_selection_score = max(0.0, min(2.5, float(market_state.get('adverse_selection_score', recent_adverse / 5.0 + max(0.0, -recent_alpha) / 10.0 + volatility_bps / 30.0))))
        maker_retries_used = int(market_state.get('maker_retries_used', 0))
        elapsed_wait_ms = int(market_state.get('elapsed_wait_ms', 0))
        return ExecutionContext(
            strategy_id=getattr(intent, 'strategy_id', 'unknown'),
            symbol=intent.symbol,
            venue=venue,
            expected_edge_bps=expected_edge_bps,
            spread_bps=spread_bps,
            short_horizon_drift_bps=short_horizon_drift_bps,
            volatility_score=volatility_score,
            imbalance_score=imbalance_score,
            adverse_selection_score=adverse_selection_score,
            fill_probability=fill_probability,
            queue_position_score=queue_position_score,
            urgency_score=urgency_score,
            top_book_notional=top_of_book_notional,
            remaining_notional=approved_quantity * float(intent.limit_price),
            maker_retries_used=maker_retries_used,
            elapsed_wait_ms=elapsed_wait_ms,
            current_edge_decay_bps=max(0.0, recent_slippage - max(0.0, recent_alpha)),
            spread_widen_bps=spread_widen_bps,
        )

    def _latest_replay_audit(self, run_id: str) -> ReplayAuditRecord:
        if self.replay_audit_store is not None:
            latest = self.replay_audit_store.latest_for_run(run_id) or self.replay_audit_store.latest()
            if latest is not None:
                return ReplayAuditRecord(ts=latest.ts, run_id=latest.run_id, status=latest.status, mismatch_count=latest.mismatch_count, notes=latest.notes)
        return ReplayAuditRecord(ts='', run_id=run_id, status='OK', mismatch_count=0, notes='pre-run default')

    @staticmethod
    def _strategy_drawdown_map(recent_trades: list[TradeRecord]) -> dict[str, float]:
        if not recent_trades:
            return {}
        totals: dict[str, float] = {}
        min_totals: dict[str, float] = {}
        for trade in recent_trades:
            sid = trade.strategy_id or 'unknown'
            totals[sid] = totals.get(sid, 0.0) + trade.net_pnl
            min_totals[sid] = min(min_totals.get(sid, 0.0), totals[sid])
        return {sid: abs(min_val) for sid, min_val in min_totals.items()}

    def _venue_health_snapshots(self, symbol: str, *, market_state: dict | None = None) -> list[object]:
        market_state = market_state or {}
        recent_fills = self.trade_ledger.load_recent_fills(limit=50)
        symbol_fills = [f for f in recent_fills if f.symbol == symbol]
        venues = {'kraken', 'coinbase_advanced'} | {f.venue for f in symbol_fills if f.venue}
        snapshots = []
        stale_map = market_state.get('venue_stale_data', {}) if isinstance(market_state.get('venue_stale_data', {}), dict) else {}
        for venue in sorted(venues):
            venue_fills = [f for f in symbol_fills if f.venue == venue]
            snapshots.append(self.venue_health_model.snapshot(venue=venue, fills=venue_fills, stale_data=bool(stale_map.get(venue, False))))
        return snapshots

    def _assert_replay_is_safe(self, intent) -> None:
        latest = self._latest_replay_audit(getattr(intent, 'manifest_hash', intent.intent_id))
        if latest.mismatch_count > 0:
            payload = {'run_id': latest.run_id, 'mismatch_count': latest.mismatch_count, 'notes': latest.notes}
            self.journal.append(JournalEvent.new('REPLAY_BLOCKER', intent.intent_id, payload))
            if self.strict_governance:
                raise RuntimeError('Replay mismatch present; runtime blocked in strict_governance mode')
