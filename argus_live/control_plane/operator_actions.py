from __future__ import annotations

from argus_live.monitoring.audit_log import AuditLogger, AuditRecord
from argus_live.state.operator_state import OperatorStateStore


def halt(store: OperatorStateStore, audit: AuditLogger, actor: str) -> None:
    state = store.load()
    store.save(state.__class__(halted=True, frozen=state.frozen, reconciliation_ack_required=state.reconciliation_ack_required, last_ack_actor=state.last_ack_actor, last_ack_at_utc=state.last_ack_at_utc))
    audit.write(AuditRecord.new(actor=actor, action="halt", reason_code="OPERATOR_HALT", payload={}))


def freeze(store: OperatorStateStore, audit: AuditLogger, actor: str) -> None:
    state = store.load()
    store.save(state.__class__(halted=state.halted, frozen=True, reconciliation_ack_required=state.reconciliation_ack_required, last_ack_actor=state.last_ack_actor, last_ack_at_utc=state.last_ack_at_utc))
    audit.write(AuditRecord.new(actor=actor, action="freeze", reason_code="OPERATOR_FREEZE", payload={}))


def resume(store: OperatorStateStore, audit: AuditLogger, actor: str) -> None:
    state = store.load()
    store.save(state.__class__(halted=False, frozen=False, reconciliation_ack_required=state.reconciliation_ack_required, last_ack_actor=state.last_ack_actor, last_ack_at_utc=state.last_ack_at_utc))
    audit.write(AuditRecord.new(actor=actor, action="resume", reason_code="OPERATOR_RESUME", payload={}))


def reconcile_ack(store: OperatorStateStore, audit: AuditLogger, actor: str) -> None:
    state = store.load()
    updated = state.with_ack(actor)
    store.save(updated)
    audit.write(AuditRecord.new(actor=actor, action="reconcile_ack", reason_code="RECONCILE_ACK", payload={}))
