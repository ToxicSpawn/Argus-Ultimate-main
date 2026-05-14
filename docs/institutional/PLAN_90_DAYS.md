# 90-Day Institutional Plan

## Phase 1 (Days 1-30): Control Baseline
1. Finalize governance artifacts and owner assignments.
2. Enforce readiness checks in CI and Windows ops scripts.
3. Close critical technical blockers: reconciliation, idempotency, fail-closed, key hygiene.
4. Run initial 7-day soak and incident drill.

## Phase 2 (Days 31-60): Validation Depth
1. Complete walk-forward promotion pack and deterministic replay checks.
2. Run chaos suite (exchange timeout/disconnect, DB failover, clock skew).
3. Produce first monthly control attestation pack.
4. Execute DR drill and record RTO/RPO evidence.

## Phase 3 (Days 61-90): External Readiness
1. Independent review of risk/compliance controls.
2. Security hardening closure and pentest remediation evidence.
3. Treasury/counterparty controls and concentration limits approval.
4. Final 30-day soak pass plus full readiness gate pass.

## Non-Negotiable Gates
1. No manual override for kill-switch and reconciliation ownership controls.
2. No promotion to live while readiness status is `FAIL`.
3. No production candidate without immutable bundle hash and signed approval record.
