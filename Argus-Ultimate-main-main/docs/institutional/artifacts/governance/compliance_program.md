# Compliance Program

Owner: Compliance Lead  
Review cadence: Monthly

## Scope
This program governs monitoring, recordkeeping, escalation, and control attestations for Argus operations across paper, backtest, and live environments.

## Mandatory Controls
1. Market abuse prevention controls are active and tested.
2. Decision and execution records are retained in immutable audit stores.
3. Reconciliation ownership controls are enforced with operator acknowledgement.
4. Promotion to live requires passing readiness gates and independent approvals.

## Monitoring and Escalation
1. Daily review of reject histogram, error rate, reconciliation events, and freeze state.
2. Incident severity classification and escalation path to risk/compliance owners.
3. Monthly control attestation package signed by accountable owners.

## Evidence
1. `reports/soak_gate_latest.json`
2. `reports/walk_forward_latest.json`
3. `data/unified_trades.db` decision snapshots and decision events
