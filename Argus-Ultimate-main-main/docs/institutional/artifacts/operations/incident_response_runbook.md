# Incident Response Runbook

Owner: Operations  
Review cadence: Quarterly

## Severity
1. `SEV1`: Trading safety/control failure (halt immediately).
2. `SEV2`: Material degradation (risk-limited mode, no promotion).
3. `SEV3`: Non-critical operational issue.

## Immediate Actions
1. Confirm kill-switch and reconciliation freeze state.
2. Stop new intents if state is ambiguous.
3. Capture timeline from `decision_events` and JSONL logs.
4. Notify on-call owner and compliance owner.

## Recovery
1. Reconcile positions, open orders, and balances to venue truth.
2. Obtain operator acknowledgement before resuming new intents.
3. Produce postmortem with root cause, blast radius, and prevention actions.
